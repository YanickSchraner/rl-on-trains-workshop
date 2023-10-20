import os
import random
import sys
from argparse import ArgumentParser, Namespace
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Union, Type

import PIL
import numpy as np
import psutil
import torch
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.utils.rendertools import RenderTool
from torch.utils.tensorboard import SummaryWriter

# base_dir = Path(__file__).resolve().parent.parent
# sys.path.append(str(base_dir))
from utils.dddqn_policy import DDDQNPolicy
from utils.timer import Timer
from utils.observation_utils import normalize_observation
from utils.dddqn_rnd_policy import DDDQNRNDPolicy
from utils.test import FlatlandTester


def creat_observation_builder(obs_params):
    predictor = ShortestPathPredictorForRailEnv(obs_params.observation_max_path_depth)
    print("\nUsing standard TreeObs")

    def check_is_observation_valid(observation):
        return observation

    def get_normalized_observation(observation):
        return normalize_observation(observation, obs_params.observation_tree_depth, obs_params.observation_radius)

    tree_observation = TreeObsForRailEnv(max_depth=obs_params.observation_tree_depth, predictor=predictor)
    tree_observation.check_is_observation_valid = check_is_observation_valid
    tree_observation.get_normalized_observation = get_normalized_observation

    # Calculate the state size given the depth of the tree observation and the number of features
    n_features_per_node = tree_observation.observation_dim
    n_nodes = sum([np.power(4, i) for i in range(obs_params.observation_tree_depth + 1)])
    state_size = n_features_per_node * n_nodes

    return tree_observation, state_size


def create_rail_env(env_params, observation_builder):
    n_agents = env_params.n_agents
    x_dim = env_params.x_dim
    y_dim = env_params.y_dim
    n_cities = env_params.n_cities
    max_rails_between_cities = env_params.max_rails_between_cities
    max_rails_in_city = env_params.max_rails_in_city
    seed = env_params.seed

    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=env_params.malfunction_rate,
        min_duration=20,
        max_duration=50
    )

    return RailEnv(
        width=x_dim, height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rails_in_city
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=n_agents,
        malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
        obs_builder_object=observation_builder,
        random_seed=seed
    )


# Render the environment
# (You would usually reuse the same RenderTool)
def render_env(env):
    from IPython.display import Image, display
    env_renderer = RenderTool(env, gl="PGL")
    env_renderer.render_env()

    image = env_renderer.get_image()
    pil_image = PIL.Image.fromarray(image)
    display(pil_image)


def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼"]
    return " ".join(f'{action} {action_prob:.2f}' for action, action_prob in zip(actions, action_probs))


def train_agent(policy_cls: Union[Type[DDDQNPolicy], Type[DDDQNRNDPolicy]], train_params, train_env_params,
                eval_env_params, obs_params):
    # Unique ID for this training based on starting date
    training_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join('runs', training_id)

    # Ensure existing log dir
    os.makedirs(log_dir, exist_ok=True)

    # Set the seeds
    random.seed(train_env_params.seed)
    np.random.seed(train_env_params.seed)

    # Observation builder
    observation_builder, state_size = creat_observation_builder(obs_params)

    # The action space of flatland is 5 discrete actions
    action_size = 5

    # Setup two separate environments for training and evaluation
    train_env = create_rail_env(train_env_params, observation_builder)
    train_env.reset(regenerate_schedule=True, regenerate_rail=True)
    eval_env = create_rail_env(eval_env_params, observation_builder)
    eval_env.reset(regenerate_schedule=True, regenerate_rail=True)

    # setup env testing
    tester = FlatlandTester()

    # Initialize Policy
    policy = policy_cls(state_size, action_size, train_params)
    print(policy)

    # Load existing policy
    if train_params.load_policy != "":
        policy.load(train_params.load_policy)

    # Max number of steps per episode
    # This is the official formula used during evaluations
    # See details in flatland.envs.schedule_generators.sparse_schedule_generator
    # max_steps = int(4 * 2 * (env.height + env.width + (n_agents / n_cities)))
    max_steps = train_env._max_episode_steps

    action_count = [0] * action_size
    action_dict = dict()
    agent_obs = [None] * train_env.get_num_agents()
    agent_prev_obs = [None] * train_env.get_num_agents()
    agent_prev_action = [2] * train_env.get_num_agents()
    update_values = [False] * train_env.get_num_agents()

    # Smoothed values used as target for hyperparameter tuning
    smoothed_eval_normalized_score = -1.0
    smoothed_eval_completion = 0.0

    scores_window = deque(maxlen=train_params.checkpoint_interval)
    completion_window = deque(maxlen=train_params.checkpoint_interval)

    # TensorBoard writer
    def normalize_params(params: dict):
        hparams_dict = dict()
        for k, v in params.items():
            if v is None or isinstance(v, (bool, int, float, str)):
                hparams_dict[k] = v
            else:
                hparams_dict[k] = str(v)
        return hparams_dict

    writer = SummaryWriter(log_dir=os.path.join(log_dir, 'logs'))
    writer.add_hparams(normalize_params(vars(train_params)), {})
    writer.add_hparams(normalize_params(vars(train_env_params)), {})
    writer.add_hparams(normalize_params(vars(obs_params)), {})

    training_timer = Timer()
    training_timer.start()

    print(
        f"\n🚉 Training {train_env.get_num_agents()} trains on {train_env.width}x{train_env.height} grid "
        f"for {train_params.n_episodes} episodes, evaluating for {train_params.n_evaluation_episodes} episodes "
        f"every {train_params.checkpoint_interval} episodes. Training id '{training_id}'.\n")

    eps = train_params.eps_start

    for episode_idx in range(train_params.n_episodes + 1):
        step_timer = Timer()
        reset_timer = Timer()
        learn_timer = Timer()
        preproc_timer = Timer()
        inference_timer = Timer()

        # Reset environment
        reset_timer.start()
        obs, info = train_env.reset(regenerate_rail=True, regenerate_schedule=True)
        reset_timer.end()

        # if train_params.render:
        #     env_renderer.set_new_rail()

        score = 0
        nb_steps = 0
        actions_taken = []

        # Build initial agent-specific observations
        for agent in train_env.get_agent_handles():
            if train_env.obs_builder.check_is_observation_valid(obs[agent]):
                agent_obs[agent] = train_env.obs_builder.get_normalized_observation(obs[agent])
                agent_prev_obs[agent] = agent_obs[agent].copy()

        # Run episode
        train_logs = defaultdict(list)
        for step in range(max_steps - 1):
            inference_timer.start()
            for agent in train_env.get_agent_handles():
                if info['action_required'][agent]:
                    update_values[agent] = True
                    action = policy.act(agent_obs[agent], eps=eps)
                    action_count[action] += 1
                    actions_taken.append(action)
                else:
                    # An action is not required if the train hasn't joined the railway network,
                    # if it already reached its target, or if is currently malfunctioning.
                    update_values[agent] = False
                    action = 0
                action_dict.update({agent: action})
            inference_timer.end()

            # Environment step
            step_timer.start()
            next_obs, all_rewards, done, info = train_env.step(action_dict)
            step_timer.end()

            # Render an episode at some interval
            if train_params.render and episode_idx % train_params.checkpoint_interval == 0:
                render_env(train_env)
                #  env_renderer.render_env(
                #     show=True,
                #     frames=False,
                #     show_observations=False,
                #     show_predictions=False,
                # )

            # Update replay buffer and train agent
            for agent in train_env.get_agent_handles():
                if update_values[agent] or done['__all__']:
                    # Only learn from timesteps where somethings happened
                    learn_timer.start()
                    update_logs = policy.step(
                        agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent], agent_obs[agent],
                        done[agent])
                    for log_key, logs in update_logs.items():
                        train_logs[log_key].append(logs.cpu().detach().numpy())
                    learn_timer.end()

                    agent_prev_obs[agent] = agent_obs[agent].copy()
                    agent_prev_action[agent] = action_dict[agent]

                # Preprocess the new observations
                if train_env.obs_builder.check_is_observation_valid(next_obs[agent]):
                    preproc_timer.start()
                    agent_obs[agent] = train_env.obs_builder.get_normalized_observation(next_obs[agent])
                    preproc_timer.end()

                score += all_rewards[agent]

            nb_steps = step

            if done['__all__']:
                break

        # Epsilon decay
        eps = max(train_params.eps_end, train_params.eps_decay * eps)

        # Collect information about training
        tasks_finished = sum(done[idx] for idx in train_env.get_agent_handles())
        completion = tasks_finished / max(1, train_env.get_num_agents())
        normalized_score = score / (max_steps * train_env.get_num_agents())
        action_probs = action_count / max(1, np.sum(action_count))

        scores_window.append(normalized_score)
        completion_window.append(completion)
        smoothed_normalized_score = np.mean(scores_window)
        smoothed_completion = np.mean(completion_window)

        # Reset action counts
        action_count = [0] * action_size

        # Print logs
        print(f'\r🚂 Episode {episode_idx}'
              f'\t🏆 Score: {normalized_score + 1.0:.3f} (Avg: {smoothed_normalized_score + 1.0:.3f})'
              f'\t💯 Done: {100 * completion:5.1f}% (Avg: {100 * smoothed_completion:5.1f}%)'
              f'\t🎲 Epsilon: {eps:.3f}'
              f'\t🔀 Action Probs: {format_action_prob(action_probs)}', end="")

        if 'rnd/intrinsic_rewards' in train_logs:
            print(f'\t🤔 Curiosity Reward: {np.concatenate(train_logs["rnd/intrinsic_rewards"]).mean():6.4f}', end="")

        # Test policy with env tests
        if episode_idx % train_params.checkpoint_interval == 0:
            test_policy(tester, policy, observation_builder, eps)

        # Evaluate policy and log results at some interval
        if episode_idx % train_params.checkpoint_interval == 0 and train_params.n_evaluation_episodes > 0:
            scores, completions, nb_steps_eval = eval_policy(eval_env, policy, train_params, obs_params)

            smoothing = 0.1
            smoothed_eval_normalized_score = \
                smoothed_eval_normalized_score * smoothing + np.mean(scores) * (1.0 - smoothing)
            smoothed_eval_completion = \
                smoothed_eval_completion * smoothing + np.mean(completions) * (1.0 - smoothing)

            writer.add_scalar("evaluation/smoothed_score", smoothed_eval_normalized_score, episode_idx)
            writer.add_scalar("evaluation/smoothed_completion", smoothed_eval_completion, episode_idx)
            writer.add_scalar("evaluation/scores/min", np.min(scores), episode_idx)
            writer.add_scalar("evaluation/scores/max", np.max(scores), episode_idx)
            writer.add_scalar("evaluation/scores/mean", np.mean(scores), episode_idx)
            writer.add_scalar("evaluation/scores/std", np.std(scores), episode_idx)
            writer.add_histogram("evaluation/scores", np.array(scores), episode_idx)
            writer.add_scalar("evaluation/completions/min", np.min(completions), episode_idx)
            writer.add_scalar("evaluation/completions/max", np.max(completions), episode_idx)
            writer.add_scalar("evaluation/completions/mean", np.mean(completions), episode_idx)
            writer.add_scalar("evaluation/completions/std", np.std(completions), episode_idx)
            writer.add_histogram("evaluation/completions", np.array(completions), episode_idx)
            writer.add_scalar("evaluation/nb_steps/min", np.min(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps/max", np.max(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps/mean", np.mean(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps/std", np.std(nb_steps_eval), episode_idx)
            writer.add_histogram("evaluation/nb_steps", np.array(nb_steps_eval), episode_idx)

        # Save logs to tensorboard
        writer.add_histogram("actions/distribution", np.array(actions_taken), episode_idx)
        writer.add_scalar("actions/nothing", action_probs[RailEnvActions.DO_NOTHING], episode_idx)
        writer.add_scalar("actions/left", action_probs[RailEnvActions.MOVE_LEFT], episode_idx)
        writer.add_scalar("actions/forward", action_probs[RailEnvActions.MOVE_FORWARD], episode_idx)
        writer.add_scalar("actions/right", action_probs[RailEnvActions.MOVE_RIGHT], episode_idx)
        writer.add_scalar("actions/stop", action_probs[RailEnvActions.STOP_MOVING], episode_idx)
        writer.add_scalar("training/score", normalized_score, episode_idx)
        writer.add_scalar("training/smoothed_score", smoothed_normalized_score, episode_idx)
        writer.add_scalar("training/completion", np.mean(completion), episode_idx)
        writer.add_scalar("training/smoothed_completion", np.mean(smoothed_completion), episode_idx)
        writer.add_scalar("training/nb_steps", nb_steps, episode_idx)
        writer.add_scalar("training/epsilon", eps, episode_idx)
        writer.add_scalar("training/buffer_size", len(policy.memory), episode_idx)
        for log_key, logs in train_logs.items():
            logs = np.concatenate([np.atleast_1d(x) for x in logs])
            writer.add_scalar(f'training/{log_key}/min', np.min(logs), episode_idx)
            writer.add_scalar(f'training/{log_key}/max', np.max(logs), episode_idx)
            writer.add_scalar(f'training/{log_key}/mean', np.mean(logs), episode_idx)
            writer.add_scalar(f'training/{log_key}/std', np.std(logs), episode_idx)
        writer.add_scalar("timer/reset", reset_timer.get(), episode_idx)
        writer.add_scalar("timer/step", step_timer.get(), episode_idx)
        writer.add_scalar("timer/learn", learn_timer.get(), episode_idx)
        writer.add_scalar("timer/preproc", preproc_timer.get(), episode_idx)
        writer.add_scalar("timer/total", training_timer.get_current(), episode_idx)

        # Save checkpoint
        if episode_idx % train_params.checkpoint_interval == 0:
            os.makedirs(os.path.join(log_dir, 'ckpts'), exist_ok=True)
            checkpoint_path = os.path.join(log_dir, 'ckpts', f'ckpt-{episode_idx}.pth')
            torch.save(policy.qnetwork_local, checkpoint_path)
            print(f'💾 Checkpoint stored: {checkpoint_path}')


def test_policy(tester, policy, observation_builder, eps):
    # wrapper function for env testing
    def act_all(obs, reward, done, info, env):
        actions = {}
        for a in obs:
            if info['action_required'][a]:
                o = env.obs_builder.get_normalized_observation(obs[a])
                actions[a] = policy.act(o, eps=eps)
        return actions

    # run env tests
    _, test_results = tester.test_all(obs_builder=observation_builder, act_method=act_all)
    test_results_info = ""
    for test in test_results:
        t_results = test_results[test]
        for k in t_results:
            test_results_info += '✅' if t_results[k] else '❌'
    print(f'\t🎓 Tests: {test_results_info}', end="")


def eval_policy(env, policy, train_params, obs_params):
    max_steps = env._max_episode_steps

    action_dict = dict()
    scores = []
    completions = []
    nb_steps = []

    for episode_idx in range(train_params.n_evaluation_episodes):
        agent_obs = [None] * env.get_num_agents()
        score = 0.0
        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
        final_step = 0

        for step in range(max_steps - 1):
            for agent in env.get_agent_handles():
                if env.obs_builder.check_is_observation_valid(agent_obs[agent]):
                    agent_obs[agent] = env.obs_builder.get_normalized_observation(obs[agent])

                action = 0
                if info['action_required'][agent]:
                    if env.obs_builder.check_is_observation_valid(agent_obs[agent]):
                        action = policy.act(agent_obs[agent], eps=0.0)
                action_dict.update({agent: action})

            obs, all_rewards, done, info = env.step(action_dict)

            for agent in env.get_agent_handles():
                score += all_rewards[agent]

            final_step = step

            if done['__all__']:
                break

        normalized_score = score / (max_steps * env.get_num_agents())
        scores.append(normalized_score)

        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        completion = tasks_finished / max(1, env.get_num_agents())
        completions.append(completion)

        nb_steps.append(final_step)

    print(f"\t✅ Eval: score {np.mean(scores) + 1.0:.3f} done {np.mean(completions) * 100.0:.1f}%")

    return scores, completions, nb_steps


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", "--n_episodes", help="number of episodes to run", default=2500, type=int)
    parser.add_argument("-t", "--training_env_config", help="train config id (eg 0 for Test_0)", default=0, type=int)
    parser.add_argument("-e", "--evaluation_env_config", help="eval config id (eg 0 for Test_0)", default=0, type=int)
    parser.add_argument("--n_evaluation_episodes", help="number of evaluation episodes", default=50, type=int)
    parser.add_argument("--checkpoint_interval", help="checkpoint interval", default=100, type=int)
    parser.add_argument("--eps_start", help="max exploration", default=1.0, type=float)
    parser.add_argument("--eps_end", help="min exploration", default=0.01, type=float)
    parser.add_argument("--eps_decay", help="exploration decay", default=0.99, type=float)
    parser.add_argument("--buffer_size", help="replay buffer size", default=int(1e5), type=int)
    parser.add_argument("--buffer_min_size", help="min buffer size to start training", default=0, type=int)
    parser.add_argument("--batch_size", help="minibatch size", default=128, type=int)
    parser.add_argument("--gamma", help="discount factor", default=0.99, type=float)
    parser.add_argument("--tau", help="soft update of target parameters", default=1e-3, type=float)
    parser.add_argument("--learning_rate", help="learning rate", default=0.5e-4, type=float)
    parser.add_argument('--hidden_layers', nargs='+', default=[128, 128], type=int)
    parser.add_argument("--update_every", help="how often to update the network", default=8, type=int)
    parser.add_argument("--use_gpu", help="use GPU if available", default=False, type=bool)
    parser.add_argument("--num_threads", help="number of threads PyTorch can use", default=1, type=int)
    parser.add_argument("--render", help="render 1 episode in 100", default=False, action='store_true')
    parser.add_argument("--load_policy", help="policy filename (reference) to load", default="", type=str)
    parser.add_argument("--max_depth", help="max depth", default=2, type=int)
    parser.add_argument('--rnd_hidden_layers', nargs='+', default=[128, 128], type=int)
    parser.add_argument('--rnd_intrinsic_reward_weight', default=1.0, type=float)

    training_params = parser.parse_args()
    env_params = [
        {
            # Test_0
            "n_agents": 5,
            "x_dim": 25,
            "y_dim": 25,
            "n_cities": 2,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 50,
            "seed": 0
        },
        {
            # Test_1
            "n_agents": 10,
            "x_dim": 30,
            "y_dim": 30,
            "n_cities": 2,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 100,
            "seed": 0
        },
        {
            # Test_2
            "n_agents": 20,
            "x_dim": 30,
            "y_dim": 30,
            "n_cities": 3,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 200,
            "seed": 0
        },
    ]

    obs_params = {
        "observation_tree_depth": training_params.max_depth,
        "observation_radius": 10,
        "observation_max_path_depth": 30
    }


    def check_env_config(id):
        if id >= len(env_params) or id < 0:
            print("\n🛑 Invalid environment configuration, only Test_0 to Test_{} are supported.".format(
                len(env_params) - 1))
            exit(1)


    check_env_config(training_params.training_env_config)
    check_env_config(training_params.evaluation_env_config)

    training_env_params = env_params[training_params.training_env_config]
    evaluation_env_params = env_params[training_params.evaluation_env_config]

    print("\nTraining parameters:")
    pprint(vars(training_params))
    print("\nTraining environment parameters (Test_{}):".format(training_params.training_env_config))
    pprint(training_env_params)
    print("\nEvaluation environment parameters (Test_{}):".format(training_params.evaluation_env_config))
    pprint(evaluation_env_params)
    print("\nObservation parameters:")
    pprint(obs_params)

    os.environ["OMP_NUM_THREADS"] = str(training_params.num_threads)

    train_agent(DDDQNPolicy,
                training_params,
                Namespace(**training_env_params),
                Namespace(**evaluation_env_params),
                Namespace(**obs_params))
