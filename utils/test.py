import yaml
import os
import numpy as np
import json
from PIL import Image

from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator

from flatland.envs.observations import GlobalObsForRailEnv
from flatland.utils.rendertools import RenderTool

def random_action(obs, reward, done, info, env):
    return {i: np.random.randint(0, 5) for i in obs}

def always_forward(obs, reward, done, info, env):
    return {i: 2 for i in obs}

class FlatlandTester:
    comparison = {
        "random": {
            "obs_builder": GlobalObsForRailEnv(),
            "act_method": random_action
        },
        "always_forward": {
            "obs_builder": GlobalObsForRailEnv(),
            "act_method": always_forward
        }
    }

    def __init__(self, num_trials=100, output_path='output'):
        self.num_trials = num_trials
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open('tests.yaml', 'r') as f:
            self.configs = yaml.load(f, Loader=yaml.Loader)

    def test_env(self, name, obs_builder):
        config = self.configs[name]
        return self._test_env(config, obs_builder)

    def _test_env(self, config, obs_builder):
        if 'env_file' in config['rail_generator']:
            load_from_package = config['rail_generator'].get('from_package')
            path = config['rail_generator']['env_file']
            rail_generator = rail_from_file(path, load_from_package=load_from_package)
        elif 'specs_file' in config['rail_generator']:
            path = config['rail_generator']['specs_file']
            with open(path, 'r') as f:
                specs = json.load(f)
            rail_generator = sparse_rail_generator(specs, grid_mode=False,seed=0,**config['rail_generator'])
        else:
            rail_generator = sparse_rail_generator(
                grid_mode=False,
                seed=0,
                **config['rail_generator']
            )
        env = RailEnv(
            rail_generator=rail_generator,
            obs_builder_object=obs_builder,
            **config['environment']
        )
        env.reset()
        return env

    def _max_steps(self, config):
        return int(4 * 2 * (
            config['environment']['width'] +
            config['environment']['height'] +
            config['environment']['number_of_agents'] / config['rail_generator']['max_num_cities']
        ))

    def _save_rendering(self, env, filename):
        if self.output_path is not None:
            self.env_renderer = RenderTool(env, gl="PGL")
            self.env_renderer.render_env(show_observations=False)
            image = Image.fromarray(self.env_renderer.get_image())
            path = os.path.join(self.output_path, filename)
            image.save(path)

    def _save_results(self, results, filename):
        if self.output_path is not None:
            path = os.path.join(self.output_path, filename)
            with open(path, 'w') as f:
                json.dump(results, f, indent=4)

    def tests(self, obs_builder):
        for name in self.configs:
            config = self.configs[name]
            env = self._test_env(config, obs_builder)
            max_steps = self._max_steps(config)
            new_score = False
            scores = config.get('scores', {})
            for cname in self.comparison:
                if cname not in scores:
                    new_score = True
                    comparison = self.comparison[cname]
                    env = self._test_env(config, comparison['obs_builder'])
                    scores[cname] = self.score_env(env, comparison['act_method'], max_steps)
            if new_score:
                config['scores'] = scores
                with open('tests.yaml', 'w') as f:
                    yaml.dump(self.configs, f)
            yield env, config['scores'], max_steps, name

    def score_test(self, name, obs_builder, act_method):
        config = self.configs[name]
        env = self._test_env(config, obs_builder)
        max_steps = self._max_steps(config)
        self._save_rendering(env, 'env_' + name + '.png')
        score = self.score_env(env, act_method, max_steps)
        return score, config['scores']

    def score_env(self, env, act_method, max_steps):
        for t in range(self.num_trials):
            trial_reward = 0
            action = {}
            for s in range(max_steps):
                obs, rewards, done, info = env.step(action)
                trial_reward += np.sum(list(rewards.values()))
                action = act_method(obs, rewards, done, info, env)
                if done['__all__']:
                    break
            env.reset()
        return float(trial_reward / self.num_trials)

    def test(self, name, obs_builder, act_method, log=False):
        score, scores = self.score_test(name, obs_builder, act_method)
        results = {k: score >= scores[k] for k in scores}
        if log:
            print('-', name)
            for k in results:
                print('✅' if results[k] else '❌', k)
        passed = bool(np.all(list(results.values())))
        self._save_results(results, 'results_' + name + '.json')
        return passed, results

    def test_all(self, obs_builder, act_method, log=False):
        if log:
            print("testing " + str(len(self.configs.keys())) + " envs")
        results = {}
        passed = True
        for env, scores, max_steps, name in self.tests(obs_builder):
            p, r = self.test(name, obs_builder, act_method, log)
            results[name] = r
            passed = passed and p
        self._save_results(results, 'test_results.json')
        return passed, results
