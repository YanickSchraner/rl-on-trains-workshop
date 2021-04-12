import random
from abc import abstractmethod, ABC
from collections import defaultdict, Counter
from logging import Logger
import torch
from gym.vector.utils import spaces
import matplotlib.pyplot as plt

from torch import LongTensor, Tensor


class NoLogger(Logger):
    def log_dict(self, global_step: int, values: dict) -> None:
        pass


class Agent(ABC):

    def __init__(self, env):
        self.env = env

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        pass


class DQNAgent(Agent):
    """
    Deep Q-network agent (DQN) implementation
    Uses a NN to approximate the Q-function, a replay memory buffer
    and a target network.
    """

    def __init__(self, env, dqn_factory, gamma, epsilon_start, epsilon_decay, epsilon_end, memory_size, batch_size,
                 target_update_interval, logger: Logger = None):
        # Save parameters
        self.env = env
        self.dqn_factory = dqn_factory  # Factory to create q-networks + optimizers
        self.gamma = gamma  # Discount factor
        self.epsilon_start = epsilon_start  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay after each episode
        self.epsilon_end = epsilon_end  # Minimum value
        self.memory_size = memory_size  # Size of the replay buffer
        self.batch_size = batch_size  # Batch size
        self.target_update_interval = target_update_interval  # Update rate
        self.is_greedy = False  # Does the agent behave greedily?
        self.logger = logger or NoLogger()

    def reset(self):
        # Create networks with episode counter to know when to update them
        self.qnetwork, self.optimizer = self.dqn_factory.create_qnetwork(target_qnetwork=False)
        self.target_qnetwork, _ = self.dqn_factory.create_qnetwork(target_qnetwork=True)
        self.num_episode = 0
        self.episode_reward = 0
        self.total_steps = 0

        # Reset exploration rate
        self.epsilon = self.epsilon_start
        self.epsilons = []

        # Create new replay memory
        self.memory = deque(maxlen=self.memory_size)

    def save(self, path):
        torch.save(self.qnetwork, path)

    def load(self, path):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        self.qnetwork = torch.load(path, map_location=device)

    def act(self, state):
        # Exploration rate
        epsilon = 0.01 if self.is_greedy else self.epsilon

        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            q_values = self.qnetwork([state])[0]
            return q_values.argmax().item()  # Greedy action

    def learn(self, state, action, reward, next_state, done):
        # Memorize experience
        self.memory.append((state, action, reward, next_state, done))
        self.episode_reward += reward
        self.total_steps += 1

        # End of episode
        if done:
            self.num_episode += 1  # Episode counter
            self.logger.log_dict(self.total_steps, {
                'episode_reward': self.episode_reward,
                'memory_size': len(self.memory),
            })
            self.epsilons.append(self.epsilon)  # Log epsilon value

            # Epislone decay
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)
            self.episode_reward = 0

        # Periodically update target network with current one
        if self.num_episode % self.target_update_interval == 0:
            self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())

        # Train when we have enough experiences in the replay memory
        if len(self.memory) > self.batch_size:
            # Sample batch of experience
            batch = random.sample(self.memory, self.batch_size)
            state, action, reward, next_state, done = zip(*batch)

            action = LongTensor(action)
            reward = Tensor(reward)
            done = Tensor(done)

            if torch.cuda.is_available():
                action = action.cuda()
                reward = reward.cuda()
                done = done.cuda()

            # Q-value for current state given current action
            q_values = self.qnetwork(state)
            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

            # Compute the TD target
            next_q_values = self.target_qnetwork(next_state)
            next_q_value = next_q_values.max(1)[0]

            td_target = reward + self.gamma * next_q_value * (1 - done)

            # Optimize quadratic loss
            loss = (q_value - td_target.detach()).pow(2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.logger.log_dict(self.total_steps, {
                'dqn/loss': loss.data.cpu().numpy(),
                'dqn/reward': reward.mean().data.cpu().numpy(),
            })

    def inspect_memory(self, top_n=10, max_col=80):
        # Functions to encode/decode states
        encode_state = lambda s: tuple(spaces.flatten(self.env.observation_space, s))
        decode_state = lambda s: spaces.unflatten(self.env.observation_space, s)

        # Function to create barchart from counter
        def count_barchart(counter, ax, xlabel=None, normalize=True):
            # Sort and extract key, counts
            sorted_tuples = counter.most_common()
            sorted_keys = [key for key, count in sorted_tuples]
            sorted_counts = [count for key, count in sorted_tuples]

            # Normalize counts
            if normalize:
                total = sum(counters['reward'].values())
                sorted_counts = [c / total for c in sorted_counts]

            # Plotting
            x_indexes = range(len(sorted_counts))
            ax.bar(x_indexes, sorted_counts)
            ax.set_xticks(x_indexes)
            ax.set_xticklabels(sorted_keys)
            ax.set_ylabel('proportion')
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            ax.set_title('Replay Memory')

        # Function to print top states from counter
        def top_states(counter):
            for i, (state, count) in enumerate(counter.most_common(top_n), 1):
                state_label = str(decode_state(state))
                state_label = state_label.replace('\n', ' ')
                state_label = state_label[:max_col] + '..' if len(state_label) > max_col else state_label
                print('{:>2}) Count: {} state: {}'.format(i, count, state_label))

        # Count statistics
        counters = defaultdict(Counter)
        for state, action, reward, next_state, done in self.memory:
            counters['state'][encode_state(state)] += 1
            counters['action'][action] += 1
            counters['reward'][reward] += 1
            counters['next_state'][encode_state(next_state)] += 1
            counters['done'][done] += 1

        # Plot reward/action
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        count_barchart(counters['reward'], ax1, 'rewards')
        count_barchart(counters['action'], ax2, 'actions')
        plt.plot()
        plt.show()

        # Print top states
        print('Top state:')
        top_states(counters['state'])
        print()

        print('Top next_state:')
        top_states(counters['next_state'])
        print()

        # Done signal
        print('Proportion of done: {:.2f}%'.format(100 * counters['done'][True] / sum(counters['done'].values())))
