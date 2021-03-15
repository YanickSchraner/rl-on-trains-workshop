from base64 import b64encode
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from utils.utils import set_seed


class MultiAgentTrainer():
    """A class to train agents in a multi-agent environment"""

    def __init__(self, env, agents, reset_agents, seed=None):
        # Save parameters
        self.env, self.agents, self.seed = env, agents, seed

        # Create log of rewards and reset agents
        self.rewards_log = {key: [] for key in self.agents.keys()}
        self.reset(reset_agents)

    def reset(self, reset_agents):
        # Set seed for reproducibility
        if self.seed is not None:
            set_seed(self.env, self.seed)

        # Reset agents and clear log of rewards
        for key, agent in self.agents.items():
            self.rewards_log[key].clear()

            if reset_agents:
                agent.reset()

    def train(self, n_steps):
        # Reset env. and get initial observations
        states = self.env.reset()
        
        # Set greedy flag
        for key, agent in self.agents.items():
            agent.is_greedy = False

        for i in tqdm(range(n_steps), 'Training agents'):
            # Select actions based on current states
            actions = {key: agent.act(states[key]) for key, agent in self.agents.items()}

            # Perform the selected action
            next_states, rewards, dones, _ = self.env.step(actions)

            # Learn from experience
            for key, agent in self.agents.items():
                agent.learn(states[key], actions[key], rewards[key], next_states[key], dones[key])
                self.rewards_log[key].append(rewards[key])
            states = next_states


def test_agents(env, agents, n_steps, seed=None):
    """Function to test agents"""

    # Initialization
    if seed is not None:
        set_seed(env, seed=seed)
    states = env.reset()
    rewards_log = defaultdict(list)

    # Set greedy flag
    for key, agent in agents.items():
        agent.is_greedy = True

    for _ in tqdm(range(n_steps), 'Testing agents'):
        # Select actions based on current states
        actions = {key: agent.act(states[key]) for key, agent in agents.items()}

        # Perform the selected action
        next_states, rewards, dones, _ = env.step(actions)

        # Save rewards
        for key, reward in rewards.items():
            rewards_log[key].append(reward)

        states = next_states

    return rewards_log

