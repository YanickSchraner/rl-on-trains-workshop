# Below are implementations of some standard RL agents
from agents import Agent


class RandomAgent(Agent):
    """Random agent"""

    def act(self, state):
        return self.env.action_space.sample()

    def reset(self):
        pass

    def learn(self, state, action, reward, next_state, done):
        raise NotImplementedError
