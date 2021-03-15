from abc import ABC, abstractmethod


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
