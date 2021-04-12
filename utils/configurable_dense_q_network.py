import random
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from collections import deque

import gym.spaces as spaces
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor, LongTensor


class DenseQNetwork(nn.Module):
    """
    A dense Q-network for OpenAI Gym Environments
    The network flattens the obs/action spaces and adds dense layers in between
    """

    def __init__(self, observation_space, action_space, hidden_layers=[]):
        # Create network
        super().__init__()  # Initialize module

        self.input_size = observation_space
        self.output_size = action_space
        self.hidden_layers = hidden_layers

        self.network = nn.Sequential()
        hidden_layers = hidden_layers + [self.output_size]
        for i, hidden_size in enumerate(hidden_layers):
            # Create layer
            in_features = self.input_size if i == 0 else hidden_layers[i - 1]
            out_features = hidden_layers[i]
            layer = nn.Linear(in_features, out_features)

            # Add layer + activation
            if i > 0:
                self.network.add_module('dense_act_{}'.format(i), nn.ReLU())
            self.network.add_module('dense_{}'.format(i + 1), layer)

    def forward(self, states):
        return self.network(states)


class DQNFactoryTemplate():
    """
    A template class to generate custom Q-networks and their optimizers
    """

    def create_qnetwork(self, target_qnetwork):
        # Should return network + optimizer
        raise NotImplementedError


class DenseQNetworkFactory(DQNFactoryTemplate):
    """
    A Q-network factory for dense Q-networks
    """

    def __init__(self, env, hidden_layers=[]):
        self.env = env
        self.hidden_layers = hidden_layers

    def create_qnetwork(self, target_qnetwork):
        network = DenseQNetwork(self.env, self.hidden_layers)
        optimizer = optim.Adam(network.parameters())
        return network, optimizer
