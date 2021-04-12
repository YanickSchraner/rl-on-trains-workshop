from typing import List

import torch
from torch.functional import F

from utils.dddqn_policy import DDDQNPolicy


class MLP(torch.nn.Module):
    def __init__(self, input_size: int, output_sizes: List[int], activate_final: bool = False):
        super().__init__()
        self._layers = torch.nn.ModuleList()
        for output_size in output_sizes:
            self._layers.append(torch.nn.Linear(in_features=input_size, out_features=output_size))
            input_size = output_size
        self._activate_final = activate_final

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mlp_out = inputs
        for i, layer in enumerate(self._layers):
            mlp_out = layer(mlp_out)
            if i < len(self._layers) - 1 or self._activate_final:
                mlp_out = F.relu(mlp_out, inplace=True)
        return mlp_out


class DDDQNRNDPolicy(DDDQNPolicy):
    """Dueling Double DQN policy with Random Network Distillation"""

    def __init__(self, state_size, action_size, parameters, evaluation_mode=False):
        super().__init__(state_size, action_size, parameters, evaluation_mode)
        self.predictor_network = MLP(state_size, parameters.rnd_hidden_layers, activate_final=False).to(self.device)
        self.target_network = MLP(state_size, parameters.rnd_hidden_layers, activate_final=False).to(self.device)
        self.intrinsic_reward_weight = parameters.rnd_intrinsic_reward_weight
        if not self.evaluation_mode:
            self.optimizer.add_param_group({'params': self.predictor_network.parameters()})

    def _learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample()

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        if self.double_dqn:
            # Double DQN
            q_best_action = self.qnetwork_local(next_states).max(1)[1]
            q_targets_next = self.qnetwork_target(next_states).gather(1, q_best_action.unsqueeze(-1))
        else:
            # DQN
            q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(-1)

        # Compute intrinsic rewards
        predicted_features = self.predictor_network(next_states)
        with torch.no_grad():
            target_features = self.target_network(next_states)
        extrinsic_rewards = rewards
        intrinsic_rewards = torch.mean(torch.square(target_features - predicted_features), dim=-1, keepdim=True)
        rewards = extrinsic_rewards + self.intrinsic_reward_weight * intrinsic_rewards

        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Compute loss
        dqn_loss = F.mse_loss(q_expected, q_targets)

        # Compute rand loss
        rnd_loss = F.mse_loss(predicted_features, target_features)
        loss = dqn_loss + rnd_loss

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        return {
            'dqn/loss': dqn_loss,
            'rnd/loss': rnd_loss,
            'rnd/extrinsic_rewards': extrinsic_rewards,
            'rnd/intrinsic_rewards': intrinsic_rewards,
            'rnd/rewards': rewards,
        }
