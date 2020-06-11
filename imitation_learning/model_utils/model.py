# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn as nn


class DynamicsModel(nn.Module):
    def __init__(self, representation_size, action_shape):
        super().__init__()

        self.action_linear = nn.Linear(action_shape, representation_size)
        self.trunk = nn.Sequential(
            nn.Linear(representation_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, representation_size),
        )

    def forward(self, state, action):
        action_emb = self.action_linear(action)
        return self.trunk(torch.cat([state, action_emb], dim=-1))


class RewardModel(nn.Module):
    def __init__(self, representation_size, action_shape):
        super().__init__()

        self.action_linear = nn.Linear(action_shape, representation_size)
        self.trunk = nn.Sequential(
            nn.Linear(representation_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state, action):
        action_emb = self.action_linear(action)
        return self.trunk(torch.cat([state, action_emb], dim=-1))


class ActorModel(nn.Module):
    def __init__(self, representation_size, action_shape):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(representation_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_shape),
        )

    def forward(self, state):
        return self.trunk(state)


class DiscriminatorModel(nn.Module):
    def __init__(self, representation_size, num_envs):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(representation_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_envs),
        )

    def forward(self, state):
        return self.trunk(state)
