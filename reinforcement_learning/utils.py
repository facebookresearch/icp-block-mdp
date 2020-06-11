# Copyright (c) Facebook, Inc. and its affiliates.

import math
import os
import random
from collections import deque

import gym
import numpy as np
import torch
import torch.nn.functional as F
from dm_control import suite
from numpy import linalg as LA
from torch import distributions as pyd
from torch import nn

import dmc2gym


def create_orthonormal_matrix(p):
    # create the positive definite matrix P
    A = np.random.rand(p, p)
    P = (A + np.transpose(A)) / 2 + p * np.eye(p)

    # get the subset of its eigenvectors
    vals, vecs = LA.eig(P)
    w = vecs[:, 0:p]
    return w


def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == "ball_in_cup_catch":
        domain_name = "ball_in_cup"
        task_name = "catch"
    else:
        domain_name = cfg.env.split("_")[0]
        task_name = "_".join(cfg.env.split("_")[1:])

    env = suite.load(domain_name, task_name)
    obs_space = int(sum([np.prod(s.shape) for s in env.observation_spec().values()]))
    train_factors = [np.eye(cfg.noise_dims) + i for i in range(cfg.num_train_envs)]
    test_factors = [create_orthonormal_matrix(cfg.noise_dims)]
    # train_factors = [1, 2, 3]
    # test_factors = [4]
    train_envs = [
        dmc2gym.make(
            domain_name=domain_name,
            task_name=task_name,
            noise=cfg.noise,
            mult_factor=train_factors[idx],
            idx=idx,
            seed=cfg.seed,
            visualize_reward=True,
        )
        for idx in range(cfg.num_train_envs)
    ]
    test_envs = [
        dmc2gym.make(
            domain_name=domain_name,
            task_name=task_name,
            noise=cfg.noise,
            mult_factor=test_factors[idx],
            idx=idx + cfg.num_train_envs,
            seed=cfg.seed,
            visualize_reward=True,
        )
        for idx in range(len(test_factors))
    ]
    [env.seed(cfg.seed) for env in train_envs]
    assert train_envs[0].action_space.low.min() >= -1
    assert train_envs[0].action_space.high.max() <= 1

    return train_envs, test_envs


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None
    ):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()
