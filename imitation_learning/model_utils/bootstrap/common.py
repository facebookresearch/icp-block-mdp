# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
from argparse import Namespace

import torch

import utils
from ml_logger.logbook import LogBook
from ml_logger.logbook import make_config as make_logbook_config


def make_logbook(args: Namespace) -> LogBook:
    logbook_config = make_logbook_config(
        logger_file_path=args.logger_file_path, id="0",
    )

    logbook = LogBook(config=logbook_config)

    logbook.write_config_log(config=vars(args))
    return logbook


def create_multi_env_replay_buffer(
    args: argparse.Namespace, env: utils.FrameStack, device: torch.device, num_envs: int
) -> utils.ReplayBuffer:
    """"Method to create a multi env replay buffer"""
    return utils.MultiEnvReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        num_envs=num_envs,
    )


def create_replay_buffer(
    args: argparse.Namespace, env: utils.FrameStack, device: torch.device
) -> utils.ReplayBuffer:
    """"Method to create a replay buffer"""
    return utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
    )
