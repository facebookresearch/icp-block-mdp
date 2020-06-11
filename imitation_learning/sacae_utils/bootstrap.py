# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json
import os
from argparse import Namespace
from typing import Optional, Union

import torch
from torch import nn

import dmc2gym
import utils
from ml_logger.logbook import LogBook
from ml_logger.logbook import make_config as make_logbook_config
from sacae import sacae, sacae_vec
from sacae.logger import Logger
from sacae.vec_logger import VecLogger
from sacae.video import VideoRecorder

AgentType = Union[sacae.SacAeAgent, sacae_vec.SacAeAgent]


def validate_env(env):
    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1


def make_dirs_and_recorders(args: Namespace):

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, "video"))
    model_dir = utils.make_dir(os.path.join(args.work_dir, args.save_model_path))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, args.save_buffer_path))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    return video_dir, model_dir, buffer_dir, video


def make_expert(
    obs_shape, action_shape, args: Namespace, device: torch.device,
) -> AgentType:

    return sacae_vec.SacAeAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        hidden_dim=args.hidden_dim,
        discount=args.discount,
        init_temperature=args.init_temperature,
        alpha_lr=args.alpha_lr,
        alpha_beta=args.alpha_beta,
        actor_lr=args.actor_lr,
        actor_beta=args.actor_beta,
        actor_log_std_min=args.actor_log_std_min,
        actor_log_std_max=args.actor_log_std_max,
        actor_update_freq=args.actor_update_freq,
        critic_lr=args.critic_lr,
        critic_beta=args.critic_beta,
        critic_tau=args.critic_tau,
        critic_target_update_freq=args.critic_target_update_freq,
        encoder_type=args.encoder_type,
        encoder_feature_dim=args.encoder_feature_dim,
        encoder_lr=args.encoder_lr,
        encoder_tau=args.encoder_tau,
        decoder_type=args.decoder_type,
        decoder_lr=args.decoder_lr,
        decoder_update_freq=args.decoder_update_freq,
        decoder_latent_lambda=args.decoder_latent_lambda,
        decoder_weight_lambda=args.decoder_weight_lambda,
        num_layers=args.num_layers,
        num_filters=args.num_filters,
    )


def bootstrap_expert(args: Namespace):

    utils.set_seed_everywhere(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fns_to_make_train_envs = [
        utils.fn_to_make_env(args=args, seed=seed, resource_files=None, camera_id=0)
        for seed in range(args.num_train_envs)
    ]

    fns_to_make_eval_envs = [
        utils.fn_to_make_env(args=args, seed=seed, resource_files=None, camera_id=0)
        for seed in range(args.num_eval_envs)
    ]

    vec_train_envs = utils.make_vec_envs(
        fns_to_make_envs=fns_to_make_train_envs, device=None
    )

    vec_eval_envs = utils.make_vec_envs(
        fns_to_make_envs=fns_to_make_eval_envs, device=None
    )

    dummy_env = utils.make_env(args, 0, resource_files=None, camera_id=0)

    video_dir, model_dir, buffer_dir, video = make_dirs_and_recorders(args=args)

    validate_env(dummy_env)

    replay_buffer = utils.MultiEnvReplayBuffer(
        obs_shape=dummy_env.observation_space.shape,
        action_shape=dummy_env.action_space.shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        num_envs=args.num_train_envs,
    )

    agent = make_expert(
        obs_shape=dummy_env.observation_space.shape,
        action_shape=dummy_env.action_space.shape,
        args=args,
        device=device,
    )

    L = VecLogger(args.work_dir, use_tb=args.save_tb, num_envs=args.num_train_envs)

    max_episode_steps = dummy_env._max_episode_steps
    return (
        vec_train_envs,
        vec_eval_envs,
        max_episode_steps,
        video_dir,
        model_dir,
        buffer_dir,
        video,
        device,
        replay_buffer,
        agent,
        L,
    )


def bootstrap_agent(args: Namespace, obs_shape, action_size, device, encoder):

    video_dir, model_dir, buffer_dir, video = make_dirs_and_recorders(args=args)

    agent = sacae_bootstrap.make_agent(
        obs_shape=obs_shape,
        action_shape=(action_size,),
        args=args,
        device=device,
        is_vec=True,
        is_irm=True,
        encoder=encoder,
    )

    L = VecLogger(args.work_dir, use_tb=args.save_tb, num_envs=args.num_train_envs)

    return (
        video_dir,
        model_dir,
        buffer_dir,
        video,
        device,
        agent,
        L,
    )
