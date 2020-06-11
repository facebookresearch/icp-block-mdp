# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import json
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd, nn, optim
from torch.utils.data import DataLoader, Dataset

import argument_parser
import utils
from main import make_fns_to_make_train_and_eval_envs
from main_aux import bootstrap_models_and_optimizers, bootstrap_setup
from ml_logger.logbook import LogBook
from ml_logger.logbook import make_config as make_logbook_config
from model import Decoder, DynamicsModel, Encoder, RewardModel
from rl.logger import Logger
from sac_ae.sac_ae import SacAeAgent

# from video import VideoRecorder


def compute_encoder_and_dynamics_loss(obs, action, next_obs, encoder, dynamics_model):
    state = encoder(obs)
    pred_next_state = dynamics_model(state, action)
    true_next_state = encoder(next_obs).detach()
    return F.mse_loss(pred_next_state, true_next_state), state


def bootstrap_setup_for_rl(args: argparse.Namespace):
    """Method to bootstrap the setup"""

    utils.set_seed_everywhere(args.seed)

    (
        logbook,
        device,
        train_envs,
        eval_envs,
        obs_shape,
        action_size,
        train_replay_buffer,
        eval_replay_buffer,
        logging_dict,
    ) = bootstrap_setup(args)

    args.video_dir = utils.make_dir(os.path.join(args.work_dir, "video"))
    args.model_dir = utils.make_dir(os.path.join(args.work_dir, "model"))
    args.buffer_dir = utils.make_dir(os.path.join(args.work_dir, "buffer"))

    # video = VideoRecorder(video_dir if args.save_video else None)

    logging_dict = {
        "steps": [],
        "model_error_in_latent_state": [],
        "model_error_in_eta_state": [],
        "reward_error": [],
        "decoding_error": [],
        "test_model_error_in_latent_state": [],
        "test_model_error_in_eta_state": [],
        "test_reward_error": [],
        "test_decoding_error": [],
    }

    logger = Logger(args.work_dir, use_tb=args.save_tb, logbook=logbook)

    # train_envs =  utils.make_vec_envs(envs = train_envs,
    #               device=None,
    #               num_frame_stack=args.frame_stack)

    # eval_envs =  utils.make_vec_envs(envs = eval_envs,
    #               device=None,
    #               num_frame_stack=args.frame_stack)

    (
        fns_to_make_train_envs,
        fns_to_make_eval_envs,
    ) = make_fns_to_make_train_and_eval_envs(args=args)

    max_episode_steps = train_envs[0]._max_episode_steps

    train_envs = utils.make_vec_envs(
        fns_to_make_envs=fns_to_make_train_envs,
        device=None,
        num_frame_stack=args.frame_stack,
    )

    eval_envs = utils.make_vec_envs(
        fns_to_make_envs=fns_to_make_eval_envs,
        device=None,
        num_frame_stack=args.frame_stack,
    )

    return (
        logbook,
        device,
        train_envs,
        eval_envs,
        obs_shape,
        action_size,
        train_replay_buffer,
        eval_replay_buffer,
        logging_dict,
        logger,
        max_episode_steps,
    )


def bootstrap_agent(
    obs_shape: Tuple[int, int, int],
    action_shape,
    args: argparse.Namespace,
    device: torch.device,
    phi_encoder: torch.nn.Module,
) -> SacAeAgent:
    if args.agent == "sac_ae":
        return SacAeAgent(
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
            phi_encoder=phi_encoder,
        )
    else:
        assert "agent is not supported: %s" % args.agent
