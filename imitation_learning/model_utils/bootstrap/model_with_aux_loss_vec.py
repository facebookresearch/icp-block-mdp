# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os
from argparse import Namespace

import torch

import utils
from model_utils.bootstrap.common import (create_multi_env_replay_buffer,
                                          make_logbook)
from model_utils.env import make_fns_to_make_train_and_eval_envs


def bootstrap_envs_and_buffer(args: Namespace):
    """Method to bootstrap the envs, buffer and related objects"""

    logbook = make_logbook(args=args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    utils.make_dir(args.work_dir)

    with open(os.path.join(args.work_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    dummy_env = utils.make_dummy_env(args=args)

    obs_shape = dummy_env.observation_space.shape
    action_size = dummy_env.action_space.shape[0]

    train_replay_buffer = create_multi_env_replay_buffer(
        args=args, env=dummy_env, device=device, num_envs=args.num_train_envs
    )

    eval_replay_buffer = create_multi_env_replay_buffer(
        args=args, env=dummy_env, device=device, num_envs=args.num_eval_envs
    )

    (
        fns_to_make_train_envs,
        fns_to_make_eval_envs,
    ) = make_fns_to_make_train_and_eval_envs(args=args)

    max_episode_steps = dummy_env._max_episode_steps

    vec_train_envs = utils.make_vec_envs(
        fns_to_make_envs=fns_to_make_train_envs, device=None,
    )

    vec_eval_envs = utils.make_vec_envs(
        fns_to_make_envs=fns_to_make_eval_envs, device=None,
    )

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
        "discriminator_loss": [],
        "encoder_discriminator_loss": [],
        "test_encoder_discriminator_loss": [],
        "actor_error": [],
        "test_actor_error": [],
        "discriminator_error": [],
        "encoder_discriminator_error": [],
        "test_encoder_discriminator_error": [],
    }

    return (
        logbook,
        device,
        vec_train_envs,
        vec_eval_envs,
        obs_shape,
        action_size,
        train_replay_buffer,
        eval_replay_buffer,
        logging_dict,
        max_episode_steps,
    )
