# Copyright (c) Facebook, Inc. and its affiliates.
"""Method to train the one encoder baseline for imitation learning"""
import json
import os
from argparse import Namespace
from time import time

import torch

import utils
from argument_parser import parse_args
from model_utils.bootstrap.common import create_multi_env_replay_buffer, make_logbook
from model_utils.env import make_fns_to_make_train_and_eval_envs
from sacae.vec_logger import VecLogger
from sacae_utils import bootstrap as sacae_bootstrap


def bootstrap_envs_and_buffer(args: Namespace):
    """Method to bootstrap the envs, buffer and related objects"""

    logbook = make_logbook(args=args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    utils.make_dir(args.work_dir)

    with open(os.path.join(args.work_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    dummy_env = utils.make_dummy_env(args=args)

    pixel_space_obs = dummy_env.env.env._get_observation_space_for_pixel_space(
        args.image_size, args.image_size
    )
    state_space_obs = dummy_env.env.env._get_observation_space_for_state_space()

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
    }

    return (
        logbook,
        device,
        vec_train_envs,
        vec_eval_envs,
        state_space_obs,
        pixel_space_obs,
        action_size,
        train_replay_buffer,
        eval_replay_buffer,
        logging_dict,
        max_episode_steps,
    )


def bootstrap_agent(args: Namespace, obs_shape, action_size, device):

    video_dir, model_dir, buffer_dir, video = sacae_bootstrap.make_dirs_and_recorders(
        args=args
    )

    agent = sacae_bootstrap.make_expert(
        obs_shape=obs_shape, action_shape=(action_size,), args=args, device=device,
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


def main():
    args = parse_args(should_use_model=True, should_use_rl=True)

    args.load_model = True
    args.return_both_pixel_and_state = True
    args.change_angle = True
    args.num_train_envs = 2
    args.num_eval_envs = 1

    (
        logbook,
        device,
        vec_train_envs,
        vec_eval_envs,
        state_space_obs,
        pixel_space_obs,
        action_size,
        train_replay_buffer,
        eval_replay_buffer,
        logging_dict,
        max_episode_steps,
    ) = bootstrap_envs_and_buffer(args)

    args.encoder_type = "identity"
    args.decoder_type = "identity"
    (video_dir, model_dir, buffer_dir, video, device, agent, L,) = bootstrap_agent(
        args, state_space_obs.shape, action_size, device
    )

    model_dir, step = args.load_model_path.rsplit("_", 1)
    agent.load(
        model_dir=model_dir, step=step,
    )

    start = time()
    # collect data across environments

    train_replay_buffer = utils.collect_both_state_and_obs_using_policy_vec(
        vec_env=vec_train_envs,
        num_samples=50000,
        replay_buffer=train_replay_buffer,
        policy=agent,
        save_video=args.save_video,
    )

    train_replay_buffer.save(f"{args.save_buffer_path}/train")

    eval_replay_buffer = utils.collect_both_state_and_obs_using_policy_vec(
        vec_env=vec_eval_envs,
        num_samples=50000,
        replay_buffer=eval_replay_buffer,
        policy=agent,
        save_video=False,
    )

    eval_replay_buffer.save(f"{args.save_buffer_path}/eval")

    end = time()
    print(f" Time to collect {args.num_samples} datapoints = {end - start}")


if __name__ == "__main__":
    main()
