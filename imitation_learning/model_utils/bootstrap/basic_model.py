# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os
from argparse import Namespace
from typing import Tuple

import numpy as np
import torch

import utils
from model_utils.bootstrap.common import create_multi_env_replay_buffer, make_logbook
from model_utils.env import make_train_and_eval_envs
from model_utils.model import DynamicsModel
from sacae.decoder import make_decoder
from sacae.encoder import make_encoder


def bootstrap_envs_and_buffer(args: Namespace):
    """Method to bootstrap the envs, buffer and related objects"""

    logbook = make_logbook(args=args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    args.work_dir = os.path.join(
        args.work_dir,
        args.domain_name + "_" + args.task_name,
        args.exp_name,
        str(args.seed),
    )

    utils.make_dir(args.work_dir)

    with open(os.path.join(args.work_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    train_envs, eval_envs = make_train_and_eval_envs(args=args)
    # print('Train env backgrounds: ', [train_env.bg_color for train_env in train_envs])
    # print('Eval env backgrounds: ', [eval_env.bg_color for eval_env in eval_envs])

    dummy_env = train_envs[0]

    obs_shape = dummy_env.observation_space.shape
    action_size = dummy_env.action_space.shape[0]

    train_replay_buffer = create_multi_env_replay_buffer(
        args=args, env=train_envs[0], device=device, num_envs=args.num_train_envs
    )

    eval_replay_buffer = create_multi_env_replay_buffer(
        args=args, env=train_envs[0], device=device, num_envs=args.num_eval_envs
    )

    logging_dict = {
        "model_error": [],
        "decoding_error": [],
        "eval_model_error": [],
        "steps": [],
    }

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
    )


def bootstrap_models_and_optimizers(
    args: Namespace,
    obs_shape: Tuple[int, int, int],
    action_size: int,
    device: torch.device,
):
    """Method to bootstrap the models and optimizers"""

    phi = make_encoder(
        encoder_type=args.encoder_type,
        obs_shape=obs_shape,
        feature_dim=args.encoder_feature_dim,
        num_layers=args.num_layers,
        num_filters=args.num_filters,
    ).to(device)

    dynamics_model = DynamicsModel(
        representation_size=args.encoder_feature_dim, action_shape=action_size
    ).to(device)
    decoders = [
        make_decoder(
            decoder_type=args.decoder_type,
            obs_shape=obs_shape,
            feature_dim=args.encoder_feature_dim,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
        ).to(device)
        for i in range(args.num_train_envs)
    ]
    opt = torch.optim.Adam(
        list(phi.parameters()) + list(dynamics_model.parameters()), lr=args.lr
    )
    decoder_opt = torch.optim.Adam(
        np.concatenate([list(decoder.parameters()) for decoder in decoders]), lr=args.lr
    )

    return phi, dynamics_model, decoders, opt, decoder_opt
