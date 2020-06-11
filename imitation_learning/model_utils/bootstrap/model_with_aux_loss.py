# Copyright (c) Facebook, Inc. and its affiliates.
import os
from argparse import Namespace
from typing import Optional, Tuple

import torch

import utils
from ml_logger.logbook import LogBook
from model_utils.bootstrap import basic_model as basic_bootstrap
from model_utils.bootstrap.common import create_multi_env_replay_buffer, make_logbook
from model_utils.env import make_train_and_eval_envs
from model_utils.model import ActorModel, DiscriminatorModel, DynamicsModel, RewardModel
from sacae.decoder import make_decoder
from sacae.encoder import make_encoder


def bootstrap_envs_and_buffer(args: Namespace):
    """Method to bootstrap the envs, buffer and related objects"""

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
    ) = basic_bootstrap.bootstrap_envs_and_buffer(args=args)

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
    logbook: Optional[LogBook],
):
    """Method to bootstrap the models and optimizers"""

    phi_encoder = make_encoder(
        encoder_type=args.encoder_type,
        obs_shape=obs_shape,
        feature_dim=args.encoder_feature_dim,
        num_layers=args.num_layers,
        num_filters=args.num_filters,
    ).to(device)

    params_to_add = list(phi_encoder.parameters())

    phi_dynamics_model = DynamicsModel(
        representation_size=args.encoder_feature_dim, action_shape=action_size
    ).to(device)

    params_to_add += list(phi_dynamics_model.parameters())

    if args.use_discriminator:
        discriminator = DiscriminatorModel(
            representation_size=args.encoder_feature_dim, num_envs=args.num_train_envs
        ).to(device)
        discriminator_opt = torch.optim.Adam(
            list(discriminator.parameters()), lr=args.lr,
        )
    else:
        discriminator = None
        discriminator_opt = None

    if args.use_reward:
        reward_model = RewardModel(
            representation_size=args.encoder_feature_dim, action_shape=action_size
        ).to(device)
        params_to_add += list(reward_model.parameters())
    else:
        reward_model = None

    if args.use_actor:
        actor_model = ActorModel(
            representation_size=args.encoder_feature_dim, action_shape=action_size
        ).to(device)
        params_to_add += list(actor_model.parameters())

    else:
        actor_model = None

    def flatten(_list):
        # Taken from https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
        return [item for sublist in _list for item in sublist]

    if args.use_single_encoder_decoder:
        eta_encoders = None
        eta_dynamics_models = None
        decoders = [
            make_decoder(
                decoder_type=args.decoder_type,
                obs_shape=obs_shape,
                feature_dim=args.encoder_feature_dim,
                num_layers=args.num_layers,
                num_filters=args.num_filters,
            ).to(device)
        ]
        params_to_add += flatten([list(decoder.parameters()) for decoder in decoders])
    else:
        num_models_to_make = args.num_train_envs

        eta_encoders = [
            make_encoder(
                encoder_type=args.encoder_type,
                obs_shape=obs_shape,
                feature_dim=args.encoder_feature_dim,
                num_layers=args.num_layers,
                num_filters=args.num_filters,
            ).to(device)
            for i in range(num_models_to_make)
        ]
        eta_dynamics_models = [
            DynamicsModel(
                representation_size=args.encoder_feature_dim, action_shape=action_size
            ).to(device)
            for i in range(num_models_to_make)
        ]

        decoders = [
            make_decoder(
                decoder_type=args.decoder_type,
                obs_shape=obs_shape,
                feature_dim=args.encoder_feature_dim * 2,
                num_layers=args.num_layers,
                num_filters=args.num_filters,
            ).to(device)
            for i in range(num_models_to_make)
        ]

        params_to_add += (
            flatten([list(decoder.parameters()) for decoder in decoders])
            + flatten(
                [
                    list(dynamics_model.parameters())
                    for dynamics_model in eta_dynamics_models
                ]
            )
            + flatten([list(encoder.parameters()) for encoder in eta_encoders])
        )

    opt = torch.optim.Adam(list(params_to_add), lr=args.lr,)

    if logbook:
        logbook.write_message(f"args.load_model: {args.load_model}")
    if args.load_model:

        if os.path.exists(args.load_model_path):

            if args.load_model_path.endswith(".pt"):
                path_to_load_model = args.load_model_path
            else:
                epochs = [
                    int(x.split(".pt")[0]) for x in os.listdir(args.load_model_path)
                ]
                epoch_to_select = max(epochs)
                path_to_load_model = os.path.join(
                    args.load_model_path, f"{epoch_to_select}.pt"
                )

            state_dict = torch.load(path_to_load_model)

            phi_encoder.load_state_dict(state_dict["phi_encoder"])
            phi_dynamics_model.load_state_dict(state_dict["phi_dynamics_model"])
            reward_model.load_state_dict(state_dict["reward_model"])
            for idx, eta_encoder in enumerate(eta_encoders):
                eta_encoder.load_state_dict(state_dict["eta_encoders"][idx])
            for idx, eta_dynamics_model in enumerate(eta_dynamics_models):
                eta_dynamics_model.load_state_dict(
                    state_dict["eta_dynamics_models"][idx]
                )
            for idx, decoder in enumerate(decoders):
                decoder.load_state_dict(state_dict["decoders"][idx])
            opt.load_state_dict(state_dict["opt"])

            if logbook:
                logbook.write_message_logs(
                    {"message": f"Loading model from {path_to_load_model}"}
                )

    model = {
        "phi_encoder": phi_encoder,
        "phi_dynamics_model": phi_dynamics_model,
        "reward_model": reward_model,
        "eta_encoders": eta_encoders,
        "eta_dynamics_models": eta_dynamics_models,
        "decoders": decoders,
        "discriminator_model": discriminator,
        "opt": opt,
        "discriminator_opt": discriminator_opt,
        "actor_model": actor_model,
    }
    return model
