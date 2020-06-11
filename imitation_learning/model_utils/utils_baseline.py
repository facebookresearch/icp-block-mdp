# Copyright (c) Facebook, Inc. and its affiliates.

from argparse import Namespace
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

import utils
from ml_logger.logbook import LogBook

Envs_Type = Union[List[utils.FrameStack], utils.VecPyTorch]


def train_model(
    args: Namespace,
    logbook: LogBook,
    device: torch.device,
    train_envs: Envs_Type,
    eval_envs: Envs_Type,
    obs_shape: Tuple[int, int, int],
    action_size: int,
    train_replay_buffer: utils.MultiEnvReplayBuffer,
    eval_replay_buffer: utils.MultiEnvReplayBuffer,
    logging_dict: Dict,
    models: Dict,
    num_iters: Optional[int] = None,
    iteration_start_index: int = 0,
):
    # Iteration start index is used for logging

    if num_iters is None:
        num_iters = args.num_iters
    for iteration in range(iteration_start_index, num_iters + iteration_start_index):
        train_metrics = train_iter(
            args=args, train_replay_buffer=train_replay_buffer, models=models
        )

        if iteration % args.log_interval == 0:

            test_metrics = eval_iter(
                args=args, eval_replay_buffer=eval_replay_buffer, models=models
            )

            metrics_to_log = {"steps": iteration}
            for key in train_metrics:
                if train_metrics[key] is not None:
                    metrics_to_log[f"train_{key}"] = train_metrics[key].item()

            for key in test_metrics:
                if test_metrics[key] is not None:
                    metrics_to_log[f"test_{key}"] = test_metrics[key].item()

            logbook.write_metric_log(metric=metrics_to_log)


def compute_encoder_and_dynamics_loss(
    args, obs, action, next_obs, encoder, dynamics_model
):
    state = encoder(obs)
    pred_next_state = dynamics_model(state, action)
    true_next_state = encoder(next_obs).detach()
    penalty = None
    return (F.mse_loss(pred_next_state, true_next_state), state)


def train_iter(
    args: Namespace, train_replay_buffer: utils.MultiEnvReplayBuffer, models: Dict,
):
    metrics = get_default_metrics_dict()
    for env_idx in range(args.num_train_envs):
        obses, actions, rewards, next_obses, not_dones = train_replay_buffer.sample(
            env_idx
        )
        if len(models["decoders"]) == 1:
            current_decoder = models["decoders"][0]  # only use one decoder
        else:
            current_decoder = models["decoders"][env_idx]

        current_eta_encoder = None
        current_eta_dynamics_model = None
        if models["eta_encoders"] is not None:
            if len(models["eta_encoders"]) == 1:
                current_eta_encoder = models["eta_encoders"][0]
            else:
                current_eta_encoder = models["eta_encoders"][env_idx]
        if models["eta_dynamics_models"] is not None:
            if len(models["eta_dynamics_models"]) == 1:
                current_eta_dynamics_model = models["eta_dynamics_models"][0]
            else:
                current_eta_dynamics_model = models["eta_dynamics_models"][env_idx]

        current_metrics = compute_loss_using_buffer(
            args=args,
            obs=obses,
            actions=actions,
            next_obs=next_obses,
            rewards=rewards,
            shared_models=models,
            eta_encoder=current_eta_encoder,
            eta_dynamics_model=current_eta_dynamics_model,
            decoder=current_decoder,
            env_idx=env_idx,
        )

        for key in current_metrics:
            if current_metrics[key] is not None:
                if metrics[key] is None:
                    metrics[key] = current_metrics[key]
                else:
                    metrics[key] += current_metrics[key]

    if not models["discriminator_model"] is None:
        models["discriminator_opt"].zero_grad()
        metrics["discriminator_error"].backward()
        models["discriminator_opt"].step()

    list_of_loss_for_opt = [
        "model_error_in_latent_state",
        "reward_error",
        "decoder_error",
        "actor_error",
        "encoder_discriminator_error",
    ]

    loss = 0
    for loss_for_opt in list_of_loss_for_opt:
        if metrics[loss_for_opt] is not None:
            loss += metrics[loss_for_opt]

    models["opt"].zero_grad()
    loss.backward()
    models["opt"].step()

    return metrics


def eval_iter(
    args: Namespace, eval_replay_buffer: utils.MultiEnvReplayBuffer, models: Dict
):
    metrics = get_default_metrics_dict()

    with torch.no_grad():

        for i in range(args.num_eval_envs):
            obses, actions, rewards, next_obses, not_dones = eval_replay_buffer.sample(
                i
            )

            current_metrics = compute_loss_using_buffer(
                args=args,
                obs=obses,
                actions=actions,
                next_obs=next_obses,
                rewards=rewards,
                shared_models=models,
                eta_encoder=None,
                eta_dynamics_model=None,
                decoder=None,
                env_idx=None,
            )

            for key in current_metrics:
                if current_metrics[key] is not None:
                    if metrics[key] is None:
                        metrics[key] = current_metrics[key]
                    else:
                        metrics[key] += current_metrics[key]

    return metrics


def compute_loss_using_buffer(
    args,
    obs,
    actions,
    next_obs,
    rewards,
    shared_models,
    eta_encoder,
    eta_dynamics_model,
    decoder,
    env_idx,
):
    phi_encoder = shared_models["phi_encoder"]
    phi_dynamics_model = shared_models["phi_dynamics_model"]
    reward_model = shared_models["reward_model"]
    actor_model = shared_models["actor_model"]
    discriminator_model = shared_models["discriminator_model"]

    metrics = get_default_metrics_dict()
    (
        metrics["model_error_in_latent_state"],
        latent_state,
    ) = compute_encoder_and_dynamics_loss(
        args=args,
        obs=obs,
        action=actions,
        next_obs=next_obs,
        encoder=phi_encoder,
        dynamics_model=phi_dynamics_model,
    )

    if reward_model is not None:
        predicted_reward = reward_model(latent_state, actions)
        metrics["reward_error"] = F.mse_loss(predicted_reward, rewards)

    if actor_model is not None:
        predicted_action = actor_model(latent_state)
        metrics["actor_error"] = F.mse_loss(predicted_action, actions)

    if discriminator_model is not None:
        discriminator_pred = discriminator_model(latent_state.detach())
        batch_size = discriminator_pred.shape[0]
        device = discriminator_pred.device
        if env_idx is not None:
            metrics["discriminator_error"] = F.cross_entropy(
                discriminator_pred,
                torch.ones(batch_size, dtype=torch.long, device=device) * env_idx,
            )
        discriminator_pred = discriminator_model(latent_state)
        metrics["encoder_discriminator_error"] = torch.mean(
            F.softmax(discriminator_pred, dim=1)
            * F.log_softmax(discriminator_pred, dim=1)
        )

    if eta_encoder is None or eta_dynamics_model is None:
        inp_for_decoder = phi_encoder(next_obs)

    else:

        _, eta_state = compute_encoder_and_dynamics_loss(
            args=args,
            obs=obs,
            action=actions,
            next_obs=next_obs,
            encoder=eta_encoder,
            dynamics_model=eta_dynamics_model,
        )

        inp_for_decoder = torch.cat(
            [encoder(next_obs) for encoder in [phi_encoder, eta_encoder]], dim=1,
        )

    if decoder is not None:
        pred_next_obs = decoder(inp_for_decoder)
        metrics["decoder_error"] = F.mse_loss(pred_next_obs, next_obs)

    return metrics


def get_default_metrics_dict():
    return {
        "model_error_in_latent_state": None,
        "reward_error": None,
        "actor_error": None,
        "decoder_error": None,
        "encoder_discriminator_error": None,
        "discriminator_error": None,
    }
