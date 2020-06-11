# Copyright (c) Facebook, Inc. and its affiliates.
import os
from argparse import Namespace
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

import utils
from ml_logger.logbook import LogBook

Envs_Type = Union[List[utils.FrameStack], utils.VecPyTorch]


def run_model_with_aux_loss(
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
    phi_encoder: torch.nn.Module,
    phi_dynamics_model: torch.nn.Module,
    reward_model: torch.nn.Module,
    eta_encoders: List[torch.nn.Module],
    eta_dynamics_models: List[torch.nn.Module],
    decoders: List[torch.nn.Module],
    opt: torch.optim.Optimizer,
    num_iters: Optional[int] = None,
    iteration_start_index: int = 0,
):
    # Iteration start index is used during IRM training

    if num_iters is None:
        num_iters = args.num_iters
    for iteration in range(iteration_start_index, num_iters + iteration_start_index):
        (
            model_error_in_latent_state,
            reward_error,
            model_error_in_eta_state,
            decoder_error,
        ) = train_iter(
            args=args,
            train_replay_buffer=train_replay_buffer,
            phi_encoder=phi_encoder,
            phi_dynamics_model=phi_dynamics_model,
            reward_model=reward_model,
            eta_encoders=eta_encoders,
            eta_dynamics_models=eta_dynamics_models,
            decoders=decoders,
            opt=opt,
        )

        if iteration % args.log_interval == 0:

            print(
                f"Iteration {iteration}: Mean train set model error: {model_error_in_latent_state.mean()}, decoding error: {decoder_error.mean()}%%"
            )

            (
                test_model_error_in_latent_state,
                test_reward_error,
                test_model_error_in_eta_state,
                test_decoder_error,
            ) = eval_iter(
                args=args,
                eval_replay_buffer=eval_replay_buffer,
                phi_encoder=phi_encoder,
                phi_dynamics_model=phi_dynamics_model,
                reward_model=reward_model,
                eta_encoders=eta_encoders,
                eta_dynamics_models=eta_dynamics_models,
                decoders=decoders,
                opt=opt,
            )

            # test_model_error_in_latent_state = 0
            # test_reward_error = 0
            # test_model_error_in_eta_state = 0
            # test_decoder_error = 0

            current_metrics = {
                "steps": iteration,
                "model_error_in_latent_state": model_error_in_latent_state.item(),
                "model_error_in_eta_state": model_error_in_eta_state.item(),
                "reward_error": reward_error.item(),
                "decoding_error": decoder_error.item(),
                "test_model_error_in_latent_state": test_model_error_in_latent_state.item(),
                # "test_model_error_in_eta_state": test_model_error_in_eta_state.item(),
                "test_reward_error": test_reward_error.item(),
                # "test_decoding_error": test_decoder_error.item(),
            }

            for key in current_metrics:
                logging_dict[key].append(current_metrics[key])

            # logging_dict["eval_model_error"].append(test_error.item())
            print(
                f"Mean test set model error: {test_model_error_in_latent_state.item()}"
            )

            logbook.write_metric_logs(metrics=current_metrics)

            torch.save(logging_dict, os.path.join(args.work_dir, "logging_dict.pt"))

            if args.save_model:

                state_dict = {
                    "phi_encoder": phi_encoder.state_dict(),
                    "phi_dynamics_model": phi_dynamics_model.state_dict(),
                    "reward_model": reward_model.state_dict(),
                    "eta_encoders": [
                        eta_encoder.state_dict() for eta_encoder in eta_encoders
                    ],
                    "eta_dynamics_models": [
                        eta_dynamics_model.state_dict()
                        for eta_dynamics_model in eta_dynamics_models
                    ],
                    "decoders": [decoder.state_dict() for decoder in decoders],
                    "opt": opt.state_dict(),
                    "epoch": iteration,
                }

                if args.save_model_path.endswith(".pt"):
                    path_to_save_model = args.save_model_path
                else:
                    path_to_save_model = os.path.join(
                        args.save_model_path, f"{iteration}.pt"
                    )

                torch.save(state_dict, path_to_save_model)

                print(f"Saved model at {path_to_save_model}")


def compute_encoder_and_dynamics_loss(obs, action, next_obs, encoder, dynamics_model):
    state = encoder(obs)
    pred_next_state = dynamics_model(state, action)
    true_next_state = encoder(next_obs).detach()
    return F.mse_loss(pred_next_state, true_next_state), state


def train_iter(
    args: Namespace,
    train_replay_buffer: utils.MultiEnvReplayBuffer,
    phi_encoder: nn.Module,
    phi_dynamics_model: nn.Module,
    reward_model: nn.Module,
    eta_encoders: List[nn.Module],
    eta_dynamics_models: List[nn.Module],
    decoders: List[nn.Module],
    opt: torch.optim,
):
    should_use_one_decoder = args.one_decoder or len(decoders) == 1
    model_error_in_latent_state = 0
    reward_error = 0
    model_error_in_eta_state = 0
    decoder_error = 0
    for i in range(args.num_train_envs):
        obses, actions, rewards, next_obses, not_dones = train_replay_buffer.sample(i)
        if should_use_one_decoder:
            current_decoder = decoders[0]  # only use one decoder
        else:
            current_decoder = decoders[i]

        current_eta_encoder = None
        current_eta_dynamics_model = None
        if eta_encoders is not None:
            if len(eta_encoders) == 1:
                current_eta_encoder = eta_encoders[0]
            else:
                current_eta_encoder = eta_encoders[i]
        if eta_dynamics_models is not None:
            if len(eta_dynamics_models) == 1:
                current_eta_dynamics_model = eta_dynamics_models[0]
            else:
                current_eta_dynamics_model = eta_dynamics_models[i]

        (
            current_error_in_latent_state,
            current_reward_error,
            current_error_in_eta_state,
            current_decoder_error,
        ) = compute_loss_using_buffer(
            obs=obses,
            actions=actions,
            next_obs=next_obses,
            rewards=rewards,
            phi_encoder=phi_encoder,
            phi_dynamics_model=phi_dynamics_model,
            reward_model=reward_model,
            eta_encoder=current_eta_encoder,
            eta_dynamics_model=current_eta_dynamics_model,
            decoder=current_decoder,
        )

        model_error_in_latent_state += current_error_in_latent_state
        reward_error += current_reward_error
        model_error_in_eta_state += current_error_in_eta_state
        decoder_error += current_decoder_error

    opt.zero_grad()
    (
        model_error_in_latent_state
        + reward_error
        + model_error_in_eta_state
        + decoder_error
    ).backward()
    opt.step()

    return (
        model_error_in_latent_state,
        reward_error,
        model_error_in_eta_state,
        decoder_error,
    )


def eval_iter(
    args: Namespace,
    eval_replay_buffer: utils.MultiEnvReplayBuffer,
    phi_encoder: nn.Module,
    phi_dynamics_model: nn.Module,
    reward_model: nn.Module,
    eta_encoders: List[nn.Module],
    eta_dynamics_models: List[nn.Module],
    decoders: List[nn.Module],
    opt: torch.optim,
):

    should_use_one_decoder = args.one_decoder or len(decoders) == 1

    model_error_in_latent_state = 0
    reward_error = 0
    model_error_in_eta_state = 0
    decoder_error = 0

    with torch.no_grad():

        for i in range(args.num_eval_envs):
            obses, actions, rewards, next_obses, not_dones = eval_replay_buffer.sample(
                i
            )

            if should_use_one_decoder:
                current_decoder = decoders[0]  # only use one decoder
            else:
                current_decoder = decoders[i]

            # (
            #     current_error_in_latent_state,
            #     current_reward_error,
            #     current_error_in_eta_state,
            #     current_decoder_error,
            # ) = compute_loss_using_buffer(
            #     obs=obses,
            #     actions=actions,
            #     next_obs=next_obses,
            #     rewards=rewards,
            #     phi_encoder=phi_encoder,
            #     phi_dynamics_model=phi_dynamics_model,
            #     reward_model=reward_model,
            #     eta_encoder=eta_encoders[i],
            #     eta_dynamics_model=eta_dynamics_models[i],
            #     decoder=current_decoder,
            # )

            (
                current_error_in_latent_state,
                current_reward_error,
                current_error_in_eta_state,
                current_decoder_error,
            ) = compute_loss_using_buffer(
                obs=obses,
                actions=actions,
                next_obs=next_obses,
                rewards=rewards,
                phi_encoder=phi_encoder,
                phi_dynamics_model=phi_dynamics_model,
                reward_model=reward_model,
                eta_encoder=None,
                eta_dynamics_model=None,
                decoder=None,
            )

            model_error_in_latent_state += current_error_in_latent_state
            reward_error += current_reward_error
            model_error_in_eta_state += current_error_in_eta_state
            decoder_error += current_decoder_error

    return (
        model_error_in_latent_state,
        reward_error,
        model_error_in_eta_state,
        decoder_error,
    )


def compute_loss_using_buffer(
    obs,
    actions,
    next_obs,
    rewards,
    phi_encoder,
    phi_dynamics_model,
    reward_model,
    eta_encoder,
    eta_dynamics_model,
    decoder,
):
    (current_error_in_latent_state, latent_state,) = compute_encoder_and_dynamics_loss(
        obs=obs,
        action=actions,
        next_obs=next_obs,
        encoder=phi_encoder,
        dynamics_model=phi_dynamics_model,
    )

    predicted_reward = reward_model(latent_state, actions)
    current_reward_error = F.mse_loss(predicted_reward, rewards)

    if eta_encoder is None or eta_dynamics_model is None:
        current_error_in_eta_state = torch.tensor(0)
        inp_for_decoder = phi_encoder(next_obs)

    else:

        current_error_in_eta_state, eta_state = compute_encoder_and_dynamics_loss(
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

        current_decoder_error = F.mse_loss(pred_next_obs, next_obs)

    else:
        current_decoder_error = torch.tensor(0)

    return (
        current_error_in_latent_state,
        current_reward_error,
        current_error_in_eta_state,
        current_decoder_error,
    )
