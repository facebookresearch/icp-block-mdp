# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd, nn, optim
from torch.utils.data import DataLoader, Dataset

import utils
from model import Decoder, DynamicsModel, Encoder


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument("--domain_name", default="cheetah")
    parser.add_argument("--task_name", default="run")
    parser.add_argument("--image_size", default=84, type=int)
    parser.add_argument("--action_repeat", default=1, type=int)
    parser.add_argument("--frame_stack", default=3, type=int)
    parser.add_argument("--num_envs", default=2, type=int)
    # replay buffer
    parser.add_argument("--replay_buffer_capacity", default=1000000, type=int)
    parser.add_argument("--num_samples", default=50000, type=int)
    # training
    parser.add_argument("--num_iters", default=100000, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument(
        "--one_decoder", action="store_true", help="baseline with single decoder"
    )
    # encoder/decoder
    parser.add_argument("--encoder_type", default="identity", type=str)
    parser.add_argument("--encoder_feature_dim", default=50, type=int)
    parser.add_argument("--encoder_lr", default=1e-3, type=float)
    parser.add_argument("--encoder_tau", default=0.05, type=float)
    parser.add_argument("--num_layers", default=4, type=int)
    parser.add_argument("--num_filters", default=32, type=int)
    # misc
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--exp_name", default="local", type=str)
    parser.add_argument("--log_interval", default=1000, type=int)
    parser.add_argument("--work_dir", default=".", type=str)
    parser.add_argument("--save_tb", default=False, action="store_true")
    parser.add_argument("--save_model", default=False, action="store_true")
    parser.add_argument("--save_buffer", default=False, action="store_true")
    parser.add_argument("--save_video", default=False, action="store_true")

    args = parser.parse_args()
    return args


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args.work_dir = os.path.join(
        args.work_dir,
        args.domain_name + "_" + args.task_name,
        args.exp_name,
        str(args.seed),
    )
    os.makedirs(args.work_dir, exist_ok=True)
    with open(os.path.join(args.work_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    train_envs = [
        utils.make_env(np.random.randint(0, 255), args) for i in range(args.num_envs)
    ]
    eval_envs = [utils.make_env(np.random.randint(0, 255), args) for i in range(5)]
    print("Train env backgrounds: ", [train_env.bg_color for train_env in train_envs])
    print("Eval env backgrounds: ", [eval_env.bg_color for eval_env in eval_envs])

    obs_shape = train_envs[0].observation_space.shape
    action_size = train_envs[0].action_space.shape[0]

    phi = Encoder(obs_shape, args.encoder_feature_dim).to(device)
    model = DynamicsModel(args.encoder_feature_dim, action_size).to(device)
    decoders = [
        Decoder(obs_shape, args.encoder_feature_dim).to(device)
        for i in range(args.num_envs)
    ]
    opt = torch.optim.Adam(
        list(phi.parameters()) + list(model.parameters()), lr=args.lr
    )
    decoder_opt = torch.optim.Adam(
        np.concatenate([list(decoder.parameters()) for decoder in decoders]), lr=args.lr
    )

    train_replay_buffer = utils.ReplayBuffer(
        obs_shape=train_envs[0].observation_space.shape,
        action_shape=train_envs[0].action_space.shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
    )
    eval_replay_buffer = utils.ReplayBuffer(
        obs_shape=train_envs[0].observation_space.shape,
        action_shape=train_envs[0].action_space.shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
    )

    logging_dict = {
        "model_error": [],
        "decoding_error": [],
        "eval_model_error": [],
        "steps": [],
    }

    # collect data across environments
    for env_id in range(args.num_envs):
        train_replay_buffer = utils.collect_random_data(
            train_envs[env_id],
            env_id,
            args.num_samples,
            train_replay_buffer,
            save_video=args.save_video,
        )
        eval_replay_buffer = utils.collect_random_data(
            eval_envs[env_id], env_id, args.num_samples, eval_replay_buffer
        )

    # Train loop
    for iteration in range(args.num_iters):
        model_error = 0
        decoder_error = 0
        for i in range(args.num_envs):
            obses, actions, rewards, next_obses, not_dones = train_replay_buffer.sample(
                i
            )
            latent = phi(obses)
            pred_next_latent = model(latent, actions)
            true_next_latent = phi(next_obses).detach()
            error_e = F.mse_loss(pred_next_latent, true_next_latent)
            model_error += error_e

            if args.one_decoder:
                pred_next_obses = decoders[0](pred_next_latent)  # only use one decoder
            else:
                pred_next_obses = decoders[i](pred_next_latent)
            decoder_error_e = F.mse_loss(pred_next_obses, next_obses)
            decoder_error += decoder_error_e

        opt.zero_grad()
        model_error.backward(retain_graph=True)
        opt.step()

        decoder_opt.zero_grad()
        decoder_error.backward()
        decoder_opt.step()
        if iteration % args.log_interval == 0:
            with torch.no_grad():
                logging_dict["steps"].append(iteration)
                logging_dict["model_error"].append(model_error.item())
                logging_dict["decoding_error"].append(decoder_error.item())
                print(
                    f"Iteration {iteration}: Mean train set model error: {model_error.mean()}, decoding error: {decoder_error.mean()}%%"
                )

                # Evaluate on test environment
                (
                    obses,
                    actions,
                    rewards,
                    next_obses,
                    not_dones,
                ) = eval_replay_buffer.sample()
                with torch.no_grad():
                    latent = phi(obses)
                    pred_next_latent = model(latent, actions)
                    true_next_latent = phi(next_obses).detach()
                    test_error = F.mse_loss(pred_next_latent, true_next_latent)
                logging_dict["eval_model_error"].append(test_error.item())
                print(f"Mean test set error: {test_error}")
            torch.save(logging_dict, os.path.join(args.work_dir, "logging_dict.pt"))


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
