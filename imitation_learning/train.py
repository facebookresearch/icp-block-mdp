# Copyright (c) Facebook, Inc. and its affiliates.
from argparse import Namespace
from time import time

import argument_parser
from model_utils.bootstrap.model_with_aux_loss import bootstrap_models_and_optimizers
from model_utils.bootstrap.model_with_aux_loss_vec import bootstrap_envs_and_buffer
from model_utils.utils_baseline import train_model


def main(args: Namespace) -> None:

    (
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
    ) = bootstrap_envs_and_buffer(args)

    train_replay_buffer.load(f"{args.load_buffer_path}/train")

    eval_replay_buffer.load(f"{args.load_buffer_path}/eval")

    args.use_actor = True

    models = bootstrap_models_and_optimizers(
        args=args,
        obs_shape=obs_shape,
        action_size=action_size,
        device=device,
        logbook=logbook,
    )

    # Train loop
    train_model(
        args=args,
        logbook=logbook,
        device=device,
        train_envs=vec_train_envs,
        eval_envs=vec_eval_envs,
        obs_shape=obs_shape,
        action_size=action_size,
        models=models,
        train_replay_buffer=train_replay_buffer,
        eval_replay_buffer=eval_replay_buffer,
        logging_dict=logging_dict,
        num_iters=args.num_iters,
    )


if __name__ == "__main__":
    args = argument_parser.parse_args(should_use_model=True, should_use_rl=False)
    main(args)
