# Copyright (c) Facebook, Inc. and its affiliates.
from argument_parser import parse_args
from sacae_utils.bootstrap import bootstrap_expert
from sacae_utils.utils import train_agent


def main():
    args = parse_args(should_use_model=False, should_use_rl=True)
    args.encoder_type = "identity"
    args.decoder_type = "identity"
    args.num_train_envs = 1
    args.num_eval_envs = 1

    (
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
    ) = bootstrap_expert(args=args)

    train_agent(
        args=args,
        vec_train_envs=vec_train_envs,
        vec_eval_envs=vec_eval_envs,
        L=L,
        agent=agent,
        video=video,
        model_dir=model_dir,
        train_replay_buffer=replay_buffer,
        eval_replay_buffer=None,
        buffer_dir=buffer_dir,
        max_episode_steps=max_episode_steps,
    )


if __name__ == "__main__":
    main()
