# Copyright (c) Facebook, Inc. and its affiliates.
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args(should_use_model=False, should_use_rl=False) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument(
        "--description",
        default="Testing that running the vectorized and non vectorized code gives similar performance",
    )
    parser.add_argument("--domain_name", default="cheetah")
    parser.add_argument("--task_name", default="run")
    parser.add_argument("--image_size", default=84, type=int)
    parser.add_argument("--action_repeat", default=4, type=int)
    parser.add_argument("--frame_stack", default=3, type=int)
    parser.add_argument("--num_train_envs", default=1, type=int)
    parser.add_argument("--num_eval_envs", default=1, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--change_angle", default=True, action="store_true")
    # replay buffer
    parser.add_argument("--replay_buffer_capacity", default=1000000, type=int)
    parser.add_argument("--num_samples", default=50000, type=int)

    # encoder/decoder
    parser.add_argument("--encoder_type", default="pixel", type=str)
    parser.add_argument("--decoder_type", default="pixel", type=str)
    parser.add_argument("--encoder_feature_dim", default=200, type=int)
    parser.add_argument("--encoder_lr", default=1e-3, type=float)
    parser.add_argument("--encoder_tau", default=0.05, type=float)
    parser.add_argument("--num_layers", default=6, type=int)
    parser.add_argument("--num_filters", default=32, type=int)
    parser.add_argument(
        "--use_single_encoder_decoder", default=False, action="store_true"
    )
    parser.add_argument("--use_reward", default=False, action="store_true")
    parser.add_argument("--use_actor", default=False, action="store_true")
    parser.add_argument("--use_discriminator", default=False, action="store_true")

    # misc
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--exp_name", default="testing", type=str)
    parser.add_argument("--log_interval", default=1000, type=int)
    parser.add_argument("--save_tb", default=False, action="store_true")
    parser.add_argument("--save_model", default=False, action="store_true")
    parser.add_argument("--load_model", default=False, action="store_true")
    parser.add_argument("--save_model_path", default="model", type=str)
    parser.add_argument("--load_model_path", default="model", type=str)
    parser.add_argument("--save_buffer", default=False, action="store_true")
    parser.add_argument("--save_buffer_path", default="buffer", type=str)
    parser.add_argument("--load_buffer_path", default="buffer", type=str)
    parser.add_argument("--save_video", default=False, action="store_true")

    parser.add_argument(
        "--work_dir", default=".", type=str,
    )

    if should_use_model:

        parser.add_argument("--render_size", default=64, type=int)

        # training
        parser.add_argument("--num_iters", default=100000, type=int)
        parser.add_argument("--lr", default=1e-3, type=float)

        parser.add_argument(
            "--one_decoder", action="store_true", help="baseline with single decoder"
        )

        # misc

        parser.add_argument(
            "--logger_file_path",
            default="/private/home/sodhani/projects/causRLity/log.jsonl",
            type=str,
        )

    if should_use_rl:

        # train
        parser.add_argument("--agent", default="sac_ae", type=str)
        parser.add_argument("--init_steps", default=1000, type=int)
        parser.add_argument("--num_train_steps", default=1000000, type=int)
        parser.add_argument("--hidden_dim", default=1024, type=int)

        # eval
        parser.add_argument("--eval_freq", default=10000, type=int)
        parser.add_argument("--num_eval_episodes", default=10, type=int)

        # critic
        parser.add_argument("--critic_lr", default=1e-3, type=float)
        parser.add_argument("--critic_beta", default=0.9, type=float)
        parser.add_argument("--critic_tau", default=0.01, type=float)
        parser.add_argument("--critic_target_update_freq", default=2, type=int)

        # actor
        parser.add_argument("--actor_lr", default=1e-3, type=float)
        parser.add_argument("--actor_beta", default=0.9, type=float)
        parser.add_argument("--actor_log_std_min", default=-10, type=float)
        parser.add_argument("--actor_log_std_max", default=2, type=float)
        parser.add_argument("--actor_update_freq", default=2, type=int)

        # decoder
        parser.add_argument("--decoder_lr", default=1e-3, type=float)
        parser.add_argument("--decoder_update_freq", default=1, type=int)
        parser.add_argument("--decoder_latent_lambda", default=1e-6, type=float)
        parser.add_argument("--decoder_weight_lambda", default=1e-7, type=float)
        # sac
        parser.add_argument("--discount", default=0.99, type=float)
        parser.add_argument("--init_temperature", default=0.1, type=float)
        parser.add_argument("--alpha_lr", default=1e-4, type=float)
        parser.add_argument("--alpha_beta", default=0.5, type=float)

    args = parser.parse_args()
    return args
