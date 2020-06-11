# Copyright (c) Facebook, Inc. and its affiliates.
import time
from argparse import Namespace
from typing import Optional

import numpy as np
import torch

import utils
from sacae.sacae_vec import SacAeAgent
from sacae.vec_logger import VecLogger

# def evaluate_one_env_from_list_of_envs(
#     env: utils.FrameStack,
#     agent: sac_ae_vec.SacAeAgent,
#     video,
#     num_episodes: int,
#     L: vec_logger.VecLogger,
#     step: int,
#     env_idx: int,
# ):
#     """Evaluate one env from a list of envs"""
#     for i in range(num_episodes):
#         obs = env.reset()
#         video.init(enabled=(i == 0))
#         done = False
#         episode_reward = 0
#         while not done:
#             with utils.eval_mode(agent):
#                 action = agent.select_action(obs)
#             obs, reward, done, _ = env.step(action)
#             video.record(env)
#             episode_reward += reward

#         video.save("%d.mp4" % step)
#         L.log(f"eval/episode_reward", episode_reward, step, env_idx=env_idx)
#         L.dump(step, env_idx=env_idx)


# def evaluate_list_of_envs(
#     envs: List[utils.FrameStack],
#     agent: sac_ae_vec.SacAeAgent,
#     video,
#     num_episodes: int,
#     L: vec_logger.VecLogger,
#     step: int,
#     episode: int,
# ):
#     for env_idx, env in enumerate(envs):
#         L.log("eval/episode", episode, step, env_idx=env_idx)
#         evaluate_one_env_from_list_of_envs(
#             env, agent, video, num_episodes, L, step, env_idx
#         )


def evaluate_agent(
    vec_eval_envs: utils.VecPyTorch,
    agent: SacAeAgent,
    replay_buffer: utils.MultiEnvReplayBuffer,
    video,
    num_episodes: int,
    L: VecLogger,
    step: int,
    args: Namespace,
    max_episode_steps: int = 1000,
):
    num_envs = args.num_eval_envs
    mode = "eval"

    def make_vector_using_val(val):
        return np.full(num_envs, val)

    def make_tensor_using_fn(fn):
        return torch.tensor([fn() for _ in range(num_envs)])

    def make_tensor_using_val(val):
        return torch.tensor(make_vector_using_val(val))

    episode, episode_reward, done = [make_vector_using_val(x) for x in [0, 0.0, True]]

    obs = vec_eval_envs.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    episode_reward = make_vector_using_val(0.0)
    episode_step = make_vector_using_val(0)
    start_time = make_vector_using_val(time.time())

    should_add_reward = make_vector_using_val(1.0)
    # This masking array is used to make sure that no environment adds rewards for more than num_epsiodes

    for _ in range(num_episodes * max_episode_steps):

        with utils.eval_mode(agent):
            action = agent.sample_action(obs.float())

        next_obs, reward, done, _ = vec_eval_envs.step(action)

        for env_idx in range(num_envs):
            if done[env_idx] and episode[env_idx] <= num_episodes:
                done[env_idx] = False
                episode[env_idx] += 1

        should_add_reward = (episode < num_episodes).astype(float)

        reward = reward.numpy()[:, 0]
        episode_reward += reward * should_add_reward
        obs = next_obs
        if isinstance(obs, tuple):
            obs = obs[0]

        condition = episode_step + 1 == max_episode_steps
        done_bool = condition * 0 + (1 - condition) * done.astype(float)

        if replay_buffer is not None:
            for env_idx in range(num_envs):
                replay_buffer.add(
                    obs=obs[env_idx],
                    action=action[env_idx],
                    reward=reward[env_idx],
                    next_obs=next_obs[env_idx],
                    done=done_bool[env_idx],
                    env_id=env_idx,
                )

    (
        start_time,
        episode_reward,
        episode_step,
        episode,
        done,
        L,
    ) = log_metrics_and_update_state(
        num_envs=num_envs,
        step=step,
        mode=mode,
        L=L,
        start_time=start_time,
        episode_reward=episode_reward / num_episodes,
        episode=episode,
        episode_step=episode_step,
        done=done,
        should_log_env_idx=make_vector_using_val(True),
    )


def train_agent(
    args: Namespace,
    vec_train_envs: utils.VecPyTorch,
    vec_eval_envs: utils.VecPyTorch,
    L: VecLogger,
    agent: SacAeAgent,
    video,
    model_dir,
    train_replay_buffer: utils.MultiEnvReplayBuffer,
    eval_replay_buffer: utils.MultiEnvReplayBuffer,
    buffer_dir,
    max_episode_steps: int,
    num_train_steps: Optional[int] = None,
    step_start_index: int = 0,
    episode_start_index: int = 0,
):

    num_envs = args.num_train_envs

    def make_vector_using_val(val):
        return np.full(num_envs, val)

    def make_tensor_using_fn(fn):
        return torch.tensor([fn() for _ in range(num_envs)])

    def make_tensor_using_val(val):
        return torch.tensor(make_vector_using_val(val))

    if num_train_steps is None:
        num_train_steps = args.num_train_steps

    episode, episode_reward, done = [
        make_vector_using_val(x) for x in [episode_start_index, 0.0, True]
    ]

    obs = vec_train_envs.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    episode_reward = make_vector_using_val(0.0)
    episode_step = make_vector_using_val(0)
    start_time = make_vector_using_val(time.time())

    for step in range(step_start_index, step_start_index + num_train_steps):

        # evaluate agent periodically
        if step > 0 and args.eval_freq > 0 and step % args.eval_freq == 0:

            evaluate_agent(
                vec_eval_envs=vec_eval_envs,
                agent=agent,
                replay_buffer=eval_replay_buffer,
                video=video,
                num_episodes=args.num_eval_episodes,
                L=L,
                step=step,
                args=args,
                max_episode_steps=max_episode_steps,
            )
            if args.save_model:
                agent.save(model_dir, step)
            print(f"Saving the model at {model_dir} after {step} steps.")

        # sample action for data collection
        if step < args.init_steps:
            action = make_tensor_using_fn(vec_train_envs.action_space.sample)
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs.float())

        # run training update

        if step >= args.init_steps:
            num_updates = args.init_steps if step == args.init_steps else 1
            for _ in range(num_updates):
                for env_idx in range(args.num_train_envs):
                    agent.update(train_replay_buffer, L, step, env_idx=env_idx)

        next_obs, reward, done, _ = vec_train_envs.step(action)

        # allow infinit bootstrap
        condition = episode_step + 1 == max_episode_steps
        done_bool = condition * 0 + (1 - condition) * done.astype(float)
        reward = reward.numpy()[:, 0]
        episode_reward += reward

        for env_idx in range(args.num_train_envs):
            train_replay_buffer.add(
                obs=obs[env_idx],
                action=action[env_idx],
                reward=reward[env_idx],
                next_obs=next_obs[env_idx],
                done=done_bool[env_idx],
                env_id=env_idx,
            )

        (
            start_time,
            episode_reward,
            episode_step,
            episode,
            done,
            L,
        ) = log_metrics_and_update_state(
            num_envs=args.num_train_envs,
            step=step,
            mode="train",
            L=L,
            start_time=start_time,
            episode_reward=episode_reward,
            episode=episode,
            episode_step=episode_step,
            done=done,
            should_log_env_idx=done * (step > 0),
        )

        obs = next_obs
        if isinstance(obs, tuple):
            obs = obs[0]
        episode_step += 1


def log_metrics_and_update_state(
    num_envs: int,
    step: int,
    mode: str,
    L: VecLogger,
    start_time: np.array,
    episode_reward: np.array,
    episode: np.array,
    episode_step: np.array,
    done: np.array,
    should_log_env_idx: np.array,
):
    for env_idx in range(num_envs):
        if should_log_env_idx[env_idx]:
            L.log(
                f"{mode}/duration",
                time.time() - start_time[env_idx],
                step,
                env_idx=env_idx,
            )
            start_time[env_idx] = time.time()

            L.log(
                f"{mode}/episode_reward",
                episode_reward[env_idx],
                step,
                env_idx=env_idx,
            )
            done[env_idx] = False
            episode_reward[env_idx] = 0
            episode_step[env_idx] = 0
            episode[env_idx] += 1

            L.log(f"{mode}/episode", episode[env_idx], step, env_idx=env_idx)

            L.dump(step, env_idx=env_idx, mode=mode)
    return (start_time, episode_reward, episode_step, episode, done, L)
