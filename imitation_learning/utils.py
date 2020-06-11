# The different vectorized envs have been taken from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/envs.py

import glob
import os
import random
from collections import defaultdict, deque
from random import sample
from typing import List, Optional, Union

import gym
import numpy as np
import skvideo.io
import torch
import torch.nn as nn

import dmc2gym
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from sacae.sacae import SacAeAgent


class MultiEnvReplayBuffer(object):
    """Buffer to store environment transitions for multiple environments"""

    def __init__(
        self, obs_shape, action_shape, capacity, batch_size, device, num_envs: int
    ):
        self.env_id_to_replay_buffer_map = [
            ReplayBuffer(
                obs_shape=obs_shape,
                action_shape=action_shape,
                capacity=int(capacity / num_envs),
                batch_size=batch_size,
                device=device,
            )
            for _ in range(num_envs)
        ]
        self.num_envs = num_envs

    def add(self, env_id, obs, action, reward, next_obs, done):
        self.env_id_to_replay_buffer_map[env_id].add(
            obs, action, reward, next_obs, done
        )

    def add_loop(self, obs, action, reward, next_obs, done):
        for env_id in range(self.num_envs):
            self.env_id_to_replay_buffer_map[env_id].add(
                obs=obs[env_id],
                action=action[env_id],
                reward=reward[env_id],
                next_obs=next_obs[env_id],
                done=done[env_id],
            )

    def sample(self, env_id: Optional[int] = None):
        if env_id is None:
            env_id = random.randint(0, self.num_envs - 1)
        return self.env_id_to_replay_buffer_map[env_id].sample()

    def save(self, save_dir):
        for idx, replay_buffer in enumerate(self.env_id_to_replay_buffer_map):
            replay_buffer.save(f"{save_dir}/{idx}")

    def load(self, save_dir):
        for idx, replay_buffer in enumerate(self.env_id_to_replay_buffer_map):
            replay_buffer.load(f"{save_dir}/{idx}")


class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, action_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, "%d_%d.pt" % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save : self.idx],
            self.next_obses[self.last_save : self.idx],
            self.actions[self.last_save : self.idx],
            self.rewards[self.last_save : self.idx],
            self.not_dones[self.last_save : self.idx],
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split("_")[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split(".")[0].split("_")]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype,
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs, info = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device, num_envs):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        self.num_envs = num_envs
        # TODO: Fix data types

    def reset(self):
        obs, info = self.venv.reset()
        # obs = torch.from_numpy(obs).float().to(self.device)
        obs = torch.from_numpy(obs).to(self.device)
        state = torch.cat([torch.from_numpy(x["state"]).unsqueeze(0) for x in info]).to(
            self.device
        )
        return obs, state

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        # obs = torch.from_numpy(obs).float().to(self.device)
        obs = torch.from_numpy(obs).to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        state = torch.cat([torch.from_numpy(x["state"]).unsqueeze(0) for x in info]).to(
            self.device
        )
        return obs, reward, done, state


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device("cpu")
        self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype
        )
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, : -self.shape_dim0] = self.stacked_obs[:, self.shape_dim0 :]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0 :] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0 :] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()


def make_env(
    args, seed: int, resource_files: Optional[Union[List[str], str]], camera_id: int
):

    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        resource_files=resource_files,
        img_source=None,
        total_frames=None,
        seed=args.seed + seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == "pixel"),
        height=args.image_size,
        width=args.image_size,
        frame_skip=args.action_repeat,
        camera_id=camera_id,
    )
    env.seed(args.seed)
    if args.encoder_type == "pixel":
        env = FrameStack(env, k=args.frame_stack)
    return env


def make_dummy_env(args):

    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        resource_files=None,
        img_source=None,
        total_frames=None,
        seed=args.seed + 1234,
        visualize_reward=False,
        from_pixels=(args.encoder_type == "pixel"),
        height=args.image_size,
        width=args.image_size,
        frame_skip=args.action_repeat,
    )
    env.seed(args.seed)
    if args.encoder_type == "pixel":
        env = FrameStack(env, k=args.frame_stack)
    return env


def fn_to_make_env(
    args, seed: int, resource_files: Union[List[str], str], camera_id: int
):
    def fn():
        return make_env(
            args=args, seed=seed, resource_files=resource_files, camera_id=camera_id
        )

    return fn


def make_vec_envs(fns_to_make_envs, device):

    if len(fns_to_make_envs) > 1:
        envs = ShmemVecEnv(fns_to_make_envs, context="spawn")
    else:
        envs = DummyVecEnv(fns_to_make_envs)

    envs = VecPyTorch(envs, device, len(fns_to_make_envs))

    return envs


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError:
        pass
    return dir_path


def collect_data_using_fn(
    env: FrameStack,
    env_id: int,
    num_samples: int,
    replay_buffer: MultiEnvReplayBuffer,
    fn_to_get_action,
    save_video=False,
):
    obs = env.reset()
    if save_video:
        frames = [obs]
    for i in range(num_samples):
        action = fn_to_get_action(obs)
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(env_id, obs, action, reward, next_obs, done)
        if save_video:
            frames.append(next_obs)
        obs = next_obs
        if done:
            obs = env.reset()
            if save_video:
                skvideo.io.vwrite(
                    f"video/{str(env)}_{env.unwrapped._img_source}.mp4", frames
                )
                save_video = False
    return replay_buffer


def collect_random_data(
    env: FrameStack,
    env_id: int,
    num_samples: int,
    replay_buffer: MultiEnvReplayBuffer,
    save_video: bool = False,
):
    fn_to_get_action = lambda obs: env.action_space.sample()
    return collect_data_using_fn(
        env, env_id, num_samples, replay_buffer, fn_to_get_action, save_video
    )


def collect_data_using_policy(
    env: FrameStack,
    env_id: int,
    num_samples: int,
    replay_buffer: MultiEnvReplayBuffer,
    policy: SacAeAgent,
    save_video: bool = False,
):
    fn_to_get_action = lambda obs: policy.sample_action(obs)
    return collect_data_using_fn(
        env, env_id, num_samples, replay_buffer, fn_to_get_action, save_video
    )


def collect_both_state_and_obs_using_policy_vec(
    vec_env: VecPyTorch,
    num_samples: int,
    replay_buffer: MultiEnvReplayBuffer,
    policy: SacAeAgent,
    save_video: bool = False,
):
    def fn_to_get_action(obs):
        return policy.sample_action(obs.float())

    return collect_data_using_fn_vec(
        vec_env, num_samples, replay_buffer, fn_to_get_action, save_video
    )


def collect_data_using_fn_vec(
    vec_env: VecPyTorch,
    num_samples: int,
    replay_buffer: MultiEnvReplayBuffer,
    fn_to_get_action,
    save_video: bool = False,
):

    make_vector_using_val = lambda val: np.full(vec_env.num_envs, val)
    obs, state = vec_env.reset()
    if save_video:
        frames = [obs]
    save_video = make_vector_using_val(save_video)
    for i in range(num_samples):
        action = fn_to_get_action(state)
        next_obs, reward, done, next_state = vec_env.step(action)
        replay_buffer.add_loop(obs, action, reward, next_obs, done)
        obs = next_obs
        state = next_state
        save_frame = False
        if save_frame:
            frames.append(next_obs)
            for env_id in range(vec_env.num_envs):
                if done[env_id] and save_video[env_id]:
                    env_frames = torch.cat(
                        [x[env_id].unsqueeze(0) for x in frames], dim=0
                    )
                    for env_id in range(vec_env.num_envs):
                        skvideo.io.vwrite(
                            f"video/env_id_{env_id}.mp4",
                            env_frames
                            # f"video/{str(env)}_{env.unwrapped._img_source}.mp4", frames
                        )
                    save_video[env_id] = False
            save_frame = np.any(save_video)
    return replay_buffer


def collect_random_data_vec(
    vec_env: VecPyTorch,
    num_samples: int,
    replay_buffer: MultiEnvReplayBuffer,
    save_video: bool = False,
):
    make_tensor_using_fn = lambda fn: torch.tensor(
        [fn() for _ in range(vec_env.num_envs)]
    )
    fn_to_get_action = lambda obs: make_tensor_using_fn(vec_env.action_space.sample)
    return collect_data_using_fn_vec(
        vec_env, num_samples, replay_buffer, fn_to_get_action, save_video
    )


def collect_data_using_policy_vec(
    vec_env: VecPyTorch,
    num_samples: int,
    replay_buffer: MultiEnvReplayBuffer,
    policy: SacAeAgent,
    save_video: bool = False,
):
    fn_to_get_action = lambda obs: policy.sample_action(obs.float())
    return collect_data_using_fn_vec(
        vec_env, num_samples, replay_buffer, fn_to_get_action, save_video
    )


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2 ** bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2 ** (8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


def populate_buffer_with_random_data(
    envs: List[FrameStack],
    buffer: MultiEnvReplayBuffer,
    save_video: bool,
    num_samples: int,
):

    for env_id, env in enumerate(envs):
        buffer = collect_random_data(
            env=env,
            env_id=env_id,
            num_samples=num_samples,
            replay_buffer=buffer,
            save_video=save_video,
        )

    return buffer
