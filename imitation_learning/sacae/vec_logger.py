# Copyright (c) Facebook, Inc. and its affiliates.
import os
import shutil
from typing import Optional

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from sacae.logger import MetersGroup

FORMAT_CONFIG = {
    "rl": {
        "train": [
            ("env_idx", "ENV_ID", "int"),
            ("episode", "E", "int"),
            ("step", "S", "int"),
            ("duration", "D", "time"),
            ("episode_reward", "R", "float"),
            ("batch_reward", "BR", "float"),
            ("actor_loss", "ALOSS", "float"),
            ("critic_loss", "CLOSS", "float"),
            ("ae_loss", "RLOSS", "float"),
        ],
        "eval": [
            ("env_idx", "ENV_ID", "int"),
            ("step", "S", "int"),
            ("episode_reward", "ER", "float"),
        ],
    }
}


class MetersGroupWithIdx(MetersGroup):
    def __init__(self, file_name, formating, env_idx):
        super().__init__(file_name=file_name, formating=formating)
        self.env_idx = env_idx

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data["step"] = step
        data["env_idx"] = self.env_idx
        self._dump_to_file(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class VecLogger(object):
    """Vectorized Logger"""

    def __init__(
        self, log_dir: str, use_tb: bool = True, config: str = "rl", num_envs: int = 1
    ):
        self._log_dir = log_dir
        if use_tb:
            tb_dir = os.path.join(log_dir, "tb")
            if os.path.exists(tb_dir):
                shutil.rmtree(tb_dir)
            self._sw = [SummaryWriter(tb_dir) for _ in range(num_envs)]
        else:
            self._sw = None
        self._train_mg = [
            MetersGroupWithIdx(
                os.path.join(log_dir, "train.log"),
                formating=FORMAT_CONFIG[config]["train"],
                env_idx=env_idx,
            )
            for env_idx in range(num_envs)
        ]
        self._eval_mg = [
            MetersGroupWithIdx(
                os.path.join(log_dir, "eval.log"),
                formating=FORMAT_CONFIG[config]["eval"],
                env_idx=env_idx,
            )
            for env_idx in range(num_envs)
        ]

    def _try_sw_log(self, key, value, step, env_idx):
        if self._sw is not None:
            self._sw[env_idx].add_scalar(key, value, step)

    def _try_sw_log_image(self, key, image, step, env_idx):
        if self._sw is not None:
            assert image.dim() == 3
            grid = torchvision.utils.make_grid(image.unsqueeze(1))
            self._sw[env_idx].add_image(key, grid, step)

    def _try_sw_log_video(self, key, frames, step, env_idx):
        if self._sw is not None:
            frames = torch.from_numpy(np.array(frames))
            frames = frames.unsqueeze(0)
            self._sw[env_idx].add_video(key, frames, step, fps=30)

    def _try_sw_log_histogram(self, key, histogram, step, env_idx):
        if self._sw is not None:
            self._sw[env_idx].add_histogram(key, histogram, step)

    def log(self, key, value, step, n=1, env_idx=None):
        assert key.startswith("train") or key.startswith("eval")
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value / n, step, env_idx=env_idx)
        mg = (
            self._train_mg[env_idx]
            if key.startswith("train")
            else self._eval_mg[env_idx]
        )
        mg.log(key, value, n)

    def log_param(self, key, param, step, env_idx):
        self.log_histogram(key + "_w", param.weight.data, step, env_idx=env_idx)
        if hasattr(param.weight, "grad") and param.weight.grad is not None:
            self.log_histogram(
                key + "_w_g", param.weight.grad.data, step, env_idx=env_idx
            )
        if hasattr(param, "bias"):
            self.log_histogram(key + "_b", param.bias.data, step, env_idx=env_idx)
            if hasattr(param.bias, "grad") and param.bias.grad is not None:
                self.log_histogram(
                    key + "_b_g", param.bias.grad.data, step, env_idx=env_idx
                )

    def log_image(self, key, image, step, env_idx):
        assert key.startswith("train") or key.startswith("eval")
        self._try_sw_log_image(key, image, step, env_idx=env_idx)

    def log_video(self, key, frames, step, env_idx):
        assert key.startswith("train") or key.startswith("eval")
        self._try_sw_log_video(key, frames, step, env_idx=env_idx)

    def log_histogram(self, key, histogram, step, env_idx):
        assert key.startswith("train") or key.startswith("eval")
        self._try_sw_log_histogram(key, histogram, step, env_idx=env_idx)

    def dump(self, step, env_idx, mode: Optional[str] = None) -> None:
        if mode is None:
            self._train_mg[env_idx].dump(step, "train")
            self._eval_mg[env_idx].dump(step, "eval")
        elif mode == "train":
            self._train_mg[env_idx].dump(step, "train")
        elif mode == "eval":
            self._eval_mg[env_idx].dump(step, "eval")
