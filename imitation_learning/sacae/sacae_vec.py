# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

import sacae.utils as utils
from sacae import sacae
from sacae.decoder import make_vec_decoder
from sacae.encoder import make_vec_encoder
from sacae.logger import Logger
from sacae.vec_logger import VecLogger

LoggerType = Union[Logger, VecLogger]


class Actor(sacae.Actor):
    """MLP actor network."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        hidden_dim: int,
        encoder_type: str,
        encoder_feature_dim: int,
        log_std_min: float,
        log_std_max: float,
        num_layers: int,
        num_filters: int,
        encoder: Optional[torch.nn.Module] = None,
    ):
        if encoder is None:
            encoder_cls = make_vec_encoder
        else:
            encoder_cls = None
        super().__init__(
            obs_shape=obs_shape,
            action_shape=action_shape,
            hidden_dim=hidden_dim,
            encoder_type=encoder_type,
            encoder_feature_dim=encoder_feature_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            num_layers=num_layers,
            num_filters=num_filters,
            encoder=encoder,
            encoder_cls=encoder_cls,
        )

    def log(
        self,
        L: LoggerType,
        step: int,
        log_freq: int = sacae.LOG_FREQ,
        env_idx: Optional[int] = None,
    ) -> None:
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram("train_actor/%s_hist" % k, v, step, env_idx=env_idx)

        L.log_param("train_actor/fc1", self.trunk[0], step, env_idx=env_idx)
        L.log_param("train_actor/fc2", self.trunk[2], step, env_idx=env_idx)
        L.log_param("train_actor/fc3", self.trunk[4], step, env_idx=env_idx)


class Critic(sacae.Critic):
    """Critic network, employes two q-functions."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        hidden_dim: int,
        encoder_type: str,
        encoder_feature_dim: int,
        num_layers: int,
        num_filters: int,
    ):
        super().__init__(
            obs_shape=obs_shape,
            action_shape=action_shape,
            hidden_dim=hidden_dim,
            encoder_type=encoder_type,
            encoder_feature_dim=encoder_feature_dim,
            num_layers=num_layers,
            num_filters=num_filters,
            encoder_cls=make_vec_encoder,
        )

    def log(
        self,
        L: LoggerType,
        step: int,
        log_freq: int = sacae.LOG_FREQ,
        env_idx: Optional[int] = None,
    ) -> None:
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq, env_idx=env_idx)

        for k, v in self.outputs.items():
            L.log_histogram("train_critic/%s_hist" % k, v, step, env_idx=env_idx)

        for i in range(3):
            L.log_param(
                "train_critic/q1_fc%d" % i, self.Q1.trunk[i * 2], step, env_idx=env_idx
            )
            L.log_param(
                "train_critic/q2_fc%d" % i, self.Q2.trunk[i * 2], step, env_idx=env_idx
            )


class SacAeAgent(sacae.SacAeAgent):
    """SAC+AE algorithm."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim: int = 256,
        discount: int = 0.99,
        init_temperature: float = 0.01,
        alpha_lr: float = 1e-3,
        alpha_beta: float = 0.9,
        actor_lr: float = 1e-3,
        actor_beta: float = 0.9,
        actor_log_std_min: int = -10,
        actor_log_std_max: int = 2,
        actor_update_freq: int = 2,
        critic_lr: float = 1e-3,
        critic_beta: float = 0.9,
        critic_tau: float = 0.005,
        critic_target_update_freq: int = 2,
        encoder_type: str = "pixel",
        encoder_feature_dim: int = 50,
        encoder_lr: float = 1e-3,
        encoder_tau: float = 0.005,
        decoder_type: str = "pixel",
        decoder_lr: float = 1e-3,
        decoder_update_freq: int = 1,
        decoder_latent_lambda: float = 0.0,
        decoder_weight_lambda: float = 0.0,
        num_layers: int = 4,
        num_filters: int = 32,
        update_encoder_via_rl: bool = False,
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.decoder_update_freq = decoder_update_freq
        self.decoder_latent_lambda = decoder_latent_lambda
        self.update_encoder_via_rl = update_encoder_via_rl

        self.actor = Actor(
            obs_shape,
            action_shape,
            hidden_dim,
            encoder_type,
            encoder_feature_dim,
            actor_log_std_min,
            actor_log_std_max,
            num_layers,
            num_filters,
        ).to(device)

        self.critic = Critic(
            obs_shape,
            action_shape,
            hidden_dim,
            encoder_type,
            encoder_feature_dim,
            num_layers,
            num_filters,
        ).to(device)

        self.critic_target = Critic(
            obs_shape,
            action_shape,
            hidden_dim,
            encoder_type,
            encoder_feature_dim,
            num_layers,
            num_filters,
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        self.decoder = None
        if decoder_type != "identity":
            # create decoder
            self.decoder = make_vec_decoder(
                decoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters
            ).to(device)
            self.decoder.apply(sacae.weight_init)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            # optimizer for decoder
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(),
                lr=decoder_lr,
                weight_decay=decoder_weight_lambda,
            )

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.train()
        self.critic_target.train()

    def update_critic(
        self,
        obs,
        action,
        reward,
        next_obs,
        not_done,
        L: LoggerType,
        step: int,
        env_idx: int,
    ):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=not self.update_encoder_via_rl
        )
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        L.log("train_critic/loss", critic_loss, step, env_idx=env_idx)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step, env_idx=env_idx)

    def update_actor_and_alpha(self, obs, L: LoggerType, step: int, env_idx: int):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        L.log("train_actor/loss", actor_loss, step, env_idx=env_idx)
        L.log("train_actor/target_entropy", self.target_entropy, step, env_idx=env_idx)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(
            dim=-1
        )
        L.log("train_actor/entropy", entropy.mean(), step, env_idx=env_idx)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step, env_idx=env_idx)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        L.log("train_alpha/loss", alpha_loss, step, env_idx=env_idx)
        L.log("train_alpha/value", self.alpha, step, env_idx=env_idx)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_decoder(self, obs, target_obs, L: LoggerType, step: int, env_idx: int):
        h = self.critic.encoder(obs)

        if target_obs.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = utils.preprocess_obs(target_obs)
        rec_obs = self.decoder(h)
        rec_loss = F.mse_loss(target_obs, rec_obs)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        loss = rec_loss + self.decoder_latent_lambda * latent_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        L.log("train/ae_loss", loss, step, env_idx=env_idx)

        self.decoder.log(L, step, log_freq=sacae.LOG_FREQ, env_idx=env_idx)

    def update(self, replay_buffer, L: Logger, step: int, env_idx: int):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(env_id=env_idx)

        L.log("train/batch_reward", reward.mean(), step, env_idx=env_idx)

        self.update_critic(
            obs, action, reward, next_obs, not_done, L, step, env_idx=env_idx
        )

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step, env_idx=env_idx)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder, self.encoder_tau,
            )

        if self.decoder is not None and step % self.decoder_update_freq == 0:
            self.update_decoder(obs, obs, L, step, env_idx=env_idx)
