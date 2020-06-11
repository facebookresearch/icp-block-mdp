# Copyright (c) Facebook, Inc. and its affiliates.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd, optim

import hydra
import utils
from agent import Agent
from decoder import make_decoder
from encoder import make_encoder


def make_dynamics_model(feature_dim, hidden_dim, action_shape):
    model = nn.Sequential(
        nn.Linear(feature_dim + action_shape, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, feature_dim),
    )
    return model


def irm_penalty(logits, labels):
    scale = torch.tensor(1.0).cuda().requires_grad_()
    loss = F.mse_loss(logits * scale, labels)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


class SACAgent(Agent):
    """SAC algorithm."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        action_range,
        device,
        encoder_type,
        encoder_feature_dim,
        critic_cfg,
        actor_cfg,
        discount,
        init_temperature,
        alpha_lr,
        alpha_betas,
        actor_lr,
        actor_betas,
        actor_update_frequency,
        critic_lr,
        critic_betas,
        critic_tau,
        critic_target_update_frequency,
        batch_size,
    ):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.encoder = make_encoder(
            encoder_type, obs_dim, encoder_feature_dim, 2, 32
        ).to(self.device)

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic.encoder = self.encoder

        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.encoder = self.encoder
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)
        self.actor.encoder = self.encoder

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=actor_betas
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=critic_betas
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=alpha_betas
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, logger, step):
        # with torch.no_grad():
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        logger.log("train_critic/loss", critic_loss, step)

        # add L1 penalty
        L1_reg = torch.tensor(0.0, requires_grad=True).to(self.device)
        for name, param in self.critic.encoder.named_parameters():
            if "weight" in name:
                L1_reg = L1_reg + torch.norm(param, 1)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        (critic_loss + 1e-5 * L1_reg).backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        dist = self.actor(obs, detach=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log("train_actor/loss", actor_loss, step)
        logger.log("train_actor/target_entropy", self.target_entropy, step)
        logger.log("train_actor/entropy", -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        logger.log("train_alpha/loss", alpha_loss, step)
        logger.log("train_alpha/value", self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
            self.batch_size
        )

        logger.log("train/batch_reward", reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done_no_max, logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)


class CausalAgent(Agent):
    """SAC algorithm."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        action_range,
        device,
        encoder_type,
        num_envs,
        c_ent,
        kld,
        critic_cfg,
        actor_cfg,
        discount,
        init_temperature,
        alpha_lr,
        encoder_lr,
        c_ent_iters,
        alpha_betas,
        actor_lr,
        actor_betas,
        actor_update_frequency,
        decoder_lr,
        decoder_weight_lambda,
        critic_lr,
        critic_betas,
        critic_tau,
        encoder_feature_dim,
        decoder_latent_lambda,
        critic_target_update_frequency,
        batch_size,
    ):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.num_envs = num_envs
        self.encoder_tau = 0.005
        self.decoder_latent_lambda = decoder_latent_lambda
        self.encoder_type = encoder_type
        self.c_ent = c_ent
        self.c_ent_iters = c_ent_iters
        self.kld = kld
        self.encoder = make_encoder(
            encoder_type, obs_dim, encoder_feature_dim, 2, 32
        ).to(self.device)

        self.task_specific_encoders = [
            make_encoder(encoder_type, obs_dim, encoder_feature_dim, 2, 32).to(device)
            for i in range(self.num_envs)
        ]

        self.model = make_dynamics_model(encoder_feature_dim, 200, action_dim).to(
            device
        )
        self.task_specific_models = [
            make_dynamics_model(encoder_feature_dim, 200, action_dim).to(device)
            for i in range(self.num_envs)
        ]
        self.reward_model = nn.Sequential(
            nn.Linear(encoder_feature_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
        ).to(device)
        self.decoder = nn.Sequential(
            nn.Linear(encoder_feature_dim * 2, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, obs_dim),
        ).to(device)
        self.classifier = nn.Sequential(
            nn.Linear(encoder_feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_envs),
        ).to(device)

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic.encoder = self.encoder
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.encoder = make_encoder(
            encoder_type, obs_dim, encoder_feature_dim, 2, 32
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)
        self.actor.encoder = self.encoder

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=actor_betas
        )

        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.encoder.parameters()),
            lr=critic_lr,
            betas=critic_betas,
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=alpha_betas
        )

        self.classifier_optimizer = torch.optim.Adam(
            self.classifier.parameters(), lr=actor_lr, betas=actor_betas
        )

        # optimizer for critic encoder for reconstruction loss
        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=encoder_lr
        )
        # optimizer for decoder
        task_specific_parameters = [
            params
            for t in (self.task_specific_encoders + self.task_specific_models)
            for params in list(t.parameters())
        ]
        self.decoder_optimizer = torch.optim.Adam(
            list(self.decoder.parameters())
            + list(self.model.parameters())
            + list(self.reward_model.parameters())
            + task_specific_parameters
            + list(self.encoder.parameters()),
            lr=decoder_lr,
            weight_decay=decoder_weight_lambda,
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs, detach=True)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, logger, step):
        dist = self.actor(next_obs, detach=False)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, detach=False)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        logger.log("train_critic/loss", critic_loss, step)

        return critic_loss

    def update_actor_and_alpha(self, obs, logger, step):
        dist = self.actor(obs, detach=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log("train_actor/loss", actor_loss, step)
        logger.log("train_actor/target_entropy", self.target_entropy, step)
        logger.log("train_actor/entropy", -log_prob.mean(), step)

        self.actor.log(logger, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        logger.log("train_alpha/loss", alpha_loss, step)
        logger.log("train_alpha/value", self.alpha, step)
        return actor_loss, alpha_loss

    def update_decoder(self, obs, action, reward, target_obs, logger, step, i):
        h = self.critic.encoder(obs)

        task_specific_h = self.task_specific_encoders[i](obs)

        next_h = self.model(torch.cat([h, action], dim=-1))
        next_task_specific_h = self.task_specific_models[i](
            torch.cat([task_specific_h, action], dim=-1)
        )
        r_hat = self.reward_model(next_h)

        rec_obs = self.decoder(torch.cat([next_h, next_task_specific_h], dim=-1))
        rec_loss = F.mse_loss(target_obs, rec_obs)
        rew_loss = F.mse_loss(r_hat, reward)
        logger.log("train_encoder/rc_loss", rec_loss.item(), step)

        # autoencoder loss
        # rec_obs = self.decoder(torch.cat([h, task_specific_h], dim=-1))
        # rec_loss = F.mse_loss(obs, rec_obs)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        # add L1 penalty
        L1_reg = torch.tensor(0.0, requires_grad=True).to(self.device)
        for name, param in self.critic.encoder.named_parameters():
            if "weight" in name:
                L1_reg = L1_reg + torch.norm(param, 1)

        # get classifier entropy
        h = self.critic.encoder(obs)
        probs = F.softmax(self.classifier(h))
        entropy = -1.0 * (probs * torch.log2(probs + 1e-12)).sum(dim=1).mean()
        logger.log("train_classifier/entropy", entropy, step)

        # compute information bottleneck
        KLD = 0.0
        if self.encoder_type == "variational":
            mu, logvar = self.critic.encoder.encode(obs)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            logger.log("train_encoder/KLD", self.kld * KLD.item(), step)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        if step > self.c_ent_iters:
            c_ent = self.c_ent
        else:
            c_ent = 0
        (
            rec_loss
            + self.decoder_latent_lambda * L1_reg
            - c_ent * entropy
            + self.kld * KLD
        ).backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        logger.log("train/model_loss", rec_loss, step)
        logger.log("train/reward_loss", rew_loss, step)

    def update_classifier(self, obs, env_id):
        h = self.critic.encoder(obs)
        pred_labels = self.classifier(h)
        classifier_loss = F.cross_entropy(pred_labels, env_id)

        self.classifier_optimizer.zero_grad()
        classifier_loss.backward()
        self.classifier_optimizer.step()

    def update(self, replay_buffer, logger, step):
        total_actor_loss, total_alpha_loss, total_critic_loss, obses, env_ids = (
            [],
            [],
            [],
            [],
            [],
        )
        for env_id in range(self.num_envs):
            (
                obs,
                action,
                reward,
                next_obs,
                not_done,
                not_done_no_max,
            ) = replay_buffer.sample(self.batch_size, env_id)
            obses.append(obs)
            env_ids.append(torch.ones_like(reward).long() * env_id)

            logger.log("train/batch_reward", reward.mean(), step)

            critic_loss = self.update_critic(
                obs, action, reward, next_obs, not_done_no_max, logger, step
            )
            total_critic_loss.append(critic_loss)

            if step % self.actor_update_frequency == 0:
                actor_loss, alpha_loss = self.update_actor_and_alpha(obs, logger, step)
                total_actor_loss.append(actor_loss)
                total_alpha_loss.append(alpha_loss)

            self.update_decoder(obs, action, reward, next_obs, logger, step, env_id)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        torch.stack(total_critic_loss).mean().backward()
        self.critic_optimizer.step()
        self.critic.log(logger, step)

        # Optimize classifier
        self.update_classifier(
            torch.cat(obses, dim=0), torch.cat(env_ids, dim=0).squeeze()
        )

        if step % self.actor_update_frequency == 0:
            # optimize the actor
            self.actor_optimizer.zero_grad()
            torch.stack(total_actor_loss).mean().backward()
            self.actor_optimizer.step()

            self.actor.log(logger, step)

            self.log_alpha_optimizer.zero_grad()
            torch.stack(total_alpha_loss).mean().backward()
            self.log_alpha_optimizer.step()

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)


class IRMAgent(Agent):
    """IRM algorithm."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        action_range,
        device,
        encoder_type,
        critic_cfg,
        actor_cfg,
        discount,
        init_temperature,
        alpha_lr,
        l2_regularizer_weight,
        alpha_betas,
        actor_lr,
        actor_betas,
        actor_update_frequency,
        critic_lr,
        critic_betas,
        critic_tau,
        num_envs,
        encoder_feature_dim,
        critic_target_update_frequency,
        batch_size,
        penalty_anneal_iters,
        penalty_weight,
    ):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.num_envs = num_envs
        self.l2_regularizer_weight = l2_regularizer_weight
        self.penalty_anneal_iters = penalty_anneal_iters
        self.penalty_weight = penalty_weight

        self.encoder = make_encoder(
            encoder_type, obs_dim, encoder_feature_dim, 2, 32
        ).to(self.device)
        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic.encoder = self.encoder

        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.encoder = make_encoder(
            encoder_type, obs_dim, encoder_feature_dim, 2, 32
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)
        self.actor.encoder = self.encoder

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=actor_betas
        )

        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.critic.encoder.parameters()),
            lr=critic_lr,
            betas=critic_betas,
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=alpha_betas
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, logger, step):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2, h = self.critic(obs, action, return_latent=True)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        self.irm_penalty = irm_penalty(current_Q1, target_Q) + irm_penalty(
            current_Q2, target_Q
        )
        logger.log("train_critic/loss", critic_loss, step)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        # add L1 penalty
        L1_reg = torch.tensor(0.0, requires_grad=True).to(self.device)
        for name, param in self.critic.encoder.named_parameters():
            if "weight" in name:
                L1_reg = L1_reg + torch.norm(param, 1)

        return critic_loss + self.l2_regularizer_weight * L1_reg, target_V.mean()

    def update_actor_and_alpha(self, obs, logger, step):
        dist = self.actor(obs, detach=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log("train_actor/loss", actor_loss, step)
        logger.log("train_actor/target_entropy", self.target_entropy, step)
        logger.log("train_actor/entropy", -log_prob.mean(), step)

        self.actor.log(logger, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        logger.log("train_alpha/loss", alpha_loss, step)
        logger.log("train_alpha/value", self.alpha, step)
        return actor_loss, alpha_loss

    def update(self, replay_buffer, logger, step):
        total_actor_loss, total_alpha_loss, total_critic_loss = [], [], []
        target_vs = []
        irm_penalties = []
        for env_id in range(self.num_envs):
            (
                obs,
                action,
                reward,
                next_obs,
                not_done,
                not_done_no_max,
            ) = replay_buffer.sample(self.batch_size, env_id)

            logger.log("train/batch_reward", reward.mean(), step)

            critic_loss, target_v = self.update_critic(
                obs, action, reward, next_obs, not_done_no_max, logger, step
            )
            total_critic_loss.append(critic_loss)
            target_vs.append(target_v)

            if step % self.actor_update_frequency == 0:
                actor_loss, alpha_loss = self.update_actor_and_alpha(obs, logger, step)
                total_actor_loss.append(actor_loss)
                total_alpha_loss.append(alpha_loss)

            irm_penalties.append(self.irm_penalty)

        # Optimize the critic
        train_penalty = torch.stack(irm_penalties).mean()
        penalty_weight = (
            self.penalty_weight if step >= self.penalty_anneal_iters else 1.0
        )
        logger.log("train_encoder/penalty", train_penalty, step)
        total_critic_loss = torch.stack(total_critic_loss).mean()
        total_critic_loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            total_critic_loss /= penalty_weight

        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        self.critic_optimizer.step()
        self.critic.log(logger, step)

        if step % self.actor_update_frequency == 0:
            # optimize the actor
            self.actor_optimizer.zero_grad()
            torch.stack(total_actor_loss).mean().backward()
            self.actor_optimizer.step()

            self.actor.log(logger, step)

            self.log_alpha_optimizer.zero_grad()
            torch.stack(total_alpha_loss).mean().backward()
            self.log_alpha_optimizer.step()

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)
