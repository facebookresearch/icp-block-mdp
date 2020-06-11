# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


OUT_DIM = {2: 39, 4: 35, 6: 31}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList([nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)])
        # self.convs = nn.ModuleList([nn.Conv2d(3, num_filters, 3, stride=2)])
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.0
        self.outputs["obs"] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs["conv1"] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs["conv%s" % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):

        if detach:
            with torch.no_grad():
                return self._forward(obs)

        return self._forward(obs)

    def _forward(self, obs):

        h = self.forward_conv(obs)

        h_fc = self.fc(h)
        self.outputs["fc"] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs["ln"] = h_norm

        out = torch.tanh(h_norm)
        self.outputs["tanh"] = out

        return out

    # def forward(self, obs, detach=False):
    #     h = self.forward_conv(obs)

    #     if detach:
    #         h = h.detach()

    #     h_fc = self.fc(h)
    #     self.outputs["fc"] = h_fc

    #     h_norm = self.ln(h_fc)
    #     self.outputs["ln"] = h_norm

    #     out = torch.tanh(h_norm)
    #     self.outputs["tanh"] = out

    #     return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram("train_encoder/%s_hist" % k, v, step)
            if len(v.shape) > 2:
                L.log_image("train_encoder/%s_img" % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param("train_encoder/conv%s" % (i + 1), self.convs[i], step)
        L.log_param("train_encoder/fc", self.fc, step)
        L.log_param("train_encoder/ln", self.ln, step)


class VecPixelEncoder(PixelEncoder):
    """Convolutional encoder of pixels observations."""

    def __init__(
        self, obs_shape, feature_dim: int, num_layers: int = 2, num_filters: int = 32
    ):
        super().__init__(
            obs_shape=obs_shape,
            feature_dim=feature_dim,
            num_layers=num_layers,
            num_filters=num_filters,
        )

    def log(self, L, step, log_freq, env_idx):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram("train_encoder/%s_hist" % k, v, step, env_idx=env_idx)
            if len(v.shape) > 2:
                L.log_image("train_encoder/%s_img" % k, v[0], step, env_idx=env_idx)

        for i in range(self.num_layers):
            L.log_param(
                "train_encoder/conv%s" % (i + 1), self.convs[i], step, env_idx=env_idx
            )
        L.log_param("train_encoder/fc", self.fc, step, env_idx=env_idx)
        L.log_param("train_encoder/ln", self.ln, step, env_idx=env_idx)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


class VecIdentityEncoder(IdentityEncoder):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters):
        super().__init__(
            obs_shape=obs_shape,
            feature_dim=feature_dim,
            num_layers=num_layers,
            num_filters=num_filters,
        )

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq, env_idx):
        pass


_AVAILABLE_ENCODERS = {"pixel": PixelEncoder, "identity": IdentityEncoder}

_AVAILABLE_VEC_ENCODERS = {"pixel": VecPixelEncoder, "identity": VecIdentityEncoder}


def make_encoder(encoder_type, obs_shape, feature_dim, num_layers, num_filters):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters
    )


def make_vec_encoder(encoder_type, obs_shape, feature_dim, num_layers, num_filters):
    assert encoder_type in _AVAILABLE_VEC_ENCODERS
    return _AVAILABLE_VEC_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters
    )
