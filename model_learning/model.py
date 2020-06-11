# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


OUT_DIM = {2: 39, 4: 35, 6: 31}


class Encoder(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList([nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)])
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

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
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs["fc"] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs["ln"] = h_norm

        out = torch.tanh(h_norm)
        self.outputs["tanh"] = out

        return out

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


class DynamicsModel(nn.Module):
    def __init__(self, representation_size, action_shape):
        super().__init__()

        self.action_linear = nn.Linear(action_shape, representation_size)
        self.trunk = nn.Sequential(
            nn.Linear(representation_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, representation_size),
        )

    def forward(self, state, action):
        action_emb = self.action_linear(action)
        return self.trunk(torch.cat([state, action_emb], dim=-1))


class Decoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = OUT_DIM[num_layers]

        self.fc = nn.Linear(feature_dim, num_filters * self.out_dim * self.out_dim)

        self.deconvs = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1)
            )
        self.deconvs.append(
            nn.ConvTranspose2d(num_filters, obs_shape[0], 3, stride=2, output_padding=1)
        )

        self.outputs = dict()

    def forward(self, h):
        h = torch.relu(self.fc(h))
        self.outputs["fc"] = h

        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)
        self.outputs["deconv1"] = deconv

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
            self.outputs["deconv%s" % (i + 1)] = deconv

        obs = self.deconvs[-1](deconv)
        self.outputs["obs"] = obs

        return obs

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram("train_decoder/%s_hist" % k, v, step)
            if len(v.shape) > 2:
                L.log_image("train_decoder/%s_i" % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param("train_decoder/deconv%s" % (i + 1), self.deconvs[i], step)
        L.log_param("train_decoder/fc", self.fc, step)
