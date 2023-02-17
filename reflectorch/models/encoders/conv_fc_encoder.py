# -*- coding: utf-8 -*-

import math
from torch import nn, cat, split

from nflows.nn.nets import ResidualNet

from reflectorch.models.encoders.conv_encoder import ConvEncoder
from reflectorch.models.utils import activation_by_name


class ConvFCEncoder(nn.Module):
    def __init__(self,
                 hidden_dims: tuple = (16, 32, 64),
                 latent_dim: int = 64,
                 avpool: int = 4,
                 use_batch_norm: bool = False,
                 in_features: int = 64,
                 hidden_features: int = 64,
                 num_blocks: int = 3,
                 conv_activation: str = 'lrelu',
                 fc_activation: str = 'relu',
                 use_selu_init: bool = False,
                 pretrained_conv: str = None,
                 ):
        super().__init__()
        self.conv = ConvEncoder(
            hidden_dims=hidden_dims, latent_dim=latent_dim, avpool=avpool, use_batch_norm=use_batch_norm,
            activation=activation_by_name(conv_activation)
        )
        self.fc = ResidualNet(
            in_features, latent_dim, hidden_features, num_blocks=num_blocks, use_batch_norm=use_batch_norm,
            activation=activation_by_name(fc_activation)
        )
        self.cat_fc = nn.Linear(latent_dim * 2, latent_dim)

        if use_selu_init and conv_activation == 'selu':
            self.conv.apply(selu_init)

        if use_selu_init and fc_activation == 'selu':
            self.fc.apply(selu_init)

        if pretrained_conv:
            self.conv.load_weights(pretrained_conv)

    def forward(self, x):
        x = cat([self.conv(x), self.fc(x)], dim=-1)
        x = self.cat_fc(x)
        return x


class SeqConvFCEncoder(nn.Module):
    def __init__(self,
                 hidden_dims: tuple = (32, 64, 128, 256, 512),
                 latent_dim: int = 128,
                 conv_latent_dim: int = 64,
                 avpool: int = 4,
                 use_batch_norm: bool = False,
                 in_features: int = 64,
                 hidden_features: int = 256,
                 num_blocks: int = 3,
                 conv_activation: str = 'lrelu',
                 fc_activation: str = 'relu',
                 use_selu_init: bool = False,
                 sigmas_input: bool = False,
                 pretrained_conv: str = None,
                 ):
        super().__init__()

        self.sigmas_input = sigmas_input

        self.conv = ConvEncoder(
            hidden_dims=hidden_dims,
            latent_dim=conv_latent_dim,
            avpool=avpool,
            use_batch_norm=use_batch_norm,
            activation=activation_by_name(conv_activation),
        )

        fc_in_dim = in_features + conv_latent_dim

        if self.sigmas_input:
            fc_in_dim += in_features
            self._split = [in_features, in_features]
        else:
            self._split = None

        self.fc = ResidualNet(
            fc_in_dim,
            latent_dim,
            hidden_features,
            num_blocks=num_blocks,
            use_batch_norm=use_batch_norm,
            activation=activation_by_name(fc_activation),
        )

        if use_selu_init and conv_activation == 'selu':
            self.conv.apply(selu_init)

        if use_selu_init and fc_activation == 'selu':
            self.fc.apply(selu_init)

        if pretrained_conv:
            self.conv.load_weights(pretrained_conv)

    def forward(self, x):
        if self.sigmas_input:
            return self._forward_with_sigmas(x)
        else:
            return self._forward(x)

    def _forward_with_sigmas(self, x):
        if isinstance(x, dict):
            curves, sigmas = x['scaled_curves'], x['scaled_sigmas']
        else:
            curves, sigmas = split(x, self._split, dim=-1)

        x = cat([curves, sigmas, self.conv(curves)], dim=-1)
        x = self.fc(x)
        return x

    def _forward(self, x):
        x = cat([self.conv(x), x], dim=-1)
        x = self.fc(x)
        return x


class SubPriorConvFCEncoder(nn.Module):
    def __init__(self,
                 hidden_dims: tuple = (16, 32, 64),
                 latent_dim: int = 64,
                 conv_latent_dim: int = 64,
                 avpool: int = 8,
                 use_batch_norm: bool = False,
                 in_features: int = 64,
                 prior_in_features: int = 16,
                 hidden_features: int = 64,
                 num_blocks: int = 3,
                 conv_activation: str = 'lrelu',
                 fc_activation: str = 'gelu',
                 use_selu_init: bool = False,
                 pass_bounds: bool = False,
                 sigmas_input: bool = False,
                 pretrained_conv: str = None,
                 ):
        super().__init__()
        self.pass_bounds = pass_bounds
        self.sigmas_input = sigmas_input

        self.conv = ConvEncoder(
            hidden_dims=hidden_dims, latent_dim=conv_latent_dim, avpool=avpool,
            use_batch_norm=use_batch_norm, activation=activation_by_name(conv_activation)
        )

        fc_in_dim = in_features + conv_latent_dim + prior_in_features

        if self.sigmas_input:
            fc_in_dim += in_features

        self.fc = ResidualNet(
            fc_in_dim,
            latent_dim,
            hidden_features,
            activation=activation_by_name(fc_activation),
            num_blocks=num_blocks,
            use_batch_norm=use_batch_norm
        )
        if self.sigmas_input:
            self._split = [in_features, in_features, prior_in_features]
        else:
            self._split = [in_features, prior_in_features]

        if use_selu_init and conv_activation == 'selu':
            self.conv.apply(selu_init)

        if use_selu_init and fc_activation == 'selu':
            self.fc.apply(selu_init)

        if pretrained_conv:
            self.conv.load_weights(pretrained_conv)

    def forward(self, x):
        if self.sigmas_input:
            return self._forward_with_sigmas(x)
        else:
            return self._forward(x)

    def _forward_with_sigmas(self, x):
        if isinstance(x, dict):
            curves, sigmas, bounds = x['scaled_curves'], x['scaled_sigmas'], x['scaled_bounds']
        else:
            curves, sigmas, bounds = split(x, self._split, dim=-1)
        x = cat([curves, sigmas, self.conv(curves), bounds], dim=-1)
        x = self.fc(x)
        if self.pass_bounds:
            x = cat([x, bounds], dim=-1)
        return x

    def _forward(self, x):
        if isinstance(x, dict):
            curves, bounds = x['scaled_curves'], x['scaled_bounds']
        else:
            curves, bounds = split(x, self._split, dim=-1)

        x = cat([curves, self.conv(curves), bounds], dim=-1)
        x = self.fc(x)
        if self.pass_bounds:
            x = cat([x, bounds], dim=-1)
        return x


def selu_init(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        m.weight.data.normal_(0.0, 0.5 / math.sqrt(m.weight.numel()))
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        size = m.weight.size()
        fan_in = size[0]

        m.weight.data.normal_(0.0, 1.0 / math.sqrt(fan_in))
        m.bias.data.fill_(0)
