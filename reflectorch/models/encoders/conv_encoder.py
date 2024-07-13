# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path

import torch
from torch import nn, load

from reflectorch.models.activations import activation_by_name
from reflectorch.paths import SAVED_MODELS_DIR

__all__ = [
    "ConvEncoder",
    "ConvDecoder",
    "ConvAutoencoder",
    "ConvVAE",
]

logger = logging.getLogger(__name__)


class ConvEncoder(nn.Module):
    """A 1D CNN encoder / embedding network

    Args:
        in_channels (int, optional): the number of input channels. Defaults to 1.
        hidden_channels (tuple, optional): the number of intermediate channels of each convolutional layer. Defaults to (32, 64, 128, 256, 512).
        dim_latent (int, optional): the dimension of the output latent embedding. Defaults to 64.
        dim_avpool (int, optional): the output size of the adaptive average pooling layer. Defaults to 1.
        use_batch_norm (bool, optional): whether to use batch normalization. Defaults to True.
        activation (str, optional): the type of activation function. Defaults to 'relu'.
    """
    def __init__(self,
                 in_channels: int = 1,
                 hidden_channels: tuple = (32, 64, 128, 256, 512),
                 dim_latent: int = 64,
                 dim_avpool: int = 1,
                 use_batch_norm: bool = True,
                 activation: str = 'relu',
                 ):
        super().__init__()

        modules = []

        activation = activation_by_name(activation)

        for h in hidden_channels:
            layers = [
                nn.Conv1d(in_channels, out_channels=h, kernel_size=3, stride=2, padding=1),
                activation(),
            ]

            if use_batch_norm:
                layers.insert(1, nn.BatchNorm1d(h))

            modules.append(nn.Sequential(*layers))
            in_channels = h

        self.core = nn.Sequential(*modules)
        self.avpool = nn.AdaptiveAvgPool1d(dim_avpool)
        self.fc = nn.Linear(hidden_channels[-1] * dim_avpool, dim_latent)

    def forward(self, x):
        """"""
        if len(x.shape) < 3:
            x = x.unsqueeze(1)
        x = self.core(x)
        x = self.avpool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x

    def load_weights(self, path: str or Path = None, strict: bool = False):
        if not path:
            return

        if isinstance(path, str):
            if not path.endswith('.pt'):
                path = path + '.pt'
            path = SAVED_MODELS_DIR / path

        if not path.is_file():
            logger.error(f'File {str(path)} is not found.')
            return
        try:
            state_dict = load(path)
            self.load_state_dict(state_dict, strict=strict)
        except Exception as err:
            logger.exception(err)


class ConvDecoder(nn.Module):
    """A 1D CNN decoder

    Args:
        hidden_dims (tuple, optional): the number of intermediate channels of each convolutional layer. Defaults to (512, 256, 128, 64, 32).
        latent_dim (int, optional): the dimension of the input latent embedding. Defaults to 64.
        in_size (int, optional): the initial size for upscaling. Defaults to 8.
        use_batch_norm (bool, optional): whether to use batch normalization. Defaults to True.
        activation (str, optional): the type of activation function. Defaults to 'relu'.
    """
    def __init__(self, 
                 hidden_channels: tuple = (512, 256, 128, 64, 32), 
                 dim_latent: int = 64, 
                 in_size: int = 8,
                 use_batch_norm: bool = True,
                 activation: str = 'relu',
                 ):

        super().__init__()

        self.in_size = in_size
        modules = []

        self.decoder_input = nn.Linear(dim_latent, hidden_channels[0] * in_size)

        activation = activation_by_name(activation)

        for i in range(len(hidden_channels) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        hidden_channels[i],
                        hidden_channels[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm1d(hidden_channels[i + 1]) if use_batch_norm else nn.Identity(),
                    activation(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(hidden_channels[-1],
                               hidden_channels[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm1d(hidden_channels[-1]) if use_batch_norm else nn.Identity(),
            activation(),
            nn.Conv1d(hidden_channels[-1], out_channels=1,
                      kernel_size=3, padding=1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.decoder_input(x).view(batch_size, -1, self.in_size)
        x = self.decoder(x)
        x = self.final_layer(x).flatten(1)
        return x


class ConvAutoencoder(nn.Module):
    """A 1D convolutional denoising autoencoder"""
    def __init__(self,
                 in_channels: int = 1,
                 encoder_hidden_channels: tuple = (32, 64, 128, 256, 512),
                 decoder_hidden_channels: tuple = (512, 256, 128, 64, 32),
                 dim_latent: int = 64,
                 dim_avpool: int = 1,
                 use_batch_norm: bool = True,
                 activation: str = 'relu',
                 decoder_in_size: int = 8,
                 **kwargs
                 ):
        super().__init__()
        self.encoder = ConvEncoder(in_channels, encoder_hidden_channels, dim_latent, dim_avpool, use_batch_norm, activation, **kwargs)
        self.decoder = ConvDecoder(decoder_hidden_channels, dim_latent, decoder_in_size, use_batch_norm, activation, **kwargs)

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
class ConvVAE(nn.Module):
    """A 1D convolutional variational autoencoder"""
    def __init__(self,
                 in_channels: int = 1,
                 encoder_hidden_channels: tuple = (32, 64, 128, 256, 512),
                 decoder_hidden_channels: tuple = (512, 256, 128, 64, 32),
                 dim_latent: int = 64,
                 dim_avpool: int = 1,
                 use_batch_norm: bool = True,
                 activation: str = 'relu',
                 decoder_in_size: int = 8,
                 **kwargs
                 ):
        super().__init__()
        self.encoder = ConvEncoder(in_channels, encoder_hidden_channels, 2*dim_latent, dim_avpool, use_batch_norm, activation, **kwargs)
        self.decoder = ConvDecoder(decoder_hidden_channels, dim_latent, decoder_in_size, use_batch_norm, activation, **kwargs)

    def forward(self, x):
        z_mu, z_logvar = self.encoder(x).chunk(2, dim=-1)
        z = self.reparameterize(z_mu, z_logvar)

        x_r_mu, x_r_logvar = self.decoder(z).chunk(2, dim=-1)
        x = self.reparameterize(x_r_mu, x_r_logvar)

        return x, (z_mu, z_logvar, x_r_mu, x_r_logvar)
    
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(std)
        return mu + eps * std