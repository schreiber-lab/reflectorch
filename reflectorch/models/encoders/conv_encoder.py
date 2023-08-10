# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path

from torch import nn, load

from reflectorch.models.utils import activation_by_name
from reflectorch.paths import SAVED_MODELS_DIR

__all__ = [
    "ConvEncoder",
    "ConvDecoder",
    "Autoencoder",
]

logger = logging.getLogger(__name__)


class ConvEncoder(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 hidden_dims: tuple = (32, 64, 128, 256, 512),
                 latent_dim: int = 64,
                 avpool: int = 1,
                 use_batch_norm: bool = True,
                 activation: str = 'lrelu',
                 ):
        super().__init__()

        modules = []

        in_channels = in_channels
        activation = activation_by_name(activation)

        for h_dim in hidden_dims:
            layers = [
                nn.Conv1d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                activation,
            ]
            if use_batch_norm:
                layers.insert(1, nn.BatchNorm1d(h_dim))
            modules.append(nn.Sequential(*layers))
            in_channels = h_dim

        self.core = nn.Sequential(*modules)
        self.avpool = nn.AdaptiveAvgPool1d(avpool)
        self.fc = nn.Linear(hidden_dims[-1] * avpool, latent_dim)

    def forward(self, x):
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
    def __init__(self, hidden_dims: tuple = (512, 256, 128, 64, 32), latent_dim: int = 64, in_size: int = 8):
        super().__init__()

        self.in_size = in_size
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * in_size)

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_dims[-1], out_channels=1,
                      kernel_size=3, padding=1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.decoder_input(x).view(batch_size, -1, self.in_size)
        x = self.decoder(x)
        x = self.final_layer(x).flatten(1)
        return x


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))
