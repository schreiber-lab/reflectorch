# -*- coding: utf-8 -*-

import math
import torch
from torch import nn, cat, split

from reflectorch.models.encoders.residual_net import ResidualMLP
from reflectorch.models.encoders.conv_encoder import ConvEncoder
from reflectorch.models.encoders.fno import FnoEncoder
from reflectorch.models.utils import activation_by_name

class NetworkWithPriorsConvEmb(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 hidden_channels: tuple = (32, 64, 128, 256, 512),
                 dim_embedding: int = 128,
                 dim_avpool: int = 1,
                 embedding_net_activation: str = 'gelu',
                 use_batch_norm: bool = False,
                 dim_out: int = 8,
                 layer_width: int = 512,
                 num_blocks: int = 4,
                 repeats_per_block: int = 2,
                 mlp_activation: str = 'gelu',
                 dropout_rate: float = 0.0,
                 use_selu_init: bool = False,
                 pretrained_embedding_net: str = None,
                 residual: bool = True,
                 adaptive_activation: bool = False,
                 ):
        super().__init__()

        self.in_channels = in_channels

        self.embedding_net = ConvEncoder(
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            dim_latent=dim_embedding, 
            dim_avpool=dim_avpool,
            use_batch_norm=use_batch_norm, 
            activation=embedding_net_activation
        )
        
        self.dim_prior_bounds = 2 * dim_out
        dim_mlp_in = dim_embedding + self.dim_prior_bounds

        self.mlp = ResidualMLP(
            dim_in=dim_mlp_in,
            dim_out=dim_out,
            dim_condition=0,
            layer_width=layer_width,
            num_blocks=num_blocks,
            repeats_per_block=repeats_per_block,
            activation=mlp_activation,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            residual=residual,
            adaptive_activation=adaptive_activation,
        )

        if use_selu_init and embedding_net_activation == 'selu':
            self.embedding_net.apply(selu_init)

        if use_selu_init and mlp_activation == 'selu':
            self.mlp.apply(selu_init)

        if pretrained_embedding_net:
            self.embedding_net.load_weights(pretrained_embedding_net)


    def forward(self, x, q_values=None):
        if isinstance(x, dict):
            curves, bounds = x['scaled_curves'], x['scaled_bounds']
        else:
            curves, bounds = split(x, [x.shape[-1]-self.dim_prior_bounds, self.dim_prior_bounds], dim=-1)
            
        if q_values is not None:
            curves = cat([curves[:, None, :], q_values.float()[:, None, :]], dim=1)
        
        curves_embedding = self.embedding_net(curves)

        x = cat([curves_embedding, bounds], dim=-1)
        x = self.mlp(x)

        return x
    
    
class NetworkWithPriorsFnoEmb(nn.Module):
    def __init__(self,
                 in_channels: int = 2,
                 dim_embedding: int = 128,
                 modes: int = 16,
                 width_fno: int = 64,
                 embedding_net_activation: str = 'gelu',
                 n_fno_blocks : int = 6,
                 dim_out: int = 8,
                 layer_width: int = 64,
                 num_blocks: int = 3,
                 repeats_per_block: int = 2,
                 use_batch_norm: bool = False,
                 mlp_activation: str = 'gelu',
                 dropout_rate: float = 0.0,
                 use_selu_init: bool = False,
                 residual: bool = True,
                 adaptive_activation: bool = False,
                 ):
        super().__init__()

        self.embedding_net = FnoEncoder(
            ch_in=in_channels, 
            dim_embedding=dim_embedding, 
            modes=modes, 
            width_fno=width_fno, 
            n_fno_blocks=n_fno_blocks, 
            activation=embedding_net_activation
        )

        self.dim_prior_bounds = 2 * dim_out
        dim_mlp_in = dim_embedding + self.dim_prior_bounds

        self.mlp = ResidualMLP(
            dim_in=dim_mlp_in,
            dim_out=dim_out,
            dim_condition=None,
            layer_width=layer_width,
            num_blocks=num_blocks,
            repeats_per_block=repeats_per_block,
            activation=mlp_activation,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            residual=residual,
            adaptive_activation=adaptive_activation,
        )

        if use_selu_init and embedding_net_activation == 'selu':
            self.FnoEncoder.apply(selu_init)

        if use_selu_init and mlp_activation == 'selu':
            self.mlp.apply(selu_init)

            
    def forward(self, x, q_values=None):
        if isinstance(x, dict):
            curves, bounds = x['scaled_curves'], x['scaled_bounds']
        else:
            curves, bounds = split(x, [x.shape[-1]-self.dim_prior_bounds, self.dim_prior_bounds], dim=-1)
            
        if q_values is not None:
            q_input = 4 * q_values.float()[:, None, :] - 1
            curves = cat([curves[:, None, :], q_input], dim=1)
        else:
            curves = curves[:, None, :]

        x = cat([self.embedding_net(curves), bounds], dim=-1)
        x = self.mlp(x)

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
