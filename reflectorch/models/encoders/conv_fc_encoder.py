# -*- coding: utf-8 -*-

import math
from torch import nn, cat, split

from reflectorch.models.encoders.residual_net import ResidualMLP, ResidualMLP_FiLM
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
    


# class ConvFCEncoder(nn.Module):
#     def __init__(self,
#                  hidden_dims: tuple = (16, 32, 64),
#                  latent_dim: int = 64,
#                  avpool: int = 4,
#                  use_batch_norm: bool = False,
#                  in_features: int = 64,
#                  hidden_features: int = 64,
#                  num_blocks: int = 3,
#                  conv_activation: str = 'lrelu',
#                  fc_activation: str = 'relu',
#                  use_selu_init: bool = False,
#                  pretrained_conv: str = None,
#                  ):
#         super().__init__()
#         self.conv = ConvEncoder(
#             hidden_dims=hidden_dims, latent_dim=latent_dim, avpool=avpool, use_batch_norm=use_batch_norm,
#             activation=activation_by_name(conv_activation)
#         )
#         self.fc = ResidualMLP(
#             in_features, latent_dim, hidden_features, num_blocks=num_blocks, use_batch_norm=use_batch_norm,
#             activation=activation_by_name(fc_activation)
#         )
#         self.cat_fc = nn.Linear(latent_dim * 2, latent_dim)

#         if use_selu_init and conv_activation == 'selu':
#             self.conv.apply(selu_init)

#         if use_selu_init and fc_activation == 'selu':
#             self.fc.apply(selu_init)

#         if pretrained_conv:
#             self.conv.load_weights(pretrained_conv)

#     def forward(self, x):
#         x = cat([self.conv(x), self.fc(x)], dim=-1)
#         x = self.cat_fc(x)
#         return x
    

# class SeqConvFCEncoder(nn.Module):
#     def __init__(self,
#                  hidden_dims: tuple = (32, 64, 128, 256, 512),
#                  latent_dim: int = 128,
#                  conv_latent_dim: int = 64,
#                  avpool: int = 4,
#                  use_batch_norm: bool = False,
#                  in_features: int = 64,
#                  hidden_features: int = 256,
#                  num_blocks: int = 3,
#                  conv_activation: str = 'lrelu',
#                  fc_activation: str = 'relu',
#                  use_selu_init: bool = False,
#                  sigmas_input: bool = False,
#                  pretrained_conv: str = None,
#                  ):
#         super().__init__()

#         self.sigmas_input = sigmas_input

#         self.conv = ConvEncoder(
#             hidden_dims=hidden_dims,
#             latent_dim=conv_latent_dim,
#             avpool=avpool,
#             use_batch_norm=use_batch_norm,
#             activation=activation_by_name(conv_activation),
#         )

#         fc_in_dim = in_features + conv_latent_dim

#         if self.sigmas_input:
#             fc_in_dim += in_features
#             self._split = [in_features, in_features]
#         else:
#             self._split = None

#         self.fc = ResidualMLP(
#             fc_in_dim,
#             latent_dim,
#             hidden_features,
#             num_blocks=num_blocks,
#             use_batch_norm=use_batch_norm,
#             activation=activation_by_name(fc_activation),
#         )

#         if use_selu_init and conv_activation == 'selu':
#             self.conv.apply(selu_init)

#         if use_selu_init and fc_activation == 'selu':
#             self.fc.apply(selu_init)

#         if pretrained_conv:
#             self.conv.load_weights(pretrained_conv)

#     def forward(self, x):
#         if self.sigmas_input:
#             return self._forward_with_sigmas(x)
#         else:
#             return self._forward(x)

#     def _forward_with_sigmas(self, x):
#         if isinstance(x, dict):
#             curves, sigmas = x['scaled_curves'], x['scaled_sigmas']
#         else:
#             curves, sigmas = split(x, self._split, dim=-1)

#         x = cat([curves, sigmas, self.conv(curves)], dim=-1)
#         x = self.fc(x)
#         return x

#     def _forward(self, x):
#         x = cat([self.conv(x), x], dim=-1)
#         x = self.fc(x)
#         return x

# class SubPriorConvFCEncoder(nn.Module):
#     def __init__(self,
#                  hidden_dims: tuple = (16, 32, 64),
#                  latent_dim: int = 64,
#                  conv_latent_dim: int = 64,
#                  avpool: int = 8,
#                  use_batch_norm: bool = False,
#                  in_features: int = 64,
#                  prior_in_features: int = 16,
#                  hidden_features: int = 64,
#                  num_blocks: int = 3,
#                  conv_activation: str = 'lrelu',
#                  fc_activation: str = 'gelu',
#                  use_selu_init: bool = False,
#                  pass_bounds: bool = False,
#                  sigmas_input: bool = False,
#                  pretrained_conv: str = None,
#                  ):
#         super().__init__()
#         self.pass_bounds = pass_bounds
#         self.sigmas_input = sigmas_input

#         self.conv = ConvEncoder(
#             hidden_dims=hidden_dims, latent_dim=conv_latent_dim, avpool=avpool,
#             use_batch_norm=use_batch_norm, activation=activation_by_name(conv_activation)
#         )

#         fc_in_dim = in_features + conv_latent_dim + prior_in_features

#         if self.sigmas_input:
#             fc_in_dim += in_features

#         self.fc = ResidualMLP(
#             fc_in_dim,
#             latent_dim,
#             hidden_features,
#             activation=activation_by_name(fc_activation),
#             num_blocks=num_blocks,
#             use_batch_norm=use_batch_norm
#         )
#         if self.sigmas_input:
#             self._split = [in_features, in_features, prior_in_features]
#         else:
#             self._split = [in_features, prior_in_features]

#         if use_selu_init and conv_activation == 'selu':
#             self.conv.apply(selu_init)

#         if use_selu_init and fc_activation == 'selu':
#             self.fc.apply(selu_init)

#         if pretrained_conv:
#             self.conv.load_weights(pretrained_conv)

#     def forward(self, x):
#         if self.sigmas_input:
#             return self._forward_with_sigmas(x)
#         else:
#             return self._forward(x)

#     def _forward_with_sigmas(self, x):
#         if isinstance(x, dict):
#             curves, sigmas, bounds = x['scaled_curves'], x['scaled_sigmas'], x['scaled_bounds']
#         else:
#             curves, sigmas, bounds = split(x, self._split, dim=-1)
#         x = cat([curves, sigmas, self.conv(curves), bounds], dim=-1)
#         x = self.fc(x)
#         if self.pass_bounds:
#             x = cat([x, bounds], dim=-1)
#         return x

#     def _forward(self, x):
#         if isinstance(x, dict):
#             curves, bounds = x['scaled_curves'], x['scaled_bounds']
#         else:
#             curves, bounds = split(x, self._split, dim=-1)

#         x = cat([curves, self.conv(curves), bounds], dim=-1)
#         x = self.fc(x)
#         if self.pass_bounds:
#             x = cat([x, bounds], dim=-1)
#         return x
    

# class SubPriorNetwork(nn.Module):
#     def __init__(self, main_network, embedding_network=nn.Identity):
#         super().__init__()

#         self.main_network = main_network
#         self.embedding_network = embedding_network
    
#     def forward(self, x, q_values):

    
class SubPriorConvFCEncoder_FiLM(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 hidden_dims: tuple = (16, 32, 64),
                 latent_dim: int = 64,
                 conv_latent_dim: int = 64,
                 avpool: int = 8,
                 use_batch_norm: bool = False,
                 in_features: int = 64,
                 prior_in_features: int = 16,
                 hidden_features: int = 64,
                 num_blocks: int = 3,
                 conv_activation: str = 'gelu',
                 fc_activation: str = 'gelu',
                 use_selu_init: bool = False,
                 pass_bounds: bool = False,
                 pretrained_conv: str = None,
                 ):
        super().__init__()
        self.pass_bounds = pass_bounds
        self.in_channels = in_channels

        self.conv = ConvEncoder(
            in_channels=in_channels, hidden_dims=hidden_dims, latent_dim=conv_latent_dim, avpool=avpool,
            use_batch_norm=use_batch_norm, activation=activation_by_name(conv_activation)
        )
        
        self.prior_in_features = prior_in_features
        fc_in_dim = conv_latent_dim

        self.fc = ResidualMLP_FiLM(
            fc_in_dim,
            prior_in_features,
            latent_dim,
            hidden_features,
            activation=activation_by_name(fc_activation),
            num_blocks=num_blocks,
            use_batch_norm=use_batch_norm
        )

        if use_selu_init and conv_activation == 'selu':
            self.conv.apply(selu_init)

        if use_selu_init and fc_activation == 'selu':
            self.fc.apply(selu_init)

        if pretrained_conv:
            self.conv.load_weights(pretrained_conv)


    def forward(self, x, q_values=None):
        if isinstance(x, dict):
            curves, bounds = x['scaled_curves'], x['scaled_bounds']
        else:
            #curves, bounds = split(x, self._split, dim=-1)
            curves, bounds = split(x, [x.shape[-1]-self.prior_in_features, self.prior_in_features], dim=-1)
            
        if q_values is not None:
            curves = cat([curves[:, None, :], q_values.float()[:, None, :]], dim=1)

        x = self.conv(curves)
        x = self.fc(x, bounds)
        if self.pass_bounds:
            x = cat([x, bounds], dim=-1)
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
                 ):
        super().__init__()

        self.embedding_net = FnoEncoder(
            ch_in=in_channels, 
            dim_embedding=dim_embedding, 
            modes=modes, 
            width_fno=width_fno, 
            n_fno_blocks=n_fno_blocks, 
            activation=activation_by_name(embedding_net_activation)
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
