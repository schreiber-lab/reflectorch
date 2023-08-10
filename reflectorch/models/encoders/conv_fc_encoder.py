# -*- coding: utf-8 -*-

import math
from torch import nn, cat, split

from reflectorch.models.encoders.residual_net import ResidualNet, ResidualNet_FiLM
from reflectorch.models.encoders.conv_encoder import ConvEncoder
from reflectorch.models.encoders.fno import FNO_Enc
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
    
    
class SubPriorConvFCEncoder_V2(nn.Module): #only the embedding and the prior are input to the MLP (the reflectivity curve is not concatenated)
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
                 sigmas_input: bool = False,
                 pretrained_conv: str = None,
                 ):
        super().__init__()
        self.pass_bounds = pass_bounds
        self.sigmas_input = sigmas_input
        self.in_channels = in_channels

        self.conv = ConvEncoder(
            in_channels=in_channels, hidden_dims=hidden_dims, latent_dim=conv_latent_dim, avpool=avpool,
            use_batch_norm=use_batch_norm, activation=activation_by_name(conv_activation)
        )
        
        self.prior_in_features = prior_in_features
        fc_in_dim = conv_latent_dim + prior_in_features

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

        x = cat([self.conv(curves), bounds], dim=-1)
        x = self.fc(x)
        if self.pass_bounds:
            x = cat([x, bounds], dim=-1)
        return x
    
    
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

        self.fc = ResidualNet_FiLM(
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
    
    
class SubPrior_FNO_MLP(nn.Module):
    def __init__(self,
                 latent_dim: int = 8, #output_dim
                 hidden_features: int = 64, #hidden layer width mlp
                 num_blocks: int = 3, #res_mlp blocks
                 prior_in_features: int = 16,
                 use_batch_norm: bool = False,
                 fc_activation: str = 'gelu',
                 
                 ch_in: int = 2,
                 dim_embedding: int = 128,
                 modes: int = 16,
                 width_fno: int = 64,
                 fno_activation: str = 'gelu',
                 n_fno_blocks : int = 6,
                 
                 use_selu_init: bool = False,
                 pass_bounds: bool = False,
                 ):
        super().__init__()
        self.pass_bounds = pass_bounds

        self.fno_enc = FNO_Enc(
            ch_in=ch_in, dim_embedding=dim_embedding, modes=modes, width_fno=width_fno, 
            n_fno_blocks=n_fno_blocks, activation=activation_by_name(fno_activation)
        )

        self.prior_in_features = prior_in_features
        fc_in_dim = dim_embedding + prior_in_features

        self.fc = ResidualNet(
            fc_in_dim,
            latent_dim,
            hidden_features,
            activation=activation_by_name(fc_activation),
            num_blocks=num_blocks,
            use_batch_norm=use_batch_norm
        )

        if use_selu_init and conv_activation == 'selu':
            self.fno_enc.apply(selu_init)

        if use_selu_init and fc_activation == 'selu':
            self.fc.apply(selu_init)

            
    def forward(self, x, q_values=None):
        if isinstance(x, dict):
            curves, bounds = x['scaled_curves'], x['scaled_bounds']
        else:
            curves, bounds = split(x, [x.shape[-1]-self.prior_in_features, self.prior_in_features], dim=-1)
            
        if q_values is not None:
            q_input = 4 * q_values.float()[:, None, :] - 1
            curves = cat([curves[:, None, :], q_input], dim=1)
        else:
            curves = curves[:, None, :]

        x = cat([self.fno_enc(curves), bounds], dim=-1)
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
