# -*- coding: utf-8 -*-

import math
from typing import Optional
import torch
from torch import nn, cat, split, Tensor

from reflectorch.models.networks.residual_net import ResidualMLP
from reflectorch.models.encoders.conv_encoder import ConvEncoder
from reflectorch.models.encoders.integral_kernel_embedding import IntegralConvEmbedding
from reflectorch.models.encoders.fno import FnoEncoder
from reflectorch.models.activations import activation_by_name

class NetworkWithPriors(nn.Module):
    """MLP network with an embedding network

    .. image:: ../documentation/FigureReflectometryNetwork.png
        :width: 800px
        :align: center

    Args:
        embedding_net_type (str): the type of embedding network, either 'conv', 'fno' or 'integral_conv'.
        embedding_net_kwargs (dict): dictionary containing the keyword arguments for the embedding network.
        dim_out (int, optional): the dimension of the output produced by the MLP. Defaults to 8.
        dim_conditioning_params (int, optional): the dimension of other parameters the network is conditioned on (e.g. for the smearing coefficient dq/q)
        layer_width (int, optional): the width of a linear layer in the MLP. Defaults to 512.
        num_blocks (int, optional): the number of residual blocks in the MLP. Defaults to 4.
        repeats_per_block (int, optional): the number of normalization/activation/linear repeats in a block. Defaults to 2.
        mlp_activation (str, optional):  the type of activation function in the MLP. Defaults to 'gelu'.
        use_batch_norm (bool, optional): whether to use batch normalization in the MLP. Defaults to True.
        use_layer_norm (bool, optional): whether to use layer normalization in the MLP (if use_batch_norm is False). Defaults to False.
        dropout_rate (float, optional): dropout rate for each block. Defaults to 0.0.
        tanh_output (bool, optional): whether to apply a tanh function to the output. Defaults to False.
        use_selu_init (bool, optional): whether to use the special weights initialization for the 'selu' activation function. Defaults to False.
        pretrained_embedding_net (str, optional): the path to the weights of a pretrained embedding network. Defaults to None.
        residual (bool, optional): whether the blocks have a residual skip connection. Defaults to True.
        adaptive_activation (bool, optional): must be set to ``True`` if the activation function is adaptive. Defaults to False.
        conditioning (str, optional): the manner in which the prior bounds are provided as input to the network. Defaults to 'concat'.
      """
    def __init__(self,
                 embedding_net_type: str,  # 'conv', 'fno'
                 embedding_net_kwargs: dict,
                 pretrained_embedding_net: str = None,
                 dim_out: int = 8,
                 dim_conditioning_params: int = 0,
                 layer_width: int = 512,
                 num_blocks: int = 4,
                 repeats_per_block: int = 2,
                 mlp_activation: str = 'gelu',
                 use_batch_norm: bool = True,
                 use_layer_norm: bool = False,
                 dropout_rate: float = 0.0,
                 tanh_output: bool = False,
                 use_selu_init: bool = False,
                 residual: bool = True,
                 adaptive_activation: bool = False,
                 conditioning: str = 'concat',
                 concat_condition_first_layer: bool = True,
                 zero_init_last: bool = False,
                 zero_init_blocks: bool = False):
        super().__init__()

        self.conditioning = conditioning
        self.dim_prior_bounds = 2 * dim_out
        self.dim_conditioning_params = dim_conditioning_params
        self.tanh_output = tanh_output

        if embedding_net_type == 'conv':
            self.embedding_net = ConvEncoder(**embedding_net_kwargs)
        elif embedding_net_type == 'fno':
            self.embedding_net = FnoEncoder(**embedding_net_kwargs)
        elif embedding_net_type == 'integral_conv':
            self.embedding_net = IntegralConvEmbedding(**embedding_net_kwargs)
        elif embedding_net_type == 'no_embedding_net':
            self.embedding_net = nn.Identity()
        else:
            raise ValueError(f"Unsupported embedding_net_type: {embedding_net_type}")

        self.dim_embedding = embedding_net_kwargs['dim_embedding']

        if conditioning == 'concat':
            dim_mlp_in = self.dim_embedding + self.dim_prior_bounds + self.dim_conditioning_params
            dim_condition = 0
        elif conditioning == 'glu' or conditioning == 'film':
            dim_mlp_in = self.dim_embedding
            dim_condition = self.dim_prior_bounds + self.dim_conditioning_params
        else:
            raise NotImplementedError(f"Conditioning type '{conditioning}' is not supported.")

        self.mlp = ResidualMLP(
            dim_in=dim_mlp_in,
            dim_out=dim_out,
            dim_condition=dim_condition,
            layer_width=layer_width,
            num_blocks=num_blocks,
            repeats_per_block=repeats_per_block,
            activation=mlp_activation,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
            residual=residual,
            adaptive_activation=adaptive_activation,
            conditioning=conditioning,
            concat_condition_first_layer=concat_condition_first_layer,
            zero_init_last=zero_init_last,
            zero_init_blocks=zero_init_blocks,
        )

        if use_selu_init and embedding_net_kwargs.get('activation', None) == 'selu':
            self.embedding_net.apply(selu_init)
        if use_selu_init and mlp_activation == 'selu':
            self.mlp.apply(selu_init)

        if pretrained_embedding_net:
            self.embedding_net.load_weights(pretrained_embedding_net)


    def forward(self, curves, bounds, q_values=None, sigmas=None, conditioning_params=None, key_padding_mask=None, unscaled_q_values=None):
        """
        Args:
            scaled_curves (torch.Tensor): Input tensor of shape [batch_size, n_points] or [batch_size, n_channels, n_points].
            scaled_bounds (torch.Tensor): Tensor representing prior bounds, shape [batch_size, 2*n_params].
            scaled_q_values (torch.Tensor, optional): Tensor of shape [batch_size, n_points].
            scaled_sigmas (torch.Tensor, optional): Tensor of shape [batch_size, n_points].
            scaled_conditioning_params (torch.Tensor, optional): Additional parameters for conditioning, shape [batch_size, ...].
        """

        if curves.dim() == 2:
            curves = curves.unsqueeze(1)

        additional_channels = []
        if q_values is not None and not isinstance(self.embedding_net, IntegralConvEmbedding):
            additional_channels.append(q_values.unsqueeze(1))
        if sigmas is not None:
            additional_channels.append(sigmas.unsqueeze(1))

        if additional_channels:
            curves = torch.cat([curves] + additional_channels, dim=1)  # [batch_size, n_channels, n_points]

        if isinstance(self.embedding_net, IntegralConvEmbedding):
            x = self.embedding_net(q=unscaled_q_values.float(), y=curves.permute(0, 2, 1), drop_mask=key_padding_mask)
        else:
            x = self.embedding_net(curves)

        if self.conditioning == 'concat': 
            x = torch.cat([x, bounds] + ([conditioning_params] if conditioning_params is not None else []), dim=-1)
            x = self.mlp(x)

        elif self.conditioning in ['glu', 'film']:
            condition = torch.cat([bounds] + ([conditioning_params] if conditioning_params is not None else []), dim=-1)
            x = self.mlp(x, condition=condition)

        else:
            raise NotImplementedError(f"Conditioning type {self.conditioning} not recognized.")

        if self.tanh_output:
            x = torch.tanh(x)

        return x

class NetworkWithPriorsConvEmb(NetworkWithPriors):
    """Wrapper for back-compatibility with previous versions of the package"""
    def __init__(self, **kwargs):
        embedding_net_kwargs = {
            'in_channels': kwargs.pop('in_channels', 1),
            'hidden_channels': kwargs.pop('hidden_channels', [32, 64, 128, 256, 512]),
            'dim_embedding': kwargs.pop('dim_embedding', 128),
            'dim_avpool': kwargs.pop('dim_avpool', 1),
            'activation': kwargs.pop('embedding_net_activation', 'gelu'),
            'use_batch_norm': kwargs.pop('use_batch_norm', False),
        }

        super().__init__(
            embedding_net_type='conv',
            embedding_net_kwargs=embedding_net_kwargs,
            **kwargs
        )

class NetworkWithPriorsFnoEmb(NetworkWithPriors):
    """Wrapper for back-compatibility with previous versions of the package"""
    def __init__(self, **kwargs):
        embedding_net_kwargs = {
            'in_channels': kwargs.pop('in_channels', 2),
            'dim_embedding': kwargs.pop('dim_embedding', 128),
            'modes': kwargs.pop('modes', 16),
            'width_fno': kwargs.pop('width_fno', 64),
            'n_fno_blocks': kwargs.pop('n_fno_blocks', 6),
            'activation': kwargs.pop('embedding_net_activation', 'gelu'),
            'fusion_self_attention': kwargs.pop('fusion_self_attention', False),
        }

        super().__init__(
            embedding_net_type='fno',
            embedding_net_kwargs=embedding_net_kwargs,
            **kwargs
        )

def selu_init(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        m.weight.data.normal_(0.0, 0.5 / math.sqrt(m.weight.numel()))
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        size = m.weight.size()
        fan_in = size[0]

        m.weight.data.normal_(0.0, 1.0 / math.sqrt(fan_in))
        m.bias.data.fill_(0)
