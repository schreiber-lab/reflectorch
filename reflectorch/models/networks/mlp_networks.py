# -*- coding: utf-8 -*-

import math
from typing import Optional
import torch
from torch import nn, cat, split, Tensor

from reflectorch.models.networks.residual_net import ResidualMLP
from reflectorch.models.encoders.conv_encoder import ConvEncoder
from reflectorch.models.encoders.fno import FnoEncoder
from reflectorch.models.activations import activation_by_name

class NetworkWithPriors(nn.Module):
    """MLP network with an embedding network

    .. image:: ../documentation/FigureReflectometryNetwork.png
        :width: 800px
        :align: center

    Args:
        embedding_net_type (str): the type of embedding network, either 'conv' or 'fno'.
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
                 concat_condition_first_layer: bool = True):
        super().__init__()

        self.conditioning = conditioning
        self.dim_prior_bounds = 2 * dim_out
        self.dim_conditioning_params = dim_conditioning_params
        self.tanh_output = tanh_output

        if embedding_net_type == 'conv':
            self.embedding_net = ConvEncoder(**embedding_net_kwargs)
        elif embedding_net_type == 'fno':
            self.embedding_net = FnoEncoder(**embedding_net_kwargs)
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
        )

        if use_selu_init and embedding_net_kwargs.get('activation', None) == 'selu':
            self.embedding_net.apply(selu_init)
        if use_selu_init and mlp_activation == 'selu':
            self.mlp.apply(selu_init)

        if pretrained_embedding_net:
            self.embedding_net.load_weights(pretrained_embedding_net)


    def forward(self, curves, bounds, q_values=None, conditioning_params=None):
        """
        Args:
            scaled_curves (torch.Tensor): Input tensor of shape [batch_size, n_points] or [batch_size, n_channels, n_points].
            scaled_bounds (torch.Tensor): Tensor representing prior bounds, shape [batch_size, 2*n_params].
            scaled_q_values (torch.Tensor, optional): Tensor of shape [batch_size, n_points].
            scaled_conditioning_params (torch.Tensor, optional): Additional parameters for conditioning, shape [batch_size, ...].
        """

        if curves.dim() == 2:
            curves = curves.unsqueeze(1)

        additional_channels = []
        if q_values is not None:
            additional_channels.append(q_values.unsqueeze(1))

        if additional_channels:
            curves = torch.cat([curves] + additional_channels, dim=1)  # [batch_size, n_channels, n_points]

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

# class NetworkWithPriorsConvEmb(nn.Module):
#     """MLP network with 1D CNN embedding network

#     .. image:: ../documentation/FigureReflectometryNetwork.png
#         :width: 800px
#         :align: center

#     Args:
#         in_channels (int, optional): the number of input channels of the 1D CNN. Defaults to 1.
#         hidden_channels (tuple, optional): list with the number of channels for each layer of the 1D CNN. Defaults to (32, 64, 128, 256, 512).
#         dim_embedding (int, optional): the dimension of the embedding produced by the 1D CNN. Defaults to 128.
#         dim_avpool (int, optional): the type of activation function in the 1D CNN. Defaults to 1.
#         embedding_net_activation (str, optional): the type of activation function in the 1D CNN. Defaults to 'gelu'.
#         use_batch_norm (bool, optional): whether to use batch normalization (in both the 1D CNN and the MLP). Defaults to False.
#         dim_out (int, optional): the dimension of the output produced by the MLP. Defaults to 8.
#         layer_width (int, optional): the width of a linear layer in the MLP. Defaults to 512.
#         num_blocks (int, optional): the number of residual blocks in the MLP. Defaults to 4.
#         repeats_per_block (int, optional): the number of normalization/activation/linear repeats in a block. Defaults to 2.
#         mlp_activation (str, optional):  the type of activation function in the MLP. Defaults to 'gelu'.
#         dropout_rate (float, optional): dropout rate for each block. Defaults to 0.0.
#         use_selu_init (bool, optional): whether to use the special weights initialization for the 'selu' activation function. Defaults to False.
#         pretrained_embedding_net (str, optional): the path to the weights of a pretrained embedding network. Defaults to None.
#         residual (bool, optional): whether the blocks have a residual skip connection. Defaults to True.
#         adaptive_activation (bool, optional): must be set to ``True`` if the activation function is adaptive. Defaults to False.
#         conditioning (str, optional): the manner in which the prior bounds are provided as input to the network. Defaults to 'concat'.
#     """
#     def __init__(self,
#                  in_channels: int = 1,
#                  hidden_channels: tuple = (32, 64, 128, 256, 512),
#                  dim_embedding: int = 128,
#                  dim_avpool: int = 1,
#                  embedding_net_activation: str = 'gelu',
#                  use_batch_norm: bool = False,
#                  dim_out: int = 8,
#                  layer_width: int = 512,
#                  num_blocks: int = 4,
#                  repeats_per_block: int = 2,
#                  mlp_activation: str = 'gelu',
#                  dropout_rate: float = 0.0,
#                  use_selu_init: bool = False,
#                  pretrained_embedding_net: str = None,
#                  residual: bool = True,
#                  adaptive_activation: bool = False,
#                  conditioning: str = 'concat',
#                  ):
#         super().__init__()

#         self.in_channels = in_channels
#         self.conditioning = conditioning

#         self.embedding_net = ConvEncoder(
#             in_channels=in_channels, 
#             hidden_channels=hidden_channels, 
#             dim_latent=dim_embedding, 
#             dim_avpool=dim_avpool,
#             use_batch_norm=use_batch_norm, 
#             activation=embedding_net_activation
#         )
        
#         self.dim_prior_bounds = 2 * dim_out

#         if conditioning == 'concat':
#             dim_mlp_in = dim_embedding + self.dim_prior_bounds
#             dim_condition = 0
#         elif conditioning == 'glu' or conditioning == 'film':
#             dim_mlp_in = dim_embedding
#             dim_condition = self.dim_prior_bounds
#         else:
#             raise NotImplementedError

#         self.mlp = ResidualMLP(
#             dim_in=dim_mlp_in,
#             dim_out=dim_out,
#             dim_condition=dim_condition,
#             layer_width=layer_width,
#             num_blocks=num_blocks,
#             repeats_per_block=repeats_per_block,
#             activation=mlp_activation,
#             use_batch_norm=use_batch_norm,
#             dropout_rate=dropout_rate,
#             residual=residual,
#             adaptive_activation=adaptive_activation,
#             conditioning=conditioning,
#         )

#         if use_selu_init and embedding_net_activation == 'selu':
#             self.embedding_net.apply(selu_init)

#         if use_selu_init and mlp_activation == 'selu':
#             self.mlp.apply(selu_init)

#         if pretrained_embedding_net:
#             self.embedding_net.load_weights(pretrained_embedding_net)


#     def forward(self, curves: Tensor, bounds: Tensor, q_values: Optional[Tensor] = None):
#         """
#         Args:
#             curves (Tensor): reflectivity curves
#             bounds (Tensor): prior bounds
#             q_values (Tensor, optional): q values. Defaults to None.

#         Returns:
#             Tensor: prediction
#         """
#         if q_values is not None:
#             curves = torch.cat([curves[:, None, :], q_values[:, None, :]], dim=1)

#         if self.conditioning == 'concat': 
#             x = torch.cat([self.embedding_net(curves), bounds], dim=-1)
#             x = self.mlp(x)

#         elif self.conditioning == 'glu' or self.conditioning == 'film':
#             x = self.mlp(self.embedding_net(curves), condition=bounds)

#         return x
    
    
# class NetworkWithPriorsFnoEmb(nn.Module):
#     """MLP network with FNO embedding network

#     Args:
#         in_channels (int, optional): the number of input channels to the FNO-based embedding network. Defaults to 2.
#         dim_embedding (int, optional): the dimension of the embedding produced by the FNO. Defaults to 128.
#         modes (int, optional):  the number of Fourier modes that are utilized. Defaults to 16.
#         width_fno (int, optional): the number of channels in the FNO blocks. Defaults to 64.
#         embedding_net_activation (str, optional):  the type of activation function in the embedding network. Defaults to 'gelu'.
#         n_fno_blocks (int, optional): the number of FNO blocks. Defaults to 6.
#         fusion_self_attention (bool, optional): if ``True`` a fusion layer is used after the FNO blocks to produce the final output. Defaults to False.
#         dim_out (int, optional): the dimension of the output produced by the MLP. Defaults to 8.
#         layer_width (int, optional): the width of a linear layer in the MLP. Defaults to 512.
#         num_blocks (int, optional): the number of residual blocks in the MLP. Defaults to 4.
#         repeats_per_block (int, optional): the number of normalization/activation/linear repeats in a block. Defaults to 2.
#         use_batch_norm (bool, optional): whether to use batch normalization (only in the MLP). Defaults to False.
#         mlp_activation (str, optional):  the type of activation function in the MLP. Defaults to 'gelu'.
#         dropout_rate (float, optional): dropout rate for each block. Defaults to 0.0.
#         use_selu_init (bool, optional): whether to use the special weights initialization for the 'selu' activation function. Defaults to False.
#         residual (bool, optional): whether the blocks have a residual skip connection. Defaults to True.
#         adaptive_activation (bool, optional): must be set to ``True`` if the activation function is adaptive. Defaults to False.
#         conditioning (str, optional): the manner in which the prior bounds are provided as input to the network. Defaults to 'concat'.
#     """
#     def __init__(self,
#                  in_channels: int = 2,
#                  dim_embedding: int = 128,
#                  modes: int = 16,
#                  width_fno: int = 64,
#                  embedding_net_activation: str = 'gelu',
#                  n_fno_blocks : int = 6,
#                  fusion_self_attention: bool = False,
#                  dim_out: int = 8,
#                  layer_width: int = 512,
#                  num_blocks: int = 4,
#                  repeats_per_block: int = 2,
#                  use_batch_norm: bool = False,
#                  mlp_activation: str = 'gelu',
#                  dropout_rate: float = 0.0,
#                  use_selu_init: bool = False,
#                  residual: bool = True,
#                  adaptive_activation: bool = False,
#                  conditioning: str = 'concat',
#                  ):
#         super().__init__()

#         self.conditioning = conditioning

#         self.embedding_net = FnoEncoder(
#             ch_in=in_channels, 
#             dim_embedding=dim_embedding, 
#             modes=modes, 
#             width_fno=width_fno, 
#             n_fno_blocks=n_fno_blocks, 
#             activation=embedding_net_activation,
#             fusion_self_attention=fusion_self_attention
#         )

#         self.dim_prior_bounds = 2 * dim_out

#         if conditioning == 'concat':
#             dim_mlp_in = dim_embedding + self.dim_prior_bounds
#             dim_condition = 0
#         elif conditioning == 'glu' or conditioning == 'film':
#             dim_mlp_in = dim_embedding
#             dim_condition = self.dim_prior_bounds
#         else:
#             raise NotImplementedError

#         self.mlp = ResidualMLP(
#             dim_in=dim_mlp_in,
#             dim_out=dim_out,
#             dim_condition=dim_condition,
#             layer_width=layer_width,
#             num_blocks=num_blocks,
#             repeats_per_block=repeats_per_block,
#             activation=mlp_activation,
#             use_batch_norm=use_batch_norm,
#             dropout_rate=dropout_rate,
#             residual=residual,
#             adaptive_activation=adaptive_activation,
#             conditioning=conditioning,
#         )

#         if use_selu_init and embedding_net_activation == 'selu':
#             self.FnoEncoder.apply(selu_init)

#         if use_selu_init and mlp_activation == 'selu':
#             self.mlp.apply(selu_init)

            
#     def forward(self, curves: Tensor, bounds: Tensor, q_values: Optional[Tensor] =None):
#         """
#         Args:
#             curves (Tensor): reflectivity curves
#             bounds (Tensor): prior bounds
#             q_values (Tensor, optional): q values. Defaults to None.

#         Returns:
#             Tensor: prediction
#         """
#         if curves.dim() < 3:
#             curves = curves[:, None, :]
#         if q_values is not None:
#             curves = torch.cat([curves, q_values[:, None, :]], dim=1)

#         if self.conditioning == 'concat': 
#             x = torch.cat([self.embedding_net(curves), bounds], dim=-1)
#             x = self.mlp(x)

#         elif self.conditioning == 'glu' or self.conditioning == 'film':
#             x = self.mlp(self.embedding_net(curves), condition=bounds)

#         return x



def selu_init(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        m.weight.data.normal_(0.0, 0.5 / math.sqrt(m.weight.numel()))
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        size = m.weight.size()
        fan_in = size[0]

        m.weight.data.normal_(0.0, 1.0 / math.sqrt(fan_in))
        m.bias.data.fill_(0)
