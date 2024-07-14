# -*- coding: utf-8 -*-

import math
from typing import Optional
import torch
from torch import nn, cat, split, Tensor

from reflectorch.models.networks.residual_net import ResidualMLP
from reflectorch.models.encoders.conv_encoder import ConvEncoder
from reflectorch.models.encoders.fno import FnoEncoder
from reflectorch.models.activations import activation_by_name

class NetworkWithPriorsConvEmb(nn.Module):
    """MLP network with 1D CNN embedding network

    .. image:: ../documentation/FigureReflectometryNetwork.png
        :width: 800px
        :align: center

    Args:
        in_channels (int, optional): the number of input channels of the 1D CNN. Defaults to 1.
        hidden_channels (tuple, optional): list with the number of channels for each layer of the 1D CNN. Defaults to (32, 64, 128, 256, 512).
        dim_embedding (int, optional): the dimension of the embedding produced by the 1D CNN. Defaults to 128.
        dim_avpool (int, optional): the type of activation function in the 1D CNN. Defaults to 1.
        embedding_net_activation (str, optional): the type of activation function in the 1D CNN. Defaults to 'gelu'.
        use_batch_norm (bool, optional): whether to use batch normalization (in both the 1D CNN and the MLP). Defaults to False.
        dim_out (int, optional): the dimension of the output produced by the MLP. Defaults to 8.
        layer_width (int, optional): the width of a linear layer in the MLP. Defaults to 512.
        num_blocks (int, optional): the number of residual blocks in the MLP. Defaults to 4.
        repeats_per_block (int, optional): the number of normalization/activation/linear repeats in a block. Defaults to 2.
        mlp_activation (str, optional):  the type of activation function in the MLP. Defaults to 'gelu'.
        dropout_rate (float, optional): dropout rate for each block. Defaults to 0.0.
        use_selu_init (bool, optional): whether to use the special weights initialization for the 'selu' activation function. Defaults to False.
        pretrained_embedding_net (str, optional): the path to the weights of a pretrained embedding network. Defaults to None.
        residual (bool, optional): whether the blocks have a residual skip connection. Defaults to True.
        adaptive_activation (bool, optional): must be set to ``True`` if the activation function is adaptive. Defaults to False.
        conditioning (str, optional): the manner in which the prior bounds are provided as input to the network. Defaults to 'concat'.
    """
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
                 conditioning: str = 'concat',
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.conditioning = conditioning

        self.embedding_net = ConvEncoder(
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            dim_latent=dim_embedding, 
            dim_avpool=dim_avpool,
            use_batch_norm=use_batch_norm, 
            activation=embedding_net_activation
        )
        
        self.dim_prior_bounds = 2 * dim_out

        if conditioning == 'concat':
            dim_mlp_in = dim_embedding + self.dim_prior_bounds
            dim_condition = 0
        elif conditioning == 'glu' or conditioning == 'film':
            dim_mlp_in = dim_embedding
            dim_condition = self.dim_prior_bounds
        else:
            raise NotImplementedError

        self.mlp = ResidualMLP(
            dim_in=dim_mlp_in,
            dim_out=dim_out,
            dim_condition=dim_condition,
            layer_width=layer_width,
            num_blocks=num_blocks,
            repeats_per_block=repeats_per_block,
            activation=mlp_activation,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            residual=residual,
            adaptive_activation=adaptive_activation,
            conditioning=conditioning,
        )

        if use_selu_init and embedding_net_activation == 'selu':
            self.embedding_net.apply(selu_init)

        if use_selu_init and mlp_activation == 'selu':
            self.mlp.apply(selu_init)

        if pretrained_embedding_net:
            self.embedding_net.load_weights(pretrained_embedding_net)


    def forward(self, curves: Tensor, bounds: Tensor, q_values: Optional[Tensor] = None):
        """
        Args:
            curves (Tensor): reflectivity curves
            bounds (Tensor): prior bounds
            q_values (Tensor, optional): q values. Defaults to None.

        Returns:
            Tensor: prediction
        """
        if q_values is not None:
            curves = torch.cat([curves[:, None, :], q_values[:, None, :]], dim=1)

        if self.conditioning == 'concat': 
            x = torch.cat([self.embedding_net(curves), bounds], dim=-1)
            x = self.mlp(x)

        elif self.conditioning == 'glu' or self.conditioning == 'film':
            x = self.mlp(self.embedding_net(curves), condition=bounds)

        return x
    
    
class NetworkWithPriorsFnoEmb(nn.Module):
    """MLP network with FNO embedding network

    Args:
        in_channels (int, optional): the number of input channels to the FNO-based embedding network. Defaults to 2.
        dim_embedding (int, optional): the dimension of the embedding produced by the FNO. Defaults to 128.
        modes (int, optional):  the number of Fourier modes that are utilized. Defaults to 16.
        width_fno (int, optional): the number of channels in the FNO blocks. Defaults to 64.
        embedding_net_activation (str, optional):  the type of activation function in the embedding network. Defaults to 'gelu'.
        n_fno_blocks (int, optional): the number of FNO blocks. Defaults to 6.
        fusion_self_attention (bool, optional): if ``True`` a fusion layer is used after the FNO blocks to produce the final output. Defaults to False.
        dim_out (int, optional): the dimension of the output produced by the MLP. Defaults to 8.
        layer_width (int, optional): the width of a linear layer in the MLP. Defaults to 512.
        num_blocks (int, optional): the number of residual blocks in the MLP. Defaults to 4.
        repeats_per_block (int, optional): the number of normalization/activation/linear repeats in a block. Defaults to 2.
        use_batch_norm (bool, optional): whether to use batch normalization (only in the MLP). Defaults to False.
        mlp_activation (str, optional):  the type of activation function in the MLP. Defaults to 'gelu'.
        dropout_rate (float, optional): dropout rate for each block. Defaults to 0.0.
        use_selu_init (bool, optional): whether to use the special weights initialization for the 'selu' activation function. Defaults to False.
        residual (bool, optional): whether the blocks have a residual skip connection. Defaults to True.
        adaptive_activation (bool, optional): must be set to ``True`` if the activation function is adaptive. Defaults to False.
        conditioning (str, optional): the manner in which the prior bounds are provided as input to the network. Defaults to 'concat'.
    """
    def __init__(self,
                 in_channels: int = 2,
                 dim_embedding: int = 128,
                 modes: int = 16,
                 width_fno: int = 64,
                 embedding_net_activation: str = 'gelu',
                 n_fno_blocks : int = 6,
                 fusion_self_attention: bool = False,
                 dim_out: int = 8,
                 layer_width: int = 512,
                 num_blocks: int = 4,
                 repeats_per_block: int = 2,
                 use_batch_norm: bool = False,
                 mlp_activation: str = 'gelu',
                 dropout_rate: float = 0.0,
                 use_selu_init: bool = False,
                 residual: bool = True,
                 adaptive_activation: bool = False,
                 conditioning: str = 'concat',
                 ):
        super().__init__()

        self.conditioning = conditioning

        self.embedding_net = FnoEncoder(
            ch_in=in_channels, 
            dim_embedding=dim_embedding, 
            modes=modes, 
            width_fno=width_fno, 
            n_fno_blocks=n_fno_blocks, 
            activation=embedding_net_activation,
            fusion_self_attention=fusion_self_attention
        )

        self.dim_prior_bounds = 2 * dim_out

        if conditioning == 'concat':
            dim_mlp_in = dim_embedding + self.dim_prior_bounds
            dim_condition = 0
        elif conditioning == 'glu' or conditioning == 'film':
            dim_mlp_in = dim_embedding
            dim_condition = self.dim_prior_bounds
        else:
            raise NotImplementedError

        self.mlp = ResidualMLP(
            dim_in=dim_mlp_in,
            dim_out=dim_out,
            dim_condition=dim_condition,
            layer_width=layer_width,
            num_blocks=num_blocks,
            repeats_per_block=repeats_per_block,
            activation=mlp_activation,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            residual=residual,
            adaptive_activation=adaptive_activation,
            conditioning=conditioning,
        )

        if use_selu_init and embedding_net_activation == 'selu':
            self.FnoEncoder.apply(selu_init)

        if use_selu_init and mlp_activation == 'selu':
            self.mlp.apply(selu_init)

            
    def forward(self, curves: Tensor, bounds: Tensor, q_values: Optional[Tensor] =None):
        """
        Args:
            curves (Tensor): reflectivity curves
            bounds (Tensor): prior bounds
            q_values (Tensor, optional): q values. Defaults to None.

        Returns:
            Tensor: prediction
        """
        if curves.dim() < 3:
            curves = curves[:, None, :]
        if q_values is not None:
            curves = torch.cat([curves, q_values[:, None, :]], dim=1)

        if self.conditioning == 'concat': 
            x = torch.cat([self.embedding_net(curves), bounds], dim=-1)
            x = self.mlp(x)

        elif self.conditioning == 'glu' or self.conditioning == 'film':
            x = self.mlp(self.embedding_net(curves), condition=bounds)

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
