###as in the PANPE repository

from collections.abc import Callable

import torch
from torch import nn

from nflows.distributions import StandardNormal

from nflows.transforms import (
    PiecewiseRationalQuadraticCouplingTransform,
    ReversePermutation,
    BatchNorm,
    CompositeTransform,
    LULinear,
)

from reflectorch.models.networks.nf.nf_wrapper import FlowWrapper
from reflectorch.models.networks.residual_net import ResidualMLP
from reflectorch.models.activations import activation_by_name

__all__ = [
    "get_rq_nsf_c_flow",
    "get_residual_transform_net_fn",
]


def get_rq_nsf_c_flow(
    features: int,
    transform_net_fn,
    embedding_net: nn.Module = None,
    num_layers: int = 10,
    tail_bound: float = 10.0,
    tails: str = "linear",
    num_bins: int = 8,
    use_batch_norm_transform: bool = True,
    use_lu: bool = False,
    min_bin_width: float = 1e-6,
    min_bin_height: float = 1e-6,
    min_derivative: float = 1e-6,
    **kwargs,
) -> FlowWrapper:

    base_dist = StandardNormal(shape=[features])
    transforms = []

    n_masked, n_unmasked = features // 2, features - features // 2

    mask = torch.cat([torch.ones(n_masked), torch.zeros(n_unmasked)])

    permute_transform = LULinear if use_lu else ReversePermutation

    for layer in range(num_layers):
        transform_net = transform_net_fn
        new_mask = mask.clone()
        transforms.append(permute_transform(features=features))

        transforms.append(
            PiecewiseRationalQuadraticCouplingTransform(
                new_mask,
                transform_net,
                num_bins=num_bins,
                tail_bound=tail_bound,
                tails=tails,
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height,
                min_derivative=min_derivative,
                **kwargs,
            )
        )

        if use_batch_norm_transform:
            transforms.append(BatchNorm(features))

    transform = CompositeTransform(transforms)

    flow = FlowWrapper(transform, base_dist, embedding_net)

    return flow


def get_residual_transform_net_fn(
    context_features: int = 64,
    hidden_features: int = 64,
    activation: str = "lrelu",
    use_batch_norm: bool = True,
    num_blocks=3,
    nn_cls: Callable = ResidualMLP,
    **kwargs,
):
    def func(num_identity_features: int, num_transform_features: int) -> nn.Module:
        return nn_cls(
            dim_in=num_identity_features,
            dim_out=num_transform_features,
            dim_condition=context_features,
            layer_width=hidden_features,
            activation=activation_by_name(activation),
            num_blocks=num_blocks,
            use_batch_norm=use_batch_norm,
            **kwargs,
        )

    return func