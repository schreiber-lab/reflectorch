# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union
from math import sqrt, pi, log10

import torch
from torch import Tensor

__all__ = [
    "get_reversed_params",
    "get_density_profiles",
    "uniform_sampler",
    "logdist_sampler",
    "triangular_sampler",
    "get_param_labels",
    "get_d_rhos",
    "get_slds_from_d_rhos",
]


def uniform_sampler(low: Union[float, Tensor], high: Union[float, Tensor], *shape, device=None, dtype=None):
    if isinstance(low, Tensor):
        device, dtype = low.device, low.dtype
    return torch.rand(*shape, device=device, dtype=dtype) * (high - low) + low


def logdist_sampler(low: Union[float, Tensor], high: Union[float, Tensor], *shape, device=None, dtype=None):
    if isinstance(low, Tensor):
        device, dtype = low.device, low.dtype
        low, high = map(torch.log10, (low, high))
    else:
        low, high = map(log10, (low, high))
    return 10 ** (torch.rand(*shape, device=device, dtype=dtype) * (high - low) + low)


def triangular_sampler(low: Union[float, Tensor], high: Union[float, Tensor], *shape, device=None, dtype=None):
    if isinstance(low, Tensor):
        device, dtype = low.device, low.dtype

    x = torch.rand(*shape, device=device, dtype=dtype)

    return (high - low) * (1 - torch.sqrt(x)) + low


def get_reversed_params(thicknesses: Tensor, roughnesses: Tensor, slds: Tensor):
    reversed_slds = torch.cumsum(
        torch.flip(
            torch.diff(
                torch.cat([torch.zeros(slds.shape[0], 1).to(slds), slds], dim=-1),
                dim=-1
            ), (-1,)
        ),
        dim=-1
    )
    reversed_thicknesses = torch.flip(thicknesses, [-1])
    reversed_roughnesses = torch.flip(roughnesses, [-1])
    reversed_params = torch.cat([reversed_thicknesses, reversed_roughnesses, reversed_slds], -1)

    return reversed_params


def get_density_profiles(
        thicknesses: Tensor,
        roughnesses: Tensor,
        slds: Tensor,
        z_axis: Tensor = None,
        num: int = 1000
):
    """Generates SLD profiles (and their derivative) based on batches of thicknesses, roughnesses and layer SLDs. 
    
    The axis has its zero at the top (ambient medium) interface and is positive inside the film.

    Args:
        thicknesses (Tensor): the layer thicknesses (top to bottom)
        roughnesses (Tensor): the interlayer roughnesses (top to bottom)
        slds (Tensor): the layer SLDs (top to bottom)
        z_axis (Tensor, optional): a custom depth (z) axis. Defaults to None.
        num (int, optional): number of discretization points for the profile. Defaults to 1000.

    Returns:
        tuple: the z axis, the computed density profile rho(z) and the derivative of the density profile drho/dz(z)
    """
    assert torch.all(roughnesses >= 0), 'Negative roughness happened'
    assert torch.all(thicknesses >= 0), 'Negative thickness happened'

    sample_num = thicknesses.shape[0]

    d_rhos = get_d_rhos(slds)

    zs = torch.cumsum(torch.cat([torch.zeros(sample_num, 1).to(thicknesses), thicknesses], dim=-1), dim=-1)

    if z_axis is None:
        z_axis = torch.linspace(- zs.max() * 0.1, zs.max() * 1.1, num, device=thicknesses.device)[None]
    elif len(z_axis.shape) == 1:
        z_axis = z_axis[None]

    sigmas = roughnesses * sqrt(2)

    profile = get_erf(z_axis[:, None], zs[..., None], sigmas[..., None], d_rhos[..., None]).sum(1)

    d_profile = get_gauss(z_axis[:, None], zs[..., None], sigmas[..., None], d_rhos[..., None]).sum(1)

    z_axis = z_axis[0]

    return z_axis, profile, d_profile


def get_d_rhos(slds: Tensor) -> Tensor:
    d_rhos = torch.cat([slds[:, 0][:, None], torch.diff(slds, dim=-1)], -1)
    return d_rhos


def get_slds_from_d_rhos(d_rhos: Tensor) -> Tensor:
    slds = torch.cumsum(d_rhos, dim=-1)
    return slds


def get_erf(z, z0, sigma, amp):
    return (torch.erf((z - z0) / sigma) + 1) * amp / 2


def get_gauss(z, z0, sigma, amp):
    return amp / (sigma * sqrt(2 * pi)) * torch.exp(- (z - z0) ** 2 / 2 / sigma ** 2)


def get_param_labels(
        num_layers: int, *,
        thickness_name: str = 'Thickness',
        roughness_name: str = 'Roughness',
        sld_name: str = 'SLD',
        substrate_name: str = 'sub',
) -> List[str]:
    thickness_labels = [f'{thickness_name} L{num_layers - i}' for i in range(num_layers)]
    roughness_labels = [f'{roughness_name} L{num_layers - i}' for i in range(num_layers)] + [f'{roughness_name} {substrate_name}']
    sld_labels = [f'{sld_name} L{num_layers - i}' for i in range(num_layers)] + [f'{sld_name} {substrate_name}']
    return thickness_labels + roughness_labels + sld_labels

def get_param_labels_absorption_model(
        num_layers: int, *,
        thickness_name: str = 'Thickness',
        roughness_name: str = 'Roughness',
        real_sld_name: str = 'SLD real',
        imag_sld_name: str = 'SLD imag',
        substrate_name: str = 'sub',
) -> List[str]:
    thickness_labels = [f'{thickness_name} L{num_layers - i}' for i in range(num_layers)]
    roughness_labels = [f'{roughness_name} L{num_layers - i}' for i in range(num_layers)] + [f'{roughness_name} {substrate_name}']
    real_sld_labels = [f'{real_sld_name} L{num_layers - i}' for i in range(num_layers)] + [f'{real_sld_name} {substrate_name}']
    imag_sld_labels = [f'{imag_sld_name} L{num_layers - i}' for i in range(num_layers)] + [f'{imag_sld_name} {substrate_name}']
    return thickness_labels + roughness_labels + real_sld_labels + imag_sld_labels
