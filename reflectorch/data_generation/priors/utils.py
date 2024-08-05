from typing import Tuple

import torch
from torch import Tensor

from reflectorch.data_generation.utils import (
    get_d_rhos,
    uniform_sampler,
)


def get_max_allowed_roughness(thicknesses: Tensor, mask: Tensor = None, coef: float = 0.5):
    """gets the maximum allowed interlayer roughnesses such that they do not exceed a fraction of the thickness of either layers meeting at that interface"""
    batch_size, layers_num = thicknesses.shape
    max_roughness = torch.ones(
        batch_size, layers_num + 1, device=thicknesses.device, dtype=thicknesses.dtype
    ) * float('inf')

    boundary = thicknesses * coef
    if mask is not None:
        boundary[get_thickness_mask_from_sld_mask(mask)] = float('inf')

    max_roughness[:, :-1] = boundary
    max_roughness[:, 1:] = torch.minimum(max_roughness[:, 1:], boundary)
    return max_roughness


def get_allowed_contrast_indices(slds: Tensor, min_contrast: float, mask: Tensor = None) -> Tensor:
    d_rhos = get_d_rhos(slds)
    indices = d_rhos.abs() >= min_contrast
    if mask is not None:
        indices = indices | mask
    indices = torch.all(indices, -1)
    return indices


def params_within_bounds(params_t: Tensor, min_t: Tensor, max_t: Tensor, mask: Tensor = None) -> Tensor:
    indices = (params_t >= min_t[None]) & (params_t <= max_t[None])
    if mask is not None:
        indices = indices | mask
    indices = torch.all(indices, -1)
    return indices


def get_allowed_roughness_indices(thicknesses: Tensor, roughnesses: Tensor, mask: Tensor = None) -> Tensor:
    max_roughness = get_max_allowed_roughness(thicknesses, mask)
    indices = roughnesses <= max_roughness
    if mask is not None:
        indices = indices | mask
    indices = torch.all(indices, -1)
    return indices


def get_thickness_mask_from_sld_mask(mask: Tensor):
    return mask[:, :-1]


def generate_roughnesses(thicknesses: Tensor, roughness_range: Tuple[float, float], mask: Tensor = None):
    batch_size, layers_num = thicknesses.shape
    max_roughness = get_max_allowed_roughness(thicknesses, mask)
    max_roughness = torch.clamp_(max_roughness, max=roughness_range[1])

    roughnesses = uniform_sampler(
        roughness_range[0], max_roughness, batch_size, layers_num + 1,
        device=thicknesses.device, dtype=thicknesses.dtype
    )

    if mask is not None:
        roughnesses[mask] = 0.

    return roughnesses


def generate_thicknesses(
        thickness_range: Tuple[float, float],
        batch_size: int,
        layers_num: int,
        device: torch.device,
        dtype: torch.dtype,
        mask: Tensor = None
):
    thicknesses = uniform_sampler(
        *thickness_range, batch_size, layers_num, device=device, dtype=dtype
    )
    if mask is not None:
        thicknesses[get_thickness_mask_from_sld_mask(mask)] = 0.
    return thicknesses


def generate_slds_with_min_contrast(
        sld_range: Tuple[float, float],
        batch_size: int,
        layers_num: int,
        min_contrast: float,
        device: torch.device,
        dtype: torch.dtype,
        mask: Tensor = None,
        *,
        _depth: int = 0
):
    # rejection sampling
    slds = uniform_sampler(
        *sld_range, batch_size, layers_num + 1, device=device, dtype=dtype
    )

    if mask is not None:
        slds[mask] = 0.

    rejected_indices = ~get_allowed_contrast_indices(slds, min_contrast, mask)
    rejected_num = rejected_indices.sum(0).item()

    if rejected_num:
        if mask is not None:
            mask = mask[rejected_indices]
        slds[rejected_indices] = generate_slds_with_min_contrast(
            sld_range, rejected_num, layers_num, min_contrast, device, dtype, mask, _depth=_depth+1
        )
    return slds
