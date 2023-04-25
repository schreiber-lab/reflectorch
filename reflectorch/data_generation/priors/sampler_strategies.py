import torch
from torch import Tensor

from reflectorch.data_generation.utils import (
    uniform_sampler,
    logdist_sampler,
)

from reflectorch.data_generation.priors.utils import get_max_allowed_roughness


class SamplerStrategy(object):
    def sample(self, batch_size: int,
               total_min_bounds: Tensor,
               total_max_bounds: Tensor):
        raise NotImplementedError


class BasicSamplerStrategy(SamplerStrategy):
    def __init__(self, logdist: bool = False, relative_min_bound_width: float = 1e-2):
        self.relative_min_bound_width = relative_min_bound_width

        if logdist:
            self.widths_sampler_func = logdist_sampler
        else:
            self.widths_sampler_func = uniform_sampler

    def sample(self, batch_size: int,
               total_min_bounds: Tensor,
               total_max_bounds: Tensor):
        return basic_sampler(
            batch_size,
            total_min_bounds,
            total_max_bounds,
            self.widths_sampler_func,
            self.relative_min_bound_width
        )


class ConstrainedRoughnessSamplerStrategy(BasicSamplerStrategy):
    def __init__(self,
                 thickness_mask: Tensor,
                 roughness_mask: Tensor,
                 logdist: bool = False,
                 relative_min_bound_width: float = 1e-2,
                 max_thickness_share: float = 0.5,
                 ):
        super().__init__(logdist=logdist, relative_min_bound_width=relative_min_bound_width)
        self.thickness_mask = thickness_mask
        self.roughness_mask = roughness_mask
        self.max_thickness_share = max_thickness_share

    def sample(self, batch_size: int,
               total_min_bounds: Tensor,
               total_max_bounds: Tensor):
        device = total_min_bounds.device
        return constrained_roughness_sampler(
            batch_size,
            total_min_bounds,
            total_max_bounds,
            thickness_mask=self.thickness_mask.to(device),
            roughness_mask=self.roughness_mask.to(device),
            widths_sampler_func=self.widths_sampler_func,
            relative_min_bound_width=self.relative_min_bound_width,
            coef=self.max_thickness_share,
        )


def basic_sampler(
        batch_size: int,
        total_min_bounds: Tensor,
        total_max_bounds: Tensor,
        widths_sampler_func,
        relative_min_bound_width: float):
    delta_vector = total_max_bounds - total_min_bounds

    prior_widths = widths_sampler_func(
        relative_min_bound_width, 1.,
        batch_size, delta_vector.shape[1],
        device=total_min_bounds.device, dtype=total_min_bounds.dtype
    ) * delta_vector

    prior_centers = uniform_sampler(
        total_min_bounds + prior_widths / 2, total_max_bounds - prior_widths / 2,
        *prior_widths.shape,
        device=total_min_bounds.device, dtype=total_min_bounds.dtype
    )

    min_bounds, max_bounds = prior_centers - prior_widths / 2, prior_centers + prior_widths / 2

    params = torch.rand(
        *min_bounds.shape,
        device=min_bounds.device,
        dtype=min_bounds.dtype
    ) * (max_bounds - min_bounds) + min_bounds

    return params, min_bounds, max_bounds


def constrained_roughness_sampler(
        batch_size: int,
        total_min_bounds: Tensor,
        total_max_bounds: Tensor,
        thickness_mask: Tensor,
        roughness_mask: Tensor,
        widths_sampler_func,
        relative_min_bound_width: float = 1e-2,
        coef: float = 0.5,
):
    params, min_bounds, max_bounds = basic_sampler(
        batch_size, total_min_bounds, total_max_bounds,
        widths_sampler_func=widths_sampler_func, relative_min_bound_width=relative_min_bound_width,
    )

    max_roughness = torch.minimum(
        get_max_allowed_roughness(thicknesses=params[..., thickness_mask], coef=coef),
        total_max_bounds[..., roughness_mask]
    )
    min_roughness = total_min_bounds[..., roughness_mask]

    assert torch.all(min_roughness <= max_roughness)

    rougnhesses, min_r_bounds, max_r_bounds = basic_sampler(
        batch_size, min_roughness, max_roughness, widths_sampler_func, relative_min_bound_width
    )

    min_bounds[..., roughness_mask], max_bounds[..., roughness_mask] = min_r_bounds, max_r_bounds
    params[..., roughness_mask] = rougnhesses

    return params, min_bounds, max_bounds
