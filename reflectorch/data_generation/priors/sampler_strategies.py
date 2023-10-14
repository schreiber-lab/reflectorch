import torch
from torch import Tensor

from reflectorch.data_generation.utils import (
    uniform_sampler,
    logdist_sampler,
)

from reflectorch.data_generation.priors.utils import get_max_allowed_roughness, get_max_allowed_sld_imag


class SamplerStrategy(object):
    def sample(self, batch_size: int,
               total_min_bounds: Tensor,
               total_max_bounds: Tensor,
               total_min_delta: Tensor,
               total_max_delta: Tensor,
               ):
        raise NotImplementedError


class BasicSamplerStrategy(SamplerStrategy):
    def __init__(self, logdist: bool = False):
        if logdist:
            self.widths_sampler_func = logdist_sampler
        else:
            self.widths_sampler_func = uniform_sampler

    def sample(self, batch_size: int,
               total_min_bounds: Tensor,
               total_max_bounds: Tensor,
               total_min_delta: Tensor,
               total_max_delta: Tensor,
               ):
        return basic_sampler(
            batch_size,
            total_min_bounds,
            total_max_bounds,
            total_min_delta,
            total_max_delta,
            self.widths_sampler_func,
        )


class ConstrainedRoughnessSamplerStrategy(BasicSamplerStrategy):
    def __init__(self,
                 thickness_mask: Tensor,
                 roughness_mask: Tensor,
                 logdist: bool = False,
                 max_thickness_share: float = 0.5,
                 ):
        super().__init__(logdist=logdist)
        self.thickness_mask = thickness_mask
        self.roughness_mask = roughness_mask
        self.max_thickness_share = max_thickness_share

    def sample(self, batch_size: int,
               total_min_bounds: Tensor,
               total_max_bounds: Tensor,
               total_min_delta: Tensor,
               total_max_delta: Tensor,
               ):
        device = total_min_bounds.device
        return constrained_roughness_sampler(
            batch_size,
            total_min_bounds,
            total_max_bounds,
            total_min_delta,
            total_max_delta,
            thickness_mask=self.thickness_mask.to(device),
            roughness_mask=self.roughness_mask.to(device),
            widths_sampler_func=self.widths_sampler_func,
            coef_roughness=self.max_thickness_share,
        )
    
class ConstrainedSLDSamplerStrategy(BasicSamplerStrategy):
    def __init__(self,
                 thickness_mask: Tensor,
                 roughness_mask: Tensor,
                 sld_real_mask: Tensor,
                 sld_imag_mask: Tensor,
                 logdist: bool = False,
                 max_thickness_share: float = 0.5,
                 ):
        super().__init__(logdist=logdist)
        self.thickness_mask = thickness_mask
        self.roughness_mask = roughness_mask
        self.sld_real_mask = sld_real_mask
        self.sld_imag_mask = sld_imag_mask
        self.max_thickness_share = max_thickness_share

    def sample(self, batch_size: int,
               total_min_bounds: Tensor,
               total_max_bounds: Tensor,
               total_min_delta: Tensor,
               total_max_delta: Tensor,
               ):
        device = total_min_bounds.device
        return constrained_sld_sampler(
            batch_size,
            total_min_bounds,
            total_max_bounds,
            total_min_delta,
            total_max_delta,
            thickness_mask=self.thickness_mask.to(device),
            roughness_mask=self.roughness_mask.to(device),
            sld_real_mask=self.sld_real_mask.to(device),
            sld_imag_mask=self.sld_imag_mask.to(device),
            widths_sampler_func=self.widths_sampler_func,
            coef_roughness=self.max_thickness_share,
        )


def basic_sampler(
        batch_size: int,
        total_min_bounds: Tensor,
        total_max_bounds: Tensor,
        total_min_delta: Tensor,
        total_max_delta: Tensor,
        widths_sampler_func,
):

    delta_vector = total_max_bounds - total_min_bounds

    prior_widths = widths_sampler_func(
        total_min_delta, total_max_delta,
        batch_size, delta_vector.shape[1],
        device=total_min_bounds.device, dtype=total_min_bounds.dtype
    )

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
        total_min_delta: Tensor,
        total_max_delta: Tensor,
        thickness_mask: Tensor,
        roughness_mask: Tensor,
        widths_sampler_func,
        coef_roughness: float = 0.5,
):
    params, min_bounds, max_bounds = basic_sampler(
        batch_size, total_min_bounds, total_max_bounds, total_min_delta, total_max_delta,
        widths_sampler_func=widths_sampler_func,
    )

    max_roughness = torch.minimum(
        get_max_allowed_roughness(thicknesses=params[..., thickness_mask], coef_roughness=coef_roughness),
        total_max_bounds[..., roughness_mask]
    )
    min_roughness = total_min_bounds[..., roughness_mask]

    assert torch.all(min_roughness <= max_roughness)

    min_roughness_delta = total_min_delta[..., roughness_mask]
    max_roughness_delta = torch.minimum(total_max_delta[..., roughness_mask], max_roughness - min_roughness)

    rougnhesses, min_r_bounds, max_r_bounds = basic_sampler(
        batch_size, min_roughness, max_roughness,
        min_roughness_delta, max_roughness_delta,
        widths_sampler_func=widths_sampler_func
    )

    min_bounds[..., roughness_mask], max_bounds[..., roughness_mask] = min_r_bounds, max_r_bounds
    params[..., roughness_mask] = rougnhesses

    return params, min_bounds, max_bounds



def constrained_sld_sampler(
        batch_size: int,
        total_min_bounds: Tensor,
        total_max_bounds: Tensor,
        total_min_delta: Tensor,
        total_max_delta: Tensor,
        thickness_mask: Tensor,
        roughness_mask: Tensor,
        sld_real_mask: Tensor,
        sld_imag_mask: Tensor,
        widths_sampler_func,    
        coef_roughness: float = 0.5,
):
    params, min_bounds, max_bounds = basic_sampler(
        batch_size, total_min_bounds, total_max_bounds, total_min_delta, total_max_delta,
        widths_sampler_func=widths_sampler_func,
    )
    
    max_roughness = torch.minimum(
        get_max_allowed_roughness(thicknesses=params[..., thickness_mask], coef_roughness=coef_roughness),
        total_max_bounds[..., roughness_mask]
    )
    min_roughness = total_min_bounds[..., roughness_mask]


    max_sld_imag = torch.minimum(
        get_max_allowed_sld_imag(sld_real=params[..., sld_real_mask]),
        total_max_bounds[...,sld_imag_mask]
    )
    min_sld_imag = total_min_bounds[..., sld_imag_mask]


    assert torch.all(min_roughness <= max_roughness)

    assert torch.all(min_sld_imag <= max_sld_imag)


    min_roughness_delta = total_min_delta[..., roughness_mask]
    max_roughness_delta = torch.minimum(total_max_delta[..., roughness_mask], max_roughness - min_roughness)

    min_sld_imag_delta = total_min_delta[..., sld_imag_mask]
    max_sld_imag_delta = torch.minimum(total_max_delta[...,sld_imag_mask], max_sld_imag - min_sld_imag)


    roughnesses, min_r_bounds, max_r_bounds = basic_sampler(
        batch_size, min_roughness, max_roughness,
        min_roughness_delta, max_roughness_delta,
        widths_sampler_func=widths_sampler_func
    )

    sld_imag, min_sld_bounds, max_sld_bounds = basic_sampler(
        batch_size, min_sld_imag, max_sld_imag,
        min_sld_imag_delta, max_sld_imag_delta,
        widths_sampler_func=widths_sampler_func
    )


    min_bounds[..., roughness_mask], max_bounds[..., roughness_mask] = min_r_bounds, max_r_bounds
    params[..., roughness_mask] = roughnesses

    min_bounds[...,sld_imag_mask],max_bounds[..., sld_imag_mask] = min_sld_bounds, max_sld_bounds
    params[...,sld_imag_mask] = sld_imag
    
    
    return params, min_bounds, max_bounds
