import torch
from torch import Tensor
from functools import partial

from reflectorch.data_generation.utils import (
    uniform_sampler,
    logdist_sampler,
)

from reflectorch.data_generation.priors.utils import get_max_allowed_roughness


class SamplerStrategy(object):
    """Base class for sampler strategies"""
    def sample(self, batch_size: int,
               total_min_bounds: Tensor,
               total_max_bounds: Tensor,
               total_min_delta: Tensor,
               total_max_delta: Tensor,
               ):
        raise NotImplementedError


class BasicSamplerStrategy(SamplerStrategy):
    """Sampler strategy with no constraints on the values of the parameters.

    Args:
        logdist (bool, optional):
            Backward-compatible flag.
            If width_distribution is None:
              - False -> uniform width sampling
              - True  -> log-uniform width sampling
        width_distribution (str, optional):
            One of {"uniform", "loguniform", "exp_uniform_mixture"}.
            If provided, overrides logdist.
        exp_width_weight (float, optional):
            Mixture weight for the exp_uniform_mixture width sampler.
        exp_width_lam (float, optional):
            Scale parameter for the exp_uniform_mixture width sampler.
            Smaller values bias more strongly toward narrow widths.
    """

    def __init__(
        self,
        logdist: bool = False,
        width_distribution: str = None,
        exp_width_weight: float = 0.5,
        exp_width_lam: float = 0.05,
    ):
        self.logdist = logdist
        self.width_distribution = width_distribution
        self.exp_width_weight = exp_width_weight
        self.exp_width_lam = exp_width_lam

        if width_distribution is None:
            if logdist:
                self.widths_sampler_func = logdist_sampler
            else:
                self.widths_sampler_func = uniform_sampler

        elif width_distribution == "uniform":
            self.widths_sampler_func = uniform_sampler

        elif width_distribution == "loguniform":
            self.widths_sampler_func = logdist_sampler

        elif width_distribution == "exp_uniform_mixture":
            self.widths_sampler_func = partial(
                exp_uniform_mixture_width_sampler,
                weight=exp_width_weight,
                lam=exp_width_lam,
            )

        else:
            raise ValueError(
                f"Unknown width_distribution={width_distribution!r}. "
                "Expected one of {'uniform', 'loguniform', 'exp_uniform_mixture'}."
            )

    def sample(self, batch_size: int,
               total_min_bounds: Tensor,
               total_max_bounds: Tensor,
               total_min_delta: Tensor,
               total_max_delta: Tensor,
               ):
        """
        Args:
            batch_size (int): the batch size
            total_min_bounds (Tensor): mimimum values of the parameters
            total_max_bounds (Tensor): maximum values of the parameters
            total_min_delta (Tensor): minimum widths of the subprior intervals
            total_max_delta (Tensor): maximum widths of the subprior intervals

        Returns:
            tuple(Tensor): samples the values of the parameters and their prior bounds (params, min_bounds, max_bounds). The widths W of the subprior interval are sampled first, then the centers C of the subprior interval, such that the prior bounds are C-W/2 and C+W/2, then the parameters are sampled from [C-W/2, C+W/2] )
        """
        return basic_sampler(
            batch_size,
            total_min_bounds,
            total_max_bounds,
            total_min_delta,
            total_max_delta,
            self.widths_sampler_func,
        )


class ConstrainedRoughnessSamplerStrategy(BasicSamplerStrategy):
    """Sampler strategy where the roughnesses are constrained not to exceed a fraction of the two neighboring thicknesses 

    Args:
        thickness_mask (Tensor): indices in the tensors which correspond to thicknesses
        roughness_mask (Tensor): indices in the tensors which correspond to roughnesses
        logdist (bool, optional): if ``True`` the relative widths of the subprior intervals are sampled uniformly on a logarithmic scale instead of uniformly. Defaults to False.
        max_thickness_share (float, optional): fraction of the layer thickness that the roughness should not exceed. Defaults to 0.5.
    """
    def __init__(self,
                 thickness_mask: Tensor,
                 roughness_mask: Tensor,
                 logdist: bool = False,
                 max_thickness_share: float = 0.5,
                 max_total_thickness: float = None,
                 width_distribution: str = None,
                 exp_width_weight: float = 0.5,
                 exp_width_lam: float = 0.05,
                 ):
        super().__init__(
            logdist=logdist,
            width_distribution=width_distribution,
            exp_width_weight=exp_width_weight,
            exp_width_lam=exp_width_lam,
        )
        self.thickness_mask = thickness_mask
        self.roughness_mask = roughness_mask
        self.max_thickness_share = max_thickness_share
        self.max_total_thickness = max_total_thickness

    def sample(self, batch_size: int,
               total_min_bounds: Tensor,
               total_max_bounds: Tensor,
               total_min_delta: Tensor,
               total_max_delta: Tensor,
               ):
        """
        Args:
            batch_size (int): the batch size
            total_min_bounds (Tensor): mimimum values of the parameters
            total_max_bounds (Tensor): maximum values of the parameters
            total_min_delta (Tensor): minimum widths of the subprior intervals
            total_max_delta (Tensor): maximum widths of the subprior intervals

        Returns:
            tuple(Tensor): samples the values of the parameters and their prior bounds *(params, min_bounds, max_bounds)*, the roughnesses being constrained. The widths **W** of the subprior interval are sampled first, then the centers **C** of the subprior interval, such that the prior bounds are **C** - **W** / 2 and **C** + **W** / 2, then the parameters are sampled from [**C** - **W** / 2, **C** + **W** / 2] )
        """
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
            max_total_thickness=self.max_total_thickness,
        )

class ConstrainedRoughnessAndImgSldSamplerStrategy(BasicSamplerStrategy):
    """Sampler strategy where the roughnesses are constrained not to exceed a fraction of the two neighboring thicknesses, and the imaginary slds are constrained not to exceed a fraction of the real slds

    Args:
        thickness_mask (Tensor): indices in the tensors which correspond to thicknesses
        roughness_mask (Tensor): indices in the tensors which correspond to roughnesses
        sld_mask (Tensor): indices in the tensors which correspond to real slds
        isld_mask (Tensor): indices in the tensors which correspond to imaginary slds
        logdist (bool, optional): if ``True`` the relative widths of the subprior intervals are sampled uniformly on a logarithmic scale instead of uniformly. Defaults to False.
        max_thickness_share (float, optional): fraction of the layer thickness that the roughness should not exceed. Defaults to 0.5
        max_sld_share (float, optional): fraction of the real sld that the imaginary sld should not exceed. Defaults to 0.2.
    """
    def __init__(self,
                 thickness_mask: Tensor,
                 roughness_mask: Tensor,
                 sld_mask: Tensor,
                 isld_mask: Tensor,
                 logdist: bool = False,
                 max_thickness_share: float = 0.5,
                 max_sld_share: float = 0.2,
                 max_total_thickness: float = None,
                 width_distribution: str = None,
                 exp_width_weight: float = 0.5,
                 exp_width_lam: float = 0.05,
                 ):
        super().__init__(
            logdist=logdist,
            width_distribution=width_distribution,
            exp_width_weight=exp_width_weight,
            exp_width_lam=exp_width_lam,
        )
        self.thickness_mask = thickness_mask
        self.roughness_mask = roughness_mask
        self.sld_mask = sld_mask
        self.isld_mask = isld_mask
        self.max_thickness_share = max_thickness_share
        self.max_sld_share = max_sld_share
        self.max_total_thickness = max_total_thickness

    def sample(self, batch_size: int,
               total_min_bounds: Tensor,
               total_max_bounds: Tensor,
               total_min_delta: Tensor,
               total_max_delta: Tensor,
               ):
        """
        Args:
            batch_size (int): the batch size
            total_min_bounds (Tensor): mimimum values of the parameters
            total_max_bounds (Tensor): maximum values of the parameters
            total_min_delta (Tensor): minimum widths of the subprior intervals
            total_max_delta (Tensor): maximum widths of the subprior intervals

        Returns:
            tuple(Tensor): samples the values of the parameters and their prior bounds *(params, min_bounds, max_bounds)*, the roughnesses and imaginary slds being constrained. The widths **W** of the subprior interval are sampled first, then the centers **C** of the subprior interval, such that the prior bounds are **C** - **W** /2 and **C** + **W** / 2, then the parameters are sampled from [**C** - **W** / 2, **C** + **W** / 2] )
        """
        device = total_min_bounds.device
        return constrained_roughness_and_isld_sampler(
            batch_size,
            total_min_bounds,
            total_max_bounds,
            total_min_delta,
            total_max_delta,
            thickness_mask=self.thickness_mask.to(device),
            roughness_mask=self.roughness_mask.to(device),
            sld_mask=self.sld_mask.to(device),
            isld_mask=self.isld_mask.to(device),
            widths_sampler_func=self.widths_sampler_func,
            coef_roughness=self.max_thickness_share,
            coef_isld=self.max_sld_share,
            max_total_thickness=self.max_total_thickness,
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
        max_total_thickness: float = None,
):
    params, min_bounds, max_bounds = basic_sampler(
        batch_size, total_min_bounds, total_max_bounds, total_min_delta, total_max_delta,
        widths_sampler_func=widths_sampler_func,
    )

    if max_total_thickness is not None:
        total_thickness = max_bounds[:, thickness_mask].sum(-1)
        indices = total_thickness > max_total_thickness

        if indices.any():
            eps = 0.01
            rand_scale = torch.rand_like(total_thickness) * eps + 1 - eps
            scale_coef = max_total_thickness / total_thickness * rand_scale
            scale_coef[~indices] = 1.0
            min_bounds[:, thickness_mask] *= scale_coef[:, None]
            max_bounds[:, thickness_mask] *= scale_coef[:, None]
            params[:, thickness_mask] *= scale_coef[:, None]

            min_bounds[:, thickness_mask] = torch.clamp_min(
                min_bounds[:, thickness_mask],
                total_min_bounds[:, thickness_mask],
            )

            max_bounds[:, thickness_mask] = torch.clamp_min(
                max_bounds[:, thickness_mask],
                total_min_bounds[:, thickness_mask],
            )

            params[:, thickness_mask] = torch.clamp_min(
                params[:, thickness_mask],
                total_min_bounds[:, thickness_mask],
            )

    max_roughness = torch.minimum(
        get_max_allowed_roughness(thicknesses=params[..., thickness_mask], coef=coef_roughness),
        total_max_bounds[..., roughness_mask]
    )
    min_roughness = total_min_bounds[..., roughness_mask]

    assert torch.all(min_roughness <= max_roughness)

    min_roughness_delta = total_min_delta[..., roughness_mask]
    max_roughness_delta = torch.minimum(total_max_delta[..., roughness_mask], max_roughness - min_roughness)

    roughnesses, min_r_bounds, max_r_bounds = basic_sampler(
        batch_size, min_roughness, max_roughness,
        min_roughness_delta, max_roughness_delta,
        widths_sampler_func=widths_sampler_func
    )

    min_bounds[..., roughness_mask], max_bounds[..., roughness_mask] = min_r_bounds, max_r_bounds
    params[..., roughness_mask] = roughnesses

    return params, min_bounds, max_bounds

def constrained_roughness_and_isld_sampler(
        batch_size: int,
        total_min_bounds: Tensor,
        total_max_bounds: Tensor,
        total_min_delta: Tensor,
        total_max_delta: Tensor,
        thickness_mask: Tensor,
        roughness_mask: Tensor,
        sld_mask: Tensor,
        isld_mask: Tensor,
        widths_sampler_func,
        coef_roughness: float = 0.5,
        coef_isld: float = 0.2,
        max_total_thickness: float = None,
):
    params, min_bounds, max_bounds = basic_sampler(
        batch_size, total_min_bounds, total_max_bounds, total_min_delta, total_max_delta,
        widths_sampler_func=widths_sampler_func,
    )

    if max_total_thickness is not None:
        total_thickness = max_bounds[:, thickness_mask].sum(-1)
        indices = total_thickness > max_total_thickness

        if indices.any():
            eps = 0.01
            rand_scale = torch.rand_like(total_thickness) * eps + 1 - eps
            scale_coef = max_total_thickness / total_thickness * rand_scale
            scale_coef[~indices] = 1.0
            min_bounds[:, thickness_mask] *= scale_coef[:, None]
            max_bounds[:, thickness_mask] *= scale_coef[:, None]
            params[:, thickness_mask] *= scale_coef[:, None]

            min_bounds[:, thickness_mask] = torch.clamp_min(
                min_bounds[:, thickness_mask],
                total_min_bounds[:, thickness_mask],
            )

            max_bounds[:, thickness_mask] = torch.clamp_min(
                max_bounds[:, thickness_mask],
                total_min_bounds[:, thickness_mask],
            )

            params[:, thickness_mask] = torch.clamp_min(
                params[:, thickness_mask],
                total_min_bounds[:, thickness_mask],
            )

    max_roughness = torch.minimum(
        get_max_allowed_roughness(thicknesses=params[..., thickness_mask], coef=coef_roughness),
        total_max_bounds[..., roughness_mask]
    )
    min_roughness = total_min_bounds[..., roughness_mask]

    assert torch.all(min_roughness <= max_roughness)

    min_roughness_delta = total_min_delta[..., roughness_mask]
    max_roughness_delta = torch.minimum(total_max_delta[..., roughness_mask], max_roughness - min_roughness)

    roughnesses, min_r_bounds, max_r_bounds = basic_sampler(
        batch_size, min_roughness, max_roughness,
        min_roughness_delta, max_roughness_delta,
        widths_sampler_func=widths_sampler_func
    )

    min_bounds[..., roughness_mask], max_bounds[..., roughness_mask] = min_r_bounds, max_r_bounds
    params[..., roughness_mask] = roughnesses

    max_isld = torch.minimum(
        torch.abs(params[..., sld_mask]) * coef_isld,
        total_max_bounds[..., isld_mask]
    )
    min_isld = total_min_bounds[..., isld_mask]

    assert torch.all(min_isld <= max_isld)

    min_isld_delta = total_min_delta[..., isld_mask]
    max_isld_delta = torch.minimum(total_max_delta[..., isld_mask], max_isld - min_isld)

    islds, min_isld_bounds, max_isld_bounds = basic_sampler(
        batch_size, min_isld, max_isld,
        min_isld_delta, max_isld_delta,
        widths_sampler_func=widths_sampler_func
    )

    min_bounds[..., isld_mask], max_bounds[..., isld_mask] = min_isld_bounds, max_isld_bounds
    params[..., isld_mask] = islds


    return params, min_bounds, max_bounds


def truncated_exponential_width_sampler(
    min_widths: Tensor,
    max_widths: Tensor,
    batch_size: int,
    param_dim: int,
    lam: float = 0.05,
    device=None,
    dtype=None,
) -> Tensor:
    """
    Sample widths with density biased toward small values inside [min_widths, max_widths].

    Implemented form matches:
        y in [0, 1],  p(y) proportional to exp(-y / lam)
    then widths = min_widths + y * (max_widths - min_widths)

    lam acts like a scale:
      - small lam -> stronger concentration near min_widths
      - large lam -> closer to uniform over [min_widths, max_widths]
    """
    min_widths = min_widths.expand(batch_size, param_dim)
    max_widths = max_widths.expand(batch_size, param_dim)

    span = max_widths - min_widths
    fixed = span <= 0

    u = torch.rand(batch_size, param_dim, device=device, dtype=dtype)

    lam_t = torch.as_tensor(lam, device=device, dtype=dtype)

    # y in [0, 1] with density proportional to exp(-y / lam)
    y = -torch.log(1 - u * (1 - torch.exp(-1.0 / lam_t))) * lam_t

    widths = min_widths + y * span
    widths = torch.where(fixed, min_widths, widths)
    return widths


def exp_uniform_mixture_width_sampler(
    min_widths: Tensor,
    max_widths: Tensor,
    batch_size: int,
    param_dim: int,
    weight: float = 0.5,
    lam: float = 0.05,
    device=None,
    dtype=None,
) -> Tensor:
    """
    Mixture of:
      - uniform width sampling on [min_widths, max_widths]
      - truncated-exponential-like width sampling favoring narrow widths

    weight = probability of using the exponential-like component.
    """
    min_widths = min_widths.expand(batch_size, param_dim)
    max_widths = max_widths.expand(batch_size, param_dim)

    uniform_widths = torch.rand(
        batch_size, param_dim, device=device, dtype=dtype
    ) * (max_widths - min_widths) + min_widths

    exp_widths = truncated_exponential_width_sampler(
        min_widths=min_widths,
        max_widths=max_widths,
        batch_size=batch_size,
        param_dim=param_dim,
        lam=lam,
        device=device,
        dtype=dtype,
    )

    choose_exp = torch.rand(batch_size, param_dim, device=device, dtype=dtype) < weight
    return torch.where(choose_exp, exp_widths, uniform_widths)