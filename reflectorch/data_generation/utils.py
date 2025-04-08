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


def get_density_profiles_sld(
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

def get_density_profiles(
    thicknesses: torch.Tensor,
    roughnesses: torch.Tensor,
    slds: torch.Tensor,
    ambient_sld: torch.Tensor = None,
    z_axis: torch.Tensor = None,
    num: int = 1000,
    padding_left: float = 0.2,
    padding_right: float = 1.1,
):
    """
    Args:
        thicknesses (Tensor): finite layer thicknesses.
        roughnesses (Tensor): interface roughnesses for all transitions (ambient→layer1 ... layerN→substrate).
        slds (Tensor): SLDs for the finite layers + substrate.
        ambient_sld (Tensor, optional): SLD for the top ambient. Defaults to 0.0 if None.
        z_axis (Tensor, optional): a custom depth axis. If None, a linear axis is generated.
        num (int): number of points in the generated z-axis (if z_axis is None).
        padding_left (float): factor to extend the negative (above the surface) portion of z-axis.
        padding_right (float): factor to extend the positive (into the sample) portion of z-axis.

    Returns:
        (z_axis, profile, d_profile)
          z_axis:    1D Tensor of shape (num, ) with the depth coordinates.
          profile:   2D Tensor of shape (batch_size, num) giving the SLD at each depth.
          d_profile: 2D Tensor of shape (batch_size, num) giving d(SLD)/dz at each depth.
    """

    bs, n = thicknesses.shape
    assert roughnesses.shape == (bs, n + 1),  (
        f"Roughnesses must be (batch_size, num_layers+1). Found {roughnesses.shape} instead."
    )
    assert slds.shape == (bs, n + 1), (
        f"SLDs must be (batch_size, num_layers+1). Found {slds.shape} instead."
    )
    assert torch.all(thicknesses >= 0), "Negative thickness encountered."
    assert torch.all(roughnesses >= 0), "Negative roughness encountered."

    if ambient_sld is None:
        ambient_sld = torch.zeros((bs, 1), device=thicknesses.device)
    else:
        if ambient_sld.ndim == 1:
            ambient_sld = ambient_sld.unsqueeze(-1)

    slds_all = torch.cat([ambient_sld, slds], dim=-1)  # new dimension: n+2
    d_rhos = torch.diff(slds_all, dim=-1)  # (bs, n+1)

    interfaces = torch.cat([
        torch.zeros((bs, 1), device=thicknesses.device),  # z=0 for ambient→layer1
        thicknesses
    ], dim=-1).cumsum(dim=-1)  # now shape => (bs, n+1)

    total_thickness = interfaces[..., -1].max()
    if z_axis is None:
        z_axis = torch.linspace(
            -padding_left * total_thickness,
            padding_right * total_thickness,
            num,
            device=thicknesses.device
        )  # shape => (num,)
    if z_axis.ndim == 1:
        z_axis = z_axis.unsqueeze(0)  # shape => (1, num)

    z_b = z_axis.repeat(bs, 1).unsqueeze(1)           # (bs, 1, num)
    interfaces_b = interfaces.unsqueeze(-1)           # (bs, n+1, 1)
    sigmas_b = (roughnesses * sqrt(2)).unsqueeze(-1)  # (bs, n+1, 1)
    d_rhos_b = d_rhos.unsqueeze(-1)                   # (bs, n+1, 1)

    profile = get_erf(z_b, interfaces_b, sigmas_b, d_rhos_b).sum(dim=1)    # (bs, num)
    if ambient_sld is not None:
        profile = profile + ambient_sld

    d_profile = get_gauss(z_b, interfaces_b, sigmas_b, d_rhos_b).sum(dim=1) # (bs, num)

    return z_axis.squeeze(0), profile, d_profile

def get_param_labels(
        num_layers: int, *,
        thickness_name: str = 'Thickness',
        roughness_name: str = 'Roughness',
        sld_name: str = 'SLD',
        imag_sld_name: str = 'SLD imag',
        substrate_name: str = 'sub',
        parameterization_type: str = 'standard',
        number_top_to_bottom: bool = False,
) -> List[str]:
    def pos(i):
        return i + 1 if number_top_to_bottom else num_layers - i
        
    thickness_labels = [f'{thickness_name} L{pos(i)}' for i in range(num_layers)]
    roughness_labels = [f'{roughness_name} L{pos(i)}' for i in range(num_layers)] + [f'{roughness_name} {substrate_name}']
    sld_labels = [f'{sld_name} L{pos(i)}' for i in range(num_layers)] + [f'{sld_name} {substrate_name}']

    all_labels = thickness_labels + roughness_labels + sld_labels

    if parameterization_type == 'absorption':
        imag_sld_labels = [f'{imag_sld_name} L{pos(i)}' for i in range(num_layers)] + [f'{imag_sld_name} {substrate_name}']
        all_labels = all_labels + imag_sld_labels
        
    return all_labels
