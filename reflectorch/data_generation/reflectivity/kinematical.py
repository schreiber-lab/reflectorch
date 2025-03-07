# -*- coding: utf-8 -*-

import torch
from torch import Tensor


def kinematical_approximation(
        q: Tensor,
        thickness: Tensor,
        roughness: Tensor,
        sld: Tensor,
        *,
        apply_fresnel: bool = True,
        log: bool = False,
):
    """Simulates reflectivity curves for SLD profiles with box model parameterization using the kinematical approximation

    Args:
        q (Tensor): tensor of momentum transfer (q) values with shape [batch_size, n_points] or [n_points]
        thickness (Tensor): tensor containing the layer thicknesses (ordered from top to bottom) with shape [batch_size, n_layers]
        roughness (Tensor): tensor containing the interlayer roughnesses (ordered from top to bottom) with shape [batch_size, n_layers + 1]
        sld (Tensor): tensor containing the layer SLDs (real or complex; ordered from top to bottom) with shape [batch_size, n_layers + 1]. 
                    It includes the substrate but excludes the ambient medium which is assumed to have an SLD of 0.
        apply_fresnel (bool, optional): whether to use the Fresnel coefficient in the computation. Defaults to ``True``.
        log (bool, optional): if True the base 10 logarithm of the reflectivity curves is returned. Defaults to ``False``.

    Returns:
        Tensor: tensor containing the simulated reflectivity curves with shape [batch_size, n_points]
    """
    c_dtype = torch.complex128 if q.dtype is torch.float64 else torch.complex64

    batch_size, num_layers = thickness.shape

    q = q.to(c_dtype)

    if q.dim() == 1:
        q.unsqueeze_(0)

    if q.dim() == 2:
        q.unsqueeze_(-1)

    sld = sld * 1e-6 + 1e-30j

    drho = torch.cat([sld[..., 0][..., None], sld[..., 1:] - sld[..., :-1]], -1)[:, None]
    thickness = torch.cumsum(torch.cat([torch.zeros(batch_size, 1).to(thickness), thickness], -1), -1)[:, None]
    roughness = roughness[:, None]

    r = (drho * torch.exp(- (roughness * q) ** 2 / 2 + 1j * (q * thickness))).sum(-1).abs().float() ** 2

    if apply_fresnel:

        substrate_sld = sld[:, -1:]

        rf = _get_fresnel_reflectivity(q, substrate_sld[:, None])

        r = torch.clamp_max_(r * rf / substrate_sld.real ** 2, 1.)

    if log:
        r = torch.log10(r)

    return r


def _get_fresnel_reflectivity(q, substrate_slds):
    _RE_CONST = 0.28174103675406496 # 2/sqrt(16*pi)

    q_c = torch.sqrt(substrate_slds + 0j) / _RE_CONST * 2
    q_prime = torch.sqrt(q ** 2 - q_c ** 2 + 0j)
    r_f = ((q - q_prime) / (q + q_prime)).abs().float() ** 2

    return r_f.squeeze(-1)