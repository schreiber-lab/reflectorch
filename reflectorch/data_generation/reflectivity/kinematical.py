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

        rf = _get_resnel_reflectivity(q, substrate_sld[:, None])

        r = torch.clamp_max_(r * rf / substrate_sld.real ** 2, 1.)

    if log:
        r = torch.log10(r)

    return r


def _get_resnel_reflectivity(q, substrate_slds):
    _RE_CONST = 0.28174103675406496

    q_c = torch.sqrt(substrate_slds + 0j) / _RE_CONST * 2
    q_prime = torch.sqrt(q ** 2 - q_c ** 2 + 0j)
    r_f = ((q - q_prime) / (q + q_prime)).abs().float() ** 2

    return r_f.squeeze(-1)

