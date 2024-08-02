# -*- coding: utf-8 -*-
import math

import torch
from torch import Tensor


def abeles(
        q: Tensor,
        thickness: Tensor,
        roughness: Tensor,
        sld: Tensor,
):
    """Simulates reflectivity curves for SLD profiles with box model parameterization using the Abeles matrix method

    Args:
        q (Tensor): tensor of momentum transfer (q) values with shape [batch_size, n_points] or [n_points]
        thickness (Tensor): tensor containing the layer thicknesses (ordered from top to bottom) with shape [batch_size, n_layers]
        roughness (Tensor): tensor containing the interlayer roughnesses (ordered from top to bottom) with shape [batch_size, n_layers + 1]
        sld (Tensor): tensors containing the layer SLDs (real or complex; ordered from top to bottom) with shape [batch_size, n_layers + 1]. 
                    It includes the substrate but excludes the ambient medium which is assumed to have an SLD of 0.

    Returns:
        Tensor: tensor containing the simulated reflectivity curves with shape [batch_size, n_points]
    """
    c_dtype = torch.complex128 if q.dtype is torch.float64 else torch.complex64

    batch_size, num_layers = thickness.shape

    sld = torch.cat([torch.zeros(batch_size, 1).to(sld), sld], -1)[:, None]
    thickness = torch.cat([torch.zeros(batch_size, 1).to(thickness), thickness], -1)[:, None]
    roughness = roughness[:, None] ** 2

    sld = sld * 1e-6 + 1e-30j

    k_z0 = (q / 2).to(c_dtype)

    if k_z0.dim() == 1:
        k_z0.unsqueeze_(0)

    if k_z0.dim() == 2:
        k_z0.unsqueeze_(-1)

    k_n = torch.sqrt(k_z0 ** 2 - 4 * math.pi * sld)

    # k_n.shape - (batch, q, layers)

    k_n, k_np1 = k_n[..., :-1], k_n[..., 1:]

    beta = 1j * thickness * k_n

    exp_beta = torch.exp(beta)
    exp_m_beta = torch.exp(-beta)

    rn = (k_n - k_np1) / (k_n + k_np1) * torch.exp(- 2 * k_n * k_np1 * roughness)

    c_matrices = torch.stack([
        torch.stack([exp_beta, rn * exp_m_beta], -1),
        torch.stack([rn * exp_beta, exp_m_beta], -1),
    ], -1)

    c_matrices = [c.squeeze(-3) for c in c_matrices.split(1, -3)]

    m, c_matrices = c_matrices[0], c_matrices[1:]

    for c in c_matrices:
        m = m @ c

    r = (m[..., 1, 0] / m[..., 0, 0]).abs() ** 2
    r = torch.clamp_max_(r, 1.)

    return r


# @torch.jit.script # commented so far due to complex numbers issue
def abeles_compiled(
        q: Tensor,
        thickness: Tensor,
        roughness: Tensor,
        sld: Tensor,
):
    return abeles(q, thickness, roughness, sld)
