# -*- coding: utf-8 -*-

from math import pi

import torch
from torch import Tensor


def abeles_memory_eff(
        q: Tensor,
        thickness: Tensor,
        roughness: Tensor,
        sld: Tensor,
):
    """Simulates reflectivity curves for SLD profiles with box model parameterization using a memory-efficient implementation the Abeles matrix method. 
    It is computationally slower compared to the implementation in the 'abeles' function.

    Args:
        q (Tensor): tensor of momentum transfer (q) values with shape [batch_size, n_points] or [n_points]
        thickness (Tensor): tensor containing the layer thicknesses (ordered from top to bottom) with shape [batch_size, n_layers]
        roughness (Tensor): tensor containing the interlayer roughnesses (ordered from top to bottom) with shape [batch_size, n_layers + 1]
        sld (Tensor): tensor containing the layer SLDs (real or complex; ordered from top to bottom) with shape [batch_size, n_layers + 1]. 
                    It includes the substrate but excludes the ambient medium which is assumed to have an SLD of 0.

    Returns:
        Tensor: tensor containing the simulated reflectivity curves with shape [batch_size, n_points]
    """
    c_dtype = torch.complex128 if q.dtype is torch.float64 else torch.complex64

    batch_size, num_layers = thickness.shape

    sld = sld * 1e-6 + 1e-30j

    num_interfaces = num_layers + 1

    k_z0 = (q / 2).to(c_dtype)

    if len(k_z0.shape) == 1:
        k_z0.unsqueeze_(0)

    thickness_prev_layer = 1.  # ambient

    for interface_num in range(num_interfaces):

        prev_layer_idx = interface_num - 1
        next_layer_idx = interface_num

        if interface_num == 0:
            k_z_previous_layer = _get_relative_k_z(k_z0, torch.zeros(batch_size, 1).to(sld))
        else:
            thickness_prev_layer = thickness[:, prev_layer_idx].unsqueeze(1)
            k_z_previous_layer = _get_relative_k_z(k_z0, sld[:, prev_layer_idx].unsqueeze(1))

        k_z_next_layer = _get_relative_k_z(k_z0, sld[:, next_layer_idx].unsqueeze(1))  # (batch_num, q_num)

        reflection_matrix = _make_reflection_matrix(
            k_z_previous_layer, k_z_next_layer, roughness[:, interface_num].unsqueeze(1)
        )

        if interface_num == 0:
            total_reflectivity_matrix = reflection_matrix
        else:
            translation_matrix = _make_translation_matrix(k_z_previous_layer, thickness_prev_layer)

            total_reflectivity_matrix = torch.einsum(
                'bnmr, bmlr, bljr -> bnjr', total_reflectivity_matrix, translation_matrix, reflection_matrix
            )

    r = total_reflectivity_matrix[:, 0, 1] / total_reflectivity_matrix[:, 1, 1]

    reflectivity = torch.clamp_max_(torch.abs(r) ** 2, 1.).flatten(1)

    return reflectivity


def _get_relative_k_z(k_z0, scattering_length_density):
    return torch.sqrt(k_z0 ** 2 - 4 * pi * scattering_length_density)


def _make_reflection_matrix(k_z_previous_layer, k_z_next_layer, interface_roughness):
    p = _safe_div((k_z_previous_layer + k_z_next_layer), (2 * k_z_previous_layer)) * \
        torch.exp(-(k_z_previous_layer - k_z_next_layer) ** 2 * 0.5 * interface_roughness ** 2)

    m = _safe_div((k_z_previous_layer - k_z_next_layer), (2 * k_z_previous_layer)) * \
        torch.exp(-(k_z_previous_layer + k_z_next_layer) ** 2 * 0.5 * interface_roughness ** 2)

    return _stack_mtx(p, m, m, p)


def _stack_mtx(a11, a12, a21, a22):
    return torch.stack([
        torch.stack([a11, a12], dim=1),
        torch.stack([a21, a22], dim=1),
    ], dim=1)


def _make_translation_matrix(k_z, thickness):
    return _stack_mtx(
        torch.exp(-1j * k_z * thickness), torch.zeros_like(k_z),
        torch.zeros_like(k_z), torch.exp(1j * k_z * thickness)
    )


def _safe_div(numerator, denominator):
    return torch.where(denominator == 0, numerator, torch.divide(numerator, denominator))
