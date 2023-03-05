# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.
import math
from math import pi, sqrt, log

import numpy as np

import torch
from torch import Tensor
from torch.nn.functional import conv1d, pad


def reflectivity(
        q: Tensor,
        thickness: Tensor,
        roughness: Tensor,
        sld: Tensor,
        dq: Tensor = None,
        gauss_num: int = 51,
        constant_dq: bool = True,
        log: bool = False,
):
    q = torch.atleast_2d(q)

    if dq is None:
        reflectivity_curves = abeles_fast(q, thickness, roughness, sld)
    else:
        reflectivity_curves = abeles_constant_smearing(
            q, thickness, roughness, sld,
            dq=dq, gauss_num=gauss_num, constant_dq=constant_dq,
        )

    if log:
        reflectivity_curves = torch.log10(reflectivity_curves)
    return reflectivity_curves


def abeles_constant_smearing(
        q: Tensor,
        thickness: Tensor,
        roughness: Tensor,
        sld: Tensor,
        dq: Tensor = None,
        gauss_num: int = 51,
        constant_dq: bool = True,
):
    q_lin = _get_q_axes(q, dq, gauss_num, constant_dq=constant_dq)
    kernels = _get_t_gauss_kernels(dq, gauss_num)

    curves = abeles(q_lin, thickness, roughness, sld)

    padding = (kernels.shape[-1] - 1) // 2
    smeared_curves = conv1d(
        pad(curves[None], (padding, padding), 'reflect'), kernels[:, None], groups=kernels.shape[0],
    )[0]

    if q.shape[0] != smeared_curves.shape[0]:
        q = q.expand(smeared_curves.shape[0], *q.shape[1:])

    smeared_curves = _batch_linear_interp1d(q_lin, smeared_curves, q)

    return smeared_curves


def abeles(
        q: Tensor,
        thickness: Tensor,
        roughness: Tensor,
        sld: Tensor,
):
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


# @torch.jit.script
def abeles_fast(
        q: Tensor,
        thickness: Tensor,
        roughness: Tensor,
        sld: Tensor,
):
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


def kinematical_approximation(
        q: Tensor,
        thickness: Tensor,
        roughness: Tensor,
        sld: Tensor,
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

    substrate_sld = sld[:, -1:]

    rf = _get_resnel_reflectivity(q, substrate_sld[:, None])

    r = torch.clamp_max_(r * rf / substrate_sld.real ** 2, 1.)

    return r


def _get_resnel_reflectivity(q, substrate_slds):
    _RE_CONST = 0.28174103675406496

    q_c = torch.sqrt(substrate_slds + 0j) / _RE_CONST * 2
    q_prime = torch.sqrt(q ** 2 - q_c ** 2 + 0j)
    r_f = ((q - q_prime) / (q + q_prime)).abs().float() ** 2

    return r_f.squeeze(-1)


def kinematical_approximation_np(
        q: np.ndarray,
        thickness: np.ndarray,
        roughness: np.ndarray,
        sld: np.ndarray,
):

    if q.ndim == thickness.ndim == roughness.ndim == sld.ndim == 1:
        zero_batch = True
    else:
        zero_batch = False

    thickness = np.atleast_2d(thickness)
    roughness = np.atleast_2d(roughness)
    sld = np.atleast_2d(sld) * 1e-6 + 1e-30j
    substrate_sld = sld[:, -1:]

    batch_size, num_layers = thickness.shape

    if q.ndim == 1:
        q = q[None]

    if q.ndim == 2:
        q = q[..., None]

    drho = np.concatenate([sld[..., 0][..., None], sld[..., 1:] - sld[..., :-1]], -1)[:, None]
    thickness = np.cumsum(np.concatenate([np.zeros((batch_size, 1)), thickness], -1), -1)[:, None]
    roughness = roughness[:, None]

    r = np.abs((drho * np.exp(- (roughness * q) ** 2 / 2 + 1j * (q * thickness))).sum(-1)).astype(float) ** 2

    rf = _get_resnel_reflectivity_np(q, substrate_sld[:, None])

    r = np.clip(r * rf / np.real(substrate_sld) ** 2, None, 1.)

    if zero_batch:
        r = r[0]

    return r


def _get_resnel_reflectivity_np(q, substrate_slds):
    _RE_CONST = 0.28174103675406496

    q_c = np.sqrt(substrate_slds + 0j) / _RE_CONST * 2
    q_prime = np.sqrt(q ** 2 - q_c ** 2 + 0j)
    r_f = np.abs((q - q_prime) / (q + q_prime)).astype(float) ** 2

    return r_f[..., 0]

def abeles_np(
        q: np.ndarray,
        thickness: np.ndarray,
        roughness: np.ndarray,
        sld: np.ndarray,
):
    c_dtype = np.complex128 if q.dtype is np.float64 else np.complex64

    if q.ndim == thickness.ndim == roughness.ndim == sld.ndim == 1:
        zero_batch = True
    else:
        zero_batch = False

    thickness = np.atleast_2d(thickness)
    roughness = np.atleast_2d(roughness)
    sld = np.atleast_2d(sld)

    batch_size, num_layers = thickness.shape

    sld = np.concatenate([np.zeros((batch_size, 1)).astype(sld.dtype), sld], -1)[:, None]
    thickness = np.concatenate([np.zeros((batch_size, 1)).astype(thickness.dtype), thickness], -1)[:, None]
    roughness = roughness[:, None] ** 2

    sld = sld * 1e-6 + 1e-30j

    k_z0 = (q / 2).astype(c_dtype)

    if len(k_z0.shape) == 1:
        k_z0 = k_z0[None]

    if len(k_z0.shape) == 2:
        k_z0 = k_z0[..., None]

    k_n = np.sqrt(k_z0 ** 2 - 4 * math.pi * sld)

    # k_n.shape - (batch, q, layers)

    k_n, k_np1 = k_n[..., :-1], k_n[..., 1:]

    beta = 1j * thickness * k_n

    exp_beta = np.exp(beta)
    exp_m_beta = np.exp(-beta)

    rn = (k_n - k_np1) / (k_n + k_np1) * np.exp(- 2 * k_n * k_np1 * roughness)

    c_matrices = np.stack([
        np.stack([exp_beta, rn * exp_m_beta], -1),
        np.stack([rn * exp_beta, exp_m_beta], -1),
    ], -1)

    c_matrices = np.moveaxis(c_matrices, -3, 0)

    m, c_matrices = c_matrices[0], c_matrices[1:]

    for c in c_matrices:
        m = m @ c

    r = np.abs(m[..., 1, 0] / m[..., 0, 0]) ** 2
    r = np.clip(r, None, 1.)

    if zero_batch:
        r = r[0]

    return r


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


_FWHM = 2 * sqrt(2 * log(2.0))
_2PI_SQRT = 1. / sqrt(2 * pi)


def _batch_linspace(start: Tensor, end: Tensor, num: int):
    return torch.linspace(0, 1, int(num), device=end.device, dtype=end.dtype)[None] * (end - start) + start


def _torch_gauss(x, s):
    return _2PI_SQRT / s * torch.exp(-0.5 * x ** 2 / s / s)


def _get_t_gauss_kernels(resolutions: Tensor, gaussnum: int = 51):
    gauss_x = _batch_linspace(-1.7 * resolutions, 1.7 * resolutions, gaussnum)
    gauss_y = _torch_gauss(gauss_x, resolutions / _FWHM) * (gauss_x[:, 1] - gauss_x[:, 0])[:, None]
    return gauss_y


def _get_q_axes(q: Tensor, resolutions: Tensor, gaussnum: int = 51, constant_dq: bool = True):
    if constant_dq:
        return _get_q_axes_for_constant_dq(q, resolutions, gaussnum)
    else:
        return _get_q_axes_for_linear_dq(q, resolutions, gaussnum)


def _get_q_axes_for_linear_dq(q: Tensor, resolutions: Tensor, gaussnum: int = 51):
    gaussgpoint = (gaussnum - 1) / 2

    lowq = torch.clamp_min_(q.min(1).values, 1e-6)
    highq = q.max(1).values

    start = torch.log10(lowq) - 6 * resolutions / _FWHM
    end = torch.log10(highq * (1 + 6 * resolutions / _FWHM))

    interpnums = torch.abs(
        (torch.abs(end - start)) / (1.7 * resolutions / _FWHM / gaussgpoint)
    ).round().to(int)

    q_lin = 10 ** _batch_linspace_with_padding(start, end, interpnums)

    return q_lin


def _get_q_axes_for_constant_dq(q: Tensor, resolutions: Tensor, gaussnum: int = 51) -> Tensor:
    gaussgpoint = (gaussnum - 1) / 2

    start = q.min(1).values[:, None] - resolutions * 1.7
    end = q.max(1).values[:, None] + resolutions * 1.7

    interpnums = torch.abs(
        (torch.abs(end - start)) / (1.7 * resolutions / gaussgpoint)
    ).round().to(int)

    q_lin = _batch_linspace_with_padding(start, end, interpnums)
    q_lin = torch.clamp_min_(q_lin, 1e-6)

    return q_lin


def _batch_linspace_with_padding(start: Tensor, end: Tensor, nums: Tensor) -> Tensor:
    max_num = nums.max().int().item()

    deltas = 1 / (nums - 1)

    x = torch.clamp_min_(_batch_linspace(deltas * (nums - max_num), torch.ones_like(deltas), max_num), 0)

    x = x * (end - start) + start

    return x


def _batch_linear_interp1d(x: Tensor, y: Tensor, x_new: Tensor) -> Tensor:
    eps = torch.finfo(y.dtype).eps

    ind = torch.searchsorted(x.contiguous(), x_new.contiguous())

    ind = torch.clamp_(ind - 1, 0, x.shape[-1] - 2)
    slopes = (y[..., 1:] - y[..., :-1]) / (eps + (x[..., 1:] - x[..., :-1]))
    ind_y = ind + torch.arange(slopes.shape[0], device=slopes.device)[:, None] * y.shape[1]
    ind_slopes = ind + torch.arange(slopes.shape[0], device=slopes.device)[:, None] * slopes.shape[1]

    y_new = y.flatten()[ind_y] + slopes.flatten()[ind_slopes] * (x_new - x.flatten()[ind_y])

    return y_new
