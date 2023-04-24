# -*- coding: utf-8 -*-

import numpy as np


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

    k_n = np.sqrt(k_z0 ** 2 - 4 * np.pi * sld)

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
