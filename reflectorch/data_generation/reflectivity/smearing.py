# -*- coding: utf-8 -*-
from math import pi, sqrt, log

import torch
from torch import Tensor

from reflectorch.data_generation.reflectivity.abeles import abeles
from torch.nn.functional import conv1d, pad


def abeles_constant_smearing(
        q: Tensor,
        thickness: Tensor,
        roughness: Tensor,
        sld: Tensor,
        dq: Tensor = None,
        gauss_num: int = 31,
        constant_dq: bool = False,
        abeles_func=None,
        **abeles_kwargs
):
    abeles_func = abeles_func or abeles

    if dq.dtype != thickness.dtype:
        q = q.to(thickness)
    
    if dq.dtype != thickness.dtype:
        dq = dq.to(thickness)

    if q.shape[0] == 1:
        q = q.repeat(thickness.shape[0], 1)

    q_lin = _get_q_axes(q, dq, gauss_num, constant_dq=constant_dq)
    kernels = _get_t_gauss_kernels(dq, gauss_num)
    
    curves = abeles_func(q_lin, thickness, roughness, sld, **abeles_kwargs)

    padding = (kernels.shape[-1] - 1) // 2
    padded_curves = pad(curves, (padding, padding), 'reflect')

    smeared_curves = conv1d(
        padded_curves, kernels[:, None], groups=kernels.shape[0],
    )

    if q.shape[0] != smeared_curves.shape[0]:
        repeat_factor = smeared_curves.shape[0] // q.shape[0]
        q = q.repeat(repeat_factor, 1)
        q_lin = q_lin.repeat(repeat_factor, 1)
    
    smeared_curves = _batch_linear_interp1d(q_lin, smeared_curves, q)
        
    return smeared_curves


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


def _get_q_axes(q: Tensor, resolutions: Tensor, gaussnum: int = 51, constant_dq: bool = False):
    if constant_dq:
        return _get_q_axes_for_constant_dq(q, resolutions, gaussnum)
    else:
        return _get_q_axes_for_linear_dq(q, resolutions, gaussnum)


def _get_q_axes_for_linear_dq(q: Tensor, resolutions: Tensor, gaussnum: int = 51):
    gaussgpoint = (gaussnum - 1) / 2

    lowq = torch.clamp_min_(q.min(1).values, 1e-6)
    highq = q.max(1).values

    start = torch.log10(lowq)[:, None] - 6 * resolutions / _FWHM
    end = torch.log10(highq[:, None] * (1 + 6 * resolutions / _FWHM))

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
