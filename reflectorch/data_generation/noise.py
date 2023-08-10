# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, Tuple
from math import log10

import torch
from torch import Tensor

from reflectorch.data_generation.process_data import ProcessData
from reflectorch.data_generation.utils import uniform_sampler

__all__ = [
    "QNoiseGenerator",
    "IntensityNoiseGenerator",
    "QNormalNoiseGenerator",
    "QSystematicShiftGenerator",
    "PoissonNoiseGenerator",
    "MultiplicativeLogNormalNoiseGenerator",
    "ScalingNoise",
    "ShiftNoise",
    "BasicExpIntensityNoise",
    "BasicQNoiseGenerator",
]


class QNoiseGenerator(ProcessData):
    def apply(self, qs: Tensor, context: dict = None):
        return qs


class QNormalNoiseGenerator(QNoiseGenerator):
    def __init__(self,
                 std: Union[float, Tuple[float, float]] = (0, 1e-3),
                 add_to_context: bool = False
                 ):
        self.std = std
        self.add_to_context = add_to_context

    def apply(self, qs: Tensor, context: dict = None):
        std = self.std

        if isinstance(std, (list, tuple)):
            std = uniform_sampler(*std, qs.shape[0], 1, device=qs.device, dtype=qs.dtype)
        else:
            std = torch.empty_like(qs).fill_(std)

        noise = torch.normal(mean=0., std=std)

        if self.add_to_context and context is not None:
            context['q_stds'] = std

        qs = torch.clamp_min_(qs + noise, 0.)

        return qs


class QSystematicShiftGenerator(QNoiseGenerator):
    def __init__(self, std: float, add_to_context: bool = True):
        self.std = std
        self.add_to_context = add_to_context

    def apply(self, qs: Tensor, context: dict = None):
        if len(qs.shape) == 1:
            shape = (1,)
        else:
            shape = (qs.shape[0], 1)

        shifts = torch.normal(
            mean=0., std=self.std * torch.ones(*shape, device=qs.device, dtype=qs.dtype)
        )

        if self.add_to_context and context is not None:
            context['q_shifts'] = shifts

        qs = torch.clamp_min_(qs + shifts, 0.)

        return qs


class BasicQNoiseGenerator(QNoiseGenerator):
    def __init__(self,
                 shift_std: float = 1e-3,
                 noise_std: Union[float, Tuple[float, float]] = (0, 1e-3),
                 add_to_context: bool = False,
                 ):
        self.q_shift = QSystematicShiftGenerator(shift_std, add_to_context=add_to_context)
        self.q_noise = QNormalNoiseGenerator(noise_std, add_to_context=add_to_context)

    def apply(self, qs: Tensor, context: dict = None):
        qs = torch.atleast_2d(qs)
        qs = self.q_shift.apply(qs, context)
        qs = self.q_noise.apply(qs, context)
        return qs


class IntensityNoiseGenerator(ProcessData):
    def apply(self, curves: Tensor, context: dict = None):
        raise NotImplementedError


class MultiplicativeLogNormalNoiseGenerator(IntensityNoiseGenerator):
    def __init__(self, std: Union[float, Tuple[float, float]], base: float = 10, add_to_context: bool = False):
        self.std = std
        self.base = base
        self.add_to_context = add_to_context

    def apply(self, curves: Tensor, context: dict = None):
        std = self.std

        if isinstance(std, (list, tuple)):
            std = uniform_sampler(*std, curves.shape[0], 1, device=curves.device, dtype=curves.dtype)
        else:
            std = torch.ones_like(curves) * std

        noise = self.base ** torch.normal(mean=0., std=std)

        if self.add_to_context and context is not None:
            context['std_lognormal'] = std

        return noise * curves


class PoissonNoiseGenerator(IntensityNoiseGenerator):
    def __init__(self,
                 relative_errors: Tuple[float, float] = (0.05, 0.35),
                 abs_errors: float = 1e-8,
                 add_to_context: bool = False,
                 consistent_rel_err: bool = True,
                 logdist: bool = False,
                 ):
        self.relative_errors = relative_errors
        self.abs_errors = abs_errors
        self.add_to_context = add_to_context
        self.consistent_rel_err = consistent_rel_err
        self.logdist = logdist

    def apply(self, curves: Tensor, context: dict = None):
        if self.consistent_rel_err:
            sigmas = self._gen_consistent_sigmas(curves)
        else:
            sigmas = self._gen_sigmas(curves)

        intensities = curves / sigmas ** 2
        curves = torch.poisson(intensities * curves) / intensities

        if self.add_to_context and context is not None:
            context['sigmas'] = sigmas
        return curves

    def _gen_consistent_sigmas(self, curves):
        rel_err = torch.rand(curves.shape[0], device=curves.device, dtype=curves.dtype) * (
                self.relative_errors[1] - self.relative_errors[0]
        ) + self.relative_errors[0]
        sigmas = curves * rel_err[:, None] + self.abs_errors
        return sigmas

    def _gen_sigmas(self, curves):
        if not self.logdist:
            rel_err = torch.rand_like(curves) * (
                    self.relative_errors[1] - self.relative_errors[0]
            ) + self.relative_errors[0]
        else:
            rel_err = torch.rand_like(curves) * (
                    log10(self.relative_errors[1]) - log10(self.relative_errors[0])
            ) + log10(self.relative_errors[0])
            rel_err = 10 ** rel_err

        sigmas = curves * rel_err + self.abs_errors
        return sigmas


class ScalingNoise(IntensityNoiseGenerator):
    def __init__(self,
                 scale_range: tuple = (-0.2e-2, 0.2e-2),
                 add_to_context: bool = False,
                 ):
        self.scale_range = scale_range
        self.add_to_context = add_to_context

    def apply(self, curves: Tensor, context: dict = None):
        scales = uniform_sampler(
            *self.scale_range, curves.shape[0], 1,
            device=curves.device, dtype=curves.dtype
        )
        if self.add_to_context and context is not None:
            context['intensity_scales'] = scales

        curves = curves ** (1 + scales)

        return curves


class ShiftNoise(IntensityNoiseGenerator):
    def __init__(self,
                 shift_range: tuple = (-0.1, 0.2e-2),
                 add_to_context: bool = False,
                 ):
        self.shift_range = shift_range
        self.add_to_context = add_to_context

    def apply(self, curves: Tensor, context: dict = None):
        intensity_shifts = uniform_sampler(
            *self.shift_range, curves.shape[0], 1,
            device=curves.device, dtype=curves.dtype
        )
        if self.add_to_context and context is not None:
            context['intensity_shifts'] = intensity_shifts

        curves = curves * (1 + intensity_shifts)

        return curves


class BasicExpIntensityNoise(IntensityNoiseGenerator):
    def __init__(self,
                 relative_errors: Tuple[float, float] = (0.001, 0.15),
                 abs_errors: float = 1e-8,
                 scale_range: tuple = (-2e-2, 2e-2),
                 shift_range: tuple = (-0.1, 0.2e-2),
                 apply_shift: bool = False,
                 apply_scaling: bool = False,
                 consistent_rel_err: bool = False,
                 add_to_context: bool = False,
                 logdist: bool = False,
                 ):
        self.poisson_noise = PoissonNoiseGenerator(
            relative_errors=relative_errors,
            abs_errors=abs_errors,
            consistent_rel_err=consistent_rel_err,
            add_to_context=add_to_context,
            logdist=logdist,
        )
        self.scaling_noise = ScalingNoise(
            scale_range=scale_range, add_to_context=add_to_context
        ) if apply_scaling else None
        self.shift_noise = ShiftNoise(
            shift_range=shift_range, add_to_context=add_to_context
        ) if apply_shift else None

    def apply(self, curves: Tensor, context: dict = None):
        if self.scaling_noise:
            curves = self.scaling_noise(curves, context)
        if self.shift_noise:
            curves = self.shift_noise(curves, context)
        curves = self.poisson_noise(curves, context)

        return curves
