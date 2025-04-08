from typing import List, Union, Tuple
from math import log10

import torch
from torch import Tensor

from reflectorch.data_generation.process_data import ProcessData
from reflectorch.data_generation.utils import logdist_sampler, uniform_sampler

__all__ = [
    "QNoiseGenerator",
    "IntensityNoiseGenerator",
    "QNormalNoiseGenerator",
    "QSystematicShiftGenerator",
    "PoissonNoiseGenerator",
    "MultiplicativeLogNormalNoiseGenerator",
    "ScalingNoise",
    "ShiftNoise",
    "BackgroundNoise",
    "BasicExpIntensityNoise",
    "GaussianExpIntensityNoise",
    "BasicQNoiseGenerator",
]


class QNoiseGenerator(ProcessData):
    """Base class for q noise generators"""
    def apply(self, qs: Tensor, context: dict = None):
        return qs


class QNormalNoiseGenerator(QNoiseGenerator):
    """Q noise generator which adds to each q value of the reflectivity curve a noise sampled from a normal distribution.

    Args:
        std (Union[float, Tuple[float, float]], optional): the standard deviation of the normal distribution (the same for all curves in the batch if provided as a float, 
                                                           or uniformly sampled for each curve in the batch if provided as a tuple)
    """
    def __init__(self,
                 std: Union[float, Tuple[float, float]] = (0, 1e-3),
                 add_to_context: bool = False
                 ):
        self.std = std
        self.add_to_context = add_to_context

    def apply(self, qs: Tensor, context: dict = None):
        """applies noise to the q values"""
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
    """Q noise generator which samples a q shift (for each curve in the batch) from a normal distribution adds it to all q values of the curve

    Args:
        std (float): the standard deviation of the normal distribution
    """
    def __init__(self, std: float, add_to_context: bool = True):
        self.std = std
        self.add_to_context = add_to_context

    def apply(self, qs: Tensor, context: dict = None):
        """applies systematic shifts to the q values"""
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
    """Q noise generator which applies both systematic shifts (same change for all q points in the curve) and random noise (different changes per q point in the curve)

    Args:
        shift_std (float, optional): the standard deviation of the normal distribution for systematic q shifts 
                                    (i.e. same change applied to all q points in the curve). Defaults to 1e-3.
        noise_std (Union[float, Tuple[float, float]], optional): the standard deviation of the normal distribution for random q noise 
                                    (i.e. different changes applied to each q point in the curve). The standard deviation is the same 
                                    for all curves in the batch if provided as a float, or uniformly sampled for each curve in the batch if provided as a tuple. 
                                    Defaults to (0, 1e-3).
    """
    def __init__(self,
                 apply_systematic_shifts: bool = True,
                 shift_std: float = 1e-3,
                 apply_gaussian_noise: bool = False,
                 noise_std: Union[float, Tuple[float, float]] = (0, 1e-3),
                 add_to_context: bool = False,
                 ):
        self.q_shift = QSystematicShiftGenerator(shift_std, add_to_context=add_to_context) if apply_systematic_shifts else None
        self.q_noise = QNormalNoiseGenerator(noise_std, add_to_context=add_to_context) if apply_gaussian_noise else None

    def apply(self, qs: Tensor, context: dict = None):
        """applies noise to the q values"""
        qs = torch.atleast_2d(qs)
        if self.q_shift:
            qs = self.q_shift.apply(qs, context)
        if self.q_noise:
            qs = self.q_noise.apply(qs, context)
        return qs


class IntensityNoiseGenerator(ProcessData):
    """Base class for intensity noise generators"""
    def apply(self, curves: Tensor, context: dict = None):
        raise NotImplementedError


class MultiplicativeLogNormalNoiseGenerator(IntensityNoiseGenerator):
    """Noise generator which applies noise as :math:`R_n = R * b^{\epsilon}` , where :math:`b` is a base and :math:`\epsilon` is sampled from the normal distribution :math:`\epsilon \sim \mathcal{N}(0, std)` .
    In logarithmic space, this translates to :math:`\log_b(R_n) = \log_b(R) + \epsilon` .


    Args:
        std (Union[float, Tuple[float, float]]): the standard deviation of the normal distribution from which the noise is sampled. The standard deviation is the same 
                                                for all curves in the batch if provided as a float, or uniformly sampled for each curve in the batch if provided as a tuple.
        base (float, optional): the base of the logarithm. Defaults to 10.
    """
    def __init__(self, std: Union[float, Tuple[float, float]], base: float = 10, add_to_context: bool = False):
        self.std = std
        self.base = base
        self.add_to_context = add_to_context

    def apply(self, curves: Tensor, context: dict = None):
        """applies noise to the curves"""
        std = self.std

        if isinstance(std, (list, tuple)):
            std = uniform_sampler(*std, curves.shape[0], 1, device=curves.device, dtype=curves.dtype)
        else:
            std = torch.ones_like(curves) * std

        noise = self.base ** torch.normal(mean=0., std=std)

        if self.add_to_context and context is not None:
            context['std_lognormal'] = std

        return noise * curves

class GaussianNoiseGenerator(IntensityNoiseGenerator):
    """Noise generator which applies noise as R_n = R + eps, with eps~N(0, sigmas) and sigmas = relative_errors * R

    Args:
        relative_errors (Union[float, Tuple[float, float], List[Tuple[float, float]]])
        consistent_relative_errors (bool): If True the relative_error is the same for all point of a curve, otherwise it is sampled uniformly.
    """

    def __init__(self, relative_errors: Union[float, Tuple[float, float], List[float], List[Tuple[float, float]]], 
                 consistent_rel_err: bool = False,
                 add_to_context: bool = False):
        self.relative_errors = relative_errors
        self.consistent_rel_err = consistent_rel_err
        self.add_to_context = add_to_context

    def apply(self, curves: Tensor, context: dict = None):
        """Applies Gaussian noise to the curves."""
        relative_errors = self.relative_errors
        num_channels = curves.shape[1] if curves.dim() == 3 else 1

        if isinstance(relative_errors, float):
            relative_errors = torch.ones_like(curves) * relative_errors

        elif isinstance(relative_errors, (list, tuple)) and isinstance(relative_errors[0], float):
            if self.consistent_rel_err:
                relative_errors = uniform_sampler(*relative_errors, curves.shape[0], num_channels, device=curves.device, dtype=curves.dtype)
                if num_channels > 1:
                    relative_errors = relative_errors.unsqueeze(-1)
            else:
                relative_errors = uniform_sampler(*relative_errors, *curves.shape, device=curves.device, dtype=curves.dtype)

        else:
            if self.consistent_rel_err:
                relative_errors = torch.stack([uniform_sampler(*item, curves.shape[0], 1, device=curves.device, dtype=curves.dtype) for item in relative_errors], dim=1)
            else:
                relative_errors = torch.stack([uniform_sampler(*item, curves.shape[0], curves.shape[-1], device=curves.device, dtype=curves.dtype) for item in relative_errors], dim=1)

        sigmas = relative_errors * curves
        noise = torch.normal(mean=0., std=sigmas).clamp_min_(0.0)

        if self.add_to_context and context is not None:
            context['relative_errors'] = relative_errors
            context['sigmas'] = sigmas
            
        return curves + noise

class PoissonNoiseGenerator(IntensityNoiseGenerator):
    """Noise generator which applies Poisson noise to the reflectivity curves

    Args:
        relative_errors (Tuple[float, float], optional): the range of relative errors to apply to the intensity curves. Defaults to (0.05, 0.35).
        abs_errors (float, optional): a small constant added to prevent division by zero. Defaults to 1e-8.
        consistent_rel_err (bool, optional): If ``True``, the same relative error is used for all points in a curve.
        logdist (bool, optional): If ``True``, the relative errors in are sampled in logarithmic space. Defaults to False.
    """
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
        """applies noise to the curves"""
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
    """Noise generator which applies scaling noise to reflectivity curves (equivalent to a vertical stretch or compression of the curve in the logarithmic domain). 
    The output is R^(1 + scale_factor), which corresponds in logarithmic domain to (1 + scale_factor) * log(R).

    Args:
        scale_range (tuple, optional): the range of scaling factors (one factor sampled per curve in the batch). Defaults to (-0.2e-2, 0.2e-2).
    """
    def __init__(self,
                 scale_range: tuple = (-0.2e-2, 0.2e-2),
                 add_to_context: bool = False,
                 ):
        self.scale_range = scale_range
        self.add_to_context = add_to_context

    def apply(self, curves: Tensor, context: dict = None):
        """applies noise to the curves"""
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
        """Noise generator which applies shifting noise to reflectivity curves (equivalent to a vertical shift of the entire curve in the logarithmic domain). 
        The output is R * (1 + shift_factor), which corresponds in logarithmic domain to log(R) + log(1 + shift_factor).
        Args:
            shift_range (tuple, optional): the range of shift factors (one factor sampled per curve in the batch). Defaults to (-0.1, 0.2e-2).
        """
        self.shift_range = shift_range
        self.add_to_context = add_to_context

    def apply(self, curves: Tensor, context: dict = None):
        """applies noise to the curves"""
        intensity_shifts = uniform_sampler(
            *self.shift_range, curves.shape[0], 1,
            device=curves.device, dtype=curves.dtype
        )
        if self.add_to_context and context is not None:
            context['intensity_shifts'] = intensity_shifts

        curves = curves * (1 + intensity_shifts)

        return curves
    
class BackgroundNoise(IntensityNoiseGenerator):
    """
    Noise generator which adds a constant background to reflectivity curves.

    Args:
        background_range (tuple, optional): The range from which the background value is sampled. Defaults to (1.0e-10, 1.0e-8).
        add_to_context (bool, optional): If True, adds generated noise parameters to the context dictionary. Defaults to False.
    """
    def __init__(self,
                 background_range: tuple = (1.0e-10, 1.0e-8),
                 add_to_context: bool = False,
                 ):
        self.background_range = background_range
        self.add_to_context = add_to_context

    def apply(self, curves: Tensor, context: dict = None) -> Tensor:
        """applies background noise to the curves"""
        backgrounds = logdist_sampler(
            *self.background_range, curves.shape[0], 1,
            device=curves.device, dtype=curves.dtype
        )
        if self.add_to_context and context is not None:
            context['backgrounds'] = backgrounds

        curves = curves + backgrounds

        return curves

class GaussianExpIntensityNoise(IntensityNoiseGenerator):
    """
    A composite noise generator that applies Gaussian, shift and background noise to reflectivity curves.

    This class combines three types of noise:
    1. Gaussian noise: Applies Gaussian noise (to account for count-based Poisson noise as well as other sources of error)
    2. Shift noise: Applies a multiplicative shift to the curves, equivalent to a vertical shift in logarithmic space.
    3. Background noise: Adds a constant background value to the curves.

    Args:
        relative_errors (Union[float, Tuple[float, float], List[Tuple[float, float]]]): The range of relative errors for Gaussian noise. Defaults to (0.001, 0.15).
        consistent_rel_err (bool, optional): If True, uses a consistent relative error for Gaussian noise across all points in a curve. Defaults to False.
        shift_range (tuple, optional): The range of shift factors for shift noise. Defaults to (-0.1, 0.2e-2).
        background_range (tuple, optional): The range from which the background value is sampled. Defaults to (1.0e-10, 1.0e-8).
        apply_shift (bool, optional): If True, applies shift noise to the curves. Defaults to False.
        apply_background (bool, optional): If True, applies background noise to the curves. Defaults to False.
        same_background_across_channels(bool, optional): If True, the same background is applied to all channels of a multi-channel curve. Defaults to False.
        add_to_context (bool, optional): If True, adds generated noise parameters to the context dictionary. Defaults to False.
    """
    def __init__(self,
                 relative_errors: Tuple[float, float] = (0.001, 0.15),
                 consistent_rel_err: bool = False,
                 apply_shift: bool = False,
                 shift_range: tuple = (-0.1, 0.2e-2),
                 apply_background: bool = False,
                 background_range: tuple = (1.0e-10, 1.0e-8),
                 same_background_across_channels: bool = False,
                 add_to_context: bool = False,
                 ):
        
        self.gaussian_noise = GaussianNoiseGenerator(
            relative_errors=relative_errors,
            consistent_rel_err=consistent_rel_err,
            add_to_context=add_to_context,
        )

        self.shift_noise = ShiftNoise(
            shift_range=shift_range, add_to_context=add_to_context
        ) if apply_shift else None

        self.background_noise = BackgroundNoise(
            background_range=background_range, add_to_context=add_to_context
        ) if apply_background else None

    def apply(self, curves: Tensor, context: dict = None):
        """applies the specified types of noise to the input curves"""
        if self.shift_noise:
            curves = self.shift_noise(curves, context)
        
        if self.background_noise:
            curves = self.background_noise.apply(curves, context)

        curves = self.gaussian_noise(curves, context)
        
        return curves

class BasicExpIntensityNoise(IntensityNoiseGenerator):
    """
    A composite noise generator that applies Poisson, scaling, shift and background noise to reflectivity curves.

    This class combines four types of noise:

    1. **Poisson noise**: Simulates count-based noise common in photon counting experiments.
    2. **Scaling noise**: Applies a scaling transformation to the curves, equivalent to a vertical stretch or compression in logarithmic space.
    3. **Shift noise**: Applies a multiplicative shift to the curves, equivalent to a vertical shift in logarithmic space.
    4. **Background noise**: Adds a constant background value to the curves.

    Args:
        relative_errors (Tuple[float, float], optional): The range of relative errors for Poisson noise. Defaults to (0.001, 0.15).
        abs_errors (float, optional): A small constant added to prevent division by zero in Poisson noise. Defaults to 1e-8.
        scale_range (tuple, optional): The range of scaling factors for scaling noise. Defaults to (-2e-2, 2e-2).
        shift_range (tuple, optional): The range of shift factors for shift noise. Defaults to (-0.1, 0.2e-2).
        background_range (tuple, optional): The range from which the background value is sampled. Defaults to (1.0e-10, 1.0e-8).
        apply_shift (bool, optional): If True, applies shift noise to the curves. Defaults to False.
        apply_scaling (bool, optional): If True, applies scaling noise to the curves. Defaults to False.
        apply_background (bool, optional): If True, applies background noise to the curves. Defaults to False.
        consistent_rel_err (bool, optional): If True, uses a consistent relative error for Poisson noise across all points in a curve. Defaults to False.
        add_to_context (bool, optional): If True, adds generated noise parameters to the context dictionary. Defaults to False.
        logdist (bool, optional): If True, samples relative errors for Poisson noise in logarithmic space. Defaults to False.
    """
    def __init__(self,
                 relative_errors: Tuple[float, float] = (0.001, 0.15),
                 abs_errors: float = 1e-8,
                 scale_range: tuple = (-2e-2, 2e-2),
                 shift_range: tuple = (-0.1, 0.2e-2),
                 background_range: tuple = (1.0e-10, 1.0e-8),
                 apply_shift: bool = False,
                 apply_scaling: bool = False,
                 apply_background: bool = False,
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

        self.background_noise = BackgroundNoise(
            background_range=background_range, add_to_context=add_to_context
        ) if apply_background else None

    def apply(self, curves: Tensor, context: dict = None):
        """applies the specified types of noise to the input curves"""
        if self.scaling_noise:
            curves = self.scaling_noise(curves, context)
        if self.shift_noise:
            curves = self.shift_noise(curves, context)
        curves = self.poisson_noise(curves, context)

        if self.background_noise:
            curves = self.background_noise.apply(curves, context)

        return curves