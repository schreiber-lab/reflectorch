from pathlib import Path

import torch
from torch import Tensor

from reflectorch.data_generation.priors import PriorSampler
from reflectorch.paths import SAVED_MODELS_DIR


class CurvesScaler(object):
    """Base class for curve scalers"""
    def scale(self, curves: Tensor):
        raise NotImplementedError

    def restore(self, curves: Tensor):
        raise NotImplementedError


class LogAffineCurvesScaler(CurvesScaler):
    """    Curve scaler which scales the reflectivity curves according to the logarithmic affine transformation:
    :math:`\log_{10}(R + eps) \cdot weight + bias`.

    Args:
        weight (float): multiplication factor in the transformation
        bias (float): addition term in the transformation
        eps (float): sets the minimum intensity value of the reflectivity curves which is considered
    """
    def __init__(self, weight: float = 0.1, bias: float = 0.5, eps: float = 1e-10):
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def scale(self, curves: Tensor):
        """scales the reflectivity curves to a ML-friendly range

        Args:
            curves (Tensor): original reflectivity curves

        Returns:
            Tensor: reflectivity curves scaled to a ML-friendly range
        """
        return torch.log10(curves + self.eps) * self.weight + self.bias

    def restore(self, curves: Tensor):
        """restores the physical reflectivity curves

        Args:
            curves (Tensor): scaled reflectivity curves

        Returns:
            Tensor: reflectivity curves restored to the physical range
        """
        return 10 ** ((curves - self.bias) / self.weight) - self.eps


class MeanNormalizationCurvesScaler(CurvesScaler):
    """Curve scaler which scales the reflectivity curves by the precomputed mean of a batch of curves

    Args:
        path (str, optional): path to the precomputed mean of the curves, only used if ``curves_mean`` is None. Defaults to None.
        curves_mean (Tensor, optional): the precomputed mean of the curves. Defaults to None.
        device (torch.device, optional): the Pytorch device. Defaults to 'cuda'.
    """

    def __init__(self, path: str = None, curves_mean: Tensor = None, device: torch.device = 'cuda'):
        if curves_mean is None:
            curves_mean = torch.load(self.get_path(path))
        self.curves_mean = curves_mean.to(device)

    def scale(self, curves: Tensor):
        """scales the reflectivity curves to a ML-friendly range

        Args:
            curves (Tensor): original reflectivity curves

        Returns:
            Tensor: reflectivity curves scaled to a ML-friendly range
        """
        self.curves_mean = self.curves_mean.to(curves)
        return curves / self.curves_mean - 1

    def restore(self, curves: Tensor):
        """restores the physical reflectivity curves

        Args:
            curves (Tensor): scaled reflectivity curves

        Returns:
            Tensor: reflectivity curves restored to the physical range
        """
        self.curves_mean = self.curves_mean.to(curves)
        return (curves + 1) * self.curves_mean

    @staticmethod
    def save(prior_sampler: PriorSampler, q: Tensor, path: str, num: int = 16384):
        """computes the mean of a batch of reflectivity curves and saves it

        Args:
            prior_sampler (PriorSampler): the prior sampler
            q (Tensor): the q values
            path (str): the path for saving the mean of the curves
            num (int, optional): the number of curves used to compute the mean. Defaults to 16384.
        """
        params = prior_sampler.sample(num)
        curves_mean = params.reflectivity(q, log=False).mean(0).cpu()
        torch.save(curves_mean, MeanNormalizationCurvesScaler.get_path(path))

    @staticmethod
    def get_path(path: str) -> Path:
        if not path.endswith('.pt'):
            path = path + '.pt'
        return SAVED_MODELS_DIR / path
