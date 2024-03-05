# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import torch
from torch import Tensor

from reflectorch.data_generation.priors import PriorSampler
from reflectorch.paths import SAVED_MODELS_DIR


class CurvesScaler(object):
    def scale(self, curves: Tensor):
        raise NotImplementedError

    def restore(self, curves: Tensor):
        raise NotImplementedError


class LogAffineCurvesScaler(CurvesScaler):
    """Contains functions for scaling the reflectivity curves to a ML-friendly range and restoring scale curves to the physical range

    Args:
        weight (float): 
        bias (float): 
        eps (float): 
    """
    def __init__(self, weight: float = 0.1, bias: float = 0.5, eps: float = 1e-10):
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def scale(self, curves: Tensor):
        return torch.log10(curves + self.eps) * self.weight + self.bias

    def restore(self, curves: Tensor):
        return 10 ** ((curves - self.bias) / self.weight) - self.eps


class MeanNormalization(CurvesScaler):
    def __init__(self, path: str = None, curves_mean: Tensor = None, device: torch.device = 'cuda'):
        if curves_mean is None:
            curves_mean = torch.load(self.get_path(path))
        self.curves_mean = curves_mean.to(device)

    def scale(self, curves: Tensor):
        self.curves_mean = self.curves_mean.to(curves)
        return curves / self.curves_mean - 1

    def restore(self, curves: Tensor):
        self.curves_mean = self.curves_mean.to(curves)
        return (curves + 1) * self.curves_mean

    @staticmethod
    def save(prior_sampler: PriorSampler, q: Tensor, path: str, num: int = 16384):
        params = prior_sampler.sample(num)
        curves_mean = params.reflectivity(q, log=False).mean(0).cpu()
        torch.save(curves_mean, MeanNormalization.get_path(path))

    @staticmethod
    def get_path(path: str) -> Path:
        if not path.endswith('.pt'):
            path = path + '.pt'
        return SAVED_MODELS_DIR / path
