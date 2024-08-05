from typing import Union, Tuple
from math import log

import torch
from torch import Tensor

from reflectorch.data_generation.priors.base import PriorSampler
from reflectorch.data_generation.priors.params import Params
from reflectorch.data_generation.utils import get_param_labels

__all__ = [
    'SingleParamPrior',
    'UniformParamPrior',
    'GaussianParamPrior',
    'TruncatedGaussianParamPrior',
    'SimplePriorSampler',

]


class SingleParamPrior(object):
    def sample(self, batch_num: int, device=None, dtype=None):
        raise NotImplementedError

    def log_prob(self, params: Tensor):
        raise NotImplementedError

    def to_conf(self):
        vars_dict = {k: v for k, v in vars(self).items() if not k.startswith('_')}
        return {
            'cls': self.__class__.__name__,
            'kwargs': vars_dict
        }

    def scale(self, params: Tensor) -> Tensor:
        raise NotImplementedError

    def restore(self, params: Tensor) -> Tensor:
        raise NotImplementedError


class SimplePriorSampler(PriorSampler):
    def __init__(self,
                 *params: Union[Tuple[float, float], SingleParamPrior],
                 device=None, dtype=None, param_cls=None, scaling_prior: PriorSampler = None,
                 ):
        if param_cls is not None:
            self.PARAM_CLS = param_cls
        elif scaling_prior is not None:
            self.PARAM_CLS = scaling_prior.PARAM_CLS
        self.param_priors: Tuple[SingleParamPrior, ...] = self._init_params(params)
        self._num_layers = self.PARAM_CLS.size2layers_num(len(params))
        self.device = device
        self.dtype = dtype
        self.scaling_prior = scaling_prior

    @property
    def max_num_layers(self) -> int:
        return self._num_layers

    def _init_params(self, params: Tuple[Union[Tuple[float, float], SingleParamPrior], ...]):
        assert len(params) == self.PARAM_CLS.layers_num2size(self.PARAM_CLS.size2layers_num(len(params)))
        params = tuple(
            param if isinstance(param, SingleParamPrior) else UniformParamPrior(*param)
            for param in params
        )
        return params

    def sample(self, batch_size: int) -> Params:
        t_params = torch.stack([
            param.sample(batch_size, device=self.device, dtype=self.dtype) for param in self.param_priors
        ], -1)
        params = self.PARAM_CLS.from_tensor(t_params)
        return params

    def scale_params(self, params: Params) -> Tensor:
        if self.scaling_prior:
            return self.scaling_prior.scale_params(params)

        t_params = params.as_tensor()

        scaled_params = torch.stack(
            [param_prior.scale(param) for param, param_prior in zip(t_params.T, self.param_priors)], -1
        )

        return scaled_params

    def restore_params(self, scaled_params: Tensor) -> Params:
        if self.scaling_prior:
            return self.scaling_prior.restore_params(scaled_params)
        t_params = torch.stack(
            [param_prior.restore(param) for param, param_prior in zip(scaled_params.T, self.param_priors)], -1
        )
        return self.PARAM_CLS.from_tensor(t_params)

    def log_prob(self, params: Params) -> Tensor:
        t_params = params.as_tensor()
        log_probs = torch.stack(
            [param_prior.log_prob(param) for param, param_prior in zip(t_params.T, self.param_priors)], -1
        )
        return log_probs.sum(1)

    def get_indices_within_domain(self, params: Params) -> Tensor:
        log_probs = self.log_prob(params)
        return torch.isfinite(log_probs)

    def get_indices_within_bounds(self, params: Params) -> Tensor:
        return self.get_indices_within_domain(params)

    def __repr__(self):
        layers_num = self.PARAM_CLS.size2layers_num(len(self.param_priors))
        labels = get_param_labels(layers_num)
        prior_str = '\n\t'.join(f'{label}: {param_prior}' for label, param_prior in zip(labels, self.param_priors))
        return f'SimplePriorSampler(\n\t{prior_str}\n)'


class UniformParamPrior(SingleParamPrior):
    def __init__(self, min_value: float, max_value: float, device=None, dtype=None):
        assert min_value < max_value
        self.min_value, self.max_value, self.delta = min_value, max_value, max_value - min_value
        self._lob_prob_const = - log(self.delta)
        self.device = device
        self.dtype = dtype

    def sample(self, batch_num: int, device=None, dtype=None):
        params = torch.rand(
            batch_num, device=(device or self.device), dtype=(dtype or self.dtype)
        ) * self.delta + self.min_value
        return params

    def log_prob(self, params: Tensor):
        log_probs = torch.fill_(torch.ones_like(params), self._lob_prob_const)
        log_probs[(params < self.min_value) | (params > self.max_value)] = - float('inf')
        return log_probs

    def scale(self, params: Tensor) -> Tensor:
        return (params - self.min_value) / self.delta

    def restore(self, params: Tensor) -> Tensor:
        return params * self.delta + self.min_value

    def __repr__(self):
        return f'UniformParamPrior(min={self.min_value}, max={self.max_value})'


class GaussianParamPrior(SingleParamPrior):
    _GAUSS_SCALE_CONST: float = 4.

    def __init__(self, mean: float, std: float, device=None, dtype=None):
        assert std > 0
        self.mean = mean
        self.std = std
        self.device = device
        self.dtype = dtype

    def sample(self, batch_num: int, device=None, dtype=None):
        params = torch.normal(
            self.mean, self.std, (batch_num,), device=(device or self.device), dtype=(dtype or self.dtype)
        )
        return params

    def log_prob(self, params: Tensor):
        # ignore constant
        log_probs = - ((params - self.mean) / self.std) ** 2 / 2
        return log_probs

    def scale(self, params: Tensor) -> Tensor:
        return (params - self.mean) / (self.std * self._GAUSS_SCALE_CONST)

    def restore(self, params: Tensor) -> Tensor:
        return params * self.std * self._GAUSS_SCALE_CONST + self.mean

    def __repr__(self):
        return f'GaussianParamPrior(mean={self.mean}, std={self.std})'


class TruncatedGaussianParamPrior(GaussianParamPrior):
    def sample(self, batch_num: int, device=None, dtype=None) -> Tensor:
        params = torch.normal(
            self.mean, self.std, (batch_num,), device=(device or self.device), dtype=(dtype or self.dtype)
        )
        negative_params = params < 0.
        num_negative_params = negative_params.sum().item()
        if num_negative_params:
            params[negative_params] = self.sample(num_negative_params, device=device, dtype=dtype)
        return params

    def log_prob(self, params: Tensor) -> Tensor:
        # ignore constant
        log_probs: Tensor = - ((params - self.mean) / self.std) ** 2 / 2
        log_probs[params < 0.] = - float('inf')
        return log_probs

    def __repr__(self):
        return f'TruncatedGaussianParamPrior(mean={self.mean}, std={self.std})'
