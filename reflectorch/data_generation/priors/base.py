# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor

from reflectorch.data_generation.priors.params import Params

__all__ = [
    "PriorSampler",
]


class PriorSampler(object):
    """Base class for prior samplers"""
    
    PARAM_CLS = Params

    @property
    def param_dim(self) -> int:
        return self.PARAM_CLS.layers_num2size(self.max_num_layers)

    @property
    def max_num_layers(self) -> int:
        raise NotImplementedError

    def sample(self, batch_size: int) -> Params:
        raise NotImplementedError

    def scale_params(self, params: Params) -> Tensor:
        raise NotImplementedError

    def restore_params(self, scaled_params: Tensor) -> Params:
        raise NotImplementedError

    def log_prob(self, params: Params) -> Tensor:
        raise NotImplementedError

    def get_indices_within_domain(self, params: Params) -> Tensor:
        raise NotImplementedError

    def get_indices_within_bounds(self, params: Params) -> Tensor:
        raise NotImplementedError

    def filter_params(self, params: Params) -> Params:
        indices = self.get_indices_within_domain(params)
        return params[indices]

    def clamp_params(self, params: Params) -> Params:
        raise NotImplementedError

    def __repr__(self):
        args = ', '.join(f'{k}={str(v)[:10]}' for k, v in vars(self).items())
        return f'{self.__class__.__name__}({args})'
