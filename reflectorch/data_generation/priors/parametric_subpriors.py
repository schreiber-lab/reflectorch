# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Dict, Type, List

import torch
from torch import Tensor

from reflectorch.data_generation.priors.base import PriorSampler
from reflectorch.data_generation.priors.params import AbstractParams
from reflectorch.data_generation.priors.no_constraints import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
)

from reflectorch.data_generation.priors.parametric_models import (
    MULTILAYER_MODELS,
    ParametricModel,
)
from reflectorch.data_generation.priors.scaler_mixin import ScalerMixin


class ParametricParams(AbstractParams):  # TODO change the name ParametricParams
    __slots__ = (
        'parameters',
        'min_bounds',
        'max_bounds',
        'max_num_layers',
        'param_model',
    )
    PARAM_NAMES = __slots__
    PARAM_MODEL_CLS: Type[ParametricModel]
    MAX_NUM_LAYERS: int = 30

    def __init__(self,
                 parameters: Tensor,
                 min_bounds: Tensor,
                 max_bounds: Tensor,
                 max_num_layers: int = None,
                 param_model: ParametricModel = None,
                 ):

        max_num_layers = max_num_layers or self.MAX_NUM_LAYERS
        self.param_model = param_model or self.PARAM_MODEL_CLS(max_num_layers)
        self.max_num_layers = max_num_layers
        self.parameters = parameters
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds

    def get_param_labels(self) -> List[str]:
        return self.param_model.get_param_labels()

    def reflectivity(self, q: Tensor, log: bool = False, **kwargs):
        return self.param_model.reflectivity(q, self.parameters, log=log, **kwargs)

    @property
    def max_layer_num(self) -> int:  # keep for back compatibility but TODO: unify api among different params
        return self.max_num_layers

    @property
    def num_params(self) -> int:
        return self.param_model.param_dim

    @property
    def thicknesses(self):
        params = self.param_model.to_standard_params(self.parameters)
        return params['thickness']

    @property
    def roughnesses(self):
        params = self.param_model.to_standard_params(self.parameters)
        return params['roughness']

    @property
    def slds(self):
        params = self.param_model.to_standard_params(self.parameters)
        return params['sld']

    @staticmethod
    def rearrange_context_from_params(
            scaled_params: Tensor,
            context: Tensor,
            inference: bool = False,
            from_params: bool = False,
    ):
        if inference:
            if from_params:
                num_params = scaled_params.shape[-1] // 3
                scaled_params = scaled_params[:, num_params:]
            context = torch.cat([context, scaled_params], dim=-1)
            return context

        num_params = scaled_params.shape[-1] // 3
        assert num_params * 3 == scaled_params.shape[-1]
        scaled_params, bound_context = torch.split(scaled_params, [num_params, 2 * num_params], dim=-1)
        context = torch.cat([context, bound_context], dim=-1)
        return scaled_params, context

    @staticmethod
    def restore_params_from_context(scaled_params: Tensor, context: Tensor):
        num_params = scaled_params.shape[-1]
        scaled_bounds = context[:, -2 * num_params:]
        scaled_params = torch.cat([scaled_params, scaled_bounds], dim=-1)
        return scaled_params

    def as_tensor(self, add_bounds: bool = True, **kwargs) -> Tensor:
        if not add_bounds:
            return self.parameters
        return torch.cat([self.parameters, self.min_bounds, self.max_bounds], -1)

    @classmethod
    def from_tensor(cls, params: Tensor, **kwargs):
        num_params = params.shape[-1] // 3

        params, min_bounds, max_bounds = torch.split(
            params, [num_params, num_params, num_params], dim=-1
        )

        return cls(
            params,
            min_bounds,
            max_bounds,
            **kwargs
        )

    def scale_with_q(self, q_ratio: float):
        self.parameters = self.param_model.scale_with_q(self.parameters, q_ratio)
        self.min_bounds = self.param_model.scale_with_q(self.min_bounds, q_ratio)
        self.max_bounds = self.param_model.scale_with_q(self.max_bounds, q_ratio)


class SubpriorParametricSampler(PriorSampler, ScalerMixin):
    PARAM_CLS = ParametricParams

    def __init__(self,
                 param_ranges: Dict[str, Tuple[float, float]],
                 bound_width_ranges: Dict[str, Tuple[float, float]],
                 model_name: str,
                 device: torch.device = DEFAULT_DEVICE,
                 dtype: torch.dtype = DEFAULT_DTYPE,
                 max_num_layers: int = 50,
                 logdist: bool = False,
                 scaled_range: Tuple[float, float] = (-1., 1.),
                 **kwargs
                 ):
        self.scaled_range = scaled_range
        self.param_model: ParametricModel = MULTILAYER_MODELS[model_name](
            max_num_layers,
            logdist=logdist,
            **kwargs
        )
        self.device = device
        self.dtype = dtype
        self.num_layers = max_num_layers

        self.PARAM_CLS.PARAM_MODEL_CLS = MULTILAYER_MODELS[model_name]
        self.PARAM_CLS.MAX_NUM_LAYERS = max_num_layers

        self._param_dim = self.param_model.param_dim
        self.min_bounds, self.max_bounds, self.min_delta, self.max_delta = self.param_model.init_bounds(
            param_ranges, bound_width_ranges, device=device, dtype=dtype
        )

    @property
    def max_num_layers(self) -> int:
        return self.num_layers

    @property
    def param_dim(self) -> int:
        return self._param_dim

    def sample(self, batch_size: int) -> ParametricParams:
        params, min_bounds, max_bounds = self.param_model.sample(
            batch_size, self.min_bounds, self.max_bounds, self.min_delta, self.max_delta
        )

        params = ParametricParams(
            parameters=params,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            max_num_layers=self.max_num_layers,
            param_model=self.param_model,
        )

        return params

    def scale_params(self, params: ParametricParams) -> Tensor:
        scaled_params = torch.cat([
            self._scale(params.parameters, params.min_bounds, params.max_bounds),
            self._scale(params.min_bounds, self.min_bounds, self.max_bounds),
            self._scale(params.max_bounds, self.min_bounds, self.max_bounds),
        ], -1)
        return scaled_params

    def restore_params(self, scaled_params: Tensor) -> ParametricParams:
        num_params = scaled_params.shape[-1] // 3
        scaled_params, scaled_min_bounds, scaled_max_bounds = torch.split(
            scaled_params, num_params, -1
        )
        min_bounds = self._restore(scaled_min_bounds, self.min_bounds, self.max_bounds)
        max_bounds = self._restore(scaled_max_bounds, self.min_bounds, self.max_bounds)
        params = self._restore(scaled_params, min_bounds, max_bounds)

        return ParametricParams(
            parameters=params,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            max_num_layers=self.max_num_layers,
            param_model=self.param_model,
        )

    def log_prob(self, params: ParametricParams) -> Tensor:
        log_prob = torch.zeros(params.batch_size, device=self.device, dtype=self.dtype)
        log_prob[~self.get_indices_within_bounds(params)] = -float('inf')
        return log_prob

    def get_indices_within_domain(self, params: ParametricParams) -> Tensor:
        return self.get_indices_within_bounds(params)

    def get_indices_within_bounds(self, params: ParametricParams) -> Tensor:
        return (
                torch.all(params.parameters >= params.min_bounds, -1) &
                torch.all(params.parameters <= params.max_bounds, -1)
        )

    def filter_params(self, params: ParametricParams) -> ParametricParams:
        indices = self.get_indices_within_domain(params)
        return params[indices]

    def clamp_params(
            self, params: ParametricParams, inplace: bool = False
    ) -> ParametricParams:
        if inplace:
            params.parameters = torch.clamp_(params.parameters, params.min_bounds, params.max_bounds)
            return params

        return ParametricParams(
            parameters=torch.clamp(params.parameters, params.min_bounds, params.max_bounds),
            min_bounds=params.min_bounds.clone(),
            max_bounds=params.max_bounds.clone(),
            max_num_layers=self.max_num_layers,
            param_model=self.param_model,
        )
