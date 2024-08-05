import logging
from functools import lru_cache
from typing import Tuple
from math import sqrt

import torch
from torch import Tensor

from reflectorch.data_generation.utils import (
    get_slds_from_d_rhos,
    uniform_sampler,
)

from reflectorch.data_generation.priors.utils import (
    get_allowed_roughness_indices,
    generate_roughnesses,
    params_within_bounds,
)
from reflectorch.data_generation.priors.scaler_mixin import ScalerMixin
from reflectorch.data_generation.priors.base import PriorSampler
from reflectorch.data_generation.priors.params import Params

__all__ = [
    "BasicPriorSampler",
    "DEFAULT_ROUGHNESS_RANGE",
    "DEFAULT_THICKNESS_RANGE",
    "DEFAULT_SLD_RANGE",
    "DEFAULT_NUM_LAYERS",
    "DEFAULT_DEVICE",
    "DEFAULT_DTYPE",
    "DEFAULT_SCALED_RANGE",
    "DEFAULT_USE_DRHO",
]

DEFAULT_THICKNESS_RANGE: Tuple[float, float] = (1., 500.)
DEFAULT_ROUGHNESS_RANGE: Tuple[float, float] = (0., 50.)
DEFAULT_SLD_RANGE: Tuple[float, float] = (-10., 30.)
DEFAULT_NUM_LAYERS: int = 5
DEFAULT_USE_DRHO: bool = False
DEFAULT_DEVICE: torch.device = torch.device('cuda')
DEFAULT_DTYPE: torch.dtype = torch.float64
DEFAULT_SCALED_RANGE: Tuple[float, float] = (-sqrt(3.), sqrt(3.))


class BasicPriorSampler(PriorSampler, ScalerMixin):
    """Prior samplers for thicknesses, roughnesses and slds"""
    def __init__(self,
                 thickness_range: Tuple[float, float] = DEFAULT_THICKNESS_RANGE,
                 roughness_range: Tuple[float, float] = DEFAULT_ROUGHNESS_RANGE,
                 sld_range: Tuple[float, float] = DEFAULT_SLD_RANGE,
                 num_layers: int = DEFAULT_NUM_LAYERS,
                 use_drho: bool = DEFAULT_USE_DRHO,
                 device: torch.device = DEFAULT_DEVICE,
                 dtype: torch.dtype = DEFAULT_DTYPE,
                 scaled_range: Tuple[float, float] = DEFAULT_SCALED_RANGE,
                 restrict_roughnesses: bool = True,
                 ):
        self.logger = logging.getLogger(__name__)
        self.thickness_range = thickness_range
        self.roughness_range = roughness_range
        self.sld_range = sld_range
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.scaled_range = scaled_range
        self.use_drho = use_drho
        self.restrict_roughnesses = restrict_roughnesses

    @property
    def max_num_layers(self) -> int:
        return self.num_layers

    @lru_cache()
    def min_vector(self, layers_num, drho: bool = False):
        if drho:
            sld_min = self.sld_range[0] - self.sld_range[1]
        else:
            sld_min = self.sld_range[0]

        return torch.tensor(
            [self.thickness_range[0]] * layers_num +
            [self.roughness_range[0]] * (layers_num + 1) +
            [sld_min] * (layers_num + 1),
            device=self.device,
            dtype=self.dtype
        )

    @lru_cache()
    def max_vector(self, layers_num, drho: bool = False):
        if drho:
            sld_max = self.sld_range[1] - self.sld_range[0]
        else:
            sld_max = self.sld_range[1]
        return torch.tensor(
            [self.thickness_range[1]] * layers_num +
            [self.roughness_range[1]] * (layers_num + 1) +
            [sld_max] * (layers_num + 1),
            device=self.device,
            dtype=self.dtype
        )

    @lru_cache()
    def delta_vector(self, layers_num, drho: bool = False):
        return self._get_delta_vector(self.min_vector(layers_num, drho), self.max_vector(layers_num, drho))

    def restore_params(self, scaled_params: Tensor) -> Params:
        layers_num = self.PARAM_CLS.size2layers_num(scaled_params.shape[-1])

        params_t = self._restore(
            scaled_params,
            self.min_vector(layers_num, drho=self.use_drho).to(scaled_params),
            self.max_vector(layers_num, drho=self.use_drho).to(scaled_params),
        )

        params = self.PARAM_CLS.from_tensor(params_t)

        if self.use_drho:
            params.slds = get_slds_from_d_rhos(params.slds)

        return params

    def scale_params(self, params: Params) -> Tensor:
        layers_num = params.max_layer_num

        return self._scale(
            params.as_tensor(use_drho=self.use_drho),
            self.min_vector(layers_num, drho=self.use_drho).to(params.thicknesses),
            self.max_vector(layers_num, drho=self.use_drho).to(params.thicknesses),
        )

    def get_indices_within_bounds(self, params: Params) -> Tensor:
        layer_num = params.max_layer_num

        return params_within_bounds(
            params.as_tensor(),
            self.min_vector(layer_num),
            self.max_vector(layer_num),
        )

    def get_indices_within_domain(self, params: Params) -> Tensor:
        if self.restrict_roughnesses:
            indices = (
                    self.get_indices_within_bounds(params) &
                    self.get_allowed_roughness_indices(params)
            )
        else:
            indices = self.get_indices_within_bounds(params)
        return indices

    def clamp_params(self, params: Params) -> Params:
        layer_num = params.max_layer_num
        params = params.as_tensor()
        params = torch.clamp(
            params,
            self.min_vector(layer_num),
            self.max_vector(layer_num),
        )
        params = Params.from_tensor(params)
        return params

    @staticmethod
    def get_allowed_roughness_indices(params: Params) -> Tensor:
        return get_allowed_roughness_indices(params.thicknesses, params.roughnesses)

    def log_prob(self, params: Params) -> Tensor:
        # so far we ignore non-uniform distribution of roughnesses and slds.
        log_prob = torch.zeros(params.batch_size, device=params.device, dtype=params.dtype)
        indices = self.get_indices_within_bounds(params)
        log_prob[~indices] = float('-inf')
        return log_prob

    def sample(self, batch_size: int) -> Params:
        slds = self.generate_slds(batch_size)
        thicknesses = self.generate_thicknesses(batch_size)
        roughnesses = self.generate_roughnesses(thicknesses)

        params = Params(thicknesses, roughnesses, slds)

        return params

    def generate_slds(self, batch_size: int):
        return uniform_sampler(
            *self.sld_range, batch_size,
            self.num_layers + 1,
            device=self.device,
            dtype=self.dtype
        )

    def generate_thicknesses(self, batch_size: int):
        return uniform_sampler(
            *self.thickness_range, batch_size,
            self.num_layers,
            device=self.device,
            dtype=self.dtype
        )

    def generate_roughnesses(self, thicknesses: Tensor) -> Tensor:
        if self.restrict_roughnesses:
            return generate_roughnesses(thicknesses, self.roughness_range)
        else:
            return uniform_sampler(
                *self.roughness_range, thicknesses.shape[0],
                self.num_layers + 1,
                device=self.device,
                dtype=self.dtype
            )
