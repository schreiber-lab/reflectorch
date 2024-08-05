from typing import Tuple, Dict

import numpy as np
import torch
from torch import Tensor

from reflectorch.data_generation.priors.base import PriorSampler
from reflectorch.data_generation.priors.params import Params
from reflectorch.data_generation.priors.no_constraints import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
)

from reflectorch.data_generation.priors.multilayer_models import MULTILAYER_MODELS, MultilayerModel
from reflectorch.utils import to_t


class MultilayerStructureParams(Params):
    pass


class SimpleMultilayerSampler(PriorSampler):
    PARAM_CLS = MultilayerStructureParams

    def __init__(self,
                 params: Dict[str, Tuple[float, float]],
                 model_name: str,
                 device: torch.device = DEFAULT_DEVICE,
                 dtype: torch.dtype = DEFAULT_DTYPE,
                 max_num_layers: int = 50,
                 ):
        self.multilayer_model: MultilayerModel = MULTILAYER_MODELS[model_name](max_num_layers)
        self.device = device
        self.dtype = dtype
        self.num_layers = max_num_layers
        ordered_bounds = [params[k] for k in self.multilayer_model.PARAMETER_NAMES]
        self._np_bounds = np.array(ordered_bounds).T
        self.min_bounds, self.max_bounds = torch.tensor(ordered_bounds, device=device, dtype=dtype).T[:, None]
        self._param_dim = len(params)

    @property
    def max_num_layers(self) -> int:
        return self.num_layers

    @property
    def param_dim(self) -> int:
        return self._param_dim

    def sample(self, batch_size: int) -> MultilayerStructureParams:
        return self.optimized_sample(batch_size)[0]

    def optimized_sample(self, batch_size: int) -> Tuple[MultilayerStructureParams, Tensor]:
        scaled_params = torch.rand(
            batch_size,
            self.min_bounds.shape[-1],
            device=self.min_bounds.device,
            dtype=self.min_bounds.dtype,
        )

        targets = self.restore_params(scaled_params)

        return targets, scaled_params

    def get_np_bounds(self):
        return np.array(self._np_bounds)

    def restore_np_params(self, params: np.ndarray):
        p = self.multilayer_model.to_standard_params(
            torch.atleast_2d(to_t(params))
        )

        return {
            'thickness': p['thicknesses'].squeeze().cpu().numpy(),
            'roughness': p['roughnesses'].squeeze().cpu().numpy(),
            'sld': p['slds'].squeeze().cpu().numpy()
        }

    def restore_params2parametrized(self, scaled_params: Tensor) -> Tensor:
        return scaled_params * (self.max_bounds - self.min_bounds) + self.min_bounds

    def restore_params(self, scaled_params: Tensor) -> MultilayerStructureParams:
        return self.to_standard_params(self.restore_params2parametrized(scaled_params))

    def to_standard_params(self, params: Tensor) -> MultilayerStructureParams:
        return MultilayerStructureParams(**self.multilayer_model.to_standard_params(params))

    def scale_params(self, params: Params) -> Tensor:
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
