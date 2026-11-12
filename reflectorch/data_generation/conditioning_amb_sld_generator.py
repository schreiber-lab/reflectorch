from typing import Tuple

import torch
from torch import Tensor

from reflectorch.data_generation.priors.no_constraints import DEFAULT_DEVICE, DEFAULT_DTYPE
from reflectorch.data_generation.utils import uniform_sampler

__all__ = [
    "ConditioningAmbientSLDGenerator",
]


class ConditioningAmbientSLDGenerator:
    def __init__(
        self,
        value_range: Tuple[float, float],
        device: torch.device = DEFAULT_DEVICE,
        dtype: torch.dtype = DEFAULT_DTYPE,
        scaled_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        self.value_range = value_range
        self.device = device
        self.dtype = dtype
        self.scaled_range = scaled_range

        self._min = float(value_range[0])
        self._max = float(value_range[1])

        self._scaled_min = float(scaled_range[0])
        self._scaled_max = float(scaled_range[1])

        self._length = self._scaled_max - self._scaled_min
        self._bias = self._scaled_min

    def get_batch(self, batch_size: int) -> Tensor:
        u = torch.rand((batch_size, 1), device=self.device, dtype=self.dtype)
        return self._min + u * (self._max - self._min)

    def scale(self, x: Tensor) -> Tensor:
        return (x - self._min) / (self._max - self._min) * self._length + self._bias

    def restore(self, x_scaled: Tensor) -> Tensor:
        return (x_scaled - self._bias) / self._length * (self._max - self._min) + self._min