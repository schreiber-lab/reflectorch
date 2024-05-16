from typing import Tuple

import torch
from torch import Tensor


class ScalerMixin:
    """Provides functionality to multiple inheritance classes for scaling the parameters to a specified range and restoring them to the original range."""
    @staticmethod
    def _get_delta_vector(min_vector: Tensor, max_vector: Tensor):
        delta_vector = max_vector - min_vector
        delta_vector[delta_vector == 0.] = 1.
        return delta_vector

    def _scale(self, params_t: Tensor, min_vector: Tensor, max_vector: Tensor):
        """scale the parameters to a specific range
        Args:
            params_t (Tensor): the values of the parameters
            min_vector (Tensor): minimum possible values of each parameter
            max_vector (Tensor): maximum possible values of each parameter

        Returns:
            Tensor: the scaled parameters
        """
        if params_t.dim() == 2:
            min_vector = torch.atleast_2d(min_vector)
            max_vector = torch.atleast_2d(max_vector)

        delta_vector = max_vector - min_vector
        delta_vector[delta_vector == 0.] = 1.
        scaled_params = (
                                params_t - min_vector
                        ) / self._get_delta_vector(min_vector, max_vector) * self._length + self._bias
        return scaled_params

    def _restore(self, scaled_params: Tensor, min_vector: Tensor, max_vector: Tensor):
        """restores the parameters to their original range
        Args:
            scaled_params: (Tensor): the scaled parameters
            min_vector (Tensor): minimum possible values of each parameter
            max_vector (Tensor): maximum possible values of each parameter

        Returns:
            Tensor: the restored parameters
        """
        if scaled_params.dim() == 2:
            min_vector = torch.atleast_2d(min_vector)
            max_vector = torch.atleast_2d(max_vector)

        params_t = (
                           scaled_params - self._bias
                   ) / self._length * self._get_delta_vector(min_vector, max_vector) + min_vector
        return params_t

    @property
    def scaled_range(self) -> Tuple[float, float]:
        return self._scaled_range

    @scaled_range.setter
    def scaled_range(self, scaled_range: Tuple[float, float]):
        """sets the range used for scaling the parameters"""
        self._scaled_range = scaled_range
        self._length = scaled_range[1] - scaled_range[0]
        self._init_bias = (scaled_range[0] + scaled_range[1]) / 2
        self._bias = (self._init_bias - 0.5 * self._length)
