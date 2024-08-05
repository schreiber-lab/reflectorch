from typing import Tuple, Union, List

import torch
from torch import Tensor

from reflectorch.data_generation.utils import (
    uniform_sampler,
    logdist_sampler,
)

from reflectorch.data_generation.priors.scaler_mixin import ScalerMixin
from reflectorch.data_generation.priors.base import PriorSampler
from reflectorch.data_generation.priors.subprior_sampler import UniformSubPriorParams
from reflectorch.data_generation.priors.params import Params
from reflectorch.data_generation.priors.utils import get_max_allowed_roughness
from reflectorch.data_generation.priors.no_constraints import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
)


class ExpUniformSubPriorSampler(PriorSampler, ScalerMixin):
    PARAM_CLS = UniformSubPriorParams

    def __init__(self,
                 params: List[Union[float, Tuple[float, float], Tuple[float, float, float, float]]],
                 device: torch.device = DEFAULT_DEVICE,
                 dtype: torch.dtype = DEFAULT_DTYPE,
                 scaled_range: Tuple[float, float] = (-1, 1),
                 logdist: bool = False,
                 relative_min_bound_width: float = 1e-4,
                 smaller_roughnesses: bool = True,
                 ):
        self.device = device
        self.dtype = dtype
        self.scaled_range = scaled_range
        self.relative_min_bound_width = relative_min_bound_width
        self.logdist = logdist
        self.smaller_roughnesses = smaller_roughnesses
        self._init_params(*params)

    @property
    def max_num_layers(self) -> int:
        return self.num_layers

    def _init_params(self, *params: Union[float, Tuple[float, float], Tuple[float, float, float, float]]):
        self.num_layers = (len(params) - 2) // 3
        self._total_num_params = len(params)

        fixed_mask = []
        bounds = []
        delta_bounds = []
        param_dim = 0

        for param in params:
            if isinstance(param, (float, int)):
                deltas = (0, 0)
                param = (param, param)
                fixed_mask.append(True)
            else:
                param_dim += 1
                fixed_mask.append(False)

                if len(param) == 4:
                    param, deltas = param[:2], param[2:]
                else:
                    max_delta = param[1] - param[0]
                    deltas = (max_delta * self.relative_min_bound_width, max_delta)

            bounds.append(param)
            delta_bounds.append(deltas)

        self.fixed_mask = torch.tensor(fixed_mask).to(self.device)
        self.fitted_mask = ~self.fixed_mask
        self.roughnesses_mask = torch.zeros_like(self.fitted_mask)
        self.roughnesses_mask[self.num_layers: self.num_layers * 2 + 1] = True

        self.min_bounds, self.max_bounds = torch.tensor(bounds).to(self.device).to(self.dtype).T
        self.min_deltas, self.max_deltas = map(torch.atleast_2d, torch.tensor(delta_bounds).to(self.min_bounds).T)
        self._param_dim = param_dim
        self._num_fixed = self.fixed_mask.sum()

        self.fixed_params = self.min_bounds[self.fixed_mask]

    @property
    def param_dim(self) -> int:
        return self._param_dim

    def sample(self, batch_size: int) -> UniformSubPriorParams:
        min_bounds, max_bounds = self.sample_bounds(batch_size)

        params = torch.rand(
            *min_bounds.shape,
            device=self.device,
            dtype=self.dtype
        ) * (max_bounds - min_bounds) + min_bounds

        thicknesses, roughnesses, slds = torch.split(
            params, [self.max_num_layers, self.max_num_layers + 1, self.max_num_layers + 1], dim=-1
        )

        if self.smaller_roughnesses:
            fitted_r_mask = self.fitted_mask.clone()
            fitted_r_mask[~self.roughnesses_mask] = False

            min_roughness = self.min_bounds[fitted_r_mask]
            max_roughness = torch.clamp(
                get_max_allowed_roughness(thicknesses)[..., self.fitted_mask[self.roughnesses_mask]],
                min_roughness,
                self.max_bounds[fitted_r_mask]
            )

            min_vector = self.min_bounds.clone()[None].repeat(batch_size, 1)
            max_vector = self.max_bounds.clone()[None].repeat(batch_size, 1)

            max_vector[..., fitted_r_mask] = max_roughness

            assert torch.all(max_vector[..., fitted_r_mask] == max_roughness)

            min_deltas = self.min_deltas.clone().repeat(batch_size, 1)
            max_deltas = self.max_deltas.clone().repeat(batch_size, 1)

            max_deltas[..., fitted_r_mask] = torch.clamp_max(
                max_deltas[..., fitted_r_mask],
                max_roughness - min_roughness,
            )

            fitted_mask = torch.zeros_like(self.fitted_mask)
            fitted_mask[fitted_r_mask] = True

            updated_min_bounds, updated_max_bounds = self._sample_bounds(
                batch_size, min_vector, max_vector, min_deltas, max_deltas, fitted_mask
            )

            min_bounds[..., fitted_mask], max_bounds[..., fitted_mask] = (
                updated_min_bounds[..., fitted_mask], updated_max_bounds[..., fitted_mask]
            )

            params[..., fitted_mask] = torch.rand(
                batch_size, fitted_mask.sum().item(),
                device=self.device,
                dtype=self.dtype
            ) * (max_bounds[..., fitted_mask] - min_bounds[..., fitted_mask]) + min_bounds[..., fitted_mask]

            thicknesses, roughnesses, slds = torch.split(
                params, [self.max_num_layers, self.max_num_layers + 1, self.max_num_layers + 1], dim=-1
            )

        params = UniformSubPriorParams(thicknesses, roughnesses, slds, min_bounds, max_bounds)

        return params

    def scale_params(self, params: UniformSubPriorParams) -> Tensor:
        params_t = params.as_tensor(add_bounds=False)

        scaled_params = self._scale(params_t, params.min_bounds, params.max_bounds)[..., self.fitted_mask]

        scaled_min_bounds = self._scale(params.min_bounds, self.min_bounds, self.max_bounds)[..., self.fitted_mask]

        scaled_max_bounds = self._scale(params.max_bounds, self.min_bounds, self.max_bounds)[..., self.fitted_mask]

        scaled_params = torch.cat([scaled_params, scaled_min_bounds, scaled_max_bounds], -1)

        return scaled_params

    def scale_bounds(self, bounds: Tensor) -> Tensor:
        return self._scale(bounds, self.min_bounds, self.max_bounds)[..., self.fitted_mask]

    def restore_params(self, scaled_params: Tensor) -> UniformSubPriorParams:
        scaled_params, scaled_min_bounds, scaled_max_bounds = torch.split(
            scaled_params, [self.param_dim, self.param_dim, self.param_dim], dim=1
        )

        min_bounds = self._restore(
            scaled_min_bounds, self.min_bounds[self.fitted_mask], self.max_bounds[self.fitted_mask]
        )
        max_bounds = self._restore(
            scaled_max_bounds, self.min_bounds[self.fitted_mask], self.max_bounds[self.fitted_mask]
        )

        restored_params = self._restore(scaled_params, min_bounds, max_bounds)

        params_t = torch.cat(
            [
                self._cat_restored_with_fixed_vector(restored_params),
                self._cat_restored_with_fixed_vector(min_bounds),
                self._cat_restored_with_fixed_vector(max_bounds),
            ], -1
        )

        params = UniformSubPriorParams.from_tensor(params_t)

        return params

    def _cat_restored_with_fixed_vector(self, restored_t: Tensor) -> Tensor:
        return self._cat_fitted_fixed_t(restored_t, self.fixed_params)

    def _cat_fitted_fixed_t(self, fitted_t: Tensor, fixed_t: Tensor, fitted_mask: Tensor = None) -> Tensor:
        if fitted_mask is None:
            fitted_mask = self.fitted_mask
            fixed_mask = self.fixed_mask
        else:
            fixed_mask = ~fitted_mask

        total_num_params = self.fitted_mask.sum().item() + self.fixed_mask.sum().item()

        batch_size = fitted_t.shape[0]

        concat_t = torch.empty(
            batch_size, total_num_params, device=fitted_t.device, dtype=fitted_t.dtype
        )
        concat_t[:, fitted_mask] = fitted_t
        concat_t[:, fixed_mask] = fixed_t[None].expand(batch_size, -1)

        return concat_t

    def log_prob(self, params: UniformSubPriorParams) -> Tensor:
        log_prob = torch.zeros(params.batch_size, device=params.device, dtype=params.dtype)
        indices = self.get_indices_within_bounds(params)
        log_prob[~indices] = float('-inf')
        return log_prob

    def get_indices_within_bounds(self, params: UniformSubPriorParams) -> Tensor:
        t_params = torch.cat([
            params.thicknesses,
            params.roughnesses,
            params.slds
        ], dim=-1)

        indices = (
                torch.all(t_params >= params.min_bounds, dim=-1) &
                torch.all(t_params <= params.max_bounds, dim=-1)
        )

        return indices

    def clamp_params(self, params: UniformSubPriorParams) -> UniformSubPriorParams:
        params = UniformSubPriorParams.from_tensor(
            torch.cat([
                torch.clamp(
                    params.as_tensor(add_bounds=False),
                    params.min_bounds, params.max_bounds
                ),
                params.min_bounds, params.max_bounds
            ], dim=1)
        )
        return params

    def get_indices_within_domain(self, params: UniformSubPriorParams) -> Tensor:
        return self.get_indices_within_bounds(params)

    def sample_bounds(self, batch_size: int):
        return self._sample_bounds(
            batch_size,
            self.min_bounds,
            self.max_bounds,
            self.min_deltas,
            self.max_deltas,
            self.fitted_mask,
        )

    def _sample_bounds(self, batch_size, min_vector, max_vector, min_deltas, max_deltas, fitted_mask):
        if self.logdist:
            widths_sampler_func = logdist_sampler
        else:
            widths_sampler_func = uniform_sampler

        num_fitted = fitted_mask.sum().item()
        num_fixed = fitted_mask.numel() - num_fitted

        prior_widths = widths_sampler_func(
            min_deltas[..., fitted_mask], max_deltas[..., fitted_mask],
            batch_size, num_fitted,
            device=self.device, dtype=self.dtype
        )

        prior_widths = self._cat_fitted_fixed_t(prior_widths, torch.zeros(num_fixed).to(prior_widths), fitted_mask)

        prior_centers = uniform_sampler(
            min_vector + prior_widths / 2, max_vector - prior_widths / 2,
            *prior_widths.shape,
            device=self.device, dtype=self.dtype
        )

        min_bounds, max_bounds = prior_centers - prior_widths / 2, prior_centers + prior_widths / 2

        return min_bounds, max_bounds

    @staticmethod
    def scale_bounds_with_q(bounds: Tensor, q_ratio: float) -> Tensor:
        params = Params.from_tensor(torch.atleast_2d(bounds).clone())
        params.scale_with_q(q_ratio)
        return params.as_tensor().squeeze()

    def clamp_bounds(self, bounds: Tensor) -> Tensor:
        return torch.clamp(
            torch.atleast_2d(bounds), torch.atleast_2d(self.min_bounds), torch.atleast_2d(self.max_bounds)
        )
