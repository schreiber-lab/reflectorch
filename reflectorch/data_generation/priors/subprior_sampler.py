from functools import lru_cache
from typing import Tuple

import torch
from torch import Tensor

from reflectorch.data_generation.utils import (
    uniform_sampler,
    logdist_sampler,
    triangular_sampler,
    get_slds_from_d_rhos,
)

from reflectorch.data_generation.priors.params import Params
from reflectorch.data_generation.priors.no_constraints import (
    BasicPriorSampler,
    DEFAULT_ROUGHNESS_RANGE,
    DEFAULT_THICKNESS_RANGE,
    DEFAULT_SLD_RANGE,
    DEFAULT_NUM_LAYERS,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_SCALED_RANGE,
    DEFAULT_USE_DRHO,
)


class UniformSubPriorParams(Params):
    """Parameters class for thicknesses, roughnesses and slds, together with their subprior bounds."""
    __slots__ = ('thicknesses', 'roughnesses', 'slds', 'min_bounds', 'max_bounds')
    PARAM_NAMES = __slots__

    def __init__(self,
                 thicknesses: Tensor,
                 roughnesses: Tensor,
                 slds: Tensor,
                 min_bounds: Tensor,
                 max_bounds: Tensor,
                 ):
        super().__init__(thicknesses, roughnesses, slds)
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds

    @staticmethod
    def rearrange_context_from_params(
            scaled_params: Tensor, context: Tensor, inference: bool = False, from_params: bool = False
    ):
        if inference:
            if from_params:
                num_params = scaled_params.shape[1] // 3
                scaled_params = scaled_params[:, num_params:]
            context = torch.cat([context, scaled_params], dim=-1)
            return context

        num_params = scaled_params.shape[1] // 3
        assert num_params * 3 == scaled_params.shape[1]
        scaled_params, bound_context = torch.split(scaled_params, [num_params, 2 * num_params], dim=-1)
        context = torch.cat([context, bound_context], dim=-1)
        return scaled_params, context

    @staticmethod
    def restore_params_from_context(scaled_params: Tensor, context: Tensor):
        num_params = scaled_params.shape[-1]
        scaled_bounds = context[:, -2 * num_params:]
        scaled_params = torch.cat([scaled_params, scaled_bounds], dim=-1)
        return scaled_params

    @staticmethod
    def input_context_split(t_params):
        num_params = t_params.shape[1] // 3
        return torch.split(t_params, [num_params, 2 * num_params])

    def as_tensor(self, use_drho: bool = False, add_bounds: bool = True) -> Tensor:
        t_list = [self.thicknesses, self.roughnesses]
        if use_drho:
            t_list.append(self.d_rhos)
        else:
            t_list.append(self.slds)
        if add_bounds:
            t_list += [self.min_bounds, self.max_bounds]
        return torch.cat(t_list, -1)

    @classmethod
    def from_tensor(cls, params: Tensor):
        layers_num = (params.shape[-1] - 6) // 9
        num_params = 3 * layers_num + 2

        thicknesses, roughnesses, slds, min_bounds, max_bounds = torch.split(
            params,
            [layers_num, layers_num + 1, layers_num + 1, num_params, num_params],
            dim=-1
        )

        return cls(thicknesses, roughnesses, slds, min_bounds, max_bounds)

    @property
    def num_params(self) -> int:
        return self.layers_num2size(self.max_layer_num)

    @staticmethod
    def size2layers_num(size: int) -> int:
        return (size - 6) // 9

    @staticmethod
    def layers_num2size(layers_num: int) -> int:
        return layers_num * 9 + 6

    def scale_with_q(self, q_ratio: float):
        super().scale_with_q(q_ratio)

        layer_num = self.max_layer_num
        scales = torch.tensor(
            [1 / q_ratio] * (2 * layer_num + 1) + [q_ratio ** 2] * (layer_num + 1),
            device=self.device, dtype=self.dtype
        )

        self.min_bounds *= scales
        self.max_bounds *= scales


class UniformSubPriorSampler(BasicPriorSampler):
    """Prior sampler for thicknesses, roughnesses, slds and their subprior bounds

    Args:
        thickness_range (Tuple[float, float], optional): the range of the layer thicknesses. Defaults to DEFAULT_THICKNESS_RANGE.
        roughness_range (Tuple[float, float], optional): the range of the interlayer roughnesses. Defaults to DEFAULT_ROUGHNESS_RANGE.
        sld_range (Tuple[float, float], optional): the range of the layer SLDs. Defaults to DEFAULT_SLD_RANGE.
        num_layers (int, optional): the number of layers. Defaults to DEFAULT_NUM_LAYERS.
        use_drho (bool, optional): whether to use differences in SLD values between neighboring layers instead of the actual SLD values. Defaults to DEFAULT_USE_DRHO.
        device (torch.device, optional): the Pytorch device. Defaults to DEFAULT_DEVICE.
        dtype (torch.dtype, optional): the Pytorch data type. Defaults to DEFAULT_DTYPE.
        scaled_range (Tuple[float, float], optional): the range for scaling the parameters. Defaults to DEFAULT_SCALED_RANGE.
        scale_by_subpriors (bool, optional): if True the film parameters are scaled with respect to their subprior bounds. Defaults to False.
        smaller_roughnesses (bool, optional): if True the sampled roughnesses are biased towards smaller values. Defaults to False.
        logdist (bool, optional): if True the relative widths of the subprior intervals are sampled uniformly on a logarithmic scale instead of uniformly. Defaults to False.
        relative_min_bound_width (float, optional): defines the interval [relative_min_bound_width, 1.0] from which the relative bound widths for each parameter are sampled. Defaults to 1e-2.
    """
    PARAM_CLS = UniformSubPriorParams

    def __init__(self,
                 thickness_range: Tuple[float, float] = DEFAULT_THICKNESS_RANGE,
                 roughness_range: Tuple[float, float] = DEFAULT_ROUGHNESS_RANGE,
                 sld_range: Tuple[float, float] = DEFAULT_SLD_RANGE,
                 num_layers: int = DEFAULT_NUM_LAYERS,
                 use_drho: bool = DEFAULT_USE_DRHO,
                 device: torch.device = DEFAULT_DEVICE,
                 dtype: torch.dtype = DEFAULT_DTYPE,
                 scaled_range: Tuple[float, float] = DEFAULT_SCALED_RANGE,
                 scale_by_subpriors: bool = False,
                 smaller_roughnesses: bool = False,
                 logdist: bool = False,
                 relative_min_bound_width: float = 1e-2,
                 ):
        super().__init__(
            thickness_range,
            roughness_range,
            sld_range,
            num_layers,
            use_drho,
            device,
            dtype,
            scaled_range,
        )

        self.scale_by_subpriors = scale_by_subpriors
        self.smaller_roughnesses = smaller_roughnesses
        self.logdist = logdist
        self.relative_min_bound_width = relative_min_bound_width

    @property
    def max_num_layers(self) -> int:
        return self.num_layers

    @property
    def param_dim(self) -> int:
        return self.max_num_layers * 3 + 2

    @lru_cache()
    def min_vector(self, layers_num, drho: bool = False):
        min_vector = super().min_vector(layers_num, drho)
        min_vector = torch.cat([min_vector, min_vector, min_vector], dim=0)
        return min_vector

    def scale_params(self, params: UniformSubPriorParams) -> Tensor:
        scaled_params = super().scale_params(params)

        if self.scale_by_subpriors:
            params_t = params.as_tensor(use_drho=self.use_drho, add_bounds=False)
            scaled_params[:, :self.param_dim] = self._scale(params_t, params.min_bounds, params.max_bounds)

        return scaled_params

    def restore_params(self, scaled_params: Tensor) -> Params:
        if not self.scale_by_subpriors:
            return super().restore_params(scaled_params)

        scaled_params, scaled_min_bounds, scaled_max_bounds = torch.split(
            scaled_params, [self.param_dim, self.param_dim, self.param_dim], dim=1
        )

        min_vector = super().min_vector(self.max_num_layers, self.use_drho)
        max_vector = super().max_vector(self.max_num_layers, self.use_drho)

        min_bounds = self._restore(scaled_min_bounds, min_vector, max_vector)
        max_bounds = self._restore(scaled_max_bounds, min_vector, max_vector)

        param_t = self._restore(scaled_params, min_bounds, max_bounds)
        param_t = torch.cat([param_t, min_bounds, max_bounds], dim=-1)

        params = UniformSubPriorParams.from_tensor(param_t)

        if self.use_drho:
            params.slds = get_slds_from_d_rhos(params.slds)
        return params

    @lru_cache()
    def max_vector(self, layers_num, drho: bool = False):
        max_vector = super().max_vector(layers_num, drho)
        max_vector = torch.cat([max_vector, max_vector, max_vector], dim=0)
        return max_vector

    @lru_cache()
    def delta_vector(self, layers_num, drho: bool = False):
        delta_vector = self.max_vector(layers_num, drho) - self.min_vector(layers_num, drho)
        delta_vector[delta_vector == 0.] = 1.
        return delta_vector

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

        params = UniformSubPriorParams(thicknesses, roughnesses, slds, min_bounds, max_bounds)

        return params

    def sample_bounds(self, batch_size: int):
        min_vector, max_vector = (
            super().min_vector(self.num_layers)[None],
            super().max_vector(self.num_layers)[None]
        )

        delta_vector = max_vector - min_vector

        if self.logdist:
            widths_sampler_func = logdist_sampler
        else:
            widths_sampler_func = uniform_sampler

        prior_widths = widths_sampler_func(
            self.relative_min_bound_width, 1.,
            batch_size, delta_vector.shape[1],
            device=self.device, dtype=self.dtype
        ) * delta_vector

        prior_centers = uniform_sampler(
            min_vector + prior_widths / 2, max_vector - prior_widths / 2,
            *prior_widths.shape,
            device=self.device, dtype=self.dtype
        )

        if self.smaller_roughnesses:
            idx_min, idx_max = self.num_layers, self.num_layers * 2 + 1
            prior_centers[:, idx_min:idx_max] = triangular_sampler(
                min_vector[:, idx_min:idx_max] + prior_widths[:, idx_min:idx_max] / 2,
                max_vector[:, idx_min:idx_max] - prior_widths[:, idx_min:idx_max] / 2,
                batch_size, self.num_layers + 1,
                device=self.device, dtype=self.dtype
            )

        min_bounds, max_bounds = prior_centers - prior_widths / 2, prior_centers + prior_widths / 2

        return min_bounds, max_bounds

    def scale_bounds(self, bounds: Tensor) -> Tensor:
        layers_num = bounds.shape[-1] // 2

        return self._scale(
            bounds,
            self.min_vector(layers_num, drho=self.use_drho).to(bounds),
            self.max_vector(layers_num, drho=self.use_drho).to(bounds),
        )


class NarrowSldUniformSubPriorSampler(UniformSubPriorSampler):
    """Prior sampler for thicknesses, roughnesses, slds and their subprior bounds. The subprior bound widths for SLDs are restricted to be lower than a specified value. """
    def __init__(self,
                 thickness_range: Tuple[float, float] = DEFAULT_THICKNESS_RANGE,
                 roughness_range: Tuple[float, float] = DEFAULT_ROUGHNESS_RANGE,
                 sld_range: Tuple[float, float] = DEFAULT_SLD_RANGE,
                 num_layers: int = DEFAULT_NUM_LAYERS,
                 use_drho: bool = DEFAULT_USE_DRHO,
                 device: torch.device = DEFAULT_DEVICE,
                 dtype: torch.dtype = DEFAULT_DTYPE,
                 scaled_range: Tuple[float, float] = DEFAULT_SCALED_RANGE,
                 scale_by_subpriors: bool = False,
                 max_sld_prior_width: float = 10.,
                 ):
        super().__init__(
            thickness_range,
            roughness_range,
            sld_range,
            num_layers,
            use_drho,
            device,
            dtype,
            scaled_range,
            scale_by_subpriors,
        )

        self.max_sld_prior_width = max_sld_prior_width

    def sample_bounds(self, batch_size: int):
        min_vector, max_vector = (
            BasicPriorSampler.min_vector(self, self.num_layers),
            BasicPriorSampler.max_vector(self, self.num_layers),
        )

        delta_vector = max_vector - min_vector
        delta_vector[-self.num_layers:] = self.max_sld_prior_width

        prior_widths = uniform_sampler(
            delta_vector * self.relative_min_bound_width, delta_vector,
            batch_size, min_vector.shape[0],
            device=self.device, dtype=self.dtype
        )

        prior_centers = uniform_sampler(
            min_vector + prior_widths / 2, max_vector - prior_widths / 2,
            *prior_widths.shape,
            device=self.device, dtype=self.dtype
        )

        return prior_centers - prior_widths / 2, prior_centers + prior_widths / 2
