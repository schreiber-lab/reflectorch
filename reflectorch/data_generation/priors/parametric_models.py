from typing import Tuple, Dict, List

import torch
from torch import Tensor

from reflectorch.data_generation.reflectivity import (
    reflectivity,
    abeles_memory_eff,
    kinematical_approximation,
)
from reflectorch.data_generation.utils import (
    get_param_labels,
)
from reflectorch.data_generation.priors.sampler_strategies import (
    SamplerStrategy,
    BasicSamplerStrategy,
    ConstrainedRoughnessSamplerStrategy,
    ConstrainedRoughnessAndImgSldSamplerStrategy,
)

__all__ = [
    "MULTILAYER_MODELS",
    "ParametricModel",
]


class ParametricModel(object):
    NAME: str = ''
    PARAMETER_NAMES: Tuple[str, ...]

    def __init__(self, max_num_layers: int, **kwargs):
        self.max_num_layers = max_num_layers
        self._sampler_strategy = self._init_sampler_strategy(**kwargs)

    def _init_sampler_strategy(self, **kwargs):
        return BasicSamplerStrategy(**kwargs)

    @property
    def param_dim(self) -> int:
        return len(self.PARAMETER_NAMES)

    @property
    def sampler_strategy(self) -> SamplerStrategy:
        return self._sampler_strategy

    def reflectivity(self, q, parametrized_model: Tensor, **kwargs) -> Tensor:
        params = self.to_standard_params(parametrized_model)
        return reflectivity(q, **params, **kwargs)

    def to_standard_params(self, parametrized_model: Tensor) -> dict:
        raise NotImplementedError

    def from_standard_params(self, params: dict) -> Tensor:
        raise NotImplementedError

    def scale_with_q(self, parametrized_model: Tensor, q_ratio: float) -> Tensor:
        raise NotImplementedError

    def init_bounds(self,
                    param_ranges: Dict[str, Tuple[float, float]],
                    bound_width_ranges: Dict[str, Tuple[float, float]],
                    device=None,
                    dtype=None,
                    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        ordered_bounds = [param_ranges[k] for k in self.PARAMETER_NAMES]
        delta_bounds = [bound_width_ranges[k] for k in self.PARAMETER_NAMES]

        min_bounds, max_bounds = torch.tensor(ordered_bounds, device=device, dtype=dtype).T[:, None]
        min_deltas, max_deltas = torch.tensor(delta_bounds, device=device, dtype=dtype).T[:, None]

        return min_bounds, max_bounds, min_deltas, max_deltas

    def get_param_labels(self) -> List[str]:
        return list(self.PARAMETER_NAMES)

    def sample(self, batch_size: int,
               total_min_bounds: Tensor,
               total_max_bounds: Tensor,
               total_min_delta: Tensor,
               total_max_delta: Tensor,
               ):
        return self.sampler_strategy.sample(
            batch_size,
            total_min_bounds,
            total_max_bounds,
            total_min_delta,
            total_max_delta,
        )


class StandardModel(ParametricModel):
    NAME = 'standard_model'

    PARAMETER_NAMES = (
        "thicknesses",
        "roughnesses",
        "slds",
    )

    @property
    def param_dim(self) -> int:
        return 3 * self.max_num_layers + 2

    def _init_sampler_strategy(self,
                               constrained_roughness: bool = True,
                               max_thickness_share: float = 0.5,
                               **kwargs):
        if constrained_roughness:
            num_params = self.param_dim
            thickness_mask = torch.zeros(num_params, dtype=torch.bool)
            roughness_mask = torch.zeros(num_params, dtype=torch.bool)
            thickness_mask[:self.max_num_layers] = True
            roughness_mask[self.max_num_layers:2 * self.max_num_layers + 1] = True
            return ConstrainedRoughnessSamplerStrategy(
                thickness_mask, roughness_mask,
                max_thickness_share=max_thickness_share,
                **kwargs
            )
        else:
            return BasicSamplerStrategy(**kwargs)

    def to_standard_params(self, parametrized_model: Tensor) -> dict:
        return self._params2dict(parametrized_model)

    def init_bounds(self,
                    param_ranges: Dict[str, Tuple[float, float]],
                    bound_width_ranges: Dict[str, Tuple[float, float]],
                    device=None,
                    dtype=None,
                    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        other_ranges = [param_ranges[k] for k in self.PARAMETER_NAMES[3:]]
        other_delta_bounds = [bound_width_ranges[k] for k in self.PARAMETER_NAMES[3:]]

        ordered_bounds = (
                [param_ranges["thicknesses"]] * self.max_num_layers +
                [param_ranges["roughnesses"]] * (self.max_num_layers + 1) +
                [param_ranges["slds"]] * (self.max_num_layers + 1) +
                other_ranges
        )
        delta_bounds = (
                [bound_width_ranges["thicknesses"]] * self.max_num_layers +
                [bound_width_ranges["roughnesses"]] * (self.max_num_layers + 1) +
                [bound_width_ranges["slds"]] * (self.max_num_layers + 1) +
                other_delta_bounds
        )

        min_bounds, max_bounds = torch.tensor(ordered_bounds, device=device, dtype=dtype).T[:, None]
        min_deltas, max_deltas = torch.tensor(delta_bounds, device=device, dtype=dtype).T[:, None]

        return min_bounds, max_bounds, min_deltas, max_deltas

    def get_param_labels(self) -> List[str]:
        return get_param_labels(self.max_num_layers)

    @staticmethod
    def _params2dict(parametrized_model: Tensor):
        num_params = parametrized_model.shape[-1]
        num_layers = (num_params - 2) // 3
        assert num_layers * 3 + 2 == num_params

        d, sigma, sld = torch.split(
            parametrized_model, [num_layers, num_layers + 1, num_layers + 1], -1
        )
        params = dict(
            thickness=d,
            roughness=sigma,
            sld=sld
        )

        return params

    def reflectivity(self, q, parametrized_model: Tensor, **kwargs) -> Tensor:
        return reflectivity(
            q, **self._params2dict(parametrized_model), **kwargs
        )


class ModelWithAbsorption(StandardModel):
    NAME = 'model_with_absorption'

    PARAMETER_NAMES = (
        "thicknesses",
        "roughnesses",
        "slds",
        "islds",
    )

    @property
    def param_dim(self) -> int:
        return 4 * self.max_num_layers + 3
    
    def _init_sampler_strategy(self,
                               constrained_roughness: bool = True,
                               constrained_isld: bool = True,
                               max_thickness_share: float = 0.5,
                               max_sld_share: float = 0.2,
                               **kwargs):
        if constrained_roughness:
            num_params = self.param_dim
            thickness_mask = torch.zeros(num_params, dtype=torch.bool)
            roughness_mask = torch.zeros(num_params, dtype=torch.bool)
            thickness_mask[:self.max_num_layers] = True
            roughness_mask[self.max_num_layers:2 * self.max_num_layers + 1] = True

            if constrained_isld:
                sld_mask = torch.zeros(num_params, dtype=torch.bool)
                isld_mask = torch.zeros(num_params, dtype=torch.bool)
                sld_mask[2 * self.max_num_layers + 1:3 * self.max_num_layers + 2] = True
                isld_mask[3 * self.max_num_layers + 2:] = True
                return ConstrainedRoughnessAndImgSldSamplerStrategy(
                    thickness_mask, roughness_mask, sld_mask, isld_mask,
                    max_thickness_share=max_thickness_share, max_sld_share=max_sld_share
                )
            else:
                return ConstrainedRoughnessSamplerStrategy(
                    thickness_mask, roughness_mask,
                    max_thickness_share=max_thickness_share,
                    **kwargs
            )
        else:
            return BasicSamplerStrategy(**kwargs)

    def init_bounds(self,
                    param_ranges: Dict[str, Tuple[float, float]],
                    bound_width_ranges: Dict[str, Tuple[float, float]],
                    device=None,
                    dtype=None,
                    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        other_ranges = [param_ranges[k] for k in self.PARAMETER_NAMES[4:]]
        other_delta_bounds = [bound_width_ranges[k] for k in self.PARAMETER_NAMES[4:]]

        ordered_bounds = (
                [param_ranges["thicknesses"]] * self.max_num_layers +
                [param_ranges["roughnesses"]] * (self.max_num_layers + 1) +
                [param_ranges["slds"]] * (self.max_num_layers + 1) +
                [param_ranges["islds"]] * (self.max_num_layers + 1) +
                other_ranges
        )
        delta_bounds = (
                [bound_width_ranges["thicknesses"]] * self.max_num_layers +
                [bound_width_ranges["roughnesses"]] * (self.max_num_layers + 1) +
                [bound_width_ranges["slds"]] * (self.max_num_layers + 1) +
                [bound_width_ranges["islds"]] * (self.max_num_layers + 1) +
                other_delta_bounds
        )

        min_bounds, max_bounds = torch.tensor(ordered_bounds, device=device, dtype=dtype).T[:, None]
        min_deltas, max_deltas = torch.tensor(delta_bounds, device=device, dtype=dtype).T[:, None]

        return min_bounds, max_bounds, min_deltas, max_deltas

    def get_param_labels(self) -> List[str]:
        raise NotImplementedError
        # return get_param_labels(self.max_num_layers)

    @staticmethod
    def _params2dict(parametrized_model: Tensor):
        num_params = parametrized_model.shape[-1]
        num_layers = (num_params - 4) // 3
        assert num_layers * 4 + 3 == num_params

        d, sigma, sld, isld = torch.split(
            parametrized_model, [num_layers, num_layers + 1, num_layers + 1, num_layers + 1], -1
        )
        params = dict(
            thickness=d,
            roughness=sigma,
            sld=sld + 1j * isld
        )

        return params

    def reflectivity(self, q, parametrized_model: Tensor, **kwargs) -> Tensor:
        return reflectivity(
            q, **self._params2dict(parametrized_model), **kwargs
        )


class ModelWithShifts(StandardModel):
    NAME = 'model_with_shifts'

    PARAMETER_NAMES = (
        "thicknesses",
        "roughnesses",
        "slds",
        "q_shift",
        "norm_shift",
    )

    @property
    def param_dim(self) -> int:
        return 3 * self.max_num_layers + 4

    def to_standard_params(self, parametrized_model: Tensor) -> dict:
        params = self._params2dict(parametrized_model)
        params.pop('q_shift')
        params.pop('norm_shift')

        return params

    def get_param_labels(self) -> List[str]:
        return get_param_labels(self.max_num_layers) + [r"$\Delta q$ (Å$^{{-1}}$)", r"$\Delta I$"]

    @staticmethod
    def _params2dict(parametrized_model: Tensor):
        num_params = parametrized_model.shape[-1]
        num_layers = (num_params - 4) // 3
        assert num_layers * 3 + 4 == num_params

        d, sigma, sld, q_shift, norm_shift = torch.split(
            parametrized_model, [num_layers, num_layers + 1, num_layers + 1, 1, 1], -1
        )
        params = dict(
            thickness=d,
            roughness=sigma,
            sld=sld,
            q_shift=q_shift,
            norm_shift=norm_shift,
        )

        return params

    def reflectivity(self, q, parametrized_model: Tensor, **kwargs) -> Tensor:
        return reflectivity_with_shifts(
            q, **self._params2dict(parametrized_model), **kwargs
        )


class NoFresnelModel(StandardModel):
    NAME = 'no_fresnel_model'

    def reflectivity(self, q, parametrized_model: Tensor, **kwargs) -> Tensor:
        return kinematical_approximation(
            q, **self._params2dict(parametrized_model), apply_fresnel=False, **kwargs
        )


class BasicMultilayerModel1(ParametricModel):
    NAME = 'repeating_multilayer_v1'

    PARAMETER_NAMES = (
        "d_full_rel",
        "rel_sigmas",
        "d_block",
        "s_block_rel",
        "r_block",
        "dr",
        "d3_rel",
        "s3_rel",
        "r3",
        "d_sio2",
        "s_sio2",
        "s_si",
        "r_sio2",
        "r_si",
    )

    def to_standard_params(self, parametrized_model: Tensor) -> dict:
        return multilayer_model1(parametrized_model, self.max_num_layers)

    def reflectivity(self, q, parametrized_model: Tensor, **kwargs) -> Tensor:
        params = self.to_standard_params(parametrized_model)
        return reflectivity(q, abeles_func=abeles_memory_eff, **params, **kwargs)


class BasicMultilayerModel2(BasicMultilayerModel1):
    NAME = 'repeating_multilayer_v2'

    PARAMETER_NAMES = (
        "d_full_rel",
        "rel_sigmas",
        "dr_sigmoid_rel_pos",
        "dr_sigmoid_rel_width",
        "d_block",
        "s_block_rel",
        "r_block",
        "dr",
        "d3_rel",
        "s3_rel",
        "r3",
        "d_sio2",
        "s_sio2",
        "s_si",
        "r_sio2",
        "r_si",
    )

    def to_standard_params(self, parametrized_model: Tensor) -> dict:
        return multilayer_model2(parametrized_model, self.max_num_layers)


class BasicMultilayerModel3(BasicMultilayerModel1):
    NAME = 'repeating_multilayer_v3'

    PARAMETER_NAMES = (
        "d_full_rel",
        "rel_sigmas",
        "dr_sigmoid_rel_pos",
        "dr_sigmoid_rel_width",
        "d_block1_rel",
        "d_block",
        "s_block_rel",
        "r_block",
        "dr",
        "d3_rel",
        "s3_rel",
        "r3",
        "d_sio2",
        "s_sio2",
        "s_si",
        "r_sio2",
        "r_si",
    )

    def to_standard_params(self, parametrized_model: Tensor) -> dict:
        return multilayer_model3(parametrized_model, self.max_num_layers)


class MultilayerModel1WithShifts(BasicMultilayerModel1):
    NAME = 'repeating_multilayer_v1_with_shifts'

    PARAMETER_NAMES = (
        "d_full_rel",
        "rel_sigmas",
        "d_block",
        "s_block_rel",
        "r_block",
        "dr",
        "d3_rel",
        "s3_rel",
        "r3",
        "d_sio2",
        "s_sio2",
        "s_si",
        "r_sio2",
        "r_si",
        "q_shift",
        "norm_shift",
    )

    def reflectivity(self, q, parametrized_model: Tensor, **kwargs) -> Tensor:
        q_shift, norm_shift = parametrized_model[..., -2:].T[..., None]
        return reflectivity_with_shifts(
            q, q_shift=q_shift, norm_shift=norm_shift, abeles_func=abeles_memory_eff,
            **self.to_standard_params(parametrized_model), **kwargs
        )


class MultilayerModel3WithShifts(BasicMultilayerModel3):
    NAME = 'repeating_multilayer_v3_with_shifts'

    PARAMETER_NAMES = (
        "d_full_rel",
        "rel_sigmas",
        "dr_sigmoid_rel_pos",
        "dr_sigmoid_rel_width",
        "d_block1_rel",
        "d_block",
        "s_block_rel",
        "r_block",
        "dr",
        "d3_rel",
        "s3_rel",
        "r3",
        "d_sio2",
        "s_sio2",
        "s_si",
        "r_sio2",
        "r_si",
        "q_shift",
        "norm_shift",
    )

    def reflectivity(self, q, parametrized_model: Tensor, **kwargs) -> Tensor:
        q_shift, norm_shift = parametrized_model[..., -2:].T[..., None]
        return reflectivity_with_shifts(
            q, q_shift=q_shift, norm_shift=norm_shift, abeles_func=abeles_memory_eff,
            **self.to_standard_params(parametrized_model), **kwargs
        )


def reflectivity_with_shifts(q, thickness, roughness, sld, q_shift, norm_shift, **kwargs):
    q = torch.atleast_2d(q) + q_shift
    return reflectivity(q, thickness, roughness, sld, **kwargs) * norm_shift


MULTILAYER_MODELS = {
    'standard_model': StandardModel,
    'model_with_absorption': ModelWithAbsorption,
    'model_with_shifts': ModelWithShifts,
    'no_fresnel_model': NoFresnelModel,
    'repeating_multilayer_v1': BasicMultilayerModel1,
    'repeating_multilayer_v2': BasicMultilayerModel2,
    'repeating_multilayer_v3': BasicMultilayerModel3,
    'repeating_multilayer_v1_with_shifts': MultilayerModel1WithShifts,
    'repeating_multilayer_v3_with_shifts': MultilayerModel3WithShifts,
}


def multilayer_model1(parametrized_model: Tensor, d_full_rel_max: int = 30) -> dict:
    n = d_full_rel_max

    (
        d_full_rel,
        rel_sigmas,
        d_block,
        s_block_rel,
        r_block,
        dr,
        d3_rel,
        s3_rel,
        r3,
        d_sio2,
        s_sio2,
        s_si,
        r_sio2,
        r_si,
        *_,
    ) = parametrized_model.T

    batch_size = parametrized_model.shape[0]

    r_positions = 2 * n - torch.arange(2 * n, dtype=dr.dtype, device=dr.device)[None].repeat(batch_size, 1)

    r_modulations = torch.sigmoid(-(r_positions - 2 * d_full_rel[..., None]) / rel_sigmas[..., None])

    r_block = r_block[:, None].repeat(1, n)
    dr = dr[:, None].repeat(1, n)

    sld_blocks = torch.stack([r_block, r_block + dr], -1).flatten(1)

    sld_blocks = r_modulations * sld_blocks

    d3 = d3_rel * d_block

    thicknesses = torch.cat(
        [(d_block / 2)[:, None].repeat(1, n * 2), d3[:, None], d_sio2[:, None]], -1
    )

    s_block = s_block_rel * d_block

    roughnesses = torch.cat(
        [s_block[:, None].repeat(1, n * 2), (s3_rel * d3)[:, None], s_sio2[:, None], s_si[:, None]], -1
    )

    slds = torch.cat(
        [sld_blocks, r3[:, None], r_sio2[:, None], r_si[:, None]], -1
    )

    params = dict(
        thickness=thicknesses,
        roughness=roughnesses,
        sld=slds
    )
    return params


def multilayer_model2(parametrized_model: Tensor, d_full_rel_max: int = 30) -> dict:
    n = d_full_rel_max

    (
        d_full_rel,
        rel_sigmas,
        dr_sigmoid_rel_pos,
        dr_sigmoid_rel_width,
        d_block,
        s_block_rel,
        r_block,
        dr,
        d3_rel,
        s3_rel,
        r3,
        d_sio2,
        s_sio2,
        s_si,
        r_sio2,
        r_si,
        *_,
    ) = parametrized_model.T

    batch_size = parametrized_model.shape[0]

    r_positions = 2 * n - torch.arange(2 * n, dtype=dr.dtype, device=dr.device)[None].repeat(batch_size, 1)

    r_modulations = torch.sigmoid(-(r_positions - 2 * d_full_rel[..., None]) / rel_sigmas[..., None])

    r_block = r_block[:, None].repeat(1, n)
    dr = dr[:, None].repeat(1, n)

    dr_positions = r_positions[:, ::2]

    dr_modulations = torch.sigmoid(
        -(dr_positions - (2 * d_full_rel * dr_sigmoid_rel_pos)[..., None]) / dr_sigmoid_rel_width[..., None]
    )

    dr = dr * dr_modulations

    sld_blocks = torch.stack([r_block, r_block + dr], -1).flatten(1)

    sld_blocks = r_modulations * sld_blocks

    d3 = d3_rel * d_block

    thicknesses = torch.cat(
        [(d_block / 2)[:, None].repeat(1, n * 2), d3[:, None], d_sio2[:, None]], -1
    )

    s_block = s_block_rel * d_block

    roughnesses = torch.cat(
        [s_block[:, None].repeat(1, n * 2), (s3_rel * d3)[:, None], s_sio2[:, None], s_si[:, None]], -1
    )

    slds = torch.cat(
        [sld_blocks, r3[:, None], r_sio2[:, None], r_si[:, None]], -1
    )

    params = dict(
        thickness=thicknesses,
        roughness=roughnesses,
        sld=slds
    )
    return params


def multilayer_model3(parametrized_model: Tensor, d_full_rel_max: int = 30):
    n = d_full_rel_max

    (
        d_full_rel,
        rel_sigmas,
        dr_sigmoid_rel_pos,
        dr_sigmoid_rel_width,
        d_block1_rel,
        d_block,
        s_block_rel,
        r_block,
        dr,
        d3_rel,
        s3_rel,
        r3,
        d_sio2,
        s_sio2,
        s_si,
        r_sio2,
        r_si,
        *_,
    ) = parametrized_model.T

    batch_size = parametrized_model.shape[0]

    r_positions = 2 * n - torch.arange(2 * n, dtype=dr.dtype, device=dr.device)[None].repeat(batch_size, 1)

    r_modulations = torch.sigmoid(
        -(
                r_positions - 2 * d_full_rel[..., None]
        ) / rel_sigmas[..., None]
    )

    dr_positions = r_positions[:, ::2]

    dr_modulations = dr[..., None] * (1 - torch.sigmoid(
        -(
                dr_positions - 2 * d_full_rel[..., None] + 2 * dr_sigmoid_rel_pos[..., None]
        ) / dr_sigmoid_rel_width[..., None]
    ))

    r_block = r_block[..., None].repeat(1, n)
    dr = dr[..., None].repeat(1, n)

    sld_blocks = torch.stack(
        [
            r_block + dr_modulations * (1 - d_block1_rel[..., None]),
            r_block + dr - dr_modulations * d_block1_rel[..., None]
        ], -1).flatten(1)

    sld_blocks = r_modulations * sld_blocks

    d3 = d3_rel * d_block

    d1, d2 = d_block * d_block1_rel, d_block * (1 - d_block1_rel)

    thickness_blocks = torch.stack([d1[:, None].repeat(1, n), d2[:, None].repeat(1, n)], -1).flatten(1)

    thicknesses = torch.cat(
        [thickness_blocks, d3[:, None], d_sio2[:, None]], -1
    )

    s_block = s_block_rel * d_block

    roughnesses = torch.cat(
        [s_block[:, None].repeat(1, n * 2), (s3_rel * d3)[:, None], s_sio2[:, None], s_si[:, None]], -1
    )

    slds = torch.cat(
        [sld_blocks, r3[:, None], r_sio2[:, None], r_si[:, None]], -1
    )

    params = dict(
        thickness=thicknesses,
        roughness=roughnesses,
        sld=slds
    )
    return params
