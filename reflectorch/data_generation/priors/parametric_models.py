from typing import Optional, Tuple, Dict, List

import torch
from torch import Tensor

from reflectorch.data_generation.reflectivity import (
    reflectivity,
    abeles_memory_eff,
    kinematical_approximation,
)
from reflectorch.data_generation.utils import (
    get_density_profiles,
    get_param_labels,
    get_param_labels_latex,
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
    """Base class for parameterizations of the SLD profile.

    Args:
        max_num_layers (int): the number of layers
    """
    NAME: str = ''
    PARAMETER_NAMES: Tuple[str, ...]

    def __init__(self, max_num_layers: int, **kwargs):
        self.max_num_layers = max_num_layers
        self._sampler_strategy = self._init_sampler_strategy(**kwargs)

    def _init_sampler_strategy(self, nuisance_params_dim: int = 0, **kwargs):
        return BasicSamplerStrategy(**kwargs)

    @property
    def param_dim(self) -> int:
        """get the number of parameters

        Returns:
            int:
        """
        return len(self.PARAMETER_NAMES)

    @property
    def sampler_strategy(self) -> SamplerStrategy:
        """get the sampler strategy

        Returns:
            SamplerStrategy:
        """
        return self._sampler_strategy

    def reflectivity(self, q, parametrized_model: Tensor, **kwargs) -> Tensor:
        """computes the reflectivity curves

        Args:
            q: the reciprocal space (q) positions
            parametrized_model (Tensor): the values of the parameters

        Returns:
            Tensor: the computed reflectivity curves
        """
        params = self.to_standard_params(parametrized_model)
        return reflectivity(q, **params, **kwargs)
    
    def sld_profile(
        self,
        parametrized_model: torch.Tensor,
        *,
        ambient_sld: Optional[torch.Tensor] = None,
        z_axis: Optional[torch.Tensor] = None,
        num: int = 1000,
        padding_left: float = 0.2,
        padding_right: float = 1.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the continuous SLD profile rho(z) corresponding to `parametrized_model`.

        Returns:
            z_axis: (num,) depth axis
            profile: (B,num) SLD profile
        """
        params = self.to_standard_params(parametrized_model)

        thickness = params["thickness"]
        roughness = params["roughness"]
        sld = params["sld"]

        ambient_sld = ambient_sld.clone().to(thickness) if ambient_sld is not None else None

        z, profile, _ = get_density_profiles(
            thicknesses=thickness,
            roughnesses=roughness,
            slds=sld,
            ambient_sld=ambient_sld,
            z_axis=z_axis,
            num=num,
            padding_left=padding_left,
            padding_right=padding_right,
        )
        return z, profile

    def available_profile_types(self) -> List[str]:
        return ["sld"]

    def profile(
        self,
        parametrized_model: torch.Tensor,
        *,
        profile_type: str = "sld",
        ambient_sld: Optional[torch.Tensor] = None,
        z_axis: Optional[torch.Tensor] = None,
        num: int = 1000,
        padding_left: float = 0.2,
        padding_right: float = 1.1,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        profile_type = str(profile_type).lower()

        if profile_type == "sld":
            return self.sld_profile(
                parametrized_model,
                ambient_sld=ambient_sld,
                z_axis=z_axis,
                num=num,
                padding_left=padding_left,
                padding_right=padding_right,
                **kwargs,
            )

        method = getattr(self, f"{profile_type}_profile", None)
        if method is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support profile_type={profile_type!r}"
            )

        return method(
            parametrized_model,
            z_axis=z_axis,
            num=num,
            padding_left=padding_left,
            padding_right=padding_right,
            **kwargs,
        )

    def supports_zero_ambient_sld_shift(self) -> bool:
        return True

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
        """initializes arrays storing individually the upper and lower bounds from the dictionaries of parameter and bound width ranges 

        Args:
            param_ranges (Dict[str, Tuple[float, float]]): parameter ranges
            bound_width_ranges (Dict[str, Tuple[float, float]]): bound width ranges
            device (optional): the Pytorch device. Defaults to None.
            dtype (optional): the Pytorch datatype. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        ordered_bounds = [param_ranges[k] for k in self.PARAMETER_NAMES]
        delta_bounds = [bound_width_ranges[k] for k in self.PARAMETER_NAMES]

        min_bounds, max_bounds = torch.tensor(ordered_bounds, device=device, dtype=dtype).T[:, None]
        min_deltas, max_deltas = torch.tensor(delta_bounds, device=device, dtype=dtype).T[:, None]

        return min_bounds, max_bounds, min_deltas, max_deltas

    def get_param_labels(self, **kwargs) -> List[str]:
        """get the list with the name of the parameters

        Returns:
            List[str]:
        """
        return list(self.PARAMETER_NAMES)

    def get_param_labels_latex(self, **kwargs) -> List[str]:
        """
        Return LaTeX-formatted parameter labels for plotting.

        Subclasses should override this when a compact symbolic representation is needed.
        """
        return self.get_param_labels(**kwargs)

    def sample(self, batch_size: int,
               total_min_bounds: Tensor,
               total_max_bounds: Tensor,
               total_min_delta: Tensor,
               total_max_delta: Tensor,
               ):
        """samples the parameter values and their prior bounds

        Args:
            batch_size (int): the batch size
            total_min_bounds (Tensor): lower bounds of the parameter ranges
            total_max_bounds (Tensor): upper bounds of the parameter ranges
            total_min_delta (Tensor): lower widths of the subprior intervals
            total_max_delta (Tensor): upper widths of the subprior intervals

        Returns:
            Tensor: sampled parameters
        """
        return self.sampler_strategy.sample(
            batch_size,
            total_min_bounds,
            total_max_bounds,
            total_min_delta,
            total_max_delta,
        )


class StandardModel(ParametricModel):
    """Parameterization for the standard box model. The parameters are the thicknesses, roughnesses and real sld values of the layers."""
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
                               nuisance_params_dim: int = 0,
                               **kwargs):
        if constrained_roughness:
            num_params = self.param_dim + nuisance_params_dim
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

    def get_param_labels(self, **kwargs) -> List[str]:
        return get_param_labels(self.max_num_layers, **kwargs)
    
    def get_param_labels_latex(self, **kwargs) -> List[str]:
        return get_param_labels_latex(
            self.max_num_layers,
            parameterization_type="standard",
            **kwargs,
        )

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
    
    def get_sld_indices(self):
        return slice(2*self.max_num_layers+1, 3*self.max_num_layers+2)
    
    def scale_with_q(self, parametrized_model: Tensor, q_ratio: float) -> Tensor:
        out = parametrized_model.clone()
        out[..., 0:2*self.max_num_layers+1] = out[..., 0:2*self.max_num_layers+1] / q_ratio #thicknesses & roughnesses
        out[..., 2*self.max_num_layers+1:3*self.max_num_layers+2] = out[..., 2*self.max_num_layers+1:3*self.max_num_layers+2] * q_ratio**2 #slds
        
        return out  
    
    def logdet_scale_with_q(self, batch_size: int, q_ratio: Tensor) -> Tensor:
        # thicknesses: n_L dimensions scaled by 1/q_ratio
        # roughnesses: n_L+1 dimensions scaled by 1/q_ratio
        # sls: n_L+1 dimensions scaled by q_ratio**2
        # n_L* (-log(q_ratio)) + (n_L + 1)* (-log(q_ratio)) + (n_L + 1) * (2 * log(q_ratio)) = n_L*log(q_ratio).
        q_ratio = q_ratio.reshape(batch_size)
        return torch.log(q_ratio)
    
    def scale_total_ranges_with_q(
        self,
        min_total_ranges: Tensor,
        max_total_ranges: Tensor,
        q_ratio_min: float,
        q_ratio_max: float,
    ):
        min_out = min_total_ranges.clone()
        max_out = max_total_ranges.clone()

        # thicknesses & roughnesses
        min_out[..., 0:2*self.max_num_layers+1] = (
            min_out[..., 0:2*self.max_num_layers+1] / q_ratio_max
        )
        max_out[..., 0:2*self.max_num_layers+1] = (
            max_out[..., 0:2*self.max_num_layers+1] / q_ratio_min
        )

        # slds
        min_sld = min_out[..., 2*self.max_num_layers+1:3*self.max_num_layers+2]
        max_sld = max_out[..., 2*self.max_num_layers+1:3*self.max_num_layers+2]

        min_out[..., 2*self.max_num_layers+1:3*self.max_num_layers+2] = torch.where(
            min_sld >= 0,
            min_sld * q_ratio_min**2,
            min_sld * q_ratio_max**2,
        )
        max_out[..., 2*self.max_num_layers+1:3*self.max_num_layers+2] = torch.where(
            max_sld >= 0,
            max_sld * q_ratio_max**2,
            max_sld * q_ratio_min**2,
        )

        return min_out, max_out


class ModelWithAbsorption(StandardModel):
    """Parameterization for the box model in which the imaginary sld values of the layers are additional parameters."""
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
                               nuisance_params_dim: int = 0,
                               **kwargs):
        if constrained_roughness:
            num_params = self.param_dim + nuisance_params_dim
            thickness_mask = torch.zeros(num_params, dtype=torch.bool)
            roughness_mask = torch.zeros(num_params, dtype=torch.bool)
            thickness_mask[:self.max_num_layers] = True
            roughness_mask[self.max_num_layers:2 * self.max_num_layers + 1] = True

            if constrained_isld:
                sld_mask = torch.zeros(num_params, dtype=torch.bool)
                isld_mask = torch.zeros(num_params, dtype=torch.bool)
                sld_mask[2 * self.max_num_layers + 1:3 * self.max_num_layers + 2] = True
                isld_mask[3 * self.max_num_layers + 2:4 * self.max_num_layers + 3] = True
                return ConstrainedRoughnessAndImgSldSamplerStrategy(
                    thickness_mask, roughness_mask, sld_mask, isld_mask,
                    max_thickness_share=max_thickness_share, max_sld_share=max_sld_share,
                    **kwargs
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

    def get_param_labels(self, **kwargs) -> List[str]:
        return get_param_labels(self.max_num_layers, parameterization_type='absorption', **kwargs)
    
    def get_param_labels_latex(self, **kwargs) -> List[str]:
        return get_param_labels_latex(
            self.max_num_layers,
            parameterization_type="absorption",
            **kwargs,
        )
    
    @staticmethod
    def _params2dict(parametrized_model: Tensor):
        num_params = parametrized_model.shape[-1]
        num_layers = (num_params - 3) // 4
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
    
    def sld_profile(
        self,
        parametrized_model: torch.Tensor,
        *,
        ambient_sld: Optional[torch.Tensor] = None,
        z_axis: Optional[torch.Tensor] = None,
        num: int = 1000,
        padding_left: float = 0.2,
        padding_right: float = 1.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        """
        params = self.to_standard_params(parametrized_model)

        thickness = params["thickness"]
        roughness = params["roughness"]
        sld = params["sld"].real

        ambient_sld = torch.tensor(ambient_sld).to(thickness) if ambient_sld is not None else None

        z, profile, _ = get_density_profiles(
            thicknesses=thickness,
            roughnesses=roughness,
            slds=sld,
            ambient_sld=ambient_sld,
            z_axis=z_axis,
            num=num,
            padding_left=padding_left,
            padding_right=padding_right,
        )
        return z, profile
    
    def imag_sld_profile(
        self,
        parametrized_model: torch.Tensor,
        *,
        z_axis: Optional[torch.Tensor] = None,
        num: int = 1000,
        padding_left: float = 0.2,
        padding_right: float = 1.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        """
        params = self.to_standard_params(parametrized_model)

        thickness = params["thickness"]
        roughness = params["roughness"]
        sld = params["sld"].imag

        z, profile, _ = get_density_profiles(
            thicknesses=thickness,
            roughnesses=roughness,
            slds=sld,
            z_axis=z_axis,
            num=num,
            padding_left=padding_left,
            padding_right=padding_right,
        )
        return z, profile
    
    def available_profile_types(self) -> List[str]:
        return ["sld", "imag_sld"]

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
    """Parameterization for a thin film composed of repeating identical monolayers, each monolayer consisting of two boxes with distinct SLDs. 
    A sigmoid envelope modulating the SLD profile of the monolayers defines the film thickness and the roughness at the top interface. 
    A second sigmoid envelope can be used to modulate the amplitude of the monolayer SLDs as a function of the displacement from the position of the first sigmoid. 
    These two sigmoids allow one to model a thin film that is coherently ordered up to a certain coherent thickness and gets incoherently ordered or amorphous toward the top of the film.
    In addition, a layer between the substrate and the multilayer (”phase layer”) is introduced to account for the interface structure, 
    which does not necessarily have to be identical to the multilayer period.
    """

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


MULTILAYER_MODELS = {
    'standard_model': StandardModel,
    'model_with_absorption': ModelWithAbsorption,
    'no_fresnel_model': NoFresnelModel,
    'repeating_multilayer_v1': BasicMultilayerModel1,
    'repeating_multilayer_v2': BasicMultilayerModel2,
    'repeating_multilayer_v3': BasicMultilayerModel3,
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


class NuisanceParamsWrapper(ParametricModel):
    """
    Wraps a base model (e.g. StandardModel) to add nuisance parameters, allowing independent enabling/disabling.

    Args:
        base_model (ParametricModel): The base parametric model.
        nuisance_params_config (Dict[str, bool]): Dictionary where keys are parameter names
                                                  and values are `True` (enable) or `False` (disable).
    """

    def __init__(self, base_model: ParametricModel, nuisance_params_config: Dict[str, bool] = None, **kwargs):
        self.base_model = base_model
        self.nuisance_params_config = nuisance_params_config or {}

        self.enabled_nuisance_params = [name for name, is_enabled in self.nuisance_params_config.items() if is_enabled]

        self.PARAMETER_NAMES = self.base_model.PARAMETER_NAMES + tuple(self.enabled_nuisance_params)
        self._param_dim = self.base_model.param_dim + len(self.enabled_nuisance_params)
        
        super().__init__(base_model.max_num_layers, **kwargs)

    def _init_sampler_strategy(self, **kwargs):
        return self.base_model._init_sampler_strategy(nuisance_params_dim=len(self.enabled_nuisance_params), **kwargs)

    @property
    def param_dim(self) -> int:
        return self._param_dim

    def to_standard_params(self, parametrized_model: Tensor) -> dict:
        """Extracts base model parameters only."""
        base_dim = self.base_model.param_dim
        base_part = parametrized_model[..., :base_dim]
        return self.base_model.to_standard_params(base_part)

    def reflectivity(self, q, parametrized_model: Tensor, **kwargs) -> Tensor:
        """Computes reflectivity with optional nuisance parameter shifts."""
        base_dim = self.base_model.param_dim
        base_params = parametrized_model[..., :base_dim]
        nuisance_part = parametrized_model[..., base_dim:]

        nuisance_dict = {param: nuisance_part[..., i].unsqueeze(-1) for i, param in enumerate(self.enabled_nuisance_params)}
        if "log10_background" in nuisance_dict:
            nuisance_dict["background"] = 10 ** nuisance_dict.pop("log10_background")

        return self.base_model.reflectivity(q, base_params, **nuisance_dict, **kwargs)

    def init_bounds(self, param_ranges: Dict[str, Tuple[float, float]],
                    bound_width_ranges: Dict[str, Tuple[float, float]], device=None, dtype=None):
        """Initialize bounds for enabled nuisance parameters."""
        min_bounds_base, max_bounds_base, min_deltas_base, max_deltas_base = self.base_model.init_bounds(
            param_ranges, bound_width_ranges, device, dtype)

        ordered_bounds_nuisance = [param_ranges[k] for k in self.enabled_nuisance_params]
        delta_bounds_nuisance = [bound_width_ranges[k] for k in self.enabled_nuisance_params]

        if ordered_bounds_nuisance:
            min_bounds_nuisance, max_bounds_nuisance = torch.tensor(ordered_bounds_nuisance, device=device, dtype=dtype).T[:, None]
            min_deltas_nuisance, max_deltas_nuisance = torch.tensor(delta_bounds_nuisance, device=device, dtype=dtype).T[:, None]

            min_bounds = torch.cat([min_bounds_base, min_bounds_nuisance], dim=-1)
            max_bounds = torch.cat([max_bounds_base, max_bounds_nuisance], dim=-1)
            min_deltas = torch.cat([min_deltas_base, min_deltas_nuisance], dim=-1)
            max_deltas = torch.cat([max_deltas_base, max_deltas_nuisance], dim=-1)
        else:
            min_bounds, max_bounds, min_deltas, max_deltas = min_bounds_base, max_bounds_base, min_deltas_base, max_deltas_base

        return min_bounds, max_bounds, min_deltas, max_deltas

    def sld_profile(
        self,
        parametrized_model: torch.Tensor,
        *,
        ambient_sld: Optional[torch.Tensor] = None,
        z_axis: Optional[torch.Tensor] = None,
        num: int = 1000,
        padding_left: float = 0.2,
        padding_right: float = 1.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the continuous SLD profile rho(z) corresponding to `parametrized_model`.

        Returns:
            z_axis: (num,) depth axis
            profile: (B,num) SLD profile
        """
        base_dim = self.base_model.param_dim
        base_params = parametrized_model[..., :base_dim]

        return self.base_model.sld_profile(
            parametrized_model=base_params,
            ambient_sld=ambient_sld,
            z_axis=z_axis,
            num=num,
            padding_left=padding_left,
            padding_right=padding_right,
        )
    
    def profile(
        self,
        parametrized_model: torch.Tensor,
        *,
        profile_type: str = "sld",
        ambient_sld: Optional[torch.Tensor] = None,
        z_axis: Optional[torch.Tensor] = None,
        num: int = 1000,
        padding_left: float = 0.2,
        padding_right: float = 1.1,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base_dim = self.base_model.param_dim
        base_params = parametrized_model[..., :base_dim]

        return self.base_model.profile(
            base_params,
            profile_type=profile_type,
            ambient_sld=ambient_sld,
            z_axis=z_axis,
            num=num,
            padding_left=padding_left,
            padding_right=padding_right,
            **kwargs,
        )
    
    def get_sld_indices(self):
        return self.base_model.get_sld_indices()
    
    def get_param_labels(self, **kwargs) -> List[str]:
        return self.base_model.get_param_labels(**kwargs) + self.enabled_nuisance_params

    def get_param_labels(self, **kwargs) -> List[str]:
        names = self.base_model.get_param_labels(**kwargs)

        pretty = {
            "q_shift": "q shift",
            "r_scale": "intensity scale",
            "log10_background": "log10 background",
        }
        names += [pretty.get(name, name) for name in self.enabled_nuisance_params]
        return names

    def get_param_labels_latex(self, **kwargs) -> List[str]:
        names = self.base_model.get_param_labels_latex(**kwargs)

        pretty = {
            "q_shift": r"$q_{sh}$",
            "r_scale": r"$r_{sc}$",
            "log10_background": r"$bkg.$",
        }
        names += [pretty.get(name, rf"${name}$") for name in self.enabled_nuisance_params]
        return names

    def available_profile_types(self) -> List[str]:
        return self.base_model.available_profile_types()
    
    def supports_zero_ambient_sld_shift(self) -> bool:
        return self.base_model.supports_zero_ambient_sld_shift()
    
    def scale_with_q(self, parametrized_model: Tensor, q_ratio: float) -> Tensor:
        parametrized_model = self.base_model.scale_with_q(parametrized_model, q_ratio)

        if 'q_shift' in self.enabled_nuisance_params:
            q_shift_idx = self.base_model.param_dim + self.enabled_nuisance_params.index('q_shift')
            parametrized_model[..., q_shift_idx:q_shift_idx+1] = parametrized_model[..., q_shift_idx:q_shift_idx+1] * q_ratio
        
        return parametrized_model
    
    def logdet_scale_with_q(self, batch_size: int, q_ratio: Tensor) -> Tensor:
        q_ratio = q_ratio.reshape(batch_size)

        logdet = self.base_model.logdet_scale_with_q(batch_size=batch_size, q_ratio=q_ratio)

        if 'q_shift' in self.enabled_nuisance_params:
            logdet = logdet + torch.log(q_ratio)

        return logdet
    
    def scale_total_ranges_with_q(
        self,
        min_total_ranges: Tensor,
        max_total_ranges: Tensor,
        q_ratio_min: float,
        q_ratio_max: float,
    ):
        min_ranges, max_ranges = self.base_model.scale_total_ranges_with_q(
            min_total_ranges, max_total_ranges, q_ratio_min, q_ratio_max
        )

        if 'q_shift' in self.enabled_nuisance_params:
            q_shift_idx = self.base_model.param_dim + self.enabled_nuisance_params.index('q_shift')

            min_q_shift = min_ranges[..., q_shift_idx]
            max_q_shift = max_ranges[..., q_shift_idx]

            min_ranges[..., q_shift_idx] = torch.where(
                min_q_shift >= 0,
                min_q_shift * q_ratio_min,
                min_q_shift * q_ratio_max,
            )
            max_ranges[..., q_shift_idx] = torch.where(
                max_q_shift >= 0,
                max_q_shift * q_ratio_max,
                max_q_shift * q_ratio_min,
            )

        return min_ranges, max_ranges