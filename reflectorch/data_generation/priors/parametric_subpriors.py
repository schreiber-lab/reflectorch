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
    NuisanceParamsWrapper,
    ParametricModel,
)
from reflectorch.data_generation.priors.scaler_mixin import ScalerMixin


class BasicParams(AbstractParams):
    """Parameter class compatible with different parameterizations of the SLD profile. It stores the parameters as well as their minimum and maximum subprior bounds.

        Args:
            parameters (Tensor): the values of the thin film parameters
            min_bounds (Tensor): the minimum subprior bounds of the parameters
            max_bounds (Tensor): the maximum subprior bounds of the parameters
            max_num_layers (int, optional): the maximum number of layers (for box model parameterizations it is the number of layers). Defaults to None.
            param_model (ParametricModel, optional): the parametric model. Defaults to the box model parameterization with number of layers given by max_num_layers.
        """
    
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

    def get_param_labels(self, **kwargs) -> List[str]:
        """gets the parameter labels"""
        return self.param_model.get_param_labels(**kwargs)

    def reflectivity(self, q: Tensor, log: bool = False, **kwargs):
        r"""computes the reflectivity curves directly from the parameters

        Args:
            q (Tensor): the q values
            log (bool, optional): whether to apply logarithm to the curves. Defaults to False.

        Returns:
            Tensor: the simulated reflectivity curves
        """
        return self.param_model.reflectivity(q, self.parameters, log=log, **kwargs)

    @property
    def max_layer_num(self) -> int:  # keep for back compatibility but TODO: unify api among different params
        """gets the maximum number of layers"""
        return self.max_num_layers

    @property
    def num_params(self) -> int:
        """get the number of parameters (parameter dimensionality)"""
        return self.param_model.param_dim

    @property
    def thicknesses(self):
        """gets the thicknesses"""
        params = self.param_model.to_standard_params(self.parameters)
        return params['thickness']

    @property
    def roughnesses(self):
        """gets the roughnesses"""
        params = self.param_model.to_standard_params(self.parameters)
        return params['roughness']

    @property
    def slds(self):
        """gets the slds"""
        params = self.param_model.to_standard_params(self.parameters)
        return params['sld']
        
    @property
    def real_slds(self):
        """gets the real part of the slds"""
        params = self.param_model.to_standard_params(self.parameters)
        return params['sld'].real
    
    @property
    def imag_slds(self):
        """gets the imaginary part of the slds (only for complex dtypes)"""
        params = self.param_model.to_standard_params(self.parameters)
        return params['sld'].imag

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
        """converts the instance of the class to a Pytorch tensor

        Args:
            add_bounds (bool, optional): whether to add the subprior bounds to the tensor. Defaults to True.

        Returns:
            Tensor: the Pytorch tensor obtained from the instance of the class
        """
        if not add_bounds:
            return self.parameters
        return torch.cat([self.parameters, self.min_bounds, self.max_bounds], -1)

    @classmethod
    def from_tensor(cls, params: Tensor, **kwargs):
        """initializes an instance of the class from a Pytorch tensor

        Args:
            params (Tensor): Pytorch tensor containing the parameter values, min subprior bounds and max subprior bounds

        Returns:
            BasicParams: the instance of the class
        """
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
        """scales the parameters based on the q ratio

        Args:
            q_ratio (float): the scaling ratio
        """
        self.parameters = self.param_model.scale_with_q(self.parameters, q_ratio)
        self.min_bounds = self.param_model.scale_with_q(self.min_bounds, q_ratio)
        self.max_bounds = self.param_model.scale_with_q(self.max_bounds, q_ratio)


class SubpriorParametricSampler(PriorSampler, ScalerMixin):
    PARAM_CLS = BasicParams

    def __init__(self,
                 param_ranges: Dict[str, Tuple[float, float]],
                 bound_width_ranges: Dict[str, Tuple[float, float]],
                 model_name: str,
                 device: torch.device = DEFAULT_DEVICE,
                 dtype: torch.dtype = DEFAULT_DTYPE,
                 max_num_layers: int = 50,
                 logdist: bool = False,
                 scale_params_by_ranges = False,
                 scaled_range: Tuple[float, float] = (-1., 1.),
                 **kwargs
                 ):
        """Prior sampler for the parameters of a parametric model and their subprior bounds

        Args:
            param_ranges (Dict[str, Tuple[float, float]]): dictionary containing the name of each type of parameter together with its range
            bound_width_ranges (Dict[str, Tuple[float, float]]): dictionary containing the name of each type of parameter together with the range for sampling the widths of the subprior interval
            model_name (str): the name of the parametric model
            device (torch.device, optional): the Pytorch device. Defaults to DEFAULT_DEVICE.
            dtype (torch.dtype, optional): the Pytorch data type. Defaults to DEFAULT_DTYPE.
            max_num_layers (int, optional): the maximum number of layers (for box model parameterizations it is the number of layers). Defaults to 50.
            logdist (bool, optional): if True the relative widths of the subprior intervals are sampled uniformly on a logarithmic scale instead of uniformly. Defaults to False.
            scale_params_by_ranges (bool, optional): if True the parameters are scaled with respect to their ranges instead of being scaled with respect to their prior bounds. Defaults to False.
            scaled_range (Tuple[float, float], optional): the range for scaling the parameters. Defaults to (-1., 1.)
        """
        self.scaled_range = scaled_range

        self.shift_param_config = kwargs.pop('shift_param_config', {})

        base_model: ParametricModel = MULTILAYER_MODELS[model_name](max_num_layers, logdist=logdist, **kwargs)
        if any(self.shift_param_config.values()):
            self.param_model = NuisanceParamsWrapper(
                base_model=base_model,
                nuisance_params_config=self.shift_param_config,
                **kwargs,
            )
        else:
            self.param_model = base_model

        self.device = device
        self.dtype = dtype
        self.num_layers = max_num_layers

        self.PARAM_CLS.PARAM_MODEL_CLS = MULTILAYER_MODELS[model_name]
        self.PARAM_CLS.MAX_NUM_LAYERS = max_num_layers

        self._param_dim = self.param_model.param_dim
        self.min_bounds, self.max_bounds, self.min_delta, self.max_delta = self.param_model.init_bounds(
            param_ranges, bound_width_ranges, device=device, dtype=dtype
        )
        
        self.param_ranges = param_ranges
        self.bound_width_ranges = bound_width_ranges
        self.model_name = model_name
        self.logdist = logdist
        self.scale_params_by_ranges = scale_params_by_ranges

    @property
    def max_num_layers(self) -> int:
        """gets the maximum number of layers"""
        return self.num_layers

    @property
    def param_dim(self) -> int:
        """get the number of parameters (parameter dimensionality)"""
        return self._param_dim

    def sample(self, batch_size: int) -> BasicParams:
        """sample a batch of parameters

        Args:
            batch_size (int): the batch size

        Returns:
            BasicParams: sampled parameters
        """
        params, min_bounds, max_bounds = self.param_model.sample(
            batch_size, self.min_bounds, self.max_bounds, self.min_delta, self.max_delta
        )

        params = BasicParams(
            parameters=params,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            max_num_layers=self.max_num_layers,
            param_model=self.param_model,
        )

        return params

    def scale_params(self, params: BasicParams) -> Tensor:
        """scale the parameters to a ML-friendly range

        Args:
            params (BasicParams): the parameters to be scaled

        Returns:
            Tensor: the scaled parameters
        """
        if self.scale_params_by_ranges:
            scaled_params = torch.cat([
                self._scale(params.parameters, self.min_bounds, self.max_bounds), #parameters and subprior bounds are scaled with respect to the parameter ranges
                self._scale(params.min_bounds, self.min_bounds, self.max_bounds), 
                self._scale(params.max_bounds, self.min_bounds, self.max_bounds),
            ], -1)
            return scaled_params
        else:
            scaled_params = torch.cat([
                self._scale(params.parameters, params.min_bounds, params.max_bounds), #each parameter scaled with respect to its subprior bounds
                self._scale(params.min_bounds, self.min_bounds, self.max_bounds), #the subprior bounds are scaled with respect to the parameter ranges
                self._scale(params.max_bounds, self.min_bounds, self.max_bounds),
            ], -1)
            return scaled_params

    def restore_params(self, scaled_params: Tensor) -> BasicParams:
        """restore the parameters to their original range

        Args:
            scaled_params (Tensor): the scaled parameters

        Returns:
            BasicParams: the parameters restored to their original range
        """
        num_params = scaled_params.shape[-1] // 3
        scaled_params, scaled_min_bounds, scaled_max_bounds = torch.split(
            scaled_params, num_params, -1
        )
        if self.scale_params_by_ranges:
            min_bounds = self._restore(scaled_min_bounds, self.min_bounds, self.max_bounds)
            max_bounds = self._restore(scaled_max_bounds, self.min_bounds, self.max_bounds)
            params = self._restore(scaled_params, self.min_bounds, self.max_bounds)
        else:
            min_bounds = self._restore(scaled_min_bounds, self.min_bounds, self.max_bounds)
            max_bounds = self._restore(scaled_max_bounds, self.min_bounds, self.max_bounds)
            params = self._restore(scaled_params, min_bounds, max_bounds)

        return BasicParams(
            parameters=params,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            max_num_layers=self.max_num_layers,
            param_model=self.param_model,
        )
    
    def scale_bounds(self, bounds: Tensor) -> Tensor:
        return self._scale(bounds, self.min_bounds, self.max_bounds)

    def log_prob(self, params: BasicParams) -> Tensor:
        log_prob = torch.zeros(params.batch_size, device=self.device, dtype=self.dtype)
        log_prob[~self.get_indices_within_bounds(params)] = -float('inf')
        return log_prob

    def get_indices_within_domain(self, params: BasicParams) -> Tensor:
        return self.get_indices_within_bounds(params)

    def get_indices_within_bounds(self, params: BasicParams) -> Tensor:
        return (
                torch.all(params.parameters >= params.min_bounds, -1) &
                torch.all(params.parameters <= params.max_bounds, -1)
        )

    def filter_params(self, params: BasicParams) -> BasicParams:
        indices = self.get_indices_within_domain(params)
        return params[indices]

    def clamp_params(
            self, params: BasicParams, inplace: bool = False
    ) -> BasicParams:
        if inplace:
            params.parameters = torch.clamp_(params.parameters, params.min_bounds, params.max_bounds)
            return params

        return BasicParams(
            parameters=torch.clamp(params.parameters, params.min_bounds, params.max_bounds),
            min_bounds=params.min_bounds.clone(),
            max_bounds=params.max_bounds.clone(),
            max_num_layers=self.max_num_layers,
            param_model=self.param_model,
        )
