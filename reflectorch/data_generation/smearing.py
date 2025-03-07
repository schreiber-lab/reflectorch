import torch
from torch import Tensor

from reflectorch.data_generation.priors.parametric_subpriors import BasicParams


class Smearing(object):
    """Class which applies resolution smearing to the reflectivity curves. 
    The intensity at a q point will be the average of the intensities of neighbouring q points, weighted by a gaussian profile.

    Args:
        sigma_range (tuple, optional): the range for sampling the resolutions. Defaults to (0.01, 0.1).
        constant_dq (bool, optional): if ``True`` the smearing is constant (the resolution is given by the constant dq at each point in the curve) 
                                    otherwise the smearing is linear (the resolution is given by the constant dq/q at each point in the curve). Defaults to True.
        gauss_num (int, optional): the number of interpolating gaussian profiles. Defaults to 31.
        share_smeared (float, optional): the share of curves in the batch for which the resolution smearing is applied. Defaults to 0.2.
    """
    def __init__(self,
                 sigma_range: tuple = (0.01, 0.1),
                 constant_dq: bool = False,
                 gauss_num: int = 31,
                 share_smeared: float = 0.2,
                 ):
        self.sigma_min, self.sigma_max = sigma_range
        self.sigma_delta = self.sigma_max - self.sigma_min
        self.constant_dq = constant_dq
        self.gauss_num = gauss_num
        self.share_smeared = share_smeared

    def __repr__(self):
        return f'Smearing(({self.sigma_min}, {self.sigma_max})'

    def generate_resolutions(self, batch_size: int, device=None, dtype=None):
        num_smeared = int(batch_size * self.share_smeared)
        if not num_smeared:
            return None, None
        dq = torch.rand(num_smeared, 1, device=device, dtype=dtype) * self.sigma_delta + self.sigma_min
        indices = torch.zeros(batch_size, device=device, dtype=torch.bool)
        indices[torch.randperm(batch_size, device=device)[:num_smeared]] = True
        return dq, indices
    
    def scale_resolutions(self, resolutions: Tensor) -> Tensor:
        """Scales the q-resolution values to [-1,1] range using the internal sigma range"""
        sigma_min = 0.0 if self.share_smeared != 1.0 else self.sigma_min
        return 2 * (resolutions - sigma_min) / (self.sigma_max - sigma_min) - 1

    def get_curves(self, q_values: Tensor, params: BasicParams, refl_kwargs:dict = None):
        refl_kwargs = refl_kwargs or {}

        dq, indices = self.generate_resolutions(params.batch_size, device=params.device, dtype=params.dtype)
        q_resolutions = torch.zeros(q_values.shape[0], 1, dtype=q_values.dtype, device=q_values.device)

        if dq is None:
            return params.reflectivity(q_values, **refl_kwargs), q_resolutions
        
        refl_kwargs_not_smeared = {}
        refl_kwargs_smeared = {}
        for key, value in refl_kwargs.items():
            if isinstance(value, torch.Tensor) and value.shape[0] == params.batch_size:
                refl_kwargs_not_smeared[key] = value[~indices]
                refl_kwargs_smeared[key] = value[indices]
            else:
                refl_kwargs_not_smeared[key] = value
                refl_kwargs_smeared[key] = value

         # Compute unsmeared reflectivity
        if (~indices).sum().item():
            if q_values.dim() == 2 and q_values.shape[0] > 1:
                q = q_values[~indices]
            else:
                q = q_values

            reflectivity_not_smeared = params[~indices].reflectivity(q, **refl_kwargs_not_smeared)
        else:
            reflectivity_not_smeared = None

        # Compute smeared reflectivity
        if indices.sum().item():
            if q_values.dim() == 2 and q_values.shape[0] > 1:
                q = q_values[indices]
            else:
                q = q_values

            reflectivity_smeared = params[indices].reflectivity(
                q, dq=dq, constant_dq=self.constant_dq, gauss_num=self.gauss_num, **refl_kwargs_smeared
            )
        else:
            reflectivity_smeared = None

        curves = torch.empty(params.batch_size, q_values.shape[-1], device=params.device, dtype=params.dtype)
        
        if (~indices).sum().item():
            curves[~indices] = reflectivity_not_smeared
        
        curves[indices] = reflectivity_smeared

        q_resolutions[indices] = dq

        return curves, q_resolutions