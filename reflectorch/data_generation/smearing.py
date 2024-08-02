import torch
from torch import Tensor

from reflectorch.data_generation.priors.parametric_subpriors import BasicParams


class Smearing(object):
    """Class which applies resolution smearing to the reflectivity curves. 
    The intensity at a q point will be the average of the intensities of neighbouring q points, weighted by a gaussian profile.

    Args:
        sigma_range (tuple, optional): the range for sampling the resolutions. Defaults to (1e-4, 5e-3).
        constant_dq (bool, optional): if ``True`` the smearing is constant (the resolution is given by the constant dq at each point in the curve) 
                                    otherwise the smearing is linear (the resolution is given by the constant dq/q at each point in the curve). Defaults to True.
        gauss_num (int, optional): the number of interpolating gaussian profiles. Defaults to 31.
        share_smeared (float, optional): the share of curves in the batch for which the resolution smearing is applied. Defaults to 0.2.
    """
    def __init__(self,
                 sigma_range: tuple = (1e-4, 5e-3),
                 constant_dq: bool = True,
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

    def get_curves(self, q_values: Tensor, params: BasicParams):
        dq, indices = self.generate_resolutions(params.batch_size, device=params.device, dtype=params.dtype)

        if dq is None:
            return params.reflectivity(q_values, log=False)

        curves = torch.empty(params.batch_size, q_values.shape[-1], device=params.device, dtype=params.dtype)

        if (~indices).sum().item():
            if q_values.dim() == 2 and q_values.shape[0] > 1:
                q = q_values[~indices]
            else:
                q = q_values

            curves[~indices] = params[~indices].reflectivity(q, log=False)

        if indices.sum().item():
            if q_values.dim() == 2 and q_values.shape[0] > 1:
                q = q_values[indices]
            else:
                q = q_values

            curves[indices] = params[indices].reflectivity(
                q, dq=dq, constant_dq=self.constant_dq, log=False, gauss_num=self.gauss_num
            )

        return curves
