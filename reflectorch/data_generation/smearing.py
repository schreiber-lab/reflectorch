import torch
from torch import Tensor

from reflectorch.data_generation.priors.params import Params


class Smearing(object):
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

    def get_curves(self, q_values: Tensor, params: Params):
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
