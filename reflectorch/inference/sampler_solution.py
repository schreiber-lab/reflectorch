import torch
from torch import Tensor

from reflectorch.data_generation.priors.utils import uniform_sampler
from reflectorch.data_generation.priors.subprior_sampler import UniformSubPriorParams
from reflectorch.data_generation.priors.params import Params
from reflectorch.data_generation.likelihoods import LogLikelihood


def simple_sampler_solution(
        likelihood: LogLikelihood,
        predicted_params: UniformSubPriorParams,
        min_bounds: Tensor,
        max_bounds: Tensor,
        total_min_bounds: Tensor,
        total_max_bounds: Tensor,
        num: int = 2 ** 15,
        coef: float = 0.1,
) -> UniformSubPriorParams:

    params_t = predicted_params.as_tensor(add_bounds=False)

    delta = (max_bounds - min_bounds) * coef
    min_bounds = torch.clamp(params_t - delta, total_min_bounds, total_max_bounds)
    max_bounds = torch.clamp(params_t + delta, total_min_bounds, total_max_bounds)

    sampled_params_t = uniform_sampler(min_bounds, max_bounds, num, params_t.shape[-1])

    sampled_params_t[0] = params_t[0]
    sampled_params = Params.from_tensor(sampled_params_t)
    sampled_curves = sampled_params.reflectivity(likelihood.q)
    log_probs = likelihood.calc_log_likelihood(sampled_curves)
    best_idx = torch.argmax(log_probs)
    best_param = sampled_params_t[best_idx]
    best_param = UniformSubPriorParams.from_tensor(
        torch.cat([torch.atleast_2d(best_param), torch.atleast_2d(min_bounds), torch.atleast_2d(max_bounds)], -1)
    )
    return best_param
