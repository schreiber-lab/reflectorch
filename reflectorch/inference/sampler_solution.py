import torch
from torch import Tensor

from reflectorch.data_generation.priors.utils import uniform_sampler
from reflectorch.data_generation.priors.subprior_sampler import UniformSubPriorParams
from reflectorch.data_generation.priors.params import Params
from reflectorch.data_generation.likelihoods import LogLikelihood


def simple_sampler_solution(
        likelihood: LogLikelihood,
        predicted_params: UniformSubPriorParams,
        total_min_bounds: Tensor,
        total_max_bounds: Tensor,
        num: int = 2 ** 15,
        coef: float = 0.1,
) -> UniformSubPriorParams:
    sampled_params_t = sample_around_params(predicted_params, total_min_bounds, total_max_bounds, num=num, coef=coef)
    sampled_params = Params.from_tensor(sampled_params_t)
    return get_best_mse_param(sampled_params, likelihood, predicted_params.min_bounds, predicted_params.max_bounds)


def sample_around_params(predicted_params: UniformSubPriorParams,
                         total_min_bounds: Tensor,
                         total_max_bounds: Tensor,
                         num: int = 2 ** 15,
                         coef: float = 0.1,
                         ) -> Tensor:
    params_t = predicted_params.as_tensor(add_bounds=False)

    delta = (predicted_params.max_bounds - predicted_params.min_bounds) * coef
    min_bounds = torch.clamp(params_t - delta, total_min_bounds, total_max_bounds)
    max_bounds = torch.clamp(params_t + delta, total_min_bounds, total_max_bounds)

    sampled_params_t = uniform_sampler(min_bounds, max_bounds, num, params_t.shape[-1])
    sampled_params_t[0] = params_t[0]

    return sampled_params_t


def get_best_mse_param(
        params: Params,
        likelihood: LogLikelihood,
        min_bounds: Tensor = None,
        max_bounds: Tensor = None,
):
    sampled_curves = params.reflectivity(likelihood.q)
    log_probs = likelihood.calc_log_likelihood(sampled_curves)
    best_idx = torch.argmax(log_probs)
    best_param = params[best_idx:best_idx + 1]

    if min_bounds is not None:
        best_param = UniformSubPriorParams.from_tensor(
            torch.cat([best_param.as_tensor(), torch.atleast_2d(min_bounds), torch.atleast_2d(max_bounds)], -1)
        )
    return best_param
