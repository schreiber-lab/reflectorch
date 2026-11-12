from tqdm import trange

import torch
from torch import nn

from reflectorch.data_generation import LogLikelihood


class ReflGradientFit(nn.Module):
    def __init__(
        self,
        q: torch.Tensor,
        exp_curve: torch.Tensor,
        prior_sampler,
        init_params: torch.Tensor,
        sigmas: torch.Tensor = None,
        rel_err: float = 0.1,
        abs_err: float = 1e-7,
        lr: float = 1e-2,
        optim_cls=None,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.q = q.to(device)
        self.exp_curve = exp_curve.to(device)
        self.prior_sampler = prior_sampler

        if sigmas is None:
            sigmas = exp_curve * rel_err + abs_err
        self.sigmas = sigmas.to(device)

        self.likelihood = LogLikelihood(self.q, self.exp_curve, prior_sampler, self.sigmas)

        # init_params: [batch_size, num_params]
        if init_params.ndim == 1:
            init_params = init_params.unsqueeze(0)

        self.init_params = init_params.to(device).clone().detach()
        self.params_to_fit = nn.Parameter(self.init_params.clone().detach())
        self.batch_size = self.init_params.shape[0]

        optim_cls = optim_cls or torch.optim.Adam
        self.optim = optim_cls([self.params_to_fit], lr=lr)
        self.losses = []

    def params(self):
        return self.params_to_fit  # [batch_size, num_params]

    def refl(self):
        return self.prior_sampler.param_model.reflectivity(self.q, self.params())

    def calc_log_likelihood(self):
        return self.likelihood.calc_log_likelihood(self.refl())

    def calc_loss(self):
        return -self.calc_log_likelihood().mean()

    def run(self, num_iterations=200, disable_tqdm=False):
        pbar = trange(num_iterations, disable=disable_tqdm)
        for _ in pbar:
            self.optim.zero_grad()
            loss = self.calc_loss()
            loss.backward()
            self.optim.step()
            self.losses.append(loss.item())
            pbar.set_description(f"Loss = {loss.item():.2e}")

        with torch.no_grad():
            losses = -self.likelihood.calc_log_likelihood(self.refl())  # [batch_size]
            best_idx = losses.argmin()
            best_params = self.params()[best_idx].detach().cpu()
            best_curve = self.refl()[best_idx].detach().cpu()
            return best_params, best_curve, losses[best_idx].item()
