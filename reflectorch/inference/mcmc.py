# as in the PANPE and torchemcee repositories

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.fft import fft, ifft
from tqdm import trange

from typing import Tuple, Optional
from math import sqrt

class MCMCState:
    """A state of MCMC chains."""

    __slots__ = ("coords", "log_prob")

    def __init__(self, coords: Tensor, log_prob: Tensor):
        """Create a new state.

        Args:
               coords: The coordinates of the walkers. Tensor of shape (num_walkers, ndim).
               log_prob: The log probability of the walkers. Tensor of shape (num_walkers,).
        """
        self.coords = coords
        self.log_prob = log_prob

    @property
    def device(self):
        return self.coords.device

    @property
    def dtype(self):
        return self.coords.dtype

    def __repr__(self):
        return "State({0}, log_prob={1})".format(self.coords, self.log_prob)


class MCMCBackend:
    """
    A backend for storing states of MCMC chains.

    Args:
        num_walkers: The number of walkers in the chain.
        device: The device to store the states on.
        thin_by: The number of steps to thin the chain by.

    """

    def __init__(
        self, num_walkers: int, device: torch.device = "cpu", thin_by: int = 1
    ):
        self.coords = []
        self.log_probs = []
        self.num_walkers = num_walkers
        self.accepted = torch.zeros(num_walkers).to(device)
        self.device = device
        self.thin_by = thin_by
        self._iteration = 0

    @property
    def accepted_fractions(self) -> Tensor:
        return (
            self.accepted / self.iteration if self.iteration else self.accepted
        ).clone()

    @property
    def iteration(self) -> int:
        return self._iteration

    def save_state(self, state: MCMCState, accepted: Tensor):
        self.accepted += accepted.to(self.device)

        if self._iteration % self.thin_by == 0:
            self._save_state(state)

        self._iteration += 1

    def _save_state(self, state: MCMCState):
        self.coords.append(state.coords.to(self.device).clone())
        self.log_probs.append(state.log_prob.to(self.device).clone())

    def get_chain(self, flat: bool = False) -> Tensor:
        if flat:
            return torch.cat(self.coords, dim=0)
        else:
            return torch.stack(self.coords, dim=0)

    @property
    def chain_size(self):
        return sum(c.shape[0] for c in self.coords)

    def __repr__(self):
        return (
            f"Backend(iterations={self.iteration}, num_walkers={self.num_walkers}, chain_size={self.chain_size},"
            f"accepted_fractions={self.accepted_fractions})"
        )


class MCMCSampler:
    """
    A class for running MCMC chains.

    Args:
        ndim: The number of dimensions of the target distribution.
        num_walkers: The number of walkers in the chain.
        log_prob_fn: A vectorized function that computes log probabilities of the target distribution.
        backend: A backend for storing the states of the chain.
        thin_by: The number of steps to thin the chain by.
    """

    def __init__(
        self,
        ndim: int,
        num_walkers: int,
        log_prob_fn,
        backend: MCMCBackend = None,
        thin_by: int = 1,
    ):
        self.ndim, self.num_walkers = ndim, num_walkers
        self.backend = backend or MCMCBackend(self.num_walkers, thin_by=thin_by)
        self.log_prob_fn = log_prob_fn

    def run(
        self, init_coords: Tensor, num_steps: int, *, burn_in: int = 0, **kwargs
    ) -> MCMCState:
        """
        Run MCMC.

        Args:
            init_coords: The initial coordinates of the walkers. Tensor of shape (num_walkers, ndim).
            num_steps: Number of steps to run the chain for.
            burn_in: Number of steps to discard at the beginning of the chain.

        Returns:
            The final state of the chains.

        """

        backend, state = run_mcmc(
            init_coords,
            self.log_prob_fn,
            num_steps,
            backend=self.backend,
            burn_in=burn_in,
            **kwargs,
        )
        return state

    def get_chain(self):
        return self.backend.get_chain()

    @property
    def accepted_fractions(self) -> Tensor:
        return self.backend.accepted_fractions

    def __repr__(self):
        return f"MCMCSampler(backend={repr(self.backend)})"


class MCMCMove:
    """
    A class for proposing new states of the MCMC chain.
    """

    def step(self, state: MCMCState, log_prob_fun) -> Tuple[MCMCState, Tensor]:
        """
        Propose a new state of the chain.

        Args:
            state: The current state of the chain.
            log_prob_fun: The vectorized log probability function of the target distribution.

        Returns:
            A tuple of the new state and a boolean tensor indicating whether the proposal was accepted.

        """
        raise NotImplementedError


def run_mcmc(
    init_coords: Tensor,
    log_prob_fn,
    num_steps: int,
    backend: MCMCBackend = None,
    *,
    moves: MCMCMove or Tuple[Tuple[MCMCMove, float], ...] = None,
    burn_in: int = 0,
    thin_by: int = 1,
    disable_tqdm: bool = False,
) -> Tuple[MCMCBackend, MCMCState]:
    """
    Run MCMC chains.

    Args:
        init_coords: The initial coordinates of the walkers. Tensor of shape (num_walkers, ndim).
        log_prob_fn: The log probability function of the target distribution.
        num_steps: The number of steps to run the chain for.
        backend: The backend to store the states of the chain.
        moves: A tuple of tuples containing MCMC moves and their probabilities.
         If None, the default is a single DEMove with weight 1.
        burn_in: The number of steps to discard at the beginning of the chain.
        thin_by: The number of steps to thin the chain by. The default is 1. Ignored if backend is not None.
        disable_tqdm: Whether to disable the progress bar.

    Returns:
        A tuple of the backend and the final state of the chain.

    """

    if moves is None:
        moves, weights = (DEMove(),), (1.0,)
    elif isinstance(moves, MCMCMove):
        moves, weights = (moves,), (1.0,)
    else:
        moves, weights = zip(*moves)

    weights = np.array(weights) / np.sum(weights)

    state = MCMCState(init_coords, log_prob_fn(init_coords))

    num_walkers, ndim = init_coords.shape

    backend = backend or MCMCBackend(num_walkers, thin_by=thin_by)

    pbar = trange(num_steps, disable=disable_tqdm)

    for step_idx in pbar:
        move = np.random.choice(moves, p=weights)
        state, accepted = move.step(state, log_prob_fn)

        if step_idx >= burn_in:
            backend.save_state(state, accepted)

    return backend, state


class RedBlueMove(MCMCMove):
    """
    A class for red-blue MCMC moves based on emcee package.
    """

    def step(self, state: MCMCState, log_prob_fun) -> Tuple[MCMCState, Tensor]:
        """
        Propose a new state of the population.
        Args:
            state: The current state of the population. The state is modified in place.
            log_prob_fun: The vectorized log probability function of the target distribution.

        Returns:
            A tuple of the updated state and a boolean tensor indicating whether the proposal was accepted.

        """
        nwalkers, ndim = state.coords.shape
        device, dtype = state.device, state.dtype

        accepted = torch.zeros((nwalkers,), dtype=torch.bool, device=device)

        all_indices = torch.arange(nwalkers, device=device)

        split_num_indices = shuffle(all_indices % 2)

        for split_num in range(2):
            updated_indices = split_num_indices == split_num
            updated_walkers, source_walkers = (
                state.coords[updated_indices],
                state.coords[~updated_indices],
            )
            proposed_coords, factors = self._get_proposal(
                updated_walkers, source_walkers
            )
            new_log_probs = log_prob_fun(proposed_coords)

            sampled_rands = torch.log(
                torch.rand(factors.shape[0], device=device, dtype=dtype)
            )

            lnpdiff = (
                factors + new_log_probs - state.log_prob[all_indices[updated_indices]]
            )

            accepted[updated_indices] = lnpdiff > sampled_rands

            new_state = MCMCState(proposed_coords, log_prob=new_log_probs)
            state = _update_state(state, new_state, accepted, updated_indices)

        return state, accepted

    def _get_proposal(self, updated_walkers: Tensor, source_walkers: Tensor):
        raise NotImplementedError


class StretchMove(RedBlueMove):
    """
    A stretch move.
    """

    def __init__(self, a: float = 2.0):
        self.a = a

    def _get_proposal(self, s, c):
        return stretch_move(s, c, self.a)


class DEMove(RedBlueMove):
    """
    A differential evolution MCMC move.
    """

    def __init__(self, sigma: float = 1e-5, g0: float = None):
        self.sigma = sigma
        self.g0 = g0

    def _get_proposal(self, s, c):
        return de_move(s, c, self.sigma, self.g0)


def _update_state(old_state, new_state, accepted, subset):
    m1 = subset & accepted
    m2 = accepted[subset]
    old_state.coords[m1] = new_state.coords[m2]
    old_state.log_prob[m1] = new_state.log_prob[m2]
    return old_state


def stretch_move(s: Tensor, c: Tensor, a: float):
    """
    A stretch move.
    """

    ns, nc, ndim = s.shape[0], c.shape[0], c.shape[1]

    zz = ((a - 1.0) * torch.rand(ns, device=s.device, dtype=s.dtype) + 1) ** 2.0 / a
    factors = (ndim - 1.0) * torch.log(zz)
    rint = torch.randint(nc, size=(ns,), device=s.device)

    return c[rint] - (c[rint] - s) * zz[:, None], factors


def de_move(
    updated_walkers: Tensor,
    source_walkers: Tensor,
    sigma: float = 1e-5,
    g0: float = None,
):
    """
    Differential evolution MCMC move.
    """

    u_num, ndim = updated_walkers.shape
    s_num = source_walkers.shape[0]

    if g0 is None:
        g0 = 2.38 / sqrt(2 * ndim)

    # sample pairs of walkers from the c population that exclude pairs of same walkers

    # Get the lower triangle indices
    rows, cols = torch.tril_indices(s_num, s_num, -1, device=updated_walkers.device)

    # Combine rows-cols and cols-rows pairs
    pairs = torch.stack([torch.cat([rows, cols]), torch.cat([cols, rows])], dim=1)

    # Sample from the pairs
    indices = torch.randint(0, s_num * (s_num - 1), (u_num,), dtype=torch.long)
    pairs = pairs[indices]

    # Get the differences between the sampled pairs of source walkers
    diffs = torch.diff(source_walkers[pairs], dim=1).squeeze(dim=1)  # (ns, ndim)

    # Sample a gamma value for each walker following Nelson et al. (2013) https://doi.org/10.1088/0067-0049/210/1/11
    gamma = g0 * (1 + sigma * torch.randn(u_num, 1).to(updated_walkers))  # (ns, 1)

    q = updated_walkers + gamma * diffs

    return q, torch.zeros_like(updated_walkers[..., 0])


def shuffle(t: Tensor) -> Tensor:
    """
    Shuffle a tensor along the first dimension.
    Args:
        t: tensor to shuffle.

    Returns:
        The shuffled tensor.
    """
    idx = torch.randperm(t.shape[0], device=t.device)
    return t[idx].view(t.size())



##############################################################################################


class HMCMove(MCMCMove):
    def __init__(self, num_steps_per_sample: int = 10, step_size: float = 0.3):
        self.num_steps_per_sample = num_steps_per_sample
        self.step_size = step_size

    def step(self, state: MCMCState, log_prob_fun) -> Tuple[MCMCState, Tensor]:
        params, accepted = batched_hamiltonian_mc_step(
            log_prob_func=log_prob_fun,
            params_init=state.coords,
            num_steps_per_sample=self.num_steps_per_sample,
            step_size=self.step_size,
        )

        new_state = MCMCState(params, log_prob=log_prob_fun(params))
        return new_state, accepted


def batched_hamiltonian_mc_step(
        log_prob_func,
        params_init: Tensor,
        num_steps_per_sample: int = 10,
        step_size: float = 0.1,
) -> Tuple[Tensor, Tensor]:
    """
    Perform a Hamiltonian Monte Carlo step in a batch.

    Args:
        log_prob_func (callable): Log probability function of the target distribution that supports batched execution.
        params_init (Tensor): Initial parameters with shape (num_chains, dim) or (dim, ) for the HMC step.
        num_steps_per_sample (int, optional): Number of leapfrog steps per sample. Defaults to 10.
        step_size (float, optional): Step size for the leapfrog integration. Defaults to 0.1.

    Returns:
        Tuple[Tensor, Tensor]: Tuple containing the updated parameters with shape (num_chains, dim)
                               and a boolean tensor with shape (num_chains, ) indicating
                               whether each sample was accepted or not.
    """

    assert params_init.dim() in (1, 2), f"Wrong dimensionality of the initial params. " \
                                        f"Expected dim == 1 or dim == 2, got {params_init.dim()}"

    # if 1-dim params for a single chain are provided, extend for 2-dim
    params_init = torch.atleast_2d(params_init)

    # clone params to perform further in-place operations
    params = params_init.clone()

    # generate auxiliary momentum values
    momentum = torch.randn_like(params)

    # calculate hamiltonian for the initial parameters (momentum will be changed in-place afterwards)
    ham = hamiltonian(params, momentum, log_prob_func)

    # perform batched leapfrog step:
    params, momentum, finite_indices = batched_leapfrog(
        params, momentum, log_prob_func,
        steps=num_steps_per_sample, step_size=step_size,
    )

    # calculate hamiltonian for the new parameters:
    new_ham = torch.ones_like(ham) * float('inf')
    new_ham[finite_indices] = hamiltonian(
        params[finite_indices], momentum[finite_indices], log_prob_func
    )

    # accept / reject proposals via a Metropolis update
    rho = torch.clamp_max(ham - new_ham, 0.)
    rejection_condition = rho < torch.log(torch.rand_like(rho))
    num_rejected = rejection_condition.sum().item()
    if num_rejected > 0:
        # return initial parameters instead of rejected proposals
        params[rejection_condition] = params_init[rejection_condition]

    accepted = ~rejection_condition

    return params, accepted


def hamiltonian(params: Tensor, momentum: Tensor, log_prob_func) -> Tensor:
    """
    Calculate the Hamiltonian for a given set of parameters and momentum.

    Args:
        params (Tensor): Parameters with shape (num_chains, dim).
        momentum (Tensor): Momentum with shape (num_chains, dim) associated with the parameters.
        log_prob_func (callable): Log probability function of the target distribution.

    Returns:
        Tensor: Hamiltonian values with shape (num_chains, ) for the given parameters and momentum.
    """
    return -log_prob_func(params) + 0.5 * torch.sum(momentum * momentum, dim=-1)


def batched_leapfrog(
        params: Tensor, momentum: Tensor,
        log_prob_func,
        steps: int = 10,
        step_size: float = 0.1,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform the leapfrog integration step in the Hamiltonian Monte Carlo algorithm for a batch of parameters.

    Args:
        params (Tensor): Parameters with shape (num_chains, dim).
        momentum (Tensor): Momentum with shape (num_chains, dim) associated with the parameters.
        log_prob_func (callable): Log probability function of the target distribution.
        steps (int, optional): Number of leapfrog steps. Defaults to 10.
        step_size (float, optional): Step size for the leapfrog integration. Defaults to 0.1.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Tuple containing the updated parameters, momentum, and a boolean
                                       tensor indicating whether each sample was finite or not.
    """

    grad, finite_indices = _get_batched_params_grad(params, log_prob_func)
    momentum[finite_indices] += 0.5 * step_size * grad

    for n in range(steps):
        if finite_indices.sum().item() == 0:
            # all the proposals are outside the prior distribution, so we stop here
            break

        params[finite_indices] += step_size * momentum[finite_indices]

        grad, finite_indices = _get_batched_params_grad(params, log_prob_func, finite_indices)

        momentum[finite_indices] += step_size * grad

    momentum[finite_indices] -= 0.5 * step_size * grad

    return params, momentum, finite_indices


def _get_batched_params_grad(p, func, finite_indices=None):
    """
    Calculate the gradient of the log probability function for a batch of parameters.

    Args:
        p (Tensor): Parameters with shape (num_chains, dim).
        func (callable): Log probability function of the target distribution.
        finite_indices (Tensor, optional): Boolean tensor indicating which samples have finite log prob.
                                           Defaults to None.

    Returns:
        Tuple[Tensor, Tensor]: Tuple containing the gradients tensor of the log probability function
                               w.r.t. parameters with finite log prob and
                               a boolean tensor indicating which samples have finite log prob.
    """

    if finite_indices is not None:
        p = p[finite_indices]

    p.detach_().requires_grad_()
    log_probs = func(p)
    new_finite_indices = torch.isfinite(log_probs)
    if finite_indices is not None:
        finite_indices[finite_indices.clone()] = new_finite_indices
    else:
        finite_indices = new_finite_indices
    grad = torch.autograd.grad(log_probs.sum(), p, allow_unused=True)[0][new_finite_indices]
    p.detach_()
    assert grad.shape[0] == finite_indices.sum().item(), f"{grad.shape[0]} != {finite_indices.sum().item()}"
    return grad, finite_indices

def get_tau_estimation(y: Tensor):
    acf = batched_autocorr_func(y)
    taus = torch.cumsum(acf, dim=0) * 2 - 1
    window = auto_window_from_taus(taus)
    return taus[window]


def batched_autocorr_func(y: Tensor, window: int = None, reduce: bool = True):
    assert len(y.shape) == 2

    batch_size, sequence_size = y.shape
    n = next_pow_two(sequence_size)

    # Compute the FFT and then the auto-correlation function
    f = fft(y - y.mean(-1)[..., None], n=2 * n)
    acf = torch.real(ifft(f * torch.conj(f)))

    if window:
        acf = acf[:, :window]

    # Optionally normalize
    if reduce:
        acf = torch.mean(acf, 0)
        acf = acf / acf[0]
    else:
        acf = acf / acf[:, 0]

    return acf


def auto_window_from_taus(taus: Tensor, c: float = 5):
    num_taus = taus.shape[0]
    m = torch.arange(num_taus, device=taus.device, dtype=taus.dtype) < c * taus
    if torch.any(m):
        return torch.where(~m)[0].min()
    return num_taus - 1


def next_pow_two(n):
    """
    Find the next power of two for a given number.

    Args:
        n: int, the number to find the next power of two for

    Returns:
        int, the next power of two for the given number

    """
    i = 1
    while i < n:
        i = i << 1
    return i

def summarize_ensemble_run(
    backend,
    param_names=None,
    max_plot_walkers=32,
    max_plot_params=6,
):
    """
    Single-run diagnostics for an ensemble MCMC backend.
    """
    chain = backend.get_chain(flat=False).detach().cpu()   # [draw, walker, dim]
    logp = torch.stack(backend.log_probs, dim=0).detach().cpu()  # [draw, walker]

    n_draw, n_walk, dim = chain.shape

    accepted = backend.accepted_fractions.detach().cpu().numpy()

    out = {
        "n_draw": int(n_draw),
        "n_walkers": int(n_walk),
        "dim": int(dim),
        "accept_mean": float(np.mean(accepted)),
        "accept_min": float(np.min(accepted)),
        "accept_max": float(np.max(accepted)),
        "accept_per_walker": accepted,
    }

    tau_per_param = []
    ess_per_param = []

    for j in range(dim):
        y = chain[:, :, j].T  # [walker, draw]
        try:
            tau_j = float(get_tau_estimation(y).item())
            ess_j = (n_draw * n_walk) / tau_j if tau_j > 0 else np.nan
        except Exception:
            tau_j = np.nan
            ess_j = np.nan
        tau_per_param.append(tau_j)
        ess_per_param.append(ess_j)

    out["tau_per_param"] = np.array(tau_per_param)
    out["ess_per_param"] = np.array(ess_per_param)

    walkers_to_plot = np.linspace(0, n_walk - 1, min(n_walk, max_plot_walkers)).round().astype(int)

    n_params_plot = min(dim, max_plot_params)
    fig1, axes = plt.subplots(n_params_plot + 1, 1, figsize=(10, 2.2 * (n_params_plot + 1)), sharex=True)

    for w in walkers_to_plot:
        axes[0].plot(logp[:, w].numpy(), alpha=0.5)
    axes[0].set_ylabel("log prob")
    axes[0].set_title("Trace diagnostics")

    for i in range(n_params_plot):
        name = param_names[i] if (param_names is not None and i < len(param_names)) else f"p{i}"
        for w in walkers_to_plot:
            axes[i + 1].plot(chain[:, w, i].numpy(), alpha=0.5)
        axes[i + 1].set_ylabel(name)

    axes[-1].set_xlabel("saved draw")
    plt.tight_layout()

    return out, fig1