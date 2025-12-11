
import warnings
import joblib
from joblib import Parallel, delayed
import numpy as np
from scipy.optimize import minimize, curve_fit
import torch

from reflectorch.data_generation.priors.base import PriorSampler
from reflectorch.data_generation.reflectivity import abeles_np

__all__ = [
    "standard_refl_fit",
    "refl_fit",
    "fit_refl_curve",
    "restore_masked_params",
    "get_fit_with_growth",
    "batch_standard_refl_fit",
]


def standard_restore_params(fitted_params) -> dict:
    num_layers = (fitted_params.size - 2) // 3

    return dict(
        thickness=fitted_params[:num_layers],
        roughness=fitted_params[num_layers:2 * num_layers + 1],
        sld=fitted_params[2 * num_layers + 1:],
    )


def mse_loss(curve1, curve2):
    return np.sum((curve1 - curve2) ** 2)

def standard_refl_fit(
        q: np.ndarray, curve: np.ndarray,
        init_params: np.ndarray,
        bounds: np.ndarray = None,
        refl_generator=abeles_np,
        restore_params_func=standard_restore_params,
        scale_curve_func=np.log10,
        **kwargs
):
    if bounds is not None:
        kwargs['bounds'] = bounds
        init_params = np.clip(init_params, *bounds)

    res = curve_fit(
        standard_get_scaled_curve_func(
            refl_generator=refl_generator,
            restore_params_func=restore_params_func,
            scale_curve_func=scale_curve_func,
        ),
        q, scale_curve_func(curve),
        p0=init_params, **kwargs
    )

    curve = refl_generator(q, **restore_params_func(res[0]))
    return res[0], curve



def batch_refl_fit(
        q: np.ndarray, 
        curves: np.ndarray,
        init_params: np.ndarray, # (n_curves, n_params)
        prior_sampler: PriorSampler,
        bounds: np.ndarray = None,
        error_bars: np.ndarray = None,
        scale_curve_func=np.log10,
        method: str = 'trf', #'lm', 'trf'
        polishing_max_steps: int = None,
        reflectivity_kwargs: dict = None,
        n_jobs: int = -1,
        verbose: int = 5,
        **kwargs
):
    """
    Fit (polished fit) multiple reflectivity curves in parallel using joblib.

    Parameters
    ----------
    q : np.ndarray
        1D array of momentum transfer values (same for all curves).
    curves : np.ndarray
        2D array of reflectivity curves with shape (n_curves, n_q).
    init_params : np.ndarray
        2D array of initial parameter guesses (n_curves, n_params).
    prior_sampler : PriorSampler
        The prior sampler.
    bounds : np.ndarray, optional
        Bounds for the parameters, shape (2, n_params). Shared by all the curves. Default: None.
    error_bars : np.ndarray, optional
        Error bars for the curves, shape (n_curves, n_q). Default: None.
    scale_curve_func : callable, optional
        Function to scale the curves. Default: `np.log10`.
    method : str, optional
        The method to use for the fitting. Default: 'trf'.
    polishing_max_steps : int, optional
        The maximum number of function evaluations for the polishing step. Default: None.
    reflectivity_kwargs : dict, optional
        Keyword arguments for the reflectivity function. Default: None.
    n_jobs : int, optional
        The number of jobs to run in parallel. Default: -1 (all CPUs).
    verbose : int, optional
        The verbosity level for joblib. Default: 5.
    **kwargs : dict
        Extra keyword arguments passed to `scipy.optimize.curve_fit`.

    Returns
    -------
    params_array : np.ndarray
        Array of fitted parameter values for each curve, shape (n_curves, n_params).
    error_bars_array : np.ndarray
        Array of error bars for the fitted parameter values, shape (n_curves, n_params).
    curves_array : np.ndarray
        Array of fitted reflectivity curves, shape (n_curves, n_q).
    """
    if bounds is not None:
        if bounds.ndim == 2:
            bounds = bounds[None].repeat(curves.shape[0], 0)
        elif bounds.ndim == 3:
            assert bounds.shape[0] == curves.shape[0], f"Bounds must have the same number of curves as the number of curves, got {bounds.shape[0]} and {curves.shape[0]}"
        else:
            raise ValueError(f"Bounds must be a 2D or 3D array, got {bounds.ndim}D array")
    else:
        bounds = [None] * curves.shape[0]
    
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(refl_fit)(
            q=q, curve=curve, init_params=init_params,
            bounds=bound,
            prior_sampler=prior_sampler,
            error_bars=error_bars,
            method=method,
            scale_curve_func=scale_curve_func,
            polishing_max_steps=polishing_max_steps,
            reflectivity_kwargs=reflectivity_kwargs,
            **kwargs
        )
        for curve, init_params, bound in zip(curves, init_params, bounds)
    )

    params_array, error_bars, curves_array = zip(*results)
    return np.array(params_array), np.array(error_bars), np.array(curves_array)

import warnings
from scipy.optimize import curve_fit

def refl_fit(
        q: np.ndarray, 
        curve: np.ndarray,
        init_params: np.ndarray,
        prior_sampler: PriorSampler,
        bounds: np.ndarray = None,
        error_bars: np.ndarray = None,
        scale_curve_func=np.log10,
        method: str = 'trf', #'lm', 'trf'
        polishing_max_steps: int = None,
        reflectivity_kwargs: dict = None,
        **kwargs
):
    # --- 1. Bounds Handling ---
    if bounds is not None:
        if bounds.ndim != 2:
            raise ValueError(f"Bounds must be a 2D array, got {bounds.ndim}D array")
        
        # Introduce a small perturbation for fixed bounds to prevent optimizer issues
        epsilon = 1e-6
        adjusted_bounds = bounds.copy()
        for i in range(bounds.shape[1]): 
            if bounds[0, i] == bounds[1, i]:
                adjusted_bounds[0, i] -= epsilon
                adjusted_bounds[1, i] += epsilon

        init_params = np.clip(init_params, *adjusted_bounds)
        
        if method != 'lm':
            # Scipy expects a tuple (lower_bound, upper_bound)
            # Assuming adjusted_bounds shape is (2, n_params)
            kwargs['bounds'] = (adjusted_bounds[0], adjusted_bounds[1])

    # --- 2. Input Scaling & Prep ---
    reflectivity_kwargs = reflectivity_kwargs or {}
    for key, value in reflectivity_kwargs.items():
        if isinstance(value, float):
            reflectivity_kwargs[key] = torch.tensor([[value]], dtype=torch.float64)
        elif isinstance(value, np.ndarray):
            reflectivity_kwargs[key] = torch.tensor(value, dtype=torch.float32).unsqueeze(0)      

    curve = np.clip(curve, a_min=1e-12, a_max=None)

    if error_bars is not None and scale_curve_func == np.log10:
        error_bars = np.clip(error_bars, a_min=1e-20, a_max=None)
        scaled_error_bars = error_bars / (curve * np.log(10))
    else:
        scaled_error_bars = None  

    # Define maxfev/max_nfev
    if polishing_max_steps is not None:
        if method == 'lm':
            kwargs['maxfev'] = polishing_max_steps
        else:
            kwargs['max_nfev'] = polishing_max_steps

    # --- 3. Robust Fitting (Try...Except) ---
    fit_func = get_scaled_curve_func(
        scale_curve_func=scale_curve_func,
        prior_sampler=prior_sampler,
        reflectivity_kwargs=reflectivity_kwargs,
    )
    
    y_data_flat = scale_curve_func(curve).reshape(-1)

    try:
        # Attempt the fit
        popt, pcov = curve_fit(
            f=fit_func,
            xdata=q, 
            ydata=y_data_flat,
            p0=init_params,
            sigma=scaled_error_bars,
            absolute_sigma=True,
            method=method,
            **kwargs
        )
    except RuntimeError:
        # FALLBACK: If fit fails (maxfev exceeded), use initial params
        # You could optionally retry here with higher max_nfev if desired
        warnings.warn("Optimization failed to converge. Returning initial parameters.")
        popt = init_params
        pcov = None

    # --- 4. Result Construction ---
    # Re-calculate the curve using the "best" parameters found (popt)
    curve_fitted = prior_sampler.param_model.reflectivity(
        torch.tensor(q, dtype=torch.float64), 
        torch.tensor(popt, dtype=torch.float64).unsqueeze(0), 
        **reflectivity_kwargs
    ).squeeze().numpy()

    # Calculate error bars (check if covariance matrix is valid)
    if pcov is not None and np.ndim(pcov) == 2 and np.all(np.isfinite(pcov)):
        pol_param_errs = np.sqrt(np.diag(pcov))
    else: 
        # Return NaNs if fit failed or covariance is infinite
        pol_param_errs = np.full_like(popt, np.nan)

    return popt, pol_param_errs, curve_fitted
    



def get_fit_with_growth(
        q: np.ndarray, 
        curve: np.ndarray,
        init_params: np.ndarray,
        bounds: np.ndarray = None,
        init_d_change: float = 0.,
        max_d_change: float = 30.,
        scale_curve_func=np.log10,
        **kwargs
):
    init_params = np.array(list(init_params) + [init_d_change])
    if bounds is not None:
        bounds = np.concatenate([bounds, np.array([0, max_d_change])[..., None]], -1)

    params, curve = standard_refl_fit(
        q, 
        curve, 
        init_params, 
        bounds, 
        refl_generator=growth_reflectivity,
        restore_params_func=get_restore_params_with_growth_func(q_size=q.size, d_idx=0),
        scale_curve_func=scale_curve_func, 
        **kwargs
    )
    
    params[0] += params[-1] / 2
    return params, curve


def fit_refl_curve(q: np.ndarray, curve: np.ndarray,
                   init_params: np.ndarray,
                   bounds: np.ndarray = None,
                   refl_generator=abeles_np,
                   restore_params_func=standard_restore_params,
                   scale_curve_func=np.log10,
                   **kwargs
                   ) -> np.ndarray:
    fitting_func = get_fitting_func(
        q=q, curve=curve,
        refl_generator=refl_generator,
        restore_params_func=restore_params_func,
        scale_curve_func=scale_curve_func,
    )

    res = minimize(fitting_func, init_params, bounds=bounds, **kwargs)

    if not res.success:
        warnings.warn(f"Minimization did not converge.")
    return res.x

def standard_get_scaled_curve_func(
        refl_generator=abeles_np,
        restore_params_func=standard_restore_params,
        scale_curve_func=np.log10,
):
    def scaled_curve_func(q, *fitted_params):
        fitted_params = restore_params_func(np.asarray(fitted_params))
        fitted_curve = refl_generator(q, **fitted_params)
        scaled_curve = scale_curve_func(fitted_curve)
        return scaled_curve

    return scaled_curve_func

def get_scaled_curve_func(
        scale_curve_func=np.log10,
        prior_sampler: PriorSampler = None,
        reflectivity_kwargs: dict = None,
):  
    reflectivity_kwargs = reflectivity_kwargs or {}
    
    def scaled_curve_func(q, *fitted_params):
        q_tensor = torch.from_numpy(q).to(torch.float64)
        fitted_params_tensor = torch.tensor(fitted_params, dtype=torch.float64).unsqueeze(0)
        
        fitted_curve_tensor = prior_sampler.param_model.reflectivity(q_tensor, fitted_params_tensor, **reflectivity_kwargs)
        fitted_curve = fitted_curve_tensor.squeeze().numpy()
        
        scaled_curve = scale_curve_func(fitted_curve)

        return scaled_curve.reshape(-1)

    return scaled_curve_func


def get_fitting_func(
        q: np.ndarray,
        curve: np.ndarray,
        refl_generator=abeles_np,
        restore_params_func=standard_restore_params,
        scale_curve_func=np.log10,
        loss_func=mse_loss,
):
    scaled_curve = scale_curve_func(curve)

    def fitting_func(fitted_params):
        fitted_params = restore_params_func(fitted_params)
        fitted_curve = refl_generator(q, **fitted_params)
        loss = loss_func(scale_curve_func(fitted_curve), scaled_curve)
        return loss

    return fitting_func


def restore_masked_params(fixed_params, fixed_mask):
    def restore_params(fitted_params) -> dict:
        params = np.empty_like(fixed_mask).astype(fitted_params.dtype)
        params[fixed_mask] = fixed_params
        params[~fixed_mask] = fitted_params
        return standard_restore_params(params)

    return restore_params


def base_params2growth(base_params: dict, d_shift: np.ndarray, d_idx: int = 0) -> dict:
    d_init = base_params['thickness'][None]
    q_size = d_shift.size
    d = d_init.repeat(q_size, 0)
    d[:, d_idx] = d[:, d_idx] + d_shift

    roughness = np.broadcast_to(base_params['roughness'][None], (q_size, base_params['roughness'].size))
    sld = np.broadcast_to(base_params['sld'][None], (q_size, base_params['sld'].size))

    return {
        'thickness': d,
        'roughness': roughness,
        'sld': sld,
    }


def get_restore_params_with_growth_func(q_size: int, d_idx: int = 0):
    def restore_params_with_growth(fitted_params) -> dict:
        fitted_params, delta_d = fitted_params[:-1], fitted_params[-1]
        base_params = standard_restore_params(fitted_params)
        d_shift = np.linspace(0, delta_d, q_size)
        return base_params2growth(base_params, d_shift, d_idx)

    return restore_params_with_growth


def growth_reflectivity(q: np.ndarray, **kwargs):
    return abeles_np(q[..., None], **kwargs).flatten()
