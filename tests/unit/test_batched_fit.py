
import numpy as np

from reflectorch.inference.scipy_fitter import batch_refl_fit, refl_fit


def test_batch_refl_fit(data_batch, dataset):
    prior_sampler = dataset.prior_sampler
    q = data_batch['q_values'].numpy()[0]
    noisy_curves = data_batch['noisy_curves'].numpy()
    params = data_batch['params']
    init_params = params.parameters.cpu().numpy()
    min_bounds = params.min_bounds.cpu().numpy()
    max_bounds = params.max_bounds.cpu().numpy()

    assert np.all(init_params >= min_bounds) and np.all(init_params <= max_bounds)

    bounds = np.stack([min_bounds, max_bounds], 1)

    params_loop, curves_loop = _get_refl_fit_in_a_loop(q, noisy_curves, init_params, bounds, prior_sampler)

    params_batch, curves_batch = batch_refl_fit(
        q=q,
        curves=noisy_curves,
        init_params=init_params,
        bounds=bounds,
        prior_sampler=prior_sampler,
    )
    assert params_loop.shape == init_params.shape
    assert curves_loop.shape == noisy_curves.shape
    assert params_batch.shape == init_params.shape
    assert curves_batch.shape == noisy_curves.shape
    assert np.allclose(params_loop, params_batch)
    assert np.allclose(curves_loop, curves_batch)


def _get_refl_fit_in_a_loop(q, noisy_curves, init_params, bounds, prior_sampler):
    params_list = []
    curves_list = []
    for noisy_curve, init_params, bound in zip(noisy_curves, init_params, bounds):
        params, curve = refl_fit(
            q=q, curve=noisy_curve, init_params=init_params, 
            bounds=bound, prior_sampler=prior_sampler
        )
        params_list.append(params)
        curves_list.append(curve)
    return np.array(params_list), np.array(curves_list)
