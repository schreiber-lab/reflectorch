import pytest
from reflectorch import ConstantQ, BasicExpIntensityNoise, BasicQNoiseGenerator, LogAffineCurvesScaler, SubpriorParametricSampler, \
      Smearing, BasicDataset, ReflectivityDataLoader

@pytest.mark.parametrize("calc_denoised_curves", [False, True])
def test_dataloader(calc_denoised_curves):
    prior_sampler = SubpriorParametricSampler(
        param_ranges={
            'thicknesses': [5.0, 300.0],
            'roughnesses': [0.0, 30.0],
            'slds': [0.0, 50.0],
        },
        bound_width_ranges={
            'thicknesses': [1e-2, 300.0],
            'roughnesses': [1e-2, 30.0],
            'slds': [1e-2, 5.0],
        },
        model_name = 'standard_model',
        device='cpu',
        max_num_layers=3,
        constrained_roughness = True,
        max_thickness_share = 0.5,
        scale_params_by_ranges=False,
    )

    q_generator = ConstantQ(
    q = [0.02, 0.15, 128], 
    device = 'cpu')

    intensity_noise_generator = BasicExpIntensityNoise(
    relative_errors = [0.0, 0.2],
    abs_errors = 0.0,
    consistent_rel_err = True,
    logdist = False,
    apply_shift = True,
    shift_range = [-0.3, 0.3],
    apply_scaling = True,
    scale_range = [-0.02, 0.02],
    apply_background = True,
    background_range = [1.0e-10, 1.0e-8],
    )

    q_noise_generator = BasicQNoiseGenerator(
        shift_std = 1.0e-3,
        noise_std = [0., 1.0e-3],
    )

    smearing_generator = Smearing(
        sigma_range = (0.0001, 0.005), 
        constant_dq = True, 
        gauss_num = 31, 
        share_smeared = 0.2,
    )

    curves_scaler = LogAffineCurvesScaler(
        weight = 0.2,
        bias = 1.0,
        eps = 1.0e-10,
    )

    data_loader = ReflectivityDataLoader(
        q_generator=q_generator, 
        prior_sampler=prior_sampler,
        intensity_noise=intensity_noise_generator,
        q_noise=q_noise_generator,
        smearing=smearing_generator,
        curves_scaler=curves_scaler,
        calc_denoised_curves=calc_denoised_curves,
        )