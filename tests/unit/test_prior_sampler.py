import pytest
import torch
from reflectorch import SubpriorParametricSampler

@pytest.mark.parametrize("batch_size", [1, 128])
@pytest.mark.parametrize("max_num_layers", [1, 2, 3, 4, 5, 10])
@pytest.mark.parametrize("constrained_roughness", [True, False])
def test_param_sampler_standard(batch_size, max_num_layers, constrained_roughness):
    param_sampler = SubpriorParametricSampler(
        param_ranges={
            'thicknesses': [10.0, 500.0],
            'roughnesses': [0.0, 60.0],
            'slds': [-20.0, 100.0],
        },
        bound_width_ranges={
            'thicknesses': [1e-2, 500.0],
            'roughnesses': [1e-2, 60.0],
            'slds': [1e-2, 10.0],
        },
        model_name = 'standard_model',
        device='cpu',
        max_num_layers=max_num_layers,
        constrained_roughness = constrained_roughness,
        max_thickness_share = 0.5,
        scale_params_by_ranges=False,
    )

    sampled_params = param_sampler.sample(batch_size=batch_size)
    assert sampled_params.parameters.shape == (batch_size, 3*max_num_layers+2)
    assert torch.all(sampled_params.min_bounds <= sampled_params.parameters ) and torch.all(sampled_params.parameters <= sampled_params.max_bounds)


@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("max_num_layers", [1, 3, 5])
@pytest.mark.parametrize("constrained_roughness", [True, False])
@pytest.mark.parametrize("constrained_isld", [True, False])
def test_param_sampler_absorption(batch_size, max_num_layers, constrained_roughness, constrained_isld):
    param_sampler = SubpriorParametricSampler(
        param_ranges={
            'thicknesses': [10.0, 500.0],
            'roughnesses': [0.0, 60.0],
            'slds': [-20.0, 100.0],
            'islds': [0.0, 10.0],
        },
        bound_width_ranges={
            'thicknesses': [1e-2, 500.0],
            'roughnesses': [1e-2, 60.0],
            'slds': [1e-3, 20.0],
            'islds': [1e-2, 10.0],
        },
        model_name = 'model_with_absorption',
        device='cpu',
        max_num_layers=max_num_layers,
        constrained_roughness = constrained_roughness,
        max_thickness_share = 0.5,
        constrained_isld = constrained_isld,
        max_sld_share = 0.2,
        scale_params_by_ranges=False,
    )

    sampled_params = param_sampler.sample(batch_size=batch_size)
    assert sampled_params.parameters.shape == (batch_size, 4*max_num_layers+3)
    assert torch.all(sampled_params.min_bounds <= sampled_params.parameters ) and torch.all(sampled_params.parameters <= sampled_params.max_bounds)