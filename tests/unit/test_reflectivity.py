import pytest
import torch
from reflectorch.data_generation.reflectivity import reflectivity, abeles, abeles_memory_eff, kinematical_approximation

@pytest.mark.parametrize("abeles_func", [abeles, abeles_memory_eff, kinematical_approximation])
def test_reflectivity_two_layers(abeles_func):

    thickness_tensor = torch.tensor([
        [50, 100], # element 1 in the batch (n_layers thicknesses from top/ambient to bottom/substrate)
        [80, 40], # element 2 in the batch
        [600, 100], # element 3 in the batch
    ])
    
    roughness_tensor = torch.tensor([
        [1.0, 5.0, 3.0], # (n_layers + 1 roughnesses from top/ambient to bottom/substrate)
        [15.5, 1.0, 2.0],
        [0.0, 0.0, 50.0],
    ])

    sld_tensor = torch.tensor([
        [20.0, 12.1, 18.5], # (n_layers + 1 slds from top/ambient to bottom/substrate)
        [-5.0, 10.5, 12.0],
        [15.0 + 1j*5, 5.99 + 1j*0.09, 25.0],
    ])

    q_min = 0.02
    q_max = 0.15
    n_q = 128 
    q = torch.linspace(q_min, q_max, n_q)

    refl_curves = reflectivity(
        q=q, 
        thickness=thickness_tensor, 
        roughness=roughness_tensor, 
        sld=sld_tensor,
        abeles_func=abeles_func,
    )
    
    batch_size = thickness_tensor.shape[0]

    assert refl_curves.shape == (batch_size, n_q)


@pytest.mark.parametrize("abeles_func", [abeles, abeles_memory_eff, kinematical_approximation])
def test_reflectivity_four_layers(abeles_func):

    thickness_tensor = torch.tensor([
        [50, 100, 10, 100],
        [80, 40, 1, 60],
        [600, 100, 50, 50],
    ])
    
    roughness_tensor = torch.tensor([
        [1.0, 5.0, 3.0, 0.5, 4.0],
        [15.5, 1.0, 2.0, 0.5, 4.0],
        [0.0, 0.0, 50.0, 12.3, 6.2],
    ])

    sld_tensor = torch.tensor([
        [20.0, 12.1, 18.5, 1.0, 25.0],
        [-5.0, 10.5, 12.0, -2.15, 15.0],
        [15.0 + 1j*5, 5.99 + 1j*0.09, 25.0, 10.1 + 1j*4.3, 16.0],
    ])

    q_min = 0.01
    q_max = 0.3
    n_q = 200
    q = torch.linspace(q_min, q_max, n_q)

    refl_curves = reflectivity(
        q=q, 
        thickness=thickness_tensor, 
        roughness=roughness_tensor, 
        sld=sld_tensor,
        abeles_func=abeles_func,
    )
    
    batch_size = thickness_tensor.shape[0]

    assert refl_curves.shape == (batch_size, n_q)

def test_abeles_vs_abelesmemoryeff():

    thickness_tensor = torch.tensor([
        [50, 100, 10, 100],
        [80, 40, 1, 60],
        [600, 100, 50, 50],
    ])
    
    roughness_tensor = torch.tensor([
        [1.0, 5.0, 3.0, 0.5, 4.0],
        [15.5, 1.0, 2.0, 0.5, 4.0],
        [0.0, 0.0, 50.0, 12.3, 6.2],
    ])

    sld_tensor = torch.tensor([
        [20.0, 12.1, 18.5, 1.0, 25.0],
        [-5.0, 10.5, 12.0, -2.15, 15.0],
        [15.0 + 1j*5, 5.99 + 1j*0.09, 25.0, 10.1 + 1j*4.3, 16.0],
    ])

    q_min = 0.01
    q_max = 0.3
    n_q = 200
    q = torch.linspace(q_min, q_max, n_q)

    refl_curves_abeles = reflectivity(
        q=q, 
        thickness=thickness_tensor, 
        roughness=roughness_tensor, 
        sld=sld_tensor,
        abeles_func=abeles,
    )
    
    refl_curves_abeles_memory_eff = reflectivity(
        q=q, 
        thickness=thickness_tensor, 
        roughness=roughness_tensor, 
        sld=sld_tensor,
        abeles_func=abeles_memory_eff,
    )

    assert torch.allclose(refl_curves_abeles, refl_curves_abeles_memory_eff, atol=1e-10)