import pytest
import torch
from reflectorch import ResidualMLP


@pytest.mark.parametrize("dim_in", [256])
@pytest.mark.parametrize("dim_out", [10])
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("repeats_per_block", [1, 2])
@pytest.mark.parametrize("activation", ['relu', 'rowdy'])
@pytest.mark.parametrize("dropout_rate", [0, 0.2])
def test_mlp(dim_in, dim_out, residual, repeats_per_block, activation, dropout_rate):
    adaptive_activation = True if activation == 'rowdy' else False

    mlp = ResidualMLP(
        dim_in = dim_in,
        dim_out = dim_out,
        dim_condition = 0,
        layer_width = 128,
        num_blocks = 4,
        repeats_per_block= repeats_per_block,
        activation = activation,
        use_batch_norm = True,
        dropout_rate = dropout_rate,
        residual = residual,
        adaptive_activation = adaptive_activation,
        conditioning = 'concat',
    )

    batch_size = 8
    input_tensor = torch.randn((batch_size, dim_in))

    assert mlp(input_tensor).shape == (batch_size, dim_out)

