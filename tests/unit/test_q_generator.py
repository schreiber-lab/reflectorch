import pytest
from reflectorch import ConstantQ, VariableQ


@pytest.mark.parametrize("batch_size", [1, 128])
@pytest.mark.parametrize("q_range", [(0.01, 0.3, 128), (0.05, 0.1, 200)])
def test_constantq(batch_size, q_range):
    q_generator = ConstantQ(q=q_range, device='cpu')
    q_values = q_generator.get_batch(batch_size=batch_size)

    assert q_values.shape == (batch_size, q_range[-1])
    assert q_values.min().float() == q_range[0]
    assert q_values.max().float() == q_range[1]

@pytest.mark.parametrize("batch_size", [1, 128])
@pytest.mark.parametrize("q_min_range", [(0.005, 0.05), (0.01, 0.01)])
@pytest.mark.parametrize("q_max_range", [(0.15, 0.5), (0.3, 0.3)])
@pytest.mark.parametrize("n_q_range", [(64, 256), (128, 128)])
def test_variableq(batch_size, q_min_range, q_max_range, n_q_range):
    q_generator = VariableQ(q_min_range, q_max_range, n_q_range, device='cpu')
    q_values = q_generator.get_batch(batch_size=batch_size)

    assert n_q_range[0] <= q_values.shape[-1] <= n_q_range[1]
    assert q_values.min().float() >= q_min_range[0]
    assert q_values.max().float() <= q_max_range[1]