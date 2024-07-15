from reflectorch import ConstantQ, VariableQ


def test_constantq():
    q_range = (0.01, 0.3, 128)
    q_generator = ConstantQ(q=q_range, device='cpu')

    batch_size = 32
    q_values = q_generator.get_batch(batch_size=batch_size)

    assert q_values.shape == (batch_size, q_range[-1])
    assert q_values.min() == q_range[0]
    assert q_values.max() == q_range[1]

def test_variableq():
    q_min_range = (0.005, 0.05)
    q_max_range = (0.15, 0.5)
    n_q_range = (64, 256)
    q_generator = VariableQ(q_min_range, q_max_range, n_q_range, device='cpu')

    batch_size = 32
    q_values = q_generator.get_batch(batch_size=batch_size)

    assert n_q_range[0] <= q_values.shape[-1] <= n_q_range[1]
    assert q_values.min() >= q_min_range[0]
    assert q_values.max() <= q_max_range[1]
