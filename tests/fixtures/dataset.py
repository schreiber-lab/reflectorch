import pytest
import torch

from reflectorch.paths import TEST_DATA_PATH
from reflectorch.runs.utils import init_dset
from reflectorch.runs.config import load_config

@pytest.fixture(
    params=["test_config.yaml"],
    scope="session"
)
def dataset(request):
    config = load_config(request.param, config_dir=TEST_DATA_PATH)
    return init_dset(config['dset'])


@pytest.fixture(scope="session")
def data_batch(dataset):
    dataset.calc_denoised_curves = True
    dataset.add_noisy_curves = True
    batch = dataset.get_batch(batch_size=2)
    # move to cpu
    batch = {
        key: value.cpu() 
        if isinstance(value, torch.Tensor) else value 
        for key, value in batch.items()}
    return batch
