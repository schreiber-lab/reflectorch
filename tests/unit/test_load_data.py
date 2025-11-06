from reflectorch.inference.loading_data import load_mft_data
from reflectorch.paths import TEST_DATA_PATH

def test_load_mft_data(mft_data):
    data = mft_data
    assert len(data.shape) == 2
    assert data.shape[0] == 4
    assert data.shape[1] > 10
