import pytest
import numpy as np
from reflectorch.paths import ROOT_DIR

TEST_DATA_PATH = ROOT_DIR / 'tests' / 'data'


@pytest.fixture(
    params=["demo_scan.csv"],
    scope="session"
)
def raw_data_with_preprocessing_params(request):
    scan_path = TEST_DATA_PATH / request.param
    data = np.loadtxt(str(scan_path), skiprows=1, delimiter=',')

    raw_data = dict(
        intensity=data[:, 5],
        scattering_angle=data[:, 0] * 2,
        attenuation=data[:, 4],
    )

    preprocessing_params = dict(
        q_interp=np.linspace(0.02, 0.15, 64),
        wavelength=1,
        beam_width=0.2,
        sample_length=10.,
    )

    return raw_data, preprocessing_params


@pytest.fixture(
    params=['test_preprocessed_curve_1'],
    scope="session",
)
def preprocessed_data(request):
    path = TEST_DATA_PATH / f'{request.param}.npz'
    data = dict(np.load(str(path)))
    return data