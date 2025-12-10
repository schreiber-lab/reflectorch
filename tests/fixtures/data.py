import pytest
import numpy as np

from reflectorch.inference.loading_data import load_mft_data

from reflectorch.paths import TEST_DATA_PATH

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
    params=["test_data.mft"],
    scope="session"
)
def mft_data_filepath(request):
    return TEST_DATA_PATH / request.param


@pytest.fixture(scope="session")
def mft_data(mft_data_filepath):
    return load_mft_data(mft_data_filepath)


@pytest.fixture(scope="session")
def simulator():
    prior_bounds = [
        (1., 400.), (1., 10.),                 # thickness (L2, L1)
        (0., 20.), (0., 15.), (0., 15.),       # roughness (Air/L2, L2/L1, L1/Sub)
        (10., 13.), (20., 21.), (20., 21.)     # SLDs (L2, L1, Sub)
    ]



