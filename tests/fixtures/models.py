import pytest

from reflectorch.inference import InferenceModel


@pytest.fixture(
    params=['l2q64_new_sub_1'],
)
def model_name(request):
    return request.param


@pytest.fixture
def inference_model(model_name):
    return InferenceModel(model_name)
