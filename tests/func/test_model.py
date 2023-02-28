import pytest

import numpy as np


def test_inference_model(inference_model, preprocessed_data):
    predicted_params = inference_model.predict_from_preprocessed_curve(
        preprocessed_data['preprocessed_curve'], preprocessed_data['priors']
    )
    min_bounds, max_bounds = preprocessed_data['priors'].T
    assert predicted_params.shape == preprocessed_data['params'].shape
    assert np.all(max_bounds >= predicted_params)
    assert np.all(min_bounds <= predicted_params)
