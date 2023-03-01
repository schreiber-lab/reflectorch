import pytest

import numpy as np


def test_inference_model(inference_model, preprocessed_data):
    predicted_dict = inference_model.predict_from_preprocessed_curve(
        preprocessed_data['preprocessed_curve'], preprocessed_data['priors']
    )
    min_bounds, max_bounds = preprocessed_data['priors'].T
    predicted_params = predicted_dict['params']
    assert predicted_params.shape == preprocessed_data['params'].shape
    assert np.all(max_bounds >= predicted_params)
    assert np.all(min_bounds <= predicted_params)
    assert predicted_dict['sld_profile'].shape == predicted_dict['sld_x_axis'].shape

