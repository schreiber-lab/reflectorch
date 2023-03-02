import numpy as np


def test_predict_from_preprocessed_curve_inference_model(inference_model, preprocessed_data):
    predicted_dict = inference_model.predict_from_preprocessed_curve(
        preprocessed_data['preprocessed_curve'], preprocessed_data['priors']
    )
    min_bounds, max_bounds = preprocessed_data['priors'].T.cpu().numpy()
    predicted_params = predicted_dict['params']
    assert predicted_params.shape == preprocessed_data['params'].shape
    assert np.all(max_bounds >= predicted_params)
    assert np.all(min_bounds <= predicted_params)
    assert predicted_dict['sld_profile'].shape == predicted_dict['sld_x_axis'].shape
    assert predicted_dict['curve_predicted'].shape == preprocessed_data['preprocessed_curve'].shape

#
# def test_predict_inference_model(inference_model, raw_data_input):
#     pass
