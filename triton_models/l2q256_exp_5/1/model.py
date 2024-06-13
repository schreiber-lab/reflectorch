import numpy as np

from reflectorch.inference import InferenceModel

try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    import warnings

    warnings.warn("triton_python_backend_utils is not installed!")
    pb_utils = None


class TritonPythonModel:
    INPUT_KEYS = {
        "tth": "scattering_angle",
        "intensity": "intensity",
        "transm": "attenuation",
        "priors": "priors",
    }

    PREPROCESS_KEYS = (
        "beam_width",
        "sample_length",
        "wavelength",
    )

    POSTPROCESS_KEYS = (
        "fit_growth",
        "max_d_change",
    )

    OUTPUT_KEYS_NP32 = {
        "q": "q_values",
        "q_interp": "q_interp",
        "refl": "curve",
        "refl_interp": "curve_interp",
        "refl_predicted": "curve_predicted",
        "refl_predicted_polished": "curve_predicted_polished",
        "parameters": "params",
        "parameters_polished": "params_polished",
        "sld_profile": "sld_profile",
        "sld_x_axis": "sld_x_axis",
        "sld_profile_polished": "sld_profile_polished",
    }

    OUTPUT_KEYS_STR = {
        "parameter_names": "param_names",
    }

    def initialize(self, args):
        self.model = InferenceModel('l2q256_exp_5')

    def execute(self, requests):
        """
        Request is expected to be of type
        Tuple[]
        :param request:
        :return:
        """

        responses = [self.process_request(request) for request in requests]
        return responses

    def process_request(self, request):
        preprocess_dict = _get_input_dict(request, self.PREPROCESS_KEYS)
        predict_input = _get_input_dict(request, self.INPUT_KEYS)
        postprocess_dict = _get_input_dict(request, self.POSTPROCESS_KEYS)

        self.model.set_preprocessing_parameters(**preprocess_dict)

        parameter_dict = self.model.predict(
            **predict_input,
            **postprocess_dict,
        )

        default_arr = - np.ones(256)

        response_tensors = pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor(client_key, np.asarray(parameter_dict.get(server_key, default_arr)).astype(np.float32))
                for client_key, server_key in self.OUTPUT_KEYS_NP32.items()
            ])
        #                    + [
        #         pb_utils.Tensor(client_key, np.asarray(parameter_dict[server_key]).astype(np.str))
        #         for client_key, server_key in self.OUTPUT_KEYS_STR.items()
        #     ]
        # )

        return response_tensors


def _get_input_dict(request, keys):
    if isinstance(keys, dict):
        return {v: _to_scalar(pb_utils.get_input_tensor_by_name(request, k).as_numpy()) for k, v in keys.items()}
    return {key: _to_scalar(pb_utils.get_input_tensor_by_name(request, key).as_numpy()) for key in keys}


def _to_scalar(value):
    if np.asarray(value).size == 1:
        value = value.item()
    return value
