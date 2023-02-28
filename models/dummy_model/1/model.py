import numpy as np

from reflectorch.inference import InferenceModel

try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    import warnings
    warnings.warn("triton_python_backend_utils is not installed!")
    pb_utils = None


class MLServer(object):
    REFL_KEYS = (
        "intensity",
        "scattering_angle",
        "attenuation",
    )

    PREPROCESS_KEYS = (
        "wavelength",
        "beam_width",
        "sample_length",
    )

    def get_dummy_input(self):
        res = {key: np.zeros(100) for key in self.REFL_KEYS}
        res.update({key: 1. for key in self.PREPROCESS_KEYS})
        res['model_name'] = 'model_1'
        return res

    def initialize(self):
        self.model = InferenceModel()

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
        refl_input = {key: request.get_input_tensor_by_name(request, key).as_numpy() for key in self.REFL_KEYS}

        preprocessing_input = {key: pb_utils.get_input_tensor_by_name(request, key).as_numpy()[0] for key in
                               self.PREPROCESS_KEYS}

        model_name = str(pb_utils.get_input_tensor_by_name(request, "model_name").as_numpy()[0])

        self.model.set_preprocessing_parameters(**preprocessing_input)

        res = {
            "curve": np.zeros(10),
            "curve_interp": np.zeros(10),
            "q_values": np.zeros(10),
            "q_interp": np.zeros(10),
            "parameters": np.zeros(10),
        }

        response_tensors = pb_utils.InferenceResponse(
            output_tensors=[pb_utils.Tensor(key, value.astype(np.float32)) for key, value in res.items()]
        )

        return response_tensors
