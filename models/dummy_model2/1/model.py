import numpy as np

from reflectorch.inference import InferenceModel

try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    import warnings

    warnings.warn("triton_python_backend_utils is not installed!")
    pb_utils = None


class TritonPythonModel:
    REFL_KEYS = (
        "preprocessed_curve",
        "priors",
    )

    OUTPUT_KEYS = (
        "params",
        "curve_predicted",
        "sld_x_axis",
        "sld_profile",
    )

    def initialize(self, args):
        self.model = InferenceModel('l2q64_new_sub_1')

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
        refl_input = {key: pb_utils.get_input_tensor_by_name(request, key).as_numpy() for key in self.REFL_KEYS}

        parameter_dict = self.model.predict_from_preprocessed_curve(
            curve=refl_input["preprocessed_curve"], priors=refl_input["priors"], polish=True
        )

        response_tensors = pb_utils.InferenceResponse(
            output_tensors=[pb_utils.Tensor(key, parameter_dict[key].astype(np.float32)) for key in self.OUTPUT_KEYS]
        )

        return response_tensors
