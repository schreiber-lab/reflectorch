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
        "curve",
        "priors",
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

        parameters = self.model.predict_from_preprocessed_curve(**refl_input)

        response_tensors = pb_utils.InferenceResponse(
            output_tensors=[pb_utils.Tensor("parameters", parameters.astype(np.float32))]
        )

        return response_tensors
