from typing import Tuple

import torch
from torch import Tensor
import numpy as np

from reflectorch.inference.inference_model import (
    InferenceModel,
    get_prediction_array,
)
from reflectorch.data_generation.utils import get_density_profiles, get_param_labels
from reflectorch.data_generation.reflectivity import abeles_np

from reflectorch.data_generation.priors import (
    MultilayerStructureParams,
    SimpleMultilayerSampler,
)
from reflectorch.inference.record_time import print_time
from reflectorch.inference.sampler_solution import simple_sampler_solution
from reflectorch.inference.scipy_fitter import standard_refl_fit


class MultilayerInferenceModel(InferenceModel):
    def predict(self,
                intensity: np.ndarray,
                scattering_angle: np.ndarray,
                attenuation: np.ndarray,
                priors: np.ndarray = None,
                preprocessing_parameters: dict = None,
                polish: bool = True,
                use_raw_q: bool = False,
                **kwargs
                ) -> dict:

        with print_time("everything"):
            with print_time("preprocess"):
                preprocessed_dict = self.preprocess(
                    intensity, scattering_angle, attenuation, **(preprocessing_parameters or {})
                )

            preprocessed_curve = preprocessed_dict["curve_interp"]

            if use_raw_q:
                raw_curve, raw_q = preprocessed_dict["curve"], preprocessed_dict["q_values"]
            else:
                raw_curve, raw_q = None, None

            with print_time("predict_from_preprocessed_curve"):
                preprocessed_dict.update(self.predict_from_preprocessed_curve(
                    preprocessed_curve, priors,
                    raw_curve=raw_curve,
                    raw_q=raw_q,
                    polish=polish,
                ))

            return preprocessed_dict

    def predict_from_preprocessed_curve(self,
                                        curve: np.ndarray,
                                        priors: np.ndarray = None, *,  # ignore the priors so far
                                        polish: bool = True,
                                        raw_curve: np.ndarray = None,
                                        raw_q: np.ndarray = None,
                                        clip_prediction: bool = True,
                                        **kwargs
                                        ) -> dict:

        scaled_curve = self._scale_curve(curve)

        predicted_params, parametrized = self._simple_prediction(scaled_curve)

        # if use_sampler:
        #     predicted_params: MultilayerStructureParams = self._sampler_solution(
        #         curve, predicted_params,
        #     )

        # if clip_prediction:
        #     predicted_params = self._prior_sampler.clamp_params(predicted_params)

        if raw_curve is None:
            raw_curve = curve
        if raw_q is None:
            raw_q = self.q.squeeze().cpu().numpy()
            raw_q_t = self.q
        else:
            raw_q_t = torch.from_numpy(raw_q).to(self.q)

        # if q_ratio != 1.:
        #     predicted_params.scale_with_q(q_ratio)
        #     raw_q = raw_q * q_ratio
        #     raw_q_t = raw_q_t * q_ratio

        prediction_dict = {
            "params": get_prediction_array(predicted_params),
            "param_names": list(self._prior_sampler.multilayer_model.PARAMETER_NAMES),
            "curve_predicted": predicted_params.reflectivity(raw_q_t).squeeze().cpu().numpy()
        }

        # sld_x_axis, sld_profile, _ = get_density_profiles(
        #     predicted_params.thicknesses, predicted_params.roughnesses, predicted_params.slds, num=1024,
        # )
        #
        # prediction_dict['sld_profile'] = sld_profile.squeeze().cpu().numpy()
        # prediction_dict['sld_x_axis'] = sld_x_axis.squeeze().cpu().numpy()

        if polish:
            prediction_dict.update(self._polish_prediction(
                raw_q, raw_curve, parametrized, q_values=prediction_dict['q_values']
            ))

        return prediction_dict

    def _simple_prediction(self, scaled_curve) -> Tuple[MultilayerStructureParams, Tensor]:
        with torch.no_grad():
            self.trainer.model.eval()
            scaled_params = self.trainer.model(scaled_curve)

        predicted_params, parametrized = self._restore_predicted_params(scaled_params)
        return predicted_params, parametrized

    def _restore_predicted_params(self, scaled_params: Tensor) -> Tuple[MultilayerStructureParams, Tensor]:
        parametrized = self._prior_sampler.restore_params2parametrized(scaled_params)
        predicted_params: MultilayerStructureParams = self._prior_sampler.restore_params(scaled_params)
        return predicted_params, parametrized

    @print_time
    def _sampler_solution(
            self,
            curve: Tensor or np.ndarray,
            predicted_params: MultilayerStructureParams,
    ) -> MultilayerStructureParams:

        if not isinstance(curve, Tensor):
            curve = torch.from_numpy(curve).float()
        curve = curve.to(self.q)

        refined_params = simple_sampler_solution(
            self._get_likelihood(curve),
            predicted_params,
            self._prior_sampler.min_bounds,
            self._prior_sampler.max_bounds,
            num=self._sampling_num, coef=0.1,
        )

        return refined_params

    @property
    def _prior_sampler(self) -> SimpleMultilayerSampler:
        return self.trainer.loader.prior_sampler

    @print_time
    def _polish_prediction(self,
                           q: np.ndarray,
                           curve: np.ndarray,
                           predicted_params: Tensor,
                           q_values: np.ndarray
                           ) -> dict:

        params = predicted_params.squeeze().cpu().numpy()
        polished_params_dict = {}

        try:
            polished_params_arr, curve_polished = standard_refl_fit(
                q, curve, params, restore_params_func=self._prior_sampler.restore_np_params,
                bounds=self._prior_sampler.get_np_bounds(),
            )
            params = self._prior_sampler.restore_np_params(polished_params_arr)
            curve_polished = abeles_np(q_values, **params)

        except Exception as err:
            self.log.exception(err)
            polished_params_arr = params
            curve_polished = np.zeros_like(q)

        polished_params_dict['params_polished'] = polished_params_arr
        polished_params_dict['curve_polished'] = curve_polished

        return polished_params_dict