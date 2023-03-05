from typing import Tuple

import torch
from torch import Tensor
import numpy as np

from reflectorch.inference.inference_model import (
    InferenceModel,
)
from reflectorch.data_generation.reflectivity import kinematical_approximation_np, abeles_np

from reflectorch.data_generation.priors import (
    MultilayerStructureParams,
    SimpleMultilayerSampler,
)
from reflectorch.inference.record_time import print_time
from reflectorch.inference.scipy_fitter import standard_refl_fit
from reflectorch.inference.multilayer_fitter import MultilayerFit


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

            raw_curve, raw_q = preprocessed_dict["curve"], preprocessed_dict["q_values"]

            with print_time("predict_from_preprocessed_curve"):
                preprocessed_dict.update(self.predict_from_preprocessed_curve(
                    preprocessed_curve, priors,
                    raw_curve=(raw_curve if use_raw_q else None),
                    raw_q=raw_q,
                    polish=polish,
                    use_raw_q=use_raw_q,
                    **kwargs
                ))

            return preprocessed_dict

    def predict_from_preprocessed_curve(self,
                                        curve: np.ndarray,
                                        priors: np.ndarray = None, *,  # ignore the priors so far
                                        polish: bool = True,
                                        raw_curve: np.ndarray = None,
                                        raw_q: np.ndarray = None,
                                        clip_prediction: bool = True,
                                        use_raw_q: bool = False,
                                        use_sampler: bool = False,
                                        fitted_time_limit: float = 3.,
                                        sampler_rel_bounds: float = 0.3,
                                        polish_with_abeles: bool = False,
                                        **kwargs
                                        ) -> dict:

        scaled_curve = self._scale_curve(curve)

        predicted_params, parametrized = self._simple_prediction(scaled_curve)

        if use_sampler:
            parametrized: Tensor = self._sampler_solution(
                curve, parametrized,
                time_limit=fitted_time_limit,
                rel_bounds=sampler_rel_bounds,
            )

        init_raw_q = raw_q

        if raw_curve is None:
            raw_curve = curve
            raw_q = self.q.squeeze().cpu().numpy()
            raw_q_t = self.q
        else:
            raw_q_t = torch.from_numpy(raw_q).to(self.q)

        # if q_ratio != 1.:
        #     predicted_params.scale_with_q(q_ratio)
        #     raw_q = raw_q * q_ratio
        #     raw_q_t = raw_q_t * q_ratio

        prediction_dict = {
            "params": parametrized.squeeze().cpu().numpy(),
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
                raw_q, raw_curve, parametrized, q_values=init_raw_q, use_kinematical=True
            ))
        if polish_with_abeles:
            prediction_dict.update(self._polish_prediction(
                raw_q, raw_curve, parametrized, q_values=init_raw_q, use_kinematical=False
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
            predicted_params: Tensor,
            batch_size: int = 2 ** 13,
            time_limit: float = 3.,
            rel_bounds: float = 0.3,
    ) -> Tensor:

        fit_obj = MultilayerFit.from_prediction(
            predicted_params, self._prior_sampler, self.q, torch.as_tensor(curve).to(self.q),
            batch_size=batch_size, rel_bounds=rel_bounds,
        )

        fit_obj.run_fixed_time(time_limit)

        best_params = fit_obj.get_best_solution()

        return best_params

    @property
    def _prior_sampler(self) -> SimpleMultilayerSampler:
        return self.trainer.loader.prior_sampler

    @print_time
    def _polish_prediction(self,
                           q: np.ndarray,
                           curve: np.ndarray,
                           predicted_params: Tensor,
                           q_values: np.ndarray,
                           use_kinematical: bool = True,
                           ) -> dict:

        params = predicted_params.squeeze().cpu().numpy()
        polished_params_dict = {}

        if use_kinematical:
            refl_generator = kinematical_approximation_np
        else:
            refl_generator = abeles_np

        try:
            polished_params_arr, curve_polished = standard_refl_fit(
                q, curve, params, restore_params_func=self._prior_sampler.restore_np_params,
                refl_generator=refl_generator,
                bounds=self._prior_sampler.get_np_bounds(),
            )
            params = self._prior_sampler.restore_np_params(polished_params_arr)
            if q_values is None:
                q_values = q
            curve_polished = abeles_np(q_values, **params)

        except Exception as err:
            self.log.exception(err)
            polished_params_arr = params
            curve_polished = np.zeros_like(q)

        polished_params_dict['params_polished'] = polished_params_arr
        polished_params_dict['curve_polished'] = curve_polished

        return polished_params_dict
