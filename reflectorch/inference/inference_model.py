import logging

from time import perf_counter
import numpy as np
import torch
from torch import Tensor

from reflectorch.data_generation.priors import Params, ExpUniformSubPriorSampler, UniformSubPriorParams
from reflectorch.data_generation.utils import get_density_profiles, get_param_labels
from reflectorch.runs.utils import (
    get_trainer_by_name, train_from_config
)
from reflectorch.runs.config import load_config
from reflectorch.ml.trainers import PointEstimatorTrainer
from reflectorch.data_generation.likelihoods import LogLikelihood

from reflectorch.inference.preprocess_exp import StandardPreprocessing
from reflectorch.inference.scipy_fitter import standard_refl_fit
from reflectorch.inference.sampler_solution import simple_sampler_solution
from reflectorch.inference.record_time import print_time
from reflectorch.utils import to_t


class InferenceModel(object):
    def __init__(self, name: str = None, trainer: PointEstimatorTrainer = None, preprocessing_parameters: dict = None,
                 num_sampling: int = 2 ** 13):
        self.log = logging.getLogger(__name__)
        self.model_name = name
        self.trainer = trainer
        self.q = None
        self.preprocessing = StandardPreprocessing(**(preprocessing_parameters or {}))
        self._sampling_num = num_sampling

        if trainer is None and self.model_name is not None:
            self.load_model(self.model_name)
        elif trainer is not None:
            self._set_trainer(trainer, preprocessing_parameters)

    ### API methods ###

    def load_model(self, name: str) -> None:
        self.log.debug(f"loading model {name}")
        if self.model_name == name and self.trainer is not None:
            return
        self.model_name = name
        self._set_trainer(get_trainer_by_name(name))
        self.log.info(f"Model {name} is loaded.")

    def train_model(self, name: str):
        self.model_name = name
        self.trainer = train_from_config(load_config(name))

    def set_preprocessing_parameters(self, **kwargs) -> None:
        self.preprocessing.set_parameters(**kwargs)

    def preprocess(self,
                   intensity: np.ndarray,
                   scattering_angle: np.ndarray,
                   attenuation: np.ndarray,
                   update_params: bool = False,
                   **kwargs) -> dict:
        if update_params:
            self.preprocessing.set_parameters(**kwargs)
        preprocessed_dict = self.preprocessing(intensity, scattering_angle, attenuation, **kwargs)
        return preprocessed_dict

    def predict(self,
                intensity: np.ndarray,
                scattering_angle: np.ndarray,
                attenuation: np.ndarray,
                priors: np.ndarray,
                preprocessing_parameters: dict = None,
                polish: bool = True,
                use_sampler: bool = True,
                ) -> dict:

        with print_time("everything"):
            with print_time("preprocess"):
                preprocessed_dict = self.preprocess(
                    intensity, scattering_angle, attenuation, **(preprocessing_parameters or {})
                )

            preprocessed_curve = preprocessed_dict["curve_interp"]
            raw_curve, raw_q = preprocessed_dict["curve"], preprocessed_dict["q_values"]
            q_ratio = preprocessed_dict["q_ratio"]

            with print_time("predict_from_preprocessed_curve"):
                preprocessed_dict.update(self.predict_from_preprocessed_curve(
                    preprocessed_curve, priors, raw_curve=raw_curve, raw_q=raw_q, polish=polish, q_ratio=q_ratio,
                    use_sampler=use_sampler,
                ))

            return preprocessed_dict

    def predict_from_preprocessed_curve(self,
                                        curve: np.ndarray,
                                        priors: np.ndarray, *,
                                        polish: bool = True,
                                        raw_curve: np.ndarray = None,
                                        raw_q: np.ndarray = None,
                                        clip_prediction: bool = True,
                                        q_ratio: float = 1.,
                                        use_sampler: bool = True,
                                        ) -> dict:
        context, min_bounds, max_bounds = self._input2context(curve, priors, q_ratio)

        with torch.no_grad():
            self.trainer.model.eval()
            scaled_params = self.trainer.model(context)

        predicted_params: UniformSubPriorParams = self._restore_predicted_params(scaled_params, context)

        if use_sampler:
            with print_time("sampler"):
                predicted_params: UniformSubPriorParams = self._sampler_solution(
                    curve, predicted_params, min_bounds, max_bounds,
                )

        if clip_prediction:
            predicted_params = self._prior_sampler.clamp_params(predicted_params)

        if raw_curve is None:
            raw_curve = curve
        if raw_q is None:
            raw_q = self.q.squeeze().cpu().numpy()
            raw_q_t = self.q
        else:
            raw_q_t = torch.from_numpy(raw_q).to(self.q)

        if q_ratio != 1.:
            predicted_params.scale_with_q(q_ratio)
            raw_q = raw_q * q_ratio
            raw_q_t = raw_q_t * q_ratio

        prediction_dict = {
            "params": _get_prediction_array(predicted_params),
            "param_names": get_param_labels(
                predicted_params.max_layer_num,
                thickness_name='d',
                roughness_name='sigma',
                sld_name='rho',
            ),
            "curve_predicted": predicted_params.reflectivity(raw_q_t).squeeze().cpu().numpy()
        }

        sld_x_axis, sld_profile, _ = get_density_profiles(
            predicted_params.thicknesses, predicted_params.roughnesses, predicted_params.slds, num=1024,
        )

        prediction_dict['sld_profile'] = sld_profile.squeeze().cpu().numpy()
        prediction_dict['sld_x_axis'] = sld_x_axis.squeeze().cpu().numpy()

        if polish:
            polished_params = self._polish_prediction(raw_q, raw_curve, predicted_params, priors)
            prediction_dict['params_polished'] = _get_prediction_array(polished_params)
            prediction_dict['curve_polished'] = polished_params.reflectivity(raw_q_t).squeeze().cpu().numpy()

            sld_x_axis_polished, sld_profile_polished, _ = get_density_profiles(
                polished_params.thicknesses, polished_params.roughnesses, polished_params.slds, z_axis=sld_x_axis,
            )

            prediction_dict['sld_profile_polished'] = sld_profile_polished.squeeze().cpu().numpy()

        return prediction_dict

    ### some shortcut methods for data processing ###

    def _polish_prediction(self,
                           q: np.ndarray,
                           curve: np.ndarray,
                           predicted_params: Params,
                           priors: np.ndarray
                           ) -> Params:
        params = torch.cat([
            predicted_params.thicknesses.squeeze(),
            predicted_params.roughnesses.squeeze(),
            predicted_params.slds.squeeze()
        ]).cpu().numpy()

        try:
            polished_params = standard_refl_fit(q, curve, params, bounds=priors.T)
            polished_params = Params.from_tensor(torch.from_numpy(polished_params[None]).to(self.q))
        except Exception as err:
            self.log.exception(err)
            polished_params = predicted_params

        return polished_params

    def _restore_predicted_params(self, scaled_params: Tensor, context: Tensor) -> UniformSubPriorParams:
        predicted_params: UniformSubPriorParams = self.trainer.loader.prior_sampler.restore_params(
            self.trainer.loader.prior_sampler.PARAM_CLS.restore_params_from_context(scaled_params, context)
        )
        return predicted_params

    def _input2context(self, curve: np.ndarray, priors: np.ndarray, q_ratio: float = 1.):
        scaled_curve = self._scale_curve(curve)
        scaled_bounds, min_bounds, max_bounds = self._scale_priors(priors, q_ratio)
        scaled_input = torch.cat([scaled_curve, scaled_bounds], -1)
        return scaled_input, min_bounds, max_bounds

    def _scale_curve(self, curve: np.ndarray or Tensor):
        if not isinstance(curve, Tensor):
            curve = torch.from_numpy(curve).float()
        curve = torch.atleast_2d(curve).to(self.q)
        scaled_curve = self.trainer.loader.curves_scaler.scale(curve)
        return scaled_curve.float()

    def _scale_priors(self, priors: np.ndarray or Tensor, q_ratio: float = 1.):
        if not isinstance(priors, Tensor):
            priors = torch.from_numpy(priors)

        priors = priors.float().clone()

        priors = priors.to(self.q).T
        priors = self._prior_sampler.scale_bounds_with_q(priors, 1 / q_ratio)
        priors = self._prior_sampler.clamp_bounds(priors)

        min_bounds, max_bounds = priors[:, None].to(self.q)
        prior_sampler = self._prior_sampler
        scaled_bounds = torch.cat([
            prior_sampler.scale_bounds(min_bounds), prior_sampler.scale_bounds(max_bounds)
        ], -1)
        return scaled_bounds.float(), min_bounds, max_bounds

    @property
    def _prior_sampler(self) -> ExpUniformSubPriorSampler:
        return self.trainer.loader.prior_sampler

    def _set_trainer(self, trainer, preprocessing_parameters: dict = None):
        self.trainer = trainer
        self.trainer.model.eval()
        self._update_preprocessing(preprocessing_parameters)

    def _update_preprocessing(self, preprocessing_parameters: dict = None):
        self.log.debug(f"setting preprocessing_parameters {preprocessing_parameters}.")
        self.q = self.trainer.loader.q_generator.q
        self.preprocessing = StandardPreprocessing(
            self.q.cpu().squeeze().numpy(),
            **(preprocessing_parameters or {})
        )
        self.log.info(f"preprocessing params are set: {preprocessing_parameters}.")

    def _sampler_solution(
            self,
            curve: Tensor or np.ndarray,
            predicted_params: UniformSubPriorParams,
            min_bounds: Tensor,
            max_bounds: Tensor,
    ) -> UniformSubPriorParams:
        if not isinstance(curve, Tensor):
            curve = torch.from_numpy(curve).float()
        curve = curve.to(self.q)

        likelihood = LogLikelihood(
            self.q, curve, self._prior_sampler, curve * 0.1
        )
        refined_params = simple_sampler_solution(
            likelihood,
            predicted_params,
            min_bounds,
            max_bounds,
            self._prior_sampler.min_bounds,
            self._prior_sampler.max_bounds,
            num=self._sampling_num, coef=0.1,
        )
        return refined_params


def _get_prediction_array(params: Params) -> np.ndarray:
    predict_arr = torch.cat([
        params.thicknesses.squeeze(),
        params.roughnesses.squeeze(),
        params.slds.squeeze(),
    ]).cpu().numpy()

    return predict_arr
