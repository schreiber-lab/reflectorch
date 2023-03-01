import logging

import numpy as np
import torch
from torch import Tensor

from reflectorch.data_generation.priors import Params, ExpUniformSubPriorSampler
from reflectorch.data_generation.utils import get_density_profiles, get_param_labels
from reflectorch.runs.utils import (
    get_trainer_by_name, train_from_config
)
from reflectorch.runs.config import load_config
from reflectorch.inference.preprocess_exp import StandardPreprocessing
from reflectorch.ml.trainers import PointEstimatorTrainer
from reflectorch.inference.scipy_fitter import standard_refl_fit


class InferenceModel(object):
    def __init__(self, name: str = None, trainer: PointEstimatorTrainer = None, preprocessing_parameters: dict = None):
        self.log = logging.getLogger(__name__)
        self.model_name = name
        self.trainer = trainer
        self.q = None
        self.preprocessing = StandardPreprocessing(**(preprocessing_parameters or {}))

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
                ) -> dict:
        preprocessed_dict = self.preprocess(
            intensity, scattering_angle, attenuation, **(preprocessing_parameters or {})
        )
        preprocessed_curve = preprocessed_dict["curve_interp"]
        raw_curve, raw_q = preprocessed_dict["curve"], preprocessed_dict["q_values"]
        preprocessed_dict.update(self.predict_from_preprocessed_curve(preprocessed_curve, priors))
        return preprocessed_dict

    def predict_from_preprocessed_curve(self,
                                        curve: np.ndarray,
                                        priors: np.ndarray, *,
                                        polish: bool = True,
                                        raw_curve: np.ndarray = None,
                                        raw_q: np.ndarray = None,
                                        ) -> dict:
        context = self._input2context(curve, priors)

        with torch.no_grad():
            scaled_params = self.trainer.model(context)

        predicted_params = self._restore_predicted_params(scaled_params, context)

        if raw_curve is None:
            raw_curve = curve
        if raw_q is None:
            raw_q = self.q.squeeze().cpu().numpy()

        if polish:
            predicted_params = self._polish_prediction(raw_q, raw_curve, predicted_params, priors)

        prediction_dict = self._scaled_params2prediction_dict(predicted_params, raw_q)

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

        polished_params = standard_refl_fit(q, curve, params, bounds=priors.T)
        polished_params = Params.from_tensor(torch.from_numpy(polished_params[None]).to(self.q))

        return polished_params

    def _scaled_params2prediction_dict(self,
                                       predicted_params: Params,
                                       raw_q: np.ndarray = None
                                       ) -> dict:

        sld_x_axis, sld_profile, _ = get_density_profiles(
            predicted_params.thicknesses, predicted_params.roughnesses, predicted_params.slds, num=1024,
        )

        if raw_q is not None:
            raw_q = torch.from_numpy(np.atleast_2d(raw_q)).to(self.q)
        else:
            raw_q = self.q

        predicted_curve = predicted_params.reflectivity(raw_q).squeeze().cpu().numpy()

        prediction_dict = {
            "params": _get_prediction_array(predicted_params),
            "sld_x_axis": sld_x_axis.squeeze().cpu().numpy(),
            "sld_profile": sld_profile.squeeze().cpu().numpy(),
            "curve_predicted": predicted_curve,
            "param_names": get_param_labels(
                predicted_params.max_layer_num,
                thickness_name='d',
                roughness_name='sigma',
                sld_name='rho',
            )
        }

        return prediction_dict

    def _restore_predicted_params(self, scaled_params: Tensor, context: Tensor) -> Params:
        predicted_params: Params = self.trainer.loader.prior_sampler.restore_params(
            self.trainer.loader.prior_sampler.PARAM_CLS.restore_params_from_context(scaled_params, context)
        )
        return predicted_params

    def _input2context(self, curve: np.ndarray, priors: np.ndarray):
        scaled_curve = self._scale_curve(curve)
        scaled_bounds = self._scale_priors(priors)
        scaled_input = torch.cat([scaled_curve, scaled_bounds], -1)
        return scaled_input

    def _scale_curve(self, curve: np.ndarray or Tensor):
        if not isinstance(curve, Tensor):
            curve = torch.from_numpy(curve).float().to(self.q)[None]
        scaled_curve = self.trainer.loader.curves_scaler.scale(curve)
        return scaled_curve.float()

    def _scale_priors(self, priors: np.ndarray or Tensor):
        if not isinstance(priors, Tensor):
            priors = torch.from_numpy(priors).float()

        min_bounds, max_bounds = priors.T[:, None].to(self.q)
        prior_sampler = self._prior_sampler
        scaled_bounds = torch.cat([
            prior_sampler.scale_bounds(min_bounds), prior_sampler.scale_bounds(max_bounds)
        ], -1)
        return scaled_bounds.float()

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


def _get_prediction_array(params: Params) -> np.ndarray:
    predict_arr = torch.cat([
        params.thicknesses.squeeze(),
        params.roughnesses.squeeze(),
        params.slds.squeeze(),
    ]).cpu().numpy()

    return predict_arr
