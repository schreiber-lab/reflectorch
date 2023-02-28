import numpy as np
import torch
from torch import Tensor

from reflectorch.data_generation.priors import Params, ExpUniformSubPriorSampler
from reflectorch.runs.utils import (
    get_trainer_by_name, train_from_config
)
from reflectorch.runs.config import load_config
from reflectorch.inference.preprocess_exp import StandardPreprocessing
from reflectorch.ml.trainers import PointEstimatorTrainer


class InferenceModel(object):
    def __init__(self, name: str = None, trainer: PointEstimatorTrainer = None, preprocessing_parameters: dict = None):
        self.model_name = name
        self.trainer = trainer
        self.q = None
        self.preprocessing = StandardPreprocessing(**(preprocessing_parameters or {}))

        if trainer is None and self.model_name is not None:
            self.load_model(self.model_name, preprocessing_parameters=preprocessing_parameters)
        elif trainer is not None:
            self._set_trainer(trainer, preprocessing_parameters)

    ### API methods ###

    def load_model(self, name: str) -> None:
        if self.model_name == name:
            return
        self.model_name = name
        self._set_trainer(get_trainer_by_name(name))

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
        preprocessed_dict["parameters"] = self.predict_from_preprocessed_curve(preprocessed_curve, priors)
        return preprocessed_dict

    def predict_from_preprocessed_curve(self, curve: np.ndarray, priors: np.ndarray) -> np.ndarray:
        context = self._input2context(curve, priors)

        with torch.no_grad():
            scaled_params = self.trainer.model(context)

        prediction = self._scaled_params2prediction_arr(scaled_params, context)
        return prediction

    ### some shortcut methods for data processing ###

    def _scaled_params2prediction_arr(self, scaled_params: Tensor, context: Tensor):
        predicted_params: Params = self.trainer.loader.prior_sampler.restore_params(
            self.trainer.loader.prior_sampler.PARAM_CLS.restore_params_from_context(scaled_params, context)
        )
        prediction = _get_prediction_array(predicted_params)
        return prediction

    def _input2context(self, curve: np.ndarray, priors: np.ndarray):
        scaled_curve = self._scale_curve(curve)
        scaled_bounds = self._scale_priors(priors)
        scaled_input = torch.cat([scaled_curve, scaled_bounds], -1)
        return scaled_input

    def _scale_curve(self, curve: np.ndarray or Tensor):
        if not isinstance(curve, Tensor):
            curve = torch.from_numpy(curve).float().to(self.q)[None]
        scaled_curve = self.trainer.loader.curves_scaler.scale(curve)
        return scaled_curve

    def _scale_priors(self, priors: np.ndarray or Tensor):
        if not isinstance(priors, Tensor):
            priors = torch.from_numpy(priors).float()

        min_bounds, max_bounds = priors.T[:, None].to(self.q)
        prior_sampler = self._prior_sampler
        scaled_bounds = torch.cat([
            prior_sampler.scale_bounds(min_bounds), prior_sampler.scale_bounds(min_bounds)
        ], -1)
        return scaled_bounds

    @property
    def _prior_sampler(self) -> ExpUniformSubPriorSampler:
        return self.trainer.loader.prior_sampler

    def _set_trainer(self, trainer, preprocessing_parameters: dict = None):
        self.trainer = trainer
        self.trainer.model.eval()
        self._update_preprocessing(preprocessing_parameters)

    def _update_preprocessing(self, preprocessing_parameters: dict = None):
        self.q = self.trainer.loader.q_generator.q
        self.preprocessing = StandardPreprocessing(
            self.q.cpu().squeeze().numpy(),
            **(preprocessing_parameters or {})
        )


def _get_prediction_array(params: Params) -> np.ndarray:
    predict_arr = torch.cat([
        params.thicknesses.squeeze(),
        params.roughnesses.squeeze(),
        params.slds.squeeze(),
    ]).cpu().numpy()

    return predict_arr
