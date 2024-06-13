import logging

import numpy as np
import torch
from torch import Tensor
from typing import List, Tuple
import ipywidgets as widgets
from IPython.display import display

from reflectorch.data_generation.priors import Params, ExpUniformSubPriorSampler, UniformSubPriorParams
from reflectorch.data_generation.utils import get_density_profiles, get_param_labels
from reflectorch.runs.utils import (
    get_trainer_by_name, train_from_config
)
from reflectorch.runs.config import load_config
from reflectorch.ml.trainers import PointEstimatorTrainer
from reflectorch.data_generation.likelihoods import LogLikelihood

from reflectorch.inference.preprocess_exp import StandardPreprocessing
from reflectorch.inference.scipy_fitter import standard_refl_fit, get_fit_with_growth
from reflectorch.inference.sampler_solution import simple_sampler_solution, get_best_mse_param
from reflectorch.inference.record_time import print_time
from reflectorch.utils import to_t

class EasyInferenceModel(object):
    """Facilitates the inference process using pretrained models
    
    Args:
        config_name (str, optional): the name of the configuration file used to initialize the model. Defaults to None.
        config_dir (str, optional): path to the directory containing the configuration file. Defaults to None.
        model_path (str, optional): path to the saved weights of the neural network. Defaults to None.
        trainer (PointEstimatorTrainer, optional): if provided, this trainer instance is used instead of initializing from the configuration file . Defaults to None.
        preprocessing_parameters (dict, optional): dictionary of parameters for preprocessing raw data. Defaults to None.
        device (str, optional): the Pytorch device ('cuda' or 'cpu'). Defaults to 'cuda'.
    """
    def __init__(self, config_name: str = None, config_dir: str = None, model_path: str = None, trainer: PointEstimatorTrainer = None, preprocessing_parameters: dict = None, device='cuda'):
        self.config_name = config_name
        self.config_dir = config_dir
        self.model_path = model_path
        self.trainer = trainer
        self.device = device
        self.preprocessing = StandardPreprocessing(**(preprocessing_parameters or {}))

        if trainer is None and self.config_name is not None:
            self.load_model(self.config_name, self.config_dir, self.model_path)

    def load_model(self, config_name: str, config_dir: str, model_path: str) -> None:
        print(f"Loading model {config_name}")
        if self.config_name == config_name and self.trainer is not None:
            return
        
        self.config_name = config_name
        self.config_dir = config_dir
        self.model_path = model_path
        self.trainer = get_trainer_by_name(config_name=config_name, config_dir=config_dir, model_path=model_path, load_weights=True, inference_device = self.device)
        self.trainer.model.eval()
        print(f"Model {config_name} is loaded.")

    def load_model_from_hugginface(self, config_name: str):
        raise NotImplementedError

    def predict(self, reflectivity_curve: np.ndarray or Tensor, q_values: np.ndarray or Tensor, prior_bounds: List[Tuple]):
        scaled_curve = self._scale_curve(reflectivity_curve)
        scaled_prior_bounds = self._scale_prior_bounds(prior_bounds)

        if not isinstance(q_values, Tensor):
            q_values = torch.from_numpy(q_values)
        q_values = torch.atleast_2d(q_values).to(scaled_curve)

        scaled_curve_and_priors = torch.cat([scaled_curve, scaled_prior_bounds], dim=-1).float()
        with torch.no_grad():
            self.trainer.model.eval()
            if self.trainer.train_with_q_input:
                scaled_q = self.trainer.loader.q_generator.scale_q(q_values).float()
                scaled_predicted_params = self.trainer.model(scaled_curve_and_priors, scaled_q)
            else:
                scaled_predicted_params = self.trainer.model(scaled_curve_and_priors)
        
        restored_predictions = self.trainer.loader.prior_sampler.restore_params(torch.cat([scaled_predicted_params, scaled_prior_bounds], dim=-1))

        return restored_predictions
    
    def predict_using_widget(self, reflectivity_curve: np.ndarray or Tensor, q_values: np.ndarray or Tensor):

        NUM_INTERVALS = self.trainer.loader.prior_sampler.param_dim
        param_labels = get_param_labels(self.trainer.loader.prior_sampler.max_num_layers)

        print(f'Parameter ranges: {self.trainer.loader.prior_sampler.param_ranges}')
        print(f'Allowed widths of the prior bound intervals (max-min): {self.trainer.loader.prior_sampler.bound_width_ranges}')
        print(f'Please fill in the values of the minimum and maximum prior bound for each parameter and press the button!')

        def create_interval_widgets(n):
            intervals = []
            for i in range(n):
                interval_label = widgets.Label(value=f'{param_labels[i]}')
                min_val = widgets.FloatText(
                    value=0.0,
                    description='min',
                    layout=widgets.Layout(width='100px'),
                    style={'description_width': '30px'}
                )
                max_val = widgets.FloatText(
                    value=1.0,
                    description='max',
                    layout=widgets.Layout(width='100px'),
                    style={'description_width': '30px'}
                )
                interval_row = widgets.HBox([interval_label, min_val, max_val])
                intervals.append((min_val, max_val, interval_row))
            return intervals
        
        interval_widgets = create_interval_widgets(NUM_INTERVALS)
        interval_box = widgets.VBox([widget[2] for widget in interval_widgets])
        display(interval_box)

        button = widgets.Button(description="Make prediction")
        display(button)

        def store_values(b):
            values = []
            for min_widget, max_widget, _ in interval_widgets:
                values.append((min_widget.value, max_widget.value))
            array_values = np.array(values)
            
            predicted_params = self.predict(reflectivity_curve=reflectivity_curve, q_values=q_values, prior_bounds=array_values)
            print(f'The prediction is: ')
            for l, p in zip(get_param_labels(2), predicted_params.parameters.squeeze().cpu().numpy()):
                print(f'{l.ljust(14)} --> Predicted: {p:.2f}')

            return predicted_params

        button.on_click(store_values)

    @print_time
    def _qshift_prediction(self, curve, scaled_bounds, num: int = 1000, dq_coef: float = 1.) -> UniformSubPriorParams:
        q = self.q.squeeze().float()
        curve = to_t(curve).to(q)
        dq_max = (q[1] - q[0]) * dq_coef
        q_shifts = torch.linspace(-dq_max, dq_max, num).to(q)
        shifted_curves = _qshift_interp(q.squeeze(), curve, q_shifts)

        assert shifted_curves.shape == (num, q.shape[0])

        scaled_curves = self.trainer.loader.curves_scaler.scale(shifted_curves)
        context = torch.cat([scaled_curves, torch.atleast_2d(scaled_bounds).expand(scaled_curves.shape[0], -1)], -1)

        with torch.no_grad():
            self.trainer.model.eval()
            scaled_params = self.trainer.model(context)
            restored_params = self._restore_predicted_params(scaled_params, context)

            best_param = get_best_mse_param(
                restored_params,
                self._get_likelihood(curve),
            )
            return best_param
        
    @print_time
    def _polish_prediction(self,
                           q: np.ndarray,
                           curve: np.ndarray,
                           predicted_params: Params,
                           priors: np.ndarray,
                           sld_x_axis,
                           fit_growth: bool = True,
                           max_d_change: float = 5.,
                           ) -> dict:
        params = torch.cat([
            predicted_params.thicknesses.squeeze(),
            predicted_params.roughnesses.squeeze(),
            predicted_params.slds.squeeze()
        ]).cpu().numpy()

        polished_params_dict = {}

        try:
            if fit_growth:
                polished_params_arr, curve_polished = get_fit_with_growth(
                    q, curve, params, bounds=priors.T,
                    max_d_change=max_d_change,
                )
                polished_params = Params.from_tensor(torch.from_numpy(polished_params_arr[:-1][None]).to(self.q))
            else:
                polished_params_arr, curve_polished = standard_refl_fit(q, curve, params, bounds=priors.T)
                polished_params = Params.from_tensor(torch.from_numpy(polished_params_arr[None]).to(self.q))
        except Exception as err:
            self.log.exception(err)
            polished_params = predicted_params
            polished_params_arr = get_prediction_array(polished_params)
            curve_polished = np.zeros_like(q)

        polished_params_dict['params_polished'] = polished_params_arr
        polished_params_dict['curve_polished'] = curve_polished

        sld_x_axis_polished, sld_profile_polished, _ = get_density_profiles(
            polished_params.thicknesses, polished_params.roughnesses, polished_params.slds, z_axis=sld_x_axis,
        )

        polished_params_dict['sld_profile_polished'] = sld_profile_polished.squeeze().cpu().numpy()

        return polished_params_dict
    

    def _scale_curve(self, curve: np.ndarray or Tensor):
        if not isinstance(curve, Tensor):
            curve = torch.from_numpy(curve).float()
        curve = torch.atleast_2d(curve).to(self.device)
        scaled_curve = self.trainer.loader.curves_scaler.scale(curve)
        return scaled_curve
    
    def _scale_prior_bounds(self, prior_bounds: List[Tuple]):
        prior_bounds = torch.tensor(prior_bounds)
        prior_bounds = prior_bounds.to(self.device).T
        min_bounds, max_bounds = prior_bounds[:, None]

        scaled_bounds = torch.cat([
            self.trainer.loader.prior_sampler.scale_bounds(min_bounds), 
            self.trainer.loader.prior_sampler.scale_bounds(max_bounds)
        ], -1)

        return scaled_bounds.float()


def get_prediction_array(params: Params) -> np.ndarray:
    predict_arr = torch.cat([
        params.thicknesses.squeeze(),
        params.roughnesses.squeeze(),
        params.slds.squeeze(),
    ]).cpu().numpy()

    return predict_arr


def _qshift_interp(q, r, q_shifts):
    qs = q[None] + q_shifts[:, None]
    eps = torch.finfo(r.dtype).eps
    ind = torch.searchsorted(q[None].expand_as(qs).contiguous(), qs.contiguous())
    ind = torch.clamp(ind - 1, 0, q.shape[0] - 2)
    slopes = (r[1:] - r[:-1]) / (eps + (q[1:] - q[:-1]))
    return r[ind] + slopes[ind] * (qs - q[ind])
