import asyncio
import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from typing import List, Tuple, Union
import ipywidgets as widgets
from IPython.display import display
from huggingface_hub import hf_hub_download

from reflectorch.data_generation.priors import Params, ExpUniformSubPriorSampler, UniformSubPriorParams
from reflectorch.data_generation.priors.parametric_subpriors import BasicParams
from reflectorch.data_generation.q_generator import ConstantQ, VariableQ
from reflectorch.data_generation.utils import get_density_profiles, get_param_labels
from reflectorch.paths import CONFIG_DIR, ROOT_DIR, SAVED_MODELS_DIR
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
        config_name (str, optional): the name of the configuration file used to initialize the model (either with or without the '.yaml' extension). Defaults to None.
        model_name (str, optional): the name of the file containing the weights of the model (either with or without the '.pt' extension), only required if different than: `'model_' + config_name + '.pt'`. Defaults to None 
        root_dir (str, optional): path to root directory containing the 'configs' and 'saved_models' subdirectories, if different from the package root directory (ROOT_DIR). Defaults to None.
        repo_id (str, optional): the id of the Huggingface repository from which the configuration files and model weights should be downloaded automatically if not found locally (in the 'configs' and 'saved_models' subdirectories of the root directory). Defaults to 'valentinsingularity/reflectivity'.
        trainer (PointEstimatorTrainer, optional): if provided, this trainer instance is used directly instead of being initialized from the configuration file. Defaults to None.
        device (str, optional): the Pytorch device ('cuda' or 'cpu'). Defaults to 'cuda'.
    """
    def __init__(self, config_name: str = None, model_name: str = None, root_dir:str = None, repo_id: str = 'valentinsingularity/reflectivity', 
                 trainer: PointEstimatorTrainer = None, device='cuda'):
        self.config_name = config_name
        self.model_name = model_name
        self.root_dir = root_dir
        self.repo_id = repo_id
        self.trainer = trainer
        self.device = device

        if trainer is None and self.config_name is not None:
            self.load_model(self.config_name, self.model_name, self.root_dir)

    def load_model(self, config_name: str, model_name: str, root_dir: str) -> None:
        """Loads a model for inference

        Args:
            config_name (str): the name of the configuration file used to initialize the model (either with or without the '.yaml' extension).
            model_name (str): the name of the file containing the weights of the model (either with or without the '.pt' extension), only required if different than: `'model_' + config_name + '.pt'`.
            root_dir (str): path to root directory containing the 'configs' and 'saved_models' subdirectories, if different from the package root directory (ROOT_DIR).
        """
        if self.config_name == config_name and self.trainer is not None:
            return
        
        if not config_name.endswith('.yaml'):
            config_name_no_extension = config_name
            self.config_name = config_name_no_extension + '.yaml'
        else:
            config_name_no_extension = config_name[:-5]
            self.config_name = config_name
        
        self.config_dir = Path(root_dir) / 'configs' if root_dir else CONFIG_DIR
        self.model_name = model_name or 'model_' + config_name_no_extension + '.pt'
        if not self.model_name.endswith('.pt'):
            self.model_name += '.pt'
        self.model_dir = Path(root_dir) / 'saved_models' if root_dir else SAVED_MODELS_DIR

        config_path = Path(self.config_dir) / self.config_name
        if config_path.exists():
            print(f"Configuration file `{config_path}` found locally.")
        else:
            print(f"Configuration file `{config_path}` not found locally.")
            if self.repo_id is None:
                raise ValueError("repo_id must be provided to download files from Huggingface.")
            print("Downloading from Huggingface...")
            hf_hub_download(repo_id=self.repo_id, subfolder='configs', filename=self.config_name, local_dir=config_path.parents[1])

        model_path = Path(self.model_dir) / self.model_name
        if model_path.exists():
            print(f"Weights file `{model_path}` found locally.")
        else:
            print(f"Weights file `{model_path}` not found locally.")
            if self.repo_id is None:
                raise ValueError("repo_id must be provided to download files from Huggingface.")
            print("Downloading from Huggingface...")
            hf_hub_download(repo_id=self.repo_id, subfolder='saved_models', filename=self.model_name, local_dir=model_path.parents[1])

        self.trainer = get_trainer_by_name(config_name=config_name, config_dir=self.config_dir, model_path=model_path, load_weights=True, inference_device = self.device)
        self.trainer.model.eval()
        
        print(f'The model corresponds to a parameterization with {self.trainer.loader.prior_sampler.max_num_layers} layers ({self.trainer.loader.prior_sampler.param_dim} predicted parameters)')
        print(f'Parameter types and total ranges: {self.trainer.loader.prior_sampler.param_ranges}')
        print(f'Allowed widths of the prior bound intervals (max-min): {self.trainer.loader.prior_sampler.bound_width_ranges}')

        if isinstance(self.trainer.loader.q_generator, ConstantQ):
            q_min = self.trainer.loader.q_generator.q[0].item()
            q_max = self.trainer.loader.q_generator.q[-1].item()
            n_q = self.trainer.loader.q_generator.q.shape[0]
            print(f'The model was trained on curves discretized at {n_q} uniform points between between q_min={q_min} and q_max={q_max}')
        elif isinstance(self.trainer.loader.q_generator, VariableQ):
            q_min_range = self.trainer.loader.q_generator.q_min_range
            q_max_range = self.trainer.loader.q_generator.q_max_range
            n_q_range = self.trainer.loader.q_generator.n_q_range
            print(f'The model was trained on curves discretized at a number between {n_q_range[0]} and {n_q_range[1]} of uniform points between between q_min in [{q_min_range[0]}, {q_min_range[1]}] and q_max in [{q_max_range[0]}, {q_max_range[1]}]')

    def predict(self, reflectivity_curve: Union[np.ndarray, Tensor], 
                q_values: Union[np.ndarray, Tensor], 
                prior_bounds: Union[np.ndarray, List[Tuple]], 
                clip_prediction: bool = False, 
                polish_prediction: bool = False, 
                use_q_shift: bool = False, 
                fit_growth: bool = False, 
                max_d_change: float = 5.,
                calc_pred_curve: bool = True,
                calc_pred_sld_profile: bool = False,
                ):
        """Predict the thin film parameters

        Args:
            reflectivity_curve (Union[np.ndarray, Tensor]): The reflectivity curve (which has been already preprocessed, normalized and interpolated).
            q_values (Union[np.ndarray, Tensor]): The momentum transfer (q) values for the reflectivity curve (in units of reverse angstroms).
            prior_bounds (Union[np.ndarray, List[Tuple]]): the prior bounds for the thin film parameters.
            clip_prediction (bool, optional): If True the values of the predicted parameters are clipped to not be outside the interval set by the prior bounds. Defaults to False.
            polish_prediction (bool, optional): If True the neural network predictions are further polished using a simple least mean squares (LMS) fit. Defaults to False.
            use_q_shift (bool, optional): . Defaults to False.
            fit_growth (bool, optional): . Defaults to False.
            max_d_change (float, optional): . Defaults to 5.
            calc_pred_curve (bool, optional): Whether to calculate the curve corresponding to the predicted parameters. Defaults to True.
            calc_pred_sld_profile (bool, optional): Whether to calculate the SLD profile corresponding to the predicted parameters. Defaults to False.

        Returns:
            dict: dictionary containing the predictions
        """
        scaled_curve = self._scale_curve(reflectivity_curve)
        prior_bounds = np.array(prior_bounds)
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
        
        predicted_params = self.trainer.loader.prior_sampler.restore_params(torch.cat([scaled_predicted_params, scaled_prior_bounds], dim=-1))
        
        if clip_prediction:
            predicted_params = self.trainer.loader.prior_sampler.clamp_params(predicted_params)
        
        
        prediction_dict = {
            "predicted_params_object": predicted_params,
            "predicted_params_array": predicted_params.parameters.squeeze().cpu().numpy(),
            "param_names" : self.trainer.loader.prior_sampler.param_model.get_param_labels()
        }

        if calc_pred_curve:
            predicted_curve = predicted_params.reflectivity(q_values).squeeze().cpu().numpy()
            prediction_dict[ "predicted_curve"] = predicted_curve
        
        if calc_pred_sld_profile: 
            predicted_sld_xaxis, predicted_sld_profile, _ = get_density_profiles(
                predicted_params.thicknesses, predicted_params.roughnesses, predicted_params.slds, num=1024, 
            ) 
            prediction_dict['predicted_sld_profile'] = predicted_sld_profile.squeeze().cpu().numpy()
            prediction_dict['predicted_sld_xaxis'] = predicted_sld_xaxis.squeeze().cpu().numpy()
        else:
            predicted_sld_xaxis = None

        if polish_prediction:
            polished_dict = self._polish_prediction(q = q_values.squeeze().cpu().numpy(), 
                                                    curve = reflectivity_curve, 
                                                    predicted_params = predicted_params, 
                                                    priors = np.array(prior_bounds), 
                                                    sld_x_axis = predicted_sld_xaxis,
                                                    max_d_change = max_d_change, 
                                                    fit_growth = fit_growth,
                                                    calc_polished_curve = calc_pred_curve,
                                                    calc_polished_sld_profile = False,
                                                    )
            prediction_dict.update(polished_dict)

        return prediction_dict

    def predict_using_widget(self, reflectivity_curve: Union[np.ndarray, torch.Tensor], q_values: Union[np.ndarray, torch.Tensor], **kwargs):
        """Use an interactive Python widget for specifying the prior bounds before the prediction (works only in a Jupyter notebook).
        The other arguments are the same as for the `predict` method.
        """

        NUM_INTERVALS = self.trainer.loader.prior_sampler.param_dim
        param_labels = self.trainer.loader.prior_sampler.param_model.get_param_labels()
        min_bounds = self.trainer.loader.prior_sampler.min_bounds.cpu().numpy().flatten()
        max_bounds = self.trainer.loader.prior_sampler.max_bounds.cpu().numpy().flatten()
        max_deltas = self.trainer.loader.prior_sampler.max_delta.cpu().numpy().flatten()

        print(f'Parameter ranges: {self.trainer.loader.prior_sampler.param_ranges}')
        print(f'Allowed widths of the prior bound intervals (max-min): {self.trainer.loader.prior_sampler.bound_width_ranges}')
        print(f'Please fill in the values of the minimum and maximum prior bound for each parameter and press the button!')

        def create_interval_widgets(n):
            intervals = []
            for i in range(n):
                interval_label = widgets.Label(value=f'{param_labels[i]}')
                initial_max = min(max_bounds[i], min_bounds[i] + max_deltas[i])
                slider = widgets.FloatRangeSlider(
                    value=[min_bounds[i], initial_max],
                    min=min_bounds[i],
                    max=max_bounds[i],
                    step=0.01,
                    layout=widgets.Layout(width='400px'),
                    style={'description_width': '60px'}
                )
                
                def validate_range(change, slider=slider, max_width=max_deltas[i]):
                    min_val, max_val = change['new']
                    if max_val - min_val > max_width:
                        if change['name'] == 'value':
                            if change['old'][0] != min_val:
                                max_val = min_val + max_width
                            else:
                                min_val = max_val - max_width
                        slider.value = [min_val, max_val]
                
                slider.observe(validate_range, names='value')
                
                interval_row = widgets.HBox([interval_label, slider])
                intervals.append((slider, interval_row))
            return intervals
        
        interval_widgets = create_interval_widgets(NUM_INTERVALS)
        interval_box = widgets.VBox([widget[1] for widget in interval_widgets])
        display(interval_box)

        button = widgets.Button(description="Make prediction")
        display(button)

        global prediction_result
        prediction_result = None

        def store_values(b):
            global prediction_result
            values = []
            for slider, _ in interval_widgets:
                values.append((slider.value[0], slider.value[1]))
            array_values = np.array(values)
            
            prediction_result = self.predict(reflectivity_curve=reflectivity_curve, q_values=q_values, prior_bounds=array_values, **kwargs)
            print(prediction_result["predicted_params_array"])

        button.on_click(store_values)

        return prediction_result

    def _qshift_prediction(self, curve, scaled_bounds, num: int = 1000, dq_coef: float = 1.) -> BasicParams:
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
        
    def _polish_prediction(self,
                           q: np.ndarray,
                           curve: np.ndarray,
                           predicted_params: BasicParams,
                           priors: np.ndarray,
                           sld_x_axis,
                           fit_growth: bool = False,
                           max_d_change: float = 5.,
                           calc_polished_curve: bool = True,
                           calc_polished_sld_profile: bool = False,
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
                    q = q, 
                    curve = curve, 
                    init_params = params, 
                    bounds = priors.T,
                    max_d_change = max_d_change,
                )
                polished_params = BasicParams(
                    torch.from_numpy(polished_params_arr[:-1][None]),
                    torch.from_numpy(priors.T[0][None]),
                    torch.from_numpy(priors.T[1][None]),
                    )
            else:
                polished_params_arr, curve_polished = standard_refl_fit(
                    q = q, 
                    curve = curve, 
                    init_params = params, 
                    bounds=priors.T)
                polished_params = BasicParams(
                    torch.from_numpy(polished_params_arr[None]),
                    torch.from_numpy(priors.T[0][None]),
                    torch.from_numpy(priors.T[1][None]),
                    
                    )
        except Exception as err:
            polished_params = predicted_params
            polished_params_arr = get_prediction_array(polished_params)
            curve_polished = np.zeros_like(q)

        polished_params_dict['polished_params_array'] = polished_params_arr
        if calc_polished_curve:
            polished_params_dict['polished_curve'] = curve_polished

        if calc_polished_sld_profile:
            _, sld_profile_polished, _ = get_density_profiles(
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
                use_sampler: bool = False,
                use_q_shift: bool = True,
                max_d_change: float = 5.,
                fit_growth: bool = True,
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
                    use_sampler=use_sampler, use_q_shift=use_q_shift, max_d_change=max_d_change,
                    fit_growth=fit_growth,
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
                                        use_sampler: bool = False,
                                        use_q_shift: bool = True,
                                        max_d_change: float = 5.,
                                        fit_growth: bool = True,
                                        ) -> dict:

        scaled_curve = self._scale_curve(curve)
        scaled_bounds, min_bounds, max_bounds = self._scale_priors(priors, q_ratio)

        if not use_q_shift:
            predicted_params: UniformSubPriorParams = self._simple_prediction(scaled_curve, scaled_bounds)
        else:
            predicted_params: UniformSubPriorParams = self._qshift_prediction(curve, scaled_bounds)

        if use_sampler:
            predicted_params: UniformSubPriorParams = self._sampler_solution(
                curve, predicted_params,
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
            "params": get_prediction_array(predicted_params),
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
            prediction_dict.update(self._polish_prediction(
                raw_q, raw_curve, predicted_params, priors, sld_x_axis,
                max_d_change=max_d_change, fit_growth=fit_growth,
            ))

            if fit_growth and "params_polished" in prediction_dict:
                prediction_dict["param_names"].append("max_d_change")

        return prediction_dict

    ### some shortcut methods for data processing ###

    def _simple_prediction(self, scaled_curve, scaled_bounds) -> UniformSubPriorParams:
        context = torch.cat([scaled_curve, scaled_bounds], -1)

        with torch.no_grad():
            self.trainer.model.eval()
            scaled_params = self.trainer.model(context)

        predicted_params: UniformSubPriorParams = self._restore_predicted_params(scaled_params, context)
        return predicted_params

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

    @print_time
    def _sampler_solution(
            self,
            curve: Tensor or np.ndarray,
            predicted_params: UniformSubPriorParams,
    ) -> UniformSubPriorParams:

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

    def _get_likelihood(self, curve, rel_err: float = 0.1, abs_err: float = 1e-12):
        return LogLikelihood(
            self.q, curve, self._prior_sampler, curve * rel_err + abs_err
        )


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


if __name__ == '__main__':
    from reflectorch import *

    inference_model = EasyInferenceModel(
        config_name = 'mc25.yaml',
        model_name = None,
        root_dir = None,
        repo_id = None,
        trainer = None,
        preprocessing_parameters = None,
        device = 'cuda',
    )

    loaded_data_dict = torch.load(str(ROOT_DIR / 'exp_data/data_PTCDI-C3.pt'))
    curve_exp = loaded_data_dict['curves'][0]
    q_exp = loaded_data_dict['q'][0]
    print(curve_exp.shape, q_exp.shape, q_exp.min(), q_exp.max())

    prior_bounds = [(0., 400.), (0., 10.), #layer thicknesses (top to bottom)
                    (0., 30.), (0., 15.), (0., 15.), #interlayer roughnesses (top to bottom)
                    (10., 13.), (20.,21.), (20., 21.)] #real layer slds (top to bottom)

    q_model = inference_model.trainer.loader.q_generator.q.cpu().numpy()
    exp_curve_interp = interp_reflectivity(q_model, q_exp, curve_exp)

    prediction_dict = inference_model.predict(
        reflectivity_curve = exp_curve_interp,
        q_values = q_model,
        prior_bounds = prior_bounds,
        clip_prediction = True,
        polish_prediction = True,
        fit_growth = False,
        max_d_change = 5.,
        use_q_shift = False, 
    )