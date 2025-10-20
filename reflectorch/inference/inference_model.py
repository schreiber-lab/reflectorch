from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from typing import List, Tuple, Union
from huggingface_hub import hf_hub_download

from reflectorch.data_generation.priors import BasicParams
from reflectorch.data_generation.priors.parametric_models import NuisanceParamsWrapper
from reflectorch.data_generation.q_generator import ConstantQ, VariableQ, MaskedVariableQ
from reflectorch.data_generation.utils import get_density_profiles
from reflectorch.inference.preprocess_exp.interpolation import interp_reflectivity
from reflectorch.paths import CONFIG_DIR, SAVED_MODELS_DIR
from reflectorch.runs.utils import (
    get_trainer_by_name
)
from reflectorch.ml.trainers import PointEstimatorTrainer
from reflectorch.data_generation.likelihoods import LogLikelihood

from reflectorch.inference.scipy_fitter import refl_fit, get_fit_with_growth
from reflectorch.inference.sampler_solution import get_best_mse_param
from reflectorch.utils import get_filtering_mask, to_t

from huggingface_hub.utils import disable_progress_bars

# that causes some Rust related errors when downloading models from Huggingface
disable_progress_bars()


class InferenceModel(object):
    """Facilitates the inference process using pretrained models
    
    Args:
        config_name (str, optional): the name of the configuration file used to initialize the model (either with or without the '.yaml' extension). Defaults to None.
        model_name (str, optional): the name of the file containing the weights of the model (either with or without the '.pt' extension), only required if different than: `'model_' + config_name + '.pt'`. Defaults to None 
        root_dir (str, optional): path to root directory containing the 'configs' and 'saved_models' subdirectories, if different from the package root directory (ROOT_DIR). Defaults to None.
        weights_format (str, optional): format (extension) of the weights file, either 'pt' or 'safetensors'. Defaults to 'safetensors'.
        repo_id (str, optional): the id of the Huggingface repository from which the configuration files and model weights should be downloaded automatically if not found locally (in the 'configs' and 'saved_models' subdirectories of the root directory). Defaults to 'valentinsingularity/reflectivity'.
        trainer (PointEstimatorTrainer, optional): if provided, this trainer instance is used directly instead of being initialized from the configuration file. Defaults to None.
        device (str, optional): the Pytorch device ('cuda' or 'cpu'). Defaults to 'cuda'.
    """
    def __init__(self, config_name: str = None, model_name: str = None, root_dir:str = None, weights_format: str = 'safetensors',
                 repo_id: str = 'valentinsingularity/reflectivity', trainer: PointEstimatorTrainer = None, device='cuda'):
        self.config_name = config_name
        self.model_name = model_name
        self.root_dir = root_dir
        self.weights_format = weights_format
        self.repo_id = repo_id
        self.trainer = trainer
        self.device = device

        if trainer is None and self.config_name is not None:
            self.load_model(self.config_name, self.model_name, self.root_dir)

        self.prediction_result = None

    def load_model(self, config_name: str, model_name: str, root_dir: str) -> None:
        """Loads a model for inference

        Args:
            config_name (str): the name of the configuration file used to initialize the model (either with or without the '.yaml' extension).
            model_name (str): the name of the file containing the weights of the model (either with or without the '.pt' or '.safetensors'  extension), only required if different than: `'model_' + config_name + extension`.
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
        weights_extension = '.' + self.weights_format
        self.model_name = model_name or 'model_' + config_name_no_extension + weights_extension
        if not self.model_name.endswith(weights_extension):
            self.model_name += weights_extension
        self.model_dir = Path(root_dir) / 'saved_models' if root_dir else SAVED_MODELS_DIR

        def _download_with_fallback(filename: str, local_target_dir: Path, legacy_subfolder: str):
            """Try to download from repo root (new layout). If not found, retry with legacy `subfolder=legacy_subfolder`. Place result under local_target_dir using `local_dir`.
            """
            try: # new layout: files at repo root (same level as README.md)
                hf_hub_download(repo_id=self.repo_id + '/' + config_name, filename=filename, local_dir=str(local_target_dir))
            except Exception : # legacy layout fallback: e.g. subfolder='configs' or 'saved_models'
                hf_hub_download(repo_id=self.repo_id, filename=filename, subfolder=legacy_subfolder, local_dir=str(local_target_dir.parent))

        config_path = Path(self.config_dir) / self.config_name
        if config_path.exists():
            print(f"Configuration file `{config_path}` found locally.")
        else:
            print(f"Configuration file `{config_path}` not found locally.")
            if self.repo_id is None:
                raise ValueError("repo_id must be provided to download files from Huggingface.")
            print("Downloading from Huggingface...")
            _download_with_fallback(self.config_name, self.config_dir, legacy_subfolder='configs')

        model_path = Path(self.model_dir) / self.model_name
        if model_path.exists():
            print(f"Weights file `{model_path}` found locally.")
        else:
            print(f"Weights file `{model_path}` not found locally.")
            if self.repo_id is None:
                raise ValueError("repo_id must be provided to download files from Huggingface.")
            print("Downloading from Huggingface...")
            _download_with_fallback(self.model_name, self.model_dir, legacy_subfolder='saved_models')

        self.trainer = get_trainer_by_name(config_name=config_name, config_dir=self.config_dir, model_path=model_path, load_weights=True, inference_device = self.device)
        self.trainer.model.eval()
        
        param_model = self.trainer.loader.prior_sampler.param_model
        param_model_name = param_model.base_model.NAME if isinstance(param_model, NuisanceParamsWrapper) else param_model.NAME
        print(f'The model corresponds to a `{param_model_name}` parameterization with {self.trainer.loader.prior_sampler.max_num_layers} layers ({self.trainer.loader.prior_sampler.param_dim} predicted parameters)')
        print("Parameter types and total ranges:")
        for param, range_ in self.trainer.loader.prior_sampler.param_ranges.items():
            print(f"- {param}: {range_}")
        print("Allowed widths of the prior bound intervals (max-min):")
        for param, range_ in self.trainer.loader.prior_sampler.bound_width_ranges.items():
            print(f"- {param}: {range_}")

        if isinstance(self.trainer.loader.q_generator, ConstantQ):
            q_min = self.trainer.loader.q_generator.q[0].item()
            q_max = self.trainer.loader.q_generator.q[-1].item()
            n_q = self.trainer.loader.q_generator.q.shape[0]
            print(f'The model was trained on curves discretized at {n_q} uniform points between q_min={q_min} and q_max={q_max}')
        elif isinstance(self.trainer.loader.q_generator, VariableQ):
            q_min_range = self.trainer.loader.q_generator.q_min_range
            q_max_range = self.trainer.loader.q_generator.q_max_range
            n_q_range = self.trainer.loader.q_generator.n_q_range
            if n_q_range[0] == n_q_range[1]:
                n_q_fixed = n_q_range[0]
                print(f'The model was trained on curves discretized at exactly {n_q_fixed} uniform points, '
                    f'between q_min in [{q_min_range[0]}, {q_min_range[1]}] and q_max in [{q_max_range[0]}, {q_max_range[1]}]')
            else:
                print(f'The model was trained on curves discretized at a number between {n_q_range[0]} and {n_q_range[1]} '
                    f'of uniform points between q_min in [{q_min_range[0]}, {q_min_range[1]}] and q_max in [{q_max_range[0]}, {q_max_range[1]}]')
        
        if self.trainer.loader.smearing is not None:
            q_res_min = self.trainer.loader.smearing.sigma_min
            q_res_max = self.trainer.loader.smearing.sigma_max
            if self.trainer.loader.smearing.constant_dq == False:
                print(f"The model was trained with linear resolution smearing (dq/q) in the range [{q_res_min}, {q_res_max}]")
            elif self.trainer.loader.smearing.constant_dq == True:
                print(f"The model was trained with constant resolution smearing in the range [{q_res_min}, {q_res_max}]")

        additional_inputs = ["prior bounds"]
        if self.trainer.train_with_q_input:
            additional_inputs.append("q values")
        if self.trainer.condition_on_q_resolutions:
            additional_inputs.append("the resolution dq/q")
        if additional_inputs:
            inputs_str = ", ".join(additional_inputs)
            print(f"The following quantities are additional inputs to the network: {inputs_str}.")

    def preprocess_and_predict(self, 
                            reflectivity_curve: np.ndarray, 
                            q_values: np.ndarray = None, 
                            prior_bounds: Union[np.ndarray, List[Tuple]] = None, 
                            sigmas: np.ndarray = None,
                            q_resolution: Union[float, np.ndarray] = None,
                            ambient_sld: float = None,
                            clip_prediction: bool = True, 
                            polish_prediction: bool = False,
                            polishing_method: str = 'trf',
                            polishing_kwargs_reflectivity: dict = None,
                            use_sigmas_for_polishing: bool = False,
                            polishing_max_steps: int = None,
                            fit_growth: bool = False, 
                            max_d_change: float = 5.,
                            calc_pred_curve: bool = True,
                            calc_pred_sld_profile: bool = False,
                            calc_polished_sld_profile: bool = False,
                            sld_profile_padding_left: float = 0.2,
                            sld_profile_padding_right: float = 1.1,
                            kwargs_param_labels: dict = {},
                            
                            truncate_index_left: int = None,
                            truncate_index_right: int = None,
                            enable_error_bars_filtering: bool = True,
                            filter_threshold=0.3,
                            filter_remove_singles=True,
                            filter_remove_consecutives=True,
                            filter_consecutive=3,
                            filter_q_start_trunc=0.1,
                            ):
        """Preprocess experimental data (clean, truncate, filter, interpolate) and run prediction. This wrapper prepares inputs according to the model's Q generator calls `predict(...)` on the interpolated/padded data, and (optionally) performs a polishing step on the original data (pre-interpolation) 

        Args:
            reflectivity_curve (Union[np.ndarray, Tensor]): 1D array of experimental reflectivity values.
            q_values (Union[np.ndarray, Tensor]): 1D array of momentum transfer values for the reflectivity curve (in units of inverse angstroms).
            prior_bounds (Union[np.ndarray, List[Tuple]]): Prior bounds for all parameters, shape ``(num_params, 2)`` as ``[(min, max), …]``.
            sigmas (Union[np.ndarray, Tensor], optional): 1D array of experimental uncertainties (same length as `reflectivity_curve`). Used for error-bar filtering (if enabled) and for polishing (if requested).
            q_resolution (Union[float, np.ndarray], optional): The q resolution for neutron reflectometry models. Can be either a float (dq/q) for linear resolution smearing (e.g. 0.05 meaning 5% reolution smearing) or an array of dq values for pointwise resolution smearing. 
            ambient_sld (float, optional): The SLD of the fronting (i.e. ambient) medium for structure with fronting medium different than air.
            clip_prediction (bool, optional): If ``True``, the values of the predicted parameters are clipped to not be outside the interval set by the prior bounds. Defaults to True.
            polish_prediction (bool, optional): If ``True``, the neural network predictions are further polished using a simple least mean squares (LMS) fit. Defaults to False.
            polishing_method (str): {'trf', 'dogbox', 'lm'} SciPy least-squares method used for polishing.
            use_sigmas_for_polishing (bool): If ``True``, weigh residuals by `sigmas` during polishing.
            polishing_max_steps (int, optional): Maximum number of function evaluations for the SciPy optimizer.
            fit_growth (bool, optional): (Deprecated) If ``True``, an additional parameters is introduced during the LMS polishing to account for the change in the thickness of the upper layer during the in-situ measurement of the reflectivity curve (a linear growth is assumed). Defaults to False.
            max_d_change (float): The maximum possible change in the thickness of the upper layer during the in-situ measurement, relevant when polish_prediction and fit_growth are True. Defaults to 5. 
            calc_pred_curve (bool, optional): Whether to calculate the curve corresponding to the predicted parameters. Defaults to True.
            calc_pred_sld_profile (bool, optional): Whether to calculate the SLD profile corresponding to the predicted parameters. Defaults to False.
            calc_polished_sld_profile (bool, optional): Whether to calculate the SLD profile corresponding to the polished parameters. Defaults to False.
            sld_profile_padding_left (float, optional): Controls the amount of padding applied to the left side of the computed SLD profiles.
            sld_profile_padding_right (float, optional): Controls the amount of padding applied to the right side of the computed SLD profiles.
            truncate_index_left (int, optional): The data provided as input to the neural network will be truncated between the indices [truncate_index_left, truncate_index_right].
            truncate_index_right (int, optional): The data provided as input to the neural network will be truncated between the indices [truncate_index_left, truncate_index_right].
            enable_error_bars_filtering (bool, optional). If ``True``, the data points with high error bars (above a threshold) will be removed before constructing the input to the neural network (they are still used in the polishing step). Default to True.
            filter_threshold (float, optional). The relative threshold (dR/R) for error bar filtering. Defaults to 0.3.
            filter_remove_singles (float, optional). If ``True``, all isolated points exceeding the filtering threshold will be eliminated. Default to True.
            filter_remove_consecutives (float, optional). If ``True``, in the situation when a number of ``filter_consecutive`` consecutive points exceeding the filtering threshold are detected at a position higher than ``filter_q_start_trunc``, all the subsequent points in the curve are eliminated.
            
        Returns:
            dict: dictionary containing the predictions
        """
        
        ## Preprocess the data for inference (remove negative intensities, truncation, filer out points with high error bars)
        (q_values, reflectivity_curve, sigmas, q_resolution, 
         q_values_original, reflectivity_curve_original, sigmas_original, q_resolution_original) = self._preprocess_input_data(
            reflectivity_curve=reflectivity_curve,
            q_values=q_values,
            sigmas=sigmas,
            q_resolution=q_resolution,
            truncate_index_left=truncate_index_left,
            truncate_index_right=truncate_index_right,
            enable_error_bars_filtering=enable_error_bars_filtering,
            filter_threshold=filter_threshold,
            filter_remove_singles=filter_remove_singles,
            filter_remove_consecutives=filter_remove_consecutives,
            filter_consecutive=filter_consecutive,
            filter_q_start_trunc=filter_q_start_trunc,
        )

        ### Interpolate the experimental data if needed by the embedding network
        interp_data = self.interpolate_data_to_model_q(
            q_exp=q_values,
            refl_exp=reflectivity_curve,
            sigmas_exp=sigmas,
            q_res_exp=q_resolution,
            as_dict=True
        )

        q_model = interp_data["q_model"]
        reflectivity_curve_interp = interp_data["reflectivity"]
        sigmas_interp = interp_data.get("sigmas")
        q_resolution_interp = interp_data.get("q_resolution")
        key_padding_mask = interp_data.get("key_padding_mask")
        
        ### Make the prediction
        prediction_dict = self.predict(
            reflectivity_curve=reflectivity_curve_interp,
            q_values=q_model,
            sigmas=sigmas_interp,
            q_resolution=q_resolution_interp,
            key_padding_mask=key_padding_mask,
            prior_bounds=prior_bounds,
            ambient_sld=ambient_sld,
            clip_prediction=clip_prediction,
            polish_prediction=False, ###do the polishing outside the predict method on the full data
            supress_sld_amb_back_shift=True, ###do not shift back the slds by the ambient yet
            calc_pred_curve=calc_pred_curve,
            calc_pred_sld_profile=calc_pred_sld_profile,
            sld_profile_padding_left=sld_profile_padding_left,
            sld_profile_padding_right=sld_profile_padding_right,
            kwargs_param_labels=kwargs_param_labels,
        )
        
        ### Save interpolated data
        prediction_dict['q_model'] = q_model
        prediction_dict['reflectivity_curve_interp'] = reflectivity_curve_interp
        if q_resolution_interp is not None:
            prediction_dict['q_resolution_interp'] = q_resolution_interp 
        if sigmas_interp is not None:
            prediction_dict['sigmas_interp'] = sigmas_interp
        if key_padding_mask is not None:
            prediction_dict['key_padding_mask'] = key_padding_mask

        ### Shift the slds for nonzero ambient
        prior_bounds = np.array(prior_bounds)
        if ambient_sld:
            sld_indices = self._shift_slds_by_ambient(prior_bounds, ambient_sld)

        ### Perform polishing on the original data
        if polish_prediction:
            polishing_kwargs = polishing_kwargs_reflectivity or {}
            polishing_kwargs.setdefault('dq', q_resolution_original)

            polished_dict = self._polish_prediction(
                q=q_values_original,
                curve=reflectivity_curve_original,
                predicted_params=prediction_dict['predicted_params_object'],
                priors=prior_bounds,
                ambient_sld_tensor=torch.atleast_2d(torch.as_tensor(ambient_sld)) if ambient_sld is not None else None,
                calc_polished_sld_profile=calc_polished_sld_profile,
                sld_x_axis=torch.from_numpy(prediction_dict['predicted_sld_xaxis']),
                polishing_kwargs_reflectivity = polishing_kwargs,
                error_bars=sigmas_original if use_sigmas_for_polishing else None,
                polishing_method=polishing_method,
                polishing_max_steps=polishing_max_steps,
                fit_growth=fit_growth,
                max_d_change=max_d_change,
            )

            prediction_dict.update(polished_dict)
            if fit_growth and "polished_params_array" in prediction_dict:
                prediction_dict["param_names"].append("max_d_change")
        
        ### Shift back the slds for nonzero ambient
        if ambient_sld:
            self._restore_slds_after_ambient_shift(prediction_dict, sld_indices, ambient_sld)   

        return prediction_dict


    def predict(self, 
                reflectivity_curve: Union[np.ndarray, Tensor], 
                q_values: Union[np.ndarray, Tensor] = None, 
                prior_bounds: Union[np.ndarray, List[Tuple]] = None, 
                sigmas: Union[np.ndarray, Tensor] = None,
                key_padding_mask: Union[np.ndarray, Tensor] = None,
                q_resolution: Union[float, np.ndarray] = None,
                ambient_sld: float = None,
                clip_prediction: bool = True, 
                polish_prediction: bool = False,
                polishing_method: str = 'trf',
                polishing_kwargs_reflectivity: dict = None,
                polishing_max_steps: int = None,
                fit_growth: bool = False, 
                max_d_change: float = 5.,
                use_q_shift: bool = False, 
                calc_pred_curve: bool = True,
                calc_pred_sld_profile: bool = False,
                calc_polished_sld_profile: bool = False,
                sld_profile_padding_left: float = 0.2,
                sld_profile_padding_right: float = 1.1,
                supress_sld_amb_back_shift: bool = False,
                kwargs_param_labels: dict = {},
                ):
        """Predict the thin film parameters

        Args:
            reflectivity_curve (Union[np.ndarray, Tensor]): The reflectivity curve (which has been already preprocessed, normalized and interpolated).
            q_values (Union[np.ndarray, Tensor], optional): The momentum transfer (q) values for the reflectivity curve (in units of inverse angstroms).
            prior_bounds (Union[np.ndarray, List[Tuple]]): The prior bounds for the predicted parameters.
            sigmas (Union[np.ndarray, Tensor], optional): The error bars of the reflectivity curve, if available. They are used for filtering out points with high error bars if ``enable_error_bars_filtering`` is ``True``, as well as for the polishing step if ``use_sigmas_for_polishing`` is ``True``.
            key_padding_mask (Union[np.ndarray, Tensor], optional): The key padding mask required for some embedding networks.
            q_resolution (Union[float, np.ndarray], optional): The q resolution for neutron reflectometry models. Can be either a float dq/q for linear resolution smearing (e.g. 0.05 meaning 5% reolution smearing) or an array of dq values for pointwise resolution smearing. 
            ambient_sld (float, optional): The SLD of the fronting (i.e. ambient) medium for structure with fronting medium different than air.
            clip_prediction (bool, optional): If ``True``, the values of the predicted parameters are clipped to not be outside the interval set by the prior bounds. Defaults to True.
            polish_prediction (bool, optional): If ``True``, the neural network predictions are further polished using a simple least mean squares (LMS) fit. Defaults to False.
            polishing_method (str): Type of scipy method used for polishing.
            polishing_max_steps (int, optional): Sets the maximum number of steps for the polishing algorithm.
            fit_growth (bool, optional): (Deprecated) If ``True``, an additional parameters is introduced during the LMS polishing to account for the change in the thickness of the upper layer during the in-situ measurement of the reflectivity curve (a linear growth is assumed). Defaults to False.
            max_d_change (float): The maximum possible change in the thickness of the upper layer during the in-situ measurement, relevant when polish_prediction and fit_growth are True. Defaults to 5. 
            use_q_shift: (Deprecated) If ``True``, the prediction is performed for a batch of slightly shifted versions of the input curve and the best result is returned, which is meant to mitigate the influence of imperfect sample alignment, as introduced in Greco et al. (only for models with fixed q-discretization). Defaults to False.
            calc_pred_curve (bool, optional): Whether to calculate the curve corresponding to the predicted parameters. Defaults to True.
            calc_pred_sld_profile (bool, optional): Whether to calculate the SLD profile corresponding to the predicted parameters. Defaults to False.
            calc_polished_sld_profile (bool, optional): Whether to calculate the SLD profile corresponding to the polished parameters. Defaults to False.
            sld_profile_padding_left (float, optional): Controls the amount of padding applied to the left side of the computed SLD profiles.
            sld_profile_padding_right (float, optional): Controls the amount of padding applied to the right side of the computed SLD profiles.

        Returns:
            dict: dictionary containing the predictions
        """

        scaled_curve = self._scale_curve(reflectivity_curve)
        if prior_bounds is None:
            raise ValueError(f'Prior bounds were not provided')
        prior_bounds = np.array(prior_bounds)

        if ambient_sld:
            sld_indices = self._shift_slds_by_ambient(prior_bounds, ambient_sld)

        scaled_prior_bounds = self._scale_prior_bounds(prior_bounds)

        if isinstance(self.trainer.loader.q_generator, ConstantQ):
            q_values = self.trainer.loader.q_generator.q
        else:
            if q_values is None:
                raise ValueError(f'The q values were not provided')
            q_values = torch.atleast_2d(to_t(q_values)).to(scaled_curve)

        scaled_q_values = self.trainer.loader.q_generator.scale_q(q_values).to(torch.float32) if self.trainer.train_with_q_input else None

        if q_resolution is None and self.trainer.loader.smearing is not None:
            raise ValueError(f'The q resolution must be provided for NR models')
        
        if q_resolution is not None:
            q_resolution_tensor = torch.atleast_2d(torch.as_tensor(q_resolution)).to(scaled_curve)
            if isinstance(q_resolution, float):
                unscaled_q_resolutions = q_resolution_tensor
            else:
                unscaled_q_resolutions = (q_resolution_tensor / q_values).nanmean(dim=-1, keepdim=True) ##when q_values is padded with 0s, there will be nan at the padded positions
            scaled_q_resolutions = self.trainer.loader.smearing.scale_resolutions(unscaled_q_resolutions) if self.trainer.condition_on_q_resolutions else None
            scaled_conditioning_params = scaled_q_resolutions
            if polishing_kwargs_reflectivity is None:
                polishing_kwargs_reflectivity = {'dq': q_resolution}
        else:
            q_resolution_tensor = None
            scaled_conditioning_params = None

        if key_padding_mask is not None:
            key_padding_mask = torch.as_tensor(key_padding_mask, device=self.device)
            key_padding_mask = key_padding_mask.unsqueeze(0) if key_padding_mask.dim() == 1 else key_padding_mask

        if use_q_shift and not self.trainer.train_with_q_input:
            predicted_params = self._qshift_prediction(reflectivity_curve, scaled_prior_bounds, num = 1024, dq_coef = 1.)
        else:
            with torch.no_grad():
                self.trainer.model.eval()
    
                scaled_predicted_params = self.trainer.model(
                    curves=scaled_curve, 
                    bounds=scaled_prior_bounds, 
                    q_values=scaled_q_values,
                    conditioning_params = scaled_conditioning_params,
                    key_padding_mask = key_padding_mask,
                    unscaled_q_values = q_values,
                    )
                
                predicted_params = self.trainer.loader.prior_sampler.restore_params(torch.cat([scaled_predicted_params, scaled_prior_bounds], dim=-1))

        if clip_prediction:
            predicted_params = self.trainer.loader.prior_sampler.clamp_params(predicted_params)
        
        prediction_dict = {
            "predicted_params_object": predicted_params,
            "predicted_params_array": predicted_params.parameters.squeeze().cpu().numpy(),
            "param_names" : self.trainer.loader.prior_sampler.param_model.get_param_labels(**kwargs_param_labels)
        }
        
        key_padding_mask = None if key_padding_mask is None else key_padding_mask.squeeze().cpu().numpy()

        if calc_pred_curve:
            predicted_curve = predicted_params.reflectivity(q=q_values, dq=q_resolution_tensor).squeeze().cpu().numpy()
            prediction_dict[ "predicted_curve"] = predicted_curve if key_padding_mask is None else predicted_curve[key_padding_mask]
        
        ambient_sld_tensor = torch.atleast_2d(torch.as_tensor(ambient_sld, device=self.device)) if ambient_sld is not None else None
        if calc_pred_sld_profile: 
            predicted_sld_xaxis, predicted_sld_profile, _ = get_density_profiles(
                predicted_params.thicknesses, predicted_params.roughnesses, predicted_params.slds + (ambient_sld_tensor or 0), ambient_sld_tensor, 
                num=1024, padding_left=sld_profile_padding_left, padding_right=sld_profile_padding_right,
            ) 
            prediction_dict['predicted_sld_profile'] = predicted_sld_profile.squeeze().cpu().numpy()
            prediction_dict['predicted_sld_xaxis'] = predicted_sld_xaxis.squeeze().cpu().numpy()
        else:
            predicted_sld_xaxis = None
        
        refl_curve_polish = reflectivity_curve if key_padding_mask is None else reflectivity_curve[key_padding_mask]
        q_polish = q_values.squeeze().cpu().numpy() if key_padding_mask is None else q_values.squeeze().cpu().numpy()[key_padding_mask]
        prediction_dict['q_plot_pred'] = q_polish

        if polish_prediction:
            if ambient_sld_tensor:
                ambient_sld_tensor = ambient_sld_tensor.cpu()
            
            polished_dict = self._polish_prediction(
                q = q_polish, 
                curve = refl_curve_polish, 
                predicted_params = predicted_params, 
                priors = np.array(prior_bounds), 
                error_bars = sigmas,
                fit_growth = fit_growth,
                max_d_change = max_d_change, 
                calc_polished_curve = calc_pred_curve,
                calc_polished_sld_profile = calc_polished_sld_profile,
                ambient_sld_tensor=ambient_sld_tensor,
                sld_x_axis = predicted_sld_xaxis,
                polishing_method=polishing_method,
                polishing_max_steps=polishing_max_steps,
                polishing_kwargs_reflectivity=polishing_kwargs_reflectivity,
            )
            prediction_dict.update(polished_dict)

            if fit_growth and "polished_params_array" in prediction_dict:
                prediction_dict["param_names"].append("max_d_change")

        if ambient_sld and not supress_sld_amb_back_shift: #Note: the SLD shift will only be reflected in predicted_params_array but not in predicted_params_object; supress_sld_amb_back_shift is required for the 'preprocess_and_predict' method
            self._restore_slds_after_ambient_shift(prediction_dict, sld_indices, ambient_sld)   

        return prediction_dict
    
    def _polish_prediction(self,
                           q: np.ndarray,
                           curve: np.ndarray,
                           predicted_params: BasicParams,
                           priors: np.ndarray,
                           sld_x_axis,
                           ambient_sld_tensor: Tensor = None,
                           fit_growth: bool = False,
                           max_d_change: float = 5.,
                           calc_polished_curve: bool = True,
                           calc_polished_sld_profile: bool = False,
                           error_bars: np.ndarray = None,
                           polishing_method: str = 'trf',
                           polishing_max_steps: int = None,
                           polishing_kwargs_reflectivity: dict = None,
                           ) -> dict:
        params = predicted_params.parameters.squeeze().cpu().numpy()

        polished_params_dict = {}
        polishing_kwargs_reflectivity = polishing_kwargs_reflectivity or {}

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
                    self.trainer.loader.prior_sampler.max_num_layers,
                    self.trainer.loader.prior_sampler.param_model
                    )
            else:
                polished_params_arr, polished_params_err, curve_polished = refl_fit(
                    q = q, 
                    curve = curve, 
                    init_params = params, 
                    bounds=priors.T,
                    prior_sampler=self.trainer.loader.prior_sampler,
                    error_bars=error_bars,
                    method=polishing_method,
                    polishing_max_steps=polishing_max_steps,
                    reflectivity_kwargs=polishing_kwargs_reflectivity,
                )
                polished_params = BasicParams(
                    torch.from_numpy(polished_params_arr[None]),
                    torch.from_numpy(priors.T[0][None]),
                    torch.from_numpy(priors.T[1][None]),
                    self.trainer.loader.prior_sampler.max_num_layers,
                    self.trainer.loader.prior_sampler.param_model
                )
        except Exception as err:
            polished_params = predicted_params
            polished_params_arr = get_prediction_array(polished_params)
            curve_polished = np.zeros_like(q)
            polished_params_err = None

        polished_params_dict['polished_params_array'] = polished_params_arr
        
        polished_params_dict['polished_params_error_array'] = (
            np.array(polished_params_err) 
            if polished_params_err is not None 
            else np.full_like(polished_params, np.nan, dtype=np.float64)
            )
        if calc_polished_curve:
            polished_params_dict['polished_curve'] = curve_polished

        if ambient_sld_tensor is not None:
            ambient_sld_tensor = ambient_sld_tensor.to(polished_params.slds.device)


        if calc_polished_sld_profile:
            _, sld_profile_polished, _ = get_density_profiles(
                polished_params.thicknesses, polished_params.roughnesses, polished_params.slds + (ambient_sld_tensor or 0), ambient_sld_tensor, 
                z_axis=sld_x_axis.to(polished_params.slds.device),
            )
            polished_params_dict['sld_profile_polished'] = sld_profile_polished.squeeze().cpu().numpy()

        return polished_params_dict
    
    def _scale_curve(self, curve: Union[np.ndarray, Tensor]):
        if not isinstance(curve, Tensor):
            curve = torch.from_numpy(curve).float()
        curve = curve.unsqueeze(0).to(self.device)
        scaled_curve = self.trainer.loader.curves_scaler.scale(curve)
        return scaled_curve
    
    def _scale_prior_bounds(self, prior_bounds: List[Tuple]):
        try:
            prior_bounds = torch.tensor(prior_bounds)
            prior_bounds = prior_bounds.to(self.device).T
            min_bounds, max_bounds = prior_bounds[:, None]

            scaled_bounds = torch.cat([
                self.trainer.loader.prior_sampler.scale_bounds(min_bounds), 
                self.trainer.loader.prior_sampler.scale_bounds(max_bounds)
            ], -1)

            return scaled_bounds.float()

        except RuntimeError as e:
            expected_param_dim = self.trainer.loader.prior_sampler.param_dim
            actual_param_dim = prior_bounds.shape[1] if prior_bounds.ndim == 2 else len(prior_bounds)

            msg = (
                f"\n **Parameter dimension mismatch during inference!**\n"
                f"- Model expects **{expected_param_dim}** parameters.\n"
                f"- You provided **{actual_param_dim}** prior bounds.\n\n"
                f"💡This often occurs when:\n"
                f"- The model was trained with additional nuisance parameters like `r_scale`, `q_shift`, or `log10_background`,\n"
                f"  but they were not included in the `prior_bounds` passed to `.predict()`.\n"
                f"- The number of layers or parameterization type differs from the one used during training.\n\n"
                f" Check the configuration or the summary of expected parameters."
            )
            raise ValueError(msg) from e    
    
    def _shift_slds_by_ambient(self, prior_bounds: np.ndarray, ambient_sld: float):
        n_layers = self.trainer.loader.prior_sampler.max_num_layers
        sld_indices = slice(2*n_layers+1, 3*n_layers+2)
        prior_bounds[sld_indices, ...] -= ambient_sld

        training_min_bounds = self.trainer.loader.prior_sampler.min_bounds.squeeze().cpu().numpy()
        training_max_bounds = self.trainer.loader.prior_sampler.max_bounds.squeeze().cpu().numpy()
        lower_bound_check = (prior_bounds[sld_indices, 0] >= training_min_bounds[sld_indices]).all()
        upper_bound_check = (prior_bounds[sld_indices, 1] <= training_max_bounds[sld_indices]).all()
        assert lower_bound_check and upper_bound_check, "Shifting the layer SLDs by the ambient SLD exceeded the training ranges."

        return sld_indices
    
    def _restore_slds_after_ambient_shift(self, prediction_dict, sld_indices, ambient_sld):
        prediction_dict["predicted_params_array"][sld_indices] += ambient_sld
        if "polished_params_array" in prediction_dict:
            prediction_dict["polished_params_array"][sld_indices] += ambient_sld

    def _get_likelihood(self, q, curve, rel_err: float = 0.1, abs_err: float = 1e-12):
        return LogLikelihood(
            q, curve, self.trainer.loader.prior_sampler, curve * rel_err + abs_err
        )
    
    def get_param_labels(self, **kwargs):
        return self.trainer.loader.prior_sampler.param_model.get_param_labels(**kwargs)
    
    @staticmethod
    def _preprocess_input_data(
                           reflectivity_curve,
                           q_values,
                           sigmas=None,
                           q_resolution=None,
                           truncate_index_left=None,
                           truncate_index_right=None,
                           enable_error_bars_filtering=True,
                           filter_threshold=0.3,
                           filter_remove_singles=True,
                           filter_remove_consecutives=True,
                           filter_consecutive=3,
                           filter_q_start_trunc=0.1):
    
        # Save originals for polishing
        reflectivity_curve_original = reflectivity_curve.copy()
        q_values_original = q_values.copy() if q_values is not None else None
        q_resolution_original = q_resolution.copy() if isinstance(q_resolution, np.ndarray) else q_resolution
        sigmas_original = sigmas.copy() if sigmas is not None else None

        # Remove points with non-positive intensities
        nonnegative_mask = reflectivity_curve > 0.0
        reflectivity_curve = reflectivity_curve[nonnegative_mask]
        q_values = q_values[nonnegative_mask]
        if sigmas is not None:
            sigmas = sigmas[nonnegative_mask]
        if isinstance(q_resolution, np.ndarray):
            q_resolution = q_resolution[nonnegative_mask]

        # Truncate arrays
        if truncate_index_left is not None or truncate_index_right is not None:
            slice_obj = slice(truncate_index_left, truncate_index_right)
            reflectivity_curve = reflectivity_curve[slice_obj]
            q_values = q_values[slice_obj]
            if sigmas is not None:
                sigmas = sigmas[slice_obj]
            if isinstance(q_resolution, np.ndarray):
                q_resolution = q_resolution[slice_obj]

        # Filter high-error points
        if enable_error_bars_filtering and sigmas is not None:
            valid_mask = get_filtering_mask(
                q_values,
                reflectivity_curve,
                sigmas,
                threshold=filter_threshold,
                consecutive=filter_consecutive,
                remove_singles=filter_remove_singles,
                remove_consecutives=filter_remove_consecutives,
                q_start_trunc=filter_q_start_trunc
            )
            reflectivity_curve = reflectivity_curve[valid_mask]
            q_values = q_values[valid_mask]
            sigmas = sigmas[valid_mask]
            if isinstance(q_resolution, np.ndarray):
                q_resolution = q_resolution[valid_mask]

        return (q_values, reflectivity_curve, sigmas, q_resolution,
                q_values_original, reflectivity_curve_original,
                sigmas_original, q_resolution_original)
    
    def interpolate_data_to_model_q(
        self,
        q_exp,
        refl_exp,
        sigmas_exp=None,
        q_res_exp=None,
        as_dict=False
    ):
        q_generator = self.trainer.loader.q_generator

        def _pad(arr, pad_to, value=0.0):
            if arr is None:
                return None
            return np.pad(arr, (0, pad_to - len(arr)), constant_values=value)

        def _interp_or_keep(q_model, q_exp, arr):
            """Interpolate arrays, keep floats or None unchanged."""
            if arr is None:
                return None
            return np.interp(q_model, q_exp, arr) if isinstance(arr, np.ndarray) else arr

        def _pad_or_keep(arr, max_n):
            """Pad arrays, keep floats or None unchanged."""
            if arr is None:
                return None
            return _pad(arr, max_n, 0.0) if isinstance(arr, np.ndarray) else arr

        def _prepare_return(q, refl, sigmas=None, q_res=None, mask=None, as_dict=False):
            if as_dict:
                result = {"q_model": q, "reflectivity": refl}
                if sigmas is not None: result["sigmas"] = sigmas
                if q_res is not None: result["q_resolution"] = q_res
                if mask is not None: result["key_padding_mask"] = mask
                return result
            result = [q, refl]
            if sigmas is not None: result.append(sigmas)
            if q_res is not None: result.append(q_res)
            if mask is not None: result.append(mask)
            return tuple(result)

        # ConstantQ
        if isinstance(q_generator, ConstantQ):
            q_model = q_generator.q.cpu().numpy()
            refl_out = interp_reflectivity(q_model, q_exp, refl_exp)
            sigmas_out = _interp_or_keep(q_model, q_exp, sigmas_exp)
            q_res_out = _interp_or_keep(q_model, q_exp, q_res_exp)
            return _prepare_return(q_model, refl_out, sigmas_out, q_res_out, None, as_dict)

        # VariableQ
        elif isinstance(q_generator, VariableQ):
            if q_generator.n_q_range[0] == q_generator.n_q_range[1]:
                n_q_model = q_generator.n_q_range[0]
                q_min = max(q_exp.min(), q_generator.q_min_range[0])
                q_max = min(q_exp.max(), q_generator.q_max_range[1])
                if self.trainer.loader.q_generator.mode == 'logspace':
                    q_model = torch.logspace(start=torch.log10(torch.tensor(q_min, device=self.device)),
                                             end=torch.log10(torch.tensor(q_max, device=self.device)),
                                             steps=n_q_model, device=self.device).to('cpu')
                    logspace = True
                else:
                    q_model = np.linspace(q_min, q_max, n_q_model)
                    logspace = False
            else:
                return _prepare_return(q_exp, refl_exp, sigmas_exp, q_res_exp, None, as_dict)

            refl_out = interp_reflectivity(q_model, q_exp, refl_exp, logspace=logspace)
            sigmas_out = _interp_or_keep(q_model, q_exp, sigmas_exp)
            q_res_out = _interp_or_keep(q_model, q_exp, q_res_exp)
            return _prepare_return(q_model, refl_out, sigmas_out, q_res_out, None, as_dict)

        # MaskedVariableQ
        elif isinstance(q_generator, MaskedVariableQ):
            min_n, max_n = q_generator.n_q_range
            n_exp = len(q_exp)

            if min_n <= n_exp <= max_n:
                # Pad only
                q_model = _pad(q_exp, max_n, 0.0)
                refl_out = _pad(refl_exp, max_n, 0.0)
                sigmas_out = _pad_or_keep(sigmas_exp, max_n)
                q_res_out = _pad_or_keep(q_res_exp, max_n)
                key_padding_mask = np.zeros(max_n, dtype=bool)
                key_padding_mask[:n_exp] = True

            else:
                # Interpolate + pad
                n_interp = min(max(n_exp, min_n), max_n)
                q_min = max(q_exp.min(), q_generator.q_min_range[0])
                q_max = min(q_exp.max(), q_generator.q_max_range[1])
                q_interp = np.linspace(q_min, q_max, n_interp)

                refl_interp = interp_reflectivity(q_interp, q_exp, refl_exp)
                sigmas_interp = _interp_or_keep(q_interp, q_exp, sigmas_exp)
                q_res_interp = _interp_or_keep(q_interp, q_exp, q_res_exp)

                q_model = _pad(q_interp, max_n, 0.0)
                refl_out = _pad(refl_interp, max_n, 0.0)
                sigmas_out = _pad_or_keep(sigmas_interp, max_n)
                q_res_out = _pad_or_keep(q_res_interp, max_n)
                key_padding_mask = np.zeros(max_n, dtype=bool)
                key_padding_mask[:n_interp] = True

            return _prepare_return(q_model, refl_out, sigmas_out, q_res_out, key_padding_mask, as_dict)

        else:
            raise TypeError(f"Unsupported QGenerator type: {type(q_generator)}")
        
    def _qshift_prediction(self, curve, scaled_bounds, num: int = 1000, dq_coef: float = 1.) -> BasicParams:
        assert isinstance(self.trainer.loader.q_generator, ConstantQ), "Prediction with q shifts available only for models with fixed discretization"
        q = self.trainer.loader.q_generator.q.squeeze().float()
        dq_max = (q[1] - q[0]) * dq_coef
        q_shifts = torch.linspace(-dq_max, dq_max, num).to(q)

        curve = to_t(curve).to(scaled_bounds)
        shifted_curves = _qshift_interp(q.squeeze(), curve, q_shifts)

        assert shifted_curves.shape == (num, q.shape[0])

        scaled_curves = self.trainer.loader.curves_scaler.scale(shifted_curves)
        scaled_prior_bounds = torch.atleast_2d(scaled_bounds).expand(scaled_curves.shape[0], -1)

        with torch.no_grad():
            self.trainer.model.eval()
            scaled_predicted_params = self.trainer.model(scaled_curves, scaled_prior_bounds)
            restored_params = self.trainer.loader.prior_sampler.restore_params(torch.cat([scaled_predicted_params, scaled_prior_bounds], dim=-1))

            best_param = get_best_mse_param(
                restored_params,
                self._get_likelihood(q=self.trainer.loader.q_generator.q, curve=curve),
            )
            return best_param



EasyInferenceModel = InferenceModel

def get_prediction_array(params: BasicParams) -> np.ndarray:
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