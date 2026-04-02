import numpy as np
import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path
from huggingface_hub import hf_hub_download

from reflectorch.data_generation.priors import BasicParams
from reflectorch.data_generation.priors.parametric_models import NuisanceParamsWrapper
from reflectorch.data_generation.q_generator import ConstantQ, VariableQ, MaskedVariableQ
from reflectorch.inference.inference_result import PointInferenceResult, PosteriorInferenceResult
from reflectorch.inference.data import ReflectivityData
from reflectorch.inference.preprocess_exp.interpolation import interp_reflectivity
from reflectorch.paths import CONFIG_DIR, SAVED_MODELS_DIR
from reflectorch.runs.utils import get_trainer_by_name
from reflectorch.ml.trainers import Trainer, PointEstimatorTrainer, NFlowTrainer
from reflectorch.inference.scipy_fitter import refl_fit, get_fit_with_growth
from reflectorch.utils import get_filtering_mask, to_t

from huggingface_hub.utils import disable_progress_bars

# that causes some Rust related errors when downloading models from Huggingface
disable_progress_bars()

@dataclass
class PreparedInferenceInputs:
    scaled_curve: Tensor
    q_values: Tensor
    scaled_prior_bounds: Tensor
    restore_context: Optional[Dict[str, Tensor]]

    scaled_q_values: Optional[Tensor]
    scaled_sigmas: Optional[Tensor]
    scaled_conditioning_params: Optional[Tensor]
    key_padding_mask: Optional[Tensor]

    q_resolution_tensor: Optional[Tensor]
    prior_bounds_array: np.ndarray

    ambient_sld: Optional[float]
    sld_indices: Optional[Any]

    applied_ambient_shift: bool = False
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

    def __init__(
            self,
            config_name: Optional[str] = None,
            model_name: Optional[str] = None,
            root_dir: Optional[Union[str, Path]] = None,
            weights_format: str = "safetensors",
            repo_id: Optional[str] = "valentinsingularity/reflectivity",
            trainer: Optional[Trainer] = None,
            device: Union[str, torch.device] = "cuda",
    ) -> None:
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

    def load_model(
            self,
            config_name: str,
            model_name: Optional[str] = None,
            root_dir: Optional[Union[str, Path]] = None,
    ) -> None:
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

    def preprocess_and_predict(
            self, 
            reflectivity_curve: np.ndarray, 
            q_values: np.ndarray, 
            prior_bounds: Union[np.ndarray, Sequence[Tuple[float, float]]], 
            sigmas: Optional[np.ndarray] = None,
            q_resolution: Optional[Union[np.ndarray, float]] = None,
            ambient_sld: Optional[float] = None,
            clip_prediction: bool = True, 
            polish_prediction: bool = False,
            polishing_method: str = 'trf',
            polishing_kwargs_reflectivity: Optional[Dict[str, Any]] = None,
            use_sigmas_for_polishing: bool = False,
            polishing_max_nfev: Optional[int] = None,
            calc_polishing_param_errors: bool = False,
            fit_growth: bool = False, 
            max_d_change: float = 5.0,
            calc_pred_curve: bool = False,
            calc_pred_sld_profile: bool = False,
            calc_polished_sld_profile: bool = False,
            sld_profile_padding_left: float = 0.4,
            sld_profile_padding_right: float = 1.3,
            kwargs_param_labels: Optional[Dict[str, Any]] = None,

            return_result_as_dict: bool = True,
            original_data_in_inference_result: bool = True,
            
            truncate_index_left: Optional[int] = None,
            truncate_index_right: Optional[int] = None,
            enable_error_bars_filtering: bool = True,
            filter_threshold: float = 0.3,
            filter_remove_singles: bool = True,
            filter_remove_consecutives: bool = True,
            filter_consecutive: int = 3,
            filter_q_start_trunc: float = 0.1,
    ) -> Union[Dict[str, Any], PointInferenceResult]:
        """
        Preprocess experimental reflectivity data and run point inference.

        This is the high-level point-inference entry point. It removes invalid data
        points, optionally truncates and filters the input curve, adapts the data to
        the q-discretization expected by the loaded model, and then calls
        :meth:`predict` to obtain a neural point estimate. Optionally, the neural
        prediction can be polished using local numerical fitting on the original
        experimental data.

        Args:
            reflectivity_curve:
                Experimental reflectivity values of shape ``(n_q,)``.
            q_values:
                Experimental q-values of shape ``(n_q,)``.
            prior_bounds:
                Prior bounds for the model parameters, given as an array of shape
                ``(n_params, 2)`` or an equivalent sequence of ``(min, max)`` pairs.
            sigmas:
                Optional experimental uncertainties of shape ``(n_q,)``.
            q_resolution:
                Optional q-resolution information, either a scalar or a pointwise array.
            ambient_sld:
                Optional ambient scattering length density used for SLD shifting.
            clip_prediction:
                If ``True``, clamp predicted parameters to the prior bounds.
            polish_prediction:
                If ``True``, refine the neural point estimate by local fitting.
            polishing_method:
                Optimization method used for polishing.
            polishing_kwargs_reflectivity:
                Optional keyword arguments forwarded to the reflectivity simulator during
                polishing.
            use_sigmas_for_polishing:
                If ``True``, pass experimental uncertainties to the polishing routine.
            polishing_max_nfev:
                Optional maximum number of function evaluations for polishing.
            fit_growth:
                If ``True``, include an additional thickness-growth parameter during
                polishing.
            max_d_change:
                Maximum allowed thickness change when ``fit_growth`` is enabled.
            calc_pred_curve:
                If ``True``, compute the reflectivity curve implied by the neural
                prediction.
            calc_pred_sld_profile:
                If ``True``, compute the SLD profile implied by the neural prediction.
            calc_polished_sld_profile:
                If ``True``, compute the SLD profile for the polished solution.
            sld_profile_padding_left:
                Left padding factor used for the SLD profile depth axis.
            sld_profile_padding_right:
                Right padding factor used for the SLD profile depth axis.
            kwargs_param_labels:
                Optional keyword arguments forwarded to parameter-label generation.
            return_result_as_dict:
                If ``True``, return a plain dictionary. Otherwise return a
                ``PointInferenceResult`` instance.
            original_data_in_inference_result:
                If ``True``, store the original experimental data in the returned
                inference result object; otherwise store the preprocessed data.
            truncate_index_left:
                Optional left truncation index applied before inference.
            truncate_index_right:
                Optional right truncation index applied before inference.
            enable_error_bars_filtering:
                If ``True``, remove points with overly large uncertainties according to
                the filtering settings.
            filter_threshold:
                Threshold used in the uncertainty-based filtering mask.
            filter_remove_singles:
                Whether isolated bad points should be removed by the filtering mask.
            filter_remove_consecutives:
                Whether consecutive bad points should be removed by the filtering mask.
            filter_consecutive:
                Minimum run length used for consecutive-point filtering.
            filter_q_start_trunc:
                Lower-q threshold below which filtering is not applied.

        Returns:
            Either a dictionary of prediction outputs or a ``PointInferenceResult``,
            depending on ``return_result_as_dict``.
        """
        
        kwargs_param_labels = kwargs_param_labels or {}
        
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
        original_prior_bounds = prior_bounds.copy()
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
                ambient_sld=ambient_sld,
                calc_polished_sld_profile=calc_polished_sld_profile,
                sld_x_axis=torch.from_numpy(prediction_dict['predicted_sld_xaxis']) if 'predicted_sld_xaxis' in prediction_dict else None,
                polishing_kwargs_reflectivity = polishing_kwargs,
                error_bars=sigmas_original if use_sigmas_for_polishing else None,
                polishing_method=polishing_method,
                polishing_max_nfev=polishing_max_nfev,
                fit_growth=fit_growth,
                max_d_change=max_d_change,
                calc_polishing_param_errors=calc_polishing_param_errors,
            )

            prediction_dict.update(polished_dict)
        
        ### Shift back the slds for nonzero ambient
        if ambient_sld:
            self._restore_slds_after_ambient_shift(prediction_dict, sld_indices, ambient_sld)   

        if return_result_as_dict:
            return prediction_dict
        
        else:
            if original_data_in_inference_result:
                data = ReflectivityData(
                    q=q_values_original,
                    R=reflectivity_curve_original,
                    dR=sigmas_original,
                    dq=q_resolution_original,
                )
            else:
                data = ReflectivityData(
                    q=q_values,
                    R=reflectivity_curve,
                    dR=sigmas,
                    dq=q_resolution,
                )

            return PointInferenceResult(
                inference_model=self,
                prediction_dict=prediction_dict,
                data=data,
                ambient_sld=ambient_sld,
                prior_bounds=original_prior_bounds,
                device=self.device,
            )
        
    def predict(
            self,
            reflectivity_curve: Union[np.ndarray, Tensor],
            prior_bounds: Union[np.ndarray, Sequence[Tuple[float, float]]],
            q_values: Optional[Union[np.ndarray, Tensor]] = None,
            sigmas: Optional[Union[np.ndarray, Tensor]] = None,
            key_padding_mask: Optional[Union[np.ndarray, Tensor]] = None,
            q_resolution: Optional[Union[np.ndarray, Tensor, float]] = None,
            ambient_sld: Optional[float] = None,
            clip_prediction: bool = True,
            polish_prediction: bool = False,
            polishing_method: str = "trf",
            polishing_kwargs_reflectivity: Optional[Dict[str, Any]] = None,
            polishing_max_nfev: Optional[int] = None,
            calc_polishing_param_errors: bool = False,
            fit_growth: bool = False,
            max_d_change: float = 5.0,
            calc_pred_curve: bool = True,
            calc_pred_sld_profile: bool = False,
            calc_polished_sld_profile: bool = False,
            sld_profile_padding_left: float = 0.4,
            sld_profile_padding_right: float = 1.3,
            supress_sld_amb_back_shift: bool = False,
            kwargs_param_labels: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Predict a single set of thin-film parameters from preprocessed input data.

        This method performs neural point inference using a model trained with
        ``PointEstimatorTrainer``. The input reflectivity curve is assumed to already
        be preprocessed and compatible with the model input format, or to be supplied
        together with the information needed to construct that format.

        Depending on the loaded model configuration, the network may additionally
        condition on prior bounds, q-values, experimental uncertainties and q-resolution.
        The method can also optionally compute the predicted
        reflectivity curve, the corresponding SLD profile, and a polished solution
        obtained by local numerical fitting.

        Args:
            reflectivity_curve:
                Reflectivity curve to infer from.
            q_values:
                q-values associated with the curve. May be omitted for models trained
                on a fixed q-grid.
            prior_bounds:
                Prior bounds for the model parameters, given as an array of shape
                ``(n_params, 2)`` or an equivalent sequence of ``(min, max)`` pairs.
            sigmas:
                Optional experimental uncertainties for the reflectivity curve.
            key_padding_mask:
                Optional boolean mask indicating valid entries for padded variable-length
                q-grids.
            q_resolution:
                Optional q-resolution information, either as a scalar or pointwise array.
            ambient_sld:
                Optional ambient scattering length density used to shift SLD-related
                parameters during inference.
            clip_prediction:
                If ``True``, clamp predicted parameters to the prior bounds.
            polish_prediction:
                If ``True``, refine the neural prediction using local numerical fitting.
            polishing_method:
                Optimization method passed to the polishing routine.
            polishing_kwargs_reflectivity:
                Optional keyword arguments forwarded to the reflectivity simulator during
                polishing.
            polishing_max_nfev:
                Optional maximum number of function evaluations for polishing.
            fit_growth:
                If ``True``, include an additional thickness-growth parameter during
                polishing.
            max_d_change:
                Maximum allowed thickness change when ``fit_growth`` is enabled.
            calc_pred_curve:
                If ``True``, compute the reflectivity curve implied by the prediction.
            calc_pred_sld_profile:
                If ``True``, compute the SLD profile implied by the prediction.
            calc_polished_sld_profile:
                If ``True``, compute the SLD profile for the polished solution.
            sld_profile_padding_left:
                Left padding factor used for the SLD profile depth axis.
            sld_profile_padding_right:
                Right padding factor used for the SLD profile depth axis.
            supress_sld_amb_back_shift:
                If ``True``, do not shift SLD values back by the ambient SLD in the
                returned arrays.
            kwargs_param_labels:
                Optional keyword arguments forwarded to parameter-label generation.

        Returns:
            A dictionary containing the predicted parameters and, depending on the
            chosen options, additional derived outputs such as reflectivity curves,
            SLD profiles, and polished results.
        """
        kwargs_param_labels = kwargs_param_labels or {}

        self._validate_point_trainer()

        prepared = self._prepare_common_inference_inputs(
            reflectivity_curve=reflectivity_curve,
            q_values=q_values,
            prior_bounds=prior_bounds,
            sigmas=sigmas,
            key_padding_mask=key_padding_mask,
            q_resolution=q_resolution,
            ambient_sld=ambient_sld,
        )

        predicted_params = self._run_point_estimator_on_scaled_bounds(
            prepared=prepared,
            scaled_prior_bounds=prepared.scaled_prior_bounds,
            restore_context=prepared.restore_context,
        )

        if clip_prediction:
            predicted_params = self.trainer.loader.prior_sampler.clamp_params(predicted_params)
        
        prediction_dict = {
            "predicted_params_object": predicted_params,
            "predicted_params_array": predicted_params.parameters.squeeze().cpu().numpy(),
            "param_names" : self.trainer.loader.prior_sampler.param_model.get_param_labels(**kwargs_param_labels)
        }

        key_padding_mask_np = (
            None
            if prepared.key_padding_mask is None
            else prepared.key_padding_mask.squeeze().cpu().numpy()
        )

        if calc_pred_curve:
            predicted_curve = predicted_params.reflectivity(
                q=prepared.q_values,
                dq=prepared.q_resolution_tensor,
            ).squeeze().cpu().numpy()

            prediction_dict["predicted_curve"] = (
                predicted_curve
                if key_padding_mask_np is None
                else predicted_curve[..., key_padding_mask_np]
            )

        if calc_pred_sld_profile:
            predicted_sld_xaxis, profile_dict = self._compute_available_profiles(
                predicted_params,
                ambient_sld=ambient_sld,
                num=1024,
                padding_left=sld_profile_padding_left,
                padding_right=sld_profile_padding_right,
            )

            prediction_dict["predicted_sld_xaxis"] = predicted_sld_xaxis.squeeze().cpu().numpy()

            for profile_type, profile in profile_dict.items():
                prediction_dict[f"predicted_{profile_type}_profile"] = profile.squeeze().cpu().numpy()
        else:
            predicted_sld_xaxis = None

        reflectivity_curve_np = (
            reflectivity_curve.detach().cpu().numpy()
            if isinstance(reflectivity_curve, Tensor)
            else np.asarray(reflectivity_curve)
        )

        refl_curve_polish = (
            reflectivity_curve_np
            if key_padding_mask_np is None
            else reflectivity_curve_np[..., key_padding_mask_np]
        )

        q_polish = (
            prepared.q_values.squeeze().cpu().numpy()
            if key_padding_mask_np is None
            else prepared.q_values.squeeze().cpu().numpy()[key_padding_mask_np]
        )
        
        prediction_dict["q_plot_pred"] = q_polish
        
        if polish_prediction:            
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
                ambient_sld=ambient_sld,
                sld_x_axis = predicted_sld_xaxis,
                polishing_method=polishing_method,
                polishing_max_nfev=polishing_max_nfev,
                polishing_kwargs_reflectivity=polishing_kwargs_reflectivity,
                calc_polishing_param_errors=calc_polishing_param_errors,
            )
            prediction_dict.update(polished_dict)

        if prepared.applied_ambient_shift and not supress_sld_amb_back_shift: #Note: the SLD shift will only be reflected in predicted_params_array but not in predicted_params_object; supress_sld_amb_back_shift is required for the 'preprocess_and_predict' method
            self._restore_slds_after_ambient_shift(
                prediction_dict,
                prepared.sld_indices,
                ambient_sld,
            )

        return prediction_dict
    
    def _run_point_estimator_on_scaled_bounds(
            self,
            prepared: PreparedInferenceInputs,
            scaled_prior_bounds: Tensor,
            restore_context: Optional[Dict[str, Tensor]],
            scaled_curve: Optional[Tensor] = None,
            q_values: Optional[Tensor] = None,
            scaled_q_values: Optional[Tensor] = None,
            scaled_sigmas: Optional[Tensor] = None,
            scaled_conditioning_params: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
    ) -> BasicParams:
        """
        Run the point-estimator model on already-prepared scaled inputs and restore
        parameters to physical space.
        """
        scaled_curve = prepared.scaled_curve if scaled_curve is None else scaled_curve
        q_values = prepared.q_values if q_values is None else q_values
        scaled_q_values = prepared.scaled_q_values if scaled_q_values is None else scaled_q_values
        scaled_sigmas = prepared.scaled_sigmas if scaled_sigmas is None else scaled_sigmas
        scaled_conditioning_params = (
            prepared.scaled_conditioning_params
            if scaled_conditioning_params is None
            else scaled_conditioning_params
        )
        key_padding_mask = prepared.key_padding_mask if key_padding_mask is None else key_padding_mask

        with torch.no_grad():
            self.trainer.model.eval()
            scaled_predicted_params = self.trainer.model(
                curves=scaled_curve,
                bounds=scaled_prior_bounds,
                q_values=scaled_q_values,
                sigmas=scaled_sigmas,
                conditioning_params=scaled_conditioning_params,
                key_padding_mask=key_padding_mask,
                unscaled_q_values=q_values,
            )

        return self._restore_predicted_params_for_inference(
            scaled_predicted_params=scaled_predicted_params,
            scaled_prior_bounds=scaled_prior_bounds,
            restore_context=restore_context,
        )
    
    def preprocess_and_sample(
            self,
            reflectivity_curve: np.ndarray,
            q_values: np.ndarray,
            prior_bounds: Union[np.ndarray, Sequence[Tuple[float, float]]],
            sigmas: Optional[np.ndarray] = None,
            q_resolution: Optional[Union[np.ndarray, float]] = None,
            ambient_sld: Optional[float] = None,
            num_samples: int = 1000,
            sampling_batch_size: Optional[int] = None,
            sld_profile_padding_left: float = 0.4,
            sld_profile_padding_right: float = 1.3,
            kwargs_param_labels: Optional[Dict[str, Any]] = None,
            calc_sampled_curves: bool = False,
            maximum_sim_batch_size: Optional[int] = None,
            calc_sampled_sld_profiles: bool = False,
            calc_log_likelihoods: bool = False,
            clip_prediction: bool = False,
            enable_importance_sampling: bool = False,
            rel_err_factor: float = 0.2,
            return_result_as_dict: bool = True,
            original_data_in_inference_result: bool = True,
            truncate_index_left: Optional[int] = None,
            truncate_index_right: Optional[int] = None,
            enable_error_bars_filtering: bool = True,
            filter_threshold: float = 0.3,
            filter_remove_singles: bool = True,
            filter_remove_consecutives: bool = True,
            filter_consecutive: int = 3,
            filter_q_start_trunc: float = 0.1,
    ) -> Union[Dict[str, Any], PosteriorInferenceResult]:
        """
        Preprocess experimental reflectivity data and run posterior sampling.

        This is the high-level posterior-inference entry point. It removes invalid data
        points, optionally truncates and filters the input curve, adapts the data to
        the q-discretization expected by the loaded model, and then calls
        :meth:`sample` to draw posterior samples. Depending on the chosen options, it
        can also compute derived outputs such as simulated curves, SLD profiles,
        likelihood values, and proposal log-probabilities for importance sampling.

        Args:
            reflectivity_curve:
                Experimental reflectivity values of shape ``(n_q,)``.
            q_values:
                Experimental q-values of shape ``(n_q,)``.
            prior_bounds:
                Prior bounds for the model parameters, given as an array of shape
                ``(n_params, 2)`` or an equivalent sequence of ``(min, max)`` pairs.
            sigmas:
                Optional experimental uncertainties of shape ``(n_q,)``.
            q_resolution:
                Optional q-resolution information, either a scalar or a pointwise array.
            ambient_sld:
                Optional ambient scattering length density used for SLD shifting.
            num_samples:
                Number of posterior samples to draw.
            sampling_batch_size:
                Optional batch size used to draw samples in multiple chunks.
            sld_profile_padding_left:
                Left padding factor used for sampled SLD profile depth axes.
            sld_profile_padding_right:
                Right padding factor used for sampled SLD profile depth axes.
            kwargs_param_labels:
                Optional keyword arguments forwarded to parameter-label generation.
            calc_sampled_curves:
                If ``True``, simulate reflectivity curves for the sampled parameters.
            maximum_sim_batch_size:
                Maximum batch size used when simulating sampled curves.
            calc_sampled_sld_profiles:
                If ``True``, compute SLD profiles for the sampled parameters.
            calc_log_likelihoods:
                If ``True``, compute log-likelihood values for the sampled curves.
            clip_prediction:
                If ``True``, clamp sampled parameters to the prior bounds after restoring
                them to physical space.
            enable_importance_sampling:
                If ``True``, also compute proposal log-probabilities in physical
                parameter space.
            rel_err_factor:
                Relative error factor used when constructing fallback uncertainties for
                likelihood evaluation.
            return_result_as_dict:
                If ``True``, return a plain dictionary. Otherwise return a
                ``PosteriorInferenceResult`` instance.
            original_data_in_inference_result:
                If ``True``, store the original experimental data in the returned
                inference result object; otherwise store the preprocessed data.
            truncate_index_left:
                Optional left truncation index applied before inference.
            truncate_index_right:
                Optional right truncation index applied before inference.
            enable_error_bars_filtering:
                If ``True``, remove points with overly large uncertainties according to
                the filtering settings.
            filter_threshold:
                Threshold used in the uncertainty-based filtering mask.
            filter_remove_singles:
                Whether isolated bad points should be removed by the filtering mask.
            filter_remove_consecutives:
                Whether consecutive bad points should be removed by the filtering mask.
            filter_consecutive:
                Minimum run length used for consecutive-point filtering.
            filter_q_start_trunc:
                Lower-q threshold below which filtering is not applied.

        Returns:
            Either a dictionary of sampling outputs or a ``PosteriorInferenceResult``,
            depending on ``return_result_as_dict``.
        """
        
        kwargs_param_labels = kwargs_param_labels or {}
        
        ## Preprocess the data (filtering, truncation, error bar filtering)
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

        ### Interpolate or pad reflectivity data to model-specific q-discretization
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

        original_prior_bounds = prior_bounds.copy()

        ### Call the sampling method
        prediction_dict = self.sample(
            num_samples=num_samples,
            reflectivity_curve=reflectivity_curve_interp,
            q_values=q_model,
            prior_bounds=prior_bounds,
            sigmas=sigmas_interp,
            key_padding_mask=key_padding_mask,
            q_resolution=q_resolution_interp,
            ambient_sld=ambient_sld,
            sld_profile_padding_left=sld_profile_padding_left,
            sld_profile_padding_right=sld_profile_padding_right,
            sampling_batch_size=sampling_batch_size,
            enable_importance_sampling=enable_importance_sampling,
            rel_err_factor=rel_err_factor,
            calc_sampled_curves=calc_sampled_curves,
            maximum_sim_batch_size=maximum_sim_batch_size,
            calc_sampled_sld_profiles=calc_sampled_sld_profiles,
            calc_log_likelihoods=calc_log_likelihoods,
            clip_prediction=clip_prediction,
            kwargs_param_labels=kwargs_param_labels,
        )

        prediction_dict['q_model'] = q_model
        prediction_dict['reflectivity_curve_interp'] = reflectivity_curve_interp
        if q_resolution_interp is not None:
            prediction_dict['q_resolution_interp'] = q_resolution_interp
        if sigmas_interp is not None:
            prediction_dict['sigmas_interp'] = sigmas_interp
        if key_padding_mask is not None:
            prediction_dict['key_padding_mask'] = key_padding_mask

        if return_result_as_dict:
            return prediction_dict
           
        else:
            if original_data_in_inference_result:
                data = ReflectivityData(
                    q=q_values_original,
                    R=reflectivity_curve_original,
                    dR=sigmas_original,
                    dq=q_resolution_original,
                )
            else:
                data = ReflectivityData(
                    q=q_values,
                    R=reflectivity_curve,
                    dR=sigmas,
                    dq=q_resolution,
                )

            return PosteriorInferenceResult(
                inference_model=self,
                prediction_dict=prediction_dict,
                data=data,
                ambient_sld=ambient_sld,
                prior_bounds=original_prior_bounds,
                device=self.device,
            )
        
    def sample(
            self,
            num_samples: int,
            reflectivity_curve: Union[np.ndarray, Tensor],
            prior_bounds: Union[np.ndarray, Sequence[Tuple[float, float]]],
            q_values: Optional[Union[np.ndarray, Tensor]] = None,
            sigmas: Optional[Union[np.ndarray, Tensor]] = None,
            key_padding_mask: Optional[Union[np.ndarray, Tensor]] = None,
            q_resolution: Optional[Union[np.ndarray, Tensor, float]] = None,
            ambient_sld: Optional[float] = None,
            clip_prediction: bool = False,
            sld_profile_padding_left: float = 0.4,
            sld_profile_padding_right: float = 1.3,
            calc_sampled_curves: bool = False,
            maximum_sim_batch_size: Optional[int] = None,
            calc_sampled_sld_profiles: bool = False,
            calc_log_likelihoods: bool = False,
            sampling_batch_size: Optional[int] = None,
            enable_importance_sampling: bool = False,
            rel_err_factor: float = 0.2,
            kwargs_param_labels: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Draw posterior parameter samples from a neural posterior model.

        This method performs posterior sampling using a model trained with
        ``NFlowTrainer``.
        It supports optional conditioning on prior bounds, q-values, q-resolution,
        uncertainties, and can additionally compute derived quantities
        such as simulated reflectivity curves, SLD profiles, and proposal
        log-probabilities for importance sampling.

        Args:
            num_samples:
                Number of posterior samples to draw.
            reflectivity_curve:
                Reflectivity curve used for conditioning.
            q_values:
                q-values associated with the curve. May be omitted for models trained
                on a fixed q-grid.
            prior_bounds:
                Prior bounds for the model parameters, given as an array of shape
                ``(n_params, 2)`` or an equivalent sequence of ``(min, max)`` pairs.
            sigmas:
                Optional experimental uncertainties for the reflectivity curve.
            key_padding_mask:
                Optional boolean mask indicating valid entries for padded variable-length
                q-grids.
            q_resolution:
                Optional q-resolution information, either as a scalar or pointwise array.
            ambient_sld:
                Optional ambient scattering length density used to shift SLD-related
                parameters during inference.
            clip_prediction:
                If ``True``, clamp sampled parameters to the prior bounds after restoring
                them to physical space.
            polishing_kwargs_reflectivity:
                Optional reflectivity keyword arguments retained for compatibility with
                downstream polishing or simulation settings.
            sld_profile_padding_left:
                Left padding factor used for sampled SLD profile depth axes.
            sld_profile_padding_right:
                Right padding factor used for sampled SLD profile depth axes.
            calc_sampled_curves:
                If ``True``, simulate reflectivity curves for the sampled parameters.
            maximum_sim_batch_size:
                Maximum batch size used when simulating sampled curves.
            calc_sampled_sld_profiles:
                If ``True``, compute SLD profiles for the sampled parameters.
            calc_log_likelihoods:
                If ``True``, compute log-likelihood values for the sampled curves.
            sampling_batch_size:
                Optional batch size for drawing samples in multiple chunks.
            enable_importance_sampling:
                If ``True``, also compute proposal log-probabilities in physical
                parameter space.
            rel_err_factor:
                Relative error factor used when constructing fallback uncertainties for
                likelihood evaluation.
            kwargs_param_labels:
                Optional keyword arguments forwarded to parameter-label generation.

        Returns:
            A dictionary containing sampled parameters and, depending on the chosen
            options, additional derived outputs such as simulated curves, SLD profiles,
            log-likelihoods, and proposal log-probabilities.
        """
        kwargs_param_labels = kwargs_param_labels or {}

        self._validate_posterior_trainer()

        prepared = self._prepare_common_inference_inputs(
            reflectivity_curve=reflectivity_curve,
            q_values=q_values,
            prior_bounds=prior_bounds,
            sigmas=sigmas,
            key_padding_mask=key_padding_mask,
            q_resolution=q_resolution,
            ambient_sld=ambient_sld,
        )

        self.trainer.model.eval()

        ### draw samples in scaled parameter space
        scaled_log_prob = None

        if not enable_importance_sampling:

            def draw_batch_of_samples(n):
                with torch.no_grad():
                    if isinstance(self.trainer, NFlowTrainer):
                        scaled_predicted_params = self.trainer.model.sample(
                            num_samples=n,
                            curves=prepared.scaled_curve,
                            bounds=(prepared.scaled_prior_bounds if self.trainer.train_with_bounds else None),
                            q_values=prepared.scaled_q_values,
                            sigmas=prepared.scaled_sigmas,
                            conditioning_params=prepared.scaled_conditioning_params,
                            key_padding_mask=prepared.key_padding_mask,
                            unscaled_q_values=prepared.q_values,
                        )
                        scaled_predicted_params = scaled_predicted_params.squeeze(0)

                    else:
                        raise RuntimeError("Unsupported trainer type.")

                return scaled_predicted_params

            if sampling_batch_size is None or sampling_batch_size >= num_samples:
                scaled_predicted_params = draw_batch_of_samples(num_samples)
            else:
                pieces = []
                remaining = num_samples
                while remaining > 0:
                    this_n = min(sampling_batch_size, remaining)
                    pieces.append(draw_batch_of_samples(this_n))
                    remaining -= this_n
                    print(f'Sampled: {num_samples - remaining} / {num_samples}')
                scaled_predicted_params = torch.cat(pieces, dim=0)

        else:

            def draw_batch_of_samples_and_logprob(n):
                with torch.no_grad():
                    if isinstance(self.trainer, NFlowTrainer):
                        scaled_predicted_params, scaled_log_prob = self.trainer.model.sample_and_log_prob(
                            num_samples=n,
                            curves=prepared.scaled_curve,
                            bounds=(prepared.scaled_prior_bounds if self.trainer.train_with_bounds else None),
                            q_values=prepared.scaled_q_values,
                            sigmas=prepared.scaled_sigmas,
                            conditioning_params=prepared.scaled_conditioning_params,
                            key_padding_mask=prepared.key_padding_mask,
                            unscaled_q_values=prepared.q_values,
                        )
                        # shapes [1, n, d], [1, n]
                        scaled_predicted_params = scaled_predicted_params.squeeze(0)
                        scaled_log_prob = scaled_log_prob.squeeze(0)
                    else:
                        raise RuntimeError(
                            "Importance sampling is currently supported here for "
                            "NFlowTrainer."
                        )

                return scaled_predicted_params, scaled_log_prob

            if sampling_batch_size is None or sampling_batch_size >= num_samples:
                scaled_predicted_params, scaled_log_prob = draw_batch_of_samples_and_logprob(num_samples)
            else:
                params_pieces = []
                logprob_pieces = []
                remaining = num_samples
                while remaining > 0:
                    this_n = min(sampling_batch_size, remaining)
                    p, lp = draw_batch_of_samples_and_logprob(this_n)
                    params_pieces.append(p)
                    logprob_pieces.append(lp)
                    remaining -= this_n
                    print(f'Sampled: {num_samples - remaining} / {num_samples}')
                scaled_predicted_params = torch.cat(params_pieces, dim=0)
                scaled_log_prob = torch.cat(logprob_pieces, dim=0)

        ### restore to physical parameter space
        prior_sampler = self.trainer.loader.prior_sampler

        if prepared.scaled_prior_bounds.shape[0] == 1 and scaled_predicted_params.shape[0] > 1:
            scaled_prior_bounds_rep = prepared.scaled_prior_bounds.expand(scaled_predicted_params.shape[0], -1)
        else:
            scaled_prior_bounds_rep = prepared.scaled_prior_bounds

        scaled_full = torch.cat([scaled_predicted_params, scaled_prior_bounds_rep], dim=-1)

        q_logdet = None

        if not self._uses_q_params_transform():
            if enable_importance_sampling:
                predicted_params, logdet_transform = prior_sampler.restore_params_with_logdet(scaled_full)
            else:
                predicted_params = prior_sampler.restore_params(scaled_full)

        else:
            restore_context = self._expand_restore_context(
                prepared.restore_context,
                batch_size=scaled_predicted_params.shape[0],
            )

            if enable_importance_sampling:
                predicted_params_k, logdet_transform = prior_sampler.restore_params_with_logdet_custom_range(
                    scaled_full,
                    restore_context["min_range_k"],
                    restore_context["max_range_k"],
                )
            else:
                predicted_params_k = prior_sampler.restore_params_custom_range(
                    scaled_full,
                    restore_context["min_range_k"],
                    restore_context["max_range_k"],
                )

            # map transformed-space params back to physical parameter space
            q_ratio_inv = 1.0 / restore_context["q_ratio"]
            predicted_params = predicted_params_k.scale_with_q(q_ratio=q_ratio_inv)

            if enable_importance_sampling:
                # log |det d(theta_k)/d(theta)| contribution
                q_logdet = prior_sampler.param_model.logdet_scale_with_q(
                    batch_size=scaled_predicted_params.shape[0],
                    q_ratio=restore_context["q_ratio"],
                )

        if clip_prediction:
            predicted_params = self.trainer.loader.prior_sampler.clamp_params(predicted_params)

        prediction_dict = {
            "predicted_params_object": predicted_params,
            "predicted_params_array": predicted_params.parameters.squeeze().cpu().numpy(),
            "param_names": self.trainer.loader.prior_sampler.param_model.get_param_labels(**kwargs_param_labels),
        }

        key_padding_mask_np = (
            None
            if prepared.key_padding_mask is None
            else prepared.key_padding_mask.squeeze().cpu().numpy()
        )

        ### proposal log-probability in physical space
        if enable_importance_sampling:
            if not self._uses_q_params_transform():
                unscaled_log_prob = scaled_log_prob - logdet_transform
            else:
                unscaled_log_prob = scaled_log_prob - logdet_transform + q_logdet

            prediction_dict["unscaled_log_prob"] = unscaled_log_prob

        ### optional outputs
        q_resolution_tensor = prepared.q_resolution_tensor
        if calc_sampled_curves:
            if q_resolution_tensor is not None and q_resolution_tensor.shape[0] == 1:
                q_resolution_tensor = q_resolution_tensor.expand(num_samples, -1)


            predicted_curves = self._batched_reflectivity(
                predicted_params,
                prepared.q_values,
                q_resolution_tensor,
                max_batch=maximum_sim_batch_size,
            )

            prediction_dict["sampled_curves"] = (
                predicted_curves
                if key_padding_mask_np is None
                else predicted_curves[..., key_padding_mask_np]
            )
            prediction_dict["q_plot_pred"] = (
                prepared.q_values.squeeze().cpu().numpy()
                if key_padding_mask_np is None
                else prepared.q_values.squeeze().cpu().numpy()[key_padding_mask_np]
            )

            if calc_log_likelihoods:
                reflectivity_curve_np = (
                    reflectivity_curve.detach().cpu().numpy()
                    if isinstance(reflectivity_curve, Tensor)
                    else np.asarray(reflectivity_curve)
                )
                sigmas_np = (
                    None
                    if sigmas is None
                    else (
                        sigmas.detach().cpu().numpy()
                        if isinstance(sigmas, Tensor)
                        else np.asarray(sigmas)
                    )
                )

                curve_for_ll = (
                    reflectivity_curve_np
                    if key_padding_mask_np is None
                    else reflectivity_curve_np[..., key_padding_mask_np]
                )
                sigmas_for_ll = (
                    sigmas_np
                    if (sigmas_np is None or key_padding_mask_np is None)
                    else sigmas_np[..., key_padding_mask_np]
                )

                log_likelihoods = self._compute_gaussian_log_likelihoods(
                    sampled_curves=prediction_dict["sampled_curves"],
                    curve_exp=curve_for_ll,
                    sigmas_exp=sigmas_for_ll,
                    rel_err_factor=rel_err_factor,
                )
                prediction_dict["log_likelihoods"] = log_likelihoods

                best_idx = int(np.argmax(log_likelihoods))
                worst_idx = int(np.argmin(log_likelihoods))

                print(f"Index of best sample: {best_idx}  Index of worst sample: {worst_idx}")

        if calc_sampled_sld_profiles:
            sampled_sld_xaxis, profile_dict = self._compute_available_profiles(
                predicted_params,
                ambient_sld=ambient_sld,
                num=1024,
                padding_left=sld_profile_padding_left,
                padding_right=sld_profile_padding_right,
            )

            prediction_dict["sampled_sld_xaxis"] = sampled_sld_xaxis.cpu().numpy()

            for profile_type, profile in profile_dict.items():
                prediction_dict[f"sampled_{profile_type}_profiles"] = profile.cpu().numpy()

        if prepared.applied_ambient_shift: # note: this only updates the numpy arrays in prediction_dict, not predicted_params_object
            self._restore_slds_after_ambient_shift(
                prediction_dict,
                prepared.sld_indices,
                ambient_sld,
            )

        return prediction_dict
    
    def _prepare_common_inference_inputs(
            self,
            reflectivity_curve: Union[np.ndarray, Tensor],
            q_values: Optional[Union[np.ndarray, Tensor]],
            prior_bounds: Union[np.ndarray, Sequence[Tuple[float, float]]],
            sigmas: Optional[Union[np.ndarray, Tensor]] = None,
            key_padding_mask: Optional[Union[np.ndarray, Tensor]] = None,
            q_resolution: Optional[Union[np.ndarray, Tensor, float]] = None,
            ambient_sld: Optional[float] = None,
        ) -> PreparedInferenceInputs:
        """
        Prepare and normalize all common inputs used by both point prediction
        and posterior sampling.

        This includes:
        - scaling the reflectivity curve
        - handling ambient-SLD shifts on prior bounds
        - preparing q-values and masks
        - scaling prior bounds
        - preparing optional conditioning inputs (q, sigmas, dq/q)
        """
        scaled_curve = self._scale_curve(reflectivity_curve)
        prior_bounds_array = np.array(prior_bounds)

        supports_shift = self._supports_zero_ambient_sld_shift()
        sld_indices = None
        applied_ambient_shift = False

        if ambient_sld is not None and supports_shift:
            sld_indices = self._shift_slds_by_ambient(prior_bounds_array, ambient_sld)
            applied_ambient_shift = True

        # q-values
        if isinstance(self.trainer.loader.q_generator, ConstantQ):
            q_values_t = self.trainer.loader.q_generator.q
        else:
            q_values_t = torch.atleast_2d(to_t(q_values)).to(scaled_curve)

        # padding mask
        key_padding_mask_t = None
        if key_padding_mask is not None:
            key_padding_mask_t = torch.as_tensor(key_padding_mask, device=self.device)
            key_padding_mask_t = (
                key_padding_mask_t.unsqueeze(0)
                if key_padding_mask_t.dim() == 1
                else key_padding_mask_t
            )

        # bounds
        scaled_prior_bounds, restore_context = self._prepare_scaled_prior_bounds_for_network(
            prior_bounds=prior_bounds_array,
            q_values=q_values_t,
            key_padding_mask=key_padding_mask_t,
        )

        # optional scaled inputs
        scaled_q_values = (
            self.trainer.loader.q_generator.scale_q(q_values_t).to(torch.float32)
            if self.trainer.train_with_q_input
            else None
        )

        sigmas_t = None
        if sigmas is not None:
            sigmas_t = torch.atleast_2d(torch.as_tensor(sigmas, device=self.device, dtype=torch.float32))

        scaled_sigmas = (
            self.trainer.loader.curves_scaler.scale(sigmas_t)
            if (sigmas_t is not None and getattr(self.trainer, "train_with_sigmas", False))
            else None
        )

        conditioning_params_list = []

        # q-resolution handling
        if q_resolution is not None:
            q_resolution_tensor = torch.atleast_2d(torch.as_tensor(q_resolution)).to(scaled_curve)

            if isinstance(q_resolution, float):
                unscaled_q_resolutions = q_resolution_tensor
            else:
                unscaled_q_resolutions = (q_resolution_tensor / q_values_t).nanmean(dim=-1, keepdim=True)

            scaled_q_resolutions = (
                self.trainer.loader.smearing.scale_resolutions(unscaled_q_resolutions)
                if self.trainer.condition_on_q_resolutions
                else None
            )
            if scaled_q_resolutions is not None:
                conditioning_params_list.append(scaled_q_resolutions)
        else:
            q_resolution_tensor = None

        scaled_conditioning_params = torch.cat(conditioning_params_list, dim=-1) if len(conditioning_params_list) > 0 else None

        return PreparedInferenceInputs(
            scaled_curve=scaled_curve,
            q_values=q_values_t,
            scaled_prior_bounds=scaled_prior_bounds,
            restore_context=restore_context,
            scaled_q_values=scaled_q_values,
            scaled_sigmas=scaled_sigmas,
            scaled_conditioning_params=scaled_conditioning_params,
            key_padding_mask=key_padding_mask_t,
            q_resolution_tensor=q_resolution_tensor,
            prior_bounds_array=prior_bounds_array,
            ambient_sld=ambient_sld,
            sld_indices=sld_indices,
            applied_ambient_shift=applied_ambient_shift,  
        )

    def _validate_point_trainer(self) -> None:
        if not isinstance(self.trainer, PointEstimatorTrainer):
            raise RuntimeError(
                "`predict()` is only supported for models trained with "
                "`PointEstimatorTrainer`."
            )

    def _validate_posterior_trainer(self) -> None:
        if not isinstance(self.trainer, NFlowTrainer):
            raise RuntimeError(
                "`sample()` is only supported for models trained with "
                "`NFlowTrainer`."
            )

    def _polish_prediction(
            self,
            q: np.ndarray,
            curve: np.ndarray,
            predicted_params: BasicParams,
            priors: np.ndarray,
            sld_x_axis: Optional[Tensor],
            ambient_sld: Optional[float] = None,
            fit_growth: bool = False,
            max_d_change: float = 5.,
            calc_polished_curve: bool = True,
            calc_polished_sld_profile: bool = False,
            error_bars: Optional[np.ndarray] = None,
            polishing_method: str = 'trf',
            polishing_max_nfev: Optional[int] = None,
            polishing_kwargs_reflectivity: Optional[Dict[str, Any]] = None,
            calc_polishing_param_errors: bool = False,
    ) -> Dict[str, Any]:
        params = predicted_params.parameters.squeeze().cpu().numpy()

        polished_params_dict = {}
        polishing_kwargs_reflectivity = polishing_kwargs_reflectivity or {}

        try:
            if fit_growth:
                polished_params_arr, polished_params_error_array, curve_polished = get_fit_with_growth(
                    q=q,
                    curve=curve,
                    init_params=params,
                    prior_sampler=self.trainer.loader.prior_sampler,
                    bounds=priors.T,
                    error_bars=error_bars,
                    method=polishing_method,
                    polishing_max_nfev=polishing_max_nfev,
                    reflectivity_kwargs=polishing_kwargs_reflectivity,
                    max_d_change=max_d_change,
                    return_param_errors=calc_polishing_param_errors,
                )
                polished_params = BasicParams(
                    torch.from_numpy(polished_params_arr[:-1][None]),
                    torch.from_numpy(priors.T[0][None]),
                    torch.from_numpy(priors.T[1][None]),
                    self.trainer.loader.prior_sampler.max_num_layers,
                    self.trainer.loader.prior_sampler.param_model,
                )
            else:
                polished_params_arr, polished_params_error_array, curve_polished = refl_fit(
                    q = q, 
                    curve = curve, 
                    init_params = params, 
                    bounds=priors.T,
                    prior_sampler=self.trainer.loader.prior_sampler,
                    error_bars=error_bars,
                    method=polishing_method,
                    polishing_max_nfev=polishing_max_nfev,
                    reflectivity_kwargs=polishing_kwargs_reflectivity,
                    return_param_errors=calc_polishing_param_errors,
                )
                polished_params = BasicParams(
                    torch.from_numpy(polished_params_arr[None]),
                    torch.from_numpy(priors.T[0][None]),
                    torch.from_numpy(priors.T[1][None]),
                    self.trainer.loader.prior_sampler.max_num_layers,
                    self.trainer.loader.prior_sampler.param_model
                )
        except Exception as err:
            print("Polishing failed.")
            polished_params = predicted_params
            polished_params_arr = params
            curve_polished = None
        
        if fit_growth:
            polished_params_dict['polished_params_array'] = polished_params_arr[:-1]
            polished_params_dict["polished_delta_d"] = float(polished_params_arr[-1])
        else:
            polished_params_dict['polished_params_array'] = polished_params_arr
        
        if calc_polishing_param_errors:
            polished_params_dict['polished_params_error_array'] = polished_params_error_array

        if calc_polished_curve:
            polished_params_dict['polished_curve'] = curve_polished

        if calc_polished_sld_profile:
            _, profile_dict = self._compute_available_profiles(
                polished_params,
                ambient_sld=ambient_sld,
                z_axis=sld_x_axis.to(polished_params.parameters.device) if sld_x_axis is not None else None,
            )

            for profile_type, profile in profile_dict.items():
                polished_params_dict[f"{profile_type}_profile_polished"] = profile.squeeze().cpu().numpy()

        return polished_params_dict

    
    def _scale_curve(self, curve: Union[np.ndarray, Tensor]) -> Tensor:
        if not isinstance(curve, Tensor):
            curve = torch.from_numpy(curve).float()
        curve = curve.unsqueeze(0).to(self.device)
        scaled_curve = self.trainer.loader.curves_scaler.scale(curve)
        return scaled_curve
    
    def _scale_prior_bounds(self, prior_bounds: Union[np.ndarray, Sequence[Tuple[float, float]]]) -> Tensor:
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
                f"  This often occurs when:\n"
                f"- The model was trained with additional nuisance parameters like `r_scale`, `q_shift`, or `log10_background`,\n"
                f"  but they were not included in the `prior_bounds` passed to `.predict()`.\n"
                f"- The number of layers or parameterization type differs from the one used during training.\n\n"
                f" Check the configuration or the summary of expected parameters."
            )
            raise ValueError(msg) from e   
    
    def _shift_slds_by_ambient(self, prior_bounds: np.ndarray, ambient_sld: float) -> Any:
        sld_indices = self.trainer.loader.prior_sampler.param_model.get_sld_indices()
        prior_bounds[sld_indices, ...] -= ambient_sld

        training_min_bounds = self.trainer.loader.prior_sampler.min_bounds.squeeze().cpu().numpy()
        training_max_bounds = self.trainer.loader.prior_sampler.max_bounds.squeeze().cpu().numpy()
        lower_bound_check = (prior_bounds[sld_indices, 0] >= training_min_bounds[sld_indices]).all()
        upper_bound_check = (prior_bounds[sld_indices, 1] <= training_max_bounds[sld_indices]).all()
        assert lower_bound_check and upper_bound_check, "Shifting the layer SLDs by the ambient SLD exceeded the training ranges."

        return sld_indices
    
    def _restore_slds_after_ambient_shift(
            self,
        prediction_dict: Dict[str, Any],
        sld_indices: Any,
        ambient_sld: float,
    ) -> None:
        prediction_dict["predicted_params_array"][..., sld_indices] += ambient_sld
        if "polished_params_array" in prediction_dict:
            prediction_dict["polished_params_array"][..., sld_indices] += ambient_sld

    def _supports_zero_ambient_sld_shift(self) -> bool:
        return self.trainer.loader.prior_sampler.param_model.supports_zero_ambient_sld_shift()

    def _compute_profile(
        self,
        params_obj: BasicParams,
        *,
        ambient_sld: Optional[float] = None,
        profile_type: str = "sld",
        z_axis: Optional[Tensor] = None,
        num: int = 1024,
        padding_left: float = 0.4,
        padding_right: float = 1.3,
    ):
        param_model = self.trainer.loader.prior_sampler.param_model
        
        params_for_profile = params_obj.parameters.clone()
        if ambient_sld is not None:
            sld_indices = self.trainer.loader.prior_sampler.param_model.get_sld_indices()
            params_for_profile[..., sld_indices] = params_for_profile[..., sld_indices] + ambient_sld

        ambient_sld_tensor = None
        if ambient_sld is not None:
            ambient_sld_tensor = torch.atleast_2d(
                torch.as_tensor(
                    ambient_sld,
                    device=params_for_profile.device,
                    dtype=params_for_profile.dtype,
                )
            )

        z, profile = param_model.profile(
            params_for_profile,
            profile_type=profile_type,
            ambient_sld=ambient_sld_tensor,
            z_axis=z_axis,
            num=num,
            padding_left=padding_left,
            padding_right=padding_right,
        )
        return z, profile
    
    def _compute_available_profiles(
        self,
        params_obj: BasicParams,
        *,
        ambient_sld: Optional[float] = None,
        z_axis: Optional[Tensor] = None,
        num: int = 1024,
        padding_left: float = 0.4,
        padding_right: float = 1.3,
    ):
        param_model = self.trainer.loader.prior_sampler.param_model
        profile_types = list(param_model.available_profile_types())

        shared_z = None
        profiles = {}

        for profile_type in profile_types:
            z, profile = self._compute_profile(
                params_obj,
                ambient_sld=ambient_sld if profile_type == "sld" else None,
                profile_type=profile_type,
                z_axis=z_axis,
                num=num,
                padding_left=padding_left,
                padding_right=padding_right,
            )

            if shared_z is None:
                shared_z = z

            profiles[profile_type] = profile

        return shared_z, profiles

    def _compute_gaussian_log_likelihoods(
        self,
        sampled_curves: Union[np.ndarray, Tensor],
        curve_exp: Union[np.ndarray, Tensor],
        sigmas_exp: Optional[Union[np.ndarray, Tensor]] = None,
        rel_err_factor: float = 0.2,
    ) -> np.ndarray:
        """
        Compute Gaussian log-likelihoods for sampled curves.
        """
        sampled_t = torch.as_tensor(sampled_curves, device=self.device, dtype=torch.float32)
        curve_t = torch.as_tensor(curve_exp, device=self.device, dtype=torch.float32)

        if sigmas_exp is None:
            sigma_t = curve_t * rel_err_factor + 1e-12
        else:
            sigma_t = torch.as_tensor(sigmas_exp, device=self.device, dtype=torch.float32)

        while curve_t.ndim < sampled_t.ndim:
            curve_t = curve_t.unsqueeze(0)
            sigma_t = sigma_t.unsqueeze(0)

        const = torch.log(torch.tensor(2.0 * np.pi, device=self.device, dtype=sampled_t.dtype))
        resid = (sampled_t - curve_t) / sigma_t
        sum_dims = tuple(range(1, sampled_t.ndim))

        ll = -0.5 * torch.sum(
            resid**2 + 2.0 * torch.log(sigma_t) + const,
            dim=sum_dims,
        )

        return ll.detach().cpu().numpy()

    def _batched_reflectivity(
            self,
            params_obj: BasicParams,
            q_values: Tensor,
            q_resolution_tensor: Optional[Tensor],
            max_batch: Optional[int] = None,
    ) -> np.ndarray:
        N = params_obj.parameters.shape[0]

        if max_batch is None or max_batch >= N:
            refl_kwargs = {}
            if q_resolution_tensor is not None:
                refl_kwargs["dq"] = q_resolution_tensor

            curves = params_obj.reflectivity(q=q_values, **refl_kwargs).cpu().numpy()
            return curves

        pieces = []

        for start in range(0, N, max_batch):
            stop = min(start + max_batch, N)
            sl = slice(start, stop)

            q_resolution_tensor_slice = q_resolution_tensor[sl] if q_resolution_tensor is not None else None

            refl_kwargs = {}
            if q_resolution_tensor_slice is not None:
                refl_kwargs["dq"] = q_resolution_tensor_slice

            curves_b = params_obj[sl].reflectivity(q=q_values, **refl_kwargs)
            pieces.append(curves_b.cpu())

        return torch.cat(pieces, dim=0).numpy()
    
    def get_param_labels(self, **kwargs) -> List[str]:
        return self.trainer.loader.prior_sampler.param_model.get_param_labels(**kwargs)
      
    def _preprocess_input_data(
            self,
            reflectivity_curve: np.ndarray,
            q_values: np.ndarray,
            sigmas: Optional[np.ndarray] = None,
            q_resolution: Optional[Union[np.ndarray, float]] = None,
            truncate_index_left: Optional[int] = None,
            truncate_index_right: Optional[int] = None,
            enable_error_bars_filtering: bool = True,
            filter_threshold: float = 0.3,
            filter_remove_singles: bool = True,
            filter_remove_consecutives: bool = True,
            filter_consecutive: int = 3,
            filter_q_start_trunc: float = 0.1,
    ):
    
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
            q_exp: np.ndarray,
            refl_exp: np.ndarray,
            sigmas_exp: Optional[np.ndarray] = None,
            q_res_exp: Optional[Union[np.ndarray, float]] = None,
            as_dict: bool = False
    ) -> Union[Dict[str, Any], Tuple[Any, ...]]:
        """
        Adapt experimental reflectivity data to the q-discretization expected by the loaded model.

        Depending on the q-generator used during training, this method may:
        - interpolate the reflectivity curve and optional arrays onto a fixed model q-grid
        - leave the input grid unchanged
        - or pad/interpolate the inputs to a valid variable-length representation and return a padding mask.

        Args:
            q_exp:
                Experimental q-values of shape ``(n_q,)``.
            refl_exp:
                Experimental reflectivity values of shape ``(n_q,)``.
            sigmas_exp:
                Optional experimental uncertainties of shape ``(n_q,)``.
            q_res_exp:
                Optional q-resolution information, either as a scalar resolution value or an array of shape ``(n_q,)``.
            as_dict:
                If ``True``, return a dictionary with named fields. Otherwise return a tuple whose contents depend on the loaded q-generator.

        Returns:
            If ``as_dict`` is ``True``, returns a dictionary containing:
            - ``"q_model"``: q-values used for the model input
            - ``"reflectivity"``: reflectivity values on the model grid
            - optionally ``"sigmas"``
            - optionally ``"q_resolution"``
            - optionally ``"key_padding_mask"`` for padded variable-length inputs

            If ``as_dict`` is ``False``, returns a tuple containing:
            - ``q_model``
            - ``reflectivity``
            - optionally ``sigmas``
            - optionally ``q_resolution``
            - optionally ``key_padding_mask``
        """
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
                q_model = np.linspace(q_min, q_max, n_q_model)
            else:
                return _prepare_return(q_exp, refl_exp, sigmas_exp, q_res_exp, None, as_dict)

            refl_out = interp_reflectivity(q_model, q_exp, refl_exp)
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

    def _uses_q_params_transform(self) -> bool:
        """
        Whether the loaded model uses the q-parameter transform.
        """
        return getattr(self.trainer.loader, "use_q_params_transform", False)


    def _get_q_fixed_transform(self) -> float:
        """
        Return the reference q_max used during training for the q-parameter transform.
        """
        if not self._uses_q_params_transform():
            raise RuntimeError("q-parameter transform is not enabled for this model.")

        if not hasattr(self.trainer.loader, "q_fixed_transform"):
            raise AttributeError(
                "The loaded trainer/loader does not expose `q_fixed_transform`, "
                "but `use_q_params_transform=True`."
            )
        return self.trainer.loader.q_fixed_transform
    
    def _get_q_transform_ratio(
            self,
            q_values: Union[np.ndarray, Tensor],
            key_padding_mask: Optional[Union[np.ndarray, Tensor]] = None,
    ) -> Tensor:
        """
        Compute the per-example q-ratio

            k = q_fixed_transform / q_max_example

        used during training-time parameter normalization.

        Returns:
            Tensor of shape [B, 1].
        """
        q = torch.atleast_2d(to_t(q_values)).to(self.device)

        if key_padding_mask is not None:
            mask = torch.as_tensor(key_padding_mask, device=q.device, dtype=torch.bool)
            mask = mask.unsqueeze(0) if mask.dim() == 1 else mask
            if mask.shape != q.shape:
                raise ValueError(
                    f"key_padding_mask shape {tuple(mask.shape)} does not match "
                    f"q_values shape {tuple(q.shape)}."
                )
            q_valid = q.masked_fill(~mask, float("-inf"))
            q_max = q_valid.max(dim=-1, keepdim=True).values
        else:
            q_max = q.max(dim=-1, keepdim=True).values

        q_fixed = torch.as_tensor(
            self._get_q_fixed_transform(),
            device=q.device,
            dtype=q.dtype,
        ).view(1, 1)

        return q_fixed / q_max
    
    def _get_q_transform_total_ranges(self) -> Tuple[Tensor, Tensor]:
        """
        Return the transformed total parameter ranges used for scaling bounds when
        `use_q_params_transform=True`.

        This mirrors the training logic in BasicDataset.get_batch().
        """
        if not self._uses_q_params_transform():
            raise RuntimeError("q-parameter transform is not enabled for this model.")

        q_generator = self.trainer.loader.q_generator
        if not hasattr(q_generator, "q_max_range"):
            raise AttributeError(
                "q-parameter transform requires the q_generator to expose `q_max_range`."
            )

        q_max_range = q_generator.q_max_range
        q_fixed = self._get_q_fixed_transform()

        q_ratio_min = q_fixed / float(q_max_range[1])
        q_ratio_max = q_fixed / float(q_max_range[0])

        prior_sampler = self.trainer.loader.prior_sampler

        min_range_k, max_range_k = prior_sampler.param_model.scale_total_ranges_with_q(
            min_total_ranges=prior_sampler.min_bounds.clone(),
            max_total_ranges=prior_sampler.max_bounds.clone(),
            q_ratio_min=q_ratio_min,
            q_ratio_max=q_ratio_max,
        )

        return min_range_k.to(self.device), max_range_k.to(self.device)
    
    def _transform_prior_bounds_with_q(
            self,
            prior_bounds: Union[np.ndarray, Tensor],
            q_ratio: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Transform physical prior bounds into the fixed-q parameter frame.

        Args:
            prior_bounds: array-like of shape [dim, 2]
            q_ratio: tensor of shape [B, 1]

        Returns:
            min_bounds_k: tensor of shape [B, dim]
            max_bounds_k: tensor of shape [B, dim]
        """
        pb = torch.as_tensor(prior_bounds, device=self.device, dtype=torch.float32)
        if pb.ndim != 2 or pb.shape[1] != 2:
            raise ValueError(
                f"prior_bounds must have shape [dim, 2], got {tuple(pb.shape)}"
            )

        min_bounds = pb[:, 0].unsqueeze(0).expand(q_ratio.shape[0], -1)
        max_bounds = pb[:, 1].unsqueeze(0).expand(q_ratio.shape[0], -1)

        param_model = self.trainer.loader.prior_sampler.param_model
        min_bounds_k = param_model.scale_with_q(min_bounds, q_ratio)
        max_bounds_k = param_model.scale_with_q(max_bounds, q_ratio)

        return min_bounds_k, max_bounds_k
    
    def _prepare_scaled_prior_bounds_for_network(
            self,
            prior_bounds: Union[np.ndarray, Tensor],
            q_values: Union[np.ndarray, Tensor],
            key_padding_mask: Optional[Union[np.ndarray, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """
        Prepare the scaled bounds tensor to feed into the network.

        For ordinary models this is just the standard scaling path.
        For q-transform-trained models this mirrors the training-time transformation.

        Returns:
            scaled_prior_bounds: tensor of shape [B, 2*dim]
            restore_context: None for ordinary models, otherwise a dict containing
                the information needed to restore predicted parameters to physical
                parameter space.
        """
        if not self._uses_q_params_transform():
            return self._scale_prior_bounds(prior_bounds), None

        q_ratio = self._get_q_transform_ratio(q_values, key_padding_mask=key_padding_mask)
        min_bounds_k, max_bounds_k = self._transform_prior_bounds_with_q(prior_bounds, q_ratio)

        min_range_k, max_range_k = self._get_q_transform_total_ranges()
        min_range_k = min_range_k.expand_as(min_bounds_k)
        max_range_k = max_range_k.expand_as(max_bounds_k)

        prior_sampler = self.trainer.loader.prior_sampler
        scaled_min_bounds = prior_sampler._scale(min_bounds_k, min_range_k, max_range_k)
        scaled_max_bounds = prior_sampler._scale(max_bounds_k, min_range_k, max_range_k)

        scaled_prior_bounds = torch.cat([scaled_min_bounds, scaled_max_bounds], dim=-1).float()

        restore_context = {
            "q_ratio": q_ratio,
            "min_range_k": min_range_k,
            "max_range_k": max_range_k,
        }
        return scaled_prior_bounds, restore_context

    def _expand_restore_context(
            self,
            restore_context: Optional[Dict[str, Tensor]],
            batch_size: int,
    ) -> Optional[Dict[str, Tensor]]:
        """
        Expand restore_context tensors along batch dimension if needed.
        """
        if restore_context is None:
            return None

        out = dict(restore_context)
        for key in ("q_ratio", "min_range_k", "max_range_k"):
            value = out[key]
            if value.shape[0] == batch_size:
                continue
            if value.shape[0] != 1:
                raise ValueError(
                    f"Cannot expand restore_context['{key}'] from batch "
                    f"{value.shape[0]} to {batch_size}."
                )
            out[key] = value.expand(batch_size, -1)
        return out
        
    def _restore_predicted_params_for_inference(
            self,
            scaled_predicted_params: Tensor,
            scaled_prior_bounds: Tensor,
            restore_context: Optional[Dict[str, Tensor]] = None,
    ) -> BasicParams:
        """
        Restore network outputs to physical parameter space.

        For ordinary models:
            scaled -> physical directly via restore_params

        For q-transform-trained models:
            scaled -> transformed-space params via restore_params_custom_range
            transformed-space params -> physical params via inverse q-transform
        """
        batch_size = scaled_predicted_params.shape[0]

        if scaled_prior_bounds.shape[0] != batch_size:
            if scaled_prior_bounds.shape[0] == 1:
                scaled_prior_bounds = scaled_prior_bounds.expand(batch_size, -1)
            else:
                raise ValueError(
                    f"scaled_prior_bounds batch {scaled_prior_bounds.shape[0]} "
                    f"does not match predicted batch {batch_size}."
                )

        prior_sampler = self.trainer.loader.prior_sampler
        scaled_full = torch.cat([scaled_predicted_params, scaled_prior_bounds], dim=-1)

        if restore_context is None:
            return prior_sampler.restore_params(scaled_full)

        restore_context = self._expand_restore_context(restore_context, batch_size)

        predicted_params_k = prior_sampler.restore_params_custom_range(
            scaled_full,
            restore_context["min_range_k"],
            restore_context["max_range_k"],
        )

        q_ratio_inv = 1.0 / restore_context["q_ratio"]
        predicted_params = predicted_params_k.scale_with_q(q_ratio=q_ratio_inv)
        return predicted_params

    def plot_training_history(
            self,
            root_dir: Optional[Union[str, Path]] = None,
            model_name: Optional[str] = None,
            metrics: Union[str, Tuple[str, ...]] = ("losses",),
            figsize: Tuple[float, float] = (6, 4),
            logy_loss: bool = False,
            map_location: str = "cpu",
            return_data: bool = False,
    ) -> Any:
        import matplotlib.pyplot as plt

        if self.weights_format != "pt":
            raise ValueError(
                "plot_training_history() is only available for '.pt' checkpoints."
            )

        if self.config_name is None and model_name is None and self.model_name is None:
            raise ValueError(
                "Could not infer checkpoint name. Provide `model_name` or initialize "
                "InferenceModel with a config/model name."
            )

        if isinstance(metrics, str):
            metrics = (metrics,)
        else:
            metrics = tuple(metrics)

        allowed = {"losses", "lrs"}
        if not metrics or any(m not in allowed for m in metrics):
            raise ValueError(
                "Supported metrics are only 'losses', 'lrs', or both together."
            )

        metrics = tuple(dict.fromkeys(metrics))

        ckpt_name = model_name or self.model_name
        if ckpt_name is None:
            config_name_no_ext = (
                self.config_name[:-5] if self.config_name.endswith(".yaml") else self.config_name
            )
            ckpt_name = f"model_{config_name_no_ext}.pt"
        elif not ckpt_name.endswith(".pt"):
            ckpt_name = ckpt_name + ".pt"

        model_dir = Path(root_dir) / "saved_models" if root_dir else (
            Path(self.root_dir) / "saved_models" if self.root_dir else SAVED_MODELS_DIR
        )
        load_path = model_dir / ckpt_name

        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

        d = torch.load(load_path, map_location=map_location, weights_only=False)

        series = []
        for metric in metrics:
            if metric == "losses":
                if "losses" not in d:
                    raise ValueError(f"'losses' not found in checkpoint. Available keys: {list(d.keys())}")
                values = d["losses"]["loss"] if isinstance(d["losses"], dict) and "loss" in d["losses"] else d["losses"]
                ylabel = "loss"
            else: 
                if "lrs" not in d:
                    raise ValueError(f"'lrs' not found in checkpoint. Available keys: {list(d.keys())}")
                values = d["lrs"]
                ylabel = "learning rate"

            if torch.is_tensor(values):
                values = values.detach().cpu().numpy()
            else:
                values = np.asarray(values)

            if values.ndim != 1:
                raise ValueError(
                    f"Metric '{metric}' must be a 1D numeric sequence, got shape {values.shape}."
                )

            series.append((metric, values, ylabel))

        fig, axes = plt.subplots(
            nrows=len(series),
            ncols=1,
            figsize=(figsize[0], max(figsize[1], 3 * len(series))),
        )
        if len(series) == 1:
            axes = [axes]

        for ax, (metric, values, ylabel) in zip(axes, series):
            ax.plot(values)
            ax.set_title(metric)
            ax.set_xlabel("step")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

            if metric == "losses" and logy_loss:
                ax.set_yscale("log")

        if "best_loss" in d:
            try:
                fig.suptitle(f"Training history (best_loss={float(d['best_loss']):.6g})")
            except Exception:
                pass

        fig.tight_layout()

        if return_data:
            return fig, axes, d
        return fig, axes



EasyInferenceModel = InferenceModel