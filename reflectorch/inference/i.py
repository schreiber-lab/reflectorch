# class EasyInferenceModel(object):
#     """Facilitates the inference process using pretrained models
    
#     Args:
#         config_name (str, optional): the name of the configuration file used to initialize the model. Defaults to None.
#         config_dir (str, optional): path to the directory containing the configuration file. Defaults to None.
#         model_path (str, optional): path to the saved weights of the neural network. Defaults to None.
#         trainer (PointEstimatorTrainer, optional): if provided, this trainer instance is used instead of initializing from the configuration file . Defaults to None.
#         preprocessing_parameters (dict, optional): dictionary of parameters for preprocessing raw data. Defaults to None.
#         device (str, optional): the Pytorch device ('cuda' or 'cpu'). Defaults to 'cuda'.
#     """
#     def __init__(self, config_name: str = None, config_dir: str = None, model_path: str = None, trainer: PointEstimatorTrainer = None, preprocessing_parameters: dict = None, device='cuda', repo_id: str = None):
#         self.config_name = config_name
#         self.config_dir = Path(config_dir) if config_dir else CONFIG_DIR
#         self.model_path = Path(model_path) if model_path else (SAVED_MODELS_DIR / f"model_{self._normalize_config_name(config_name)}.pt")
#         self.trainer = trainer
#         self.device = device
#         self.repo_id = repo_id
#         self.preprocessing = StandardPreprocessing(**(preprocessing_parameters or {}))

#         if trainer is None and self.config_name is not None:
#             self.load_model(self.config_name, self.config_dir, self.model_path)

#     def _normalize_config_name(self, config_name: str) -> str:
#         if not config_name.endswith('.yaml'):
#             config_name += '.yaml'
#         return config_name

#     def get_file_path(self, file_name: str, local_path: Path, repo_id: str) -> Path:
#         file_path = local_path / file_name
#         print(f"Checking for {file_path} locally.")
#         if not file_path.exists():
#             print(f"{file_path} does not exist.")
#             if repo_id is None:
#                 raise ValueError("repo_id must be provided to download files from Hugging Face Hub.")
#             print(f"{file_name} not found locally. Downloading from Hugging Face Hub...")
#             file_path = Path(hf_hub_download(repo_id=repo_id, filename=file_name, cache_dir=str(local_path)))
#         else:
#             print(f"{file_path} exists locally.")
#         return file_path
    
#     def load_model(self, config_name: str, config_dir: Path, model_path: Path) -> None:
#         if self.config_name == config_name and self.trainer is not None:
#             return

#         config_name_with_ext = self._normalize_config_name(config_name)
#         model_file_name = f"model_{config_name_with_ext}.pt"
        
#         print(f"Loading config: {config_name_with_ext} from {config_dir}")
#         config_file_path = self.get_file_path(config_name_with_ext, config_dir, self.repo_id)
#         print(f"Loading model: {model_file_name} from {model_path.parent}")
#         model_file_path = self.get_file_path(model_file_name, model_path.parent, self.repo_id)
        
#         self.config_name = config_name_with_ext
#         self.config_dir = config_dir
#         self.model_path = model_file_path
        
#         self.trainer = get_trainer_by_name(config_name=config_name_with_ext, config_dir=str(config_dir), model_path=str(model_file_path), load_weights=True, inference_device=self.device)
#         self.trainer.model.eval()
        
#         print(f'The model corresponds to a parameterization with {self.trainer.loader.prior_sampler.max_num_layers} layers ({self.trainer.loader.prior_sampler.param_dim} predicted parameters)')
#         print(f'Parameter types and total ranges: {self.trainer.loader.prior_sampler.param_ranges}')
#         print(f'Allowed widths of the prior bound intervals (max-min): {self.trainer.loader.prior_sampler.bound_width_ranges}')

#         if isinstance(self.trainer.loader.q_generator, ConstantQ):
#             q_min = self.trainer.loader.q_generator.q[0].item()
#             q_max = self.trainer.loader.q_generator.q[-1].item()
#             n_q = self.trainer.loader.q_generator.q.shape[0]
#             print(f'The model was trained on curves discretized at {n_q} uniform points between between q_min={q_min} and q_max={q_max}')
#         elif isinstance(self.trainer.loader.q_generator, VariableQ):
#             q_min_range = self.trainer.loader.q_generator.q_min_range
#             q_max_range = self.trainer.loader.q_generator.q_max_range
#             n_q_range = self.trainer.loader.q_generator.n_q_range
#             print(f'The model was trained on curves discretized at a number between {n_q_range[0]} and {n_q_range[1]} of uniform points between between q_min={q_min} in [{q_min_range[0]}, {q_min_range[1]}] and q_max in [{q_max_range[0]}, {q_max_range[1]}]')
