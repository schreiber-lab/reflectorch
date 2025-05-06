from pathlib import Path
from typing import Tuple

import torch
import safetensors.torch
import os

from reflectorch import *
from reflectorch.runs.config import load_config

__all__ = [
    "train_from_config",
    "get_trainer_from_config",
    "get_paths_from_config",
    "get_callbacks_from_config",
    "get_trainer_by_name",
    "get_callbacks_by_name",
    "convert_pt_to_safetensors",
]


def init_from_conf(conf, **kwargs):
    """Initializes an object with class type, args and kwargs specified in the configuration

    Args:
        conf (dict): configuration dictionary

    Returns:
        Any: the initialized object
    """
    if not conf:
        return
    cls_name = conf['cls']
    if not cls_name:
        return
    cls = globals().get(cls_name)

    if not cls:
        raise ValueError(f'Unknown class {cls_name}')

    conf_args = conf.get('args', [])
    conf_kwargs = conf.get('kwargs', {})
    conf_kwargs.update(kwargs)
    return cls(*conf_args, **conf_kwargs)


def train_from_config(config: dict):
    """Train a model from a configuration dictionary

    Args:
        config (dict): configuration dictionary

    Returns:
        Trainer: the trainer object
    """

    folder_paths = get_paths_from_config(config, mkdir=True)

    trainer = get_trainer_from_config(config, folder_paths)

    callbacks = get_callbacks_from_config(config, folder_paths)

    trainer.train(
        config['training']['num_iterations'],
        callbacks, disable_tqdm=False,
        update_tqdm_freq=config['training']['update_tqdm_freq'],
        grad_accumulation_steps=config['training'].get('grad_accumulation_steps', 1)
    )

    torch.save({
        'paths': folder_paths,
        'losses': trainer.losses,
        'params': config,
    }, folder_paths['losses'])

    return trainer


def get_paths_from_config(config: dict, mkdir: bool = False):
    """Get the directory paths from a configuration dictionary

    Args:
        config (dict): configuration dictionary
        mkdir (bool, optional): option to create a new directory for the saved model weights and losses.

    Returns:
        dict: dictionary containing the folder paths
    """
    root_dir = Path(config['general']['root_dir'] or ROOT_DIR)
    name = config['general']['name']

    assert root_dir.is_dir()

    saved_models_dir = root_dir / 'saved_models'
    saved_losses_dir = root_dir / 'saved_losses'

    if mkdir:
        saved_models_dir.mkdir(exist_ok=True)
        saved_losses_dir.mkdir(exist_ok=True)

    model_path = str((saved_models_dir / f'model_{name}.pt').absolute())

    losses_path = saved_losses_dir / f'{name}_losses.pt'

    return {
        'name': name,
        'model': model_path,
        'losses': losses_path,
        'root': root_dir,
        'saved_models': saved_models_dir,
        'saved_losses': saved_losses_dir,
    }


def get_callbacks_from_config(config: dict, folder_paths: dict = None) -> Tuple['TrainerCallback', ...]:
    """Initializes the training callbacks from a configuration dictionary
    
    Returns:
        tuple: tuple of callbacks
    """
    callbacks = []

    folder_paths = folder_paths or get_paths_from_config(config)

    train_conf = config['training']
    callback_conf = dict(train_conf['callbacks'])
    save_conf = callback_conf.pop('save_best_model')

    if save_conf['enable']:
        save_model = SaveBestModel(folder_paths['model'], freq=save_conf['freq'])
        callbacks.append(save_model)

    for conf in callback_conf.values():
        callback = init_from_conf(conf)

        if callback:
            callbacks.append(callback)

    if 'logger' in train_conf.keys():
        callbacks.append(LogLosses())

    return tuple(callbacks)


def get_trainer_from_config(config: dict, folder_paths: dict = None):
    """Initializes a trainer from a configuration dictionary

    Args:
        config (dict): the configuration dictionary
        folder_paths (dict, optional): dictionary containing the folder paths

    Returns:
        Trainer: the trainer object
    """
    dset = init_dset(config['dset'])

    folder_paths = folder_paths or get_paths_from_config(config)

    model = init_network(config['model']['network'], folder_paths['saved_models'])

    train_conf = config['training']

    optim_cls = getattr(torch.optim, train_conf['optimizer'])

    logger_conf = train_conf.get('logger', None)
    logger = init_from_conf(logger_conf) if logger_conf and logger_conf.get('cls') else None 

    clip_grad_norm_max = train_conf.get('clip_grad_norm_max', None)

    trainer_cls = globals().get(train_conf['trainer_cls']) if 'trainer_cls' in train_conf else PointEstimatorTrainer
    trainer_kwargs = train_conf.get('trainer_kwargs', {})


    trainer = trainer_cls(
        model, dset, train_conf['lr'], train_conf['batch_size'], clip_grad_norm_max=clip_grad_norm_max,
        logger=logger, optim_cls=optim_cls, 
        **trainer_kwargs
    )

    if train_conf.get('train_with_q_input', False) and getattr(trainer, 'train_with_q_input', None) is not None: #only for back-compatibility with configs in older versions
        trainer.train_with_q_input = True

    return trainer


def get_trainer_by_name(config_name, config_dir=None, model_path=None, load_weights: bool = True, inference_device: str = 'cuda'):
    """Initializes a trainer object based on a configuration file (i.e. the model name) and optionally loads \
        saved weights into the network

    Args:
        config_name (str): name of the configuration file
        config_dir (str): path of the configuration directory
        model_path (str, optional): path to the network weights. The default path is 'saved_models' located in the package directory
        load_weights (bool, optional): if True the saved network weights are loaded into the network. Defaults to True.
        inference_device (str, optional): overwrites the device in the configuration file for the purpose of inference on a different device then the training was performed on. Defaults to 'cuda'.

    Returns:
        Trainer: the trainer object
    """
    config = load_config(config_name, config_dir)
    #config['model']['network']['pretrained_name'] = None

    config['model']['network']['device'] = inference_device
    config['dset']['prior_sampler']['kwargs']['device'] = inference_device
    config['dset']['q_generator']['kwargs']['device'] = inference_device

    trainer = get_trainer_from_config(config)

    num_params = sum(p.numel() for p in trainer.model.parameters())

    print(
        f'Model {config_name} loaded. Number of parameters: {num_params / 10 ** 6:.2f} M',
    )

    if not load_weights:
        return trainer

    if not model_path:
        model_name = f'model_{config_name}.pt'
        model_path = SAVED_MODELS_DIR / model_name

    if str(model_path).endswith('.pt'):
        try:
            state_dict = torch.load(model_path, map_location=inference_device, weights_only=False)
        except Exception as err:
            raise RuntimeError(f'Could not load model from {model_path}') from err

        if 'model' in state_dict:
            trainer.model.load_state_dict(state_dict['model'])
        else:
            trainer.model.load_state_dict(state_dict)

    elif str(model_path).endswith('.safetensors'):
        try:
            load_state_dict_safetensors(model=trainer.model, filename=model_path, device=inference_device)
        except Exception as err:
            raise RuntimeError(f'Could not load model from {model_path}') from err
        
    else:
        raise RuntimeError('Weigths file with unknown extension')

    return trainer
 
def get_callbacks_by_name(config_name, config_dir=None):
    """Initializes the trainer callbacks based on a configuration file

    Args:
        config_name (str): name of the configuration file
        config_dir (str): path of the configuration directory
    """
    config = load_config(config_name, config_dir)
    callbacks = get_callbacks_from_config(config)

    return callbacks


def init_network(config: dict, saved_models_dir: Path):
    """Initializes the network based on the configuration dictionary and optionally loades the weights from a pretrained model"""
    device = config.get('device', 'cuda')
    network = init_from_conf(config).to(device)
    network = load_pretrained(network, config.get('pretrained_name', None), saved_models_dir)
    return network


def load_pretrained(model, model_name: str, saved_models_dir: Path):
    """Loads the saved weights into the network"""
    if not model_name:
        return model
    
    if '.' not in model_name:
        model_name = model_name + '.pt'
    model_path = saved_models_dir / model_name

    if not model_path.is_file():
        model_path = saved_models_dir / f'model_{model_name}'

    if not model_path.is_file():
        raise FileNotFoundError(f'File {str(model_path)} does not exist.')
    
    try:
        pretrained = torch.load(model_path)
    except Exception as err:
        raise RuntimeError(f'Could not load model from {str(model_path)}') from err
    
    if 'model' in pretrained:
        pretrained = pretrained['model']
    try:
        model.load_state_dict(pretrained)
    except Exception as err:
        raise RuntimeError(f'Could not update state dict from {str(model_path)}') from err

    return model


def init_dset(config: dict):
    """Initializes the dataset / dataloader object"""
    dset_cls = globals().get(config['cls']) if 'cls' in config else ReflectivityDataLoader
    prior_sampler = init_from_conf(config['prior_sampler'])
    intensity_noise = init_from_conf(config['intensity_noise'])
    q_generator = init_from_conf(config['q_generator'])
    curves_scaler = init_from_conf(config['curves_scaler']) if 'curves_scaler' in config else None
    smearing = init_from_conf(config['smearing']) if 'smearing' in config else None
    q_noise = init_from_conf(config['q_noise']) if 'q_noise' in config else None

    dset = dset_cls(
        q_generator=q_generator,
        prior_sampler=prior_sampler,
        intensity_noise=intensity_noise,
        curves_scaler=curves_scaler,
        smearing=smearing,
        q_noise=q_noise,
    )

    return dset

def split_complex_tensors(state_dict):
    new_state_dict = {}
    for key, tensor in state_dict.items():
        if tensor.is_complex():
            new_state_dict[f"{key}_real"] = tensor.real.clone()
            new_state_dict[f"{key}_imag"] = tensor.imag.clone()
        else:
            new_state_dict[key] = tensor
    return new_state_dict

def recombine_complex_tensors(state_dict):
    new_state_dict = {}
    keys = list(state_dict.keys())
    visited = set()

    for key in keys:
        if key.endswith('_real') or key.endswith('_imag'):
            base_key = key[:-5]
            new_state_dict[base_key] = torch.complex(state_dict[base_key + '_real'], state_dict[base_key + '_imag'])
            visited.add(base_key + '_real')
            visited.add(base_key + '_imag')
        elif key not in visited:
            new_state_dict[key] = state_dict[key]

    return new_state_dict

def convert_pt_to_safetensors(input_dir):
    """Creates '.safetensors' files for all the model state dictionaries inside '.pt' files in the specified directory.

    Args:
        input_dir (str): directory containing model weights 
    """
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory {input_dir} does not exist")

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.pt'):
            pt_file_path = os.path.join(input_dir, file_name)
            safetensors_file_path = os.path.join(input_dir, file_name[:-3] + '.safetensors')

            if os.path.exists(safetensors_file_path):
                print(f"Skipping {pt_file_path}, corresponding .safetensors file already exists.")
                continue

            print(f"Converting {pt_file_path} to .safetensors format.")
            data_pt = torch.load(pt_file_path)
            model_state_dict = data_pt["model"]
            model_state_dict = split_complex_tensors(model_state_dict) #handle tensors with complex dtype which are not natively supported by safetensors

            safetensors.torch.save_file(tensors=model_state_dict, filename=safetensors_file_path)

def convert_files_to_safetensors(files):
    """
    Converts specified .pt files to .safetensors format.

    Args:
        files (str or list of str): Path(s) to .pt files containing model state dictionaries.
    """
    if isinstance(files, str):
        files = [files]

    for pt_file_path in files:
        if not pt_file_path.endswith('.pt'):
            print(f"Skipping {pt_file_path}: not a .pt file.")
            continue

        if not os.path.exists(pt_file_path):
            print(f"File {pt_file_path} does not exist.")
            continue

        safetensors_file_path = pt_file_path[:-3] + '.safetensors'

        if os.path.exists(safetensors_file_path):
            print(f"Skipping {pt_file_path}: .safetensors version already exists.")
            continue

        print(f"Converting {pt_file_path} to .safetensors format.")
        data_pt = torch.load(pt_file_path, weights_only=False)
        model_state_dict = data_pt["model"]
        model_state_dict = split_complex_tensors(model_state_dict)

        safetensors.torch.save_file(tensors=model_state_dict, filename=safetensors_file_path)

def load_state_dict_safetensors(model, filename, device):
    state_dict = safetensors.torch.load_file(filename=filename, device=device)
    state_dict = recombine_complex_tensors(state_dict)
    model.load_state_dict(state_dict)