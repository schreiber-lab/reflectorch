from pathlib import Path
from typing import Tuple

import torch

from reflectorch import *
from reflectorch.runs.config import load_config

__all__ = [
    "train_from_config",
    "get_trainer_from_config",
    "get_paths_from_config",
    "get_callbacks_from_config",
    "get_trainer_by_name",
    "get_callbacks_by_name",
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
    """Get the folder paths from a configuration dictionary

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

    if train_conf['logger']['use_neptune']:
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

    logger = None

    train_with_q_input = train_conf.get('train_with_q_input', False)
    clip_grad_norm_max = train_conf.get('clip_grad_norm_max', None)

    trainer_cls = globals().get(train_conf['trainer_cls']) if 'trainer_cls' in train_conf else PointEstimatorTrainer
    trainer_kwargs = train_conf.get('trainer_kwargs', {})


    trainer = trainer_cls(
        model, dset, train_conf['lr'], train_conf['batch_size'], clip_grad_norm_max=clip_grad_norm_max,
        logger=logger, optim_cls=optim_cls, train_with_q_input=train_with_q_input, 
        **trainer_kwargs
    )

    return trainer


def get_trainer_by_name(config_name, config_dir=None, model_path=None, load_weights: bool = True, inference_device = None):
    """Initializes a trainer object based on a configuration file (i.e. the model name) and optionally loads \
        saved weights into the network

    Args:
        config_name (str): name of the configuration file
        config_dir (str): path of the configuration directory
        model_path (str, optional): path to the network weights. The default path is 'saved_models' located in the package directory
        load_weights (bool, optional): if True the saved network weights are loaded into the network. Defaults to True.


    Returns:
        Trainer: the trainer object
    """
    config = load_config(config_name, config_dir)
    config['model']['network']['pretrained_name'] = None
    config['training']['logger']['use_neptune'] = False

    if inference_device:
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

    try:
        state_dict = torch.load(model_path)
    except Exception as err:
        raise RuntimeError(f'Could not load model from {model_path}') from err

    if 'model' in state_dict:
        trainer.model.load_state_dict(state_dict['model'])
    else:
        trainer.model.load_state_dict(state_dict)

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
