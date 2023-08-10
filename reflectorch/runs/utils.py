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
]


def init_from_conf(conf, **kwargs):
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
    folder_paths = get_paths_from_config(config, mkdir=True)

    trainer = get_trainer_from_config(config, folder_paths)

    callbacks = get_callbacks_from_config(config, folder_paths)

    trainer.train_epoch(
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
    dset = init_dset(config['dset'])

    folder_paths = folder_paths or get_paths_from_config(config)

    model = init_encoder(config['model']['encoder'], folder_paths['saved_models'])

    train_conf = config['training']

    optim_cls = getattr(torch.optim, train_conf['optimizer'])

    logger = None

    trainer_kwargs = train_conf.get('init_kwargs', {})
    
    if 'train_with_q_input' in train_conf:
        train_with_q_input = train_conf['train_with_q_input']
    else:
        train_with_q_input = False

    trainer = PointEstimatorTrainer(
        model, dset, train_conf['lr'], train_conf['batch_size'],
        logger=logger, optim_cls=optim_cls, train_with_q_input=train_with_q_input, 
        **trainer_kwargs
    )

    return trainer


def get_trainer_by_name(model_name, model_path=None, load_weights: bool = True):
    config = load_config(model_name)
    config['model']['encoder']['pretrained_name'] = None
    config['training']['logger']['use_neptune'] = False
    trainer = get_trainer_from_config(config)

    num_params = sum(p.numel() for p in trainer.model.parameters())

    print(
        f'Model {model_name} loaded. Number of parameters: {num_params / 10 ** 6:.2f} M',
    )

    if not load_weights:
        return trainer

    if not model_path:
        model_name = f'model_{model_name}.pt'
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


def init_encoder(config: dict, saved_models_dir: Path):
    encoder = init_from_conf(config).cuda()
    encoder = load_pretrained(encoder, config.get('pretrained_name', None), saved_models_dir)
    return encoder


def load_pretrained(model, model_name: str, saved_models_dir: Path):
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
    dset_cls = globals().get(config['cls']) if 'cls' in config else XrrDataLoader
    prior_sampler = init_from_conf(config['prior_sampler'])
    intensity_noise = init_from_conf(config['intensity_noise'])
    q_generator = init_from_conf(config['q_generator'], device='cuda')
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
