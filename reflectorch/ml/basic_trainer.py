from typing import Optional, Tuple, Iterable, Any, Union, Type
from collections import defaultdict

from tqdm import tqdm as standard_tqdm
from tqdm.notebook import tqdm as notebook_tqdm
import numpy as np

import torch
from torch.nn import Module

from reflectorch.ml.loggers import Logger, Loggers

from .utils import is_divisor

__all__ = [
    'Trainer',
    'TrainerCallback',
    'DataLoader',
    'PeriodicTrainerCallback',
]


class Trainer(object):
    """Trainer class
    
    Args:
        model (nn.Module): neural network
        loader (DataLoader): data loader
        lr (float): learning rate
        batch_size (int): batch size
        clip_grad_norm (int, optional): maximum norm for gradient clipping if it is not ``None``. Defaults to None.
        logger (Union[Logger, Tuple[Logger, ...], Loggers], optional): logger. Defaults to None.
        optim_cls (Type[torch.optim.Optimizer], optional): Pytorch optimizer. Defaults to torch.optim.Adam.
        optim_kwargs (dict, optional): optimizer arguments. Defaults to None.
    """

    TOTAL_LOSS_KEY: str = 'total_loss'

    def __init__(self,
                 model: Module,
                 loader: 'DataLoader',
                 lr: float,
                 batch_size: int,
                 clip_grad_norm_max: Optional[int] = None,
                 logger: Union[Logger, Tuple[Logger, ...], Loggers] = None,
                 optim_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
                 optim_kwargs: dict = None,
                 **kwargs
                 ):
        
        self.model = model
        self.loader = loader
        self.batch_size = batch_size
        self.clip_grad_norm_max = clip_grad_norm_max

        self.optim = self.configure_optimizer(optim_cls, lr=lr, **(optim_kwargs or {}))
        self.lrs = []
        self.losses = defaultdict(list)

        self.logger = _init_logger(logger)
        self.callback_params = {}
        
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.init()

    def init(self):
        pass

    def log(self, name: str, data):
        """log data"""
        self.logger.log(name, data)

    def train(self,
                    num_batches: int,
                    callbacks: Union[Tuple['TrainerCallback', ...], 'TrainerCallback'] = (),
                    disable_tqdm: bool = False,
                    use_notebook_tqdm: bool = False,
                    update_tqdm_freq: int = 1,
                    grad_accumulation_steps: int = 1,
                    ):
        """starts the training process

        Args:
            num_batches (int): total number of training iterations
            callbacks (Union[Tuple['TrainerCallback'], 'TrainerCallback']): the trainer callbacks. Defaults to ().
            disable_tqdm (bool, optional): if ``True``, the progress bar is disabled. Defaults to False.
            use_notebook_tqdm (bool, optional): should be set to ``True`` when used in a Jupyter Notebook. Defaults to False.
            update_tqdm_freq (int, optional): frequency for updating the progress bar. Defaults to 10.
            grad_accumulation_steps (int, optional): number of gradient accumulation steps. Defaults to 1.
        """

        if isinstance(callbacks, TrainerCallback):
            callbacks = (callbacks,)

        callbacks = _StackedTrainerCallbacks(list(callbacks) + [self.loader])

        tqdm_class = notebook_tqdm if use_notebook_tqdm else standard_tqdm
        pbar = tqdm_class(range(num_batches), disable=disable_tqdm)

        callbacks.start_training(self)

        for batch_num in pbar:
            self.model.train()

            self.optim.zero_grad()
            total_loss, avr_loss_dict = 0, defaultdict(list)

            for _ in range(grad_accumulation_steps):

                batch_data = self.get_batch_by_idx(batch_num)
                loss_dict = self.get_loss_dict(batch_data)
                loss = loss_dict['loss'] / grad_accumulation_steps
                total_loss += loss.item()
                _update_loss_dict(avr_loss_dict, loss_dict)

                if not torch.isfinite(loss).item():
                    raise ValueError('Loss is not finite!')

                loss.backward()

            if self.clip_grad_norm_max is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm_max)

            self.optim.step()

            avr_loss_dict = {k: np.mean(v) for k, v in avr_loss_dict.items()}
            self._update_losses(avr_loss_dict, total_loss)

            if not disable_tqdm:
                self._update_tqdm(pbar, batch_num, update_tqdm_freq)

            break_epoch = callbacks.end_batch(self, batch_num)

            if break_epoch:
                break

        callbacks.end_training(self)

    def _update_tqdm(self, pbar, batch_num: int, update_tqdm_freq: int):
        if is_divisor(batch_num, update_tqdm_freq):
            last_loss = np.mean(self.losses[self.TOTAL_LOSS_KEY][-10:])
            pbar.set_description(f'Loss = {last_loss:.2e}')

            postfix = {}
            for key in self.losses.keys():
                if key != self.TOTAL_LOSS_KEY:
                    last_value = self.losses[key][-1]
                    postfix[key] = f'{last_value:.4f}'

            postfix['lr'] = f'{self.lr():.2e}'

            pbar.set_postfix(postfix)

    def get_batch_by_idx(self, batch_num: int) -> Any:
        raise NotImplementedError

    def get_loss_dict(self, batch_data) -> dict:
        raise NotImplementedError

    def _update_losses(self, loss_dict: dict, loss: float) -> None:
        _update_loss_dict(self.losses, loss_dict)
        self.losses[self.TOTAL_LOSS_KEY].append(loss)
        self.lrs.append(self.lr())

    def configure_optimizer(self, optim_cls, lr: float, **kwargs) -> torch.optim.Optimizer:
        """configure the optimizer based on the optimizer class, the learning rate and the optimizer keyword arguments

        Args:
            optim_cls: the class of the optimizer
            lr (float): the learning rate

        Returns:
            torch.optim.Optimizer:
        """
        optim = optim_cls(self.model.parameters(), lr,  **kwargs)
        return optim

    def lr(self, param_group: int = 0) -> float:
        """get the learning rate"""
        return self.optim.param_groups[param_group]['lr']

    def set_lr(self, lr: float, param_group: int = 0) -> None:
        """set the learning rate"""
        self.optim.param_groups[param_group]['lr'] = lr


class TrainerCallback(object):
    """Base class for trainer callbacks
    """
    def start_training(self, trainer: Trainer) -> None:
        """add functionality the start of training

        Args:
            trainer (Trainer): the trainer object
        """
        pass

    def end_training(self, trainer: Trainer) -> None:
        """add functionality at the end of training

        Args:
            trainer (Trainer): the trainer object
        """
        pass

    def end_batch(self, trainer: Trainer, batch_num: int) -> Union[bool, None]:
        """add functionality at the end of the iteration / batch

        Args:
            trainer (Trainer): the trainer object
            batch_num (int): the index of the current iteration / batch

        Returns:
            Union[bool, None]:
        """
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class DataLoader(TrainerCallback):
    pass


class PeriodicTrainerCallback(TrainerCallback):
    """Base class for trainer callbacks which perform an action periodically after a number of iterations
        
    Args:
        step (int, optional): Number of iterations after which the action is repeated. Defaults to 1.
        last_epoch (int, optional): the last training  iteration for which the action is performed. Defaults to -1.
    """
    def __init__(self, step: int = 1, last_epoch: int = -1):
        self.step = step
        self.last_epoch = last_epoch

    def end_batch(self, trainer: Trainer, batch_num: int) -> Union[bool, None]:
        """add functionality at the end of the iteration / batch

        Args:
            trainer (Trainer): the trainer object
            batch_num (int): the index of the current iteration / batch

        Returns:
            Union[bool, None]:
        """
        if (
                is_divisor(batch_num, self.step) and
                (self.last_epoch == -1 or batch_num < self.last_epoch)
        ):
            return self._end_batch(trainer, batch_num)

    def _end_batch(self, trainer: Trainer, batch_num: int) -> Union[bool, None]:
        pass


class _StackedTrainerCallbacks(TrainerCallback):
    def __init__(self, callbacks: Iterable[TrainerCallback]):
        self.callbacks = tuple(callbacks)

    def start_training(self, trainer: Trainer) -> None:
        for c in self.callbacks:
            c.start_training(trainer)

    def end_training(self, trainer: Trainer) -> None:
        for c in self.callbacks:
            c.end_training(trainer)

    def end_batch(self, trainer: Trainer, batch_num: int) -> Union[bool, None]:
        break_epoch = False
        for c in self.callbacks:
            break_epoch += bool(c.end_batch(trainer, batch_num))
        return break_epoch

    def __repr__(self):
        callbacks = ", ".join(repr(c) for c in self.callbacks)
        return f'StackedTrainerCallbacks({callbacks})'


def _init_logger(logger: Union[Logger, Tuple[Logger, ...], Loggers] = None):
    if not logger:
        return Logger()
    if isinstance(logger, Logger):
        return logger
    return Loggers(*logger)


def _update_loss_dict(loss_dict: dict, new_values: dict):
    for k, v in new_values.items():
        loss_dict[k].append(v.item())
