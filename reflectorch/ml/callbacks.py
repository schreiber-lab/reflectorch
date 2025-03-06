import torch

import numpy as np

from reflectorch.ml.basic_trainer import (
    TrainerCallback,
    Trainer,
)
from reflectorch.ml.utils import is_divisor

__all__ = [
    'SaveBestModel',
    'LogLosses',
]


class SaveBestModel(TrainerCallback):
    """Callback for periodically saving the best model weights

    Args:
        path (str): path for saving the model weights
        freq (int, optional): frequency in iterations at which the current average loss is evaluated. Defaults to 50.
        average (int, optional): number of recent iterations over which the average loss is computed. Defaults to 10.
    """

    def __init__(self, path: str, freq: int = 50, average: int = 10):
        self.path = path
        self.average = average
        self._best_loss = np.inf
        self.freq = freq

    def end_batch(self, trainer: Trainer, batch_num: int) -> None:
        """checks if the current average loss has improved from the previous save, if true the model is saved

        Args:
            trainer (Trainer): the trainer object
            batch_num (int): the current iteration / batch
        """
        if is_divisor(batch_num, self.freq):

            loss = np.mean(trainer.losses['total_loss'][-self.average:])

            if loss < self._best_loss:
                self._best_loss = loss
                self.save(trainer, batch_num)

    def save(self, trainer: Trainer, batch_num: int):
        """saves a dictionary containing the network weights, the learning rates, the losses and the current \
            best loss with its corresponding iteration to the disk

        Args:
            trainer (Trainer): the trainer object
            batch_num (int): the current iteration / batch
        """
        prev_save = trainer.callback_params.pop('saved_iteration', 0)
        trainer.callback_params['saved_iteration'] = batch_num
        save_dict = {
            'model': trainer.model.state_dict(),
            'lrs': trainer.lrs,
            'losses': trainer.losses,
            'prev_save': prev_save,
            'batch_num': batch_num,
            'best_loss': self._best_loss
        }
        torch.save(save_dict, self.path)


class LogLosses(TrainerCallback):
    """Callback for logging the training losses"""
    def end_batch(self, trainer: Trainer, batch_num: int) -> None:
        """log loss at the current iteration

        Args:
            trainer (Trainer): the trainer object
            batch_num (int): the index of the current iteration / batch
        """
        for loss_name, loss_values in trainer.losses.items():
            try:
                trainer.log(f'train/{loss_name}', loss_values[-1])
            except IndexError:
                continue