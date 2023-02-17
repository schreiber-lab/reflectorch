# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

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
    'PretrainSubpriors',
]


class SaveBestModel(TrainerCallback):
    def __init__(self, path: str, freq: int = 50, average: int = 10):
        self.path = path
        self.average = average
        self._best_loss = np.inf
        self.freq = freq

    def end_batch(self, trainer: Trainer, batch_num: int) -> None:
        if is_divisor(batch_num, self.freq):

            loss = np.mean(trainer.losses['total_loss'][-self.average:])

            if loss < self._best_loss:
                self._best_loss = loss
                self.save(trainer, batch_num)

    def save(self, trainer: Trainer, batch_num: int):
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
    def end_batch(self, trainer: Trainer, batch_num: int) -> None:
        try:
            trainer.log('train/total_loss', trainer.losses[trainer.TOTAL_LOSS_KEY][-1])
        except IndexError:
            pass


class PretrainSubpriors(TrainerCallback):
    def __init__(self, num_pretrain_iterations: int):
        self.num_pretrain_iterations = num_pretrain_iterations

    def start_training(self, trainer: Trainer) -> None:
        trainer.subprior_pretraining = True

    def end_batch(self, trainer: Trainer, batch_num: int) -> None:
        if batch_num > self.num_pretrain_iterations:
            trainer.subprior_pretraining = False
