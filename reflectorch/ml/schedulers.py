# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim import lr_scheduler

import numpy as np

from reflectorch.ml.basic_trainer import Trainer, TrainerCallback, PeriodicTrainerCallback

__all__ = [
    'ScheduleBatchSize',
    'ScheduleLR',
    'StepLR',
    'CyclicLR',
    'LogCyclicLR',
    'ReduceLrOnPlateau',
    'OneCycleLR',
]


class ScheduleBatchSize(PeriodicTrainerCallback):
    def __init__(self, step: int, gamma: int = 2, last_epoch: int = -1, mode: str = 'add'):
        super().__init__(step, last_epoch)

        assert mode in ('add', 'multiply')

        self.gamma = gamma
        self.mode = mode

    def _end_batch(self, trainer: Trainer, batch_num: int) -> None:
        if self.mode == 'add':
            trainer.batch_size += self.gamma
        elif self.mode == 'multiply':
            trainer.batch_size *= self.gamma


class ScheduleLR(TrainerCallback):
    def __init__(self, lr_scheduler_cls, **kwargs):
        self.lr_scheduler_cls = lr_scheduler_cls
        self.kwargs = kwargs
        self.lr_scheduler = None

    def start_training(self, trainer: Trainer) -> None:
        self.lr_scheduler = self.lr_scheduler_cls(trainer.optim, **self.kwargs)
        trainer.callback_params['lrs'] = []

    def end_batch(self, trainer: Trainer, batch_num: int) -> None:
        self.lr_scheduler.step()


class StepLR(ScheduleLR):
    def __init__(self, step_size: int, gamma: float, last_epoch: int = -1, **kwargs):
        super().__init__(lr_scheduler.StepLR, step_size=step_size, gamma=gamma, last_epoch=last_epoch, **kwargs)

    def start_training(self, trainer: Trainer) -> None:
        trainer.optim.param_groups[0]['initial_lr'] = trainer.lr()
        super().start_training(trainer)


class CyclicLR(ScheduleLR):
    def __init__(self, base_lr, max_lr, step_size_up: int = 2000,
                 cycle_momentum: bool = False, gamma: float = 1., mode: str = 'triangular',
                 **kwargs):
        super().__init__(
            lr_scheduler.CyclicLR,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            cycle_momentum=cycle_momentum,
            gamma=gamma,
            mode=mode,
            **kwargs
        )


class LogCyclicLR(TrainerCallback):
    def __init__(self,
                 base_lr,
                 max_lr,
                 period: int = 2000,
                 gamma: float = None,
                 log: bool = True,
                 param_groups: tuple = (0,),
                 start_period: int = 25,
                 ):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.period = period
        self.gamma = gamma
        self.param_groups = param_groups
        self.log = log
        self.start_period = start_period
        self._axis = None
        self._period = None

    def get_lr(self, batch_num: int):
        return self._get_lr(batch_num)

    def _get_lr(self, batch_num):
        num_period, t = batch_num // self.period, batch_num % self.period

        if self._period != num_period:
            self._period = num_period
            if self.gamma and (num_period >= self.start_period):
                amp = (self.max_lr - self.base_lr) * (self.gamma ** (num_period - self.start_period))
                max_lr = self.base_lr + amp
            else:
                max_lr = self.max_lr

            if self.log:
                self._axis = np.logspace(np.log10(self.base_lr), np.log10(max_lr), self.period // 2)
            else:
                self._axis = np.linspace(self.base_lr, max_lr, self.period // 2)
        if t < self.period // 2:
            lr = self._axis[t]
        else:
            lr = self._axis[self.period - t - 1]
        return lr

    def end_batch(self, trainer: Trainer, batch_num: int):
        lr = self.get_lr(batch_num)
        for param_group in self.param_groups:
            trainer.set_lr(lr, param_group)


class ReduceLrOnPlateau(TrainerCallback):
    def __init__(
            self,
            gamma: float = 0.5,
            patience: int = 500,
            average: int = 50,
            loss_key: str = 'total_loss',
            param_groups: tuple = (0,),
    ):
        self.patience = patience
        self.average = average
        self.gamma = gamma
        self.loss_key = loss_key
        self.param_groups = param_groups

    def end_batch(self, trainer: Trainer, batch_num: int) -> None:
        loss = trainer.losses[self.loss_key]

        if len(loss) < self.patience:
            return

        if np.mean(loss[-self.patience:-(self.patience - self.average)]) <= np.mean(loss[-self.average:]):
            for param_group in self.param_groups:
                trainer.set_lr(trainer.lr(param_group) * self.gamma, param_group)
                
                
class OneCycleLR(ScheduleLR):
    def __init__(self, max_lr: float, total_steps: int, pct_start: float = 0.3, div_factor: float = 25., 
                 final_div_factor: float = 1e4, three_phase: bool = True, **kwargs):
        super().__init__(
            lr_scheduler.OneCycleLR,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=div_factor ,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
            **kwargs
        )
