from torch.optim import lr_scheduler

import numpy as np

from reflectorch.ml.basic_trainer import Trainer, TrainerCallback, PeriodicTrainerCallback

__all__ = [
    'ScheduleBatchSize',
    'ScheduleLR',
    'StepLR',
    'CyclicLR',
    'LogCyclicLR',
    'ReduceLROnPlateau',
    'OneCycleLR',
]


class ScheduleBatchSize(PeriodicTrainerCallback):
    """Batch size scheduler

    Args:
        step (int): number of iterations after which the batch size is modified.
        gamma (int, optional): quantity which is added to or multiplied with the current batch size. Defaults to 2.
        last_epoch (int, optional): the last training iteration for which the batch size is modified. Defaults to -1.
        mode (str, optional): ``'add'`` for addition or ``'multiply'`` for multiplication. Defaults to 'add'.
    """
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
    """Base class for learning rate schedulers

    Args:
        lr_scheduler_cls: class of the learning rate scheduler
    """
    
    def __init__(self, lr_scheduler_cls, **kwargs):
        self.lr_scheduler_cls = lr_scheduler_cls
        self.kwargs = kwargs
        self.lr_scheduler = None

    def start_training(self, trainer: Trainer) -> None:
        """initializes a learning rate scheduler based on its class and keyword arguments at the start of training

        Args:
            trainer (Trainer): the trainer object
        """
        self.lr_scheduler = self.lr_scheduler_cls(trainer.optim, **self.kwargs)
        trainer.callback_params['lrs'] = []

    def end_batch(self, trainer: Trainer, batch_num: int) -> None:
        """modifies the learning rate at the end of each iteration

        Args:
            trainer (Trainer): the trainer object
            batch_num (int): index of the current iteration
        """
        self.lr_scheduler.step()


class StepLR(ScheduleLR):
    """Learning rate scheduler which decays the learning rate of each parameter group by gamma every ``step_size`` epochs.
    
    Args:
        step_size (int): Period of learning rate decay
        gamma (float): Multiplicative factor of learning rate decay
        last_epoch (int, optional): The index of last iteration. Defaults to -1.
    """
    def __init__(self, step_size: int, gamma: float, last_epoch: int = -1, **kwargs):

        
        super().__init__(lr_scheduler.StepLR, step_size=step_size, gamma=gamma, last_epoch=last_epoch, **kwargs)

    def start_training(self, trainer: Trainer) -> None:
        trainer.optim.param_groups[0]['initial_lr'] = trainer.lr()
        super().start_training(trainer)


class CyclicLR(ScheduleLR):
    """Cyclic learning rate scheduler

    Args:
        base_lr (float): Initial learning rate which is the lower boundary in the cycle
        max_lr (float): Upper learning rate boundary in the cycle
        step_size_up (int, optional): Number of training iterations in the increasing half of a cycle. Defaults to 2000.
        cycle_momentum (bool, optional):  If True, momentum is cycled inversely to learning rate between ``base_momentum`` and ``max_momentum``. Defaults to False.
        gamma (float, optional): Constant in ``‘exp_range’`` mode scaling function: gamma^(cycle iterations). Defaults to 1.
        mode (str, optional):  One of: ``'triangular'`` (a basic triangular cycle without amplitude scaling),
              ``'triangular2'`` (a basic triangular cycle that scales initial amplitude by half each cycle), ``'exp_range'`` 
              (a cycle that scales initial amplitude by gamma^iterations at each cycle iteration). Defaults to 'triangular'.
    """
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
    """Cyclic learning rate scheduler on a logarithmic scale
    
    Args:
        base_lr (float): Lower learning rate boundary in the cycle
        max_lr (float): Upper learning rate boundary in the cycle
        period (int, optional): Number of training iterations in the cycle. Defaults to 2000.
        gamma (float, optional): Constant for scaling the amplitude as ``gamma`` ^ ``iterations``. Defaults to 1.
        start_period (int, optional): Number of starting iterations with the default learning rate.
        log (bool, optional): If ``True``, the cycle is in the logarithmic domain.
        param_groups (tupe, optional): Parameter groups of the optimizer.
    """
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


class ReduceLROnPlateau(TrainerCallback):
    """Learning rate scheduler which reduces the learning rate when the loss stops decreasing
    
    Args:
            gamma (float, optional): Multiplicative factor of learning rate decay. Defaults to 0.5.
            patience (int, optional): The number of allowed iterations with no improvement after which the learning rate will be reduced. Defaults to 500.
            average (int, optional): Size of the window over which the average loss is computed. Defaults to 50.
            loss_key (str, optional): Defaults to 'total_loss'.
            param_groups (tuple, optional): Defaults to (0,).
    
    """
    def __init__(
            self,
            gamma: float = 0.5,
            patience: int = 500,
            average: int = 50,
            loss_key: str = 'total_loss',
            param_groups: tuple = (0,),
    ):
        """
        """
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
    """One-cycle learning rate scheduler (https://arxiv.org/abs/1708.07120)

    Args:
        max_lr (float): Upper learning rate boundary in the cycle
        total_steps (int): The total number of steps in the cycle
        pct_start (float, optional): The percentage of the cycle (in number of steps) spent increasing the learning rate. Defaults to 0.3.
        div_factor (float, optional): Determines the initial learning rate via initial_lr = ``max_lr`` / ``div_factor``. Defaults to 25..
        final_div_factor (float, optional): Determines the minimum learning rate via min_lr = ``initial_lr`` / ``final_div_factor``. Defaults to 1e4.
        three_phase (bool, optional): If ``True``, use a third phase of the schedule to annihilate the learning rate according to ``final_div_factor`` instead of modifying the second phase. Defaults to True.
    """
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
