import torch
import numpy as np
from pathlib import Path
from typing import Union

from reflectorch.ml.basic_trainer import (
    TrainerCallback,
    Trainer,
)
from reflectorch.ml.utils import is_divisor

__all__ = [
    'SaveBestModel',
    'SaveModelSnapshots',
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

class SaveModelSnapshots(TrainerCallback):
    """Periodically save model snapshots without overwriting previous saves.

    Args:
        path (str): Base checkpoint path, e.g. ".../model.pt".
        freq (int): Save frequency in iterations.
        save_last (bool): Whether to also save a final snapshot at end of training.
        include_ema (bool): Whether to save EMA weights when available.
        subdir (str): Directory name for snapshots relative to base path parent.
    """

    def __init__(
        self,
        path: str,
        freq: int = 1000,
        save_last: bool = True,
        include_ema: bool = True,
        subdir: str = "checkpoints",
    ):
        self.base_path = Path(path)
        self.freq = freq
        self.save_last = save_last
        self.include_ema = include_ema
        self.subdir = subdir

    def start_training(self, trainer: Trainer) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    @property
    def snapshot_dir(self) -> Path:
        return self.base_path.parent / self.subdir

    def checkpoint_path(self, batch_num: int) -> Path:
        stem = self.base_path.stem
        suffix = self.base_path.suffix or ".pt"
        return self.snapshot_dir / f"{stem}_iter_{batch_num}{suffix}"

    def make_save_dict(self, trainer: Trainer, batch_num: int) -> dict:
        save_dict = {
            "model": trainer.model.state_dict(),
            "lrs": trainer.lrs,
            "losses": trainer.losses,
            "batch_num": batch_num,
        }
        if self.include_ema and trainer.model_ema is not None:
            save_dict["model_ema"] = trainer.model_ema.state_dict()
        return save_dict

    def save(self, trainer: Trainer, batch_num: int) -> None:
        ckpt_path = self.checkpoint_path(batch_num)
        torch.save(self.make_save_dict(trainer, batch_num), ckpt_path)

        trainer.callback_params.setdefault("saved_snapshots", []).append(str(ckpt_path))
        trainer.callback_params["last_snapshot_iteration"] = batch_num

    def end_batch(self, trainer: Trainer, batch_num: int) -> None:
        if is_divisor(batch_num, self.freq):
            self.save(trainer, batch_num)

    def end_training(self, trainer: Trainer) -> None:
        if not self.save_last:
            return

        batch_num = len(trainer.losses.get("total_loss", []))
        if trainer.callback_params.get("last_snapshot_iteration") != batch_num:
            self.save(trainer, batch_num)

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