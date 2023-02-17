# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from IPython.display import clear_output

from ...ml import TrainerCallback, Trainer

from ..matplotlib import plot_losses


class JPlotLoss(TrainerCallback):
    def __init__(self, frequency: int, log: bool = True, clear: bool = True, **kwargs):
        self.frequency = frequency
        self.log = log
        self.kwargs = kwargs
        self.clear = clear

    def end_batch(self, trainer: Trainer, batch_num: int) -> None:
        if not batch_num % self.frequency:
            if self.clear:
                clear_output(wait=True)

            plot_losses(
                trainer.losses,
                log=self.log,
                best_epoch=trainer.callback_params.get('saved_iteration', None),
                **self.kwargs
            )
