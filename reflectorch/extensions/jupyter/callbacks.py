from IPython.display import clear_output

from ...ml import TrainerCallback, Trainer

from ..matplotlib import plot_losses


class JPlotLoss(TrainerCallback):
    """Callback for plotting the loss in a Jupyter notebook
    """
    def __init__(self, frequency: int, log: bool = True, clear: bool = True, **kwargs):
        """

        Args:
            frequency (int): plotting frequency
            log (bool, optional): if True, the plot is on a logarithmic scale. Defaults to True.
            clear (bool, optional):
        """
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
