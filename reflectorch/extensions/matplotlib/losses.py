# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt


def plot_losses(
        losses: dict,
        log: bool = False,
        show: bool = True,
        title: str = 'Losses',
        x_label: str = 'Iterations',
        best_epoch: float = None,
        **kwargs
):
    func = plt.semilogy if log else plt.plot

    if len(losses) <= 2:
        losses = {'loss': losses['total_loss']}

    for k, data in losses.items():
        func(data, label=k, **kwargs)

    if best_epoch is not None:
        plt.axvline(best_epoch, ls='--', color='red')

    plt.xlabel(x_label)

    if len(losses) > 2:
        plt.legend()

    plt.title(title)

    if show:
        plt.show()
