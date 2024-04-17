# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from reflectorch.data_generation import BATCH_DATA_TYPE
from reflectorch.ml.basic_trainer import Trainer
from reflectorch.ml.dataloaders import ReflectivityDataLoader

__all__ = [
    'RealTimeSimTrainer',
    'DenoisingAETrainer',
    'PointEstimatorTrainer',
]


class RealTimeSimTrainer(Trainer):
    """Trainer with functionality to customize the sampled batch of data"""
    loader: ReflectivityDataLoader

    def get_batch_by_idx(self, batch_num: int):
        """Gets a batch of data with the default batch size"""
        batch_data = self.loader.get_batch(self.batch_size)
        return self._get_batch(batch_data)

    def get_batch_by_size(self, batch_size: int):
        """Gets a batch of data with a custom batch size"""
        batch_data = self.loader.get_batch(batch_size)
        return self._get_batch(batch_data)

    def _get_batch(self, batch_data: BATCH_DATA_TYPE):
        """Modify the batch of data sampled from the data loader"""
        raise NotImplementedError


class PointEstimatorTrainer(RealTimeSimTrainer):
    """Point estimator trainer for the inverse problem."""
    add_sigmas_to_context: bool = False
        
    def _get_batch(self, batch_data: BATCH_DATA_TYPE):
        scaled_params, context, q_values = batch_data['scaled_params'], batch_data['scaled_noisy_curves'], batch_data['q_values']

        if self.add_sigmas_to_context:
            sigmas = self.loader.curves_scaler.scale(batch_data['sigmas'])
            context = torch.cat([context, sigmas], dim=-1)
            
        scaled_params, context = scaled_params.to(torch.float32), context.to(torch.float32)
        scaled_params, context = batch_data['params'].rearrange_context_from_params(scaled_params, context)

        return scaled_params, context, q_values

    def get_loss_dict(self, batch_data):
        """Returns the regression loss"""

        scaled_params, context, q_values = batch_data

        if self.train_with_q_input:
            predicted_params = self.model(context, q_values)
        else:
            predicted_params = self.model(context)
            
        loss = self.mse(predicted_params, scaled_params)
        return {'loss': loss}

    def init(self):
        self.mse = nn.MSELoss()


class DenoisingAETrainer(RealTimeSimTrainer):
    """Trainer which can be used for training a denoising autoencoder model. Overrides _get_batch and get_loss_dict methods """
    def init(self):
        self.criterion = nn.MSELoss()
        self.loader.calc_denoised_curves = True

    def _get_batch(self, batch_data: BATCH_DATA_TYPE):
        """returns scaled curves with and without noise"""
        scaled_noisy_curves, curves = batch_data['scaled_noisy_curves'], batch_data['curves']
        scaled_curves = self.loader.curves_scaler.scale(curves)

        scaled_noisy_curves, scaled_curves = scaled_noisy_curves.to(torch.float32), scaled_curves.to(torch.float32)

        return scaled_noisy_curves, scaled_curves

    def get_loss_dict(self, batch_data):
        """returns the reconstruction loss of the autoencoder"""
        scaled_noisy_curves, scaled_curves = batch_data
        restored_curves = self.model(scaled_noisy_curves)
        loss = self.criterion(scaled_curves, restored_curves)
        return {'loss': loss}
