import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass
from typing import Optional

from reflectorch.data_generation import BATCH_DATA_TYPE
from reflectorch.ml.basic_trainer import Trainer
from reflectorch.ml.dataloaders import ReflectivityDataLoader

__all__ = [
    'RealTimeSimTrainer',
    'DenoisingAETrainer',
    'PointEstimatorTrainer',
]


@dataclass
class BasicBatchData:
    scaled_curves: torch.Tensor
    scaled_bounds: torch.Tensor
    scaled_params: torch.Tensor = None
    scaled_sigmas: Optional[torch.Tensor] = None
    scaled_q_values: Optional[torch.Tensor] = None
    scaled_denoised_curves: Optional[torch.Tensor] = None
    scaled_conditioning_params: Optional[torch.Tensor] = None

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

    def init(self):
        if getattr(self, 'use_l1_loss', False):
            self.criterion = nn.L1Loss(reduction='none')
        else:
            self.criterion = nn.MSELoss(reduction='none')
        self.use_curve_reconstruction_loss = getattr(self, 'use_curve_reconstruction_loss', False)
        self.rescale_loss_interval_width = getattr(self, 'rescale_loss_interval_width', False)
        if self.use_curve_reconstruction_loss:
            self.loader.calc_denoised_curves = True

        self.train_with_q_input = getattr(self, 'train_with_q_input', False)
        self.train_with_sigmas = getattr(self, 'train_with_sigmas', False)
        self.condition_on_q_resolutions = getattr(self, 'condition_on_q_resolutions', False)
    
    def _get_batch(self, batch_data: BATCH_DATA_TYPE) -> BasicBatchData:
        def get_scaled_or_none(key, scaler=None):
            value = batch_data.get(key)
            if value is None:
                return None
            scale_func = scaler or (lambda x: x)
            return scale_func(value).to(torch.float32)

        scaled_params = batch_data['scaled_params'].to(torch.float32)
        scaled_curves = batch_data['scaled_noisy_curves'].to(torch.float32)
        scaled_denoised_curves = get_scaled_or_none('curves', self.loader.curves_scaler.scale)
        scaled_q_values = get_scaled_or_none('q_values', self.loader.q_generator.scale_q) if self.train_with_q_input else None

        scaled_q_resolutions = get_scaled_or_none('q_resolutions', self.loader.smearing.scale_resolutions) if self.condition_on_q_resolutions else None
        conditioning_params = []
        if scaled_q_resolutions is not None:
            conditioning_params.append(scaled_q_resolutions)
        scaled_conditioning_params = torch.cat(conditioning_params, dim=-1) if len(conditioning_params) > 0 else None
    
        num_params = scaled_params.shape[-1] // 3
        assert num_params * 3 == scaled_params.shape[-1]
        scaled_params, scaled_bounds = torch.split(scaled_params, [num_params, 2 * num_params], dim=-1)

        return BasicBatchData(
            scaled_params=scaled_params,
            scaled_bounds=scaled_bounds,
            scaled_curves=scaled_curves,
            scaled_q_values=scaled_q_values,
            scaled_denoised_curves=scaled_denoised_curves,
            scaled_conditioning_params=scaled_conditioning_params,
        )

    def get_loss_dict(self, batch_data: BasicBatchData):
        """Returns the regression loss"""
        scaled_params=batch_data.scaled_params
        scaled_curves=batch_data.scaled_curves
        scaled_bounds=batch_data.scaled_bounds
        scaled_q_values=batch_data.scaled_q_values
        scaled_conditioning_params=batch_data.scaled_conditioning_params

        predicted_params = self.model(
            curves = scaled_curves,
            bounds = scaled_bounds,
            q_values = scaled_q_values,
            conditioning_params = scaled_conditioning_params,
        )

        if not self.rescale_loss_interval_width:
            loss = self.criterion(predicted_params, scaled_params).mean()
        else:
            n_params = scaled_params.shape[-1]
            b_min = scaled_bounds[..., :n_params]
            b_max = scaled_bounds[..., n_params:]
            interval_width = b_max - b_min

            base_loss  = self.criterion(predicted_params, scaled_params)
            if isinstance(self.criterion, torch.nn.MSELoss):
                width_factors = (interval_width / 2) ** 2
            elif isinstance(self.criterion, torch.nn.L1Loss):
                width_factors = interval_width / 2

            loss = (width_factors * base_loss).mean()

        return {'loss': loss}


# class PointEstimatorTrainer(RealTimeSimTrainer):
#     """Trainer for the regression inverse problem with incorporation of prior bounds"""
#     add_sigmas_to_context: bool = False
        
#     def _get_batch(self, batch_data: BATCH_DATA_TYPE):
#         scaled_params = batch_data['scaled_params'].to(torch.float32)
#         scaled_curves = batch_data['scaled_noisy_curves'].to(torch.float32)
#         if self.train_with_q_input:
#             q_values = batch_data['q_values'].to(torch.float32)
#             scaled_q_values = self.loader.q_generator.scale_q(q_values)
#         else:
#             scaled_q_values = None

#         num_params = scaled_params.shape[-1] // 3
#         assert num_params * 3 == scaled_params.shape[-1]
#         scaled_params, scaled_bounds = torch.split(scaled_params, [num_params, 2 * num_params], dim=-1)

#         return scaled_params, scaled_bounds, scaled_curves, scaled_q_values

#     def get_loss_dict(self, batch_data):
#         """computes the loss dictionary"""

#         scaled_params, scaled_bounds, scaled_curves, scaled_q_values = batch_data

#         if self.train_with_q_input:
#             predicted_params = self.model(scaled_curves, scaled_bounds, scaled_q_values)
#         else:
#             predicted_params = self.model(scaled_curves, scaled_bounds)
            
#         loss = self.mse(predicted_params, scaled_params)
#         return {'loss': loss}

#     def init(self):
#         self.mse = nn.MSELoss()


class DenoisingAETrainer(RealTimeSimTrainer):
    """Trainer which can be used for training a denoising autoencoder model. Overrides _get_batch and get_loss_dict methods """
    def init(self):
        self.loader.calc_denoised_curves = True
    
        if getattr(self, 'use_l1_loss', False):
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.MSELoss()

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
    