import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from reflectorch.data_generation import BATCH_DATA_TYPE
from reflectorch.ml.basic_trainer import Trainer
from reflectorch.ml.dataloaders import ReflectivityDataLoader

__all__ = [
    'RealTimeSimTrainer',
    'DenoisingAETrainer',
    'VAETrainer',
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
    """Trainer for the regression inverse problem with incorporation of prior bounds"""
    add_sigmas_to_context: bool = False
        
    def _get_batch(self, batch_data: BATCH_DATA_TYPE):
        scaled_params = batch_data['scaled_params'].to(torch.float32)
        scaled_curves = batch_data['scaled_noisy_curves'].to(torch.float32)
        if self.train_with_q_input:
            q_values = batch_data['q_values'].to(torch.float32)
            scaled_q_values = self.loader.q_generator.scale_q(q_values)
        else:
            scaled_q_values = None

        num_params = scaled_params.shape[-1] // 3
        assert num_params * 3 == scaled_params.shape[-1]
        scaled_params, scaled_bounds = torch.split(scaled_params, [num_params, 2 * num_params], dim=-1)

        return scaled_params, scaled_bounds, scaled_curves, scaled_q_values

    def get_loss_dict(self, batch_data):
        """computes the loss dictionary"""

        scaled_params, scaled_bounds, scaled_curves, scaled_q_values = batch_data

        if self.train_with_q_input:
            predicted_params = self.model(scaled_curves, scaled_bounds, scaled_q_values)
        else:
            predicted_params = self.model(scaled_curves, scaled_bounds)
            
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
    

class VAETrainer(DenoisingAETrainer):
    """Trainer which can be used for training a denoising autoencoder model. Overrides _get_batch and get_loss_dict methods """
    def init(self):
        self.loader.calc_denoised_curves = True
        self.freebits = 0.05
    
    def calc_kl(self, z_mu, z_logvar):
        return 0.5*(z_mu**2 + torch.exp(z_logvar) - 1 - z_logvar)
    
    def gaussian_log_prob(self, z, mu, logvar):
        return -0.5*(np.log(2*np.pi) + logvar + (z-mu)**2/torch.exp(logvar))

    def get_loss_dict(self, batch_data):
        """returns the reconstruction loss of the autoencoder"""
        scaled_noisy_curves, scaled_curves = batch_data
        _, (z_mu, z_logvar, restored_curves_mu, restored_curves_logvar) = self.model(scaled_noisy_curves)

        l_rec = -torch.mean(self.gaussian_log_prob(scaled_curves, restored_curves_mu, restored_curves_logvar), dim=-1)  
        l_kl = torch.mean(F.relu(self.calc_kl(z_mu, z_logvar) - self.freebits*np.log(2)) + self.freebits*np.log(2), dim=-1)
        loss = torch.mean(l_rec + l_kl)/np.log(2)

        l_rec = torch.mean(l_rec)
        l_kl = torch.mean(l_kl)

        return {'loss': loss}