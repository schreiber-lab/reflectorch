import numpy as np
import torch
import torch.nn.functional as F
import torchdiffeq

from torch import Tensor, nn
from dataclasses import dataclass
from typing import Optional

from reflectorch.data_generation import BATCH_DATA_TYPE
from reflectorch.ml.basic_trainer import Trainer
from reflectorch.ml.dataloaders import ReflectivityDataLoader
from reflectorch.ml.ema import ModelEma

__all__ = [
    'RealTimeSimTrainer',
    'DenoisingAETrainer',
    'PointEstimatorTrainer',
    'NeuralOdeTrainer',
    'FlowMatchingTrainer',
    'NFlowTrainer',
]


@dataclass
class BasicBatchData:
    scaled_curves: torch.Tensor
    scaled_bounds: torch.Tensor
    scaled_params: torch.Tensor = None
    scaled_sigmas: Optional[torch.Tensor] = None
    scaled_q_values: Optional[torch.Tensor] = None
    scaled_denoised_curves: Optional[torch.Tensor] = None
    key_padding_mask: Optional[torch.Tensor] = None
    scaled_conditioning_params: Optional[torch.Tensor] = None
    unscaled_q_values: Optional[torch.Tensor] = None

@dataclass
class OdeBatchData:
    scaled_curves: torch.Tensor
    t: torch.Tensor
    scaled_bounds: torch.Tensor = None
    scaled_params: torch.Tensor = None
    scaled_sigmas: Optional[torch.Tensor] = None
    scaled_q_values: Optional[torch.Tensor] = None
    scaled_denoised_curves: Optional[torch.Tensor] = None
    key_padding_mask: Optional[torch.Tensor] = None
    scaled_conditioning_params: Optional[torch.Tensor] = None
    unscaled_q_values: Optional[torch.Tensor] = None

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
        self.condition_on_ambient_sld = getattr(self, 'condition_on_ambient_sld', False)
    
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
        scaled_sigmas = get_scaled_or_none('scaled_sigmas', None) if self.train_with_sigmas else None
        key_padding_mask = batch_data.get('key_padding_mask', None)

        scaled_q_resolutions = get_scaled_or_none('q_resolutions', self.loader.smearing.scale_resolutions) if self.condition_on_q_resolutions else None
        scaled_conditioning_ambient_sld = get_scaled_or_none('scaled_conditioning_ambient_sld', None) if self.condition_on_ambient_sld else None
        conditioning_params = []
        if scaled_q_resolutions is not None:
            conditioning_params.append(scaled_q_resolutions)
        if scaled_conditioning_ambient_sld is not None:
            conditioning_params.append(scaled_conditioning_ambient_sld)
        scaled_conditioning_params = torch.cat(conditioning_params, dim=-1) if len(conditioning_params) > 0 else None
    
        num_params = scaled_params.shape[-1] // 3
        assert num_params * 3 == scaled_params.shape[-1]
        scaled_params, scaled_bounds = torch.split(scaled_params, [num_params, 2 * num_params], dim=-1)

        return BasicBatchData(
            scaled_params=scaled_params,
            scaled_bounds=scaled_bounds,
            scaled_curves=scaled_curves,
            scaled_sigmas=scaled_sigmas,
            scaled_q_values=scaled_q_values,
            scaled_denoised_curves=scaled_denoised_curves,
            scaled_conditioning_params=scaled_conditioning_params,
            unscaled_q_values=batch_data['q_values'],
            key_padding_mask=key_padding_mask,
        )

    def get_loss_dict(self, batch_data: BasicBatchData):
        """Returns the regression loss"""
        scaled_params=batch_data.scaled_params
        scaled_curves=batch_data.scaled_curves
        scaled_bounds=batch_data.scaled_bounds
        scaled_q_values=batch_data.scaled_q_values
        scaled_sigmas=batch_data.scaled_sigmas
        key_padding_mask=batch_data.key_padding_mask
        scaled_conditioning_params=batch_data.scaled_conditioning_params
        unscaled_q_values=batch_data.unscaled_q_values

        predicted_params = self.model(
            curves = scaled_curves,
            bounds = scaled_bounds,
            q_values = scaled_q_values,
            sigmas = scaled_sigmas,
            conditioning_params = scaled_conditioning_params,
            key_padding_mask = key_padding_mask,
            unscaled_q_values = unscaled_q_values,
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

class FlowMatchingTrainer(RealTimeSimTrainer):
    def init(self):
        if getattr(self, 'use_l1_loss', False):
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.MSELoss()
        
        self.variant = getattr(self, 'variant', 'tong')
        self.path_type = getattr(self, 'path_type', 'linear')
        self.sigma = getattr(self, 'sigma', 0.0)
        self.base_distribution = getattr(self, 'base_distribution', 'normal')

        self.use_ema = getattr(self, 'use_ema', False)
        if self.use_ema:
            self.ema_decay = getattr(self, 'ema_decay',  0.9999)
            self.ema_device = getattr(self, 'ema_device', 'cpu')
            self.model_ema = ModelEma(self.model, self.ema_decay, self.ema_device)

        self.train_with_bounds = getattr(self, 'train_with_bounds', False)
        self.train_with_q_input = getattr(self, 'train_with_q_input', False)
        self.train_with_sigmas = getattr(self, 'train_with_sigmas', False)
        self.condition_on_q_resolutions = getattr(self, 'condition_on_q_resolutions', False)
        self.condition_on_ambient_sld = getattr(self, 'condition_on_ambient_sld', False)

        self.track_reconstruction_metric = getattr(self, 'track_reconstruction_metric', False)
        self.reconstruction_freq = getattr(self, 'reconstruction_freq', 100)

    def _get_batch(self, batch_data: BATCH_DATA_TYPE) -> OdeBatchData:
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
        scaled_sigmas = get_scaled_or_none('scaled_sigmas', None) if self.train_with_sigmas else None
        key_padding_mask = batch_data.get('key_padding_mask', None)
        t = batch_data['t'].to(torch.float32)

        scaled_q_resolutions = get_scaled_or_none('q_resolutions', self.loader.smearing.scale_resolutions) if self.condition_on_q_resolutions else None
        scaled_conditioning_ambient_sld = get_scaled_or_none('scaled_conditioning_ambient_sld', None) if self.condition_on_ambient_sld else None
        conditioning_params = []
        if scaled_q_resolutions is not None:
            conditioning_params.append(scaled_q_resolutions)
        if scaled_conditioning_ambient_sld is not None:
            conditioning_params.append(scaled_conditioning_ambient_sld)
        scaled_conditioning_params = torch.cat(conditioning_params, dim=-1) if len(conditioning_params) > 0 else None

        num_params = scaled_params.shape[-1] // 3
        assert num_params * 3 == scaled_params.shape[-1]
        scaled_params, scaled_bounds = torch.split(scaled_params, [num_params, 2 * num_params], dim=-1)

        if not self.train_with_bounds:
            scaled_bounds = None

        return OdeBatchData(
            scaled_params=scaled_params,
            scaled_bounds=scaled_bounds,
            scaled_curves=scaled_curves,
            scaled_sigmas=scaled_sigmas,
            scaled_q_values=scaled_q_values,
            scaled_denoised_curves=scaled_denoised_curves,
            key_padding_mask=key_padding_mask,
            t=t,
            scaled_conditioning_params=scaled_conditioning_params,
            unscaled_q_values=batch_data['q_values'],
        )


    def get_loss_dict(self, batch_data: OdeBatchData):
        scaled_params=batch_data.scaled_params
        scaled_bounds=batch_data.scaled_bounds
        scaled_curves=batch_data.scaled_curves
        scaled_sigmas=batch_data.scaled_sigmas
        scaled_q_values=batch_data.scaled_q_values
        scaled_conditioning_params=batch_data.scaled_conditioning_params
        key_padding_mask=batch_data.key_padding_mask
        unscaled_q_values=batch_data.unscaled_q_values
        t = batch_data.t

        x_0 = self.sample_from_base_distribution(scaled_params.shape[0], scaled_params.shape[1], scaled_params.device, self.base_distribution)
        x_1 = scaled_params

        if self.variant == 'lipman': #one-sided
            assert self.sigma > 0
            assert self.base_distribution == 'normal'
            mu_x_t = t * x_1
            sigma_x_t = 1 - (1 - self.sigma) * t
            epsilon = torch.randn_like(mu_x_t)
            x_t = mu_x_t + sigma_x_t * epsilon
            u_t = (x_1 - (1 - self.sigma) * x_t) / (1 - (1 - self.sigma) * t)
        elif self.variant == 'tong': #two-sided
            if self.path_type == "linear":
                if self.sigma == 0:
                    x_t = (1.0 - t) * x_0 + t * x_1
                else:
                    x_t = (1.0 - t + self.sigma) * x_0 + t * x_1
                u_t = x_1 - x_0
            elif self.path_type == 'sinusoidal':
                x_t = torch.cos(0.5*torch.pi*t) * x_0 + torch.sin(0.5*torch.pi*t) * x_1
                u_t = 0.5*torch.pi*(-torch.sin(0.5*torch.pi*t)* x_0 + torch.cos(0.5*torch.pi*t) * x_1)

        v_t = self._compute_v_t(
            network=self.model,
            scaled_curves=scaled_curves, 
            x_t=x_t, 
            t=t, 
            scaled_bounds=scaled_bounds, 
            scaled_q_values=scaled_q_values, 
            scaled_sigmas=scaled_sigmas, 
            scaled_conditioning_params=scaled_conditioning_params, 
            key_padding_mask=key_padding_mask,
            unscaled_q_values=unscaled_q_values,
        )

        loss = self.criterion(u_t, v_t)

        if self.track_reconstruction_metric:
            curve_rec_error = self._compute_curve_rec_error(
                scaled_curves=scaled_curves,
                scaled_bounds=scaled_bounds,
                scaled_q_values=scaled_q_values,
                scaled_sigmas=scaled_sigmas,
                scaled_conditioning_params=scaled_conditioning_params,
                key_padding_mask=key_padding_mask,
                unscaled_q_values=unscaled_q_values,
                num_lrs=len(self.lrs),
            )
        else:
            curve_rec_error = None

        loss_dict = {'loss': loss}
        if curve_rec_error is not None:
            loss_dict['curve_rec_error'] = curve_rec_error

        return loss_dict

    def _compute_v_t(
        self,
        network,
        scaled_curves,
        x_t,
        t,
        scaled_bounds,
        scaled_q_values,
        scaled_sigmas,
        scaled_conditioning_params,
        key_padding_mask,
        unscaled_q_values=None,
        *,
        embedding: Optional[torch.Tensor] = None,
    ):  

        v_t = network(
            curves=scaled_curves,
            x_t=x_t,
            t=t,
            bounds=scaled_bounds,
            q_values=scaled_q_values,
            sigmas=scaled_sigmas,
            conditioning_params=scaled_conditioning_params,
            key_padding_mask=key_padding_mask,
            unscaled_q_values=unscaled_q_values,
            embedding=embedding,
        )

        return v_t
    
    def _compute_curve_rec_error(self, scaled_curves, scaled_bounds, scaled_q_values, scaled_sigmas, 
                                 scaled_conditioning_params, key_padding_mask, unscaled_q_values, num_lrs):
        if num_lrs == 0:
            self.last_curve_rec_error = torch.tensor([99.0], device=scaled_curves.device)
        if self.track_reconstruction_metric and (num_lrs + 1) % self.reconstruction_freq == 0:
            with torch.no_grad():
                scaled_predicted_params = self.sample(
                    scaled_curves=scaled_curves,
                    scaled_bounds=scaled_bounds,
                    scaled_q_values=scaled_q_values,
                    scaled_sigmas=scaled_sigmas,
                    scaled_conditioning_params=scaled_conditioning_params,
                    key_padding_mask=key_padding_mask,
                    num_samples=1,
                    device=scaled_curves.device,
                    method='dopri5',
                    rtol=1e-5,
                    atol=1e-5,
                ).squeeze()
                if scaled_bounds is None:
                    param_dim = self.loader.prior_sampler.param_dim
                    scaled_bounds = torch.ones(scaled_curves.shape[0], 2*param_dim).to(scaled_curves)
                restored_predictions = self.loader.prior_sampler.restore_params(
                    torch.cat([scaled_predicted_params, scaled_bounds], dim=-1)
                )
                predicted_curves = restored_predictions.reflectivity(unscaled_q_values)
                scaled_predicted_curves = self.loader.curves_scaler.scale(predicted_curves)

                curve_rec_error = torch.mean(torch.abs(scaled_curves - scaled_predicted_curves))
                self.last_curve_rec_error = curve_rec_error
        else:
            curve_rec_error = self.last_curve_rec_error
        return curve_rec_error

    def sample(
        self,
        scaled_curves,
        scaled_bounds=None,
        scaled_q_values=None,
        scaled_sigmas=None,
        scaled_conditioning_params=None,
        key_padding_mask=None,
        unscaled_q_values=None,
        num_samples=1,
        device="cuda",
        sampler: str = "ode", # "ode" or "sde"
        method="dopri5",
        rtol=1e-5,
        atol=1e-5,
        inference_with_ema=False,
        return_x0=False,
        visualize_trajectories=False,
        num_steps_fixed=100,
        sde_num_steps: int = 200,
        sde_variant: str = "singular",  # "singular" | "zero_ends"
        sde_alpha: float = np.sqrt(2.0),
        sde_eps: float = 1e-3,
    ):
        """
        """

        num_params = self.loader.prior_sampler.param_dim
        batch_size = scaled_curves.shape[0]
        total_num_samples = batch_size * num_samples
        device = device or scaled_curves.device

        x0 = self.sample_from_base_distribution(total_num_samples, num_params, device=device, distribution=self.base_distribution)

        net = self.model_ema.module if (inference_with_ema and hasattr(self, "model_ema")) else self.model
        net.eval()

        with torch.no_grad():
            emb_B = net.compute_embedding(
                curves=scaled_curves,
                q_values=scaled_q_values,
                sigmas=scaled_sigmas,
                key_padding_mask=key_padding_mask,
                unscaled_q_values=unscaled_q_values,
            )
        emb = emb_B.repeat_interleave(num_samples, dim=0)

        rep = lambda t: None if t is None else t.repeat_interleave(num_samples, dim=0)
        s_bounds = rep(scaled_bounds)
        s_q = rep(scaled_q_values)
        s_sig = rep(scaled_sigmas)
        s_cond = rep(scaled_conditioning_params)
        kpm = rep(key_padding_mask)
        us_q = rep(unscaled_q_values)

        if sampler.lower() == "ode":
            ts_ode = (
                torch.linspace(0.0, 1.0, num_steps_fixed, device=device)
                if num_steps_fixed is not None
                else torch.tensor([0.0, 1.0], device=device)
            )

            num_intervals = ts_ode.numel() - 1

            def eval_fn(x, t_scalar):
                t_rep = t_scalar.repeat(x.shape[0], 1)
                return self._compute_v_t(
                    network=net,
                    scaled_curves=None,  #not needed when embedding is provided
                    x_t=x,
                    t=t_rep,
                    scaled_bounds=s_bounds,
                    scaled_q_values=s_q,
                    scaled_sigmas=s_sig,
                    scaled_conditioning_params=s_cond,
                    key_padding_mask=kpm,
                    unscaled_q_values=us_q,
                    embedding=emb,
                )

            def ode_model_wrapper(t, x):
                return eval_fn(x, t)

            with torch.no_grad():
                states = torchdiffeq.odeint(
                    ode_model_wrapper,
                    x0,
                    ts_ode,
                    atol=atol,
                    rtol=rtol,
                    method=method,
                )
            x1 = states[-1]

        elif sampler.lower() == "sde":
            eps = sde_eps if sde_variant == "singular" else 0.0
            ts = torch.linspace(eps, 1.0 - eps, sde_num_steps + 1, device=device)
            x = x0

            traj_states = [x0] if visualize_trajectories else None

            # Euler–Maruyama
            with torch.no_grad():
                for k in range(sde_num_steps):
                    t_k, t_kp1 = ts[k], ts[k + 1]
                    dt = t_kp1 - t_k
                    t_rep = t_k.repeat(total_num_samples, 1)

                    g_tilde = self.diffusion_fn(
                        t=t_rep,
                        sde_variant=sde_variant,
                        sde_alpha=sde_alpha,
                    )
                    f_tilde = self.drift_fn(
                        x_t=x,
                        t=t_rep,
                        scaled_curves=None,
                        scaled_bounds=s_bounds,
                        scaled_q_values=s_q,
                        scaled_sigmas=s_sig,
                        scaled_conditioning_params=s_cond,
                        key_padding_mask=kpm,
                        unscaled_q_values=us_q,
                        embedding=emb,
                        inference_with_ema=inference_with_ema,
                        sde_variant=sde_variant,
                        sde_alpha=sde_alpha,
                    )

                    x = x + dt * f_tilde + torch.sqrt(dt) * g_tilde * torch.randn_like(x)

                    if visualize_trajectories:
                        traj_states.append(x)

            x1 = x

            if visualize_trajectories:
                states = torch.stack(traj_states, dim=0)
                return ts, states

        else:
            raise ValueError(f"Unknown sampler='{sampler}'. Use 'ode' or 'sde'.")

        if visualize_trajectories:
            return ts_ode, states

        x1 = x1.reshape(batch_size, num_samples, num_params).transpose(0, 1)  # (S,B,d)

        if return_x0:
            x0 = x0.reshape(batch_size, num_samples, num_params).transpose(0, 1)
            return x1, x0

        return x1

    
    def sample_from_base_distribution(self, batch_size, n_params, device, distribution):
        if distribution == 'normal':
            return torch.randn(batch_size, n_params, device=device)
        elif distribution == 'uniform':
            return torch.rand(batch_size, n_params, device=device)
        elif distribution == 'uniform_symmetric':
            return 2 * (torch.rand(batch_size, n_params, device=device) - 0.5)
        else:
            raise NotImplementedError
    
    def log_p_base(self, x, base_distribution, reduction='sum', dim=-1):
        if base_distribution == 'normal':
            log_p = -0.5 * np.log(2. * np.pi) - 0.5 * x**2.
        elif base_distribution == 'uniform':
            log_p = torch.where((x >= 0) & (x <= 1), torch.zeros_like(x), torch.full_like(x, float('-inf')))
        elif base_distribution == 'uniform_symmetric':
            log_p_value = -torch.log(torch.tensor(2.0))
            log_p = torch.where((x >= -1) & (x <= 1), torch.full_like(x, log_p_value), torch.full_like(x, float('-inf')))

        if reduction == 'mean':
            return torch.mean(log_p, dim)
        elif reduction == 'sum':
            return torch.sum(log_p, dim)
        else:
            return log_p
        
    def augmented_ode(self, t, x_and_div_t, scaled_curves, scaled_bounds, scaled_q_values, scaled_sigmas,
                       scaled_conditioning_params, key_padding_mask, unscaled_q_values, inference_with_ema):
        x_t = x_and_div_t[:, :-1].detach()
        t = t.repeat(x_t.shape[0], 1).detach()
        with torch.enable_grad():
            x_t.requires_grad = True

            if inference_with_ema:
                v_t = self._compute_v_t(self.model_ema.module, scaled_curves, x_t, t, scaled_bounds, scaled_q_values, scaled_sigmas, 
                                        scaled_conditioning_params, key_padding_mask, unscaled_q_values)
            else:
                v_t = self._compute_v_t(self.model, scaled_curves, x_t, t, scaled_bounds, scaled_q_values, scaled_sigmas, 
                                        scaled_conditioning_params, key_padding_mask, unscaled_q_values)

            div_v_t = compute_divergence(v_t, x_t)
            v_t = v_t.detach()
            div_v_t = div_v_t.detach()

            assert not torch.isnan(v_t).any(), "v_t contains NaN"
            assert not torch.isinf(v_t).any(), "v_t contains Inf"
            assert not torch.isnan(div_v_t).any(), "div_v_t contains NaN"
            assert not torch.isinf(div_v_t).any(), "div_v_t contains Inf"

        return torch.cat((v_t, -div_v_t), dim=-1)

    def log_prob(self, x_1, scaled_curves, scaled_bounds, scaled_q_values=None, scaled_sigmas=None, 
                 scaled_conditioning_params=None, key_padding_mask=None, unscaled_q_values=None,
                 method='dopri5', rtol=1e-7, atol=1e-7, inference_with_ema=False):
        num_samples, batch_size, num_params = x_1.shape
        x_1 = x_1.reshape(-1, num_params)

        scaled_curves = scaled_curves.repeat_interleave(num_samples, dim=0)
        scaled_bounds = scaled_bounds.repeat_interleave(num_samples, dim=0) if scaled_bounds is not None else None
        scaled_q_values = scaled_q_values.repeat_interleave(num_samples, dim=0) if scaled_q_values is not None else None
        scaled_sigmas = scaled_sigmas.repeat_interleave(num_samples, dim=0) if scaled_sigmas is not None else None
        scaled_conditioning_params = scaled_conditioning_params.repeat_interleave(num_samples, dim=0) if scaled_conditioning_params is not None else None
        key_padding_mask = key_padding_mask.repeat_interleave(num_samples, dim=0) if key_padding_mask is not None else None
        unscaled_q_values = unscaled_q_values.repeat_interleave(num_samples, dim=0) if unscaled_q_values is not None else None

        x_and_div_1 = torch.cat((x_1, torch.zeros((x_1.shape[0], 1), device=x_1.device)), dim=-1)

        assert not torch.isnan(x_and_div_1).any(), "x_and_div_1 contains NaN"
        assert not torch.isinf(x_and_div_1).any(), "x_and_div_1 contains Inf"

        self.model.nfe = 0
        self.model.eval()
        _, x_and_div_0 = torchdiffeq.odeint(
            lambda t, x_and_div_t: self.augmented_ode(t, x_and_div_t, scaled_curves, scaled_bounds, scaled_q_values, 
                                                      scaled_sigmas, scaled_conditioning_params, key_padding_mask, unscaled_q_values, inference_with_ema),
            x_and_div_1,
            torch.tensor([1.0, 0.0], device=x_1.device),
            atol=atol,
            rtol=rtol,
            method=method,
            options={"norm": norm_without_divergence_component},
        )

        x_0, div_0 = x_and_div_0[:, :-1], x_and_div_0[:, -1]

        log_p_base = self.log_p_base(x_0, self.base_distribution, dim=-1)
        log_p_base = log_p_base.reshape(num_samples, batch_size)
        div_0 = div_0.reshape(num_samples, batch_size)
        log_prob_result = (log_p_base - div_0).detach()

        return log_prob_result

    def sample_and_log_prob(
        self,
        *,
        scaled_curves: torch.Tensor,
        scaled_bounds: Optional[torch.Tensor] = None,
        scaled_q_values: Optional[torch.Tensor] = None,
        scaled_sigmas: Optional[torch.Tensor] = None,
        scaled_conditioning_params: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        unscaled_q_values: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        method: str = "dopri5",
        rtol: float = 1e-5,
        atol: float = 1e-5,
        num_steps_fixed: int = 100,
        device: str = "cuda",
        inference_with_ema: bool = False,
    ):
        """
        """
        B = scaled_curves.shape[0]
        d = self.loader.prior_sampler.param_dim
        S = num_samples
        tot = S * B
        device = device or scaled_curves.device

        x0 = self.sample_from_base_distribution(tot, d, device, self.base_distribution)
        log_p0 = self.log_p_base(x0, self.base_distribution, reduction="sum")

        log_det0 = torch.zeros(tot, 1, device=device)

        net = self.model_ema.module if (inference_with_ema and hasattr(self, "model_ema")) else self.model
        net.eval()

        with torch.no_grad():
            emb_B = net.compute_embedding(
                curves=scaled_curves,
                q_values=scaled_q_values,
                sigmas=scaled_sigmas,
                key_padding_mask=key_padding_mask,
                unscaled_q_values=unscaled_q_values,
            )
        emb = emb_B.repeat_interleave(S, dim=0)  # (tot,E)

        rep = lambda t: None if t is None else t.repeat_interleave(S, 0)
        s_bounds = rep(scaled_bounds)
        s_q = rep(scaled_q_values)
        s_sig = rep(scaled_sigmas)
        s_cond = rep(scaled_conditioning_params)
        kpm = rep(key_padding_mask)
        us_q = rep(unscaled_q_values)

        with torch.no_grad():
            def aug_dynamics(t, state):
                x = state[:, :-1]
                t_r = t.repeat(x.shape[0], 1)

                with torch.enable_grad():
                    x_req = x.detach().requires_grad_(True)

                    v_t = self._compute_v_t(
                        network=net,
                        scaled_curves=None,  #not needed when embedding is provided
                        x_t=x_req,
                        t=t_r,
                        scaled_bounds=s_bounds,
                        scaled_q_values=s_q,
                        scaled_sigmas=s_sig,
                        scaled_conditioning_params=s_cond,
                        key_padding_mask=kpm,
                        unscaled_q_values=us_q,
                        embedding=emb,
                    )

                    div = compute_divergence(v_t, x_req)  # (tot,1)

                return torch.cat([v_t.detach(), div.detach()], dim=-1)
            
            ts_ode = torch.linspace(0.0, 1.0, num_steps_fixed, device=device) if num_steps_fixed is not None else torch.tensor([0.0, 1.0], device=device)

            z0 = torch.cat([x0, log_det0], dim=-1)  # (tot,d+1)

            states = torchdiffeq.odeint(
                aug_dynamics,
                z0,
                ts_ode,
                atol=atol,
                rtol=rtol,
                method=method,
                options={"norm": norm_without_divergence_component},
            )
            z1 = states[-1]

        x1, log_det1 = z1[:, :-1], z1[:, -1]  # (tot,d), (tot,)

        x1 = x1.view(S, B, d)
        log_q = (log_p0 - log_det1).view(S, B)  # log q_phi(theta|x)

        return x1, log_q

    def _sigma_min(self) -> float:
        return float(getattr(self, "sigma_min", getattr(self, "sigma", 1e-3)))

    @torch.no_grad()
    def score_fn(
        self,
        *,
        x_t,
        t,
        scaled_curves=None,
        scaled_bounds=None,
        scaled_q_values=None,
        scaled_sigmas=None,
        scaled_conditioning_params=None,
        key_padding_mask=None,
        unscaled_q_values=None,
        embedding=None,
        inference_with_ema: bool = False,
        v_t=None,
    ) -> Tensor:
        """
        Score implied by the rectified-flow velocity field, in the convention t: 0->1
        """
        sigma_min = self._sigma_min()


        net = self.model_ema.module if (inference_with_ema and hasattr(self, "model_ema")) else self.model

        if v_t is None:
            v_t = self._compute_v_t(
                network=net,
                scaled_curves=scaled_curves,
                x_t=x_t,
                t=t,
                scaled_bounds=scaled_bounds,
                scaled_q_values=scaled_q_values,
                scaled_sigmas=scaled_sigmas,
                scaled_conditioning_params=scaled_conditioning_params,
                key_padding_mask=key_padding_mask,
                unscaled_q_values=unscaled_q_values,
                embedding=embedding,
            )

        return (t * v_t - x_t) / (1.0 - t + sigma_min)
    

    @torch.no_grad()
    def diffusion_fn(
        self,
        *,
        t: Tensor,
        sde_variant: str = "singular", 
        sde_alpha: float = np.sqrt(2.0),
    ) -> Tensor:
        """
        """
        sigma_min = self._sigma_min()

        if sde_variant == "singular":
            return sde_alpha * torch.sqrt(((1.0 - t) + sigma_min) / t) #i.e. torch.sqrt(2* ((1.0 - t) + sigma_min) / t), when alpha=sqrt(2)
        elif sde_variant == "zero_ends":
            return sde_alpha * torch.sqrt(t * (1.0 - t))
        else:
            raise NotImplementedError("sde_variant must be 'singular' or 'zero_ends'")

    @torch.no_grad()
    def drift_fn(
        self,
        *,
        x_t: Tensor,
        t: Tensor,
        scaled_curves=None,
        scaled_bounds=None,
        scaled_q_values=None,
        scaled_sigmas=None,
        scaled_conditioning_params=None,
        key_padding_mask=None,
        unscaled_q_values=None,
        embedding=None,
        inference_with_ema: bool = False,
        sde_variant: str = "singular",
        sde_alpha: float = np.sqrt(2.0),
    ) -> Tensor:
        """
        """

        net = self.model_ema.module if (inference_with_ema and hasattr(self, "model_ema")) else self.model

        v_t = self._compute_v_t(
            network=net,
            scaled_curves=scaled_curves,
            x_t=x_t,
            t=t,
            scaled_bounds=scaled_bounds,
            scaled_q_values=scaled_q_values,
            scaled_sigmas=scaled_sigmas,
            scaled_conditioning_params=scaled_conditioning_params,
            key_padding_mask=key_padding_mask,
            unscaled_q_values=unscaled_q_values,
            embedding=embedding,
        )

        score = self.score_fn(x_t=x_t, t=t, v_t=v_t)

        g_tilde = self.diffusion_fn(
            t=t,
            sde_variant=sde_variant,
            sde_alpha=sde_alpha,
        )
        g2 = g_tilde * g_tilde

        f_tilde = v_t + 0.5 * g2 * score
        return f_tilde

    def mean_t_fn(self, t: Tensor) -> Tensor:
        return t

    def std_fn(self, t: Tensor) -> Tensor:
        sigma_min = self._sigma_min()
        return 1.0 - t + sigma_min      


NeuralOdeTrainer = FlowMatchingTrainer


def compute_divergence(y, x, method="autograd_loop", n_est_hutchinson=100):
    if method == "autograd_loop":
        div = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        for i in range(y.shape[-1]):
            (grad_i,) = torch.autograd.grad(
                outputs=y[..., i],
                inputs=x,
                grad_outputs=torch.ones_like(y[..., i]),
                retain_graph=True,
                create_graph=False,
            )
            div = div + grad_i[..., i:i+1]
        return div
    elif method == "hutchinson":
        divs = []
        for _ in range(n_est_hutchinson):
            eps = torch.randn_like(x)
            y_eps = (y * eps).sum()
            (grad,) = torch.autograd.grad(
                outputs=y_eps,
                inputs=x,
                create_graph=False,
                retain_graph=False,
            )
            divs.append((grad * eps).sum(-1, keepdim=True))
        return torch.stack(divs, dim=0).mean(0)
    else:
        raise ValueError(method)
    
def norm_without_divergence_component(y):
    return torch.linalg.norm(y[:, :-1])
    

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
    
class NFlowTrainer(RealTimeSimTrainer):
    """
    """
    def init(self):
        self.train_with_bounds = getattr(self, 'train_with_bounds', False)
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
        scaled_q_values = get_scaled_or_none('q_values', self.loader.q_generator.scale_q) if self.train_with_q_input else None
        scaled_sigmas = get_scaled_or_none('scaled_sigmas', None) if self.train_with_sigmas else None
        key_padding_mask = batch_data.get('key_padding_mask', None)

        scaled_q_resolutions = get_scaled_or_none('q_resolutions', self.loader.smearing.scale_resolutions) if self.condition_on_q_resolutions else None
        conditioning_params = []
        if scaled_q_resolutions is not None:
            conditioning_params.append(scaled_q_resolutions)
        scaled_conditioning_params = torch.cat(conditioning_params, dim=-1) if len(conditioning_params) > 0 else None
    
        num_params = scaled_params.shape[-1] // 3
        assert num_params * 3 == scaled_params.shape[-1]
        scaled_params, scaled_bounds = torch.split(scaled_params, [num_params, 2 * num_params], dim=-1)

        if not self.train_with_bounds:
            scaled_bounds = None

        return BasicBatchData(
            scaled_params=scaled_params,
            scaled_bounds=scaled_bounds,
            scaled_curves=scaled_curves,
            scaled_sigmas=scaled_sigmas,
            scaled_q_values=scaled_q_values,
            scaled_conditioning_params=scaled_conditioning_params,
            unscaled_q_values=batch_data['q_values'],
            key_padding_mask=key_padding_mask,
        )

    def get_loss_dict(self, batch_data: BasicBatchData):
        """Returns the regression loss"""
        scaled_params=batch_data.scaled_params
        scaled_curves=batch_data.scaled_curves
        scaled_bounds=batch_data.scaled_bounds
        scaled_q_values=batch_data.scaled_q_values
        scaled_sigmas=batch_data.scaled_sigmas
        key_padding_mask=batch_data.key_padding_mask
        scaled_conditioning_params=batch_data.scaled_conditioning_params
        unscaled_q_values=batch_data.unscaled_q_values

        loss = -self.model.log_prob(
            inputs=scaled_params,
            curves = scaled_curves,
            bounds = scaled_bounds,
            q_values = scaled_q_values,
            sigmas = scaled_sigmas,
            conditioning_params = scaled_conditioning_params,
            key_padding_mask = key_padding_mask,
            unscaled_q_values = unscaled_q_values,
        ).mean()

        return {'loss': loss}