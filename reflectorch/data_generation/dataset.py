# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Union
import warnings

from torch import Tensor
import torch

from reflectorch.data_generation.priors import PriorSampler, BasicParams
from reflectorch.data_generation.noise import QNoiseGenerator, IntensityNoiseGenerator
from reflectorch.data_generation.q_generator import QGenerator
from reflectorch.data_generation.scale_curves import CurvesScaler, LogAffineCurvesScaler
from reflectorch.data_generation.smearing import Smearing

BATCH_DATA_TYPE = Dict[str, Union[Tensor, BasicParams]]


class BasicDataset(object):
    """Reflectometry dataset. It generates the q positions, samples the thin film parameters from the prior, 
    simulates the reflectivity curves and applies noise to the curves.

    Args:
        q_generator (QGenerator): the momentum transfer (q) generator
        prior_sampler (PriorSampler): the prior sampler
        intensity_noise (IntensityNoiseGenerator, optional): the intensity noise generator. Defaults to None.
        q_noise (QNoiseGenerator, optional): the q noise generator. Defaults to None.
        curves_scaler (CurvesScaler, optional): the reflectivity curve scaler. Defaults to an instance of LogAffineCurvesScaler, 
                                                which scales the curves to the range [-1, 1], the minimum considered intensity being 1e-10.
        calc_denoised_curves (bool, optional): whether to add the curves without noise to the dictionary. Defaults to False.
        smearing (Smearing, optional): curve smearing generator. Defaults to None.
    """
    def __init__(self,
                 q_generator: QGenerator,
                 prior_sampler: PriorSampler,
                 intensity_noise: IntensityNoiseGenerator = None,
                 q_noise: QNoiseGenerator = None,
                 curves_scaler: CurvesScaler = None,
                 calc_denoised_curves: bool = False,
                 smearing: Smearing = None,
                 ):
        self.q_generator = q_generator
        self.intensity_noise = intensity_noise
        self.q_noise = q_noise
        self.curves_scaler = curves_scaler or LogAffineCurvesScaler()
        self.prior_sampler = prior_sampler
        self.smearing = smearing
        self.calc_denoised_curves = calc_denoised_curves

    def update_batch_data(self, batch_data: BATCH_DATA_TYPE) -> None:
        """implement in a subclass to edit batch_data dict inplace"""
        pass

    def _sample_from_prior(self, batch_size: int):
        params: BasicParams = self.prior_sampler.sample(batch_size)
        scaled_params: Tensor = self.prior_sampler.scale_params(params)
        return params, scaled_params

    def get_batch(self, batch_size: int) -> BATCH_DATA_TYPE:
        """get a batch of data as a dictionary with keys ``params``, ``scaled_params``, ``q_values``, ``curves``, ``scaled_noisy_curves``

        Args:
            batch_size (int): the batch size
        """
        batch_data = {}

        params, scaled_params = self._sample_from_prior(batch_size)

        batch_data['params'] = params
        batch_data['scaled_params'] = scaled_params

        q_values: Tensor = self.q_generator.get_batch(batch_size, batch_data)

        if self.q_noise:
            batch_data['original_q_values'] = q_values
            q_values = self.q_noise.apply(q_values, batch_data)

        batch_data['q_values'] = q_values

        curves = self._calc_curves(q_values, params)

        if self.calc_denoised_curves:
            batch_data['curves'] = curves

        noisy_curves = curves

        if self.intensity_noise:
            noisy_curves = self.intensity_noise(noisy_curves, batch_data)

        scaled_noisy_curves = self.curves_scaler.scale(noisy_curves)
        batch_data['scaled_noisy_curves'] = scaled_noisy_curves

        is_finite = torch.all(torch.isfinite(scaled_noisy_curves), -1)
        if not torch.all(is_finite).item():
            infinite_indices = ~is_finite
            warnings.warn(f'Batch with {infinite_indices.sum().item()} curves with infinities skipped.')
            return self.get_batch(batch_size = batch_size)

        is_finite = torch.all(torch.isfinite(batch_data['scaled_noisy_curves']), -1)
        assert torch.all(is_finite).item()

        self.update_batch_data(batch_data)

        return batch_data

    def _calc_curves(self, q_values: Tensor, params: BasicParams):
        if self.smearing:
            curves = self.smearing.get_curves(q_values, params)
        else:
            curves = params.reflectivity(q_values)
        curves = curves.to(q_values)
        return curves


def _insert_batch_data(tgt_batch_data, add_batch_data, indices):
    for key in tuple(tgt_batch_data.keys()):
        value = tgt_batch_data[key]
        if isinstance(value, BasicParams) or len(value.shape) == 2:
            value[indices] = add_batch_data[key]
        else:
            warnings.warn(f'Ignore {key} while merging batch_data.')


if __name__ == '__main__':
    from reflectorch.data_generation.q_generator import ConstantQ
    from reflectorch.data_generation.priors import BasicPriorSampler, UniformSubPriorSampler
    from reflectorch.data_generation.noise import BasicExpIntensityNoise
    from reflectorch.data_generation.noise import BasicQNoiseGenerator
    from reflectorch.utils import to_np
    from time import perf_counter

    q_generator = ConstantQ((0, 0.2, 65), device='cpu')
    noise_gen = BasicExpIntensityNoise(
        relative_errors=(0.05, 0.2),
        # scale_range=(-1e-2, 1e-2),
        logdist=True,
        apply_shift=True,
    )
    q_noise_gen = BasicQNoiseGenerator(
        shift_std=5e-4,
        noise_std=(0, 1e-3),
    )
    prior_sampler = UniformSubPriorSampler(
        thickness_range=(0, 250),
        roughness_range=(0, 40),
        sld_range=(0, 60),
        num_layers=2,
        device=torch.device('cpu'),
        dtype=torch.float64,
        smaller_roughnesses=True,
        logdist=True,
        relative_min_bound_width=5e-4,
    )
    smearing = Smearing(
        sigma_range=(0.8e-3, 5e-3),
        gauss_num=31,
        share_smeared=0.5,
    )

    dataset = BasicDataset(
        q_generator,
        prior_sampler,
        noise_gen,
        q_noise=q_noise_gen,
        smearing=smearing
    )
    start = perf_counter()
    batch_data = dataset.get_batch(32)
    print(f'Total time = {(perf_counter() - start):.3f} sec ')
    print(batch_data['params'].roughnesses[:10])
    print(batch_data['scaled_noisy_curves'].min().item())

    scaled_noisy_curves = batch_data['scaled_noisy_curves']
    scaled_curves = dataset.curves_scaler.scale(
        batch_data['params'].reflectivity(q_generator.q)
    )

    try:
        import matplotlib.pyplot as plt

        for i in range(16):
            plt.plot(
                to_np(q_generator.q.squeeze().cpu().numpy()),
                to_np(scaled_curves[i])
            )
            plt.plot(
                to_np(q_generator.q.squeeze().cpu().numpy()),
                to_np(scaled_noisy_curves[i])
            )

            plt.show()
    except ImportError:
        pass
