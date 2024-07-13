from torch import Tensor


from reflectorch.data_generation import BasicDataset
from reflectorch.data_generation.reflectivity import kinematical_approximation
from reflectorch.data_generation.priors import BasicParams
from reflectorch.ml.basic_trainer import DataLoader


__all__ = [
    "ReflectivityDataLoader",
    "MultilayerDataLoader",
]


class ReflectivityDataLoader(BasicDataset, DataLoader):
    """Dataloader for reflectivity data, combining functionality from the ``BasicDataset`` (basic dataset class for reflectivity) and the ``DataLoader`` (which inherits from ``TrainerCallback``) classes"""
    pass


class MultilayerDataLoader(ReflectivityDataLoader):
    """Dataloader for reflectivity curves simulated using the kinematical approximation"""
    def _sample_from_prior(self, batch_size: int):
        return self.prior_sampler.optimized_sample(batch_size)

    def _calc_curves(self, q_values: Tensor, params: BasicParams):
        return kinematical_approximation(q_values, params.thicknesses, params.roughnesses, params.slds)