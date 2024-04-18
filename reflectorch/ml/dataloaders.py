from torch import Tensor


from reflectorch.data_generation import BasicDataset
from reflectorch.data_generation.reflectivity import kinematical_approximation
from reflectorch.data_generation.priors import Params
from reflectorch.ml.basic_trainer import DataLoader


__all__ = [
    "ReflectivityDataLoader",
    "MultilayerDataLoader",
]


class ReflectivityDataLoader(BasicDataset, DataLoader):
    """Class combining functionality from BasicDataset (basic dataset class for reflectivity) and the DataLoader (which inherits from TrainerCallback)"""
    pass


class MultilayerDataLoader(ReflectivityDataLoader):
    """Dataloader for reflectivity curves simulated using the kinematical approximation (for multilayer parameterizations)."""
    def _sample_from_prior(self, batch_size: int):
        return self.prior_sampler.optimized_sample(batch_size)

    def _calc_curves(self, q_values: Tensor, params: Params):
        return kinematical_approximation(q_values, params.thicknesses, params.roughnesses, params.slds)