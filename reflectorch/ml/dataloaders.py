from torch import Tensor


from reflectorch.data_generation import BasicDataset
from reflectorch.data_generation.reflectivity import kinematical_approximation
from reflectorch.data_generation.priors import Params
from reflectorch.ml.basic_trainer import DataLoader


__all__ = [
    "XrrDataLoader",
    "MultilayerDataset",
]


class XrrDataLoader(BasicDataset, DataLoader):
    """Class combining the functionalities of the BasicDataset and DataLoader classes
    """
    pass


class MultilayerDataset(XrrDataLoader):
    def _sample_from_prior(self, batch_size: int):
        return self.prior_sampler.optimized_sample(batch_size)

    def _calc_curves(self, q_values: Tensor, params: Params):
        return kinematical_approximation(q_values, params.thicknesses, params.roughnesses, params.slds)
