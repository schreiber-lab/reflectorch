###as in the PANPE repository

import torch
from torch import Tensor

from nflows.flows import Flow
from nflows.utils import typechecks as check


class FlowWrapper(Flow):
    @torch.no_grad()

    def log_prob(self, inputs, context=None):
        """
        Calculate log probability under the distribution.
        """
        inputs = torch.as_tensor(inputs)
        return self._log_prob(inputs, context)

    def sample(self, num_samples, context=None, batch_size=None):
        """
        Generates samples from the distribution. Samples can be generated in batches.
        """
        if not check.is_positive_int(num_samples):
            raise TypeError("Number of samples must be a positive integer.")

        if batch_size is None:
            return self._sample(num_samples, context)

        else:
            if not check.is_positive_int(batch_size):
                raise TypeError("Batch size must be a positive integer.")

            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples = [self._sample(batch_size, context) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self._sample(num_leftover, context))
            return torch.cat(samples, dim=0)