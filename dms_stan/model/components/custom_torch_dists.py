"""Holds custom torch distributions used by DMS Stan"""

from typing import Optional, ParamSpec, Sequence

import torch
import torch.distributions as dist

# Define a type variable for the parameters of a distribution
P = ParamSpec("P")


class Multinomial:
    """
    Extension of torch.distributions.Multinomial that allows inhomogeneous values
    of `total_count`. This assumes that the first dimension of input parameters
    is the batch dimension.
    """

    def __init__(
        self,
        total_count: int | torch.Tensor = 1,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        """See documentation for torch.distributions.Multinomial"""
        # Probs or logits must be provided. Not both.
        if probs is None and logits is None:
            raise ValueError("Either `probs` or `logits` must be provided.")
        if probs is not None and logits is not None:
            raise ValueError("Only one of `probs` or `logits` can be provided.")

        # Are we working with probs or logits?
        key, values = ("probs", probs) if logits is None else ("logits", logits)

        # Get the shape of all but the last dimension, which is the number of categories
        self._batch_shape = values.shape[:-1]
        self._n_categories = values.shape[-1]

        # Broadcast total_count to the same shape as the values
        total_count, values = torch.broadcast_tensors(
            torch.as_tensor(total_count), values
        )

        # The last dimension should have identical entries in each row for the total
        # count.
        assert torch.all(total_count[..., 0:1] == total_count)

        # Now we build the distributions. Start by flattening all but the last dimension.
        total_count = total_count[..., 0].flatten()
        values = values.reshape(-1, values.size(-1))
        assert total_count.ndim == 1 and values.ndim == 2
        assert (
            len(total_count)
            == len(values)
            == torch.prod(torch.tensor(self._batch_shape))  # Batch size
        )

        # Build a multinomial distribution for each batch element
        self.distributions = [
            dist.Multinomial(
                **{
                    "total_count": N.item(),
                    key: values[i],
                    "validate_args": validate_args,
                }
            )
            for i, N in enumerate(total_count)
        ]

    def _stack_and_reshape(self, values: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Converts a list of tensors to a single tensor by stacking. Reshapes
        to the original shape of the inputs.
        """
        return torch.stack(values)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """See documentation of torch.distributions.Multinomial.log_prob"""
        # Ensure that the value of our multinomial is the same shape as the batch
        # dimension of the input values.
        if value.shape[:-1] != self._batch_shape:
            raise ValueError(
                f"Value shape {value.shape[:-1]} does not match batch shape {self._batch_shape}"
            )
        if value.shape[-1] != self._n_categories:
            raise ValueError(
                f"Value shape {value.shape[-1]} does not match number of categories "
                f"{self._n_categories}"
            )

        # Flatten the value tensor at all but the last dimension.
        value = value.reshape(-1, value.size(-1))
        assert len(value) == len(self.distributions)

        # Compute the log probability for each distribution
        return self._stack_and_reshape(
            [d.log_prob(v) for d, v in zip(self.distributions, value)]
        )

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """See documentation of torch.distributions.Multinomial.sample"""
        return self._stack_and_reshape(
            [d.sample(sample_shape=sample_shape) for d in self.distributions]
        )
