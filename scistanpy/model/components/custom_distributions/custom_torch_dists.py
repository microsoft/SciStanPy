"""Holds custom torch distributions used by SciStanPy"""

from __future__ import annotations

from typing import Optional, ParamSpec

import torch
import torch.distributions as dist

# Define a type variable for the parameters of a distribution
P = ParamSpec("P")


class CustomDistribution:
    """Doesn't do anything. Just useful for type hinting."""


class Multinomial(CustomDistribution):
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

        return torch.stack(
            [d.log_prob(v) for d, v in zip(self.distributions, value)]
        ).reshape(self._batch_shape)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """See documentation of torch.distributions.Multinomial.sample"""
        # Make the samples. Each sample comes from the batch. We reshape each sample
        # to match the original shape, stack the samples, then reshape to get the
        # appropriate sample dimension
        return torch.stack(
            [d.sample(sample_shape=sample_shape) for d in self.distributions], dim=-2
        ).reshape((*sample_shape, *self._batch_shape, self._n_categories))


class MultinomialProb(Multinomial, CustomDistribution):
    """
    See documentation for `Multinomial`. This is identical to `Multinomial` but
    assumes that the input values are probabilities rather than logits.
    """

    def __init__(
        self,
        total_count: int | torch.Tensor = 1,
        probs: Optional[torch.Tensor] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        """See documentation for torch.distributions.Multinomial"""
        # Call the parent class with probs
        super().__init__(
            total_count=total_count, probs=probs, validate_args=validate_args
        )


class MultinomialLogit(Multinomial, CustomDistribution):
    """
    See documentation for `Multinomial`. This is identical to `Multinomial` but
    assumes that the input values are logits rather than probabilities.
    """

    def __init__(
        self,
        total_count: int | torch.Tensor = 1,
        logits: Optional[torch.Tensor] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        """See documentation for torch.distributions.Multinomial"""
        # Call the parent class with logits
        super().__init__(
            total_count=total_count, logits=logits, validate_args=validate_args
        )


class MultinomialLogTheta(MultinomialLogit):
    """
    Identical to MultinomialLogit, but we make sure that the sum of the exponentiated
    logits is already 1.
    """

    def __init__(
        self,
        total_count: int | torch.Tensor = 1,
        log_probs: Optional[torch.Tensor] = None,
        validate_args: Optional[bool] = None,
    ) -> None:

        # Make sure the log_probs are normalized
        assert log_probs is not None, "log_probs must be provided"
        assert torch.allclose(
            log_probs.exp().sum(dim=-1), torch.ones_like(log_probs[..., 0])
        ), "log_probs must be normalized to sum to 1"

        # Otherwise, we can just call the parent class
        super().__init__(
            total_count=total_count,
            logits=log_probs,
            validate_args=validate_args,
        )


# pylint: disable=abstract-method
class Lomax(dist.transformed_distribution.TransformedDistribution, CustomDistribution):
    """Implementation of the Lomax distribution (shifted Pareto distribution)."""

    def __init__(self, lambda_: torch.Tensor, alpha: torch.Tensor, *args, **kwargs):
        """
        Args:
            lambda_ (torch.Tensor): Scale parameter.
            alpha (torch.Tensor): Shape parameter.
            *args: Additional arguments for the base distribution.
            **kwargs: Additional keyword arguments for the base distribution.
        """
        # Build the base distribution and the transforms (just a shift in the output)
        base_dist = dist.Pareto(scale=lambda_, alpha=alpha)
        transforms = [dist.transforms.AffineTransform(loc=-lambda_, scale=1)]
        super().__init__(base_dist, transforms, *args, **kwargs)


class ExpLomax(
    dist.transformed_distribution.TransformedDistribution, CustomDistribution
):
    """Implementation of the Exponential Lomax distribution."""

    def __init__(self, lambda_: torch.Tensor, alpha: torch.Tensor, *args, **kwargs):
        """
        Args:
            lambda_ (torch.Tensor): Scale parameter.
            alpha (torch.Tensor): Shape parameter.
            *args: Additional arguments for the base distribution.
            **kwargs: Additional keyword arguments for the base distribution.
        """
        # Build the base distribution (Lomax) and transforms (log)
        base_dist = Lomax(lambda_=lambda_, alpha=alpha)
        transforms = [dist.transforms.ExpTransform().inv]
        super().__init__(base_dist, transforms, *args, **kwargs)


class ExpExponential(
    dist.transformed_distribution.TransformedDistribution, CustomDistribution
):
    """Implementation of the Exponential Exponential distribution."""

    def __init__(self, rate: torch.Tensor, *args, **kwargs):
        """
        Args:
            lambda_ (torch.Tensor): Scale parameter.
            *args: Additional arguments for the base distribution.
            **kwargs: Additional keyword arguments for the base distribution.
        """
        # Build the base distribution (Exponential) and transforms (log)
        base_dist = dist.Exponential(rate=rate)
        transforms = [dist.transforms.ExpTransform().inv]
        super().__init__(base_dist, transforms, *args, **kwargs)


class ExpDirichlet(
    dist.transformed_distribution.TransformedDistribution, CustomDistribution
):
    """Implementation of the Exponential Dirichlet distribution."""

    def __init__(self, concentration: torch.Tensor, *args, **kwargs):
        """
        Args:
            concentration (torch.Tensor): Concentration parameter.
            *args: Additional arguments for the base distribution.
            **kwargs: Additional keyword arguments for the base distribution.
        """
        # Build the base distribution (Dirichlet) and transforms (log)
        base_dist = dist.Dirichlet(concentration=concentration)
        transforms = [dist.transforms.ExpTransform().inv]
        super().__init__(base_dist, transforms, *args, **kwargs)


class ExpNormal(
    dist.transformed_distribution.TransformedDistribution, CustomDistribution
):
    """Implementation of the Exponential Normal distribution."""

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor, *args, **kwargs):
        """
        Args:
            mu (torch.Tensor): Mean parameter.
            sigma (torch.Tensor): Standard deviation parameter.
            *args: Additional arguments for the base distribution.
            **kwargs: Additional keyword arguments for the base distribution.
        """
        # Build the base distribution (Normal) and transforms (log)
        base_dist = dist.Normal(loc=mu, scale=sigma)
        transforms = [dist.transforms.ExpTransform().inv]
        super().__init__(base_dist, transforms, *args, **kwargs)


# pylint: enable=abstract-method
