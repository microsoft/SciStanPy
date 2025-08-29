"""Custom PyTorch distribution implementations for SciStanPy models.

This module provides specialized PyTorch distribution classes that extend or
modify the standard PyTorch distributions to meet specific requirements of
SciStanPy modeling. These distributions handle edge cases, provide numerical
stability improvements, and enable functionality not available in the standard
PyTorch distribution library.

Key Features:
    - **Extended Multinomial**: Support for inhomogeneous total counts
    - **Numerical Stability**: Improved log-space probability computations
    - **Custom Distributions**: Implementations of distributions not in PyTorch
    - **SciStanPy Integration**: Designed for compatibility with SciStanPy parameter types

Distribution Categories:

**Multinomial Extensions**: Enhanced multinomial distributions
    - Multinomial: Base class with inhomogeneous total count support
    - MultinomialProb: Probability-parameterized multinomial
    - MultinomialLogit: Logit-parameterized multinomial
    - MultinomialLogTheta: Normalized log-probability multinomial

**Numerically Stable Distributions**: Improved standard distributions
    - Normal: Enhanced with stable log-CDF and log-survival functions
    - LogNormal: Enhanced with stable log-space probability functions

**Custom Distribution Implementations**: New distribution types
    - Lomax: Shifted Pareto distribution
    - ExpLomax: Exponential-Lomax distribution
    - ExpExponential: Exponential-Exponential distribution
    - ExpDirichlet: Exponential-Dirichlet distribution

The distributions in this module are designed to work within PyTorch's
distribution framework while providing the specific functionality required
for probabilistic modeling in SciStanPy.
"""

from __future__ import annotations

from typing import Optional, ParamSpec, TYPE_CHECKING

import torch
import torch.distributions as dist

if TYPE_CHECKING:
    from scistanpy import custom_types

# Define a type variable for the parameters of a distribution
P = ParamSpec("P")


class CustomDistribution:
    """Base marker class for custom SciStanPy distributions.

    This class serves as a marker interface for custom distribution implementations
    in SciStanPy. It doesn't provide any functionality but is useful for type
    hinting and identifying custom distributions in the codebase.

    All custom distribution classes should inherit from this class to maintain
    consistency and enable type checking.
    """


class Multinomial(CustomDistribution):
    """Extended multinomial distribution supporting inhomogeneous total counts.

    This class extends the functionality of PyTorch's standard multinomial
    distribution to support different total counts across batch dimensions.
    The standard PyTorch implementation requires all trials to have the same
    total count, but this implementation allows each batch element to have
    its own total count.

    :param total_count: Total number of trials for each batch element. Defaults to 1.
    :type total_count: Union[custom_types.Integer, torch.Tensor]
    :param probs: Event probabilities (mutually exclusive with logits)
    :type probs: Optional[torch.Tensor]
    :param logits: Event log-odds (mutually exclusive with probs)
    :type logits: Optional[torch.Tensor]
    :param validate_args: Whether to validate arguments. Defaults to None.
    :type validate_args: Optional[bool]

    :raises ValueError: If neither or both probs and logits are provided

    Key Features:
    - Supports different total counts per batch element
    - Maintains PyTorch distribution interface compatibility
    - Efficient batched computation through internal distribution creation
    - Proper shape handling for multi-dimensional batch operations

    The implementation creates individual multinomial distributions for each
    batch element, allowing for flexible modeling scenarios where trial
    counts vary across observations.

    Example:
        >>> # Different total counts for each batch element
        >>> total_counts = torch.tensor([[10], [20], [15]])
        >>> probs = torch.tensor([[0.3, 0.4, 0.3],
        ...                       [0.2, 0.5, 0.3],
        ...                       [0.4, 0.3, 0.3]])
        >>> dist = Multinomial(total_count=total_counts, probs=probs)
        >>> samples = dist.sample()
    """

    def __init__(
        self,
        total_count: "custom_types.Integer" | torch.Tensor = 1,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        """Initialize multinomial distribution with inhomogeneous total counts.

        The initialization process validates parameters, determines batch shapes,
        and creates individual multinomial distributions for each batch element
        to enable different total counts across batches.
        """
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
        """Compute log-probabilities for observed multinomial outcomes.

        :param value: Observed counts for each category
        :type value: torch.Tensor

        :returns: Log-probabilities for the observed outcomes
        :rtype: torch.Tensor

        :raises ValueError: If value shape doesn't match expected dimensions

        The method validates that the input tensor has the correct shape
        and computes log-probabilities by calling the appropriate distribution
        for each batch element.
        """
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
        """Generate samples from the multinomial distribution.

        :param sample_shape: Shape of samples to generate. Defaults to empty.
        :type sample_shape: torch.Size

        :returns: Sampled multinomial outcomes
        :rtype: torch.Tensor

        Generates samples by calling the sample method of each individual
        distribution and properly reshaping the results to maintain the
        expected batch and sample dimensions.
        """
        # Make the samples. Each sample comes from the batch. We reshape each sample
        # to match the original shape, stack the samples, then reshape to get the
        # appropriate sample dimension
        return torch.stack(
            [d.sample(sample_shape=sample_shape) for d in self.distributions], dim=-2
        ).reshape((*sample_shape, *self._batch_shape, self._n_categories))


class MultinomialProb(Multinomial, CustomDistribution):
    """Multinomial distribution parameterized by probabilities.

    This class provides a specialized interface for multinomial distributions
    where the parameters are specified as probabilities rather than logits.
    It's a convenience wrapper around the base Multinomial class.

    :param total_count: Total number of trials for each batch element. Defaults to 1.
    :type total_count: Union[custom_types.Integer, torch.Tensor]
    :param probs: Event probabilities (must sum to 1)
    :type probs: Optional[torch.Tensor]
    :param validate_args: Whether to validate arguments. Defaults to None.
    :type validate_args: Optional[bool]

    This parameterization is natural when working with probability vectors
    that are already normalized, such as output from softmax functions or
    empirical frequency estimates.

    Example:
        >>> # Probability parameterization
        >>> probs = torch.softmax(torch.randn(3, 4), dim=-1)
        >>> total_counts = torch.tensor([[100], [200], [150]])
        >>> dist = MultinomialProb(total_count=total_counts, probs=probs)
    """

    def __init__(
        self,
        total_count: "custom_types.Integer" | torch.Tensor = 1,
        probs: Optional[torch.Tensor] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        """Initialize probability-parameterized multinomial distribution.

        :param total_count: Total trials per batch element
        :param probs: Probability parameters (must be provided)
        :param validate_args: Validation flag
        """
        # Call the parent class with probs
        super().__init__(
            total_count=total_count, probs=probs, validate_args=validate_args
        )


class MultinomialLogit(Multinomial, CustomDistribution):
    """Multinomial distribution parameterized by logits.

    This class provides a specialized interface for multinomial distributions
    where the parameters are specified as logits (log-odds) rather than
    probabilities. This parameterization is often more convenient for
    modeling and optimization.

    :param total_count: Total number of trials for each batch element. Defaults to 1.
    :type total_count: Union[custom_types.Integer, torch.Tensor]
    :param logits: Event logits (log-odds)
    :type logits: Optional[torch.Tensor]
    :param validate_args: Whether to validate arguments. Defaults to None.
    :type validate_args: Optional[bool]

    The logit parameterization is advantageous because:
    - No normalization constraints (logits can be any real numbers)
    - Better numerical properties for optimization
    - Natural output of neural networks and linear models
    - Automatic normalization through softmax transformation

    Example:
        >>> # Logit parameterization
        >>> logits = torch.randn(3, 4)  # No normalization needed
        >>> total_counts = torch.tensor([[50], [75], [100]])
        >>> dist = MultinomialLogit(total_count=total_counts, logits=logits)
    """

    def __init__(
        self,
        total_count: "custom_types.Integer" | torch.Tensor = 1,
        logits: Optional[torch.Tensor] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        """Initialize logit-parameterized multinomial distribution.

        :param total_count: Total trials per batch element
        :param logits: Logit parameters (must be provided)
        :param validate_args: Validation flag
        """
        # Call the parent class with logits
        super().__init__(
            total_count=total_count, logits=logits, validate_args=validate_args
        )


class MultinomialLogTheta(MultinomialLogit):
    """Multinomial distribution with normalized log-probabilities.

    This class extends MultinomialLogit with the additional constraint that
    the input log-probabilities must already be normalized (i.e., their
    exponentials sum to 1). This is useful when working with log-probability
    vectors that are guaranteed to be valid probability distributions.

    :param total_count: Total number of trials for each batch element. Defaults to 1.
    :type total_count: Union[custom_types.Integer, torch.Tensor]
    :param log_probs: Normalized log-probabilities (exp(log_probs) must sum to 1)
    :type log_probs: Optional[torch.Tensor]
    :param validate_args: Whether to validate arguments. Defaults to None.
    :type validate_args: Optional[bool]

    :raises AssertionError: If log_probs is None
    :raises AssertionError: If log_probs are not properly normalized

    This parameterization is particularly useful when:
    - Working with log-space normalized probability vectors
    - Ensuring numerical precision in log-space computations
    - Interfacing with other log-space probability calculations

    The normalization constraint is enforced at initialization to prevent
    invalid probability distributions.

    Example:
        >>> # Normalized log-probabilities
        >>> logits = torch.randn(3, 4)
        >>> log_probs = torch.log_softmax(logits, dim=-1)
        >>> total_counts = torch.tensor([[100], [200], [150]])
        >>> dist = MultinomialLogTheta(total_count=total_counts, log_probs=log_probs)
    """

    def __init__(
        self,
        total_count: "custom_types.Integer" | torch.Tensor = 1,
        log_probs: Optional[torch.Tensor] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        """Initialize normalized log-probability multinomial distribution.

        :param total_count: Total trials per batch element
        :param log_probs: Normalized log-probability parameters
        :param validate_args: Validation flag

        Validates that log_probs are properly normalized before initialization.
        """
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


class Normal(dist.Normal):
    """Enhanced normal distribution with numerically stable log-space functions.

    This class extends PyTorch's standard Normal distribution with improved
    implementations of log-CDF and log-survival functions that provide better
    numerical stability, particularly in the tails of the distribution.

    The enhanced methods use PyTorch's special functions that are specifically
    designed for numerical stability in extreme value computations.

    Key Improvements:
    - Numerically stable log-CDF computation using log_ndtr
    - Stable log-survival function using symmetry properties
    - Maintains full compatibility with PyTorch's Normal interface
    - Better precision for extreme tail probabilities

    These improvements are particularly important for:
    - Extreme value analysis
    - Tail probability computations
    - Log-likelihood calculations with extreme parameter values

    Example:
        >>> # Enhanced normal distribution
        >>> normal = Normal(loc=0.0, scale=1.0)
        >>> # Stable computation of very small tail probabilities
        >>> extreme_value = torch.tensor(10.0)
        >>> log_tail_prob = normal.log_cdf(extreme_value)  # Numerically stable
    """

    # pylint: disable=not-callable, abstract-method
    def log_cdf(self, value: torch.Tensor) -> torch.Tensor:
        """Compute logarithm of cumulative distribution function.

        :param value: Values at which to evaluate log-CDF
        :type value: torch.Tensor

        :returns: Log-CDF values
        :rtype: torch.Tensor

        Uses PyTorch's special.log_ndtr function for numerical stability,
        which is specifically designed to handle extreme values without
        overflow or underflow issues.
        """
        return torch.special.log_ndtr((value - self.loc) / self.scale)

    def log_sf(self, value: torch.Tensor) -> torch.Tensor:
        """Compute logarithm of survival function (1 - CDF).

        :param value: Values at which to evaluate log-survival function
        :type value: torch.Tensor

        :returns: Log-survival function values
        :rtype: torch.Tensor

        Leverages the symmetry of the normal distribution to compute the
        survival function as the CDF evaluated at the reflection about the mean.
        This approach maintains numerical stability while avoiding direct
        computation of 1 - CDF.
        """
        # We take advantage of the symmetry of the normal distribution. The CDF
        # evaluated at the reflection about the mean will be the survival function.
        return torch.special.log_ndtr((self.loc - value) / self.scale)


class LogNormal(dist.LogNormal):
    """Enhanced log-normal distribution with numerically stable log-space functions.

    This class extends PyTorch's standard LogNormal distribution with improved
    implementations of log-CDF and log-survival functions for better numerical
    stability, particularly important given the log-normal's heavy tail behavior.

    Key Improvements:
    - Stable log-CDF computation using the underlying normal distribution
    - Numerically stable log-survival function
    - Maintains compatibility with PyTorch's LogNormal interface
    - Better handling of extreme values in both tails

    The log-normal distribution is particularly sensitive to numerical issues
    because of its relationship to the normal distribution through logarithmic
    transformation and its heavy-tailed nature.

    Example:
        >>> # Enhanced log-normal distribution
        >>> lognormal = LogNormal(loc=0.0, scale=1.0)
        >>> # Stable computation for extreme values
        >>> large_value = torch.tensor(1000.0)
        >>> log_tail_prob = lognormal.log_sf(large_value)  # Numerically stable
    """

    # pylint: disable=not-callable, abstract-method
    def log_cdf(self, value: torch.Tensor) -> torch.Tensor:
        """Compute logarithm of cumulative distribution function.

        :param value: Values at which to evaluate log-CDF
        :type value: torch.Tensor

        :returns: Log-CDF values
        :rtype: torch.Tensor

        Transforms the problem to the underlying normal distribution for
        stable computation using log_ndtr.
        """
        return torch.special.log_ndtr((torch.log(value) - self.loc) / self.scale)

    def log_sf(self, value: torch.Tensor) -> torch.Tensor:
        """Compute logarithm of survival function.

        :param value: Values at which to evaluate log-survival function
        :type value: torch.Tensor

        :returns: Log-survival function values
        :rtype: torch.Tensor

        Uses the relationship between log-normal and normal distributions
        to compute stable log-survival probabilities.
        """
        return torch.special.log_ndtr((self.loc - torch.log(value)) / self.scale)


# pylint: disable=abstract-method
class Lomax(dist.transformed_distribution.TransformedDistribution, CustomDistribution):
    """Lomax distribution implementation (shifted Pareto distribution).

    The Lomax distribution is a shifted version of the Pareto distribution,
    also known as the Pareto Type II distribution. It's implemented as a
    transformed Pareto distribution with an affine transformation.

    :param lambda_: Scale parameter (must be positive)
    :type lambda_: torch.Tensor
    :param alpha: Shape parameter (must be positive)
    :type alpha: torch.Tensor
    :param args: Additional arguments for the base distribution
    :param kwargs: Additional keyword arguments for the base distribution

    Mathematical Definition:
        If X ~ Pareto(scale=λ, shape=α), then Y = X - λ ~ Lomax(λ, α)

    Properties:
    - Support: [0, ∞)
    - Heavy-tailed distribution
    - Power-law behavior in the tail

    The distribution is implemented using PyTorch's TransformedDistribution
    framework with a Pareto base distribution and an affine transformation.

    Example:
        >>> # Lomax distribution for modeling heavy-tailed phenomena
        >>> lambda_param = torch.tensor(1.0)
        >>> alpha_param = torch.tensor(2.0)
        >>> lomax = Lomax(lambda_=lambda_param, alpha=alpha_param)
        >>> samples = lomax.sample((1000,))
    """

    def __init__(self, lambda_: torch.Tensor, alpha: torch.Tensor, *args, **kwargs):
        """Initialize Lomax distribution as transformed Pareto.

        :param lambda_: Scale parameter
        :param alpha: Shape parameter
        :param args: Additional base distribution arguments
        :param kwargs: Additional base distribution keyword arguments
        """
        # Build the base distribution and the transforms (just a shift in the output)
        base_dist = dist.Pareto(scale=lambda_, alpha=alpha)
        transforms = [dist.transforms.AffineTransform(loc=-lambda_, scale=1)]
        super().__init__(base_dist, transforms, *args, **kwargs)


class ExpLomax(
    dist.transformed_distribution.TransformedDistribution, CustomDistribution
):
    """Exponential-Lomax distribution implementation.

    This distribution is created by taking the logarithm of a Lomax-distributed
    random variable. It's useful for modeling log-scale phenomena that exhibit
    heavy-tailed behavior.

    :param lambda_: Scale parameter for the underlying Lomax distribution
    :type lambda_: torch.Tensor
    :param alpha: Shape parameter for the underlying Lomax distribution
    :type alpha: torch.Tensor
    :param args: Additional arguments for the base distribution
    :param kwargs: Additional keyword arguments for the base distribution

    Mathematical Definition:
        If X ~ Lomax(λ, α), then Y = log(X) ~ ExpLomax(λ, α)

    Properties:
    - Support: (-∞, ∞)
    - Heavy-tailed in log-space
    - Useful for log-scale modeling of power-law phenomena
    - Natural for multiplicative processes

    This distribution combines the heavy-tail properties of the Lomax
    distribution with the convenience of log-scale modeling.

    Example:
        >>> # Modeling log-scale heavy-tailed data
        >>> lambda_param = torch.tensor(1.0)
        >>> alpha_param = torch.tensor(2.0)
        >>> exp_lomax = ExpLomax(lambda_=lambda_param, alpha=alpha_param)
        >>> log_samples = exp_lomax.sample((1000,))
    """

    def __init__(self, lambda_: torch.Tensor, alpha: torch.Tensor, *args, **kwargs):
        """Initialize Exponential-Lomax distribution.

        :param lambda_: Scale parameter
        :param alpha: Shape parameter
        :param args: Additional arguments
        :param kwargs: Additional keyword arguments
        """
        # Build the base distribution (Lomax) and transforms (log)
        base_dist = Lomax(lambda_=lambda_, alpha=alpha)
        transforms = [dist.transforms.ExpTransform().inv]
        super().__init__(base_dist, transforms, *args, **kwargs)


class ExpExponential(
    dist.transformed_distribution.TransformedDistribution, CustomDistribution
):
    """Exponential-Exponential distribution implementation.

    This distribution is created by taking the logarithm of an exponentially
    distributed random variable. It's also known as the Gumbel distribution
    and is useful for extreme value modeling.

    :param rate: Rate parameter for the underlying exponential distribution
    :type rate: torch.Tensor
    :param args: Additional arguments for the base distribution
    :param kwargs: Additional keyword arguments for the base distribution

    Mathematical Definition:
        If X ~ Exponential(rate), then Y = log(X) ~ ExpExponential(rate)

    Properties:
    - Support: (-∞, ∞)
    - Type I extreme value distribution (Gumbel)
    - Useful for modeling minima of exponential random variables
    - Common in survival analysis and reliability engineering

    Example:
        >>> # Extreme value modeling
        >>> rate_param = torch.tensor(1.0)
        >>> exp_exp = ExpExponential(rate=rate_param)
        >>> extreme_values = exp_exp.sample((1000,))
    """

    def __init__(self, rate: torch.Tensor, *args, **kwargs):
        """Initialize Exponential-Exponential distribution.

        :param rate: Rate parameter for base exponential distribution
        :param args: Additional arguments
        :param kwargs: Additional keyword arguments
        """
        # Build the base distribution (Exponential) and transforms (log)
        base_dist = dist.Exponential(rate=rate)
        transforms = [dist.transforms.ExpTransform().inv]
        super().__init__(base_dist, transforms, *args, **kwargs)


class ExpDirichlet(
    dist.transformed_distribution.TransformedDistribution, CustomDistribution
):
    """Exponential-Dirichlet distribution implementation.

    This distribution is created by taking the element-wise logarithm of a
    Dirichlet-distributed random vector. It's useful for modeling log-scale
    compositional data and log-probability vectors.

    :param concentration: Concentration parameters for the underlying Dirichlet
    :type concentration: torch.Tensor
    :param args: Additional arguments for the base distribution
    :param kwargs: Additional keyword arguments for the base distribution

    Mathematical Definition:
        If X ~ Dirichlet(α), then Y = log(X) ~ ExpDirichlet(α)

    Properties:
    - Support: (-∞, 0]^K where K is the number of categories
    - Log-scale simplex (sum of exponentials equals 1)
    - Natural for log-probability modeling
    - Useful in Bayesian analysis of categorical data

    This distribution is particularly valuable when working with probability
    vectors in log-space, where it maintains the simplex constraint through
    the exponential transformation.

    Example:
        >>> # Log-probability vector modeling
        >>> concentration = torch.tensor([2.0, 3.0, 1.0])
        >>> exp_dirichlet = ExpDirichlet(concentration=concentration)
        >>> log_probs = exp_dirichlet.sample((100,))
        >>> # Verify simplex constraint: exp(log_probs).sum(dim=-1) ≈ 1
    """

    def __init__(self, concentration: torch.Tensor, *args, **kwargs):
        """Initialize Exponential-Dirichlet distribution.

        :param concentration: Concentration parameter vector
        :param args: Additional arguments
        :param kwargs: Additional keyword arguments
        """
        # Build the base distribution (Dirichlet) and transforms (log)
        base_dist = dist.Dirichlet(concentration=concentration)
        transforms = [dist.transforms.ExpTransform().inv]
        super().__init__(base_dist, transforms, *args, **kwargs)


# pylint: enable=abstract-method
