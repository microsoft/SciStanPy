# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Custom SciPy distribution implementations for SciStanPy models.

This module provides extended and custom SciPy distribution classes that address
limitations in the standard SciPy distribution library for probabilistic modeling.
These implementations enable advanced batch operations, support variable batch
dimensions, and provide transformed distributions not available in the standard library.

Key Features:
    - **Enhanced Batch Support**: Extended multivariate distributions with variable batch dimensions
    - **Custom Transformations**: Log-transformed distributions with proper Jacobian corrections
    - **Alternative Parameterizations**: Logit and log-probability parameterizations for
        multinomial distributions
    - **Numerical Stability**: Improved implementations for edge cases and extreme values

Distribution Categories:

**Enhanced Multivariate Distributions**: Extended batch support
    - CustomDirichlet: Dirichlet distribution with variable batch dimensions
    - CustomMultinomial: Multinomial distribution with flexible batch handling
    - MultinomialLogit: Logit-parameterized multinomial distribution
    - MultinomialLogTheta: Log-probability parameterized multinomial distribution

**Transformed Distributions**: Log-transformed variants
    - ExpDirichlet: Log-transformed Dirichlet distribution
    - LogUnivariateScipyTransform: General log-transformation framework
    - TransformedScipyDist: Abstract base for distribution transformations

**Pre-configured Distributions**: Ready-to-use distribution instances
    - dirichlet: CustomDirichlet instance
    - multinomial: CustomMultinomial instance
    - expexponential: Log-transformed exponential distribution
    - explomax: Log-transformed Lomax distribution
    - expdirichlet: ExpDirichlet instance

The distributions in this module are designed to work within SciPy's distribution
framework while providing enhanced functionality for advanced probabilistic modeling
scenarios commonly encountered in SciStanPy applications.
"""

from __future__ import annotations

import functools
import inspect

from abc import ABC, abstractmethod
from typing import Callable, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from scipy import special
from scipy import stats

if TYPE_CHECKING:
    from scistanpy import custom_types


def _combine_args_kwargs(function: Callable, args: tuple, kwargs: dict) -> dict:
    """Combine positional and keyword arguments into a single dictionary.

    :param function: Function whose signature determines parameter names
    :type function: Callable
    :param args: Positional arguments to the function
    :type args: tuple
    :param kwargs: Keyword arguments to the function
    :type kwargs: dict

    :returns: Combined arguments as a dictionary
    :rtype: dict

    :raises ValueError: If total arguments don't match function signature

    This utility function inspects the function signature and maps positional
    arguments to their corresponding parameter names, then merges them with
    the provided keyword arguments.
    """
    # We will need the function signature to determine the arg and kwarg names
    signature = inspect.signature(function)
    paramnames = list(signature.parameters.keys())

    # Make sure that the number of args and kwargs matches the number of parameters
    if len(args) + len(kwargs) != len(paramnames):
        raise ValueError(
            f"Expected {len(paramnames)} arguments, but got {len(args) + len(kwargs)}."
        )

    # Combine args and kwargs
    combined_kwargs = dict(zip(paramnames, args))
    combined_kwargs.update(kwargs)

    return combined_kwargs


class CustomDirichlet(
    stats._multivariate.dirichlet_gen  # pylint: disable=protected-access
):
    """Enhanced Dirichlet distribution supporting variable batch dimensions.

    This class extends SciPy's standard Dirichlet distribution to support
    arbitrary batch dimensions while maintaining compatibility with the
    SciPy distribution interface. The standard SciPy implementation has
    limitations with batch operations that this class addresses.

    Key Enhancements:
    - Support for arbitrary batch dimensions in alpha parameters
    - Proper broadcasting behavior across batch dimensions
    - Consistent output shapes for all distribution methods
    - Efficient vectorized operations over batch elements

    The implementation uses a decorator pattern to extend existing SciPy
    methods with batch dimension handling while preserving the original
    mathematical properties of the Dirichlet distribution.

    Mathematical Properties:
    - Support: Probability simplex {x : Σxᵢ = 1, xᵢ ≥ 0}
    - Parameters: α = (α₁, α₂, ..., αₖ) where αᵢ > 0
    - Mean: E[Xᵢ] = αᵢ / Σⱼ αⱼ
    - Useful for modeling categorical probabilities and compositional data

    Example:
        >>> # Standard usage with batch dimensions
        >>> alpha = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
        >>> dirichlet = CustomDirichlet()
        >>> samples = dirichlet.rvs(alpha, size=100) # output_shape = (100, 3, 3)
        >>> log_probs = dirichlet.logpdf(samples, alpha)
    """

    @staticmethod
    def _expand_batch(function: Callable, expect_x: bool = False) -> Callable:
        """Decorator for adding batch dimension support to Dirichlet methods.

        :param function: SciPy Dirichlet method to wrap
        :type function: Callable
        :param expect_x: Whether the function expects an 'x' parameter. Defaults to False.
        :type expect_x: bool

        :returns: Wrapped function with batch dimension support
        :rtype: Callable

        :raises ValueError: If expected parameters are missing or unexpected

        This decorator automatically handles:
        - Parameter validation and broadcasting
        - Reshaping for batch operations
        - Vectorized computation across batch elements
        - Proper output shape reconstruction

        The decorator distinguishes between functions that operate on data (expect_x=True)
        and those that only use distribution parameters (expect_x=False).
        """

        @functools.wraps(function)
        def inner(*args, **kwargs):

            # Combine args and kwargs
            combined_kwargs = _combine_args_kwargs(function, args, kwargs)

            # Check for 'x'
            if expect_x and "x" not in combined_kwargs:
                raise ValueError("Expected 'x' parameter in the function signature.")
            elif not expect_x and "x" in combined_kwargs:
                raise ValueError("Unexpected 'x' parameter in the function signature.")

            # Get the alpha parameter and the x, if present
            param_kwargs = {"alpha": np.asarray(combined_kwargs.pop("alpha"))}
            if expect_x:
                param_kwargs["x"] = np.asarray(combined_kwargs.pop("x"))

            # Broadcast the arrays in the param kwargs and reshape to be 2D
            broadcasted_shape = np.broadcast_shapes(
                *[v.shape for v in param_kwargs.values()]
            )
            param_kwargs = {
                k: np.broadcast_to(v, broadcasted_shape).reshape(-1, v.shape[-1])
                for k, v in param_kwargs.items()
            }

            # Run the function and combine the results
            res = np.concatenate(
                [
                    function(
                        **{k: v[i] for k, v in param_kwargs.items()}, **combined_kwargs
                    )
                    for i in range(len(param_kwargs["alpha"]))
                ]
            )

            # Some sanity checks on the result
            assert (
                res.ndim == 1
            ), f"Expected result to be 1D, but got {res.ndim}D with shape {res.shape}."
            assert len(res) == len(
                param_kwargs["alpha"]
            ), f"Expected result length {len(param_kwargs['alpha'])}, but got {len(res)}."

            # Always applied to a reduction, so we reshape the result to one less
            # dimension than the broadcasted shape
            return res.reshape(*broadcasted_shape[:-1])

        return inner

    # Enhanced method implementations with batch support
    # pylint: disable=protected-access
    logpdf = _expand_batch(stats._multivariate.dirichlet_gen.logpdf, expect_x=True)
    pdf = _expand_batch(stats._multivariate.dirichlet_gen.pdf, expect_x=True)
    mean = _expand_batch(stats._multivariate.dirichlet_gen.mean)
    var = _expand_batch(stats._multivariate.dirichlet_gen.var)
    cov = _expand_batch(stats._multivariate.dirichlet_gen.cov)
    entropy = _expand_batch(stats._multivariate.dirichlet_gen.entropy)
    # pylint: enable=protected-access

    def rvs(
        self,
        alpha: npt.NDArray[np.floating],
        size: tuple["custom_types.Integer", ...] | "custom_types.Integer" | None = 1,
        random_state: "custom_types.Integer" | np.random.Generator | None = None,
    ) -> npt.NDArray[np.floating]:
        """Generate random samples from the Dirichlet distribution.

        :param alpha: Concentration parameters with shape (..., k)
        :type alpha: npt.NDArray[np.floating]
        :param size: Output shape. Defaults to 1.
        :type size: Union[tuple[custom_types.Integer, ...], custom_types.Integer, None]
        :param random_state: Random state for reproducible sampling. Defaults to None.
        :type random_state: Union[custom_types.Integer, np.random.Generator, None]

        :returns: Random samples from Dirichlet distribution
        :rtype: npt.NDArray[np.floating]

        :raises ValueError: If alpha cannot be broadcast to the specified size

        This method supports arbitrary batch dimensions in the alpha parameter
        and properly broadcasts to the requested output size while maintaining
        the simplex constraint for each sample.
        """
        # Set the size
        if size is None:
            size = alpha.shape
        elif isinstance(size, int):
            size = (size, *alpha.shape)
        else:
            size = tuple(size)

        # Broadcast alpha to the given size.
        try:
            alpha = np.broadcast_to(alpha, size)
        except ValueError as err:
            raise ValueError(
                f"Cannot broadcast alpha ({alpha.shape}) to size ({size})"
            ) from err

        # Now that alphas have been broadcasted to the correct size, we can proceed
        # by sampling just once from the Dirichlet distribution for each alpha.
        # Reshaping at the end will reconstruct the original dimensions.
        return np.stack(
            [
                super().rvs(arr, random_state=random_state)
                for arr in alpha.reshape(-1, alpha.shape[-1])
            ]
        ).reshape(size)


class CustomMultinomial(stats._multivariate.multinomial_gen):  # pylint: disable=W0212
    """Enhanced multinomial distribution supporting variable batch dimensions.

    This class extends SciPy's standard multinomial distribution to support
    arbitrary batch dimensions in both the trial count (n) and probability
    parameters (p), enabling flexible batch operations for discrete multivariate
    modeling scenarios.

    Key Enhancements:
    - Variable batch dimensions for n and p parameters
    - Proper broadcasting behavior between n and p
    - Support for different trial counts across batch elements
    - Consistent output shapes for sampling operations

    Mathematical Properties:
    - Support: {x ∈ ℕ₀ᵏ : Σxᵢ = n}
    - Parameters: n (trial count), p = (p₁, ..., pₖ) where Σpᵢ = 1
    - PMF: P(X = x) = n! / (∏xᵢ!) × ∏pᵢˣⁱ
    - Useful for modeling categorical count data

    Example:
        >>> # Batch multinomial with different trial counts
        >>> n = np.array([[10], [20], [15]])
        >>> p = np.array([[0.3, 0.4, 0.3],
        ...               [0.2, 0.5, 0.3],
        ...               [0.4, 0.3, 0.3]])
        >>> multinomial = CustomMultinomial()
        >>> samples = multinomial.rvs(n=n, p=p, size=100) # shape = (100, 3, 3)
    """

    def rvs(
        self,
        n: "custom_types.Integer" | npt.NDArray[np.integer],
        p: npt.NDArray[np.floating],
        size: tuple["custom_types.Integer", ...] | "custom_types.Integer" | None = 1,
        random_state: "custom_types.Integer" | np.random.Generator | None = None,
    ) -> npt.NDArray[np.integer]:
        """Generate random samples from the multinomial distribution.

        :param n: Number of trials (can be scalar or array)
        :type n: Union[custom_types.Integer, npt.NDArray[np.integer]]
        :param p: Event probabilities with shape (..., k)
        :type p: npt.NDArray[np.floating]
        :param size: Output shape. Defaults to 1.
        :type size: Union[tuple[custom_types.Integer, ...], custom_types.Integer, None]
        :param random_state: Random state for reproducible sampling. Defaults to None.
        :type random_state: Union[custom_types.Integer, np.random.Generator, None]

        :returns: Random samples from multinomial distribution
        :rtype: npt.NDArray[np.integer]

        :raises ValueError: If n and p cannot be broadcast to compatible shapes

        This method supports different trial counts for each batch element
        and handles broadcasting between scalar/array n and multi-dimensional p.
        """

        def try_broadcast(x, target_size):
            """Attempts to broadcast and raises an error if not possible"""
            try:
                return np.broadcast_to(x, target_size)
            except ValueError as err:
                raise ValueError(
                    f"Cannot broadcast shape {x.shape} to {target_size}"
                ) from err

        # Set the size of p
        if size is None:
            p_size = p.shape
        elif isinstance(size, int):
            p_size = (size, *p.shape)
        else:
            p_size = tuple(size)

        # Set the size of n
        n_size = list(p_size)
        n_size[-1] = 1

        # n and p must be broadcastable to their respective sizes
        n = try_broadcast(n, n_size)
        p = try_broadcast(p, p_size)

        # Reshape to 2D
        n = n.reshape(-1, 1)
        p = p.reshape(-1, p_size[-1])
        assert len(n) == len(p)

        # Take the random samples. We take 1 for each n-p pair. Reshape to the
        # target size
        return np.stack(
            [
                super().rvs(n=n_el, p=p_el, random_state=random_state)
                for n_el, p_el in zip(n, p)
            ]
        ).reshape(size)


class MultinomialLogit(CustomMultinomial):
    """Multinomial distribution with logit parameterization.

    This class provides a logit-parameterized interface to the multinomial
    distribution, where probabilities are specified as logits (log-odds)
    rather than normalized probabilities. This parameterization is often
    more convenient for modeling and optimization.

    :ivar softmax_p: Decorator that transforms logits to probabilities

    Mathematical Properties:
    - Parameterization: logits ∈ ℝᵏ (unconstrained)
    - Transformation: pᵢ = exp(logitᵢ) / Σⱼ exp(logitⱼ)
    - Natural for neural network outputs and linear models
    - Automatic normalization ensures valid probabilities

    The logit parameterization offers several advantages:
    - No normalization constraints on input parameters
    - Better numerical properties for optimization
    - Natural output space for many machine learning models
    - Automatic softmax transformation to valid probabilities

    Example:
        >>> # Logit parameterization (no normalization needed)
        >>> logits = np.random.randn(3, 4)
        >>> n = np.array([[50], [75], [100]])
        >>> multinomial_logit = MultinomialLogit()
        >>> samples = multinomial_logit.rvs(n=n, logits=logits)
    """

    @staticmethod
    def softmax_p(function: Callable) -> Callable:
        """Decorator that transforms logits to probabilities using softmax.

        :param function: Function to wrap with logit transformation
        :type function: Callable

        :returns: Wrapped function that accepts logits instead of probabilities
        :rtype: Callable

        This decorator automatically applies the softmax transformation to
        convert logits to valid probabilities before calling the underlying
        multinomial distribution methods.
        """

        @functools.wraps(function)
        def inner(self, **kwargs):
            # Apply the softmax transformation to the logits
            kwargs["p"] = special.softmax(kwargs.pop("logits"), axis=-1)
            return function(self, **kwargs)

        return inner

    # Wrapped methods with logit transformation
    pmf = softmax_p(CustomMultinomial.pmf)
    logpmf = softmax_p(CustomMultinomial.logpmf)
    rvs = softmax_p(CustomMultinomial.rvs)
    entropy = softmax_p(CustomMultinomial.entropy)
    cov = softmax_p(CustomMultinomial.cov)


class MultinomialLogTheta(CustomMultinomial):
    """Multinomial distribution with normalized log-probability parameterization.

    This class provides a log-probability parameterized interface where the
    input parameters are logarithms of probabilities that must already be
    normalized (i.e., their exponentials sum to 1). This is useful when
    working with log-probability vectors from other computations.

    :ivar exp_p: Decorator that transforms log-probabilities to probabilities

    Mathematical Properties:
    - Parameterization: log_p ∈ ℝᵏ with Σexp(log_pᵢ) = 1

    The log-probability parameterization is particularly useful when:
    - Working with log-normalized probability vectors
    - Interfacing with other log-space computations
    - Ensuring consistency with log-space model components

    Example:
        >>> # Normalized log-probabilities
        >>> logits = np.random.randn(3, 4)
        >>> log_probs = logits - scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        >>> n = np.array([[100], [200], [150]])
        >>> multinomial_log_theta = MultinomialLogTheta()
        >>> samples = multinomial_log_theta.rvs(n=n, log_p=log_probs)
    """

    @staticmethod
    def exp_p(function: Callable) -> Callable:
        """Decorator that transforms log-probabilities to probabilities.

        :param function: Function to wrap with log-probability transformation
        :type function: Callable

        :returns: Wrapped function that accepts log_p instead of probabilities
        :rtype: Callable

        :raises ValueError: If log-probabilities are not properly normalized

        This decorator validates that the exponentials of log-probabilities
        sum to 1 (within tolerance) and applies the exponential transformation
        to convert to valid probabilities.
        """

        @functools.wraps(function)
        def inner(self, **kwargs):
            # Exponentiate the log probabilities
            p = np.exp(kwargs.pop("log_p"))

            # The rows of `p` must sum to 1 within a threshold
            p_sum = p.sum(axis=-1, keepdims=True)
            if not np.allclose(p_sum, 1, atol=1e-6):
                raise ValueError(f"Rows of `p` must sum to 1, but got {p_sum.max()}")

            # Ensure total normalization
            kwargs["p"] = p / p_sum

            return function(self, **kwargs)

        return inner

    # Wrapped methods with log-probability transformation
    pmf = exp_p(CustomMultinomial.pmf)
    logpmf = exp_p(CustomMultinomial.logpmf)
    rvs = exp_p(CustomMultinomial.rvs)
    entropy = exp_p(CustomMultinomial.entropy)
    cov = exp_p(CustomMultinomial.cov)


class ExpDirichlet(CustomDirichlet):
    """Log-transformed Dirichlet distribution (Exponential-Dirichlet).

    This class implements a distribution where the logarithm of a Dirichlet-distributed
    random vector follows this distribution. It's useful for modeling log-scale
    compositional data and log-probability vectors with proper Jacobian corrections.

    Mathematical Properties:
    - If X ~ Dirichlet(α), then Y = log(X) ~ ExpDirichlet(α)
    - Support: (-∞, 0]ᵏ (log-simplex)
    - Constraint: Σexp(yᵢ) = 1
    - Natural for log-probability modeling

    The log-transformation requires proper Jacobian correction for probability
    calculations. This distribution is particularly valuable for:
    - Log-space compositional data analysis
    - Bayesian modeling of log-probability vectors
    - Numerical stability in extreme probability regimes
    - Integration with other log-space model components

    Example:
        >>> # Log-compositional data modeling
        >>> alpha = np.array([2.0, 3.0, 1.0])
        >>> exp_dirichlet = ExpDirichlet()
        >>> log_samples = exp_dirichlet.rvs(alpha, size=(1000,))
        >>> # Verify constraint: np.exp(log_samples).sum(axis=-1) ≈ 1
    """

    def logpdf(self, x, alpha):
        """Compute log probability density with Jacobian correction.

        :param x: Log-probability values
        :param alpha: Concentration parameters

        :returns: Log probability density values

        The implementation includes the proper Jacobian correction for the
        log-transformation, computed analytically for efficiency and numerical stability.
        """
        # pylint: disable=no-member
        return (
            np.sum(x * alpha, axis=-1)
            - x[..., -1]
            + special.gammaln(np.sum(alpha, axis=-1))
            - np.sum(special.gammaln(alpha), axis=-1)
        )

    def pdf(self, x, alpha):
        """Compute probability density function.

        :param x: Log-probability values
        :param alpha: Concentration parameters

        :returns: Probability density values

        Computed as the exponential of the log probability density for
        numerical stability and consistency.
        """
        return np.exp(self.logpdf(x, alpha))

    def rvs(
        self,
        alpha: npt.NDArray[np.floating],
        size: tuple["custom_types.Integer", ...] | "custom_types.Integer" | None = 1,
        random_state: "custom_types.Integer" | np.random.Generator | None = None,
    ) -> npt.NDArray[np.floating]:
        """Generate random samples from the log-transformed Dirichlet distribution.

        :param alpha: Concentration parameters
        :type alpha: npt.NDArray[np.floating]
        :param size: Output shape. Defaults to 1.
        :type size: Union[tuple[custom_types.Integer, ...], custom_types.Integer, None]
        :param random_state: Random state. Defaults to None.
        :type random_state: Union[custom_types.Integer, np.random.Generator, None]

        :returns: Log-probability samples
        :rtype: npt.NDArray[np.floating]

        Samples are generated by first sampling from the standard Dirichlet
        distribution and then applying the logarithmic transformation.
        """
        # Sample from the Dirichlet distribution and then take the logarithm
        return np.log(super().rvs(alpha, size=size, random_state=random_state))

    def mean(self, alpha):
        """Raise error for undefined moments.

        :raises NotImplementedError: Mean is not analytically available

        The moments of the log-transformed Dirichlet distribution do not
        have simple closed-form expressions.
        """
        raise NotImplementedError("Not defined for this custom distribution")

    def var(self, alpha):
        """Raise error for undefined moments.

        :raises NotImplementedError: Variance is not analytically available
        """
        raise NotImplementedError("Not defined for this custom distribution")

    def cov(self, alpha):
        """Raise error for undefined moments.

        :raises NotImplementedError: Covariance is not analytically available
        """
        raise NotImplementedError("Not defined for this custom distribution")

    def entropy(self, alpha):
        """Raise error for undefined entropy.

        :raises NotImplementedError: Entropy is not analytically available
        """
        raise NotImplementedError("Not defined for this custom distribution")


class TransformedScipyDist(ABC):
    """Abstract base class for transformed SciPy distributions.

    This class provides a framework for creating distributions that are
    transformations of existing SciPy distributions. It handles the
    mathematical details of transformation including Jacobian corrections
    for probability density functions.

    :param base_dist: Base SciPy distribution to transform
    :type base_dist: stats.rv_continuous

    Key Features:
    - Automatic Jacobian correction for probability densities
    - Proper transformation of all distribution methods
    - Maintains SciPy distribution interface compatibility
    - Support for arbitrary invertible transformations

    Subclasses must implement:
    - transform: Forward transformation function
    - inverse_transform: Inverse transformation function
    - log_jacobian_correction: Log determinant of Jacobian matrix

    The framework automatically handles:
    - PDF/log-PDF with Jacobian corrections
    - CDF through inverse transformation
    - Quantile functions through forward transformation
    - Random sampling through transformation of base samples
    """

    def __init__(self, base_dist: stats.rv_continuous):
        """Initialize transformed distribution with base distribution.

        :param base_dist: Base distribution to transform
        :type base_dist: stats.rv_continuous

        Records the base distribution for use in transformation operations.
        """
        self.base_dist = base_dist

    @abstractmethod
    def transform(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Apply forward transformation to input values.

        :param x: Input values from base distribution
        :type x: npt.NDArray[np.floating]

        :returns: Transformed values
        :rtype: npt.NDArray[np.floating]

        This method must implement the forward transformation function
        that maps from the base distribution support to the transformed
        distribution support.
        """

    @abstractmethod
    def inverse_transform(
        self, x: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Apply inverse transformation to input values.

        :param x: Input values from transformed distribution
        :type x: npt.NDArray[np.floating]

        :returns: Values in base distribution space
        :rtype: npt.NDArray[np.floating]

        This method must implement the inverse transformation function
        that maps from the transformed distribution support back to the
        base distribution support.
        """

    @abstractmethod
    def log_jacobian_correction(
        self,
        x: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute log Jacobian correction for the transformation.

        :param x: Input values in transformed space
        :type x: npt.NDArray[np.floating]

        :returns: Log Jacobian determinant values
        :rtype: npt.NDArray[np.floating]

        This method must compute the logarithm of the absolute value of
        the Jacobian determinant for the transformation, which is required
        for proper probability density calculations.
        """

    def pdf(self, x, *args, **kwargs):
        """Compute probability density function with Jacobian correction.

        :param x: Values at which to evaluate PDF
        :param args: Arguments for base distribution
        :param kwargs: Keyword arguments for base distribution

        :returns: Probability density values

        Applies the change of variables formula with proper Jacobian correction.
        """
        return self.base_dist.pdf(self.inverse_transform(x), *args, **kwargs) * np.exp(
            self.log_jacobian_correction(x)
        )

    def logpdf(self, x, *args, **kwargs):
        """Compute log probability density function with Jacobian correction.

        :param x: Values at which to evaluate log-PDF
        :param args: Arguments for base distribution
        :param kwargs: Keyword arguments for base distribution

        :returns: Log probability density values

        More numerically stable than computing log of PDF directly.
        """
        return self.base_dist.logpdf(
            self.inverse_transform(x), *args, **kwargs
        ) + self.log_jacobian_correction(x)

    def cdf(self, x, *args, **kwargs):
        """Compute cumulative distribution function.

        :param x: Values at which to evaluate CDF
        :param args: Arguments for base distribution
        :param kwargs: Keyword arguments for base distribution

        :returns: Cumulative probability values

        Uses inverse transformation to map to base distribution space.
        """
        return self.base_dist.cdf(self.inverse_transform(x), *args, **kwargs)

    def ppf(self, q, *args, **kwargs):
        """Compute percent point function (inverse CDF).

        :param q: Probability values
        :param args: Arguments for base distribution
        :param kwargs: Keyword arguments for base distribution

        :returns: Quantile values

        Uses forward transformation of base distribution quantiles.
        """
        return self.transform(self.base_dist.ppf(q, *args, **kwargs))

    def sf(self, x, *args, **kwargs):
        """Compute survival function (1 - CDF).

        :param x: Values at which to evaluate survival function
        :param args: Arguments for base distribution
        :param kwargs: Keyword arguments for base distribution

        :returns: Survival probability values
        """
        return self.base_dist.sf(self.inverse_transform(x), *args, **kwargs)

    def isf(self, q, *args, **kwargs):
        """Compute inverse survival function.

        :param q: Probability values
        :param args: Arguments for base distribution
        :param kwargs: Keyword arguments for base distribution

        :returns: Inverse survival function values
        """
        return self.transform(self.base_dist.isf(q, *args, **kwargs))

    def logsf(self, x, *args, **kwargs):
        """Compute log survival function.

        :param x: Values at which to evaluate log survival function
        :param args: Arguments for base distribution
        :param kwargs: Keyword arguments for base distribution

        :returns: Log survival probability values

        More numerically stable for small survival probabilities.
        """
        return self.base_dist.logsf(self.inverse_transform(x), *args, **kwargs)

    def rvs(self, *args, **kwargs):
        """Generate random samples from transformed distribution.

        :param args: Arguments for base distribution
        :param kwargs: Keyword arguments for base distribution

        :returns: Random samples from transformed distribution

        Generates samples from base distribution and applies transformation.
        """
        return self.transform(self.base_dist.rvs(*args, **kwargs))


class LogUnivariateScipyTransform(TransformedScipyDist):
    """Log transformation for univariate SciPy distributions.

    This class implements the natural logarithm transformation for any
    univariate SciPy distribution, creating a log-transformed variant
    with proper Jacobian corrections.

    This transformation is commonly used to:
    - Convert positive-valued distributions to real-valued distributions
    - Enable log-scale modeling of multiplicative processes
    - Improve numerical stability for heavy-tailed distributions
    - Create log-normal variants of arbitrary positive distributions

    Example:
        >>> # Create log-transformed exponential distribution
        >>> log_exponential = LogUnivariateScipyTransform(stats.expon)
        >>> # This is equivalent to a Gumbel distribution
        >>> samples = log_exponential.rvs(scale=1.0, size=1000)
    """

    transform = np.log
    inverse_transform = np.exp

    def log_jacobian_correction(
        self, x: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Compute log Jacobian correction for logarithmic transformation.

        :param x: Values in transformed (log) space
        :type x: npt.NDArray[np.floating]

        :returns: Log Jacobian determinant (equal to x for log transform)
        :rtype: npt.NDArray[np.floating]
        """
        return x


# Pre-configured distribution instances for convenient use
dirichlet = CustomDirichlet()
expexponential = LogUnivariateScipyTransform(stats.expon)
explomax = LogUnivariateScipyTransform(stats.lomax)
expdirichlet = ExpDirichlet()
multinomial = CustomMultinomial()
multinomial_logit = MultinomialLogit()
multinomial_log_theta = MultinomialLogTheta()
