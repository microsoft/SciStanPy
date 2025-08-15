"""Holds subclasses for transformed distributions."""

from __future__ import annotations

import functools
import inspect

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import numpy.typing as npt

from scipy import special
from scipy import stats


def _combine_args_kwargs(function: Callable, args: tuple, kwargs: dict) -> dict:
    """
    Combines positional and keyword arguments into a single dictionary for a function
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
    """
    Subclass of scipy's dirichlet distribution to support variable numbers of
    batch dimensions.
    """

    @staticmethod
    def _expand_batch(function: Callable, expect_x: bool = False) -> Callable:
        """
        Decorator that takes allows variable batch dimensions in the scipy Dirichlet
        distribution functions.
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

    # pylint: disable=W0212
    logpdf = _expand_batch(stats._multivariate.dirichlet_gen.logpdf, expect_x=True)
    pdf = _expand_batch(stats._multivariate.dirichlet_gen.pdf, expect_x=True)
    mean = _expand_batch(stats._multivariate.dirichlet_gen.mean)
    var = _expand_batch(stats._multivariate.dirichlet_gen.var)
    cov = _expand_batch(stats._multivariate.dirichlet_gen.cov)
    entropy = _expand_batch(stats._multivariate.dirichlet_gen.entropy)
    # pylint: enable=W0212

    def rvs(
        self,
        alpha: npt.NDArray[np.floating],
        size: tuple[int, ...] | int | None = 1,
        random_state: int | np.random.Generator | None = None,
    ) -> npt.NDArray[np.floating]:

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
    """
    Custom subclass of scipy's multinomial distribution to support variable numbers
    of batch dimensions.
    """

    def rvs(
        self,
        n: int | npt.NDArray[np.integer],
        p: npt.NDArray[np.floating],
        size: tuple[int, ...] | int | None = 1,
        random_state: int | np.random.Generator | None = None,
    ) -> npt.NDArray[np.integer]:
        """
        Generates random samples from the multinomial distribution with variable batch
        dimensions.
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
    """
    Identical to CustomMultinomial, but uses the logit of the probabilities instead
    of the probabilities themselves.
    """

    @staticmethod
    def softmax_p(function: Callable) -> Callable:
        """
        Decorator that transforms logits to probabilities before calling the function.
        """

        @functools.wraps(function)
        def inner(self, **kwargs):
            # Apply the softmax transformation to the logits
            kwargs["p"] = special.softmax(kwargs.pop("logits"), axis=-1)
            return function(self, **kwargs)

        return inner

    pmf = softmax_p(CustomMultinomial.pmf)
    logpmf = softmax_p(CustomMultinomial.logpmf)
    rvs = softmax_p(CustomMultinomial.rvs)
    entropy = softmax_p(CustomMultinomial.entropy)
    cov = softmax_p(CustomMultinomial.cov)


class MultinomialLogTheta(CustomMultinomial):
    """
    Identical to CustomMultinomial, but uses the log of the probabilities instead
    of the probabilities themselves.
    """

    @staticmethod
    def exp_p(function: Callable) -> Callable:
        """
        Decorator that transforms the log probabilities to probabilities before calling the
        function.
        """

        @functools.wraps(function)
        def inner(self, **kwargs):
            # Exponentiate the log probabilities
            kwargs["p"] = np.exp(kwargs.pop("log_p"))
            return function(self, **kwargs)

        return inner

    pmf = exp_p(CustomMultinomial.pmf)
    logpmf = exp_p(CustomMultinomial.logpmf)
    rvs = exp_p(CustomMultinomial.rvs)
    entropy = exp_p(CustomMultinomial.entropy)
    cov = exp_p(CustomMultinomial.cov)


class ExpDirichlet(CustomDirichlet):
    """
    Sublcass of the CustomDirichlet distribution that describes a random variable
    whose exponential is Dirichlet-distributed (i.e., a log-transformed Dirichlet
    distribution).
    """

    def logpdf(self, x, alpha):
        """
        Computes the log probability density function, taking into account the Jacobian
        correction for the transformation.
        """
        # pylint: disable=no-member
        return (
            np.sum(x * alpha, axis=-1)
            - x[..., -1]
            + special.gammaln(np.sum(alpha, axis=-1))
            - np.sum(special.gammaln(alpha), axis=-1)
        )

    def pdf(self, x, alpha):
        """
        Computes the probability density function, taking into account the Jacobian
        correction for the transformation.
        """
        return np.exp(self.logpdf(x, alpha))

    def rvs(
        self,
        alpha: npt.NDArray[np.floating],
        size: tuple[int, ...] | int | None = 1,
        random_state: int | np.random.Generator | None = None,
    ) -> npt.NDArray[np.floating]:
        """
        Generates random samples from the ExpDirichlet distribution.
        """
        # Sample from the Dirichlet distribution and then take the logarithm
        return np.log(super().rvs(alpha, size=size, random_state=random_state))

    def mean(self, alpha):
        raise NotImplementedError("Not defined for this custom distribution")

    def var(self, alpha):
        raise NotImplementedError("Not defined for this custom distribution")

    def cov(self, alpha):
        raise NotImplementedError("Not defined for this custom distribution")

    def entropy(self, alpha):
        raise NotImplementedError("Not defined for this custom distribution")


class TransformedScipyDist(ABC):
    """Base class for transformed scipy distributions."""

    def __init__(self, base_dist: stats.rv_continuous):
        """
        Records the distribution to be transformed before proceeding with the standard
        initialization.
        """
        self.base_dist = base_dist

    @abstractmethod
    def transform(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Returns the transformation function."""

    @abstractmethod
    def inverse_transform(
        self, x: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Returns the inverse transformation function."""

    @abstractmethod
    def log_jacobian_correction(
        self,
        x: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Returns the log Jacobian correction for the transformation."""

    def pdf(self, x, *args, **kwargs):
        """Probability density function."""
        return self.base_dist.pdf(self.inverse_transform(x), *args, **kwargs) * np.exp(
            self.log_jacobian_correction(x)
        )

    def logpdf(self, x, *args, **kwargs):
        """Logarithm of the probability density function."""
        return self.base_dist.logpdf(
            self.inverse_transform(x), *args, **kwargs
        ) + self.log_jacobian_correction(x)

    def cdf(self, x, *args, **kwargs):
        """Cumulative distribution function."""
        return self.base_dist.cdf(self.inverse_transform(x), *args, **kwargs)

    def ppf(self, q, *args, **kwargs):
        """Percent point function (inverse of cdf)."""
        return self.transform(self.base_dist.ppf(q, *args, **kwargs))

    def sf(self, x, *args, **kwargs):
        """Survival function (1 - cdf)."""
        return self.base_dist.sf(self.inverse_transform(x), *args, **kwargs)

    def isf(self, q, *args, **kwargs):
        """Inverse survival function."""
        return self.transform(self.base_dist.isf(q, *args, **kwargs))

    def logsf(self, x, *args, **kwargs):
        """Log survival function."""
        return self.base_dist.logsf(self.inverse_transform(x), *args, **kwargs)

    def rvs(self, *args, **kwargs):
        """Random variates."""
        return self.transform(self.base_dist.rvs(*args, **kwargs))


class LogUnivariateScipyTransform(TransformedScipyDist):
    """
    Transforms a univariate scipy distribution using the natural logarithm.
    """

    transform = np.log
    inverse_transform = np.exp

    def log_jacobian_correction(
        self, x: npt.NDArray[np.floating]  # pylint: disable=unused-argument
    ) -> npt.NDArray[np.floating]:
        """
        The log Jacobian correction for the transformation is simply the input value.
        """
        return x


dirichlet = CustomDirichlet()
expexponential = LogUnivariateScipyTransform(stats.expon)
explomax = LogUnivariateScipyTransform(stats.lomax)
expdirichlet = ExpDirichlet()
multinomial = CustomMultinomial()
multinomial_logit = MultinomialLogit()
multinomial_log_theta = MultinomialLogTheta()
