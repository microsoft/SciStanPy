"""Holds subclasses for transformed distributions."""

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


class CustomDirichlet(stats._multivariate.dirichlet_gen):  # pylint: disable=W0212
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

        # The trailing dimensions of the size must match the shape of the alphas
        # The last dimension is the number of categories. All others are the
        # batch dimensions
        batch_dims, trailing_dims = size[: -(alpha.ndim)], size[-(alpha.ndim) :]
        if trailing_dims != alpha.shape:
            raise ValueError(
                f"Trailing dimensions of the size ({size}) do not match the "
                f"shape of the alphas ({alpha.shape})"
            )

        # Reshape the alphas to be 2D. The last dimension is the number of
        # categories. All others are the batch dimensions.
        alphas = alpha.reshape(-1, alpha.shape[-1])

        # Sample from the Dirichlet distribution according to the batch dims. Note
        # that the axis of stacking varies depending on the number of batch dimensions.
        # This is because we want alphas to be the first dimension after the batch
        # dimensions -- we cannot assume that it is always the first dimensions as
        # we do in the above wrapped functions.
        return np.stack(
            [
                super().rvs(alpha, size=batch_dims, random_state=random_state)
                for alpha in alphas
            ],
            axis=len(batch_dims),
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
        # The dimensions of `n` must equal the leading dimensions of `p`
        if isinstance(n, np.ndarray) and n.shape != p.shape[:-1]:
            raise ValueError(
                f"Dimensions of `n` ({n.shape}) must equal the leading dimensions "
                f"of `p` ({p.shape[:-1]})"
            )

        # Set the size
        if size is None:
            size = p.shape
        elif isinstance(size, int):
            size = (size, *p.shape)
        else:
            size = tuple(size)

        # The last dimension of the size must match the shape of the p
        if size[-1] != p.shape[-1]:
            raise ValueError(
                f"Last dimension of the size ({size}) must match the shape of "
                f"the p ({p.shape[-1]})"
            )

        # Run the base distribution, ignoring the last dimension
        return super().rvs(n=n, p=p, size=size[:-1], random_state=random_state)


class MultinomialLogit(CustomMultinomial):
    """
    Identical to CustomMultinomial, but uses the logit of the probabilities instead
    of the probabilities themselves.
    """

    @staticmethod
    def softmax_p(function: Callable) -> Callable:
        """
        Decorator that transforms the probabilities to logits before calling the
        function.
        """

        @functools.wraps(function)
        def inner(**kwargs):

            # Apply the softmax transformation to the logits
            kwargs["p"] = special.softmax(kwargs.pop("logits"), axis=-1)

            return function(**kwargs)

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
        Decorator that transforms the probabilities to logits before calling the
        function.
        """

        @functools.wraps(function)
        def inner(**kwargs):

            # Exponentiate the log probabilities
            kwargs["p"] = np.exp(kwargs.pop("log_p"))

            return function(**kwargs)

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


class ScipyDistTransform(stats.rv_continuous, ABC):
    """Base class for transformed scipy distributions."""

    def __init__(self, base_dist: stats.rv_continuous, *args, **kwargs):
        """
        Records the distribution to be transformed before proceeding with the standard
        initialization.
        """
        self.base_dist = base_dist
        super().__init__(*args, **kwargs)

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

    def _pdf(self, x, *args, **kwargs):
        """Probability density function."""
        return self.base_dist.pdf(self.inverse_transform(x), *args, **kwargs) * np.exp(
            self.log_jacobian_correction(x)
        )

    def _logpdf(self, x, *args, **kwargs):
        """Logarithm of the probability density function."""
        return self.base_dist.logpdf(
            self.inverse_transform(x), *args, **kwargs
        ) + self.log_jacobian_correction(x)

    def _cdf(self, x, *args, **kwargs):
        """Cumulative distribution function."""
        return self.base_dist.cdf(self.inverse_transform(x), *args, **kwargs)

    def _ppf(self, q, *args, **kwargs):
        """Percent point function (inverse of cdf)."""
        return self.transform(self.base_dist.ppf(q, *args, **kwargs))

    def _sf(self, x, *args, **kwargs):
        """Survival function (1 - cdf)."""
        return self.base_dist.sf(self.inverse_transform(x), *args, **kwargs)

    def _isf(self, q, *args, **kwargs):
        """Inverse survival function."""
        return self.transform(self.base_dist.isf(q, *args, **kwargs))

    def _logsf(self, x, *args, **kwargs):
        """Log survival function."""
        return self.base_dist.logsf(self.inverse_transform(x), *args, **kwargs)


class LogUnivariateScipyTransform(ScipyDistTransform):
    """
    Transforms a univariate scipy distribution using the natural logarithm.
    """

    transform = np.log
    inverse_transform = np.exp

    def log_jacobian_correction(
        self, x: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """
        The log Jacobian correction for the transformation is simply the input value.
        """
        return x


dirichlet = CustomDirichlet()
expnormal = LogUnivariateScipyTransform(stats.norm, name="expnormal")
expexponential = LogUnivariateScipyTransform(stats.expon, name="expexponential")
explomax = LogUnivariateScipyTransform(stats.lomax, name="explomax")
expdirichlet = ExpDirichlet()
multinomial = CustomMultinomial()
multinomial_logit = MultinomialLogit()
multinomial_log_theta = MultinomialLogTheta()
