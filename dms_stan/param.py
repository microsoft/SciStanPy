"""Holds classes that can be used for defining models in DMS Stan models."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

import numpy as np
import numpy.typing as npt
import scipy.special as sp

import dms_stan as dms
import dms_stan.constant as dmsc


class AbstractParameter(ABC):
    """Template class for parameters used in DMS Stan models."""

    def __init__(  # pylint: disable=unused-argument
        self,
        *,
        shape: tuple[int, ...] = tuple(),
        **parameters: "CombinableParameterType",
    ):
        """Builds a parameter instance with the given shape."""
        # Not observable by default
        self.observable: bool = False

        # There must be at least one parameter
        if len(parameters) < 1:
            raise ValueError("At least one parameter must be passed to the class")

        # No incoming parameters can be observables
        if any(
            isinstance(param, AbstractParameter) and param.observable
            for param in parameters.values()
        ):
            raise ValueError("Parent parameters cannot be observables")

        # Populate the parameters
        self.parameters = {
            name: (
                val
                if isinstance(val, (AbstractParameter, dmsc.Constant))
                else np.array(val)
            )
            for name, val in parameters.items()
        }

        # Initialize the shape of the parameter. The shape must be broadcastable
        # to the shapes of the parameters.
        self.shape = np.broadcast_shapes(
            shape, *[param.shape for param in self.parameters.values()]
        )

    @abstractmethod
    def draw(self, n: int) -> Any:
        """
        Sample from the distribution that represents the parameter. This method
        should be overwritten by the subclasses, though they may choose to call
        this method to perform the following set of operations:

        1.  Separate the constants and distributions.
        2.  If there are no distributions, then copy the constants `n` times. This
            creates a draw for the constants with a new first dimension of length
            `n`.
        3.  If there are distributions, then add a singleton dimension to the constants
            and draw `n` from the distributions. Combine the expanded constants
            and draws from the distributions.

        In child classes, this should return a numpy array. In this base class,
        however, a dictionary is returned that maps from parameter names to draws.
        """
        # Separate constants and distributions.
        constants, dists = {}, {}
        for name, param in self.parameters.items():
            if isinstance(param, AbstractParameter):
                dists[name] = param
            else:
                constants[name] = param

        # If there are no distributions, then we copy the constants `n` times.
        # If there are parent distributions, then add a singleton dimension to the
        # constants and draw `n` from the parent distributions.
        if len(dists) == 0:
            draws = {
                name: np.broadcast_to(val, (n,) + val.shape)
                for name, val in constants.items()
            }
        else:
            draws = {name: param.draw(n) for name, param in dists.items()}
            draws.update({name: val[None] for name, val in constants.items()})

        # Adding that new first dimension might break broadcasting, so we need to
        # add singleton dimensions after it to bring the total number of dimensions
        # to the same as the shape of the parameter.
        finalized_draws = {}
        for name, val in draws.items():
            to_add = (self.ndim + 1) - val.ndim  # Add 1 for sample dimension
            assert to_add >= 0
            finalized_draws[name] = np.expand_dims(
                val, axis=tuple(range(1, to_add + 1))
            )

        return finalized_draws

    def get_parents(self) -> list["CombinableParameterType"]:
        """
        Gathers the parent parameters of the current parameter.

        Returns:
            list[AbstractParameter]: Parent parameters of the current parameter.
        """
        return list(self.parameters.values())

    def recurse_parents(
        self, _current_depth: int = 0
    ) -> list[tuple[int, "AbstractParameter", "AbstractParameter"]]:
        """
        Recursively calls get_parents on the current parameter to get the entire
        lineage of the parameter.

        Returns:
            list[tuple[int, AbstractParameter, AbstractParameter]]: A list of tuples
                containing the depth of the parameter in the lineage, the parent
                parameter, and the current parameter, in that order.
        """
        # Get the parents of the current parameter
        parents = self.get_parents()

        # Call `recurse_parents` on each parent
        to_return = []
        for parent in parents:

            # Skip non-parameters
            if not isinstance(parent, AbstractParameter):
                continue

            # Add the parent to the list of tuples that will be returned
            to_return.append((_current_depth, parent, self))

            # Get the parent's lineage and add it to the list
            to_return.extend(parent.recurse_parents(_current_depth + 1))

        # Return the list of tuples
        return to_return

    def __str__(self):
        return f"{self.__class__.__name__}"

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the parameter"""
        return len(self.shape)

    @property
    def togglable(self) -> bool:
        """
        Return the parameters that can be toggled. These are the ones that have
        no parents.
        """
        return not any(
            isinstance(param, AbstractParameter) for param in self.parameters.values()
        )


class TransformedParameter(AbstractParameter):
    """
    Base class representing a parameter that is the result of combining other
    parameters using mathematical operations.
    """

    def draw(self, n: int) -> npt.NDArray:
        """Sample from this parameter's distribution `n` times."""

        # Perform the operation on the draws
        return self.operation(**super().draw(n))

    @abstractmethod
    def operation(self, **draws: "SampleType") -> npt.NDArray:
        """Perform the operation on the draws"""

    # Calling this class should return the result of the operation.
    def __call__(self, *args, **kwargs):
        return self.operation(*args, **kwargs)


class BinaryTransformedParameter(TransformedParameter):
    """
    Identical to the TransformedParameter class, but only for operations involving
    two parameters. In other words, two parameters must be passed to the class.
    """

    def __init__(
        self,
        dist1: "CombinableParameterType",
        dist2: "CombinableParameterType",
        shape: tuple[int, ...] = tuple(),
    ):
        super().__init__(dist1=dist1, dist2=dist2, shape=shape)

    @abstractmethod
    def operation(  # pylint: disable=arguments-differ
        self,
        dist1: "SampleType",
        dist2: "SampleType",
    ): ...


class UnaryTransformedParameter(TransformedParameter):
    """Transformed parameter that only requires one parameter."""

    def __init__(
        self, dist1: "CombinableParameterType", shape: tuple[int, ...] = tuple()
    ):
        super().__init__(dist1=dist1, shape=shape)

    @abstractmethod
    def operation(  # pylint: disable=arguments-differ
        self, dist1: "SampleType"
    ) -> npt.NDArray: ...


class AddParameter(BinaryTransformedParameter):
    """Defines a parameter that is the sum of two other parameters."""

    def operation(
        self,
        dist1: "SampleType",
        dist2: "SampleType",
    ) -> npt.NDArray:
        return dist1 + dist2


class SubtractParameter(BinaryTransformedParameter):
    """Defines a parameter that is the difference of two other parameters."""

    def operation(
        self,
        dist1: "SampleType",
        dist2: "SampleType",
    ) -> npt.NDArray:
        return dist1 - dist2


class MultiplyParameter(BinaryTransformedParameter):
    """Defines a parameter that is the product of two other parameters."""

    def operation(
        self,
        dist1: "SampleType",
        dist2: "SampleType",
    ) -> npt.NDArray:
        return dist1 * dist2


class DivideParameter(BinaryTransformedParameter):
    """Defines a parameter that is the quotient of two other parameters."""

    def operation(
        self,
        dist1: "SampleType",
        dist2: "SampleType",
    ) -> npt.NDArray:
        return dist1 / dist2


class PowerParameter(BinaryTransformedParameter):
    """Defines a parameter raised to the power of another parameter."""

    def operation(
        self,
        dist1: "SampleType",
        dist2: "SampleType",
    ) -> npt.NDArray:
        return dist1**dist2


class NegateParameter(UnaryTransformedParameter):
    """Defines a parameter that is the negative of another parameter."""

    def __init__(self, dist1: "CombinableParameterType"):
        super().__init__(dist1)

    def operation(self, dist1: "SampleType") -> npt.NDArray:
        return -dist1


class AbsParameter(UnaryTransformedParameter):
    """Defines a parameter that is the absolute value of another."""

    def operation(self, dist1: "SampleType") -> npt.NDArray:
        return np.abs(dist1)


class LogParameter(UnaryTransformedParameter):
    """Defines a parameter that is the natural logarithm of another."""

    def operation(self, dist1: "SampleType") -> npt.NDArray:
        return np.log(dist1)


class ExpParameter(UnaryTransformedParameter):
    """Defines a parameter that is the exponential of another."""

    def operation(self, dist1: "SampleType") -> npt.NDArray:
        return np.exp(dist1)


class NormalizeParameter(UnaryTransformedParameter):
    """Defines a parameter that is normalized to sum to 1."""

    def operation(self, dist1: "SampleType") -> npt.NDArray:
        return dist1 / np.sum(dist1)


class NormalizeLogParameter(UnaryTransformedParameter):
    """
    Defines a parameter that is normalized such that exp(x) sums to 1. By extension,
    this assumes that the input is log-transformed.
    """

    def operation(self, dist1: "SampleType") -> npt.NDArray:
        return dist1 - sp.logsumexp(dist1)


class Parameter(AbstractParameter):
    """Base class for parameters used in DMS Stan"""

    def __init__(
        self,
        numpy_dist: str,
        stan_to_np_names: dict[str, str],
        stan_to_np_transforms: Optional[
            dict[str, Callable[[npt.NDArray], npt.NDArray]]
        ] = None,
        seed: Optional[Union[np.random.Generator, int]] = None,
        shape: tuple[int, ...] = tuple(),
        **parameters,
    ):
        """
        Sets up random number generation and handles all parameters on which this
        parameter depends.
        """
        # Initialize the parameters
        super().__init__(shape=shape, **parameters)

        # Store the seed and the numpy distribution
        self._numpy_dist = numpy_dist
        self._seed = seed

        # Default value for the transforms dictionary is an empty dictionary
        stan_to_np_transforms = stan_to_np_transforms or {}

        # All parameter names must be in the stan_to_np_names dictionary
        if missing_names := set(parameters.keys()) - set(stan_to_np_names.keys()):
            raise ValueError(
                f"Missing names in stan_to_np_names: {', '.join(missing_names)}"
            )

        # Any key in the `stan_to_np_transforms` dictionary must be in `stan_to_np_names`
        # dictionary as well
        if not set(stan_to_np_transforms.keys()).issubset(stan_to_np_names.keys()):
            raise ValueError(
                "All keys in `stan_to_np_transforms` must be in `stan_to_np_names`"
            )

        # Store the stan names to numpy names dictionary and the numpy distribution
        # transformation dictionary
        self.stan_to_np_names = stan_to_np_names
        self.stan_to_np_transforms = stan_to_np_transforms

    def draw(self, n: int) -> npt.NDArray:
        """Sample from the distribution that represents the parameter `n` times"""
        # Get draws from the parent parameters
        draws = super().draw(n)

        # Perform transforms
        for name, transform in self.stan_to_np_transforms.items():
            draws[name] = transform(draws[name])

        # Rename the parameters to the names used by numpy
        draws = {self.stan_to_np_names[name]: val for name, val in draws.items()}

        # Sample from this distribution using numpy. Alter the shape to account
        # for the new first dimension of length `n`.
        return self.numpy_dist(**draws, size=(n,) + self.shape)

    def as_observable(self):
        """Redefines the parameter as an observable variable (i.e., data)"""
        self.observable = True
        return self

    def as_unobservable(self):
        """Redefines the parameter as an unobservable variable (i.e., a parameter)"""
        self.observable = False
        return self

    @property
    def rng(self) -> np.random.Generator:
        """Return the random number generator"""
        if self._seed is None:
            return dms.RNG
        elif isinstance(self._seed, int):
            return np.random.default_rng(self._seed)
        else:
            return self._seed

    @property
    def numpy_dist(self) -> Callable[..., npt.NDArray]:
        """Returns the numpy distribution function"""
        return getattr(self.rng, self._numpy_dist)


class Distribution(Parameter):
    """
    Defines distributions, which are a special type of parameter. This class is
    a passthrough to the Parameter class, but it is used to differentiate between
    parameters and distributions in the code.
    """


class ContinuousDistribution(Distribution):
    """Base class for parameters represented by continuous distributions."""

    def __add__(self, other: "CombinableParameterType"):
        return AddParameter(self, other)

    def __radd__(self, other: "CombinableParameterType"):
        return AddParameter(other, self)

    def __sub__(self, other: "CombinableParameterType"):
        return SubtractParameter(self, other)

    def __rsub__(self, other: "CombinableParameterType"):
        return SubtractParameter(other, self)

    def __mul__(self, other: "CombinableParameterType"):
        return MultiplyParameter(self, other)

    def __rmul__(self, other: "CombinableParameterType"):
        return MultiplyParameter(other, self)

    def __truediv__(self, other: "CombinableParameterType"):
        return DivideParameter(self, other)

    def __rtruediv__(self, other: "CombinableParameterType"):
        return DivideParameter(other, self)

    def __pow__(self, other: "CombinableParameterType"):
        return PowerParameter(self, other)

    def __rpow__(self, other: "CombinableParameterType"):
        return PowerParameter(other, self)

    def __neg__(self):
        return NegateParameter(self)


class DiscreteDistribution(Distribution):
    """
    Base class for parameters represented by discrete distributions. This is
    more-or-less a passthrough to the Parameter class; however, the default for
    discrete distributions is to set the observable attribute to True.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observable = True


class Normal(ContinuousDistribution):
    """Parameter that is represented by the normal distribution."""

    def __init__(
        self,
        *,
        mu: "ContinuousParameterType",
        sigma: "ContinuousParameterType",
        **kwargs,
    ):
        # Sigma must be positive
        if not isinstance(sigma, AbstractParameter) and sigma <= 0:
            raise ValueError("`sigma` must be positive")

        super().__init__(
            numpy_dist="normal",
            stan_to_np_names={"mu": "loc", "sigma": "scale"},
            mu=mu,
            sigma=sigma,
            **kwargs,
        )


class HalfNormal(Normal):
    """Parameter that is represented by the half-normal distribution."""

    def __init__(self, *, sigma: "ContinuousParameterType", **kwargs):
        super().__init__(mu=0, sigma=sigma, **kwargs)

    # Overwrite the draw method to ensure that the drawn values are positive
    def draw(self, n: int) -> npt.NDArray:
        return np.abs(super().draw(n))


class UnitNormal(Normal):
    """Parameter that is represented by the unit normal distribution."""

    def __init__(self, **kwargs):
        super().__init__(mu=0, sigma=1, **kwargs)


class LogNormal(ContinuousDistribution):
    """Parameter that is represented by the log-normal distribution."""

    def __init__(
        self,
        mu: "ContinuousParameterType",
        sigma: "ContinuousParameterType",
        **kwargs,
    ):
        # Sigma must be positive
        if not isinstance(sigma, AbstractParameter) and sigma <= 0:
            raise ValueError("`sigma` must be positive")
        super().__init__(
            numpy_dist="lognormal",
            stan_to_np_names={"mu": "mean", "sigma": "sigma"},
            mu=mu,
            sigma=sigma,
            **kwargs,
        )


class Beta(ContinuousDistribution):
    """Defines the beta distribution."""

    def __init__(
        self,
        *,
        alpha: "ContinuousParameterType",
        beta: "ContinuousParameterType",
        **kwargs,
    ):

        # Alpha and beta must be positive
        if not isinstance(alpha, AbstractParameter) and alpha <= 0:
            raise ValueError("`alpha` must be positive")
        if not isinstance(beta, AbstractParameter) and beta <= 0:
            raise ValueError("`beta` must be positive")

        super().__init__(
            numpy_dist="beta",
            stan_to_np_names={"alpha": "a", "beta": "b"},
            alpha=alpha,
            beta=beta,
            **kwargs,
        )


class Gamma(ContinuousDistribution):
    """Defines the gamma distribution."""

    def __init__(
        self,
        *,
        alpha: "ContinuousParameterType",
        beta: "ContinuousParameterType",
        **kwargs,
    ):

        # Alpha and beta must be positive
        if not isinstance(alpha, AbstractParameter) and alpha <= 0:
            raise ValueError("`alpha` must be positive")
        if not isinstance(beta, AbstractParameter) and beta <= 0:
            raise ValueError("`beta` must be positive")

        super().__init__(
            numpy_dist="gamma",
            stan_to_np_names={"alpha": "shape", "beta": "scale"},
            stan_to_np_transforms={"beta": lambda x: 1 / x},
            alpha=alpha,
            beta=beta,
            **kwargs,
        )


class Exponential(ContinuousDistribution):
    """Defines the exponential distribution."""

    def __init__(self, *, beta: "ContinuousParameterType", **kwargs):
        # Beta must be positive
        if not isinstance(beta, AbstractParameter) and beta <= 0:
            raise ValueError("`beta` must be positive")

        super().__init__(
            numpy_dist="exponential",
            stan_to_np_names={"beta": "scale"},
            stan_to_np_transforms={"beta": lambda x: 1 / x},
            beta=beta,
            **kwargs,
        )


class Dirichlet(ContinuousDistribution):
    """Defines the Dirichlet distribution."""

    def __init__(self, *, alpha: Union[AbstractParameter, npt.ArrayLike], **kwargs):
        # All alpha values must be positive
        if not isinstance(alpha, AbstractParameter) and np.all(np.array(alpha) <= 0):
            raise ValueError("All `alpha` values must be positive")

        super().__init__(
            numpy_dist="dirichlet",
            stan_to_np_names={"alpha": "alpha"},
            alpha=alpha,
            **kwargs,
        )


class Binomial(DiscreteDistribution):
    """Parameter that is represented by the binomial distribution"""

    def __init__(
        self,
        *,
        theta: "ContinuousParameterType",
        N: "DiscreteParameterType",
        **kwargs,
    ):
        # Theta must be between 0 and 1
        if not isinstance(theta, AbstractParameter) and not 0 <= theta <= 1:
            raise ValueError("`theta` must be between 0 and 1")

        super().__init__(
            numpy_dist="binomial",
            stan_to_np_names={"N": "n", "theta": "p"},
            N=N,
            theta=theta,
            **kwargs,
        )


class Poisson(DiscreteDistribution):
    """Parameter that is represented by the Poisson distribution."""

    def __init__(self, *, lambda_: "ContinuousParameterType", **kwargs):

        # Lambda must be positive
        if not isinstance(lambda_, AbstractParameter) and lambda_ <= 0:
            raise ValueError("`lambda_` must be positive")

        super().__init__(
            numpy_dist="poisson",
            stan_to_np_names={"lambda_": "lam"},
            lambda_=lambda_,
            **kwargs,
        )


class Multinomial(DiscreteDistribution):
    """Defines the multinomial distribution."""

    def __init__(
        self,
        *,
        theta: Union[AbstractParameter, npt.ArrayLike],
        N: Optional[Union[AbstractParameter, int]] = None,
        **kwargs,
    ):
        # Thetas must sum to 1
        if not isinstance(theta, AbstractParameter) and abs(np.sum(theta) % 1) > 1e-6:
            raise ValueError("All arrays of thetas must sum to 1")

        # Run the parent class's init
        super().__init__(
            numpy_dist="multinomial",
            stan_to_np_names={"N": "n", "theta": "pvals"},
            N=N,
            theta=theta,
            **kwargs,
        )

    def draw(self, n: int) -> npt.NDArray:
        # There must be a value for `N` in the parameters if we are sampling
        if self.parameters.get("N") is None:
            raise ValueError(
                "Sampling from a multinomial distribution is only possible when "
                "'N' is provided'"
            )

        return super().draw(n)


# Define custom types for this module
SampleType = Union[int, float, npt.NDArray]
ContinuousParameterType = Union[
    ContinuousDistribution,
    TransformedParameter,
    dmsc.Constant,
    float,
    npt.NDArray[np.floating],
]
DiscreteParameterType = Union[
    DiscreteDistribution,
    TransformedParameter,
    dmsc.Constant,
    int,
    npt.NDArray[np.integer],
]
CombinableParameterType = Union[ContinuousParameterType, DiscreteParameterType]
