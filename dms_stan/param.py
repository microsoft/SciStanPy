"""Holds classes that can be used for defining models in DMS Stan models."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

import numpy as np
import numpy.typing as npt

# TODO: Different parameters need different starting seeds!


class AbstractParameter(ABC):
    """Template class for parameters used in DMS Stan models."""

    # Define special types for parameters
    SampleType = Union[int, float, npt.NDArray]
    CombinableParameterType = Union["ContinuousParameter", int, float, npt.NDArray]

    @abstractmethod
    def sample(self, n: int) -> npt.NDArray:
        """Sample from the distribution that represents the parameter."""

    def __str__(self):
        return f"{self.__class__.__name__}"


class TransformedParameter(AbstractParameter):
    """
    Base class representing a parameter that is the result of combining other
    parameters using mathematical operations.
    """

    def __init__(
        self,
        param1: AbstractParameter.CombinableParameterType,
        param2: Optional[AbstractParameter.CombinableParameterType],
    ):
        """
        Create a transformed parameter by combining two parameters that are represented
        by continuous distributions.
        """
        # Store the two parameters
        self.param1 = param1
        self.param2 = param2

    @abstractmethod
    def sample(self, n: int) -> AbstractParameter.SampleType:
        # Sample from the first distribution
        return (
            self.param1.sample(n)
            if isinstance(self.param1, ContinuousParameter)
            else self.param1
        )

    @abstractmethod
    def operation(
        self,
        sample1: AbstractParameter.SampleType,
        sample2: Optional[AbstractParameter.SampleType],
    ) -> npt.NDArray:
        """Perform the operation on the samples"""


class BinaryTransformedParameter(TransformedParameter):
    """
    Identical to the TransformedParameter class, but only for operations involving
    two parameters. In other words, two parameters must be passed to the class.
    """

    def __init__(
        self,
        dist1: AbstractParameter.CombinableParameterType,
        dist2: AbstractParameter.CombinableParameterType,
    ):
        super().__init__(dist1, dist2)

    def sample(self, n: int) -> npt.NDArray:
        # Sample from the first distribution using the parent class's method
        sample1 = super().sample(n)

        # Sample from the second distribution
        sample2 = (
            self.param2.sample(n)
            if isinstance(self.param2, ContinuousParameter)
            else self.param2
        )

        # Perform the operation
        return self.operation(sample1, sample2)

    @abstractmethod
    def operation(
        self,
        sample1: AbstractParameter.SampleType,
        sample2: AbstractParameter.SampleType,
    ): ...


class UnaryTransformedParameter(TransformedParameter):
    """Transformed parameter that only requires one parameter."""

    def __init__(self, dist1: "ContinuousParameter"):
        super().__init__(dist1, None)

    def sample(self, n: int) -> npt.NDArray:
        # Sample from the first distribution using the parent class's method, then
        # perform the operation
        return self.operation(super().sample(n))

    # pylint: disable=arguments-differ
    @abstractmethod
    def operation(self, sample1: AbstractParameter.SampleType) -> npt.NDArray: ...

    # pylint: enable=arguments-differ


class AddParameter(BinaryTransformedParameter):
    """Defines a parameter that is the sum of two other parameters."""

    def operation(
        self,
        sample1: AbstractParameter.SampleType,
        sample2: AbstractParameter.SampleType,
    ) -> npt.NDArray:
        return sample1 + sample2


class SubtractParameter(BinaryTransformedParameter):
    """Defines a parameter that is the difference of two other parameters."""

    def operation(
        self,
        sample1: AbstractParameter.SampleType,
        sample2: AbstractParameter.SampleType,
    ) -> npt.NDArray:
        return sample1 - sample2


class MultiplyParameter(BinaryTransformedParameter):
    """Defines a parameter that is the product of two other parameters."""

    def operation(
        self,
        sample1: AbstractParameter.SampleType,
        sample2: AbstractParameter.SampleType,
    ) -> npt.NDArray:
        return sample1 * sample2


class DivideParameter(BinaryTransformedParameter):
    """Defines a parameter that is the quotient of two other parameters."""

    def operation(
        self,
        sample1: AbstractParameter.SampleType,
        sample2: AbstractParameter.SampleType,
    ) -> npt.NDArray:
        return sample1 / sample2


class PowerParameter(BinaryTransformedParameter):
    """Defines a parameter raised to the power of another parameter."""

    def operation(
        self,
        sample1: AbstractParameter.SampleType,
        sample2: AbstractParameter.SampleType,
    ) -> npt.NDArray:
        return sample1**sample2


class NegateParameter(UnaryTransformedParameter):
    """Defines a parameter that is the negative of another parameter."""

    def __init__(self, dist1: AbstractParameter.CombinableParameterType):
        super().__init__(dist1)

    def operation(self, sample1: AbstractParameter.SampleType) -> npt.NDArray:
        return -sample1


class AbsParameter(UnaryTransformedParameter):
    """Defines a parameter that is the absolute value of another."""

    def operation(self, sample1: AbstractParameter.SampleType) -> npt.NDArray:
        return np.abs(sample1)


class LogParameter(UnaryTransformedParameter):
    """Defines a parameter that is the natural logarithm of another."""

    def operation(self, sample1: AbstractParameter.SampleType) -> npt.NDArray:
        return np.log(sample1)


class ExpParameter(UnaryTransformedParameter):
    """Defines a parameter that is the exponential of another."""

    def operation(self, sample1: AbstractParameter.SampleType) -> npt.NDArray:
        return np.exp(sample1)


class Parameter(AbstractParameter):
    """Base class for parameters used in DMS Stan"""

    def __init__(
        self,
        numpy_dist: str,
        stan_to_np_names: dict[str, str],
        *,
        seed: int = 2,
        **parameters,
    ):
        """
        Sets up random number generation and handles all parameters on which this
        parameter depends.
        """
        # Run the parent class's init
        super().__init__()

        # Define variables with types that are created within this function
        self.rng: np.random.Generator
        self.numpy_dist: Callable[..., npt.NDArray]
        self.parameters: dict[str, Parameter] = {}
        self.constants: dict[str, Any] = {}

        # Define the random number generator
        self.rng = np.random.default_rng(seed)

        # Store the numpy distribution
        self.numpy_dist = getattr(self.rng, numpy_dist)

        # Populate the parameters and constants
        for paramname, val in parameters.items():
            if isinstance(val, AbstractParameter):
                self.parameters[paramname] = val
            else:
                self.constants[paramname] = val

        # All parameter names must be in the stan_to_np_names dictionary
        if missing_names := set(parameters.keys()) - set(stan_to_np_names.keys()):
            raise ValueError(
                f"Missing names in stan_to_np_names: {', '.join(missing_names)}"
            )

        # Store the stan names to numpy names dictionary
        self.stan_to_np_names = stan_to_np_names

    def sample(self, n: int) -> npt.NDArray:
        # Sample from the parameter distributions
        param_draws = {
            self.stan_to_np_names[name]: param.sample(n)
            for name, param in self.parameters.items()
        }

        # Add on constants
        param_draws.update(
            {self.stan_to_np_names[name]: val for name, val in self.constants.items()}
        )

        # Sample from this distribution using numpy
        return self.numpy_dist(**param_draws, size=n)


class ContinuousParameter(Parameter):
    """Base class for parameters represented by continuous distributions."""

    def __add__(self, other: AbstractParameter.CombinableParameterType):
        return AddParameter(self, other)

    def __radd__(self, other: AbstractParameter.CombinableParameterType):
        return AddParameter(other, self)

    def __sub__(self, other: AbstractParameter.CombinableParameterType):
        return SubtractParameter(self, other)

    def __rsub__(self, other: AbstractParameter.CombinableParameterType):
        return SubtractParameter(other, self)

    def __mul__(self, other: AbstractParameter.CombinableParameterType):
        return MultiplyParameter(self, other)

    def __rmul__(self, other: AbstractParameter.CombinableParameterType):
        return MultiplyParameter(other, self)

    def __truediv__(self, other: AbstractParameter.CombinableParameterType):
        return DivideParameter(self, other)

    def __rtruediv__(self, other: AbstractParameter.CombinableParameterType):
        return DivideParameter(other, self)

    def __pow__(self, other: AbstractParameter.CombinableParameterType):
        return PowerParameter(self, other)

    def __rpow__(self, other: AbstractParameter.CombinableParameterType):
        return PowerParameter(other, self)

    def __neg__(self):
        return NegateParameter(self)


class DiscreteParameter(Parameter):
    """Base class for parameters represented by discrete distributions."""


class Normal(ContinuousParameter):
    """Parameter that is represented by the normal distribution."""

    def __init__(
        self,
        *,
        mu: Union[AbstractParameter, float],
        sigma: Union[AbstractParameter, float],
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

    def __init__(self, *, sigma: Union[AbstractParameter, float], **kwargs):
        super().__init__(mu=0, sigma=sigma, **kwargs)

    # Overwrite the sample method to ensure that the sampled values are positive
    def sample(self, n: int) -> npt.NDArray:
        return np.abs(super().sample(n))


class UnitNormal(Normal):
    """Parameter that is represented by the unit normal distribution."""

    def __init__(self, **kwargs):
        super().__init__(mu=0, sigma=1, **kwargs)


class Beta(ContinuousParameter):
    """Defines the beta distribution."""

    def __init__(
        self,
        *,
        alpha: Union[AbstractParameter, float],
        beta: Union[AbstractParameter, float],
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


class Dirichlet(ContinuousParameter):
    """Defines the Dirichlet distribution."""

    def __init__(self, *, alpha: Union[AbstractParameter, npt.ArrayLike], **kwargs):
        # All alpha values must be positive
        if isinstance(alpha, AbstractParameter):
            test_alpha = np.array(alpha.distribution.sample(1))
        else:
            test_alpha = np.array(alpha)
        if not np.all(test_alpha > 0):
            raise ValueError("All `alpha` values must be positive")

        super().__init__(
            numpy_dist="dirichlet",
            stan_to_np_names={"alpha": "alpha"},
            alpha=alpha,
            **kwargs,
        )


class Binomial(DiscreteParameter):
    """Parameter that is represented by the binomial distribution"""

    def __init__(
        self,
        *,
        theta: Union[AbstractParameter, float],
        N: Union[AbstractParameter, int],
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


class Poisson(DiscreteParameter):
    """Parameter that is represented by the Poisson distribution."""

    def __init__(self, *, lambda_: Union[AbstractParameter, float], **kwargs):

        # Lambda must be positive
        if not isinstance(lambda_, AbstractParameter) and lambda_ <= 0:
            raise ValueError("`lambda_` must be positive")

        super().__init__(
            numpy_dist="poisson",
            stan_to_np_names={"lambda_": "lam"},
            lambda_=lambda_,
            **kwargs,
        )


class Multinomial(DiscreteParameter):
    """Defines the multinomial distribution."""

    def __init__(
        self,
        *,
        theta: Union[AbstractParameter, npt.ArrayLike],
        N: Optional[Union[AbstractParameter, int]] = None,
        **kwargs,
    ):
        # Sample the theta parameter if it is a distribution
        if isinstance(theta, AbstractParameter):
            sampled = np.array(theta.distribution.sample(1))
            sampled = sampled.unsqueeze(0) if sampled.ndim == 1 else sampled

        # Otherwise make sure that it is a 1D array
        else:
            sampled = np.expand_dims(np.array(theta), 0)
            if sampled.ndim != 2:
                raise ValueError("Thetas must be passed as a 1D array")

        # Whether passed as a parameter or not, the thetas must sum to 1
        if not np.allclose(sampled.sum(axis=-1), 1):
            raise ValueError("All arrays of thetas must sum to 1")

        # Run the parent class's init
        super().__init__(
            numpy_dist="multinomial",
            stan_to_np_names={"N": "n", "theta": "pvals"},
            N=N,
            theta=theta,
            **kwargs,
        )

    def sample(self, n: int) -> npt.NDArray:
        # There must be a value for `N` in the parameters if we are sampling
        if (self.parameters.get("N") is None) and (self.constants.get("N") is None):
            raise ValueError(
                "Sampling from a multinomial distribution is only possible when "
                "'N' is provided'"
            )

        return super().sample(n)
