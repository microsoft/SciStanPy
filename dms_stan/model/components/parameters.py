"""Holds classes that can be used for defining models in DMS Stan models."""

from abc import abstractmethod
from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt
import torch.distributions as dist

import dms_stan as dms
import dms_stan.model.components as dms_components

from .abstract_classes import AbstractParameter, AbstractPassthrough
from .pytorch import ParameterContainer
from .transformed_parameters import (
    AddParameter,
    DivideParameter,
    MultiplyParameter,
    NegateParameter,
    PowerParameter,
    SubtractParameter,
    TransformedParameter,
)


class Parameter(AbstractParameter):
    """Base class for parameters used in DMS Stan"""

    _torch_container_class = ParameterContainer

    def __init__(
        self,
        numpy_dist: str,
        torch_dist,
        stan_to_np_names: dict[str, str],
        stan_to_torch_names: dict[str, str],
        stan_to_np_transforms: Optional[
            dict[str, Callable[[npt.NDArray], npt.NDArray]]
        ] = None,
        seed: Optional[Union[np.random.Generator, int]] = None,
        shape: Optional[tuple[int, ...]] = None,
        **parameters,
    ):
        """
        Sets up random number generation and handles all parameters on which this
        parameter depends.
        """
        # Initialize the parameters
        super().__init__(shape=shape, **parameters)

        # Store the seed and distributions
        self._numpy_dist = numpy_dist
        self._torch_dist = torch_dist
        self._seed = seed

        # Default value for the transforms dictionary is an empty dictionary
        stan_to_np_transforms = stan_to_np_transforms or {}

        # All parameter names must be in the stan_to_np_names dictionary
        if missing_names := set(parameters.keys()) - set(stan_to_np_names.keys()):
            raise ValueError(
                f"Missing names in stan_to_np_names: {', '.join(missing_names)}"
            )

        # All parameter names must be in the stan_to_torch_names dictionary
        if missing_names := set(parameters.keys()) - set(stan_to_torch_names.keys()):
            raise ValueError(
                f"Missing names in stan_to_torch_names: {', '.join(missing_names)}"
            )

        # Any key in the `stan_to_np_transforms` dictionary must be in `stan_to_np_names`
        # dictionary as well
        if not set(stan_to_np_transforms.keys()).issubset(stan_to_np_names.keys()):
            raise ValueError(
                "All keys in `stan_to_np_transforms` must be in `stan_to_np_names`"
            )

        # Store the stan names to names dictionaries and the numpy distribution
        # transformation dictionary
        self.stan_to_np_names = stan_to_np_names
        self.stan_to_np_transforms = stan_to_np_transforms
        self.stan_to_torch_names = stan_to_torch_names

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
        return self.numpy_dist(**draws, size=(n,) + self.draw_shape)

    def as_observable(self):
        """Redefines the parameter as an observable variable (i.e., data)"""
        self.observable = True
        return self

    def as_unobservable(self):
        """Redefines the parameter as an unobservable variable (i.e., a parameter)"""
        self.observable = False
        return self

    def get_target_incrementation(self, index_opts: tuple[str, ...]) -> str:
        """Return the Stan target incrementation for this parameter."""
        return (
            f"{self.get_indexed_varname(index_opts)} ~ "
            + self.get_stan_distribution(index_opts)
        )

    # TODO: We need to handle the set hyperparameter values. How do we populate
    # them in the distribution?
    def get_stan_distribution(self, index_opts: tuple[str, ...]) -> str:
        """Return the Stan distribution for this parameter"""

        # Recursively gather the transformations until we hit a non-transformed
        # parameter or a recorded variable
        to_format: dict[str, str] = {}
        for name, param in self.parameters.items():

            # If the parameter is a constant or another parameter, record
            if isinstance(param, (AbstractPassthrough, Parameter)):
                to_format[name] = param.get_indexed_varname(index_opts)

            # If the parameter is transformed and not named, the computation is
            # happening in the model. Otherwise, the computation has already happened
            # in the transformed parameters block.
            elif isinstance(param, TransformedParameter):
                if param.is_named:
                    to_format[name] = param.get_stan_transformation(index_opts)
                else:
                    to_format[name] = param.get_indexed_varname(index_opts)

            # Otherwise, raise an error
            else:
                raise TypeError(f"Unknown model component type {type(param)}")

        # Format the distribution
        return self.format_stan_distribution(**to_format)

    @abstractmethod
    def format_stan_distribution(self, **param_vals: str) -> str:
        """Return the base Stan distribution for this parameter"""

    @property
    def rng(self) -> np.random.Generator:
        """Return the random number generator"""
        if self._seed is None:
            return dms.RNG
        elif isinstance(self._seed, int):
            self._seed = np.random.default_rng(self._seed)
        return self._seed

    @property
    def numpy_dist(self) -> Callable[..., npt.NDArray]:
        """Returns the numpy distribution function"""
        return getattr(self.rng, self._numpy_dist)

    @property
    def torch_dist(self):
        """Returns the torch distribution class"""
        return self._torch_dist


class ContinuousDistribution(Parameter):
    """Base class for parameters represented by continuous distributions."""

    def __add__(self, other: "dms_components.custom_types.CombinableParameterType"):
        return AddParameter(self, other)

    def __radd__(self, other: "dms_components.custom_types.CombinableParameterType"):
        return AddParameter(other, self)

    def __sub__(self, other: "dms_components.custom_types.CombinableParameterType"):
        return SubtractParameter(self, other)

    def __rsub__(self, other: "dms_components.custom_types.CombinableParameterType"):
        return SubtractParameter(other, self)

    def __mul__(self, other: "dms_components.custom_types.CombinableParameterType"):
        return MultiplyParameter(self, other)

    def __rmul__(self, other: "dms_components.custom_types.CombinableParameterType"):
        return MultiplyParameter(other, self)

    def __truediv__(self, other: "dms_components.custom_types.CombinableParameterType"):
        return DivideParameter(self, other)

    def __rtruediv__(
        self, other: "dms_components.custom_types.CombinableParameterType"
    ):
        return DivideParameter(other, self)

    def __pow__(self, other: "dms_components.custom_types.CombinableParameterType"):
        return PowerParameter(self, other)

    def __rpow__(self, other: "dms_components.custom_types.CombinableParameterType"):
        return PowerParameter(other, self)

    def __neg__(self):
        return NegateParameter(self)


class DiscreteDistribution(Parameter):
    """
    Base class for parameters represented by discrete distributions. This is
    more-or-less a passthrough to the Parameter class; however, the default for
    discrete distributions is to set the observable attribute to True.
    """

    base_stan_dtype: str = "int"
    stan_lower_bound: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observable = True


class Normal(ContinuousDistribution):
    """Parameter that is represented by the normal distribution."""

    POSITIVE_PARAMS = {"sigma"}

    def __init__(
        self,
        *,
        mu: "dms_components.custom_types.ContinuousParameterType",
        sigma: "dms_components.custom_types.ContinuousParameterType",
        **kwargs,
    ):

        super().__init__(
            numpy_dist="normal",
            torch_dist=dist.normal.Normal,
            stan_to_np_names={"mu": "loc", "sigma": "scale"},
            stan_to_torch_names={"mu": "loc", "sigma": "scale"},
            mu=mu,
            sigma=sigma,
            **kwargs,
        )

    def format_stan_distribution(  # pylint: disable=arguments-differ
        self, mu: str, sigma: str
    ) -> str:
        return f"normal({mu}, {sigma})"


class HalfNormal(Normal):
    """Parameter that is represented by the half-normal distribution."""

    stan_lower_bound: float = 0.0

    def __init__(
        self,
        *,
        sigma: "dms_components.custom_types.ContinuousParameterType",
        **kwargs,
    ):
        super().__init__(mu=0.0, sigma=sigma, **kwargs)

    # Overwrite the draw method to ensure that the drawn values are positive
    def draw(self, n: int) -> npt.NDArray:
        return np.abs(super().draw(n))


class UnitNormal(Normal):
    """Parameter that is represented by the unit normal distribution."""

    def __init__(self, **kwargs):
        super().__init__(mu=0.0, sigma=1.0, **kwargs)


class LogNormal(ContinuousDistribution):
    """Parameter that is represented by the log-normal distribution."""

    POSITIVE_PARAMS = {"sigma"}
    stan_lower_bound: float = 0.0

    def __init__(
        self,
        mu: "dms_components.custom_types.ContinuousParameterType",
        sigma: "dms_components.custom_types.ContinuousParameterType",
        **kwargs,
    ):
        super().__init__(
            numpy_dist="lognormal",
            torch_dist=dist.log_normal.LogNormal,
            stan_to_np_names={"mu": "mean", "sigma": "sigma"},
            stan_to_torch_names={"mu": "loc", "sigma": "scale"},
            mu=mu,
            sigma=sigma,
            **kwargs,
        )

    def format_stan_distribution(  # pylint: disable=arguments-differ
        self, mu: str, sigma: str
    ) -> str:
        return f"lognormal({mu}, {sigma})"


class Beta(ContinuousDistribution):
    """Defines the beta distribution."""

    POSITIVE_PARAMS = {"alpha", "beta"}
    stan_lower_bound: float = 0.0
    stan_upper_bound: float = 1.0

    def __init__(
        self,
        *,
        alpha: "dms_components.custom_types.ContinuousParameterType",
        beta: "dms_components.custom_types.ContinuousParameterType",
        **kwargs,
    ):

        super().__init__(
            numpy_dist="beta",
            torch_dist=dist.beta.Beta,
            stan_to_np_names={"alpha": "a", "beta": "b"},
            stan_to_torch_names={"alpha": "concentration1", "beta": "concentration0"},
            alpha=alpha,
            beta=beta,
            **kwargs,
        )

    def format_stan_distribution(  # pylint: disable=arguments-differ
        self, alpha: str, beta: str
    ) -> str:
        return f"beta({alpha}, {beta})"


class Gamma(ContinuousDistribution):
    """Defines the gamma distribution."""

    POSITIVE_PARAMS = {"alpha", "beta"}

    stan_lower_bound: float = 0.0

    def __init__(
        self,
        *,
        alpha: "dms_components.custom_types.ContinuousParameterType",
        beta: "dms_components.custom_types.ContinuousParameterType",
        **kwargs,
    ):

        super().__init__(
            numpy_dist="gamma",
            torch_dist=dist.gamma.Gamma,
            stan_to_np_names={"alpha": "shape", "beta": "scale"},
            stan_to_torch_names={"alpha": "concentration", "beta": "rate"},
            stan_to_np_transforms={"beta": lambda x: 1 / x},
            alpha=alpha,
            beta=beta,
            **kwargs,
        )

    def format_stan_distribution(  # pylint: disable=arguments-differ
        self, alpha: str, beta: str
    ) -> str:
        return f"gamma({alpha}, {beta})"


class Exponential(ContinuousDistribution):
    """Defines the exponential distribution."""

    POSITIVE_PARAMS = {"beta"}

    # Overwrite the stan data type
    stan_lower_bound: float = 0.0

    def __init__(
        self,
        *,
        beta: "dms_components.custom_types.ContinuousParameterType",
        **kwargs,
    ):

        super().__init__(
            numpy_dist="exponential",
            torch_dist=dist.exponential.Exponential,
            stan_to_np_names={"beta": "scale"},
            stan_to_torch_names={"beta": "rate"},
            stan_to_np_transforms={"beta": lambda x: 1 / x},
            beta=beta,
            **kwargs,
        )

    def format_stan_distribution(  # pylint: disable=arguments-differ
        self, beta: str
    ) -> str:
        return f"exponential({beta})"


class Dirichlet(ContinuousDistribution):
    """Defines the Dirichlet distribution."""

    POSITIVE_PARAMS = {"alpha"}
    base_stan_dtype: str = "simplex"

    def __init__(
        self,
        *,
        alpha: Union[AbstractParameter, npt.ArrayLike],
        **kwargs,
    ):

        super().__init__(
            numpy_dist="dirichlet",
            torch_dist=dist.dirichlet.Dirichlet,
            stan_to_np_names={"alpha": "alpha"},
            stan_to_torch_names={"alpha": "concentration"},
            alpha=alpha,
            **kwargs,
        )

    def format_stan_distribution(  # pylint: disable=arguments-differ
        self, alpha: str
    ) -> str:
        return f"dirichlet({alpha})"


class Binomial(DiscreteDistribution):
    """Parameter that is represented by the binomial distribution"""

    POSITIVE_PARAMS = {"theta", "N"}

    def __init__(
        self,
        *,
        theta: "dms_components.custom_types.ContinuousParameterType",
        N: "dms_components.custom_types.DiscreteParameterType",
        **kwargs,
    ):

        super().__init__(
            numpy_dist="binomial",
            torch_dist=dist.binomial.Binomial,
            stan_to_np_names={"N": "n", "theta": "p"},
            stan_to_torch_names={"N": "total_count", "theta": "probs"},
            N=N,
            theta=theta,
            **kwargs,
        )

    def format_stan_distribution(  # pylint: disable=arguments-differ
        self, N: str, theta: str  # pylint: disable="invalid-name"
    ) -> str:
        return f"binomial({N}, {theta})"


class Poisson(DiscreteDistribution):
    """Parameter that is represented by the Poisson distribution."""

    POSITIVE_PARAMS = {"lambda_"}

    def __init__(
        self,
        *,
        lambda_: "dms_components.custom_types.ContinuousParameterType",
        **kwargs,
    ):

        super().__init__(
            numpy_dist="poisson",
            torch_dist=dist.poisson.Poisson,
            stan_to_np_names={"lambda_": "lam"},
            stan_to_torch_names={"lambda_": "rate"},
            lambda_=lambda_,
            **kwargs,
        )

    def format_stan_distribution(  # pylint: disable=arguments-differ
        self, lambda_: str
    ) -> str:
        return f"poisson({lambda_})"


class Multinomial(DiscreteDistribution):
    """Defines the multinomial distribution."""

    SIMPLEX_PARAMS = {"theta"}

    def __init__(
        self,
        *,
        theta: Union[AbstractParameter, npt.ArrayLike],
        N: Optional[Union[AbstractParameter, int]] = None,
        **kwargs,
    ):

        # Run the parent class's init
        super().__init__(
            numpy_dist="multinomial",
            torch_dist=dist.multinomial.Multinomial,
            stan_to_np_names={"N": "n", "theta": "pvals"},
            stan_to_torch_names={"N": "total_count", "theta": "probs"},
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

    def format_stan_distribution(  # pylint: disable=arguments-differ, unused-argument
        self, N: str, theta: str
    ) -> str:
        return f"multinomial({theta})"
