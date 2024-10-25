"""Holds classes that can be used for defining models in DMS Stan models."""

from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.distributions as dist

import dms_stan as dms

from .abstract_model_component import AbstractModelComponent
from .transformed_parameters import (
    AddParameter,
    DivideParameter,
    MultiplyParameter,
    NegateParameter,
    PowerParameter,
    SubtractParameter,
)


class Parameter(AbstractModelComponent):
    """Base class for parameters used in DMS Stan"""

    def __init__(
        self,
        numpy_dist: str,
        torch_dist,
        stan_to_np_names: dict[str, str],
        stan_to_torch_names: dict[str, str],
        stan_to_np_transforms: Optional[
            dict[str, Callable[[npt.NDArray], npt.NDArray]]
        ] = None,
        shape: tuple[int, ...] = (),
        **parameters,
    ):
        """
        Sets up random number generation and handles all parameters on which this
        parameter depends.
        """
        # Initialize the parameters
        super().__init__(shape=shape, **parameters)

        # Store the distributions
        self._numpy_dist = numpy_dist
        self._torch_dist = torch_dist

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

    def _draw(self, n: int, level_draws: dict[str, npt.NDArray]) -> npt.NDArray:
        """Sample from the distribution that represents the parameter `n` times"""
        # Perform transforms
        for name, transform in self.stan_to_np_transforms.items():
            level_draws[name] = transform(level_draws[name])

        # Rename the parameters to the names used by numpy
        level_draws = {
            self.stan_to_np_names[name]: val for name, val in level_draws.items()
        }

        # Sample from this distribution using numpy. Alter the shape to account
        # for the new first dimension of length `n`.
        return self.numpy_dist(**level_draws, size=(n,) + self.draw_shape)

    def as_observable(self) -> "Parameter":
        """Redefines the parameter as an observable variable (i.e., data)"""
        self.observable = True
        return self

    def get_target_incrementation(self, index_opts: tuple[str, ...]) -> str:
        """Return the Stan target incrementation for this parameter."""
        return f"{self.get_indexed_varname(index_opts)} ~ " + self.get_stan_code(
            index_opts
        )

    def _handle_transformation_code(
        self, param: AbstractModelComponent, index_opts: tuple[str, ...]
    ) -> str:
        if param.is_named:
            return param.get_stan_code(index_opts)
        else:
            return param.get_indexed_varname(index_opts)

    def calculate_log_prob(
        self, observed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculates the log probability of the parameters given the observed data.

        Args:
            observed (Optional[torch.Tensor], optional): The observed value. This
                only needs to be provided for observed parameters. Latent parameters
                will automatically identify the child parameters and in turn use
                that parameter's parameters as the observed value. Defaults to None.

        Returns:
            torch.Tensor: Log probability of the parameters given the observed data.
        """
        # Calculate log probability using the observed data and the distribution
        return self.torch_dist_instance.log_prob(
            self.get_torch_observables(observed)
        ).sum()

    @property
    def rng(self) -> np.random.Generator:
        """Return the random number generator"""
        return dms.RNG

    @property
    def numpy_dist(self) -> Callable[..., npt.NDArray]:
        """Returns the numpy distribution function"""
        return getattr(self.rng, self._numpy_dist)

    @property
    def torch_dist(self) -> type[dist.Distribution]:
        """Returns the torch distribution class"""
        return self._torch_dist

    @property
    def torch_dist_instance(self) -> dist.Distribution:
        """Returns an instance of the torch distribution class"""
        return self.torch_dist(
            **{
                self.stan_to_torch_names[name]: param
                for name, param in self.torch_parameters.items()
            }
        )


class ContinuousDistribution(Parameter):
    """Base class for parameters represented by continuous distributions."""

    def __add__(self, other: "dms.custom_types.CombinableParameterType"):
        return AddParameter(self, other)

    def __radd__(self, other: "dms.custom_types.CombinableParameterType"):
        return AddParameter(other, self)

    def __sub__(self, other: "dms.custom_types.CombinableParameterType"):
        return SubtractParameter(self, other)

    def __rsub__(self, other: "dms.custom_types.CombinableParameterType"):
        return SubtractParameter(other, self)

    def __mul__(self, other: "dms.custom_types.CombinableParameterType"):
        return MultiplyParameter(self, other)

    def __rmul__(self, other: "dms.custom_types.CombinableParameterType"):
        return MultiplyParameter(other, self)

    def __truediv__(self, other: "dms.custom_types.CombinableParameterType"):
        return DivideParameter(self, other)

    def __rtruediv__(self, other: "dms.custom_types.CombinableParameterType"):
        return DivideParameter(other, self)

    def __pow__(self, other: "dms.custom_types.CombinableParameterType"):
        return PowerParameter(self, other)

    def __rpow__(self, other: "dms.custom_types.CombinableParameterType"):
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
        mu: "dms.custom_types.ContinuousParameterType",
        sigma: "dms.custom_types.ContinuousParameterType",
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

    def format_stan_code(  # pylint: disable=arguments-differ
        self, mu: str, sigma: str
    ) -> str:
        return f"normal({mu}, {sigma})"


class HalfNormal(Normal):
    """Parameter that is represented by the half-normal distribution."""

    stan_lower_bound: float = 0.0

    def __init__(
        self,
        *,
        sigma: "dms.custom_types.ContinuousParameterType",
        **kwargs,
    ):
        super().__init__(mu=0.0, sigma=sigma, **kwargs)

    # Overwrite the draw method to ensure that the drawn values are positive
    def draw(
        self,
        n: int,
        _drawn: Optional[dict["AbstractModelComponent", npt.NDArray]] = None,
    ) -> tuple[npt.NDArray, dict["AbstractModelComponent", npt.NDArray]]:
        # Draw from the normal distribution and take the absolute value
        draws, _drawn = super().draw(n, _drawn=_drawn)
        draws = np.abs(draws)

        # Update the drawn values
        _drawn[self] = draws

        return draws, _drawn


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
        mu: "dms.custom_types.ContinuousParameterType",
        sigma: "dms.custom_types.ContinuousParameterType",
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

    def format_stan_code(  # pylint: disable=arguments-differ
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
        alpha: "dms.custom_types.ContinuousParameterType",
        beta: "dms.custom_types.ContinuousParameterType",
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

    def format_stan_code(  # pylint: disable=arguments-differ
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
        alpha: "dms.custom_types.ContinuousParameterType",
        beta: "dms.custom_types.ContinuousParameterType",
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

    def format_stan_code(  # pylint: disable=arguments-differ
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
        beta: "dms.custom_types.ContinuousParameterType",
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

    def format_stan_code(self, beta: str) -> str:  # pylint: disable=arguments-differ
        return f"exponential({beta})"


class Dirichlet(ContinuousDistribution):
    """Defines the Dirichlet distribution."""

    POSITIVE_PARAMS = {"alpha"}
    base_stan_dtype: str = "simplex"

    def __init__(
        self,
        *,
        alpha: Union[AbstractModelComponent, npt.ArrayLike],
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

    def format_stan_code(self, alpha: str) -> str:  # pylint: disable=arguments-differ
        return f"dirichlet({alpha})"


class Binomial(DiscreteDistribution):
    """Parameter that is represented by the binomial distribution"""

    POSITIVE_PARAMS = {"theta", "N"}

    def __init__(
        self,
        *,
        theta: "dms.custom_types.ContinuousParameterType",
        N: "dms.custom_types.DiscreteParameterType",
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

    def format_stan_code(  # pylint: disable=arguments-differ
        self, N: str, theta: str  # pylint: disable="invalid-name"
    ) -> str:
        return f"binomial({N}, {theta})"


class Poisson(DiscreteDistribution):
    """Parameter that is represented by the Poisson distribution."""

    POSITIVE_PARAMS = {"lambda_"}

    def __init__(
        self,
        *,
        lambda_: "dms.custom_types.ContinuousParameterType",
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

    def format_stan_code(self, lambda_: str) -> str:  # pylint: disable=arguments-differ
        return f"poisson({lambda_})"


class Multinomial(DiscreteDistribution):
    """Defines the multinomial distribution."""

    SIMPLEX_PARAMS = {"theta"}

    def __init__(
        self,
        *,
        theta: Union[AbstractModelComponent, npt.ArrayLike],
        N: Optional[Union[AbstractModelComponent, int]] = None,
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

    def draw(
        self,
        n: int,
        _drawn: Optional[dict["AbstractModelComponent", npt.NDArray]] = None,
    ) -> tuple[npt.NDArray, dict["AbstractModelComponent", npt.NDArray]]:
        # There must be a value for `N` in the parameters if we are sampling
        if self._parents.get("N") is None:
            raise ValueError(
                "Sampling from a multinomial distribution is only possible when "
                "'N' is provided'"
            )

        return super().draw(n, _drawn=_drawn)

    def format_stan_code(  # pylint: disable=arguments-differ, unused-argument
        self, N: str, theta: str
    ) -> str:
        return f"multinomial({theta})"
