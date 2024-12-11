"""Holds classes that can be used for defining models in DMS Stan models."""

from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.distributions as dist
import torch.nn as nn

import dms_stan as dms

from .abstract_model_component import AbstractModelComponent
from .constants import Constant
from .transformed_parameters import TransformableParameter


class Parameter(AbstractModelComponent):
    """Base class for parameters used in DMS Stan"""

    def __init__(
        self,
        numpy_dist: str,
        torch_dist: type[dist.distribution.Distribution],
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

        # Initialize a parametrization using PyTorch
        self._torch_parametrization: Optional[nn.Parameter] = None

    def init_pytorch(
        self,
        init_val: Optional[Union[npt.NDArray, torch.Tensor]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Sets up the parameters needed for training a Pytorch model."""
        # This cannot be called if the parameter is an observable
        if self.observable:
            raise ValueError("Observables do not have a torch parametrization")

        # If no initialization value is provided, then we draw one
        if init_val is None:
            init_val, _ = self.draw(1, seed=seed)
            init_val = np.squeeze(init_val, axis=0)

        # If the initialization value is a numpy array, convert it to a tensor
        if isinstance(init_val, np.ndarray):
            init_val = torch.from_numpy(init_val)

        # The shape of the initialization value must match the shape of the
        # parameter being initialized
        if init_val.shape != self.shape:
            raise ValueError(
                f"The shape of the initialization value must match the shape of the "
                f"parameter. Expected: {self.shape}, provided: {init_val.shape}"
            )

        # Initialize the parameter
        self._torch_parametrization = nn.Parameter(init_val)

    def _draw(
        self, n: int, level_draws: dict[str, npt.NDArray], seed: Optional[int]
    ) -> npt.NDArray:
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
        return self.get_numpy_dist(seed=seed)(**level_draws, size=(n,) + self.shape)

    def as_observable(self) -> "Parameter":
        """Redefines the parameter as an observable variable (i.e., data)"""

        # Set the observable attribute to True
        self.observable = True

        # We do not have a torch parameterization for observables
        self._torch_parametrization = None

        return self

    def get_transformation_assignment(self, index_opts: tuple[str, ...]) -> str:
        """Null opp for parameters by default"""
        return ""

    def get_target_incrementation(self, index_opts: tuple[str, ...]) -> str:
        """Return the Stan target incrementation for this parameter."""
        return f"{self.get_indexed_varname(index_opts)} ~ " + self.get_stan_code(
            index_opts
        )

    def _handle_transformation_code(
        self, param: AbstractModelComponent, index_opts: tuple[str, ...]
    ) -> str:
        if param.is_named:
            return param.get_indexed_varname(index_opts)
        else:
            return param.get_stan_code(index_opts)

    def get_torch_logprob(
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
        # Observed parameters must have an observed value.
        if self.observable and observed is None:
            raise ValueError("Observed parameters must have an observed value.")

        # If this is not an observable, then we should not have an observed value
        if not self.observable and observed is not None:
            raise ValueError("Latent parameters should not have an observed value.")

        # Calculate log probability using the observed data and the distribution
        return self.torch_dist_instance.log_prob(
            observed if self.observable else self.torch_parametrization
        ).sum()

    def get_rng(self, seed: Optional[int] = None) -> np.random.Generator:
        """Return the random number generator"""
        # Return the global random number generator if no seed is provided. Otherwise,
        # return a new random number generator with the provided seed.
        if seed is None:
            return dms.RNG
        return np.random.default_rng(seed)

    def get_numpy_dist(self, seed: Optional[int] = None) -> Callable[..., npt.NDArray]:
        """Returns the numpy distribution function"""
        return getattr(self.get_rng(seed=seed), self._numpy_dist)

    @property
    def torch_dist(self) -> type[dist.Distribution]:
        """Returns the torch distribution class"""
        return self._torch_dist

    @property
    def torch_dist_instance(self) -> dist.Distribution:
        """Returns an instance of the torch distribution class"""
        return self.torch_dist(
            **{
                self.stan_to_torch_names[name]: param.torch_parametrization
                for name, param in self._parents.items()
            }
        )

    @property
    def is_hyperparameter(self) -> bool:
        """Returns `True` if all parents are constants. False otherwise."""
        return all(isinstance(parent, Constant) for parent in self.parents)

    @property
    def torch_parametrization(self) -> torch.Tensor:

        # If the parameter is an observable, there is no torch parametrization
        if self.observable:
            raise ValueError("Observables do not have a torch parametrization")

        # Just return the parameter if no bounds
        if (
            self.LOWER_BOUND is None
            and self.UPPER_BOUND is None
            and not self.IS_SIMPLEX
        ):
            return self._torch_parametrization

        # Address bounds. First is if we have both bounds, then we need to transform
        # the parameter to be bounded between the two bounds.
        if self.LOWER_BOUND is not None and self.UPPER_BOUND is not None:
            return self.LOWER_BOUND + (
                self.UPPER_BOUND - self.LOWER_BOUND
            ) * torch.sigmoid(self._torch_parametrization)

        # If not both bounds, then we must have one bound. We assume the parameter
        # is defined in the log space and exponentiate it to get the positive value.
        exp_param = torch.exp(self._torch_parametrization)

        # If a simplex, normalize. We assume that the simplex is the last dimension.
        if self.IS_SIMPLEX:
            return exp_param / torch.sum(exp_param, dim=-1)

        # Now if we only have a lower bound
        elif self.LOWER_BOUND is not None:
            return self.LOWER_BOUND + exp_param

        # If we only have an upper bound
        elif self.UPPER_BOUND is not None:
            return self.UPPER_BOUND - exp_param

        # We should never get here
        raise AssertionError("Invalid bounds")


class ContinuousDistribution(Parameter, TransformableParameter):
    """Base class for parameters represented by continuous distributions."""


class DiscreteDistribution(Parameter):
    """
    Base class for parameters represented by discrete distributions. This is
    more-or-less a passthrough to the Parameter class; however, the default for
    discrete distributions is to set the observable attribute to True.
    """

    BASE_STAN_DTYPE: str = "int"
    LOWER_BOUND: int = 0

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

    def get_transformation_assignment(self, index_opts: tuple[str, ...]) -> str:
        """
        If a hierarchical model is used and this is not a hyperparameter (i.e.,
        its parents are not constants but other parameters), then we want to non-
        center the parameter. This is done by redefining this parameter as the
        transformation of a draw from a unit normal distribution.
        """
        # If this is a hyperparameter or observable, then we use the parent method
        if self.is_hyperparameter or self.observable:
            return super().get_transformation_assignment(index_opts)

        # Get our formattables
        mu_declaration = self._get_formattables(self.mu, index_opts)
        sigma_declaration = self._get_formattables(self.sigma, index_opts)

        # Otherwise, we redefine this parameter as the transformation of a draw
        # from a unit normal distribution
        return (
            f"{self.get_indexed_varname(index_opts)} = {mu_declaration} + {sigma_declaration} "
            f"* {self.get_indexed_varname(index_opts, _name_override=self.noncentered_varname)}"
        )

    def get_target_incrementation(self, index_opts: tuple[str, ...]) -> str:
        """
        If a hierarchical model is used, then the target variable is incremented
        by the log probability of the non-centered parameter. Otherwise, we use
        the parent method.
        """
        # If this is a hyperparameter or observable, then we use the parent method
        if self.is_hyperparameter or self.observable:
            return super().get_target_incrementation(index_opts)

        # Otherwise, we increment the target variable by the log probability of
        # the non-centered parameter
        return (
            self.get_indexed_varname(
                index_opts, _name_override=self.noncentered_varname
            )
            + " ~ std_normal()"
        )

    @property
    def noncentered_varname(self) -> str:
        """Return the non-centered variable name"""
        return f"{self.model_varname}_raw"


class HalfNormal(Normal):
    """Parameter that is represented by the half-normal distribution."""

    LOWER_BOUND: float = 0.0

    def __init__(
        self,
        *,
        sigma: "dms.custom_types.ContinuousParameterType",
        **kwargs,
    ):
        super().__init__(mu=0.0, sigma=sigma, **kwargs)

    # Overwrite the draw method to ensure that the drawn values are positive
    def _draw(
        self, n: int, level_draws: dict[str, npt.NDArray], seed: Optional[int]
    ) -> npt.NDArray:
        return np.abs(super()._draw(n, level_draws, seed=seed))


class UnitNormal(Normal):
    """Parameter that is represented by the unit normal distribution."""

    def __init__(self, **kwargs):
        super().__init__(mu=0.0, sigma=1.0, **kwargs)


class LogNormal(ContinuousDistribution):
    """Parameter that is represented by the log-normal distribution."""

    POSITIVE_PARAMS = {"sigma"}
    LOWER_BOUND: float = 0.0

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
    LOWER_BOUND: float = 0.0
    UPPER_BOUND: float = 1.0

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

    LOWER_BOUND: float = 0.0

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
    LOWER_BOUND: float = 0.0

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
    BASE_STAN_DTYPE: str = "simplex"

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
        *,
        _drawn: Optional[dict["AbstractModelComponent", npt.NDArray]] = None,
        seed: Optional[int] = None,
    ) -> tuple[npt.NDArray, dict["AbstractModelComponent", npt.NDArray]]:
        # There must be a value for `N` in the parameters if we are sampling
        if self._parents.get("N") is None:
            raise ValueError(
                "Sampling from a multinomial distribution is only possible when "
                "'N' is provided'"
            )

        return super().draw(n, _drawn=_drawn, seed=seed)

    def format_stan_code(  # pylint: disable=arguments-differ, unused-argument
        self, N: str, theta: str
    ) -> str:
        return f"multinomial({theta})"
