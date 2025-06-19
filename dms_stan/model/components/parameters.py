"""Holds classes that can be used for defining models in DMS Stan models."""

import inspect

from abc import abstractmethod
from functools import partial
from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.distributions as dist
import torch.nn as nn

from scipy import special

import dms_stan as dms

from dms_stan.model.components import custom_torch_dists
from .abstract_model_component import AbstractModelComponent
from .constants import Constant
from .transformed_data import (
    MultinomialCoefficient,
    SharedAlphaDirichlet,
    TransformedData,
)
from .transformed_parameters import TransformableParameter

# pylint: disable=too-many-lines


def _inverse_transform(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Simple inverse transformation function. Defined at top-level rather than as
    a lambda function to avoid pickling issues.
    """
    return 1 / x


# TODO: Make sure samples from torch distributions obey the same bounds as noted
# in the classes.
# TODO: Make sure samples from Stan distributions obey the same bounds as noted
# in the classes.

# TODO: Most of the arguments to `__init__` of `Parameter` should be converted to
# class attributes, as they are not expected to change per instance.


class Parameter(AbstractModelComponent):
    """Base class for parameters used in DMS Stan"""

    STAN_DIST: str = ""  # The Stan distribution name

    def __init__(
        self,
        numpy_dist: str,
        torch_dist: (
            type[dist.distribution.Distribution]
            | type[custom_torch_dists.CustomDistribution]
        ),
        stan_to_np_names: dict[str, str],
        stan_to_torch_names: dict[str, str],
        stan_to_np_transforms: Optional[
            dict[str, Callable[[npt.NDArray], npt.NDArray]]
        ] = None,
        **kwargs,
    ):
        """
        Sets up random number generation and handles all parameters on which this
        parameter depends.
        """
        # Initialize the parameters
        super().__init__(**kwargs)

        # Make sure the STAN_DIST class attribute is defined
        if self.STAN_DIST == "":
            raise NotImplementedError("The STAN_DIST class attribute must be defined")

        # Parameters can be manually set as observables, so we need a flag to
        # track this
        self._observable = False

        # Store the distributions
        self._numpy_dist = numpy_dist
        self._torch_dist = torch_dist

        # Default value for the transforms dictionary is an empty dictionary
        stan_to_np_transforms = stan_to_np_transforms or {}

        # Identify the parameters. This is anything that is not defined as an argument
        # in the AbstractModelComponent __init__ method.
        parameters = {
            k: v
            for k, v in kwargs.items()
            if k not in inspect.signature(AbstractModelComponent.__init__).parameters
        }

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

        # If no initialization value is provided, then we create one on the range
        # of -1 to 1. This is done by drawing from the distribution.
        if init_val is None:
            init_val = np.squeeze(
                self.get_rng(seed=seed).uniform(
                    low=-1.0, high=1.0, size=(1,) + self.shape
                ),
                axis=0,
            )

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

    def _transform_and_rename_np(
        self, level_draws: dict[str, npt.NDArray]
    ) -> dict[str, npt.NDArray]:
        """Transforms the numpy level draws to the correct format"""
        # Perform transforms
        for name, transform in self.stan_to_np_transforms.items():
            level_draws[name] = transform(level_draws[name])

        # Rename the parameters to the names used by numpy
        return {self.stan_to_np_names[name]: val for name, val in level_draws.items()}

    def _draw(
        self, n: int, level_draws: dict[str, npt.NDArray], seed: Optional[int]
    ) -> npt.NDArray:
        """Sample from the distribution that represents the parameter `n` times"""
        # Perform any necessary transformations and rename the parameters
        level_draws = self._transform_and_rename_np(level_draws)

        # Sample from this distribution using numpy. Alter the shape to account
        # for the new first dimension of length `n`.
        return self.get_numpy_dist(seed=seed)(**level_draws, size=(n,) + self.shape)

    def as_observable(self) -> "Parameter":
        """Redefines the parameter as an observable variable (i.e., data)"""

        # Set the observable attribute to True
        self._observable = True

        # We do not have a torch parameterization for observables
        self._torch_parametrization = None

        return self

    def get_target_incrementation(self, index_opts: tuple[str, ...]) -> str:
        """Return the Stan target incrementation for this parameter."""
        # Determine the left side and operator
        left_side = f"{self.get_indexed_varname(index_opts)} ~ "

        # Get the right-hand-side of the incrementation
        right_side = self.get_right_side(index_opts)

        # Put it all together
        return left_side + right_side

    def get_generated_quantities(self, index_opts: tuple[str, ...]) -> str:
        """Return the Stan code for the generated quantities block."""
        return (
            self.get_indexed_varname(index_opts, _name_override=self.generated_varname)
            + f" = {self.get_right_side(index_opts, dist_suffix='rng')}"
        )

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

    @abstractmethod
    def _write_dist_args(self, **to_format: str) -> str:
        """Writes the distribution arguments in the correct format"""

    def get_right_side(
        self, index_opts: tuple[str, ...] | None, dist_suffix: str = ""
    ) -> str:
        # Get the formattables
        formattables = super().get_right_side(index_opts=index_opts)

        # Build the distribution argument and format the Stan code
        suffix = "" if dist_suffix == "" else f"_{dist_suffix}"
        code = f"{self.STAN_DIST}{suffix}({self._write_dist_args(**formattables)})"

        return code

    def get_transformed_data_declaration(self) -> str:
        """Returns the Stan code for the transformed data block if there is any"""
        # None by default
        return ""

    def __str__(self) -> str:
        right_side = (
            self.get_right_side(None)
            .replace("[start:end]", "")
            .replace("__", ".")
            .capitalize()
        )
        return f"{self.model_varname} ~ {right_side}"

    @property
    def torch_dist(self) -> type["dms.custom_types.DMSStanDistribution"]:
        """Returns the torch distribution class"""
        return self._torch_dist

    @property
    def torch_dist_instance(self) -> "dms.custom_types.DMSStanDistribution":
        """Returns an instance of the torch distribution class"""
        return self.torch_dist(
            **{
                self.stan_to_torch_names[name]: torch.broadcast_to(
                    param.torch_parametrization, self.shape
                )
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
            return exp_param / torch.sum(exp_param, dim=-1, keepdim=True)

        # Now if we only have a lower bound
        elif self.LOWER_BOUND is not None:
            return self.LOWER_BOUND + exp_param

        # If we only have an upper bound
        elif self.UPPER_BOUND is not None:
            return self.UPPER_BOUND - exp_param

        # We should never get here
        raise AssertionError("Invalid bounds")

    @property
    def generated_varname(self) -> str:
        """Return the generated variable name"""
        # Only available for observables
        if not self.observable:
            raise ValueError("Generated variables are only available for observables")

        return f"{self.model_varname}_ppc"

    @property
    def stan_generated_quantity_declaration(self) -> str:
        """Returns the Stan generated quantity declaration for this parameter."""
        return self.declare_stan_variable(self.generated_varname)

    @property
    def observable(self) -> bool:
        """Observable if the parameter has no children or it is set as such."""
        return self._observable or all(
            isinstance(child, TransformedData) for child in self._children
        )


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


class Normal(ContinuousDistribution):
    """Parameter that is represented by the normal distribution."""

    POSITIVE_PARAMS = {"sigma"}
    STAN_DIST = "normal"

    def __init__(
        self,
        *,
        mu: "dms.custom_types.ContinuousParameterType",
        sigma: "dms.custom_types.ContinuousParameterType",
        noncentered: bool = True,
        **kwargs,
    ):
        # Build the instance
        super().__init__(
            numpy_dist="normal",
            torch_dist=dist.normal.Normal,
            stan_to_np_names={"mu": "loc", "sigma": "scale"},
            stan_to_torch_names={"mu": "loc", "sigma": "scale"},
            mu=mu,
            sigma=sigma,
            **kwargs,
        )

        # Are we using non-centered parameterization?
        self._noncentered = noncentered

    def _write_dist_args(  # pylint: disable=arguments-differ
        self, mu: str, sigma: str
    ) -> str:
        return f"{mu}, {sigma}"

    def get_transformation_assignment(self, index_opts: tuple[str, ...]) -> str:
        """
        If a hierarchical model is used and this is not a hyperparameter (i.e.,
        its parents are not constants but other parameters), then we want to non-
        center the parameter. This is done by redefining this parameter as the
        transformation of a draw from a unit normal distribution.
        """
        # If this is centered, then we use the parent method
        if not self.is_noncentered:
            return super().get_transformation_assignment(index_opts)

        # Get our formattables
        formattables = super(Parameter, self).get_right_side(index_opts)
        mu_declaration, sigma_declaration = formattables["mu"], formattables["sigma"]
        raw_declaration = self.get_indexed_varname(
            index_opts, _name_override=self.raw_varname
        )

        # Otherwise, we redefine this parameter as the transformation of a draw
        # from a unit normal distribution
        return (
            f"{self.get_indexed_varname(index_opts)} = {mu_declaration} + {sigma_declaration} "
            f".* {raw_declaration}"
        )

    def get_target_incrementation(self, index_opts: tuple[str, ...]) -> str:
        # Run the parent method
        parent_incrementation = super().get_target_incrementation(index_opts)

        # If not noncentered, we are done
        if not self.is_noncentered:
            return parent_incrementation

        # Otherwise, replace the default variable name with the non-centered variable
        # name
        default_name = self.get_indexed_varname(index_opts)
        new_name = self.get_indexed_varname(index_opts, _name_override=self.raw_varname)

        return parent_incrementation.replace(f"{default_name} ~", f"{new_name} ~")

    def get_right_side_components(self) -> list[AbstractModelComponent]:
        # If noncentered, then there aren't any right-hand-side components
        if self.is_noncentered:
            return []

        # Otherwise, we use the parent method
        return super().get_right_side_components()

    def get_right_side(
        self, index_opts: tuple[str, ...] | None, dist_suffix: str = ""
    ) -> str:
        # If not noncentered, run the parent method
        if not self.is_noncentered:
            return super().get_right_side(index_opts, dist_suffix=dist_suffix)

        # Otherwise, make sure we do not have the suffix set and return the standard
        # normal distribution
        assert (
            dist_suffix == ""
        ), "Non-centered parameters should not have a distribution suffix"

        return "std_normal()"

    @property
    def raw_varname(self) -> str:
        """Return the non-centered variable name"""
        return f"{self.stan_model_varname}_raw"

    @property
    def is_noncentered(self) -> bool:
        """
        Can only be noncentered if the parameter is not a hyperparameter, observable,
        and we did not set `noncentered` to False at initialization.
        """
        return self._noncentered and not self.is_hyperparameter and not self.observable


class HalfNormal(ContinuousDistribution):
    """Parameter that is represented by the half-normal distribution."""

    LOWER_BOUND: float = 0.0
    STAN_DIST = "normal"

    def __init__(
        self,
        *,
        sigma: "dms.custom_types.ContinuousParameterType",
        **kwargs,
    ):
        super().__init__(
            numpy_dist="normal",
            torch_dist=dist.half_normal.HalfNormal,
            stan_to_np_names={"sigma": "scale"},
            stan_to_torch_names={"sigma": "scale"},
            sigma=sigma,
            **kwargs,
        )

    def get_numpy_dist(self, seed: Optional[int] = None) -> Callable[..., npt.NDArray]:
        """Returns the absolute value of the numpy distribution function"""
        base_dist = super().get_numpy_dist(seed=seed)

        def half_normal(**kwargs):
            return np.abs(base_dist(loc=0.0, **kwargs))

        return half_normal

    def _write_dist_args(self, sigma: str) -> str:  # pylint: disable=arguments-differ
        return f"0, {sigma}"


class UnitNormal(Normal):
    """Parameter that is represented by the unit normal distribution."""

    def __init__(self, **kwargs):
        super().__init__(mu=0.0, sigma=1.0, noncentered=False, **kwargs)

        # Sigma is not togglable
        self.sigma.is_togglable = False


class LogNormal(ContinuousDistribution):
    """Parameter that is represented by the log-normal distribution."""

    POSITIVE_PARAMS = {"sigma"}
    LOWER_BOUND: float = 0.0
    STAN_DIST = "lognormal"

    def __init__(
        self,
        *,
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

    def _write_dist_args(  # pylint: disable=arguments-differ
        self, mu: str, sigma: str
    ) -> str:
        return f"{mu}, {sigma}"


class Beta(ContinuousDistribution):
    """Defines the beta distribution."""

    POSITIVE_PARAMS = {"alpha", "beta"}
    LOWER_BOUND: float = 0.0
    UPPER_BOUND: float = 1.0
    STAN_DIST = "beta"

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

    def _write_dist_args(  # pylint: disable=arguments-differ
        self, alpha: str, beta: str
    ) -> str:
        return f"{alpha}, {beta}"


class Gamma(ContinuousDistribution):
    """Defines the gamma distribution."""

    POSITIVE_PARAMS = {"alpha", "beta"}
    LOWER_BOUND: float = 0.0
    STAN_DIST = "gamma"

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
            stan_to_np_transforms={"beta": _inverse_transform},
            alpha=alpha,
            beta=beta,
            **kwargs,
        )

    def _write_dist_args(  # pylint: disable=arguments-differ
        self, alpha: str, beta: str
    ) -> str:
        return f"{alpha}, {beta}"


class InverseGamma(ContinuousDistribution):
    """Defines the inverse gamma distribution."""

    POSITIVE_PARAMS = {"alpha", "beta"}
    LOWER_BOUND: float = 0.0
    STAN_DIST = "inv_gamma"

    def __init__(
        self,
        *,
        alpha: "dms.custom_types.ContinuousParameterType",
        beta: "dms.custom_types.ContinuousParameterType",
        **kwargs,
    ):

        super().__init__(
            numpy_dist="gamma",
            torch_dist=dist.inverse_gamma.InverseGamma,
            stan_to_np_names={"alpha": "shape", "beta": "scale"},
            stan_to_torch_names={"alpha": "concentration", "beta": "rate"},
            stan_to_np_transforms={"beta": _inverse_transform},
            alpha=alpha,
            beta=beta,
            **kwargs,
        )

    def get_numpy_dist(self, seed: Optional[int] = None) -> Callable[..., npt.NDArray]:
        """Builds the numpy distribution function"""
        # Get the base distribution
        np_dist = super().get_numpy_dist(seed=seed)

        def inverse_gamma_dist(*args, **kwargs) -> npt.NDArray[np.floating]:
            return 1 / np_dist(*args, **kwargs)

        return inverse_gamma_dist

    def _write_dist_args(  # pylint: disable=arguments-differ
        self, alpha: str, beta: str
    ) -> str:
        return f"{alpha}, {beta}"


class Exponential(ContinuousDistribution):
    """Defines the exponential distribution."""

    POSITIVE_PARAMS = {"beta"}
    LOWER_BOUND: float = 0.0
    STAN_DIST = "exponential"

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
            stan_to_np_transforms={"beta": _inverse_transform},
            beta=beta,
            **kwargs,
        )

    def _write_dist_args(self, beta: str) -> str:  # pylint: disable=arguments-differ
        return beta


class ExpExponential(Exponential):
    """Defines the Exp-Exponential distribution, which is the distribution of y if
    exp(y) follows an Exponential distribution. Equivalently, if y follows an
    Exponential distribution, then log(y) follows an Exp-Exponential distribution.
    """

    LOWER_BOUND = None
    STAN_DIST = "expexponential"

    def __init__(
        self,
        *,
        beta: "dms.custom_types.ContinuousParameterType",
        **kwargs,
    ):

        super(Exponential, self).__init__(
            numpy_dist="exponential",
            torch_dist=custom_torch_dists.ExpExponential,
            stan_to_np_names={"beta": "scale"},
            stan_to_torch_names={"beta": "rate"},
            stan_to_np_transforms={"beta": _inverse_transform},
            beta=beta,
            **kwargs,
        )

    def get_numpy_dist(self, seed: Optional[int] = None) -> Callable[..., npt.NDArray]:
        # The base distribution is the Exponential distribution
        np_dist = super().get_numpy_dist(seed=seed)

        # Wrap the exponential distribution to take the log of the draw
        def expexponential_dist(
            beta: npt.NDArray,
            size: int | tuple[int, ...] | None = None,
        ) -> npt.NDArray:
            return np.log(np_dist(scale=beta, size=size))

        return expexponential_dist

    def get_supporting_functions(self) -> list[str]:
        # We need to extend the set of supporting functions to include the custom
        # Stan functions for the Exp-Exponential distribution
        return super().get_supporting_functions() + [
            "#include expexponential.stanfunctions"
        ]


class Lomax(ContinuousDistribution):
    """
    Defines the Pareto Type II distribution with the values for mu set to 0 (the
    Lomax distribution).
    """

    LOWER_BOUND: float = 0.0
    POSITIVE_PARAMS = {"lambda_", "alpha"}
    STAN_DIST = "pareto_type_2"

    def __init__(
        self,
        *,
        lambda_: "dms.custom_types.ContinuousParameterType",
        alpha: "dms.custom_types.ContinuousParameterType",
        **kwargs,
    ):

        super().__init__(
            numpy_dist="pareto",
            torch_dist=custom_torch_dists.Lomax,
            stan_to_np_names={
                "lambda_": "lambda_",
                "alpha": "a",
            },  # lambda_ is not used
            stan_to_torch_names={"lambda_": "lambda_", "alpha": "alpha"},
            lambda_=lambda_,
            alpha=alpha,
            **kwargs,
        )

    def get_numpy_dist(self, seed: Optional[int] = None) -> Callable[..., npt.NDArray]:

        # Get the base distribution
        np_dist = super().get_numpy_dist(seed=seed)

        # Wrap the numpy distribution to handle the lambda_ parameter
        def lomax_dist(
            lambda_: npt.NDArray,
            a: npt.NDArray,
            size: int | tuple[int, ...] | None = None,
        ) -> npt.NDArray:

            # Call the base distribution with the 'a' parameter. This is because
            # the numpy inbuilt assumes that lambda_ = 1.
            base_draw = np_dist(a=a, size=size)

            # Now we need to scale the draw appropriately to account for different
            # values of lambda_
            return base_draw * lambda_

        return lomax_dist

    def _write_dist_args(  # pylint: disable=arguments-differ
        self, lambda_: str, alpha: str
    ) -> str:
        return f"0.0, {lambda_}, {alpha}"


class ExpLomax(Lomax):
    """
    Defines the Exp-Lomax distribution, which is the distribution of y if exp(y)
    follows a Lomax distribution. Equivalently, if y follows a Lomax distribution,
    then log(y) follows an Exp-Lomax distribution.
    """

    LOWER_BOUND = None
    STAN_DIST = "explomax"

    def __init__(
        self,
        *,
        lambda_: "dms.custom_types.ContinuousParameterType",
        alpha: "dms.custom_types.ContinuousParameterType",
        **kwargs,
    ):

        super(Lomax, self).__init__(
            numpy_dist="pareto",
            torch_dist=custom_torch_dists.ExpLomax,
            stan_to_np_names={
                "lambda_": "lambda_",
                "alpha": "a",
            },  # lambda_ is not used
            stan_to_torch_names={"lambda_": "lambda_", "alpha": "alpha"},
            lambda_=lambda_,
            alpha=alpha,
            **kwargs,
        )

    def get_numpy_dist(self, seed: Optional[int] = None) -> Callable[..., npt.NDArray]:
        # The base distribution is the Lomax distribution
        np_dist = super().get_numpy_dist(seed=seed)

        # Wrap the lomax distribution to take the log of the draw
        def explomax_dist(
            lambda_: npt.NDArray,
            a: npt.NDArray,
            size: int | tuple[int, ...] | None = None,
        ) -> npt.NDArray:
            return np.log(np_dist(lambda_=lambda_, a=a, size=size))

        return explomax_dist

    def _write_dist_args(self, lambda_: str, alpha: str) -> str:
        return f"{lambda_}, {alpha}"

    def get_supporting_functions(self) -> list[str]:
        # We need to extend the set of supporting functions to include the custom
        # Stan functions for the Exp-Lomax distribution
        return super().get_supporting_functions() + ["#include explomax.stanfunctions"]


class Dirichlet(ContinuousDistribution):
    """Defines the Dirichlet distribution."""

    BASE_STAN_DTYPE = "simplex"
    IS_SIMPLEX = True
    STAN_DIST = "dirichlet"
    POSITIVE_PARAMS = {"alpha"}

    def __init__(
        self,
        *,
        alpha: Union[AbstractModelComponent, npt.ArrayLike],
        **kwargs,
    ):
        # If a float or int is provided, then "shape" must be provided too. We will
        # create a numpy array filled of that shape filled with the value
        enforce_uniformity = True
        if isinstance(alpha, (float, int)):
            if "shape" not in kwargs:
                raise ValueError(
                    "If alpha is a float or int, then shape must be provided"
                )
            alpha = np.full(kwargs["shape"], float(alpha))
        elif isinstance(alpha, Constant) and isinstance(alpha.value, (float, int)):
            alpha.value = np.full(alpha.shape, float(alpha.value))
        else:
            enforce_uniformity = False

        # Run the parent class's init
        super().__init__(
            numpy_dist="dirichlet",
            torch_dist=dist.dirichlet.Dirichlet,
            stan_to_np_names={"alpha": "alpha"},
            stan_to_torch_names={"alpha": "concentration"},
            alpha=alpha,
            **kwargs,
        )

        # Set `enforce_uniformity` appropriately
        self.alpha.enforce_uniformity = enforce_uniformity

    def get_numpy_dist(self, seed: Optional[int] = None) -> Callable[..., npt.NDArray]:
        """
        The dirichlet distribution in numpy cannot be batched. This is a wrapper
        around that distribution to allow for batching.
        """
        # Get the base distribution
        np_dist = super().get_numpy_dist(seed=seed)

        def dirichlet_dist(
            alpha: npt.NDArray, size: int | tuple[int, ...] | None = None
        ) -> npt.NDArray:

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

            # Sample from the Dirichlet distribution according to the batch dims
            return np.stack(
                [np_dist(alpha, size=batch_dims) for alpha in alphas],
                axis=len(batch_dims),
            ).reshape(size)

        return dirichlet_dist

    def _write_dist_args(self, alpha: str) -> str:  # pylint: disable=arguments-differ
        return alpha


class ExpDirichlet(Dirichlet):
    """Defines the Exp-Dirichlet distribution, which is the distribution of y if
    exp(y) follows a Dirichlet distribution. Equivalently, if y follows a Dirichlet
    distribution, then log(y) follows an Exp-Dirichlet distribution.
    """

    BASE_STAN_DTYPE = "real"
    IS_SIMPLEX = False
    IS_LOG_SIMPLEX = True
    LOWER_BOUND = None
    UPPER_BOUND = 0.0
    STAN_DIST = "expdirichlet"

    def __init__(
        self,
        *,
        alpha: Union[AbstractModelComponent, npt.ArrayLike],
        **kwargs,
    ):
        super(Dirichlet, self).__init__(
            numpy_dist="dirichlet",
            torch_dist=custom_torch_dists.ExpDirichlet,
            stan_to_np_names={"alpha": "alpha"},
            stan_to_torch_names={"alpha": "concentration"},
            alpha=alpha,
            **kwargs,
        )

    def get_numpy_dist(self, seed: Optional[int] = None) -> Callable[..., npt.NDArray]:
        # The base distribution is the Dirichlet distribution
        np_dist = super().get_numpy_dist(seed=seed)

        # Wrap the dirichlet distribution to take the log of the draw
        def expdirichlet_dist(
            alpha: npt.NDArray, size: int | tuple[int, ...] | None = None
        ) -> npt.NDArray:
            return np.log(np_dist(alpha=alpha, size=size))

        return expdirichlet_dist

    def get_supporting_functions(self) -> list[str]:
        # We need to extend the set of supporting functions to include the custom
        # Stan functions for the Exp-Dirichlet distribution
        return super().get_supporting_functions() + [
            "#include expdirichlet.stanfunctions"
        ]

    def get_transformation_assignment(self, index_opts: tuple[str, ...]) -> str:
        """
        There is no 'log simplex' type in Stan, so we need to redefine the base
        vector definition to be constrained to the log simplex.
        """
        # We constrain and adjust the Jacobian for the transformation
        raw_varname = self.get_indexed_varname(
            index_opts, _name_override=self.raw_varname
        )
        transformed_varname = self.get_indexed_varname(index_opts)

        return f"{transformed_varname} = logsoftmax_transform_lp({raw_varname})"

    def get_right_side(
        self, index_opts: tuple[str, ...] | None, dist_suffix: str = ""
    ) -> str:
        # If no suffix is provided, determine whether we are using the normalized
        # or unnormalized version of the distribution. We use unnormalized when
        # the parameter is a hyperparameter with no parents.
        if dist_suffix == "":
            dist_suffix = "unnorm" if self.is_hyperparameter else "norm"

        # Now we just run the parent method
        return super().get_right_side(index_opts, dist_suffix=dist_suffix)

    @property
    def raw_varname(self) -> str:
        """Return the raw variable name for the Exp-Dirichlet distribution"""
        return f"{self.stan_model_varname}_raw"


class Binomial(DiscreteDistribution):
    """Parameter that is represented by the binomial distribution"""

    POSITIVE_PARAMS = {"theta", "N"}
    STAN_DIST = "binomial"

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

    def _write_dist_args(  # pylint: disable=arguments-differ
        self, N: str, theta: str
    ) -> str:
        return f"{N}, {theta}"


class Poisson(DiscreteDistribution):
    """Parameter that is represented by the Poisson distribution."""

    POSITIVE_PARAMS = {"lambda_"}
    STAN_DIST = "poisson"

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

    def _write_dist_args(self, lambda_: str) -> str:  # pylint: disable=arguments-differ
        return lambda_


class _MultinomialBase(DiscreteDistribution):
    """Defines the base multinomial distribution."""

    def __init__(
        self,
        *,
        torch_dist: type[custom_torch_dists.CustomDistribution],
        stan_to_np_names: dict[str, str],
        stan_to_torch_names: dict[str, str],
        N: Union[AbstractModelComponent, int, npt.NDArray[np.integer]],
        **kwargs,
    ):

        # Run the parent class's init
        super().__init__(
            numpy_dist="multinomial",
            torch_dist=torch_dist,
            stan_to_np_names=stan_to_np_names,
            stan_to_torch_names=stan_to_torch_names,
            stan_to_np_transforms={"N": partial(np.squeeze, axis=-1)},
            N=N,
            **kwargs,
        )

    def get_numpy_dist(self, seed: Optional[int] = None) -> Callable[..., npt.NDArray]:
        """Returns the multinomial distribution function"""
        # Get the base distribution
        np_dist = super().get_numpy_dist(seed=seed)

        # The last dimension is ignored in the multinomial distribution by default
        def multinomial_dist(
            n: int | npt.NDArray[np.integer],
            pvals: npt.NDArray[np.floating],
            size: int | tuple[int, ...] | None = None,
        ) -> npt.NDArray[np.integer]:
            # The dimensions of `n` must equal the leading dimensions of `pvals`
            if isinstance(n, np.ndarray) and n.shape != pvals.shape[:-1]:
                raise ValueError(
                    f"Dimensions of `n` ({n.shape}) must equal the leading dimensions "
                    f"of `pvals` ({pvals.shape[:-1]})"
                )

            # Set the size
            if size is None:
                size = pvals.shape
            elif isinstance(size, int):
                size = (size, *pvals.shape)
            else:
                size = tuple(size)

            # The last dimension of the size must match the shape of the pvals
            if size[-1] != pvals.shape[-1]:
                raise ValueError(
                    f"Last dimension of the size ({size}) must match the shape of "
                    f"the pvals ({pvals.shape[-1]})"
                )

            # Run the base distribution, ignoring the last dimension
            return np_dist(n=n, pvals=pvals, size=size[:-1])

        return multinomial_dist

    def get_target_incrementation(self, index_opts: tuple[str, ...]) -> str:
        # We need to strip the N parameter from the declaration as this is implicit
        # in the distribution as defined in Stan
        raw = super().get_target_incrementation(index_opts)

        # Remove the N parameter
        assert raw.count(", ") == 1, "Invalid target incrementation: " + raw
        raw, _ = raw.split(", ")
        return raw + ")"


class Multinomial(_MultinomialBase):
    """Defines the multinomial distribution."""

    SIMPLEX_PARAMS = {"theta"}
    STAN_DIST = "multinomial"

    def __init__(
        self,
        *,
        theta: "dms.custom_types.ContinuousParameterType",
        N: "dms.custom_types.DiscreteParameterType",
        **kwargs,
    ):
        super().__init__(
            torch_dist=custom_torch_dists.Multinomial,
            stan_to_np_names={"N": "n", "theta": "pvals"},
            stan_to_torch_names={"N": "total_count", "theta": "probs"},
            theta=theta,
            N=N,
            **kwargs,
        )

    def _write_dist_args(  # pylint: disable=arguments-differ
        self, theta: str, N: str
    ) -> str:
        return f"{theta}, {N}"


class MultinomialLogit(_MultinomialBase):
    """
    Defines the multinomial distribution for modeling logit-transformed simplex
    parameters. In other words, this is the multinomial distribution parametrized
    by `ln(theta)` rather than `theta`. This is useful for modeling extremely
    high-dimensional multinomial distributions where the simplex parameterization
    is numerically unstable.
    """

    STAN_DIST = "multinomial_logit"

    def __init__(
        self,
        *,
        gamma: "dms.custom_types.ContinuousParameterType",
        N: "dms.custom_types.DiscreteParameterType",
        **kwargs,
    ):
        super().__init__(
            torch_dist=custom_torch_dists.MultinomialLogit,
            stan_to_np_names={"N": "n", "gamma": "logits"},
            stan_to_torch_names={"N": "total_count", "gamma": "logits"},
            gamma=gamma,
            N=N,
            **kwargs,
        )

    def _write_dist_args(  # pylint: disable=arguments-differ
        self, gamma: str, N: str
    ) -> str:
        return f"{gamma}, {N}"

    def get_numpy_dist(self, seed: Optional[int] = None) -> Callable[..., npt.NDArray]:
        """
        Override the numpy distribution of the multinomial distribution to apply the
        log transformation.
        """

        # The new function applies the softmax transformation to the output of the
        # multinomial distribution (over the last dimension)
        base_dist = super().get_numpy_dist(seed=seed)

        def multinomial_logit(
            n: int | npt.NDArray[np.integer],
            logits: npt.NDArray[np.floating],
            size: int | tuple[int, ...] | None = None,
        ):
            # Run the base distribution with the logits softmaxed
            return base_dist(n=n, pvals=special.softmax(logits, axis=-1), size=size)

        return multinomial_logit
