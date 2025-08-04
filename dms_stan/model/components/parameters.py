"""Holds classes that can be used for defining models in DMS Stan models."""

import re

from abc import ABCMeta
from functools import partial
from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.distributions as dist
import torch.nn as nn

from scipy import stats

import dms_stan
from dms_stan.model.components import abstract_model_component, constants
from dms_stan.model.components.custom_distributions import (
    custom_scipy_dists,
    custom_torch_dists,
)
from dms_stan.model.components.transformations import (
    cdfs,
    transformed_data,
    transformed_parameters,
)


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


class ParameterMeta(ABCMeta):
    """
    Metaclass for Parameter subclass construction. This makes sure that each subclass
    has an appropriately defined `CDF`, `SF`, `LOG_CDF`, and `LOG_SF` class.
    """

    def __init__(cls, name, bases, attrs):
        """
        Assigns the CDF, SF, LOG_CDF, and LOG_SF classes to the Parameter subclass.
        """
        # Run the parent class's __init__ method
        super().__init__(name, bases, attrs)

        # Add CDF, SF, LOG_CDF, and LOG_SF classes to the Parameter subclass
        cls.CDF = type(f"{name}CDF", (cdfs.CDF,), {"PARAMETER": cls})
        cls.SF = type(f"{name}SF", (cdfs.SurvivalFunction,), {"PARAMETER": cls})
        cls.LOG_CDF = type(f"{name}LOG_CDF", (cdfs.LogCDF,), {"PARAMETER": cls})
        cls.LOG_SF = type(
            f"{name}LOG_SF", (cdfs.LogSurvivalFunction,), {"PARAMETER": cls}
        )


class Parameter(
    abstract_model_component.AbstractModelComponent, metaclass=ParameterMeta
):
    """Base class for parameters used in DMS Stan"""

    STAN_DIST: str = ""  # The Stan distribution name
    HAS_RAW_VARNAME: bool = False  # Whether the parameter has a raw variable name
    CDF = None  # Transformed parameter class for the CDF
    SF = None  # Transformed parameter class for the survival function (CCDF)
    LOG_CDF = None  # Transformed parameter class for the log CDF
    LOG_SF = None  # Transformed parameter for the log survival function (log CCDF)
    SCIPY_DIST: type[stats.rv_continuous] | type[stats.rv_discrete] | None = (
        None  # The SciPy distribution
    )
    TORCH_DIST: (
        type[dist.distribution.Distribution]
        | type[custom_torch_dists.CustomDistribution]
        | None
    ) = None  # The PyTorch distribution class
    STAN_TO_SCIPY_NAMES: dict[str, str] = {}  # Mapping from Stan to SciPy names
    STAN_TO_TORCH_NAMES: dict[str, str] = {}  # Mapping from Stan to PyTorch names
    STAN_TO_SCIPY_TRANSFORMS: dict[str, Callable[[npt.NDArray], npt.NDArray]] = {}

    def __init__(self, **kwargs):
        """
        Sets up random number generation and handles all parameters on which this
        parameter depends.
        """
        # Confirm that class attributes are set correctly
        if missing_attributes := [
            attr
            for attr in (
                "STAN_DIST",
                "CDF",
                "CCDF",
                "LOG_CDF",
                "LOG_CCDF",
                "SCIPY_DIST",
                "TORCH_DIST",
                "STAN_TO_SCIPY_NAMES",
                "STAN_TO_TORCH_NAMES",
            )
            if not getattr(self, attr)
        ]:
            raise NotImplementedError(
                f"The following class attributes must be defined: {', '.join(missing_attributes)}"
            )

        # Make sure we have the expected parameters
        kwargset = set(kwargs.keys())
        if additional_params := kwargset - self.STAN_TO_SCIPY_NAMES.keys():
            raise TypeError(
                f"Unexpected parameters {additional_params} passed to {self}."
            )
        if missing_params := self.STAN_TO_SCIPY_NAMES.keys() - kwargset:
            raise TypeError(f"Missing parameters {missing_params} for {self}.")

        # Initialize the parameters
        super().__init__(**kwargs)

        # Parameters can be manually set as observables, so we need a flag to
        # track this
        self._observable = False

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

    def _draw(
        self, n: int, level_draws: dict[str, npt.NDArray], seed: Optional[int]
    ) -> dict[str, npt.NDArray]:
        """
        Applies the appropriate transforms to the scipy draws from a parent parameter
        such that we can sample from the scipy distribution of this parameter.
        """
        # Transform and rename the draws from the previous level.
        level_draws = {
            self.STAN_TO_SCIPY_NAMES[name]: self.STAN_TO_SCIPY_TRANSFORMS.get(
                name, lambda x: x
            )(draw)
            for name, draw in level_draws.items()
        }

        # Draw from the scipy distribution
        return self.scipy_dist_instance.rvs(
            **level_draws, size=(n,) + self.shape, random_state=seed
        )

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

        return self.torch_dist_instance.log_prob(
            observed if self.observable else self.torch_parametrization
        ).sum()

    def get_rng(self, seed: Optional[int] = None) -> np.random.Generator:
        """Return the random number generator"""
        # Return the global random number generator if no seed is provided. Otherwise,
        # return a new random number generator with the provided seed.
        if seed is None:
            return dms_stan.RNG
        return np.random.default_rng(seed)

    def write_dist_args(self, **to_format: str) -> str:
        """
        Writes the distribution arguments in the correct format. Default is to concatenate
        values with commas as delimiters. The order is set up `STAN_TO_SCIPY_NAMES`.
        """
        return ", ".join(to_format[name] for name in self.STAN_TO_SCIPY_NAMES)

    def get_right_side(
        self, index_opts: tuple[str, ...] | None, dist_suffix: str = ""
    ) -> str:
        # Get the formattables
        formattables = super().get_right_side(index_opts=index_opts)

        # Build the distribution argument and format the Stan code
        suffix = "" if dist_suffix == "" else f"_{dist_suffix}"
        code = f"{self.STAN_DIST}{suffix}({self.write_dist_args(**formattables)})"

        return code

    def get_transformed_data_declaration(self) -> str:
        """Returns the Stan code for the transformed data block if there is any"""
        # None by default
        return ""

    def cdf(
        self: Union["Parameter", None] = None,
        **params: "dms.custom_types.CombinableParameterType",
    ) -> cdfs.CDF:
        """
        Can be used as a class method or instance method to return the CDF of the
        parameter. If called as a class method, the parameters must be provided.
        Otherwise, the parent parameters are used to calculate the CDF.
        """

    def ccdf(
        self: Union["Parameter", None] = None,
        **params: "dms.custom_types.CombinableParameterType",
    ) -> cdfs.SurvivalFunction:
        """
        Can be used as a class method or instance method to return the complementary
        CDF of the parameter. If called as a class method, the parameters must be
        provided. Otherwise, the parent parameters are used to calculate the CCDF.
        """

    def log_cdf(
        self: Union["Parameter", None] = None,
        **params: "dms.custom_types.CombinableParameterType",
    ) -> cdfs.LogCDF:
        """
        Can be used as a class method or instance method to return the log CDF of the
        parameter. If called as a class method, the parameters must be provided.
        Otherwise, the parent parameters are used to calculate the log CDF.
        """

    def log_ccdf(
        self: Union["Parameter", None] = None,
        **params: "dms.custom_types.CombinableParameterType",
    ) -> cdfs.LogSurvivalFunction:
        """
        Can be used as a class method or instance method to return the log CCDF of the
        parameter. If called as a class method, the parameters must be provided.
        Otherwise, the parent parameters are used to calculate the log CCDF.
        """

    def __str__(self) -> str:
        right_side = (
            self.get_right_side(None)
            .replace("[start:end]", "")
            .replace("__", ".")
            .capitalize()
        )
        return f"{self.model_varname} ~ {right_side}"

    @property
    def torch_dist_instance(self) -> "dms.custom_types.DMSStanDistribution":
        """Returns an instance of the torch distribution class"""
        return self.TORCH_DIST(  # pylint: disable=not-callable
            **{
                self.STAN_TO_TORCH_NAMES[name]: torch.broadcast_to(
                    param.torch_parametrization, self.shape
                )
                for name, param in self._parents.items()
            }
        )

    @property
    def scipy_dist_instance(self) -> stats.rv_continuous | stats.rv_discrete:
        """Returns an instance of the scipy distribution class"""
        return self.SCIPY_DIST()  # pylint: disable=not-callable

    @property
    def is_hyperparameter(self) -> bool:
        """Returns `True` if all parents are constants. False otherwise."""
        return all(isinstance(parent, constants.Constant) for parent in self.parents)

    @property
    def torch_parametrization(self) -> torch.Tensor:

        # If the parameter is an observable, there is no torch parametrization
        if self.observable:
            raise ValueError("Observables do not have a torch parametrization")

        # Just return the parameter if there are no bounds
        if (
            self.LOWER_BOUND is None
            and self.UPPER_BOUND is None
            and not self.IS_SIMPLEX
            and not self.IS_LOG_SIMPLEX
        ):
            return self._torch_parametrization

        # Set bounds where we have both upper and lower
        if self.IS_SIMPLEX:
            return torch.softmax(self._torch_parametrization, dim=-1)
        elif self.IS_LOG_SIMPLEX:
            return torch.log_softmax(self._torch_parametrization, dim=-1)
        elif self.LOWER_BOUND is not None and self.UPPER_BOUND is not None:
            return self.LOWER_BOUND + (
                self.UPPER_BOUND - self.LOWER_BOUND
            ) * torch.sigmoid(self._torch_parametrization)

        # If not both bounds, then we must have one bound. We assume the parameter
        # is defined in the log space and exponentiate it to get the positive value.
        exp_param = torch.exp(self._torch_parametrization)

        # Now if we only have a lower bound
        if self.LOWER_BOUND is not None:
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
            isinstance(child, transformed_data.TransformedData)
            for child in self._children
        )

    @property
    def raw_varname(self) -> str:
        """
        Some parameters are defined in terms of others. That "other" parameter's
        name is the raw variable name. This returns an empty string by default,
        indicating no raw variable name.
        """
        return f"{self.stan_model_varname}_raw" if self.HAS_RAW_VARNAME else ""

    @property
    def raw_stan_parameter_declaration(self) -> str:
        """Declares the raw Stan parameter for this parameter."""
        if self.HAS_RAW_VARNAME:
            return self.declare_stan_variable(self.raw_varname)

        return ""


class ContinuousDistribution(Parameter, transformed_parameters.TransformableParameter):
    """Base class for parameters represented by continuous distributions."""


class DiscreteDistribution(Parameter):
    """Base class for parameters represented by discrete distributions"""

    BASE_STAN_DTYPE: str = "int"
    LOWER_BOUND: int = 0


class Normal(ContinuousDistribution):
    """Parameter that is represented by the normal distribution."""

    POSITIVE_PARAMS = {"sigma"}
    STAN_DIST = "normal"
    SCIPY_DIST = stats.norm
    TORCH_DIST = dist.normal.Normal
    STAN_TO_SCIPY_NAMES = {"mu": "loc", "sigma": "scale"}
    STAN_TO_TORCH_NAMES = {"mu": "loc", "sigma": "scale"}

    def __init__(
        self,
        *,
        mu: "dms.custom_types.ContinuousParameterType",
        sigma: "dms.custom_types.ContinuousParameterType",
        noncentered: bool = True,
        **kwargs,
    ):
        # Build the instance
        super().__init__(mu=mu, sigma=sigma, **kwargs)

        # Are we using non-centered parameterization?
        self._noncentered = noncentered

    def write_dist_args(  # pylint: disable=arguments-differ
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

    def get_right_side_components(
        self,
    ) -> list[abstract_model_component.AbstractModelComponent]:
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
    def is_noncentered(self) -> bool:
        """
        Can only be noncentered if the parameter is not a hyperparameter, observable,
        and we did not set `noncentered` to False at initialization.
        """
        return self._noncentered and not self.is_hyperparameter and not self.observable

    HAS_RAW_VARNAME = is_noncentered  # Raw varname only if noncentered


class ExpNormal(ContinuousDistribution):
    """
    A parameter whose exponential is normally distributed. This is distinct from
    the LogNormal distribution, which describes a parameter whose logarithm is
    normally distributed.
    """

    POSITIVE_PARAMS = {"sigma"}
    STAN_DIST = "expnormal"
    SCIPY_DIST = custom_scipy_dists.expnormal
    TORCH_DIST = custom_torch_dists.ExpNormal
    STAN_TO_SCIPY_NAMES = {"mu": "loc", "sigma": "scale"}
    STAN_TO_TORCH_NAMES = {"mu": "loc", "sigma": "scale"}

    def get_supporting_functions(self) -> list[str]:
        """
        Returns the Stan functions that are needed to support this distribution.
        """
        # We need the expnormal function for the Stan model
        return super().get_supporting_functions() + ["#include expnormal.stanfunctions"]


class HalfNormal(ContinuousDistribution):
    """Parameter that is represented by the half-normal distribution."""

    LOWER_BOUND: float = 0.0
    STAN_DIST = "normal"
    SCIPY_DIST = stats.halfnorm
    TORCH_DIST = dist.half_normal.HalfNormal
    STAN_TO_SCIPY_NAMES = {"sigma": "scale"}
    STAN_TO_TORCH_NAMES = {"sigma": "scale"}

    def write_dist_args(self, sigma: str) -> str:  # pylint: disable=arguments-differ
        return f"0, {sigma}"


class UnitNormal(Normal):
    """Parameter that is represented by the unit normal distribution."""

    STAN_DIST = "std_normal"

    def __init__(self, **kwargs):
        super().__init__(mu=0.0, sigma=1.0, noncentered=False, **kwargs)

        # Sigma is not togglable
        self.sigma.is_togglable = False

    def write_dist_args(
        self, mu: str, sigma: str
    ) -> str:  # pylint: disable=arguments-differ
        # No arguments needed for the unit normal distribution in Stan.
        return ""


class LogNormal(ContinuousDistribution):
    """Parameter that is represented by the log-normal distribution."""

    POSITIVE_PARAMS = {"sigma"}
    LOWER_BOUND: float = 0.0
    STAN_DIST = "lognormal"
    SCIPY_DIST = stats.lognorm
    TORCH_DIST = dist.log_normal.LogNormal
    STAN_TO_SCIPY_NAMES = {"mu": "loc", "sigma": "scale"}
    STAN_TO_TORCH_NAMES = {"mu": "loc", "sigma": "scale"}


class Beta(ContinuousDistribution):
    """Defines the beta distribution."""

    POSITIVE_PARAMS = {"alpha", "beta"}
    LOWER_BOUND: float = 0.0
    UPPER_BOUND: float = 1.0
    STAN_DIST = "beta"
    SCIPY_DIST = stats.beta
    TORCH_DIST = dist.beta.Beta
    STAN_TO_SCIPY_NAMES = {"alpha": "a", "beta": "b"}
    STAN_TO_TORCH_NAMES = {"alpha": "concentration1", "beta": "concentration0"}


class Gamma(ContinuousDistribution):
    """Defines the gamma distribution."""

    POSITIVE_PARAMS = {"alpha", "beta"}
    LOWER_BOUND: float = 0.0
    STAN_DIST = "gamma"
    SCIPY_DIST = stats.gamma
    TORCH_DIST = dist.gamma.Gamma
    STAN_TO_SCIPY_NAMES = {"alpha": "a", "beta": "scale"}
    STAN_TO_TORCH_NAMES = {"alpha": "concentration", "beta": "rate"}
    STAN_TO_SCIPY_TRANSFORMS = {
        "beta": _inverse_transform
    }  # Transform beta to match the scipy distribution's scale parameter


class InverseGamma(ContinuousDistribution):
    """Defines the inverse gamma distribution."""

    POSITIVE_PARAMS = {"alpha", "beta"}
    LOWER_BOUND: float = 0.0
    STAN_DIST = "inv_gamma"
    SCIPY_DIST = stats.invgamma
    TORCH_DIST = dist.inverse_gamma.InverseGamma
    STAN_TO_SCIPY_NAMES = {"alpha": "a", "beta": "scale"}
    STAN_TO_TORCH_NAMES = {"alpha": "concentration", "beta": "rate"}


class Exponential(ContinuousDistribution):
    """Defines the exponential distribution."""

    POSITIVE_PARAMS = {"beta"}
    LOWER_BOUND: float = 0.0
    STAN_DIST = "exponential"
    SCIPY_DIST = stats.expon
    TORCH_DIST = dist.exponential.Exponential
    STAN_TO_SCIPY_NAMES = {"beta": "scale"}
    STAN_TO_TORCH_NAMES = {"beta": "rate"}
    STAN_TO_SCIPY_TRANSFORMS = {
        "beta": _inverse_transform
    }  # Transform beta to match the scipy distribution's scale parameter


class ExpExponential(Exponential):
    """Defines the Exp-Exponential distribution, which is the distribution of y if
    exp(y) follows an Exponential distribution. Equivalently, if y follows an
    Exponential distribution, then log(y) follows an Exp-Exponential distribution.
    """

    LOWER_BOUND = None
    STAN_DIST = "expexponential"
    SCIPY_DIST = custom_scipy_dists.expexponential
    TORCH_DIST = custom_torch_dists.ExpExponential
    STAN_TO_SCIPY_NAMES = {"beta": "scale"}
    STAN_TO_TORCH_NAMES = {"beta": "rate"}
    STAN_TO_SCIPY_TRANSFORMS = {
        "beta": _inverse_transform
    }  # Transform beta to match the scipy distribution's scale parameter

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
    SCIPY_DIST = stats.lomax
    TORCH_DIST = custom_torch_dists.Lomax
    STAN_TO_SCIPY_NAMES = {"lambda_": "scale", "alpha": "c"}
    STAN_TO_TORCH_NAMES = {"lambda_": "lambda_", "alpha": "alpha"}

    def write_dist_args(  # pylint: disable=arguments-differ
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
    SCIPY_DIST = custom_scipy_dists.explomax
    TORCH_DIST = custom_torch_dists.ExpLomax
    STAN_TO_SCIPY_NAMES = {"lambda_": "scale", "alpha": "c"}
    STAN_TO_TORCH_NAMES = {"lambda_": "lambda_", "alpha": "alpha"}

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
    SCIPY_DIST = custom_scipy_dists.dirichlet
    TORCH_DIST = dist.dirichlet.Dirichlet
    STAN_TO_SCIPY_NAMES = {"alpha": "alpha"}
    STAN_TO_TORCH_NAMES = {"alpha": "concentration"}

    def __init__(
        self,
        *,
        alpha: Union[abstract_model_component.AbstractModelComponent, npt.ArrayLike],
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
        elif isinstance(alpha, constants.Constant) and isinstance(
            alpha.value, (float, int)
        ):
            alpha.value = np.full(alpha.shape, float(alpha.value))
        else:
            enforce_uniformity = False

        # Run the init method of the class one level up in the MRO hierarchy
        super().__init__(alpha=alpha, **kwargs)

        # Set `enforce_uniformity` appropriately
        self.alpha.enforce_uniformity = enforce_uniformity  # pylint: disable=no-member


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
    HAS_RAW_VARNAME = True
    SCIPY_DIST = custom_scipy_dists.expdirichlet
    TORCH_DIST = custom_torch_dists.ExpDirichlet
    STAN_TO_SCIPY_NAMES = {"alpha": "alpha"}
    STAN_TO_TORCH_NAMES = {"alpha": "concentration"}

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

        return (
            f"{transformed_varname} = inv_ilr_log_simplex_constrain_lp({raw_varname})"
        )

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
    def raw_stan_parameter_declaration(self) -> str:
        """
        Declares the raw Stan parameter for this parameter. We must account for
        the fact that the raw parameter has K - 1 dimensions, where K is the number
        of categories in the Dirichlet distribution.
        """
        # Run the parent method to get the raw variable name
        raw_varname = super().raw_stan_parameter_declaration

        # We should always have a raw variable name if we are using the Exp-Dirichlet
        assert raw_varname, "Raw variable name should not be empty for Exp-Dirichlet"

        # Every ExpDirichlet will have a 'vector<upper=0.0>[K]' datatype. We want
        # to split the raw variable name on this signature
        array_info, index, var_info = re.match(
            r"(.*)vector<upper=0.0>\[([0-9]+)\](.+)", raw_varname
        ).groups()
        index = int(index) - 1  # Remove extra dimension for the raw variable

        # Reconstruct the raw variable name with the correct number of dimensions.
        # Note that there is no upper bound on the raw variable
        return f"{array_info}vector[{index}]{var_info}"


class Binomial(DiscreteDistribution):
    """Parameter that is represented by the binomial distribution"""

    POSITIVE_PARAMS = {"theta", "N"}
    STAN_DIST = "binomial"
    SCIPY_DIST = stats.binom
    TORCH_DIST = dist.binomial.Binomial
    STAN_TO_SCIPY_NAMES = {"N": "n", "theta": "p"}
    STAN_TO_TORCH_NAMES = {"N": "total_count", "theta": "probs"}


class Poisson(DiscreteDistribution):
    """Parameter that is represented by the Poisson distribution."""

    POSITIVE_PARAMS = {"lambda_"}
    STAN_DIST = "poisson"
    SCIPY_DIST = stats.poisson
    TORCH_DIST = dist.poisson.Poisson
    STAN_TO_SCIPY_NAMES = {"lambda_": "mu"}
    STAN_TO_TORCH_NAMES = {"lambda_": "rate"}


class _MultinomialBase(DiscreteDistribution):
    """Defines the base multinomial distribution."""

    STAN_TO_NP_TRANSFORMS = {
        "N": partial(np.squeeze, axis=-1)
    }  # Squeeze the N parameter to match the numpy distribution's expected shape

    def get_target_incrementation(self, index_opts: tuple[str, ...]) -> str:
        # We need to strip the N parameter from the declaration as this is implicit
        # in the distribution as defined in Stan
        raw = super().get_target_incrementation(index_opts)

        # Remove the N parameter
        *raw, _ = raw.split(", ")
        return ", ".join(raw) + ")"


class Multinomial(_MultinomialBase):
    """Defines the multinomial distribution."""

    SIMPLEX_PARAMS = {"theta"}
    STAN_DIST = "multinomial"
    SCIPY_DIST = custom_scipy_dists.multinomial
    TORCH_DIST = custom_torch_dists.Multinomial
    STAN_TO_SCIPY_NAMES = {"theta": "p", "N": "n"}
    STAN_TO_TORCH_NAMES = {"theta": "probs", "N": "total_count"}


class MultinomialLogit(_MultinomialBase):
    """
    Defines the multinomial distribution for modeling logit-transformed simplex
    parameters. In other words, this is the multinomial distribution parametrized
    by `ln(theta)` rather than `theta`. This is useful for modeling extremely
    high-dimensional multinomial distributions where the simplex parameterization
    is numerically unstable.
    """

    STAN_DIST = "multinomial_logit"
    SCIPY_DIST = custom_scipy_dists.multinomial_logit
    TORCH_DIST = custom_torch_dists.MultinomialLogit
    STAN_TO_SCIPY_NAMES = {"gamma": "logits", "N": "n"}
    STAN_TO_TORCH_NAMES = {"gamma": "logits", "N": "total_count"}


class MultinomialLogTheta(_MultinomialBase):
    """
    Defines the multinomial distribution in terms of the log of the theta parameter.
    """

    LOG_SIMPLEX_PARAMS = {"log_theta"}
    STAN_DIST = "multinomial_logtheta"
    SCIPY_DIST = custom_scipy_dists.multinomial_log_theta
    TORCH_DIST = custom_torch_dists.MultinomialLogTheta
    STAN_TO_SCIPY_NAMES = {"log_theta": "log_p", "N": "n"}
    STAN_TO_TORCH_NAMES = {"log_theta": "log_probs", "N": "total_count"}

    def __init__(
        self,
        *,
        log_theta: "dms.custom_types.ContinuousParameterType",
        N: "dms.custom_types.DiscreteParameterType",
        **kwargs,
    ):

        # Init the parent class with the appropriate parameters
        super().__init__(log_theta=log_theta, N=N, **kwargs)

        # By default, we allow a multinomial coefficient to be pre-calculated. This
        # assumes that the instance will be an observable parameter, so we modify
        # the `_record_child` function to remove the coefficient as soon as something
        # is added to the children.
        self._coefficient = transformed_data.LogMultinomialCoefficient(self)

    def _record_child(
        self, child: abstract_model_component.AbstractModelComponent
    ) -> None:
        """Handles removal of the coefficient when a child is added."""
        # If this is not a multinomial coefficient, then we have to remove that
        # coefficient from the children.
        if not isinstance(child, transformed_data.LogMultinomialCoefficient):
            assert len(self._children) == 1
            del self._children[0]
            self._coefficient = None

        # Otherwise, the list of children must be empty, as the coefficient will
        # be the first child added.
        else:
            assert len(self._children) == 0

        # Run the parent method to record the child
        super()._record_child(child)

    def get_supporting_functions(self) -> list[str]:
        """
        Extend the set of supporting functions to include the custom Stan functions
        for the MultinomialLogTheta distribution.
        """
        return super().get_supporting_functions() + [
            "#include multinomial.stanfunctions"
        ]

    def write_dist_args(  # pylint: disable=arguments-differ, arguments-renamed
        self, log_theta: str, N: str, coeff: str = ""
    ):
        # If the coefficient is provided, insert it in the middle of the arguments.
        # Otherwise, just return the log_theta and N parameters. This is a bit of
        # a hack to make sure that "N" is stripped off by `get_target_incrementation`
        # regardless of whether the coefficient is provided or not.
        if coeff:
            return f"{log_theta}, {coeff}, {N}"
        return f"{log_theta}, {N}"

    def get_right_side(
        self, index_opts: tuple[str, ...] | None, dist_suffix: str = ""
    ) -> str:
        """
        Override the right side to use the log softmax transformation.
        """
        # Get the formattables
        formattables = super(Parameter, self).get_right_side(index_opts)

        # If no suffix is provided and this is an observable, we want to add the
        # coefficient to the set of formattables and use manual normalization.
        # Otherwise, we just use the standard normalization.
        if dist_suffix == "":
            if self.coefficient is None:
                dist_suffix = "norm"
            else:
                formattables["coeff"] = self.coefficient.get_indexed_varname(index_opts)
                dist_suffix = "manual_norm"

        # Build the right side
        return f"{self.STAN_DIST}_{dist_suffix}({self.write_dist_args(**formattables)})"

    @property
    def coefficient(self) -> transformed_data.LogMultinomialCoefficient | None:
        """
        Returns the coefficient for the multinomial distribution, if it exists.
        """
        return self._coefficient
