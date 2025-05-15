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


class Parameter(AbstractModelComponent):
    """Base class for parameters used in DMS Stan"""

    STAN_DIST: str = ""  # The Stan distribution name
    UNNORMALIZED_LOG_PROB_SUFFIX: str = (
        ""  # Will be 'lupmf' for discrete, 'lupdf' for continuous
    )

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

        # We must have defined the `UNNORMALIZED_LOG_PROB_SUFFIX` class attribute
        if self.UNNORMALIZED_LOG_PROB_SUFFIX == "":
            raise NotImplementedError(
                "The UNNORMALIZED_LOG_PROB_SUFFIX class attribute must be defined"
            )

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
        # Get the right-hand-side of the incrementation
        right_side = self.get_right_side(index_opts)

        # Determine the left side and operator
        left_side = (
            "target += "
            if self._parallelized
            else f"{self.get_indexed_varname(index_opts)} ~ "
        )

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
        # If parallelized, the right side is a call to the `reduce_sum` function.
        if self._parallelized and dist_suffix == "":
            component_names = [
                component.get_indexed_varname(index_opts)
                for component in self.get_right_side_components()
            ]
            return (
                "reduce_sum("
                + f"{self.plp_function_name}, "
                + f"to_array_1d({self.get_indexed_varname(index_opts)}), "
                + "1"  # Automatic grainsize
                + (", " if len(component_names) > 0 else "")
                + ", ".join(component_names)
                + ")"
            )

        # Get the formattables
        formattables = super().get_right_side(index_opts=index_opts)

        # Build the distribution argument and format the Stan code
        suffix = "" if dist_suffix == "" else f"_{dist_suffix}"
        code = f"{self.STAN_DIST}{suffix}({self._write_dist_args(**formattables)})"

        return code

    def get_supporting_functions(self) -> list[str]:
        """
        Returns the Stan code for the partial log probability function that is used
        with Stan's `reduce_sum` function. This is used for parallelizing the log
        probability calculation across multiple cores.
        """

        # No declaration if the function is not defined
        if self.plp_function_name == "":
            return []

        # Get the arguments shared across all shards. These are all the variables
        # that show up on the right-hand-side of the target incrementation
        right_components = self.get_right_side_components()

        # Get the datatype for the slice of this parameter used in the function
        slice_dtype = f"array[] {'int' if self.BASE_STAN_DTYPE == 'int' else 'real'}"

        # Get the argument specification for the function.
        argspec = ", ".join(
            [
                f"{slice_dtype} {self.stan_model_varname}_slice",
                "int start",
                "int end",
                *[component.plp_argspec_vardec for component in right_components],
            ]
        )

        # Now get the function body. This will be the log probability calculated
        # using the UNNORMALIZED log probability function directly.
        right_side_sliced = self._write_dist_args(
            **super().get_right_side(index_opts=None)
        )
        body = (
            f"return {self.STAN_DIST}_{self.UNNORMALIZED_LOG_PROB_SUFFIX}("
            f"{self.stan_model_varname}_slice | {right_side_sliced})"
        )

        # Complete the function declaration
        return [
            f"real {self.plp_function_name}({argspec}) {{" + "\n\t\t" + body + ";\n}"
        ]

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

    @property
    def plp_function_name(self) -> str:
        """
        Returns the name of the partial log probability function that is used with
        Stan's `reduce_sum` function. This is used for parallelizing the log probability
        calculation across multiple cores.
        """
        # We can only use `reduce_sum` if this parameter is not a scalar and has
        # more than one element in its final dimension
        if not self._parallelized:
            return ""

        # Otherwise, the partial function is defined
        return (
            f"{self.stan_model_varname}_"
            + self.UNNORMALIZED_LOG_PROB_SUFFIX.replace("lup", "lp")
        )


class ContinuousDistribution(Parameter, TransformableParameter):
    """Base class for parameters represented by continuous distributions."""

    UNNORMALIZED_LOG_PROB_SUFFIX: str = "lupdf"


class DiscreteDistribution(Parameter):
    """
    Base class for parameters represented by discrete distributions. This is
    more-or-less a passthrough to the Parameter class; however, the default for
    discrete distributions is to set the observable attribute to True.
    """

    BASE_STAN_DTYPE: str = "int"
    LOWER_BOUND: int = 0
    UNNORMALIZED_LOG_PROB_SUFFIX: str = "lupmf"

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
            index_opts, _name_override=self.noncentered_varname
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
        new_name = self.get_indexed_varname(
            index_opts, _name_override=self.noncentered_varname
        )
        if self._parallelized:
            return parent_incrementation.replace(
                f"to_array_1d({default_name})",
                f"to_array_1d({new_name})",
            )
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

        # Otherwise, we cannot have the suffix set
        assert (
            dist_suffix == ""
        ), "Non-centered parameters should not have a distribution suffix"

        # Run the parent method if parallelized, replacing the indexed variable name
        # with the non-centered variable name
        if self._parallelized:
            return (
                super()
                .get_right_side(index_opts, dist_suffix=dist_suffix)
                .replace(
                    self.get_indexed_varname(index_opts),
                    self.get_indexed_varname(
                        index_opts, _name_override=self.noncentered_varname
                    ),
                )
            )

        # Otherwise, the right hand side is just the standard normal
        return "std_normal()"

    def get_supporting_functions(self) -> list[str]:
        # Parent method if not noncentered or if we are not parallelized
        if not self.is_noncentered or not self._parallelized:
            return super().get_supporting_functions()

        # Otherwise, declare the function
        argspec = f"array[] real {self.noncentered_varname}_slice, int start, int end"
        body = f"return std_normal_lupdf({self.noncentered_varname}_slice)"
        return [
            f"real {self.plp_function_name}({argspec}) {{" + "\n\t\t" + body + ";\n}"
        ]

    @property
    def noncentered_varname(self) -> str:
        """Return the non-centered variable name"""
        return f"{self.stan_model_varname}_raw"

    @property
    def is_noncentered(self) -> bool:
        """
        Can only be noncentered if the parameter is not a hyperparameter, observable,
        and we did not set `noncentered` to False at initialization.
        """
        return self._noncentered and not self.is_hyperparameter and not self.observable

    @property
    def plp_function_name(self) -> str:

        # Run the parent method
        # pylint: disable=assignment-from-no-return
        name = super(Normal, self.__class__).plp_function_name.fget(self)
        # pylint: enable=assignment-from-no-return

        # If there is no name or we are not noncentered, return
        if name == "" or not self.is_noncentered:
            return name

        # If we are noncentered, prepend 'std'
        return f"std_{name}"


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
        return f"0, {lambda_}, {alpha}"


class _CustomStanFunctionMixIn:
    """
    Some distributions have custom Stan functions. This is a mixin to handle
    that.
    """

    def get_supporting_functions(self) -> list[str]:
        """Builds the appropriate #include statement for the custom Stan functions"""
        # pylint: disable=no-member
        return (
            [f"#include {self.STAN_DIST}.stanfunctions"] if self._parallelized else []
        )


class Dirichlet(_CustomStanFunctionMixIn, ContinuousDistribution):
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

        # Trigger the parallelization setter. This will build transformed data
        # declarations if needed
        self._parallelized = self._parallelized

        # Placeholder for the shared alpha if we have not set it
        if not hasattr(self, "_shared_alpha"):
            self._shared_alpha: SharedAlphaDirichlet | None = None

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

    def get_right_side(
        self, index_opts: tuple[str, ...] | None, dist_suffix: str = ""
    ) -> str:
        # Parent method if provided a suffix or we are not parallelized
        if dist_suffix != "" or not self._parallelized:
            return super().get_right_side(index_opts, dist_suffix=dist_suffix)

        # If we are enforcing uniformity, then we need to use the shared alpha.
        thetas = f"to_array_1d({self.get_indexed_varname(index_opts)})"
        if self.alpha.enforce_uniformity:
            coeff = self._shared_alpha.get_indexed_varname(index_opts)
            return f"parallelized_dirichlet_uniform_alpha_lpdf({thetas} | {coeff})"

        # Otherwise, we calculate the full log probability
        alphas = self.alpha.get_indexed_varname(index_opts)
        return f"parallelized_dirichlet_lpdf({thetas} | {alphas})"

    @property
    def plp_function_name(self) -> str:
        """There is no partial log probability function for the Dirichlet distribution"""
        return ""

    # Update the parallelized setter to add a shared alpha if relevant
    @Parameter._parallelized.setter  # pylint: disable=protected-access
    def _parallelized(self, value: bool) -> None:

        # Set the parallelized attribute
        super(Dirichlet, self.__class__)._parallelized.fset(self, value)

        # If we are parallelized and enforcing uniformity, then we need to set
        # the shared alpha
        if value and self.alpha.enforce_uniformity:
            self._shared_alpha = SharedAlphaDirichlet(
                alpha=self.alpha,
                shape=self.shape[:-1] + (1,),
                parallelize=True,
            )


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


class _MultinomialBase(_CustomStanFunctionMixIn, DiscreteDistribution):
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

        # Trigger the parallelization setter. This will build transformed data
        # declarations if needed
        self._parallelized = self._parallelized

        # If we do not have a multinomial coefficient, then we need to set it
        if not hasattr(self, "_multinomial_coefficient"):
            self._multinomial_coefficient: MultinomialCoefficient | None = None

    def _record_child(self, child: AbstractModelComponent) -> None:
        # Run the inherited method
        super()._record_child(child)

        # If recording the multinomial coefficient, we are done
        if isinstance(child, MultinomialCoefficient):
            return

        # If we have any other child, then we cannot precalculate the multinomial
        # coefficient and so we need to remove the transformed data attribute
        self._multinomial_coefficient = None

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

        # Just return raw if we are parallelized or if we are precalculating the
        # multinomial coefficient
        if self._parallelized or self._multinomial_coefficient is not None:
            return raw

        # If not parallelized, remove the N parameter
        assert raw.count(", ") == 1, "Invalid target incrementation: " + raw
        raw, _ = raw.split(", ")
        return raw + ")"

    def get_right_side(
        self, index_opts: tuple[str, ...] | None, dist_suffix: str = ""
    ) -> str:
        # Parent method if provided a suffix or we are not parallelized
        if dist_suffix != "" or not self._parallelized:
            return super().get_right_side(index_opts, dist_suffix=dist_suffix)

        # Get the indexed names for the loss calculations
        ys = self.get_indexed_varname(index_opts)
        probs = self.get_stan_probs_indexed_varname(index_opts)

        # If we have a multinomial coefficient predefined, get it. Otherwise,
        # we need to calculate it each time.
        if self._multinomial_coefficient is not None:
            coeff = self._multinomial_coefficient.get_indexed_varname(index_opts)

        # Otherwise, we need to calculate the multinomial coefficient each time
        else:
            coeff = f"parallelized_multinomial_factorial_component_lpmf({ys})"

        # We always need to calculate the log probability each time. We slice over
        # theta instead of y if this is an observable to avoid copying gradients
        if self.observable:
            probs = f"to_array_1d({probs})"
            midfunc = "thetaslice_"
        else:
            midfunc = ""
        logprob = f"parallelized_multinomial_factorial_component_{midfunc}lpmf({ys} | {probs})"

        # Overall log-loss is the coefficient + the log probability
        return f"{coeff} + {logprob}"

    @abstractmethod
    def get_stan_probs_indexed_varname(self, index_opts: tuple[str, ...] | None) -> str:
        """Returns Stan code that will yield the probabilities of the distribution"""

    @property
    def plp_function_name(self) -> str:
        """There is no partial log probability function for the multinomial distribution"""
        return ""

    # Redefine the parallelized property setter to set the multinomial coefficient
    @Parameter._parallelized.setter  # pylint: disable=protected-access
    def _parallelized(self, value: bool) -> None:

        # Run the parent setter
        super(_MultinomialBase, self.__class__)._parallelized.fset(self, value)

        # If we are parallelized and have no children, then we will want a precalculated
        # multinomial coefficient
        if value and len(self._children) == 0:
            self._multinomial_coefficient = MultinomialCoefficient(
                self,
                shape=self.shape[:-1] + (1,),
                parallelize=True,
            )


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

    def get_stan_probs_indexed_varname(self, index_opts: tuple[str, ...] | None) -> str:
        return self.theta.get_indexed_varname(index_opts)


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

    def get_stan_probs_indexed_varname(self, index_opts: tuple[str, ...] | None) -> str:
        return f"softmax({self.gamma.get_indexed_varname(index_opts)})"
