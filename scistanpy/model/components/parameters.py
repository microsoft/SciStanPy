# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Parameter classes for defining probabilistic model components in SciStanPy.

This module provides the core parameter classes that serve as building blocks for
constructing probabilistic models in SciStanPy. These classes represent random
variables with specific probability distributions and handle the complex task of
translating between Python model specifications, PyTorch modules, and Stan probabilistic
programming language code.

The parameter classes are designed to be composable, allowing complex hierarchical
models to be built through simple parameter relationships while maintaining
mathematical rigor and computational efficiency.

The following distributions are currently supported in SciStanPy:

Continuous Univariate
^^^^^^^^^^^^^^^^^^^^^
- :py:class:`~scistanpy.model.components.parameters.Normal`
- :py:class:`~scistanpy.model.components.parameters.HalfNormal`
- :py:class:`~scistanpy.model.components.parameters.UnitNormal`
- :py:class:`~scistanpy.model.components.parameters.LogNormal`
- :py:class:`~scistanpy.model.components.parameters.Beta`
- :py:class:`~scistanpy.model.components.parameters.Gamma`
- :py:class:`~scistanpy.model.components.parameters.InverseGamma`
- :py:class:`~scistanpy.model.components.parameters.Exponential`
- :py:class:`~scistanpy.model.components.parameters.ExpExponential`
- :py:class:`~scistanpy.model.components.parameters.Lomax`
- :py:class:`~scistanpy.model.components.parameters.ExpLomax`

Continuous Multivariate
^^^^^^^^^^^^^^^^^^^^^^^
- :py:class:`~scistanpy.model.components.parameters.Dirichlet`
- :py:class:`~scistanpy.model.components.parameters.ExpDirichlet`

Discrete Univariate
^^^^^^^^^^^^^^^^^^^
- :py:class:`~scistanpy.model.components.parameters.Binomial`
- :py:class:`~scistanpy.model.components.parameters.Poisson`

Discrete Multivariate
^^^^^^^^^^^^^^^^^^^^^
- :py:class:`~scistanpy.model.components.parameters.Multinomial`
- :py:class:`~scistanpy.model.components.parameters.MultinomialLogit`
- :py:class:`~scistanpy.model.components.parameters.MultinomialLogTheta`

Is there a distribution you need that isn't listed here? Please open an issue or
submit a PR!
"""

from __future__ import annotations

import functools
import re

from abc import ABCMeta
from typing import Callable, Optional, overload, TYPE_CHECKING, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.distributions as dist
import torch.nn as nn

from scipy import stats

import scistanpy
from scistanpy import utils

from scistanpy.model.components import abstract_model_component
from scistanpy.model.components.custom_distributions import (
    custom_scipy_dists,
    custom_torch_dists,
)
from scistanpy.model.components.transformations import transformed_parameters

cdfs = utils.lazy_import("scistanpy.model.components.transformations.cdfs")
constants = utils.lazy_import("scistanpy.model.components.constants")
transformed_data = utils.lazy_import(
    "scistanpy.model.components.transformations.transformed_data"
)

if TYPE_CHECKING:
    from scistanpy import custom_types

# pylint: disable=too-many-lines, line-too-long


@overload
def _inverse_transform(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]: ...


@overload
def _inverse_transform(x: "custom_types.Float") -> "custom_types.Float": ...


def _inverse_transform(x):
    """Apply element-wise inverse transformation (1/x).

    Simple inverse transformation function for parameter transformations.
    Defined at module level to avoid pickling issues with lambda functions.

    :param x: Input value(s) to transform
    :type x: Union[npt.NDArray[np.floating], custom_types.Float]

    :returns: Element-wise inverse of input
    :rtype: Union[npt.NDArray[np.floating], custom_types.Float]
    """
    return 1 / x


@overload
def _exp_transform(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]: ...


@overload
def _exp_transform(x: "custom_types.Float") -> "custom_types.Float": ...


def _exp_transform(x):
    """Apply element-wise exponential transformation.

    Simple exponential transformation function for parameter transformations.
    Defined at module level to avoid pickling issues with lambda functions.

    :param x: Input array to transform
    :type x: npt.NDArray[np.floating]

    :returns: Element-wise exponential of input
    :rtype: npt.NDArray[np.floating]
    """
    return np.exp(x)


# TODO: Make sure samples from torch distributions obey the same bounds as noted
# in the classes.
# TODO: Make sure samples from Stan distributions obey the same bounds as noted
# in the classes.


class ParameterMeta(ABCMeta):
    """Metaclass for automatic CDF transform class generation.

    :param name: Name of the class being created
    :type name: str
    :param bases: Base classes for the new class
    :type bases: tuple
    :param attrs: Class attributes dictionary
    :type attrs: dict

    This metaclass automatically creates cumulative distribution function (CDF)
    and related transform classes for each Parameter subclass, enabling automatic
    generation of probabilistic transforms and survival functions.

    The metaclass creates four transform classes for each parameter and assigns
    them to the following class variables:

        - :py:attr:`Parameter.CDF <scistanpy.model.components.parameters.Parameter.CDF>`,
          based on the :py:class:`~scistanpy.model.components.transformations.
          cdfs.CDF` class and describing the cumulative distribution function for
          the parameter.
        - :py:attr:`Parameter.SF <scistanpy.model.components.parameters.Parameter.SF>`,
          based on the :py:class:`~scistanpy.model.components.transformations.
          cdfs.SurvivalFunction` class and describing the survival function (1 - CDF)
          for the parameter.
        - :py:attr:`Parameter.LOG_CDF <scistanpy.model.components.parameters.Parameter.
          LOG_CDF>`, based on the :py:class:`~scistanpy.model.components.transformations.
          cdfs.LogCDF` class and describing the logarithmic cumulative distribution
          function for the parameter.
        - :py:attr:`Parameter.LOG_SF <scistanpy.model.components.parameters.Parameter.
          LOG_SF>`, based on the :py:class:`~scistanpy.model.components.transformations.
          cdfs.LogSurvivalFunction` class and describing the logarithmic survival
          function for the parameter.
    """

    def __init__(cls, name, bases, attrs):
        """Create CDF transform classes for the Parameter subclass."""
        # Run the parent class's __init__ method
        super().__init__(name, bases, attrs)

        # Add CDF, SF, LOG_CDF, and LOG_SF classes to the Parameter subclass
        cls.CDF = type(
            "CDF",
            (cdfs.CDF,),
            {
                "PARAMETER": cls,
                "__module__": cls.__module__,
                "__qualname__": f"{cls.__qualname__}.CDF",
            },
        )
        cls.SF = type(
            "SF",
            (cdfs.SurvivalFunction,),
            {
                "PARAMETER": cls,
                "__module__": cls.__module__,
                "__qualname__": f"{cls.__qualname__}.SF",
            },
        )
        cls.LOG_CDF = type(
            "LOG_CDF",
            (cdfs.LogCDF,),
            {
                "PARAMETER": cls,
                "__module__": cls.__module__,
                "__qualname__": f"{cls.__qualname__}.LOG_CDF",
            },
        )
        cls.LOG_SF = type(
            "LOG_SF",
            (cdfs.LogSurvivalFunction,),
            {
                "PARAMETER": cls,
                "__module__": cls.__module__,
                "__qualname__": f"{cls.__qualname__}.LOG_SF",
            },
        )


class ClassOrInstanceMethod:
    """Descriptor used as a decorator to enable dual class/instance method behavior.

    This descriptor allows methods to behave differently when called as class
    methods versus instance methods, enabling flexible parameter handling for
    CDF-like functions that can use either explicit parameters or instance
    parameter values.

    :param func: Function to wrap with dual behavior
    :type func: Callable

    When called as an instance method, the descriptor automatically uses the
    instance's parameter values. When called as a class method, it requires
    explicit parameter specification.
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        """Return appropriate method based on call context.

        :param instance: Instance object if called as instance method, None for class method
        :param owner: Class that owns the method

        :returns: Configured method with appropriate parameter handling
        :rtype: Callable

        The returned method automatically handles parameter passing based on
        whether it's called as a class or instance method, validating required
        parameters and applying appropriate transformations.
        """

        @functools.wraps(self.func)
        def inner(**kwargs):
            # `x` must always be a kwarg
            if "x" not in kwargs:
                raise ValueError(
                    "Expected `x` to be a keyword argument for the CDF-like methods."
                )

            # Check for extra kwargs
            if (
                extra_kwargs := set(kwargs)
                - set(owner.STAN_TO_SCIPY_NAMES)
                - {"x", "shape"}
            ):
                raise ValueError(
                    f"Unexpected arguments passed to {self.func.__name__}: "
                    f"{extra_kwargs}."
                )

            # If the instance is not provided, then kwargs must be provided
            if instance is None:

                # Check for missing kwargs
                if missing_kwargs := set(owner.STAN_TO_SCIPY_NAMES) - set(kwargs):
                    raise ValueError(
                        f"If calling {self.func.__name__} as a static method, the "
                        f"following parameters must be provided: {missing_kwargs}."
                    )

                # Run the wrapped method using provided kwargs
                return self.func(owner, **kwargs)

            # If the instance is not provided, we run the wrapped method using the
            # parent parameter values
            return self.func(
                owner, **instance._parents, **kwargs  # pylint: disable=protected-access
            )

        return inner


class Parameter(
    abstract_model_component.AbstractModelComponent, metaclass=ParameterMeta
):
    """Base class for all probabilistic parameters in SciStanPy models.

    This class provides the foundational infrastructure for representing random
    variables with specific probability distributions. It handles the complex
    mapping between Python model specifications and Stan code generation while
    providing integration with SciPy and PyTorch ecosystems.

    :param kwargs: Distribution parameters (mu, sigma, etc. depending on subclass)

    :raises NotImplementedError: If required class attributes are missing (i.e.,
        if subclass was incorrectly defined)
    :raises TypeError: If required distribution parameters are missing

    :cvar STAN_DIST: Stan distribution name for code generation
    :type STAN_DIST: str
    :cvar HAS_RAW_VARNAME: Whether parameter uses a raw/transformed parameterization
    :type HAS_RAW_VARNAME: bool
    :cvar SCIPY_DIST: Corresponding SciPy distribution class
    :type SCIPY_DIST: Optional[Union[type[stats.rv_continuous], type[stats.rv_discrete]]]
    :cvar TORCH_DIST: Corresponding PyTorch distribution class
    :type TORCH_DIST: Optional[Union[type[dist.distribution.Distribution],
        type[custom_torch_dists.CustomDistribution]]]
    :cvar STAN_TO_SCIPY_NAMES: Parameter name mapping for SciPy interface
    :type STAN_TO_SCIPY_NAMES: dict[str, str]
    :cvar STAN_TO_TORCH_NAMES: Parameter name mapping for PyTorch interface
    :type STAN_TO_TORCH_NAMES: dict[str, str]
    :cvar STAN_TO_SCIPY_TRANSFORMS: Parameter transformation functions converting between
        Stan and SciPy parametrizations
    :type STAN_TO_SCIPY_TRANSFORMS: dict[str, Callable[[npt.NDArray], npt.NDArray]]
    :cvar CDF: Automatically generated :py:class:`~scistanpy.model.components.transformations.
        cdfs.CDF` class.
    :type CDF: type[cdfs.CDF]
    :cvar SF: Automatically generated :py:class:`~scistanpy.model.components.transformations.
        cdfs.SurvivalFunction` class.
    :type SF: type[cdfs.SurvivalFunction]
    :cvar LOG_CDF: Automatically generated :py:class:`~scistanpy.model.components.
        transformations.cdfs.LogCDF` class.
    :type LOG_CDF: type[cdfs.LogCDF]
    :cvar LOG_SF: Automatically generated :py:class:`~scistanpy.model.components.
        transformations.cdfs.LogSurvivalFunction` class.
    :type LOG_SF: type[cdfs.LogSurvivalFunction]
    """

    STAN_DIST: str = ""
    """Name of the distribution in Stan code (e.g. "normal", "binomial")."""

    HAS_RAW_VARNAME: bool = False
    """Whether the parameter intrinsically uses a raw/transformed parameterization."""

    CDF: type[cdfs.CDF]
    """
    Subclass of :py:class:`~scistanpy.model.components.transformations.cdfs.CDF`
    describing the cumulative distribution function for the parameter.
    """

    SF: type[cdfs.SurvivalFunction]
    """
    Subclass of :py:class:`~scistanpy.model.components.transformations.cdfs.SurvivalFunction`
    describing the survival function for the parameter.
    """

    LOG_CDF: type[cdfs.LogCDF]
    """
    Subclass of :py:class:`~scistanpy.model.components.transformations.cdfs.LogCDF`
    describing the log cumulative distribution function for the parameter.
    """

    LOG_SF: type[cdfs.LogSurvivalFunction]
    """
    Subclass of :py:class:`~scistanpy.model.components.transformations.cdfs.LogSurvivalFunction`
    describing the log survival function for the parameter.
    """

    SCIPY_DIST: type[stats.rv_continuous] | type[stats.rv_discrete] | None = None
    """Corresponding SciPy distribution class (e.g., `scipy.stats.norm`)."""

    TORCH_DIST: (
        type[dist.distribution.Distribution]
        | type[custom_torch_dists.CustomDistribution]
        | None
    ) = None
    """Corresponding PyTorch distribution class (e.g., `torch.distributions.Normal`)."""

    STAN_TO_SCIPY_NAMES: dict[str, str] = {}
    """
    There can be differences in parameter names between Stan and SciPy. This dictionary
    maps between the two naming conventions.
    """

    STAN_TO_TORCH_NAMES: dict[str, str] = {}
    """
    There can be differences in parameter names between Stan and PyTorch. This dictionary
    maps between the two naming conventions.
    """

    STAN_TO_SCIPY_TRANSFORMS: dict[str, Callable[[npt.NDArray], npt.NDArray]] = {}
    """
    Some distributions are parametrized differently between Stan and SciPy. This
    dictionary provides transformation functions to convert parameters from Stan's
    parametrization to SciPy's parametrization.
    """

    def __init__(self, **kwargs):
        """Initialize parameter with distribution-specific arguments."""
        # Confirm that class attributes are set correctly
        if missing_attributes := [
            attr
            for attr in (
                "STAN_DIST",
                "CDF",
                "SF",
                "LOG_CDF",
                "LOG_SF",
                "SCIPY_DIST",
                "TORCH_DIST",
                "STAN_TO_SCIPY_NAMES",
                "STAN_TO_TORCH_NAMES",
            )
            if not hasattr(self, attr)
        ]:
            raise NotImplementedError(
                f"The following class attributes must be defined: {', '.join(missing_attributes)}"
            )

        # Make sure we have the expected parameters
        if missing_params := self.STAN_TO_SCIPY_NAMES.keys() - set(kwargs.keys()):
            raise TypeError(
                f"Missing parameters {missing_params} for {self.__class__.__name__}."
            )

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
        seed: Optional[custom_types.Integer] = None,
    ) -> None:
        """Initialize PyTorch parameter on *unconstrained* space for gradient-based
        optimization.

        :param init_val: Initial parameter values on *unconstrained* space. Uniform
            between -1 and 1 if None. Defaults to None.
        :type init_val: Optional[Union[npt.NDArray, torch.Tensor]]
        :param seed: Random seed for initialization. Defaults to None.
        :type seed: Optional[custom_types.Integer]

        :raises ValueError: If called on observable parameters
        :raises ValueError: If ``init_val`` shape doesn't match parameter shape

        This method sets up the parameter for PyTorch-based optimization by
        creating a trainable ``nn.Parameter``. The initialization strategy uses
        uniform random values in [-1, 1] if no explicit values are provided.

        .. important::
            Initialization values are considered to be in unconstrained space, whether
            provided or otherwise. An appropriate transform is applied depending
            on the bounds of the distribution represented by the class to take it to
            a constrained space (e.g., exponentiation for positive distributions).

        .. note::
            Observable parameters cannot be initialized as they represent fixed
            data rather than learnable parameters.
        """
        # This cannot be called if the parameter is an observable
        if self.observable:
            raise ValueError("Observables do not have a torch parametrization")

        # If no initialization value is provided, then we create one on the range
        # of -1 to 1.
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
        self,
        level_draws: dict[str, Union[npt.NDArray, "custom_types.Float"]],
        seed: Optional[custom_types.Integer],
    ) -> Union[npt.NDArray, "custom_types.Float", "custom_types.Integer"]:
        """Draw samples from the parameter's distribution using SciPy backend.

        :param level_draws: Parameter values from parent components
        :type level_draws: dict[str, Union[npt.NDArray, custom_types.Float]]
        :param seed: Random seed for reproducible sampling
        :type seed: Optional[custom_types.Integer]

        :returns: Sampled values from the distribution
        :rtype: Union[npt.NDArray, custom_types.Float, custom_types.Integer]

        This method applies necessary parameter transformations and name mappings
        to convert from Stan parameter conventions to SciPy conventions, then
        samples from the corresponding SciPy distribution.
        """
        # Transform and rename the draws from the previous level.
        level_draws = {
            self.STAN_TO_SCIPY_NAMES[name]: self.STAN_TO_SCIPY_TRANSFORMS.get(
                name, lambda x: x
            )(draw)
            for name, draw in level_draws.items()
        }

        # Draw from the scipy distribution
        return self.__class__.SCIPY_DIST.rvs(
            **level_draws, size=(1,) + self.shape, random_state=seed
        )[0]

    def as_observable(self) -> "Parameter":
        """Mark parameter as observable (representing observed data).

        :returns: Self-reference for method chaining
        :rtype: Parameter

        Observable parameters represent known data rather than unknown variables
        to be inferred. This method:

        - Sets the observable flag to True
        - Removes PyTorch parameterization (observables aren't optimized)
        - Enables generation of appropriate Stan code for data blocks

        This method will typically not be needed, as SciStanPy automatically assigns
        parameters with no children in the depenency graph as observables.

        Example:
            >>> y_obs = Normal(mu=mu_param, sigma=sigma_param).as_observable()
        """
        # Set the observable attribute to True
        self._observable = True

        # We do not have a torch parameterization for observables
        self._torch_parametrization = None

        return self

    def get_target_incrementation(self, index_opts: tuple[str, ...]) -> str:
        """Generate Stan target increment statement for log-probability.

        :param index_opts: Potential names for indexing variables (e.g., ('i', 'j',
            'k', ...)).
        :type index_opts: tuple[str, ...]

        :returns: Stan target increment statement (e.g., "y ~ normal(mu, sigma)"
            or target += normal_lpdf(y | mu, sigma)").
        :rtype: str

        This method generates the Stan code that adds this parameter's log-probability
        (e.g., "target += normal_lpdf(y | mu, sigma)" or "y ~ normal(mu, sigma)")
        contribution to the target density.
        """
        # Determine the left side and operator
        left_side = f"{self.get_indexed_varname(index_opts)} ~ "

        # Get the right-hand-side of the incrementation
        right_side = self.get_right_side(index_opts)

        # Put it all together
        return left_side + right_side

    def get_generated_quantities(self, index_opts: tuple[str, ...]) -> str:
        """Generate Stan code for posterior predictive sampling.

        :param index_opts: Indexing options for multi-dimensional parameters
        :type index_opts: tuple[str, ...]

        :returns: Stan code for generated quantities block
        :rtype: str

        This method creates Stan code for the generated quantities block,
        enabling posterior predictive sampling.
        """
        return (
            self.get_indexed_varname(index_opts, _name_override=self.generated_varname)
            + f" = {self.get_right_side(index_opts, dist_suffix='rng')}"
        )

    def get_torch_logprob(
        self, observed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute log-probability using PyTorch backend for gradient computation.

        :param observed: Observed values for observable parameters. Defaults to
            None. Required if parameter is observable. Must be None if parameter
            is latent.
        :type observed: Optional[torch.Tensor]

        :returns: Log-probability tensor with gradient tracking
        :rtype: torch.Tensor

        :raises ValueError: If observable parameter lacks observed values
        :raises ValueError: If latent parameter has observed values

        This method computes log-probabilities using PyTorch distributions,
        enabling gradient-based optimization.
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

    def get_rng(
        self, seed: Optional[custom_types.Integer] = None
    ) -> np.random.Generator:
        """Get random number generator for sampling operations.

        :param seed: Optional seed for reproducible generation. Defaults to None.
        :type seed: Optional[custom_types.Integer]

        :returns: NumPy random number generator
        :rtype: np.random.Generator

        Returns the global :py:obj:`scistanpy.RNG` if no seed is provided, otherwise
        creates a new generator with the specified seed.
        """
        # Return the global random number generator if no seed is provided. Otherwise,
        # return a new random number generator with the provided seed.
        if seed is None:
            return scistanpy.RNG
        return np.random.default_rng(seed)

    def write_dist_args(self, **to_format: str) -> str:
        """Format distribution arguments for Stan code generation.

        :param to_format: Formatted parameter strings keyed by parameter name
        :type to_format: str

        :returns: Comma-separated argument string for Stan distribution calls
        :rtype: str

        This method creates properly formatted argument lists for Stan
        distribution functions, ordering arguments according to Stan
        conventions and handling parameter name mappings.

        Example:
            >>> param = Normal(mu=0, sigma=1)
            >>> param.write_dist_args(mu="mean", sigma="stddev")
            'mean, stddev'
        """
        return ", ".join(to_format[name] for name in self.STAN_TO_SCIPY_NAMES)

    def get_right_side(  # pylint: disable=arguments-differ
        self,
        index_opts: tuple[str, ...] | None,
        start_dims: dict[str, custom_types.Integer] | None = None,
        end_dims: dict[str, custom_types.Integer] | None = None,
        offset_adjustment: int = 0,
        dist_suffix: str = "",
    ) -> str:
        """Generate right-hand side of Stan statements.

        :param index_opts: Options for indexing variable names (e.g., ('i', 'j',
            'k', ...)). If None, no indexing is applied.
        :type index_opts: Optional[tuple[str, ...]]
        :param start_dims: First indexable dimension of parent model components.
            Defaults to None, meaning the first dimension is the first indexable
            dimension for all parents.
        :type start_dims: Optional[dict[str, custom_types.Integer]]
        :param end_dims: Last indexable dimension of parent model components. Defaults
            to None, meaning the last dimension is the last indexable dimension
            for all parents.
        :type end_dims: Optional[dict[str, custom_types.Integer]]
        :param offset_adjustment: Index offset adjustment. For example, if `index_opts`
            are ('a', 'b', 'c') and `offset_adjustment` is 1, the effective allowed
            indices will be ('b', 'c'). This argument is critical for aligning
            dimensions between parent and child model components that have different
            numbers of dimensions (i.e., as a result of broadcasting). Defaults
            to 0.
        :type offset_adjustment: int
        :param dist_suffix: Distribution function suffix (e.g., "_rng"). Defaults
            to "".
        :type dist_suffix: str

        :returns: Stan distribution call string
        :rtype: str
        """
        # Get the formattables
        formattables = super().get_right_side(
            index_opts=index_opts,
            start_dims=start_dims,
            end_dims=end_dims,
            offset_adjustment=offset_adjustment,
        )

        # Build the distribution argument and format the Stan code
        suffix = "" if dist_suffix == "" else f"_{dist_suffix}"
        code = f"{self.STAN_DIST}{suffix}({self.write_dist_args(**formattables)})"

        return code

    def get_transformed_data_declaration(self) -> str:
        """Generate the ``transformed data`` block of Stan code.

        :returns: Stan code for transformed data declarations (empty by default)
        :rtype: str

        Most parameters don't require transformed data declarations. This method
        can be overridden by subclasses that need to declare transformed data variables.
        """
        # None by default
        return ""

    def get_generated_quantity_declaration(self, force_basetype: bool = True) -> str:
        """Generate Stan variable declaration for generated quantities.

        :param force_basetype: Whether to force base type declaration. For example,
            if ``True`` and the parameter is defined as a multidimensional float,
            the returned stan dtype will not be ``array[...Ndim - 1...] vector``,
            but ``array[...NDim...] float``. Defaults to ``True``, as this is the
            format expected by generated quantities blocks.
        :type force_basetype: bool

        :returns: Stan variable declaration for posterior predictive sampling
        :rtype: str

        Creates appropriate variable declarations for the generated quantities
        block, enabling posterior predictive sampling with correct variable
        types and constraints.
        """
        return self.declare_stan_variable(
            self.generated_varname, force_basetype=force_basetype
        )

    def get_raw_stan_parameter_declaration(self, force_basetype: bool = False) -> str:
        """Generate Stan parameter declaration for raw (untransformed) variables.

        :param force_basetype: Whether to force base type declaration. See
            ``get_generated_quantity_declaration`` for more information. Defaults
            to ``False``.
        :type force_basetype: bool

        :returns: Stan parameter declaration for raw variables
        :rtype: str

        For parameters using non-centered or other reparameterizations, this generates
        declarations for the underlying raw variables that are transformed to create
        the actual parameters. The ``get_transformation_assignment`` function will
        return Stan code that converts this raw parameter to the desired parameter.
        For parameters that do not use a raw variable, this returns an empty string.
        """
        if self.HAS_RAW_VARNAME:
            return self.declare_stan_variable(
                self.raw_varname, force_basetype=force_basetype
            )

        return ""

    # pylint: disable=no-self-argument
    @ClassOrInstanceMethod
    def cdf(cls, **params: "custom_types.CombinableParameterType") -> "cdfs.CDF":
        """Create cumulative distribution function transform.

        :param params: Distribution parameters (if called as class method) and
            parameter value.
        :type params: custom_types.CombinableParameterType

        :returns: CDF transform object
        :rtype: cdfs.CDF

        .. note::
            This is a convenience method for building an instance of the
            :py:class:`~scistanpy.model.components.transforms.cdf.CDF` subclass
            associated with the parameter. It can be used as either a class method,
            in which case parameters must be explicitly provided, or an instance
            method (in which case instance parameter values will be used).

        Example:
            >>> # As class method
            >>> cdf = Normal.cdf(mu=0.0, sigma=1.0, x=data)
            >>> # As instance method
            >>> normal_param = Normal(mu=0.0, sigma=1.0)
            >>> cdf = normal_param.cdf(x=data)
        """
        return cls.CDF(**params)

    @ClassOrInstanceMethod
    def ccdf(
        cls,
        **params: "custom_types.CombinableParameterType",
    ) -> "cdfs.SurvivalFunction":
        """Create complementary CDF (survival function) transform.

        :param params: Distribution parameters (if called as class method) and
            parameter value.
        :type params: custom_types.CombinableParameterType

        :returns: Survival function transform object
        :rtype: cdfs.SurvivalFunction

        Creates survival function (1 - CDF) transforms for survival analysis.

        .. note::
            This is a convenience method for building an instance of the
            :py:class:`~scistanpy.model.components.transforms.cdf.SurvivalFunction`
            subclass associated with the parameter. It can be used as either a class
            method, in which case parameters must be explicitly provided, or an
            instance method (in which case instance parameter values will be used).

        Example:
            >>> # As class method
            >>> sf = Normal.ccdf(mu=0.0, sigma=1.0, x=data)
            >>> # As instance method
            >>> normal_param = Normal(mu=0.0, sigma=1.0)
            >>> sf = normal_param.ccdf(x=data)
        """
        return cls.SF(**params)

    @ClassOrInstanceMethod
    def log_cdf(cls, **params: "custom_types.CombinableParameterType") -> "cdfs.LogCDF":
        """Create logarithmic CDF transform.

        :param params: Distribution parameters (if called as class method) and
            parameter value.
        :type params: custom_types.CombinableParameterType

        :returns: Log-CDF transform object
        :rtype: cdfs.LogCDF

        Creates log-CDF transforms for numerical stability in extreme
        tail probability computations.

        .. note::
            This is a convenience method for building an instance of the
            :py:class:`~scistanpy.model.components.transforms.cdf.LogCDF` subclass
            associated with the parameter. It can be used as either a class method,
            in which case parameters must be explicitly provided, or an instance
            method (in which case instance parameter values will be used).
        """
        return cls.LOG_CDF(**params)

    @ClassOrInstanceMethod
    def log_ccdf(
        cls,
        **params: "custom_types.CombinableParameterType",
    ) -> "cdfs.LogSurvivalFunction":
        """Create logarithmic survival function transform.

        :param params: Distribution parameters (if called as class method) and
            parameter value.
        :type params: custom_types.CombinableParameterType

        :returns: Log-survival function transform object
        :rtype: cdfs.LogSurvivalFunction

        Creates log-survival function transforms for numerical stability
        in extreme tail probability computations.

        .. note::
            This is a convenience method for building an instance of the
            :py:class:`~scistanpy.model.components.transforms.cdf.LogSurvivalFunction`
            subclass associated with the parameter. It can be used as either a class
            method, in which case parameters must be explicitly provided, or an
            instance method (in which case instance parameter values will be used).
        """
        return cls.LOG_SF(**params)

    # pylint: enable=no-self-argument

    def __str__(self) -> str:
        """Return human-readable string representation of the parameter.

        :returns: String showing parameter name and distribution
        :rtype: str

        Creates a readable representation showing the parameter assignment
        in mathematical notation, useful for model inspection and debugging.
        """
        right_side = (
            self.get_right_side(None)
            .replace("[start:end]", "")
            .replace("__", ".")
            .capitalize()
        )
        return f"{self.model_varname} ~ {right_side}"

    @property
    def torch_dist_instance(self) -> "custom_types.SciStanPyDistribution":
        """Get PyTorch distribution instance with current parameter values.

        :returns: Configured PyTorch distribution object
        :rtype: custom_types.SciStanPyDistribution

        Creates a PyTorch distribution instance using the current (constrained)
        parameter values, enabling gradient-based computations and optimization.
        """
        return self.__class__.TORCH_DIST(  # pylint: disable=not-callable
            **{
                self.STAN_TO_TORCH_NAMES[name]: torch.broadcast_to(
                    param.torch_parametrization, self.shape
                )
                for name, param in self._parents.items()
            }
        )

    @property
    def is_hyperparameter(self) -> bool:
        """Check if parameter is a hyperparameter (has only constant parents).

        :returns: True if all parent parameters are constants
        :rtype: bool

        Hyperparameters are top-level parameters in the model hierarchy
        that depend only on fixed constants rather than other random variables.
        """
        return all(isinstance(parent, constants.Constant) for parent in self.parents)

    @property
    def torch_parametrization(self) -> torch.Tensor:
        """Get PyTorch parameter tensor with appropriate constraints applied.

        :returns: Constrained parameter tensor for optimization
        :rtype: torch.Tensor

        :raises ValueError: If called on observable parameters

        Returns the PyTorch parameter tensor with appropriate transformations
        applied to enforce bounds, simplex constraints, or other restrictions.
        The returned tensor is suitable for gradient-based optimization and obeys
        the bounds of the probability distribution.
        """

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
        """Variable name for posterior predictive sampling.

        :returns: Variable name with "_ppc" suffix for generated quantities
        :rtype: str

        :raises ValueError: If called on non-observable parameters

        Observable parameters generate posterior predictive samples in the
        ``generated quantities`` block of Stan code using this modified variable
        name.
        """
        # Only available for observables
        if not self.observable:
            raise ValueError("Generated variables are only available for observables")

        return f"{self.model_varname}_ppc"

    @property
    def observable(self) -> bool:
        """Check if parameter represents observed data.

        :returns: ``True`` if parameter is observable
        :rtype: bool

        Parameters are observable if explicitly marked as such or if they
        have no children (representing terminal nodes in the model graph).
        """
        return self._observable or all(
            isinstance(child, transformed_data.TransformedData)
            for child in self._children
        )

    @property
    def raw_varname(self) -> str:
        """Get raw variable name for reparameterized parameters.

        :returns: Raw variable name with "_raw" suffix, or empty string if there
            is no raw variable name associated with the parameter.
        :rtype: str

        Some parameters use reparameterization techniques (like non-centered
        parameterization) that require separate raw variables in Stan code. This
        property returns the appropriate raw variable name when needed.
        """
        return f"{self.stan_model_varname}_raw" if self.HAS_RAW_VARNAME else ""


class ContinuousDistribution(Parameter, transformed_parameters.TransformableParameter):
    """Base class for parameters with continuous sample spaces.

    This class extends :py:class:`~scistanpy.model.components.parameters.Parameter`
    to provide functionality specific to continuous probability distributions. It
    also inherits transformation capabilities that enable complex hierarchical model
    construction using mathematical operators.
    """


class DiscreteDistribution(Parameter):
    """Base class for parameters with discrete sample spaces.

    This class extends :py:class:`~scistanpy.model.components.parameters.Parameter`
    for discrete probability distributions, handling the specific requirements of
    integer-valued random variables.

    :cvar BASE_STAN_DTYPE: Stan data type for discrete variables ("int")
    :cvar LOWER_BOUND: Default lower bound for discrete values (0)
    """

    BASE_STAN_DTYPE: str = "int"
    """
    Updated relative to :py:class:`~scistanpy.model.components.parameters.Parameter`
    to reflect that discrete parameters are represented as integers in Stan.
    """

    LOWER_BOUND: custom_types.Integer = 0
    """
    Updated relative to :py:class:`~scistanpy.model.components.parameters.Parameter`
    to reflect that all discrete distributions currently implemented in SciStanPy
    are defined for non-negative integers. This sets the default lower bound to 0.
    """


class Normal(ContinuousDistribution):
    r"""Normal (Gaussian) distribution parameter.

    Implements the normal distribution with location (mu) and scale (sigma)
    parameters. Supports automatic non-centered parameterization for improved
    sampling in hierarchical models.

    :param mu: Location parameter (mean)
    :type mu: custom_types.ContinuousParameterType
    :param sigma: Scale parameter (standard deviation)
    :type sigma: custom_types.ContinuousParameterType
    :param noncentered: Whether to use non-centered parameterization in hierarchical
        models. Defaults to True.
    :type noncentered: bool
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        .. math::
            P(x | \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} *
            \exp\left(-\frac{((x-\mu)/\sigma)^2}{2}\right)


    Properties:

    .. list-table::

        * - Support
          - :math:`(-\infty, \infty)`
        * - Mean
          - :math:`\mu`
        * - Median
          - :math:`\mu`
        * - Mode
          - :math:`\mu`
        * - Variance
          - :math:`\sigma^2`

    In hierarchical models, the non-centered parametrization is used by default.
    In this case, a separate raw variable is introduced, and the actual parameter
    is defined as a transformation of this raw variable. This can lead to
    improved sampling efficiency in many scenarios:

        .. math::
            \begin{align*}
            z &\sim \text{Normal}(0, 1) \\
            x &= \mu + \sigma * z
            \end{align*}

    """

    POSITIVE_PARAMS = {"sigma"}
    STAN_DIST = "normal"
    SCIPY_DIST = stats.norm
    TORCH_DIST = custom_torch_dists.Normal
    STAN_TO_SCIPY_NAMES = {"mu": "loc", "sigma": "scale"}
    STAN_TO_TORCH_NAMES = {"mu": "loc", "sigma": "scale"}

    def __init__(
        self,
        *,
        mu: "custom_types.ContinuousParameterType",
        sigma: "custom_types.ContinuousParameterType",
        noncentered: bool = True,
        **kwargs,
    ):
        # Build the instance
        super().__init__(mu=mu, sigma=sigma, **kwargs)

        # Are we using non-centered parameterization?
        self._noncentered = noncentered

    def get_transformation_assignment(self, index_opts: tuple[str, ...]) -> str:
        """Generate Stan code for parameter transformation.

        :param index_opts: Potential names for indexing variables (e.g., ('i', 'j',
            'k', ...)).
        :type index_opts: tuple[str, ...]

        :returns: Stan transformation code.
        :rtype: str

        For non-centered parameterization, returns Stan code that defines the
        parameter as a transformation of a raw variable drawn from a unit normal
        distribution. Otherwise,  uses the parent class default transformation.
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
        """Generate Stan target increment with appropriate variable names.

        :param index_opts: Potential names for indexing variables (e.g., ('i', 'j',
            'k', ...)).
        :type index_opts: tuple[str, ...]

        :returns: Stan target increment statement
        :rtype: str

        For non-centered parameterization, uses the raw variable name in the target
        increment while the transformed variable is computed in the transformed
        parameters block. Otherwise, uses the parent implementation.
        """
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

    def get_right_side(
        self,
        index_opts: tuple[str, ...] | None,
        start_dims: dict[str, custom_types.Integer] | None = None,
        end_dims: dict[str, custom_types.Integer] | None = None,
        offset_adjustment: int = 0,
        dist_suffix: str = "",
    ) -> str:
        """Generate distribution call for Stan code.

        For non-centered parameterization, returns "std_normal()" for the raw variable.
        Otherwise uses the parent implementation.
        """
        # If not noncentered, run the parent method
        if not self.is_noncentered:
            return super().get_right_side(
                index_opts,
                start_dims=start_dims,
                end_dims=end_dims,
                offset_adjustment=offset_adjustment,
                dist_suffix=dist_suffix,
            )

        # Otherwise, make sure we do not have the suffix set and return the standard
        # normal distribution
        assert (
            dist_suffix == ""
        ), "Non-centered parameters should not have a distribution suffix"

        return "std_normal()"

    @property
    def is_noncentered(self) -> bool:
        """Check if parameter uses non-centered parameterization.

        :returns: ``True`` if using non-centered parameterization
        :rtype: bool

        Non-centered parameterization is used when:

        - ``noncentered`` flag is ``True`` during initialization.
        - Parameter is not a hyperparameter
        - Parameter is not observable
        """
        return self._noncentered and not self.is_hyperparameter and not self.observable

    HAS_RAW_VARNAME = is_noncentered


class HalfNormal(ContinuousDistribution):
    r"""Half-normal distribution parameter (normal truncated at zero).

    Implements the half-normal distribution, which is a normal distribution
    truncated to positive values. Commonly used for scale parameters.

    :param sigma: Scale parameter
    :type sigma: custom_types.ContinuousParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        .. math::
            P(x | \sigma) = \frac{2}{\sigma\sqrt{2\pi}} *
            \exp\left(-\frac{x^2}{2\sigma^2}\right) \text{ for } x \geq 0

    Properties:

        .. list-table::

            * - Support
              - :math:`[0, \infty)`
            * - Mean
              - :math:`\sigma\sqrt{\dfrac{2}{\pi}}`
            * - Median
              - :math:`\sigma\sqrt{2}\operatorname{erf}^{-1}(0.5)`
            * - Mode
              - :math:`0`
            * - Variance
              - :math:`\sigma^2\left(1 - \dfrac{2}{\pi}\right)`
    """

    LOWER_BOUND: custom_types.Float = 0.0
    STAN_DIST = "normal"
    SCIPY_DIST = stats.halfnorm
    TORCH_DIST = dist.half_normal.HalfNormal
    STAN_TO_SCIPY_NAMES = {"sigma": "scale"}
    STAN_TO_TORCH_NAMES = {"sigma": "scale"}

    def write_dist_args(self, sigma: str) -> str:  # pylint: disable=arguments-differ
        """Format distribution arguments for Stan.

        :param sigma: Formatted sigma parameter string
        :type sigma: str

        :returns: "0, sigma" for Stan normal distribution call
        :rtype: str
        """
        return f"0, {sigma}"


class UnitNormal(Normal):
    r"""Standard normal distribution (mu=0, sigma=1).

    Implements the standard normal distribution with fixed parameters.
    This is a convenience class for the commonly used N(0,1) distribution.

    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        .. math::
            P(x) = \frac{e^{-\frac{x^2}{2}}}{\sqrt{2\pi}}

    Properties:

    .. list-table::

        * - Support
          - :math:`(-\infty, \infty)`
        * - Mean
          - :math:`0`
        * - Median
          - :math:`0`
        * - Mode
          - :math:`0`
        * - Variance
          - :math:`1`
    """

    STAN_DIST = "std_normal"

    def __init__(self, **kwargs):
        super().__init__(mu=0.0, sigma=1.0, noncentered=False, **kwargs)

        # Sigma is not togglable
        self.sigma.is_togglable = False

    def write_dist_args(  # pylint: disable=arguments-differ, unused-argument
        self, mu: str, sigma: str
    ) -> str:
        """Return empty argument string for Stan std_normal distribution.

        :param mu: Location parameter (unused for std_normal)
        :type mu: str
        :param sigma: Scale parameter (unused for std_normal)
        :type sigma: str

        :returns: Empty string (std_normal takes no arguments)
        :rtype: str
        """
        # No arguments needed for the unit normal distribution in Stan.
        return ""


class LogNormal(ContinuousDistribution):
    r"""Log-normal distribution parameter.

    Implements the log-normal distribution where :math:`\log(X)` follows a normal
    distribution. Commonly used for modeling positive quantities with multiplicative
    effects.

    :param mu: Location parameter for underlying normal
    :type mu: custom_types.ContinuousParameterType
    :param sigma: Scale parameter for underlying normal
    :type sigma: custom_types.ContinuousParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        .. math::
            \begin{align*}
            \text{If } Y &\sim \text{Normal}(\mu, \sigma), \text{then } \\ \\
            X &\sim \text{LogNormal}(\mu, \sigma), \text{where } \\ \\
            P(x | \mu, \sigma) &= \frac{1}{x\sigma\sqrt{2\pi}} *
            \exp\left(-\frac{(\ln(x)-\mu)^2}{2\sigma^2}\right) \text{ for } x > 0
            \end{align*}

    Properties:

        .. list-table::

           * - Support
             - :math:`(0, \infty)`
           * - Mean
             - :math:`\exp\left(\mu + \frac{\sigma^2}{2}\right)`
           * - Median
             - :math:`\exp(\mu)`
           * - Mode
             - :math:`\exp(\mu - \sigma^2)`
           * - Variance
             - :math:`\left(\exp(\sigma^2) - 1\right) \exp\left(2\mu + \sigma^2\right)`
    """

    POSITIVE_PARAMS = {"sigma"}
    LOWER_BOUND: custom_types.Float = 0.0
    STAN_DIST = "lognormal"
    SCIPY_DIST = stats.lognorm
    TORCH_DIST = custom_torch_dists.LogNormal
    STAN_TO_SCIPY_NAMES = {"mu": "scale", "sigma": "s"}
    STAN_TO_TORCH_NAMES = {"mu": "loc", "sigma": "scale"}
    STAN_TO_SCIPY_TRANSFORMS = {"mu": _exp_transform}


class Beta(ContinuousDistribution):
    r"""Beta distribution parameter.

    Implements the beta distribution with shape parameters alpha and beta.
    The distribution has support on (0, 1) and is commonly used for modeling
    probabilities and proportions.

    :param alpha: First shape parameter (concentration)
    :type alpha: custom_types.ContinuousParameterType
    :param beta: Second shape parameter (concentration)
    :type beta: custom_types.ContinuousParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:

        .. math::
            \begin{align*}
            P(x | \alpha, \beta) &= \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} *
            x^{\alpha - 1} (1 - x)^{\beta - 1} \text{ for } 0 < x < 1 \\
            \text{where } \Gamma(z) &= \int_0^\infty t^{z-1} e^{-t} dt
            \end{align*}

    Properties:

        .. list-table::

           * - Support
             - :math:`(0, 1)`
           * - Mean
             - :math:`\frac{\alpha}{\alpha + \beta}`
           * - Mode
             - :math:`\frac{\alpha - 1}{\alpha + \beta - 2}` for :math:`\alpha, \beta > 1`
           * - Variance
             - :math:`\frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}`

    Common Applications:

        - Prior distributions for probabilities
        - Modeling proportions and percentages
        - Bayesian A/B testing
        - Mixture model component weights
    """

    POSITIVE_PARAMS = {"alpha", "beta"}
    LOWER_BOUND: custom_types.Float = 0.0
    UPPER_BOUND: custom_types.Float = 1.0
    STAN_DIST = "beta"
    SCIPY_DIST = stats.beta
    TORCH_DIST = dist.beta.Beta
    STAN_TO_SCIPY_NAMES = {"alpha": "a", "beta": "b"}
    STAN_TO_TORCH_NAMES = {"alpha": "concentration1", "beta": "concentration0"}


class Gamma(ContinuousDistribution):
    r"""Gamma distribution parameter.

    Implements the gamma distribution with shape (alpha) and rate (beta)
    parameters. Commonly used for modeling positive continuous quantities
    with specific shape characteristics.

    :param alpha: Shape parameter
    :type alpha: custom_types.ContinuousParameterType
    :param beta: Rate parameter
    :type beta: custom_types.ContinuousParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        .. math::
            \begin{align*}
            P(x | \alpha, \beta) &= \frac{\beta^\alpha}{\Gamma(\alpha)} *
            x^{\alpha - 1} e^{-\beta x} \text{ for } x > 0 \\
            \text{where } \Gamma(z) &= \int_0^\infty t^{z-1} e^{-t} dt
            \end{align*}

    Properties:

        .. list-table::

            * - Support
              - :math:`(0, \infty)`
            * - Mean
              - :math:`\frac{\alpha}{\beta}`
            * - Mode
              - :math:`\frac{\alpha - 1}{\beta}` for :math:`\alpha > 1`
            * - Variance
              - :math:`\frac{\alpha}{\beta^2}`
    """

    POSITIVE_PARAMS = {"alpha", "beta"}
    LOWER_BOUND: custom_types.Float = 0.0
    STAN_DIST = "gamma"
    SCIPY_DIST = stats.gamma
    TORCH_DIST = dist.gamma.Gamma
    STAN_TO_SCIPY_NAMES = {"alpha": "a", "beta": "scale"}
    STAN_TO_TORCH_NAMES = {"alpha": "concentration", "beta": "rate"}
    STAN_TO_SCIPY_TRANSFORMS = {
        "beta": _inverse_transform
    }  # Transform beta to match the scipy distribution's scale parameter


class InverseGamma(ContinuousDistribution):
    r"""Inverse gamma distribution parameter.

    Implements the inverse gamma distribution, commonly used as a conjugate
    prior for variance parameters in Bayesian analysis.

    :param alpha: Shape parameter
    :type alpha: custom_types.ContinuousParameterType
    :param beta: Scale parameter
    :type beta: custom_types.ContinuousParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        .. math::
            \begin{align*}
            P(x | \alpha, \beta) &= \frac{\beta^\alpha}{\Gamma(\alpha)} *
            x^{-\alpha - 1} e^{-\beta / x} \text{ for } x > 0 \\
            \text{where } \Gamma(z) &= \int_0^\infty t^{z-1} e^{-t} dt
            \end{align*}

    Properties:

        .. list-table::

            * - Support
              - :math:`(0, \infty)`
            * - Mean
              - :math:`\frac{\beta}{\alpha - 1}` for :math:`\alpha > 1`
            * - Mode
              - :math:`\frac{\beta}{\alpha + 1}`
            * - Variance
              - :math:`\frac{\beta^2}{(\alpha - 1)^2(\alpha - 2)}` for :math:`\alpha > 2`

    Common Applications:

        - Conjugate prior for normal variance
        - Hierarchical modeling of scale parameters
        - Bayesian regression variance modeling
"""

    POSITIVE_PARAMS = {"alpha", "beta"}
    LOWER_BOUND: custom_types.Float = 0.0
    STAN_DIST = "inv_gamma"
    SCIPY_DIST = stats.invgamma
    TORCH_DIST = dist.inverse_gamma.InverseGamma
    STAN_TO_SCIPY_NAMES = {"alpha": "a", "beta": "scale"}
    STAN_TO_TORCH_NAMES = {"alpha": "concentration", "beta": "rate"}


class Exponential(ContinuousDistribution):
    r"""Exponential distribution parameter.

    Implements the exponential distribution with rate parameter beta.
    Commonly used for modeling waiting times and survival analysis.

    :param beta: Rate parameter
    :type beta: custom_types.ContinuousParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        .. math::
            \begin{align*}
            P(x | \beta) &= \beta e^{-\beta x} \text{ for } x \geq 0
            \end{align*}

    Properties:

        .. list-table::

           * - Support
             - :math:`[0, \infty)`
           * - Mean
             - :math:`\frac{1}{\beta}`
           * - Mode
             - :math:`0`
           * - Variance
             - :math:`\frac{1}{\beta^2}`
    """

    POSITIVE_PARAMS = {"beta"}
    LOWER_BOUND: custom_types.Float = 0.0
    STAN_DIST = "exponential"
    SCIPY_DIST = stats.expon
    TORCH_DIST = dist.exponential.Exponential
    STAN_TO_SCIPY_NAMES = {"beta": "scale"}
    STAN_TO_TORCH_NAMES = {"beta": "rate"}
    STAN_TO_SCIPY_TRANSFORMS = {
        "beta": _inverse_transform
    }  # Transform beta to match the scipy distribution's scale parameter


class ExpExponential(Exponential):
    r"""Exp-Exponential distribution (log of exponential random variable).

    Implements the distribution of :math:`Y` where :math:`\exp(Y) \sim
    \text{Exponential}(\beta)`.

    :param beta: Rate parameter for the underlying exponential
    :type beta: custom_types.ContinuousParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        .. math::
            \begin{align*}
            \text{If } X &\sim \text{Exponential}(\beta), \text{then } \\ \\
            Y &= \log(X) \sim \text{ExpExponential}(\beta), \text{where } \\ \\
            P(y | \beta) &= \beta * \exp(y - \beta * \exp(y)) \text{ for } y \in (-\infty, \infty)
            \end{align*}

    Properties:

        - Support: :math:`(-\infty, \infty)`
        - Related to Gumbel distribution family
        - Useful for log-scale modeling of exponential processes

    This distribution requires custom Stan functions for implementation (see
    :doc:`../stan/stan_functions`) which are automatically included in any Stan
    program defined using this distribution.
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
        """Return list of required Stan function includes.

        :returns: List containing the expexponential.stanfunctions include
        :rtype: list[str]

        This distribution requires custom Stan functions for proper implementation
        of the exp-exponential density and random number generation.
        """
        # We need to extend the set of supporting functions to include the custom
        # Stan functions for the Exp-Exponential distribution
        return super().get_supporting_functions() + [
            "#include expexponential.stanfunctions"
        ]


class Lomax(ContinuousDistribution):
    r"""Lomax distribution (Pareto Type II with location=0).

    Implements the Lomax distribution, which is a special case of the
    Pareto Type II distribution with location parameter set to 0.

    :param lambda\_: Scale parameter
    :type lambda\_: custom_types.ContinuousParameterType
    :param alpha: Shape parameter
    :type alpha: custom_types.ContinuousParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        .. math::
            \begin{align*}
            P(x | \lambda, \alpha) &= \frac{\alpha}{\lambda} *
            \left(1 + \frac{x}{\lambda}\right)^{-\alpha - 1} \text{ for } x \geq 0
            \end{align*}

    Properties:

        .. list-table::

            * - Support
              - :math:`[0, \infty)`
            * - Mean
              - :math:`\frac{\lambda}{\alpha - 1}` for :math:`\alpha > 1`
            * - Mode
              - :math:`0`
            * - Variance
              - :math:`\frac{\lambda^2}{(\alpha - 1)^2(\alpha - 2)}` for :math:`\alpha > 2`


    Common Applications:

        - Modeling income distributions
        - Network analysis (degree distributions)
        - Reliability engineering
        - Extreme value modeling
    """

    LOWER_BOUND = 0.0
    POSITIVE_PARAMS = {"lambda_", "alpha"}
    STAN_DIST = "pareto_type_2"
    SCIPY_DIST = stats.lomax
    TORCH_DIST = custom_torch_dists.Lomax
    STAN_TO_SCIPY_NAMES = {"lambda_": "scale", "alpha": "c"}
    STAN_TO_TORCH_NAMES = {"lambda_": "lambda_", "alpha": "alpha"}

    def write_dist_args(  # pylint: disable=arguments-differ
        self, lambda_: str, alpha: str
    ) -> str:
        r"""Format arguments for Stan pareto_type_2 distribution.

        :param lambda\_: Formatted lambda parameter string
        :type lambda\_: str
        :param alpha: Formatted alpha parameter string
        :type alpha: str

        :returns: "0.0, lambda\_, alpha" for Stan distribution call
        :rtype: str

        The Lomax distribution is implemented in Stan as Pareto Type II
        with location parameter fixed at 0.0.
        """
        return f"0.0, {lambda_}, {alpha}"


class ExpLomax(ContinuousDistribution):
    r"""Exp-Lomax distribution (log of Lomax random variable).

    Implements the distribution of :math:`Y` where :math:`\exp(Y) \sim
    \text{Lomax}(\lambda, \alpha)`.

    :param lambda\_: Scale parameter for underlying Lomax
    :type lambda\_: custom_types.ContinuousParameterType
    :param alpha: Shape parameter for underlying Lomax
    :type alpha: custom_types.ContinuousParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        .. math::
            \begin{align*}
            \text{If } X &\sim \text{Lomax}(\lambda, \alpha), \text{then } \\ \\
            Y &= \log(X) \sim \text{ExpLomax}(\lambda, \alpha), \text{where } \\ \\
            P(y | \lambda, \alpha) &= \frac{\alpha}{\lambda} * \exp(y) *
            \left(1 + \frac{\exp(y)}{\lambda}\right)^{-\alpha - 1} \text{ for } y
            \in (-\infty, \infty)
            \end{align*}

    This distribution requires custom Stan functions for implementation (see
    :doc:`../stan/stan_functions`) which are automatically included in any Stan
    program defined using this distribution.
"""

    LOWER_BOUND = None
    POSITIVE_PARAMS = {"lambda_", "alpha"}
    STAN_DIST = "explomax"
    SCIPY_DIST = custom_scipy_dists.explomax
    TORCH_DIST = custom_torch_dists.ExpLomax
    STAN_TO_SCIPY_NAMES = {"lambda_": "scale", "alpha": "c"}
    STAN_TO_TORCH_NAMES = {"lambda_": "lambda_", "alpha": "alpha"}

    def get_supporting_functions(self) -> list[str]:
        """Return list of required Stan function includes.

        :returns: List containing the explomax.stanfunctions include
        :rtype: list[str]

        This distribution requires custom Stan functions for proper implementation
        of the exp-Lomax density and random number generation.
        """
        # We need to extend the set of supporting functions to include the custom
        # Stan functions for the Exp-Lomax distribution
        return super().get_supporting_functions() + ["#include explomax.stanfunctions"]


class Dirichlet(ContinuousDistribution):
    r"""Dirichlet distribution parameter.

    Implements the Dirichlet distribution for modeling probability simplexes.
    The distribution generates vectors that sum to 1, making it ideal for
    modeling categorical probabilities and mixture weights.

    :param alpha: Concentration parameters (can be scalar or array-like). If scalar,
        will be converted to the appropriate shape given by the `shape` kwarg.
    :type alpha: Union[AbstractModelComponent, npt.ArrayLike]
    :param kwargs: Additional keyword arguments including 'shape' if alpha is scalar

    Mathematical Definition:
        .. math::
            \begin{align*}
            P(x | \alpha) &= \frac{\Gamma(\sum_{i=1}^K \alpha_i)}{\prod_{i=1}^K
            \Gamma(\alpha_i)} \prod_{i=1}^K x_i^{\alpha_i - 1} \text{ for } x_i > 0,
            \sum_{i=1}^K x_i = 1 \\
            \text{where } \Gamma(z) &= \int_0^\infty t^{z-1} e^{-t} dt
            \end{align*}

    Properties:

        .. list-table::

            * - Support
              - :math:`\{x : \sum_{i=1}^K x_i = 1, x_i > 0\}`
            * - Mean
              - :math:`E[X_i] = \frac{\alpha_i}{\sum_{j=1}^K \alpha_j}`
            * - Mode
              - :math:`\frac{\alpha_i - 1}{\sum_{j=1}^K \alpha_j - K}` for all :math:`\alpha_i > 1`
            * - Variance
              - :math:`\frac{\alpha_i (\sum_{j=1}^K \alpha_j - \alpha_i)}
                {(\sum_{j=1}^K \alpha_j)^2 (\sum_{j=1}^K \alpha_j + 1)}`
                for all :math:`\alpha_i > 1`
            * - Covariance
              - :math:`\frac{-\alpha_i \alpha_j}{(\sum_{k=1}^K \alpha_k)^2
                (\sum_{k=1}^K \alpha_k + 1)}` for all :math:`i \neq j` and
                :math:`\alpha_i, \alpha_j > 1`

    Common Applications:

        - Modeling categorical probabilities

    .. note::
        SciStanPy's implementation of the Dirichlet distribution allows for defining
        concentration parameters (:math:`\alpha`) as either scalar values or arrays.
        If a scalar is provided, the `shape` keyword argument must also be specified
        to define the dimensionality of the simplex--the scalar will be expanded
        to a uniform array of that shape. This flexibility enables both symmetric
        Dirichlet distributions (where all components have the same concentration)
        and asymmetric distributions with distinct concentration parameters for each
        component.
    """

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
        """Initialize Dirichlet distribution with concentration parameters.

        :param alpha: Concentration parameters for each component
        :type alpha: Union[AbstractModelComponent, npt.ArrayLike]
        :param kwargs: Additional arguments including 'shape' if alpha is scalar

        :raises ValueError: If alpha is scalar but 'shape' is not provided

        The initialization handles both scalar and vector specifications of
        concentration parameters, automatically creating uniform arrays when
        a scalar value is provided with an explicit shape.
        """
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
    r"""Exp-Dirichlet distribution (log of Dirichlet random variable).

    Implements the distribution of :math:`Y` where :math:`\exp(Y) ~ \text{Dirichlet}(\alpha)`.
    This provides a log-simplex parameterization that can be more numerically
    stable for extreme concentration parameters and extremely high-dimensional
    simplexes, like those encountered when modeling deep mutational scanning data.

    :param alpha: Concentration parameters for underlying Dirichlet
    :type alpha: Union[AbstractModelComponent, npt.ArrayLike]
    :param kwargs: Additional keyword arguments including 'shape' if alpha is scalar

    Mathematical Definition:
        .. math::
            \begin{align*}
            \text{If } X &\sim \text{Dirichlet}(\alpha), \text{then } \\ \\
            Y &= \log(X) \sim \text{ExpDirichlet}(\alpha), \text{where } \\ \\
            P(y | \alpha) &= \frac{\sqrt{K}}{\exp{(y_K)}}\frac{\Gamma(\sum_{i=1}^K \alpha_i)}{\prod_{i=1}^K
            \Gamma(\alpha_i)} \prod_{i=1}^K \exp(y_i \alpha_i) \text{ for }
            y_i \in (-\infty, \infty),
            \sum_{i=1}^K \exp(y_i) = 1 \\
            \text{where } \Gamma(z) &= \int_0^\infty t^{z-1} e^{-t} dt
            \end{align*}

    Properties:

        - Log-probability is computed in the log-simplex space
        - More numerically stable for extreme concentration parameters
        - Suitable for high-dimensional simplexes
        - Useful for modeling compositional data on log scale

    .. note::
        The Exp-Dirichlet distribution is not natively supported in Stan, so this
        implementation includes custom Stan functions for the probability density,
        transformations, and random number generation. These functions are automatically
        included in any Stan program defined using this distribution. Special thanks
        to Sean Pinkney for assistance with deriving the log probability density
        function for the distribution; thanks also to Bob Carpenter and others for
        developing the log-simplex constraint used in SciStanPy. See
        `here <https://discourse.mc-stan.org/t/log-simplex-constraints/39782>`__
        for derivations and `here <https://github.com/bob-carpenter/transforms/tree
        /main/simplex_transforms/stan/transforms>`__ for transforms.

    .. note::
        When used as a hyperparameter (i.e., :math:`\alpha` is constant), the
        normalization constant of the distribution is also constant and can be
        ignored during MCMC sampling to improve computational efficiency. This
        implementation automatically detects when the Exp-Dirichlet is used as a
        hyperparameter and switches to the unnormalized version of the distribution
        in such cases. If :math:`\alpha` is not constant, the normalized version
        is used.
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
        """Return list of required Stan function includes.

        :returns: List containing the expdirichlet.stanfunctions include
        :rtype: list[str]

        This distribution requires custom Stan functions for proper implementation
        of the exp-Dirichlet density, constraint functions, and transformations.
        """
        # We need to extend the set of supporting functions to include the custom
        # Stan functions for the Exp-Dirichlet distribution
        return super().get_supporting_functions() + [
            "#include expdirichlet.stanfunctions"
        ]

    def get_transformation_assignment(self, index_opts: tuple[str, ...]) -> str:
        """Generate Stan code for log-simplex constraint transformation.

        :param index_opts: Indexing options for multi-dimensional parameters
        :type index_opts: tuple[str, ...]

        :returns: Stan transformation code with Jacobian adjustment
        :rtype: str

        Applies the inverse ILR (isometric log-ratio) transformation to
        convert from unconstrained K-1 dimensional space to log-simplex
        while automatically handling the Jacobian adjustment.
        """
        # We constrain and adjust the Jacobian for the transformation
        raw_varname = self.get_indexed_varname(
            index_opts, _name_override=self.raw_varname
        )
        transformed_varname = self.get_indexed_varname(index_opts)

        return f"{transformed_varname} = inv_ilr_log_simplex_constrain_jacobian({raw_varname})"

    def get_right_side(
        self,
        index_opts: tuple[str, ...] | None,
        start_dims: dict[str, custom_types.Integer] | None = None,
        end_dims: dict[str, custom_types.Integer] | None = None,
        offset_adjustment: int = 0,
        dist_suffix: str = "",
    ) -> str:
        """Generate distribution call with appropriate normalization suffix.

        When ExpDirichlet is used as a hyperparameter, the normalization coefficient
        is a constant and so will be ignored during MCMC to improve computational
        efficiency. When ExpDirichlet is not a hyperparameter (values for `alpha`
        are not constant), the normalized version is used.
        """
        # If no suffix is provided, determine whether we are using the normalized
        # or unnormalized version of the distribution. We use unnormalized when
        # the parameter is a hyperparameter with no parents.
        if dist_suffix == "":
            dist_suffix = "unnorm" if self.is_hyperparameter else "norm"

        # Now we just run the parent method
        return super().get_right_side(
            index_opts,
            start_dims=start_dims,
            end_dims=end_dims,
            offset_adjustment=offset_adjustment,
            dist_suffix=dist_suffix,
        )

    def get_raw_stan_parameter_declaration(self, force_basetype: bool = False) -> str:
        """Generate declaration for raw parameter with reduced dimensions.

        :param force_basetype: Whether to force base type declaration
        :type force_basetype: bool

        :returns: Stan parameter declaration for K-1 dimensional raw variable
        :rtype: str

        The raw parameter has K-1 dimensions instead of K to account for
        the simplex constraint. This raw variable is transformed to create the
        log-simplex constrained value.
        """
        # Run the parent method to get the raw variable name
        raw_varname = super().get_raw_stan_parameter_declaration(
            force_basetype=force_basetype
        )

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
    r"""Binomial distribution parameter.

    Implements the binomial distribution for modeling the number of successes
    in a fixed number of independent Bernoulli trials.

    :param N: Number of trials
    :type N: custom_types.DiscreteParameterType
    :param theta: Success probability
    :type theta: custom_types.ContinuousParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        .. math::
            \begin{align*}
            P(X = k | N, \theta) &= \binom{N}{k} \theta^k (1 - \theta)^{N - k}
            \text{ for } k = 0, 1, ..., N
            \end{align*}

    Properties:

        .. list-table::

              * - Support
                - :math:`\{0, 1, 2, ..., N\}`
              * - Mean
                - :math:`N \theta`
              * - Variance
                - :math:`N \theta (1 - \theta)`
              * - Mode
                - :math:`\lfloor (N + 1) \theta \rfloor`

    Common Applications:

        - Number of successes in fixed trials
        - Proportion data with known denominators
    """

    POSITIVE_PARAMS = {"theta", "N"}
    STAN_DIST = "binomial"
    SCIPY_DIST = stats.binom
    TORCH_DIST = dist.binomial.Binomial
    STAN_TO_SCIPY_NAMES = {"N": "n", "theta": "p"}
    STAN_TO_TORCH_NAMES = {"N": "total_count", "theta": "probs"}


class Poisson(DiscreteDistribution):
    r"""Poisson distribution parameter.

    Implements the Poisson distribution for modeling count data with
    a single rate parameter.

    :param lambda_: Rate parameter (mean number of events)
    :type lambda_: custom_types.ContinuousParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        .. math::
            \begin{align*}
            P(X = k | \lambda) &= \frac{\lambda^k e^{-\lambda}}{k!}
            \text{ for } k = 0, 1, 2, ...
            \end{align*}

    Properties:

        .. list-table::

            *  - Support
               - :math:`\{0, 1, 2, ...\}`
            *  - Mean
               - :math:`\lambda`
            *  - Variance
               - :math:`\lambda`
            *  - Mode
               - :math:`\lfloor \lambda \rfloor`

    Common Applications:

        - Event counting (arrivals, defects, etc.)
        - Modeling rare events
        - Count regression
        - Queueing theory
    """

    POSITIVE_PARAMS = {"lambda_"}
    STAN_DIST = "poisson"
    SCIPY_DIST = stats.poisson
    TORCH_DIST = dist.poisson.Poisson
    STAN_TO_SCIPY_NAMES = {"lambda_": "mu"}
    STAN_TO_TORCH_NAMES = {"lambda_": "rate"}


class _MultinomialBase(DiscreteDistribution):
    """Base class for multinomial distribution variants.

    This abstract base class provides common functionality for different
    parameterizations of the multinomial distribution. It handles the
    special case where the number of trials (N) is implicit in Stan
    but explicit in SciPy/PyTorch interfaces.

    The class automatically handles parameter transformation and provides
    custom target incrementation that removes the N parameter from Stan
    distribution calls since it's implicit in the multinomial definition.
    """

    STAN_TO_NP_TRANSFORMS = {
        "N": functools.partial(np.squeeze, axis=-1)
    }  # Squeeze the N parameter to match the numpy distribution's expected shape

    def get_target_incrementation(self, index_opts: tuple[str, ...]) -> str:
        """Generate target increment with N parameter removed.

        :param index_opts: Indexing options for multi-dimensional parameters
        :type index_opts: tuple[str, ...]

        :returns: Stan target increment without N parameter
        :rtype: str

        The multinomial distribution in Stan doesn't explicitly include
        the number of trials N in the function call, so this method
        removes it from the generated code.
        """
        # We need to strip the N parameter from the declaration as this is implicit
        # in the distribution as defined in Stan
        raw = super().get_target_incrementation(index_opts)

        # Remove the N parameter
        *raw, _ = raw.split(", ")
        return ", ".join(raw) + ")"


class Multinomial(_MultinomialBase):
    r"""Standard multinomial distribution parameter.

    Implements the multinomial distribution with probability vector :math:`\theta`
    and number of trials :math:`N`. Models the number of observations in each
    category for a fixed number of trials.

    :param theta: Probability vector (must sum to 1)
    :type theta: custom_types.ContinuousParameterType
    :param N: Number of trials
    :type N: custom_types.DiscreteParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:

        .. math::
            \begin{align*}
            P(X = x | N, \theta) &= \frac{N!}{\prod_{i=1}^K x_i!} \prod_{i=1}^K
            \theta_i^{x_i} \text{ for } x_i \geq 0, \sum_{i=1}^K x_i = N \\
            \end{align*}

    Properties:
        .. list-table::

            * - Support
              - :math:`\{x : \sum x_i = N, x_i \geq 0\}`
            * - Mean
              - :math:`N \theta_i`
            * - Variance
              - :math:`N \theta_i (1 - \theta_i)`
            * - Covariance
              - :math:`-N \theta_i \theta_j`

    Common Applications:

        - Categorical count data
        - Multinomial logistic regression
        - Natural language processing (word counts)
        - Genetics (allele frequencies)
        - Marketing (customer choice modeling)
    """

    SIMPLEX_PARAMS = {"theta"}
    STAN_DIST = "multinomial"
    SCIPY_DIST = custom_scipy_dists.multinomial
    TORCH_DIST = custom_torch_dists.Multinomial
    STAN_TO_SCIPY_NAMES = {"theta": "p", "N": "n"}
    STAN_TO_TORCH_NAMES = {"theta": "probs", "N": "total_count"}


class MultinomialLogit(_MultinomialBase):
    r"""Multinomial distribution with logit parameterization.

    Implements the multinomial distribution parameterized by logits (:math:`\gamma`)
    rather than probabilities. Logits are unconstrained real values that are transformed
    to probabilities via the softmax function. This parameterization is useful for
    models that naturally work with logits, such as logistic regression extensions.

    :param gamma: Logit vector (unconstrained real values)
    :type gamma: custom_types.ContinuousParameterType
    :param N: Number of trials
    :type N: custom_types.DiscreteParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        .. math::
            \begin{align*}
            \theta_i &= \frac{\exp(\gamma_i)}{\sum_{j=1}^K \exp(\gamma_j)} \\ \\
            X &\sim \text{Multinomial}(N, \theta)
            \end{align*}

    .. warning::
        It is easy to create non-identifiable models using this parameterization.
        Make sure to include appropriate priors or constraints on logits to ensure
        model identifiability.

    """

    STAN_DIST = "multinomial_logit"
    SCIPY_DIST = custom_scipy_dists.multinomial_logit
    TORCH_DIST = custom_torch_dists.MultinomialLogit
    STAN_TO_SCIPY_NAMES = {"gamma": "logits", "N": "n"}
    STAN_TO_TORCH_NAMES = {"gamma": "logits", "N": "total_count"}


class MultinomialLogTheta(_MultinomialBase):
    r"""Multinomial distribution with log-probability parameterization.

    Implements the multinomial distribution in terms of the log of the theta
    parameter. This parameterization is useful for models that naturally
    work with log-probabilities. It can use the ExpDirichlet distribution as a
    prior to enforce the log-simplex constraint and keep computations completely
    in the log-probability space. This is extremely useful for high-dimensional
    multinomials where categories may have very small probabilities (e.g., deep
    mutational scanning data).

    :param log_theta: Log probability vector (log-simplex constraint)
    :type log_theta: custom_types.ContinuousParameterType
    :param N: Number of trials
    :type N: custom_types.DiscreteParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        .. math::
            \begin{align*}
            \theta_i &= \exp(\log{\theta_i}) \quad \text{(with normalization: }
            \sum_j \exp(\log{\theta_j}) = 1) \\
            X &\sim \text{Multinomial}(N, \theta)
            \end{align*}

    .. note::
        This distribution is not natively supported in Stan, so this implementation
        includes custom Stan functions for the probability density and random
        number generation. These functions are automatically included in any Stan
        program defined using this distribution.

    .. hint::
        When used to model observed data (i.e., as an observable parameter),
        the multinomial coefficient is automatically calculated and included in
        the ``transformed data`` block of Stan code. This improves computational
        efficiency by eliminating redundant calculations during MCMC sampling.
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
        log_theta: "custom_types.ContinuousParameterType",
        N: "custom_types.DiscreteParameterType",
        **kwargs,
    ):
        """Initialize MultinomialLogTheta with automatic coefficient handling.

        :param log_theta: Log probability parameter
        :type log_theta: custom_types.ContinuousParameterType
        :param N: Number of trials parameter
        :type N: custom_types.DiscreteParameterType
        :param kwargs: Additional keyword arguments

        The initialization automatically creates a multinomial coefficient
        component that is removed if the parameter gains children (indicating
        it's not an observable).
        """

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
        """Handle removal of coefficient when parameter gains children.

        :param child: Child component being added
        :type child: AbstractModelComponent

        This method manages the multinomial coefficient based on the parameter's
        role in the model. The coefficient is automatically removed when the
        parameter gains non-coefficient children, indicating it's not purely
        an observable.
        """
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
        """Return list of required Stan function includes.

        :returns: List containing the multinomial.stanfunctions include
        :rtype: list[str]

        This distribution requires custom Stan functions for efficient
        computation of the multinomial log-theta density and normalization.
        """
        return super().get_supporting_functions() + [
            "#include multinomial.stanfunctions"
        ]

    def write_dist_args(  # pylint: disable=arguments-differ, arguments-renamed
        self, log_theta: str, N: str, coeff: str = ""
    ):
        """Format distribution arguments with optional coefficient.

        :param log_theta: Formatted log_theta parameter string
        :type log_theta: str
        :param N: Formatted N parameter string
        :type N: str
        :param coeff: Formatted coefficient string (optional). Defaults to "".
        :type coeff: str

        :returns: Formatted argument string for Stan distribution call
        :rtype: str

        The method handles inclusion of the multinomial coefficient when available,
        enabling more efficient computation by pre-calculating constant terms.
        """
        # If the coefficient is provided, insert it in the middle of the arguments.
        # Otherwise, just return the log_theta and N parameters. This is a bit of
        # a hack to make sure that "N" is stripped off by `get_target_incrementation`
        # regardless of whether the coefficient is provided or not.
        if coeff:
            return f"{log_theta}, {coeff}, {N}"
        return f"{log_theta}, {N}"

    def get_right_side(
        self,
        index_opts: tuple[str, ...] | None,
        start_dims: dict[str, custom_types.Integer] | None = None,
        end_dims: dict[str, custom_types.Integer] | None = None,
        offset_adjustment: int = 0,
        dist_suffix: str = "",
    ) -> str:
        """Generate distribution call with appropriate normalization and coefficients.

        :param index_opts: See parent method for details.
        :type index_opts: Optional[tuple[str, ...]]
        :param start_dims: See parent method for details.
        :type start_dims: Optional[dict[str, custom_types.Integer]]
        :param end_dims: See parent method for details.
        :type end_dims: Optional[dict[str, custom_types.Integer]]
        :param offset_adjustment: See parent method for details.
        :type offset_adjustment: int
        :param dist_suffix: Distribution function suffix
        :type dist_suffix: str

        :returns: Stan distribution call with proper normalization
        :rtype: str

        The method automatically selects between manual normalization (with
        coefficient) and standard normalization based on coefficient availability.
        """
        # Get the formattables
        formattables = super(Parameter, self).get_right_side(
            index_opts,
            start_dims=start_dims,
            end_dims=end_dims,
            offset_adjustment=offset_adjustment,
        )

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
        """Get the multinomial coefficient component if it exists.

        :returns: Multinomial coefficient component or None
        :rtype: Optional[LogMultinomialCoefficient]

        The coefficient is automatically created for observable parameters
        and removed when the parameter gains other children, optimizing
        computation by pre-calculating constant terms.
        """
        return self._coefficient
