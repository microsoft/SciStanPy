"""Parameter classes for defining probabilistic model components in SciStanPy.

This module provides the core parameter classes that serve as building blocks for
constructing probabilistic models in SciStanPy. These classes represent random
variables with specific probability distributions and handle the complex task of
translating between Python model specifications, PyTorch modules, and Stan probabilistic
programming language code.

Parameter Type Hierarchy:
    - **Parameter**: Base class for all probabilistic model components
    - **ContinuousDistribution**: Parameters with continuous sample spaces
    - **DiscreteDistribution**: Parameters with discrete sample spaces

Key Features:
    - **Multi-Backend Support**: Integration with SciPy, PyTorch, and Stan
    - **Automatic Parameterization**: Intelligent handling of parameter bounds and constraints
    - **Non-Centered Parameterization**: Automatic reparameterization for improved sampling
    - **Custom Distributions**: Extended distribution library beyond standard offerings
    - **Type Safety**: Comprehensive type checking and validation

Stan Code Generation:
    Each parameter class automatically generates appropriate Stan code including:
    - Variable declarations with proper constraints
    - Target increment statements for log-probability
    - Generated quantities for posterior predictive sampling
    - Support for custom Stan functions when needed

Distribution Support:
    **Continuous Distributions:**
    - Normal, HalfNormal, UnitNormal, LogNormal
    - Beta, Gamma, InverseGamma, Exponential
    - Dirichlet, ExpDirichlet (log-simplex)
    - Custom: ExpExponential, Lomax, ExpLomax

    **Discrete Distributions:**
    - Binomial, Poisson
    - Multinomial variants: standard, logit, log-theta parameterizations

Advanced Features:
    - **Automatic Reparameterization**: Non-centered parameterization for hierarchical models
    - **Constraint Handling**: Automatic bound enforcement and transformations
    - **Observable Support**: Automatic identification of observed data model components
    - **PyTorch Integration**: Native support for gradient-based optimization
    - **Custom Function Integration**: Automatic inclusion of required Stan functions

The parameter classes are designed to be composable, allowing complex hierarchical
models to be built through simple parameter relationships while maintaining
mathematical rigor and computational efficiency.
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

# pylint: disable=too-many-lines


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


def _exp_transform(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
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

    This metaclass automatically creates cumulative distribution function (CDF)
    and related transform classes for each Parameter subclass, enabling automatic
    generation of probabilistic transforms and survival functions.

    The metaclass creates four transform classes for each parameter:

    - CDF: Cumulative distribution function
    - SF: Survival function (complementary CDF)
    - LOG_CDF: Logarithmic CDF
    - LOG_SF: Logarithmic survival function

    """

    def __init__(cls, name, bases, attrs):
        """Create CDF transform classes for the Parameter subclass.

        :param name: Name of the class being created
        :type name: str
        :param bases: Base classes for the new class
        :type bases: tuple
        :param attrs: Class attributes dictionary
        :type attrs: dict

        This method automatically generates transform classes by creating
        new class types that inherit from the appropriate CDF base classes
        and reference the current parameter class.
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


class ClassOrInstanceMethod:
    """Descriptor enabling dual class/instance method behavior.

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

    :cvar STAN_DIST: Stan distribution name for code generation
    :type STAN_DIST: str
    :cvar HAS_RAW_VARNAME: Whether parameter uses a raw/transformed parameterization
    :type HAS_RAW_VARNAME: bool
    :cvar SCIPY_DIST: Corresponding SciPy distribution class
    :type SCIPY_DIST: Optional[Union[type[stats.rv_continuous], type[stats.rv_discrete]]]
    :cvar TORCH_DIST: Corresponding PyTorch distribution class
    :type TORCH_DIST: Optional[Union[type[dist.distribution.Distribution], type[custom_torch_dists.CustomDistribution]]]
    :cvar STAN_TO_SCIPY_NAMES: Parameter name mapping for SciPy interface
    :type STAN_TO_SCIPY_NAMES: dict[str, str]
    :cvar STAN_TO_TORCH_NAMES: Parameter name mapping for PyTorch interface
    :type STAN_TO_TORCH_NAMES: dict[str, str]
    :cvar STAN_TO_SCIPY_TRANSFORMS: Parameter transformation functions converting between
        Stan and SciPy parametrizations
    :type STAN_TO_SCIPY_TRANSFORMS: dict[str, Callable[[npt.NDArray], npt.NDArray]]
    :cvar CDF: Automatically generated CDF transform class
    :type CDF: type[cdfs.CDF]
    :cvar SF: Automatically generated SF transform class
    :type SF: type[cdfs.SurvivalFunction]
    :cvar LOG_CDF: Automatically generated log CDF transform class
    :type LOG_CDF: type[cdfs.LogCDF]
    :cvar LOG_SF: Automatically generated log SF transform class
    :type LOG_SF: type[cdfs.LogSurvivalFunction]

    The class automatically handles:

    - Parameter validation and type checking
    - Stan code generation for all model blocks
    - PyTorch parameter initialization and management
    - Observable/latent parameter distinction
    - Bound enforcement and constraint handling

    Key Capabilities:

    - **Multi-Backend Integration**: Works with SciPy, PyTorch, and Stan
    - **Automatic Code Generation**: Generates appropriate Stan syntax
    - **Type Safety**: Validates parameter types and constraints
    - **Flexible Parameterization**: Supports both raw and transformed parameters
    - **Observable Support**: Can represent both latent variables and observed data
    """

    STAN_DIST: str = ""
    HAS_RAW_VARNAME: bool = False
    CDF: type[cdfs.CDF]
    SF: type[cdfs.SurvivalFunction]
    LOG_CDF: type[cdfs.LogCDF]
    LOG_SF: type[cdfs.LogSurvivalFunction]
    SCIPY_DIST: type[stats.rv_continuous] | type[stats.rv_discrete] | None = None
    TORCH_DIST: (
        type[dist.distribution.Distribution]
        | type[custom_torch_dists.CustomDistribution]
        | None
    ) = None
    STAN_TO_SCIPY_NAMES: dict[str, str] = {}
    STAN_TO_TORCH_NAMES: dict[str, str] = {}
    STAN_TO_SCIPY_TRANSFORMS: dict[str, Callable[[npt.NDArray], npt.NDArray]] = {}

    def __init__(self, **kwargs):
        """Initialize parameter with distribution-specific arguments.

        :param kwargs: Distribution parameters (mu, sigma, etc. depending on subclass)

        :raises NotImplementedError: If required class attributes are missing (i.e.,
            if subclass was incorrectly defined)
        :raises TypeError: If required distribution parameters are missing

        The initialization process:
        1. Validates all required class attributes are defined
        2. Checks that all required distribution parameters are provided
        3. Initializes parent AbstractModelComponent
        4. Sets up observability tracking and PyTorch parameter placeholders
        """
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
        """Initialize PyTorch parameter on **unconstrained** space for gradient-based
        optimization.

        :param init_val: Initial parameter values. Uniform between -1 and 1 if None.
            Defaults to None.
        :type init_val: Optional[Union[npt.NDArray, torch.Tensor]]
        :param seed: Random seed for initialization. Defaults to None.
        :type seed: Optional[custom_types.Integer]

        :raises ValueError: If called on observable parameters
        :raises ValueError: If init_val shape doesn't match parameter shape

        This method sets up the parameter for PyTorch-based optimization by
        creating a trainable nn.Parameter with appropriate initialization.
        The initialization strategy uses uniform random values in [-1, 1]
        if no explicit values are provided. Note that initialization values are
        on the unconstrained space. An appropriate transform is applied depending
        on the bounds of the distribution represented by the class to take it to
        a constrained space.

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

        :param index_opts: Indexing options for multi-dimensional parameters
        :type index_opts: tuple[str, ...]

        :returns: Stan code for target increment (e.g., "y ~ normal(mu, sigma)")
        :rtype: str

        This method generates the Stan code that adds this parameter's
        log-probability contribution to the target density. It handles
        proper indexing for multi-dimensional parameters and constructs
        the appropriate distribution call with parameter values.
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
        enabling posterior predictive sampling by generating new samples
        from the parameter's distribution using fitted parameter values.
        """
        return (
            self.get_indexed_varname(index_opts, _name_override=self.generated_varname)
            + f" = {self.get_right_side(index_opts, dist_suffix='rng')}"
        )

    def get_torch_logprob(
        self, observed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute log-probability using PyTorch backend for gradient computation.

        :param observed: Observed values for observable parameters. Defaults to None.
        :type observed: Optional[torch.Tensor]

        :returns: Log-probability tensor with gradient tracking
        :rtype: torch.Tensor

        :raises ValueError: If observable parameter lacks observed values
        :raises ValueError: If latent parameter has observed values

        This method computes log-probabilities using PyTorch distributions,
        enabling gradient-based optimization. For observable parameters,
        it evaluates the likelihood of observed data. For latent parameters,
        it evaluates the prior probability of current parameter values.
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

        Returns the global SciStanPy RNG if no seed is provided, otherwise
        creates a new generator with the specified seed for reproducible
        sampling operations.
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
        """Generate right-hand side of Stan distribution statements.

        :param index_opts: Indexing options for multi-dimensional parameters
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
            numbers of dimensions. Defaults to 0.
        :type offset_adjustment: int
        :param dist_suffix: Distribution function suffix (e.g., "_rng"). Defaults to "".
        :type dist_suffix: str

        :returns: Stan distribution call string
        :rtype: str

        This method constructs the right-hand side of Stan statements,
        handling parameter indexing, distribution suffixes for different
        Stan blocks, and proper argument formatting.
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
        """Generate Stan transformed data block declarations.

        :returns: Stan code for transformed data declarations (empty by default)
        :rtype: str

        Most parameters don't require transformed data declarations. This
        method can be overridden by subclasses that need to declare
        transformed data variables.
        """
        # None by default
        return ""

    def get_generated_quantity_declaration(self, force_basetype: bool = True) -> str:
        """Generate Stan variable declaration for generated quantities.

        :param force_basetype: Whether to force base type declaration. For example,
            if `True` and the parameter is defined as a multidimension float, the
            returned stan dtype will not be `array[...Ndim - 1...] vector`, but
            `array[...NDim...] float`. Defaults to True, as this is the format
            expected by generated quantities blocks.
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
            `get_generated_quantity_declaration` for more information. Defaults to False.
        :type force_basetype: bool

        :returns: Stan parameter declaration for raw variables
        :rtype: str

        For parameters using non-centered or other reparameterizations,
        this generates declarations for the underlying raw variables that
        are transformed to create the actual parameters. The `get_transformation_assignment`
        function will return Stan code that converts this raw parameter to the
        desired parameter.
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

        Can be used as either a class method (with explicit parameters) or
        instance method (using instance parameter values) to create CDF
        transform objects for probabilistic modeling.

        Example:
            >>> # As class method
            >>> cdf = Normal.cdf(mu=0, sigma=1, x=data)
            >>> # As instance method
            >>> normal_param = Normal(mu=0, sigma=1)
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

        Creates a PyTorch distribution instance using the current parameter
        values, enabling gradient-based computations and optimization.
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
        """Get variable name for posterior predictive sampling.

        :returns: Variable name with "_ppc" suffix for generated quantities
        :rtype: str

        :raises ValueError: If called on non-observable parameters

        Observable parameters generate posterior predictive samples in the
        generated quantities block using a modified variable name.
        """
        # Only available for observables
        if not self.observable:
            raise ValueError("Generated variables are only available for observables")

        return f"{self.model_varname}_ppc"

    @property
    def observable(self) -> bool:
        """Check if parameter represents observed data.

        :returns: True if parameter is observable
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
        parameterization) that require separate raw variables. This property
        returns the appropriate raw variable name when needed.
        """
        return f"{self.stan_model_varname}_raw" if self.HAS_RAW_VARNAME else ""


class ContinuousDistribution(Parameter, transformed_parameters.TransformableParameter):
    """Base class for parameters with continuous sample spaces.

    This class extends Parameter to provide functionality specific to continuous
    probability distributions. It inherits transformation capabilities that
    enable complex hierarchical model construction.
    """


class DiscreteDistribution(Parameter):
    """Base class for parameters with discrete sample spaces.

    This class extends Parameter for discrete probability distributions,
    handling the specific requirements of integer-valued random variables.

    :cvar BASE_STAN_DTYPE: Stan data type for discrete variables ("int")
    :cvar LOWER_BOUND: Default lower bound for discrete values (0)
    """

    BASE_STAN_DTYPE: str = "int"
    LOWER_BOUND: custom_types.Integer = 0


class Normal(ContinuousDistribution):
    """Normal (Gaussian) distribution parameter.

    Implements the normal distribution with location (mu) and scale (sigma)
    parameters. Supports automatic non-centered parameterization for improved
    sampling in hierarchical models.

    :param mu: Location parameter (mean)
    :type mu: custom_types.ContinuousParameterType
    :param sigma: Scale parameter (standard deviation)
    :type sigma: custom_types.ContinuousParameterType
    :param noncentered: Whether to use non-centered parameterization. Defaults to True.
    :type noncentered: bool
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        X ~ Normal(μ, σ) where f(x) = (1/(σ√(2π))) * exp(-½((x-μ)/σ)²)

    Non-Centered Parameterization:
        When enabled for hierarchical models:
        - Raw variable: z ~ Normal(0, 1)
        - Transformed: x = μ + σ * z
        - Can improve MCMC sampling efficiency

    The non-centered parameterization is automatically applied when:
    - noncentered=True (default)
    - Parameter is not a hyperparameter
    - Parameter is not observable

    Example:
        >>> # Standard normal parameter
        >>> mu = Normal(mu=0.0, sigma=1.0)
        >>> # Hierarchical parameter with automatic non-centering
        >>> y = Normal(mu=mu, sigma=0.5)
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

        :param index_opts: Indexing options for multi-dimensional parameters
        :type index_opts: tuple[str, ...]

        :returns: Stan transformation code (non-centered if applicable)
        :rtype: str

        For non-centered parameterization, generates:
        x = mu + sigma .* z_raw

        Otherwise uses the parent class default transformation.
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

        :param index_opts: Indexing options for multi-dimensional parameters
        :type index_opts: tuple[str, ...]

        :returns: Stan target increment statement
        :rtype: str

        For non-centered parameterization, uses the raw variable name
        in the target increment while the transformed variable is computed
        in the transformed parameters block.
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

        For non-centered parameterization, returns "std_normal()" for the
        raw variable. Otherwise uses the parent implementation with full
        parameter specification.
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

        :returns: True if using non-centered parameterization
        :rtype: bool

        Non-centered parameterization is used when:
        - noncentered flag is True
        - Parameter is not a hyperparameter
        - Parameter is not observable
        """
        return self._noncentered and not self.is_hyperparameter and not self.observable

    HAS_RAW_VARNAME = is_noncentered


class HalfNormal(ContinuousDistribution):
    """Half-normal distribution parameter (normal truncated at zero).

    Implements the half-normal distribution, which is a normal distribution
    truncated to positive values. Commonly used for scale parameters.

    :param sigma: Scale parameter
    :type sigma: custom_types.ContinuousParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        X ~ HalfNormal(σ) where f(x) = (2/(σ√(2π))) * exp(-x²/(2σ²)) for x ≥ 0

    Example:
        >>> # Scale parameter for hierarchical model
        >>> tau = HalfNormal(sigma=1.0)
        >>> # Individual-level scale
        >>> sigma_i = HalfNormal(sigma=tau)
    """

    LOWER_BOUND: custom_types.Float = 0.0
    STAN_DIST = "normal"
    SCIPY_DIST = stats.halfnorm
    TORCH_DIST = dist.half_normal.HalfNormal
    STAN_TO_SCIPY_NAMES = {"sigma": "scale"}
    STAN_TO_TORCH_NAMES = {"sigma": "scale"}

    def write_dist_args(self, sigma: str) -> str:  # pylint: disable=arguments-differ
        """Format distribution arguments for Stan (location=0, scale=sigma).

        :param sigma: Formatted sigma parameter string
        :type sigma: str

        :returns: "0, sigma" for Stan normal distribution call
        :rtype: str
        """
        return f"0, {sigma}"


class UnitNormal(Normal):
    """Standard normal distribution (mu=0, sigma=1).

    Implements the standard normal distribution with fixed parameters.
    This is a convenience class for the commonly used N(0,1) distribution.

    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        X ~ N(0, 1) where f(x) = (1/√(2π)) * exp(-x²/2)

    The standard normal distribution:
    - Has fixed parameters that cannot be toggled
    - Uses the "std_normal" distribution in Stan
    - Is commonly used for prior specifications
    - Serves as the basis for non-centered parameterizations

    Example:
        >>> # Standard normal prior
        >>> z = UnitNormal()
        >>> # Used in non-centered parameterization
        >>> x = mu + sigma * z  # Conceptually
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
    """Log-normal distribution parameter.

    Implements the log-normal distribution where log(X) follows a normal
    distribution. Commonly used for modeling positive quantities with
    multiplicative effects.

    :param mu: Location parameter for underlying normal
    :type mu: custom_types.ContinuousParameterType
    :param sigma: Scale parameter for underlying normal
    :type sigma: custom_types.ContinuousParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        If Y ~ Normal(μ, σ), then X = exp(Y) ~ LogNormal(μ, σ)
        f(x) = (1/(xσ√(2π))) * exp(-½((ln(x)-μ)/σ)²) for x > 0


    Example:
        >>> # Scale parameter with log-normal prior
        >>> sigma = LogNormal(mu=0.0, sigma=1.0)
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
    """Beta distribution parameter.

    Implements the beta distribution with shape parameters alpha and beta.
    The distribution has support on (0, 1) and is commonly used for modeling
    probabilities and proportions.

    :param alpha: First shape parameter (concentration)
    :type alpha: custom_types.ContinuousParameterType
    :param beta: Second shape parameter (concentration)
    :type beta: custom_types.ContinuousParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        X ~ Beta(α, β) where f(x) = (Γ(α+β)/(Γ(α)Γ(β))) * x^(α-1) * (1-x)^(β-1)

    Properties:
    - Support: (0, 1)
    - Mean: α/(α+β)
    - Mode: (α-1)/(α+β-2) for α,β > 1
    - Variance: αβ/((α+β)²(α+β+1))

    Common Applications:
    - Prior distributions for probabilities
    - Modeling proportions and percentages
    - Bayesian A/B testing
    - Mixture model component weights

    Example:
        >>> # Uniform prior on probability
        >>> p = Beta(alpha=1.0, beta=1.0)
        >>> # Weakly informative prior favoring smaller probabilities
        >>> p_rare = Beta(alpha=1.0, beta=3.0)
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
    """Gamma distribution parameter.

    Implements the gamma distribution with shape (alpha) and rate (beta)
    parameters. Commonly used for modeling positive continuous quantities
    with specific shape characteristics.

    :param alpha: Shape parameter
    :type alpha: custom_types.ContinuousParameterType
    :param beta: Rate parameter
    :type beta: custom_types.ContinuousParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        X ~ Gamma(α, β) where f(x) = (β^α/Γ(α)) * x^(α-1) * exp(-βx)

    Properties:
    - Support: (0, ∞)
    - Mean: α/β
    - Mode: (α-1)/β for α > 1
    - Variance: α/β²

    Note on Parameterization:
    - Stan uses shape-rate parameterization: Gamma(α, β)
    - SciPy uses shape-scale parameterization: Gamma(α, 1/β)
    - Automatic transformation handles this difference

    Example:
        >>> # Precision parameter (inverse variance)
        >>> tau = Gamma(alpha=2.0, beta=1.0)
        >>> # Positive continuous variable
        >>> waiting_time = Gamma(alpha=shape_param, beta=rate_param)
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
    """Inverse gamma distribution parameter.

    Implements the inverse gamma distribution, commonly used as a conjugate
    prior for variance parameters in Bayesian analysis.

    :param alpha: Shape parameter
    :type alpha: custom_types.ContinuousParameterType
    :param beta: Scale parameter
    :type beta: custom_types.ContinuousParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        X ~ InverseGamma(α, β) where f(x) = (β^α/Γ(α)) * x^(-α-1) * exp(-β/x)

    Properties:
    - Support: (0, ∞)
    - Mean: β/(α-1) for α > 1
    - Mode: β/(α+1)
    - Variance: β²/((α-1)²(α-2)) for α > 2

    Common Applications:
    - Conjugate prior for normal variance
    - Hierarchical modeling of scale parameters
    - Bayesian regression variance modeling

    Example:
        >>> # Prior for variance parameter
        >>> sigma_sq = InverseGamma(alpha=2.0, beta=1.0)
        >>> # Hierarchical variance
        >>> tau_sq = InverseGamma(alpha=a_tau, beta=b_tau)
    """

    POSITIVE_PARAMS = {"alpha", "beta"}
    LOWER_BOUND: custom_types.Float = 0.0
    STAN_DIST = "inv_gamma"
    SCIPY_DIST = stats.invgamma
    TORCH_DIST = dist.inverse_gamma.InverseGamma
    STAN_TO_SCIPY_NAMES = {"alpha": "a", "beta": "scale"}
    STAN_TO_TORCH_NAMES = {"alpha": "concentration", "beta": "rate"}


class Exponential(ContinuousDistribution):
    """Exponential distribution parameter.

    Implements the exponential distribution with rate parameter beta.
    Commonly used for modeling waiting times and survival analysis.

    :param beta: Rate parameter
    :type beta: custom_types.ContinuousParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        X ~ Exponential(β) where f(x) = β * exp(-βx) for x ≥ 0

    Properties:
    - Support: [0, ∞)
    - Mean: 1/β
    - Mode: 0
    - Variance: 1/β²

    Note on Parameterization:
    - Stan uses rate parameterization: Exponential(β)
    - SciPy uses scale parameterization: Exponential(1/β)
    - Automatic transformation handles this difference

    Example:
        >>> # Waiting time parameter
        >>> wait_time = Exponential(beta=1.5)
        >>> # Scale parameter with exponential prior
        >>> tau = Exponential(beta=rate_prior)
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
    """Exp-Exponential distribution (log of exponential random variable).

    Implements the distribution of Y where exp(Y) ~ Exponential(β).
    This is equivalent to Y = log(X) where X ~ Exponential(β).

    :param beta: Rate parameter for the underlying exponential
    :type beta: custom_types.ContinuousParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        If X ~ Exponential(β), then Y = log(X) ~ ExpExponential(β)
        f(y) = β * exp(y - β*exp(y)) for y ∈ (-∞, ∞)

    Properties:
    - Support: (-∞, ∞)
    - Related to Gumbel distribution family
    - Useful for log-scale modeling of exponential processes

    This distribution requires custom Stan functions for implementation (see
    `expexponential.stanfunctions` in the `stan` submodule) which are automatically
    included in any Stan program defined using this distribution.

    Example:
        >>> # Log-scale waiting time
        >>> log_wait = ExpExponential(beta=1.0)
        >>> # Log-transformed scale parameter
        >>> log_scale = ExpExponential(beta=prior_rate)
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
    """Lomax distribution (Pareto Type II with location=0).

    Implements the Lomax distribution, which is a special case of the
    Pareto Type II distribution with location parameter set to 0.

    :param lambda_: Scale parameter
    :type lambda_: custom_types.ContinuousParameterType
    :param alpha: Shape parameter
    :type alpha: custom_types.ContinuousParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        X ~ Lomax(λ, α) where f(x) = (α/λ) * (1 + x/λ)^(-α-1) for x ≥ 0

    Properties:
    - Support: [0, ∞)
    - Mean: λ/(α-1) for α > 1
    - Mode: 0
    - Heavy-tailed distribution

    Common Applications:
    - Modeling income distributions
    - Network analysis (degree distributions)
    - Reliability engineering
    - Extreme value modeling

    Example:
        >>> # Heavy-tailed positive variable
        >>> wealth = Lomax(lambda_=scale_param, alpha=shape_param)
        >>> # Robust scale parameter
        >>> sigma = Lomax(lambda_=1.0, alpha=2.0)
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
    """Exp-Lomax distribution (log of Lomax random variable).

    Implements the distribution of Y where exp(Y) ~ Lomax(λ, α).

    :param lambda_: Scale parameter for underlying Lomax
    :type lambda_: custom_types.ContinuousParameterType
    :param alpha: Shape parameter for underlying Lomax
    :type alpha: custom_types.ContinuousParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        If X ~ Lomax(λ, α), then Y = log(X) ~ ExpLomax(λ, α)
        f(y) = (α/λ) * exp(y) * (1 + exp(y)/λ)^(-α-1) for y ∈ (-∞, ∞)

    Properties:
    - Support: (-∞, ∞)
    - Log-scale version of heavy-tailed Lomax
    - Useful for modeling log-transformed heavy-tailed data

    This distribution requires custom Stan functions for implementation (see
    `explomax.stanfunctions` in the `stan` submodule) which are automatically
    included in any Stan program defined using this distribution.

    Example:
        >>> # Log-scale heavy-tailed variable
        >>> log_income = ExpLomax(lambda_=scale, alpha=shape)
        >>> # Robust log-scale parameter
        >>> log_sigma = ExpLomax(lambda_=1.0, alpha=2.0)
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
    """Dirichlet distribution parameter.

    Implements the Dirichlet distribution for modeling probability simplexes.
    The distribution generates vectors that sum to 1, making it ideal for
    modeling categorical probabilities and mixture weights.

    :param alpha: Concentration parameters (can be scalar or array-like). If scalar,
        will be converted to the appropriate shape given by the `shape` kwarg.
    :type alpha: Union[AbstractModelComponent, npt.ArrayLike]
    :param kwargs: Additional keyword arguments including 'shape' if alpha is scalar

    Mathematical Definition:
        X ~ Dirichlet(α) where f(x) = (Γ(Σαᵢ)/Πᵢ Γ(αᵢ)) * Πᵢ xᵢ^(αᵢ-1)

    Properties:
    - Support: K-dimensional simplex (Σxᵢ = 1, xᵢ > 0)
    - Mean: E[Xᵢ] = αᵢ/Σαⱼ
    - Mode: (αᵢ-1)/(Σαⱼ-K) for all αᵢ > 1

    Parameter Handling:
    - If alpha is scalar, 'shape' must be provided to create uniform concentration
    - If alpha is array-like, it defines the concentration for each component

    Example:
        >>> # Uniform Dirichlet (symmetric)
        >>> p = Dirichlet(alpha=1.0, shape=(3,))
        >>> # Non-uniform concentrations
        >>> p = Dirichlet(alpha=np.array([0.5, 1.0, 2.0]))
        >>> # Hierarchical concentration
        >>> p = Dirichlet(alpha=alpha_prior)
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
    """Exp-Dirichlet distribution (log of Dirichlet random variable).

    Implements the distribution of Y where exp(Y) ~ Dirichlet(α).
    This provides a log-simplex parameterization that can be more numerically
    stable for extreme concentration parameters and extremely high-dimensional
    simplexes, like those encountered when modeling deep mutational scanning data.

    :param alpha: Concentration parameters for underlying Dirichlet
    :type alpha: Union[AbstractModelComponent, npt.ArrayLike]
    :param kwargs: Additional keyword arguments including 'shape' if alpha is scalar

    Mathematical Definition:
        If X ~ Dirichlet(α), then Y = log(X) ~ ExpDirichlet(α)
        where Σexp(Yᵢ) = 1 (log-simplex constraint)

    Properties:

    - Support: Log-simplex {y : Σexp(yᵢ) = 1}
    - Raw parameterization uses K-1 dimensions with constraint transformation
    - More numerically stable for extreme concentrations

    Stan Implementation:

    - Uses custom constraint functions for log-simplex transformation
    - Raw parameter has K-1 dimensions (reduced for constraint)
    - Special thanks to Sean Pinkney for assistance with deriving the log probability
      density function; thanks also to Bob Carpenter and others for developing the
      log-simplex constraint used in SciStanPy. See
      `here <https://discourse.mc-stan.org/t/log-simplex-constraints/39782>_` for
      derivations and
      `here <https://github.com/bob-carpenter/transforms/tree/main/simplex_transforms/stan/transforms>`_
      for transforms.

    This distribution requires custom Stan functions for implementation (see
    `expdirichlet.stanfunctions` in the `stan` submodule) which are automatically
    included in any Stan program defined using this distribution.

    Example:
        >>> # Log-simplex probabilities
        >>> log_p = ExpDirichlet(alpha=alpha_vec)
        >>> # Numerically stable log-probabilities
        >>> log_weights = ExpDirichlet(alpha=1.0, shape=(10,))
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
    """Binomial distribution parameter.

    Implements the binomial distribution for modeling the number of successes
    in a fixed number of independent Bernoulli trials.

    :param N: Number of trials
    :type N: custom_types.DiscreteParameterType
    :param theta: Success probability
    :type theta: custom_types.ContinuousParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        X ~ Binomial(N, θ) where P(X = k) = C(N,k) * θᵏ * (1-θ)^(N-k)

    Properties:
    - Support: {0, 1, 2, ..., N}
    - Mean: N * θ
    - Variance: N * θ * (1-θ)
    - Mode: floor((N+1) * θ)

    Common Applications:
    - Number of successes in fixed trials
    - Proportion data with known denominators

    Example:
        >>> # Number of successes in 10 trials
        >>> successes = Binomial(N=10, theta=success_prob)
    """

    POSITIVE_PARAMS = {"theta", "N"}
    STAN_DIST = "binomial"
    SCIPY_DIST = stats.binom
    TORCH_DIST = dist.binomial.Binomial
    STAN_TO_SCIPY_NAMES = {"N": "n", "theta": "p"}
    STAN_TO_TORCH_NAMES = {"N": "total_count", "theta": "probs"}


class Poisson(DiscreteDistribution):
    """Poisson distribution parameter.

    Implements the Poisson distribution for modeling count data with
    a single rate parameter.

    :param lambda_: Rate parameter (mean number of events)
    :type lambda_: custom_types.ContinuousParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        X ~ Poisson(λ) where P(X = k) = (λᵏ * exp(-λ)) / k!

    Properties:
    - Support: {0, 1, 2, 3, ...}
    - Mean: λ
    - Variance: λ
    - Mode: floor(λ)

    Common Applications:
    - Event counting (arrivals, defects, etc.)
    - Modeling rare events
    - Count regression
    - Queueing theory

    Example:
        >>> # Number of events per time period
        >>> counts = Poisson(lambda_=event_rate)
        >>> # Observed count data
        >>> y_counts = Poisson(lambda_=fitted_rate).as_observable()
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
    """Standard multinomial distribution parameter.

    Implements the multinomial distribution with probability vector theta
    and number of trials N. Models the number of observations in each
    category for a fixed number of trials.

    :param theta: Probability vector (must sum to 1)
    :type theta: custom_types.ContinuousParameterType
    :param N: Number of trials
    :type N: custom_types.DiscreteParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        X ~ Multinomial(N, θ) where P(X = x) = (N! / Πᵢ xᵢ!) * Πᵢ θᵢ^xᵢ

    Properties:
    - Support: {x : Σxᵢ = N, xᵢ ≥ 0}
    - Mean: E[Xᵢ] = N * θᵢ
    - Variance: Var[Xᵢ] = N * θᵢ * (1 - θᵢ)
    - Covariance: Cov[Xᵢ, Xⱼ] = -N * θᵢ * θⱼ

    Example:
        >>> # Categorical data with known totals
        >>> counts = Multinomial(theta=category_probs, N=total_trials)
    """

    SIMPLEX_PARAMS = {"theta"}
    STAN_DIST = "multinomial"
    SCIPY_DIST = custom_scipy_dists.multinomial
    TORCH_DIST = custom_torch_dists.Multinomial
    STAN_TO_SCIPY_NAMES = {"theta": "p", "N": "n"}
    STAN_TO_TORCH_NAMES = {"theta": "probs", "N": "total_count"}


class MultinomialLogit(_MultinomialBase):
    """Multinomial distribution with logit parameterization.

    Implements the multinomial distribution parameterized by logits (gamma)
    rather than probabilities.

    :param gamma: Logit vector (unconstrained real values)
    :type gamma: custom_types.ContinuousParameterType
    :param N: Number of trials
    :type N: custom_types.DiscreteParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        θᵢ = exp(γᵢ) / Σⱼ exp(γⱼ) (softmax transformation)
        X ~ Multinomial(N, θ)

    Properties:
    - Unconstrained parameterization (γ ∈ ℝᴷ)
    - Natural for logistic regression extensions
    - Automatic softmax transformation applied

    Example:
        >>> # Logit-parameterized multinomial
        >>> counts = MultinomialLogit(gamma=logit_probs, N=total_trials)
    """

    STAN_DIST = "multinomial_logit"
    SCIPY_DIST = custom_scipy_dists.multinomial_logit
    TORCH_DIST = custom_torch_dists.MultinomialLogit
    STAN_TO_SCIPY_NAMES = {"gamma": "logits", "N": "n"}
    STAN_TO_TORCH_NAMES = {"gamma": "logits", "N": "total_count"}


class MultinomialLogTheta(_MultinomialBase):
    """Multinomial distribution with log-probability parameterization.

    Implements the multinomial distribution in terms of the log of the theta
    parameter. This parameterization is useful for models that naturally
    work with log-probabilities. It can use the ExpDirichlet distribution as a
    prior to enforce the log-simplex constraint and keep computations completely
    in the log-probability space.

    :param log_theta: Log probability vector (log-simplex constraint)
    :type log_theta: custom_types.ContinuousParameterType
    :param N: Number of trials
    :type N: custom_types.DiscreteParameterType
    :param kwargs: Additional keyword arguments passed to parent class

    Mathematical Definition:
        θᵢ = exp(log_θᵢ) (with normalization: Σⱼ exp(log_θⱼ) = 1)
        X ~ Multinomial(N, θ)

    Properties:

    - Log-simplex parameterization
    - Numerically stable for small probabilities
    - Optional multinomial coefficient pre-computation

    Special Features:

    - Automatic multinomial coefficient calculation when used as observable. This
      results in improved computational efficiency by eliminating redundant calculations.
    - Coefficient removed when parameter has other children

    This distribution requires custom Stan functions for implementation (see
    `multinomial.stanfunctions` in the `stan` submodule) which are automatically
    included in any Stan program defined using this distribution.

    Example:
        >>> # Log-probability parameterized multinomial
        >>> counts = MultinomialLogTheta(log_theta=log_probs, N=total_trials)
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
