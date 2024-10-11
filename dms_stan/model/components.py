"""Holds classes that can be used for defining models in DMS Stan models."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
import scipy.special as sp
import torch.distributions as dist

import dms_stan as dms


# pylint: disable=too-many-lines
class AbstractModelComponent(ABC):
    """Base class for all core components of a DMS Stan model."""

    def __init__(self):

        # Set up a placeholder for the DMS Stan Model variable name
        self._model_varname: str = ""

    def get_indexed_varname(self, index_opts: tuple[str, ...]) -> tuple[str, bool]:
        """
        Returns the variable name used by stan with the appropriate indices. Note
        that DMS Stan always assumes that computations on the last dimension can
        be vectorized.

        Args:
            index_opts (tuple[str, ...]): The index options for the variable name.

        Returns:
            str: The variable name with the appropriate indices (if any).
            bool: Whether the variable is a vector.
        """
        # Get the number of indices needed
        n_indices = len(self.shape) - 1

        # If there are no indices, then we just return the variable name
        if n_indices <= 0:
            return self.model_varname, False

        # Vector if the last dimension is not singleton
        is_vector = self.shape[-1] != 1

        # Build the indexed variable name
        indexed_varname = f"{self.model_varname}[{','.join(index_opts[:n_indices])}]"

        return indexed_varname, is_vector

    @property
    def model_varname(self) -> str:
        """Return the DMS Stan variable name for this parameter"""
        if self._model_varname == "":
            raise ValueError(
                "DMS Stan variable name not set. This is only set when the parameter"
                "is used in a DMS Stan model."
            )
        return self._model_varname

    @model_varname.setter
    def model_varname(self, name: str) -> None:
        """Set the DMS Stan variable name for this parameter"""
        self._model_varname = name

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the parameter"""

    @property
    @abstractmethod
    def stan_dtype(self) -> str:
        """Return the Stan data type for this parameter"""

    @property
    @abstractmethod
    def stan_parameter_declaration(self) -> str:
        """Declare the parameter in Stan"""


class Constant(AbstractModelComponent):
    """
    This class is used to wrap values that are intended to stay constant in the
    Stan model. This is effectively a wrapper around the value that forwards all
    mathematical operations and attribute access to the value. Note that because
    it is a wrapper, in place operations made to the value will be reflected in
    the instance of this class.
    """

    def __init__(self, value: Union[int, float, npt.NDArray]):
        """
        Wraps the value in a Constant instance. Any numerical type is legal.
        """
        # Initialize the parent class
        super().__init__()

        # Assign the value
        self.value: npt.NDArray = np.array(value)

    def __getattr__(self, name):
        return getattr(self.value, name)

    def __repr__(self):
        return f"Constant({self.value.__repr__()})"

    def __getitem__(self, key):
        return self.value[key]

    # Mathematical operations are forwarded to the value
    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return other + self.value

    def __sub__(self, other):
        return self.value - other

    def __rsub__(self, other):
        return other - self.value

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    def __truediv__(self, other):
        return self.value / other

    def __rtruediv__(self, other):
        return other / self.value

    def __floordiv__(self, other):
        return self.value // other

    def __rflooriv__(self, other):
        return other // self.value

    def __mod__(self, other):
        return self.value % other

    def __rmod__(self, other):
        return other % self.value

    def __pow__(self, other):
        return self.value**other

    def __rpow__(self, other):
        return other**self.value

    def __matmul__(self, other):
        return self.value @ other

    def __rmatmul__(self, other):
        return other @ self.value

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Returns the shape of the value.
        """
        return self.value.shape

    @property
    def stan_dtype(self) -> str:
        """
        Returns the Stan data type of the value.
        """
        return "real" if isinstance(self.value.dtype, np.floating) else "int"

    @property
    def stan_parameter_declaration(self) -> str:
        """Declare the parameter in Stan"""
        # Get the base declaration
        declaration = f"{self.stan_dtype} {self.model_varname}"

        # If a scalar, the declaration needs no modification
        if self.value.ndim == 0:
            return declaration

        # Otherwise, it is part of an array
        else:
            str_shape = [str(s) for s in self.value.shape]
            return f"array[{','.join(str_shape)}] {declaration}"


class AbstractParameter(AbstractModelComponent):
    """Template class for parameters used in DMS Stan models."""

    # Define allowed ranges for the parameters
    POSITIVE_PARAMS: set[str] = set()
    NEGATIVE_PARAMS: set[str] = set()
    SIMPLEX_PARAMS: set[str] = set()

    # Define the class that will be used for compiling to PyTorch
    _torch_container_class: type[dms.pytorch.TorchContainer]

    # Define the stan data type
    base_stan_dtype: Literal["real", "int", "simplex"] = "real"
    stan_lower_bound: Optional[float | int] = None
    stan_upper_bound: Optional[float | int] = None

    def __init__(  # pylint: disable=unused-argument
        self,
        *,
        shape: Optional[tuple[int, ...]] = None,
        **parameters: "CombinableParameterType",
    ):
        """Builds a parameter instance with the given shape."""
        # Initialize the parent class
        super().__init__()

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

        # Populate the parameters and record this distribution as a child
        self.parameters = {
            name: (
                val.record_child(self)
                if isinstance(val, AbstractParameter)
                else np.array(val)
            )
            for name, val in parameters.items()
        }

        # All bounded parameters must be named in the parameter dictionary
        if missing_names := (
            self.POSITIVE_PARAMS | self.NEGATIVE_PARAMS | self.SIMPLEX_PARAMS
        ) - set(self.parameters.keys()):
            raise ValueError(
                f"{', '.join(missing_names)} are bounded parameters but are missing "
                "from those defined"
            )

        # Check parameter ranges
        self._check_parameter_ranges()

        # If the shape is not provided, then we use the broadcasted shape of the
        # parameters
        param_broadcast_shape = np.broadcast_shapes(
            *[param.shape for param in self.parameters.values()]
        )
        self._shape = param_broadcast_shape if shape is None else shape

        # The shape must be broadcastable to the shapes of the parameters.
        try:
            self.draw_shape = np.broadcast_shapes(shape, param_broadcast_shape)
        except ValueError as error:
            raise ValueError("Shape is not broadcastable to parent shapes") from error

        # We need a list for children
        self.children = []

        # Set up a placeholder for the Pytorch container
        self._torch_container: Optional[dms.pytorch.TorchContainer] = None

    def _check_parameter_ranges(
        self, draws: Optional[dict[str, npt.NDArray]] = None
    ) -> None:
        """Makes sure that the parameters are within the allowed ranges."""
        # The positive and negative sets must be disjoint
        if self.POSITIVE_PARAMS & self.NEGATIVE_PARAMS:
            raise ValueError("Positive and negative parameter sets must be disjoint")

        # Convert the list of simplex parameters to a set. Anything that is a simplex
        # is also positive
        positive_set = self.SIMPLEX_PARAMS | self.POSITIVE_PARAMS

        # If draws are not provided, use parameters
        checkdict = self.parameters if draws is None else draws

        # Check the parameter ranges
        for paramname, paramval in checkdict.items():

            # Skip distributions
            if isinstance(paramval, AbstractParameter):
                continue

            # Check ranges
            if paramname in positive_set and np.any(paramval <= 0):
                raise ValueError(f"{paramname} must be positive")
            elif paramname in self.NEGATIVE_PARAMS and np.any(paramval >= 0):
                raise ValueError(f"{paramname} must be negative")
            elif paramname in self.SIMPLEX_PARAMS:
                if not isinstance(paramval, np.ndarray):
                    raise ValueError(f"{paramname} must be a numpy array")
                if not np.allclose(np.sum(paramval, axis=-1), 1):
                    raise ValueError(f"{paramname} must sum to 1 over the last axis")

    def init_pytorch(self) -> None:
        """
        Sets up the parameters needed for training a Pytorch model and defines the
        Pytorch operation that will be performed on the parameter. Operations can
        be either calculation of loss or transformation of the parameter, depending
        on the subclass.
        """
        # We must have defined the PyTorch container class
        if not hasattr(self, "_torch_container_class"):
            raise ValueError("Pytorch container class not defined in the subclass")

        # Initialize the Pytorch container
        self._torch_container = self._torch_container_class(self)

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

        # Check the parameter ranges
        self._check_parameter_ranges(finalized_draws)

        return finalized_draws

    def record_child(self, child: "AbstractParameter") -> "AbstractParameter":
        """
        Records a child parameter of the current parameter. This is used to keep
        track of the lineage of the parameter.

        Args:
            child (AbstractParameter): The child parameter to record.

        Returns:
            AbstractParameter: Self.
        """

        # If the child is already in the list of children, then we don't need to
        # add it again
        assert child not in self.children, "Child already recorded"

        # Record the child
        self.children.append(child)

        return self

    def get_parents(self) -> list["CombinableParameterType"]:
        """
        Gathers the parent parameters of the current parameter.

        Returns:
            list[AbstractParameter]: Parent parameters of the current parameter.
        """
        return list(self.parameters.values())

    def get_children(self) -> list["AbstractParameter"]:
        """
        Gathers the children parameters of the current parameter.

        Returns:
            list[AbstractParameter]: Children parameters of the current parameter.
        """
        return self.children

    def recurse_parents(
        self, _current_depth: int = 0
    ) -> list[tuple[int, "AbstractParameter", "AbstractParameter"]]:
        """
        Recursively calls `get_parents` on the current parameter to get the entire
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

    def recurse_children(
        self, _current_depth: int = 0
    ) -> list[tuple[int, "AbstractParameter", "AbstractParameter"]]:
        """
        Recursively calls `get_children` on the current parameter to get the entire
        lineage of the parameter.

        Returns:
            list[tuple[int, AbstractParameter, AbstractParameter]]: A list of tuples
                containing the depth of the parameter in the lineage, the child
                parameter, and the current parameter, in that order.
        """
        # Get the children of the current parameter
        children = self.get_children()

        # Call `recurse_children` on each child
        to_return = []
        for child in children:

            # Must be a parameter
            assert isinstance(child, AbstractParameter)

            # Add the child to the list of tuples that will be returned
            to_return.append((_current_depth, child, self))

            # Get the child's lineage and add it to the list
            to_return.extend(child.recurse_children(_current_depth + 1))

        return to_return

    def __str__(self):
        return f"{self.__class__.__name__}"

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the parameter"""
        return self._shape

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

    @property
    def torch_container(self) -> dms.pytorch.TorchContainer:
        """Return the Pytorch container for this parameter. Error if not initialized."""
        if self._torch_container is None:
            raise ValueError("Pytorch container not initialized. Run `init_pytorch`.")
        return self._torch_container

    @property
    def stan_bounds(self) -> str:
        """Return the Stan bounds for this parameter"""
        # Format the lower and upper bounds
        lower = (
            "" if self.stan_lower_bound is None else f"lower={self.stan_lower_bound}"
        )
        upper = (
            "" if self.stan_upper_bound is None else f"upper={self.stan_upper_bound}"
        )

        # Combine the bounds
        if lower and upper:
            bounds = f"{lower}, {upper}"
        elif lower:
            bounds = lower
        elif upper:
            bounds = upper
        else:
            return ""

        # Return the bounds
        return f"<{bounds}>"

    @property
    def stan_dtype(self) -> str:
        """Return the Stan data type for this parameter"""

        # Get the base datatype
        dtype = self.__class__.base_stan_dtype

        # Base data type for 0-dimensional parameters. If the parameter is 0-dimensional,
        # then we can only have real or int as the data type.
        if self.ndim == 0:
            assert dtype in {"real", "int"}
            return dtype

        # Convert shape to strings
        string_shape = [str(dim) for dim in self.shape]

        # Handle different data types for different dimensions
        if dtype == "real":  # Becomes vector or array of vectors
            dtype = f"vector[{string_shape[-1]}]"
            if self.ndim > 1:
                dtype = f"array[{','.join(string_shape[:-1])}] {dtype}"

        elif dtype == "int":  # Becomes array
            dtype = f"array[{','.join(string_shape)}] int"

        elif dtype == "simplex":  # Becomes array of simplexes
            dtype = f"array[{','.join(string_shape[:-1])}] simplex[{string_shape[-1]}]"

        else:
            raise AssertionError(f"Unknown data type {dtype}")

        return dtype

    @property
    def stan_parameter_declaration(self) -> str:
        """Returns the Stan parameter declaration for this parameter."""
        return f"{self.stan_dtype}{self.stan_bounds} {self.model_varname}"


# TODO: Elementwise operations if both components are non-scalar. Otherwise default.
class TransformedParameter(AbstractParameter):
    """
    Base class representing a parameter that is the result of combining other
    parameters using mathematical operations.
    """

    _torch_container_class: dms.pytorch.TransformedContainer

    def draw(self, n: int) -> npt.NDArray:
        """Sample from this parameter's distribution `n` times."""

        # Perform the operation on the draws
        return self.operation(**super().draw(n))

    @abstractmethod
    def operation(self, **draws: "SampleType") -> npt.NDArray:
        """Perform the operation on the draws"""

    def get_stan_transformation(self, index_opts: tuple[str, ...]) -> tuple[str, bool]:
        """
        Return the Stan transformation for this parameter. This recursively calls
        the equivalent method on parent transformed parameters until we hit a
        non-transformed parameter.
        """
        # Recursively gather the transformations until we hit a non-transformed
        # parameter or a recorded variable
        to_format: dict[str, tuple[str, bool]] = {}
        for name, param in self.parameters.items():

            # If the parameter is non-transformed or named in a varaible, record
            # the variable name
            if (
                isinstance(param, (Constant, Parameter))
                or param._component_varname != ""  # pylint: disable=protected-access
            ):
                to_format[name] = param.get_indexed_varname(index_opts)

            # Otherwise, get the transformation of the parent
            elif isinstance(param, TransformedParameter):
                transformation, is_vector = param.get_stan_transformation(index_opts)
                to_format[name] = (f"( {transformation} )", is_vector)

            # Otherwise, raise an error
            else:
                raise TypeError(f"Unknown model component type {type(param)}")

        # We are a vector if any formatted variables are vectors
        is_vector = any(is_vector for _, is_vector in to_format.values())

        # Format the transformation
        return self.format_stan_transformation(**to_format), is_vector

    @abstractmethod
    def format_stan_transformation(self, **param_vals: tuple[str, bool]) -> str:
        """Return the base Stan transformation for this parameter."""

    # Calling this class should return the result of the operation.
    def __call__(self, *args, **kwargs):
        return self.operation(*args, **kwargs)


class BinaryTransformedParameter(TransformedParameter):
    """
    Identical to the TransformedParameter class, but only for operations involving
    two parameters. In other words, two parameters must be passed to the class.
    """

    _torch_container_class: dms.pytorch.BinaryTransformedContainer

    def __init__(
        self,
        dist1: "CombinableParameterType",
        dist2: "CombinableParameterType",
        shape: Optional[tuple[int, ...]] = None,
    ):
        super().__init__(dist1=dist1, dist2=dist2, shape=shape)

    @abstractmethod
    def operation(  # pylint: disable=arguments-differ
        self,
        dist1: "SampleType",
        dist2: "SampleType",
    ): ...

    @abstractmethod
    def format_stan_transformation(  # pylint: disable=arguments-differ
        self, dist1: tuple[str, bool], dist2: tuple[str, bool]
    ) -> tuple[str, str, bool]:

        # Unpack the variable names and scalar flags
        dist1_name, dist1_is_vector = dist1
        dist2_name, dist2_is_vector = dist2

        # Return the names and whether or not this will be an elementwise operation
        return dist1_name, dist2_name, dist1_is_vector and dist2_is_vector


class UnaryTransformedParameter(TransformedParameter):
    """Transformed parameter that only requires one parameter."""

    _torch_container_class: dms.pytorch.UnaryTransformedContainer

    def __init__(
        self,
        dist1: "CombinableParameterType",
        shape: Optional[tuple[int, ...]] = None,
        **kwargs: Any,
    ):
        super().__init__(dist1=dist1, shape=shape)

        # Store the kwargs for the operation
        self.operation_kwargs = kwargs

    @abstractmethod
    def operation(  # pylint: disable=arguments-differ
        self, dist1: "SampleType"
    ) -> npt.NDArray: ...

    @abstractmethod
    def format_stan_transformation(  # pylint: disable=arguments-differ
        self, dist1: tuple[str, bool]
    ) -> tuple[str, bool]: ...


class AddParameter(BinaryTransformedParameter):
    """Defines a parameter that is the sum of two other parameters."""

    _torch_container_class = dms.pytorch.AddTransformedContainer

    def operation(
        self,
        dist1: "SampleType",
        dist2: "SampleType",
    ) -> npt.NDArray:
        return dist1 + dist2

    def format_stan_transformation(
        self,
        dist1: tuple[str, bool],
        dist2: tuple[str, bool],
    ) -> str:
        # Unpack the variable names
        dist1_name, dist2_name, _ = super().format_stan_transformation(dist1, dist2)

        return f"{dist1_name} + {dist2_name}"


class SubtractParameter(BinaryTransformedParameter):
    """Defines a parameter that is the difference of two other parameters."""

    _torch_container_class = dms.pytorch.SubtractTransformedContainer

    def operation(
        self,
        dist1: "SampleType",
        dist2: "SampleType",
    ) -> npt.NDArray:
        return dist1 - dist2

    def format_stan_transformation(
        self,
        dist1: tuple[str, bool],
        dist2: tuple[str, bool],
    ) -> str:
        # Unpack the variable names
        dist1_name, dist2_name, _ = super().format_stan_transformation(dist1, dist2)

        return f"{dist1_name} - {dist2_name}"


class MultiplyParameter(BinaryTransformedParameter):
    """Defines a parameter that is the product of two other parameters."""

    _torch_container_class = dms.pytorch.MultiplyTransformedContainer

    def operation(
        self,
        dist1: "SampleType",
        dist2: "SampleType",
    ) -> npt.NDArray:
        return dist1 * dist2

    def format_stan_transformation(
        self,
        dist1: tuple[str, bool],
        dist2: tuple[str, bool],
    ) -> str:
        # Unpack the variable names and determine if this is an elementwise operation
        dist1_name, dist2_name, elementwise = super().format_stan_transformation(
            dist1, dist2
        )

        # Get the operator
        operator = ".*" if elementwise else "*"

        return f"{dist1_name} {operator} {dist2_name}"


class DivideParameter(BinaryTransformedParameter):
    """Defines a parameter that is the quotient of two other parameters."""

    _torch_container_class = dms.pytorch.DivideTransformedContainer

    def operation(
        self,
        dist1: "SampleType",
        dist2: "SampleType",
    ) -> npt.NDArray:
        return dist1 / dist2

    def format_stan_transformation(
        self,
        dist1: tuple[str, bool],
        dist2: tuple[str, bool],
    ) -> str:
        # Unpack the variable names and determine if this is an elementwise operation
        dist1_name, dist2_name, elementwise = super().format_stan_transformation(
            dist1, dist2
        )

        # Get the operator
        operator = "./" if elementwise else "/"

        return f"{dist1_name} {operator} {dist2_name}"


class PowerParameter(BinaryTransformedParameter):
    """Defines a parameter raised to the power of another parameter."""

    _torch_container_class = dms.pytorch.PowerTransformedContainer

    def operation(
        self,
        dist1: "SampleType",
        dist2: "SampleType",
    ) -> npt.NDArray:
        return dist1**dist2

    def format_stan_transformation(
        self,
        dist1: tuple[str, bool],
        dist2: tuple[str, bool],
    ) -> str:
        # Unpack the variable names and determine if this is an elementwise operation
        dist1_name, dist2_name, elementwise = super().format_stan_transformation(
            dist1, dist2
        )

        # Get the operator
        operator = ".^" if elementwise else "^"

        return f"{dist1_name} {operator} {dist2_name}"


class NegateParameter(UnaryTransformedParameter):
    """Defines a parameter that is the negative of another parameter."""

    _torch_container_class = dms.pytorch.NegateTransformedContainer

    def operation(self, dist1: "SampleType") -> npt.NDArray:
        return -dist1

    def format_stan_transformation(self, dist1: tuple[str, bool]) -> str:
        return f"-{dist1[0]}"


class AbsParameter(UnaryTransformedParameter):
    """Defines a parameter that is the absolute value of another."""

    _torch_container_class = dms.pytorch.AbsTransformedContainer
    stan_lower_bound: float = 0.0

    def operation(self, dist1: "SampleType") -> npt.NDArray:
        return np.abs(dist1, **self.operation_kwargs)

    def format_stan_transformation(self, dist1: tuple[str, bool]) -> str:
        return f"abs({dist1[0]})"


class LogParameter(UnaryTransformedParameter):
    """Defines a parameter that is the natural logarithm of another."""

    # The distribution must be positive
    POSITIVE_PARAMS = set(["dist1"])

    _torch_container_class = dms.pytorch.LogTransformedContainer
    stan_lower_bound: float = 0.0

    def operation(self, dist1: "SampleType") -> npt.NDArray:
        return np.log(dist1, **self.operation_kwargs)

    def format_stan_transformation(self, dist1: tuple[str, bool]) -> str:
        return f"log({dist1[0]})"


class ExpParameter(UnaryTransformedParameter):
    """Defines a parameter that is the exponential of another."""

    _torch_container_class = dms.pytorch.ExpTransformedContainer
    stan_lower_bound: float = 0.0

    def operation(self, dist1: "SampleType") -> npt.NDArray:
        return np.exp(dist1, **self.operation_kwargs)

    def format_stan_transformation(self, dist1: tuple[str, bool]) -> str:
        return f"exp({dist1[0]})"


class NormalizeParameter(UnaryTransformedParameter):
    """Defines a parameter that is normalized to sum to 1."""

    _torch_container_class = dms.pytorch.NormalizeTransformedContainer
    stan_lower_bound: float = 0.0
    stan_upper_bound: float = 1.0

    def operation(self, dist1: "SampleType") -> npt.NDArray:
        return dist1 / np.sum(dist1, keepdims=True, **self.operation_kwargs)

    def format_stan_transformation(self, dist1: tuple[str, bool]) -> str:
        return f"{dist1[0]} / sum({dist1[0]})"


class NormalizeLogParameter(UnaryTransformedParameter):
    """
    Defines a parameter that is normalized such that exp(x) sums to 1. By extension,
    this assumes that the input is log-transformed.
    """

    _torch_container_class = dms.pytorch.NormalizeLogTransformedContainer
    stan_upper_bound: float = 0.0

    def operation(self, dist1: "SampleType") -> npt.NDArray:
        return dist1 - sp.logsumexp(dist1, keepdims=True, **self.operation_kwargs)

    def format_stan_transformation(self, dist1: tuple[str, bool]) -> str:
        return f"{dist1[0]} - log_sum_exp({dist1[0]})"


class Growth(TransformedParameter):
    """Base class for growth models."""

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        t: "CombinableParameterType",
        shape: Optional[tuple[int, ...]] = None,
        **params: "CombinableParameterType",
    ):
        # Store all parameters as a list by calling the super class
        super().__init__(t=t, shape=shape, **params)


class LogExponentialGrowth(Growth):
    """
    A distribution that models the natural log of the `ExponentialGrowth` distribution.
    Specifically, parameters `t`, `log_A`, and `r` are used to calculate the log
    of the exponential growth model as follows:

    $$
    log(x) = log_A + rt
    $$

    Note that, with this parametrization, we guarantee that $x > 0$. It is also
    only defined for $A > 0$ and $r > 0$, assuming that the time parameter $t$ is
    always positive.

    This parametrization is particularly useful for modeling the proportions of
    different populations as is done in DMS Stan, as proportions are always positive.
    """

    _torch_container_class = dms.pytorch.LogExponentialGrowthContainer

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        t: "CombinableParameterType",
        log_A: "CombinableParameterType",
        r: "CombinableParameterType",
        shape: Optional[tuple[int, ...]] = None,
    ):
        """Initializes the LogExponentialGrowth distribution.

        Args:
            t ("SampleType"): The time parameter.

            log_A ("SampleType"): The log of the amplitude parameter.

            r ("SampleType"): The rate parameter.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super().__init__(t=t, log_A=log_A, r=r, shape=shape)

    def operation(  # pylint: disable=arguments-differ
        self,
        *,
        t: "SampleType",
        log_A: "SampleType",
        r: "SampleType",
    ) -> npt.NDArray:
        return log_A + r * t

    def format_stan_transformation(  # pylint: disable=arguments-differ
        self, t: tuple[str, bool], log_A: [str, bool], r: [str, bool]
    ) -> str:
        # Decide on operator between r and t
        operator = ".*" if t[1] and r[1] else "*"

        # Build the transformation
        return f"{log_A[0]} + {r[0]} {operator} {t[0]}"


class LogSigmoidGrowth(Growth):
    r"""
    A distribution that models the natural log of the `SigmoidGrowth` distribution.
    Specifically, parameters `t`, `log_A`, `r`, and `c` are used to calculate the
    log of the sigmoid growth model as follows:

    $$
    log(x) = log_A - log(1 + \textrm{e}^{-r(t - c)})
    $$

    As with the `LogExponentialGrowth` distribution, this parametrization guarantees
    that $x > 0$.
    """

    _torch_container_class = dms.pytorch.LogSigmoidGrowthContainer

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        t: "CombinableParameterType",
        log_A: "CombinableParameterType",
        r: "CombinableParameterType",
        c: "CombinableParameterType",
        shape: Optional[tuple[int, ...]] = None,
    ):
        """Initializes the LogSigmoidGrowth distribution.

        Args:
            t ("SampleType"): The time parameter.

            log_A ("SampleType"): The log of the amplitude parameter.

            r ("SampleType"): The rate parameter.

            c ("SampleType"): The offset parameter.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super().__init__(t=t, log_A=log_A, r=r, c=c, shape=shape)

    def operation(  # pylint: disable=arguments-differ
        self,
        *,
        t: "SampleType",
        log_A: "SampleType",
        r: "SampleType",
        c: "SampleType",
    ) -> npt.NDArray:
        return log_A - np.log(1 + np.exp(-r * (t - c)))

    def format_stan_transformation(  # pylint: disable=arguments-differ
        self, t: str, log_A: str, r: str, c: str
    ) -> str:
        # Determine the operator between r and t
        operator = ".*" if r[1] and t[1] else "*"

        return f"{log_A} - log(1 + exp(-{r} {operator} ({t} - {c})))"


class Parameter(AbstractParameter):
    """Base class for parameters used in DMS Stan"""

    _torch_container_class = dms.pytorch.ParameterContainer

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

    def get_stan_distribution(self, index_opts: tuple[str, ...]) -> str:
        """Return the Stan distribution for this parameter"""

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

    base_stan_dtype: str = "int"
    stan_lower_bound: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observable = True


class Normal(ContinuousDistribution):
    """Parameter that is represented by the normal distribution."""

    POSITIVE_PARAMS = set(["sigma"])

    def __init__(
        self,
        *,
        mu: "ContinuousParameterType",
        sigma: "ContinuousParameterType",
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


class HalfNormal(Normal):
    """Parameter that is represented by the half-normal distribution."""

    stan_lower_bound: float = 0.0

    def __init__(self, *, sigma: "ContinuousParameterType", **kwargs):
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

    POSITIVE_PARAMS = set(["sigma"])
    stan_lower_bound: float = 0.0

    def __init__(
        self,
        mu: "ContinuousParameterType",
        sigma: "ContinuousParameterType",
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


class Beta(ContinuousDistribution):
    """Defines the beta distribution."""

    POSITIVE_PARAMS = set(["alpha", "beta"])
    stan_lower_bound: float = 0.0
    stan_upper_bound: float = 1.0

    def __init__(
        self,
        *,
        alpha: "ContinuousParameterType",
        beta: "ContinuousParameterType",
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


class Gamma(ContinuousDistribution):
    """Defines the gamma distribution."""

    POSITIVE_PARAMS = set(["alpha", "beta"])

    stan_lower_bound: float = 0.0

    def __init__(
        self,
        *,
        alpha: "ContinuousParameterType",
        beta: "ContinuousParameterType",
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


class Exponential(ContinuousDistribution):
    """Defines the exponential distribution."""

    POSITIVE_PARAMS = set(["beta"])

    # Overwrite the stan data type
    stan_lower_bound: float = 0.0

    def __init__(self, *, beta: "ContinuousParameterType", **kwargs):

        super().__init__(
            numpy_dist="exponential",
            torch_dist=dist.exponential.Exponential,
            stan_to_np_names={"beta": "scale"},
            stan_to_torch_names={"beta": "rate"},
            stan_to_np_transforms={"beta": lambda x: 1 / x},
            beta=beta,
            **kwargs,
        )


class Dirichlet(ContinuousDistribution):
    """Defines the Dirichlet distribution."""

    POSITIVE_PARAMS = set(["alpha"])
    base_stan_dtype: str = "simplex"

    def __init__(self, *, alpha: Union[AbstractParameter, npt.ArrayLike], **kwargs):

        super().__init__(
            numpy_dist="dirichlet",
            torch_dist=dist.dirichlet.Dirichlet,
            stan_to_np_names={"alpha": "alpha"},
            stan_to_torch_names={"alpha": "concentration"},
            alpha=alpha,
            **kwargs,
        )


class Binomial(DiscreteDistribution):
    """Parameter that is represented by the binomial distribution"""

    POSITIVE_PARAMS = set(["theta", "N"])

    def __init__(
        self,
        *,
        theta: "ContinuousParameterType",
        N: "DiscreteParameterType",
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


class Poisson(DiscreteDistribution):
    """Parameter that is represented by the Poisson distribution."""

    POSITIVE_PARAMS = set(["lambda_"])

    def __init__(self, *, lambda_: "ContinuousParameterType", **kwargs):

        super().__init__(
            numpy_dist="poisson",
            torch_dist=dist.poisson.Poisson,
            stan_to_np_names={"lambda_": "lam"},
            stan_to_torch_names={"lambda_": "rate"},
            lambda_=lambda_,
            **kwargs,
        )


class Multinomial(DiscreteDistribution):
    """Defines the multinomial distribution."""

    SIMPLEX_PARAMS = set(["theta"])

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


# Define custom types for this module
SampleType = Union[int, float, npt.NDArray]
ContinuousParameterType = Union[
    ContinuousDistribution,
    TransformedParameter,
    Constant,
    float,
    npt.NDArray[np.floating],
]
DiscreteParameterType = Union[
    DiscreteDistribution,
    TransformedParameter,
    Constant,
    int,
    npt.NDArray[np.integer],
]
CombinableParameterType = Union[ContinuousParameterType, DiscreteParameterType]
