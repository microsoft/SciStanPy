"""Holds the abstract classes for the core components of a DMS Stan model."""

from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Union

import numpy as np
import numpy.typing as npt

import dms_stan as dms
import dms_stan.model.components as dms_components


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
        n_indices = len(self.shape)

        # If there are no indices, then we just return the variable name
        if n_indices == 0:
            return self.model_varname, False

        # Singleton dimensions get a "1" index. All others get the index options.
        indices = [
            "1" if dimsize == 1 else index_opts[i]
            for i, dimsize in enumerate(self.shape)
        ]

        # Vector if at least one dimension is not 1
        is_vector = any(dimsize != 1 for dimsize in self.shape)

        # Build the indexed variable name
        indexed_varname = f"{self.model_varname}[{','.join(indices)}]"

        return indexed_varname, is_vector

    @property
    def model_varname(self) -> str:
        """Return the DMS Stan variable name for this parameter"""
        if self._model_varname == "":
            print(type(self))
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

    @property
    def stan_code_level(self) -> int:
        """The level at which the code is written. 0 is the highest level."""
        # Strip off leading 1s. The level is the remaining dimensions.
        level = 0
        for dimsize in self.shape:
            if dimsize == 1:
                level += 1
            else:
                break

        return level


class AbstractPassthrough(AbstractModelComponent):
    """
    Abstract class for components that pass through the values of their children.
    """

    def __init__(self, value: Union[int, float, npt.NDArray]):
        """
        Wraps the value in a Constant instance. Any numerical type is legal.
        """
        # Initialize the parent class
        super().__init__()

        # Assign the value
        self.value = np.array(value)

    def __getattr__(self, name):
        return getattr(self.value, name)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value.__repr__()})"

    # Handling items in `value`
    def __getitem__(self, key):
        return self.value[key]

    def __setitem__(self, key, value):
        self.value[key] = value

    def __delitem__(self, key):
        del self.value[key]

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

    # Comparison operations are forwarded to the value
    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return self.value != other

    def __lt__(self, other):
        return self.value < other

    def __le__(self, other):
        return self.value <= other

    def __rt__(self, other):
        return self.value > other

    def __ge__(self, other):
        return self.value >= other

    def __and__(self, other):
        return self.value & other

    def __rand__(self, other):
        return other & self.value

    def __or__(self, other):
        return self.value | other

    def __ror__(self, other):
        return other | self.value

    def __contains__(self, item):
        return item in self.value

    # Unary operations are forwarded to the value
    def __neg__(self):
        return -self.value

    def __pos__(self):
        return +self.value

    def __invert__(self):
        return ~self.value

    # Other mathematical operations are forwarded to the value
    def __abs__(self):
        return abs(self.value)

    def __round__(self, n=None):
        return round(self.value, n)

    # Iterators
    def __iter__(self):
        return iter(self.value)

    def __len__(self):
        return len(self.value)

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
    _torch_container_class: type["dms.model.components.pytorch.TorchContainer"]

    # Define the stan data type
    base_stan_dtype: Literal["real", "int", "simplex"] = "real"
    stan_lower_bound: Optional[float | int] = None
    stan_upper_bound: Optional[float | int] = None

    def __init__(  # pylint: disable=unused-argument
        self,
        *,
        shape: Optional[tuple[int, ...]] = None,
        **parameters: "dms_components.custom_types.CombinableParameterType",
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
        self.parameters = {}
        for name, val in parameters.items():
            if isinstance(val, AbstractParameter):
                self.parameters[name] = val.record_child(self)
            elif isinstance(val, (int, float, np.ndarray)):
                self.parameters[name] = dms_components.Hyperparameter(val)
                self.parameters[name].child = self
                self.parameters[name].name = name
            elif isinstance(val, dms_components.Constant):
                self.parameters[name] = val
            else:
                raise TypeError(
                    f"Unexpected type passed for {name}: {type(val)}. Should be "
                    "an int, float, np.ndarray, Hyperparameter, Constant, or child "
                    "of AbstractParameter"
                )

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
        self._torch_container: Optional[
            "dms.model.components.pytorch.TorchContainer"
        ] = None

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

    def get_parents(
        self,
    ) -> list["dms_components.custom_types.CombinableParameterType"]:
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
    def torch_container(self) -> "dms.model.components.pytorch.TorchContainer":
        """Return the Pytorch container for this parameter. Error if not initialized."""
        if self._torch_container is None:
            raise ValueError("Pytorch container not initialized. Run `init_pytorch`.")
        return self._torch_container

    @property
    def is_root_node(self) -> bool:
        """Return whether or not the parameter is a root node."""
        return all(
            isinstance(param, dms_components.Hyperparameter)
            for param in self.parameters.values()
        )

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

    @property
    def is_named(self) -> bool:
        """Return whether or not the parameter has a name."""
        return self._model_varname != ""

    @property
    def hyperparameters(self):
        """Return the hyperparameters of the parameter."""
        return {
            name: param
            for name, param in self.parameters.items()
            if isinstance(param, dms_components.Hyperparameter)
        }
