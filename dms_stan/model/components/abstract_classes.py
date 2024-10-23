"""Holds the abstract classes for the core components of a DMS Stan model."""

from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Union

import numpy as np
import numpy.typing as npt

import dms_stan as dms
import dms_stan.model.components as dms_components


class AbstractModelComponent(ABC):
    """Base class for all core components of a DMS Stan model."""

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
        shape: tuple[int, ...] = (),
        **parameters: "dms_components.custom_types.CombinableParameterType",
    ):
        """Builds a parameter instance with the given shape."""

        # Define placeholder variables
        self._model_varname: str = ""  # DMS Stan Model variable name
        self.observable: bool = False  # Whether the parameter is observable
        self._parents: dict[str, "dms_components.custom_types.CombinableParameterType"]
        self._component_to_paramname: dict["AbstractModelComponent", str]
        self._shape: tuple[int, ...] = shape  # Shape of the parameter
        self._draw_shape: tuple[int, ...]  # Shape of the draws
        self._children: list["AbstractModelComponent"] = []  # Children of the component
        self._torch_container: Optional[  # For PyTorch operations
            "dms.model.components.pytorch.TorchContainer"
        ] = None

        # Validate incoming parameters
        self._validate_parameters(parameters)

        # Set parents
        self._set_parents(parameters)

        # Link parent and child parameters
        for val in self._parents.values():
            val._record_child(self)

        # Set the draw shape
        self._set_draw_shape()

    def _validate_parameters(
        self,
        parameters: dict[str, "dms_components.custom_types.CombinableParameterType"],
    ) -> None:
        """Checks inputs to the __init__ method for validity."""
        # No incoming parameters can be observables
        if any(
            isinstance(param, AbstractModelComponent) and param.observable
            for param in parameters.values()
        ):
            raise ValueError("Parent parameters cannot be observables")

        # All bounded parameters must be named in the parameter dictionary
        if missing_names := (
            self.POSITIVE_PARAMS | self.NEGATIVE_PARAMS | self.SIMPLEX_PARAMS
        ) - set(parameters.keys()):
            raise ValueError(
                f"{', '.join(missing_names)} are bounded parameters but are missing "
                "from those defined"
            )

    def _set_parents(
        self,
        parameters: dict[str, "dms_components.custom_types.CombinableParameterType"],
    ) -> None:
        """Sets the parent parameters of the current parameter."""

        # Convert any non-model components to model components
        self._parents = {
            name: (
                dms_components.Constant(value=val)
                if not isinstance(val, AbstractModelComponent)
                else val
            )
            for name, val in parameters.items()
        }

        # Map components to param names
        self._component_to_paramname = {v: k for k, v in self._parents.items()}
        assert len(self._component_to_paramname) == len(self._parents)

    def _record_child(self, child: "AbstractModelComponent") -> None:
        """
        Records a child parameter of the current parameter. This is used to keep
        track of the lineage of the parameter.

        Args:
            child (AbstractModelComponent): The child parameter to record.
        """

        # If the child is already in the list of children, then we don't need to
        # add it again
        assert child not in self._children, "Child already recorded"

        # Record the child
        self._children.append(child)

    def _set_draw_shape(self) -> None:
        """Sets the shape of the draws for the parameter."""

        # The shape must be broadcastable to the shapes of the parameters.
        try:
            self._draw_shape = np.broadcast_shapes(
                self._shape, *[param.shape for param in self.parents.values()]
            )
        except ValueError as error:
            raise ValueError("Shape is not broadcastable to parent shapes") from error

    def get_child_paramnames(self) -> dict["AbstractModelComponent", str]:
        """
        Gets the names of the parameters that this component defines in its children

        Returns:
            dict[AbstractModelComponent, str]: The child objects mapped to the names
            of their parameters defiend by this object.
        """
        # Set up the dictionary to return
        child_paramnames = {}

        # Process all children
        for child in self._children:

            # Make sure the child is not already recorded
            assert child not in child_paramnames

            # Get the name of the parameter in the child that the bound parameter
            # defines. There should only be one parameter in the child that the
            # bound parameter defines.
            names = [
                paramname
                for paramname, param in child.parameters.items()
                if param is self
            ]
            assert len(names) == 1

            # Record the name
            child_paramnames[child] = names[0]

        return child_paramnames

    def get_indexed_varname(self, index_opts: tuple[str, ...]) -> str:
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

        # Build the indexed variable name
        indexed_varname = f"{self.model_varname}[{','.join(indices)}]"

        return indexed_varname

    def init_pytorch(self) -> None:
        """
        Sets up the parameters needed for training a Pytorch model and defines the
        Pytorch operation that will be performed on the parameter. Operations can
        be either calculation of loss or transformation of the parameter, depending
        on the subclass.
        """
        # Initialize the Pytorch container
        self._torch_container = self._torch_container_class(self)

    def _check_parameter_ranges(self, draws: dict[str, npt.NDArray]) -> None:
        """Makes sure that the parameters are within the allowed ranges."""
        # The positive and negative sets must be disjoint
        if self.POSITIVE_PARAMS & self.NEGATIVE_PARAMS:
            raise ValueError("Positive and negative parameter sets must be disjoint")

        # Convert the list of simplex parameters to a set. Anything that is a simplex
        # is also positive
        positive_set = self.SIMPLEX_PARAMS | self.POSITIVE_PARAMS

        # Check the parameter ranges
        for paramname, paramval in draws.items():
            if paramname in positive_set and np.any(paramval <= 0):
                raise ValueError(f"{paramname} must be positive")
            elif paramname in self.NEGATIVE_PARAMS and np.any(paramval >= 0):
                raise ValueError(f"{paramname} must be negative")
            elif paramname in self.SIMPLEX_PARAMS:
                if not isinstance(paramval, np.ndarray):
                    raise ValueError(f"{paramname} must be a numpy array")
                if not np.allclose(np.sum(paramval, axis=-1), 1):
                    raise ValueError(f"{paramname} must sum to 1 over the last axis")

    @abstractmethod
    def _draw(self, n: int, level_draws: dict[str, npt.NDArray]) -> npt.NDArray:
        """Sample from the distribution that represents the parameter"""

    def draw(
        self,
        n: int,
        _drawn: Optional[dict["AbstractModelComponent", npt.NDArray]] = None,
    ) -> tuple[npt.NDArray, dict["AbstractModelComponent", npt.NDArray]]:
        """
        Recursively draws from the parameter and its parents. The draws for this
        component are the first object returned. The draws from the recursion are
        the second object returned.
        """
        # Build the _drawn dictionary if it is not already built
        if calling_level := (_drawn is None):
            _drawn = {}

        # Loop over the parents and draw from them if we haven't already
        level_draws: dict[str, npt.NDArray] = {}
        for paramname, parent in self._parents.items():

            # If the parent has been drawn, use the already drawn value. Otherwise,
            # draw from the parent.
            if parent in _drawn:
                parent_draw = _drawn[parent]
            else:
                parent_draw, _ = parent.draw(n, _drawn)
                _drawn[parent] = parent_draw

            # Add the parent draw to the level draws. Expand the number of dimensions
            # if necessary to account for the addition of "n" draws.
            dims_to_add = (
                self.ndim + 1
            ) - parent_draw.ndim  # Implicit prepended dimensions
            assert dims_to_add >= 0
            level_draws[paramname] = np.expand_dims(
                parent_draw, axis=tuple(range(1, dims_to_add + 1))
            )

        # Now draw from the current parameter
        draws = self._draw(n, level_draws)

        # Update the _drawn dictionary
        assert self not in _drawn
        _drawn[self] = draws

        # If the calling level, then move the last dimension to the front
        if calling_level:
            draws = np.moveaxis(draws, -1, 0)
            _drawn = {param: np.moveaxis(draw, -1, 0) for param, draw in _drawn.items()}

        return draws, _drawn

    def walk_tree(
        self, walk_down: bool = True
    ) -> list[tuple["AbstractModelComponent", "AbstractModelComponent"]]:
        """
        Walks the tree of parameters, either up or down. "up" means walking from
        the children to the parents, while "down" means walking from the parents
        to the children.

        Args:
            walk_down (bool): Whether to walk down the tree (True) or up the tree (False).

        Returns:
            list[tuple["AbstractModelComponent", "AbstractModelComponent"]]: The
            lineage. Each tuple contains the reference parameter in the first position
            and its relative (child if walking down, parent if walking up) in the
            second.
        """
        # Get the variables to loop over
        relatives = self.children if walk_down else self.parents

        # Recurse
        to_return = []
        for relative in relatives:

            # Add the current parameter and the relative parameter to the list
            to_return.append((self, relative))

            # Recurse on the relative parameter
            to_return.extend(relative.walk_tree(walk_down=walk_down))

        return to_return

    @abstractmethod
    def _handle_transformation_code(
        self, param: "AbstractModelComponent", index_opts: tuple[str, ...]
    ) -> str:
        """Handles code formatting for a transformed parameter."""

    @abstractmethod
    def format_stan_code(self, **to_format: str) -> str:
        """Formats the Stan code for the parameter."""

    def get_stan_code(self, index_opts: tuple[str, ...]) -> str:
        """Gets the Stan code for the parameter."""

        # Recursively gather the transformations until we hit a non-transformed
        # parameter or a recorded variable
        to_format: dict[str, str] = {}
        for name, param in self.parents.items():

            # If the parameter is a constant or another parameter, record
            if isinstance(param, (dms_components.Constant, dms_components.Parameter)):
                to_format[name] = param.get_indexed_varname(index_opts)

            # If the parameter is transformed and not named, the computation is
            # happening in the model. Otherwise, the computation has already happened
            # in the transformed parameters block.
            elif isinstance(param, dms_components.TransformedParameter):
                to_format[name] = self._handle_transformation_code(index_opts)

            # Otherwise, raise an error
            else:
                raise TypeError(f"Unknown model component type {type(param)}")

        # Format the code
        return self.format_stan_code(to_format)

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
    def is_named(self) -> bool:
        """Return whether the parameter has a name assigned by a model"""
        return self._model_varname != ""

    @property
    def torch_container(self) -> "dms.model.components.pytorch.TorchContainer":
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

    @property
    def stan_parameter_declaration(self) -> str:
        """Returns the Stan parameter declaration for this parameter."""
        return f"{self.stan_dtype}{self.stan_bounds} {self.model_varname}"

    @property
    def model_varname(self) -> str:
        """Return the DMS Stan variable name for this parameter"""
        # If the _model_varname variable is set, then we return it
        if self._model_varname != "":
            return self._model_varname

        # Otherwise, we automatically create the name. This is the name of the
        # child components and the name of this component as defined in that child
        # component separated by underscores.
        return "_".join(
            [
                "_".join(child.model_varname, name)
                for child, name in self.get_child_paramnames().items()
            ]
        )

    @model_varname.setter
    def model_varname(self, name: str) -> None:
        """Set the DMS Stan variable name for this parameter"""
        self._model_varname = name

    @property
    def constants(self):
        """Return the constants of the component."""
        return {
            name: component
            for name, component in self._parents.items()
            if isinstance(component, dms_components.Constant)
        }

    @property
    def draw_shape(self) -> tuple[int, ...]:
        """Return the shape of the draws for the parameter."""
        return self._draw_shape

    @property
    def parents(self) -> list["AbstractModelComponent"]:
        """
        Gathers the parent parameters of the current parameter.

        Returns:
            list[AbstractModelComponent]: Parent parameters of the current parameter.
        """
        return list(self._parents.values())

    @property
    def children(self) -> list["AbstractModelComponent"]:
        """
        Gathers the children parameters of the current parameter.

        Returns:
            list[AbstractModelComponent]: Children parameters of the current parameter.
        """
        return self._children.copy()
