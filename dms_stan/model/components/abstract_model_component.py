"""Holds the abstract classes for the core components of a DMS Stan model."""

from abc import ABC, abstractmethod
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
import torch

from dms_stan.exceptions import NumpySampleError
from dms_stan.model.components import constants, parameters, transformations


class AbstractModelComponent(ABC):
    """Base class for all core components of a DMS Stan model."""

    # Define allowed ranges for the parameters used to define this one
    POSITIVE_PARAMS: set[str] = set()
    NEGATIVE_PARAMS: set[str] = set()
    SIMPLEX_PARAMS: set[str] = set()
    LOG_SIMPLEX_PARAMS: set[str] = set()

    # Define the stan data type
    BASE_STAN_DTYPE: Literal["real", "int", "simplex"] = "real"

    # Define the bounds for this parameter
    LOWER_BOUND: Optional[float | int] = None
    UPPER_BOUND: Optional[float | int] = None
    IS_SIMPLEX: bool = False
    IS_LOG_SIMPLEX: bool = False

    def __init__(  # pylint: disable=unused-argument
        self,
        *,
        shape: tuple[int, ...] = (),
        **model_params: "dms.custom_types.CombinableParameterType",
    ):
        """Builds a parameter instance with the given shape."""

        # Define placeholder variables
        self._model_varname: str = ""  # DMS Stan Model variable name
        self._parents: dict[str, AbstractModelComponent]
        self._component_to_paramname: dict[AbstractModelComponent, str]
        self._shape: tuple[int, ...] = shape  # Shape of the parameter
        self._children: list[AbstractModelComponent] = []  # Children of the component

        # Validate incoming parameters
        self._validate_parameters(model_params)

        # Set parents
        self._set_parents(model_params)

        # Link parent and child parameters
        for val in self._parents.values():
            val._record_child(self)

        # Set the shape
        self._set_shape()

    def _validate_parameters(
        self,
        model_params: dict[str, "dms.custom_types.CombinableParameterType"],
    ) -> None:
        """Checks inputs to the __init__ method for validity."""
        # All bounded parameters must be named in the parameter dictionary
        if missing_names := (
            self.POSITIVE_PARAMS
            | self.NEGATIVE_PARAMS
            | self.SIMPLEX_PARAMS
            | self.LOG_SIMPLEX_PARAMS
        ) - set(model_params.keys()):
            raise ValueError(
                f"{', '.join(missing_names)} are bounded parameters but are missing "
                "from those defined"
            )

        # Lower bounds must be below upper bounds
        if (
            self.LOWER_BOUND is not None
            and self.UPPER_BOUND is not None
            and self.LOWER_BOUND >= self.UPPER_BOUND
        ):
            raise ValueError("Lower bound must be less than upper bound")

        # If THIS parameter is a simplex, then the upper and lower bounds cannot
        # be set
        if self.IS_SIMPLEX:
            if self.LOWER_BOUND is not None and self.LOWER_BOUND != 0.0:
                raise ValueError("Simplex parameters cannot have lower bounds")
            if self.UPPER_BOUND is not None and self.UPPER_BOUND != 1.0:
                raise ValueError("Simplex parameters cannot have upper bounds")

        # If THIS parameter is a log simplex, then the upper and lower bounds cannot
        # be set
        if self.IS_LOG_SIMPLEX:
            if self.LOWER_BOUND is not None and self.LOWER_BOUND != -np.inf:
                raise ValueError("Log simplex parameters cannot have lower bounds")
            if self.UPPER_BOUND is not None and self.UPPER_BOUND != 0.0:
                raise ValueError("Log simplex parameters cannot have upper bounds")

    def _set_parents(
        self,
        model_params: dict[str, "dms.custom_types.CombinableParameterType"],
    ) -> None:
        """Sets the parent parameters of the current parameter."""

        # Convert any non-model components to model components, making sure to
        # propagate any restrictions on
        self._parents = {}
        for name, val in model_params.items():

            # Just the value if an AbstractModelComponent
            if isinstance(val, AbstractModelComponent):
                self._parents[name] = val
                continue

            # Otherwise, convert to a constant model component with the appropriate
            # bounds
            if name in self.POSITIVE_PARAMS:
                lower_bound = 0
                upper_bound = None
            elif name in self.NEGATIVE_PARAMS or name in self.LOG_SIMPLEX_PARAMS:
                lower_bound = None
                upper_bound = 0
            elif name in self.SIMPLEX_PARAMS:
                lower_bound = 0
                upper_bound = 1
            else:
                lower_bound = None
                upper_bound = None
            self._parents[name] = constants.Constant(
                value=val,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )

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

    def _set_shape(self) -> None:
        """Sets the shape of the draws for the parameter."""
        # The shape must be broadcastable to the shapes of the parameters.
        try:
            parent_shapes = [param.shape for param in self.parents]
            broadcasted_shape = np.broadcast_shapes(self._shape, *parent_shapes)
        except ValueError as error:
            raise ValueError(
                f"Shape is not broadcastable to parent shapes while initializing instance "
                f"of {self.__class__.__name__}: {','.join(map(str, parent_shapes))} "
                f"not broadcastable to {self._shape}"
            ) from error

        # The broadcasted shape must be the same as the shape of the parameter if
        # it is not 0-dimensional.
        if broadcasted_shape != self._shape and self._shape != ():
            raise ValueError(
                "Provided shape does not match broadcasted shapes of parents while "
                f"initializing instance of {self.__class__.__name__}. {self._shape} "
                f"!= {broadcasted_shape}"
            )

        # Set the shape
        self._shape = broadcasted_shape

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
                for paramname, param in child._parents.items()  # pylint: disable=protected-access
                if param is self
            ]
            assert len(names) == 1

            # Record the name
            child_paramnames[child] = names[0]

        return child_paramnames

    def get_indexed_varname(
        self, index_opts: tuple[str, ...] | None, _name_override: str = ""
    ) -> str:
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
        # If the name override is provided, then we use that name
        base_name = _name_override or self.stan_model_varname

        # If there are no indices, then we just return the variable name
        if self.ndim == 0 or index_opts is None:
            return base_name

        # Singleton dimensions get a "1" index. All others get the index options.
        indices = [
            "1" if dimsize == 1 else index_opts[i]
            for i, dimsize in enumerate(self.shape)
        ]

        # If the last dimension is vectorized, then we don't need to index it. We
        # assume that the last dimension is always vectorized.
        if indices[-1] != "1":
            indices = indices[:-1]

        # If there are no indices, then we just return the variable name
        if len(indices) == 0:
            return base_name

        # Build the indexed variable name
        indexed_varname = f"{base_name}[{','.join(indices)}]"

        return indexed_varname

    @abstractmethod
    def _draw(
        self, n: int, level_draws: dict[str, npt.NDArray], seed: Optional[int]
    ) -> npt.NDArray:
        """Sample from the distribution that represents the parameter"""

    def draw(
        self,
        n: int,
        *,
        _drawn: Optional[dict["AbstractModelComponent", npt.NDArray]] = None,
        seed: Optional[int] = None,
    ) -> tuple[npt.NDArray, dict["AbstractModelComponent", npt.NDArray]]:
        """
        Recursively draws from the parameter and its parents. The draws for this
        component are the first object returned. The draws from the recursion are
        the second object returned.
        """
        # Build the _drawn dictionary if it is not already built
        if _drawn is None:
            _drawn = {}

        # Loop over the parents and draw from them if we haven't already
        level_draws: dict[str, npt.NDArray] = {}
        for paramname, parent in self._parents.items():

            # If the parent has been drawn, use the already drawn value. Otherwise,
            # draw from the parent.
            if parent in _drawn:
                parent_draw = _drawn[parent]
            else:
                parent_draw, _ = parent.draw(n, _drawn=_drawn, seed=seed)
                _drawn[parent] = parent_draw

            # Add the parent draw to the level draws. Expand the number of dimensions
            # if necessary to account for the addition of "n" draws.
            dims_to_add = (
                self.ndim + 1
            ) - parent_draw.ndim  # Implicit prepended dimensions
            assert (
                dims_to_add >= 0
            ), f"{self.ndim} {parent_draw.ndim} {self.model_varname} {parent.model_varname}"
            level_draws[paramname] = np.expand_dims(
                parent_draw, axis=tuple(range(1, dims_to_add + 1))
            )

        # Now draw from the current parameter
        try:
            draws = self._draw(n, level_draws, seed=seed)
        except ValueError as error:
            raise NumpySampleError(
                f"Error encountered when trying to sample from {self.model_varname}: {error}"
            ) from error

        assert self.LOWER_BOUND is None or np.all(draws >= self.LOWER_BOUND), (
            f"Draw from `{self}` must be greater than or equal to {self.LOWER_BOUND} "
            f"but got {draws.min()}"
        )
        assert self.UPPER_BOUND is None or np.all(draws <= self.UPPER_BOUND)
        assert not self.IS_SIMPLEX or np.allclose(np.sum(draws, axis=-1), 1)

        # Update the _drawn dictionary
        assert self not in _drawn
        _drawn[self] = draws

        # Test the ranges of the draws
        for paramname, param in self._parents.items():
            if paramname in self.POSITIVE_PARAMS:
                assert np.all(_drawn[param] >= 0)
            elif paramname in self.NEGATIVE_PARAMS:
                assert np.all(_drawn[param] <= 0)
            elif paramname in self.SIMPLEX_PARAMS:
                assert np.all((_drawn[param] >= 0) & (_drawn[param] <= 1))
                assert np.allclose(np.sum(_drawn[param], axis=-1), 1)
            elif paramname in self.LOG_SIMPLEX_PARAMS:
                assert np.all(_drawn[param] < 0)
                assert np.allclose(np.sum(np.exp(_drawn[param]), axis=-1), 1)

        return draws, _drawn

    def walk_tree(
        self, walk_down: bool = True, _recursion_depth: int = 1
    ) -> list[tuple[int, "AbstractModelComponent", "AbstractModelComponent"]]:
        """
        Walks the tree of parameters, either up or down. "up" means walking from
        the children to the parents, while "down" means walking from the parents
        to the children.

        Args:
            walk_down (bool): Whether to walk down the tree (True) or up the tree (False).

        Returns:
            list[tuple[int, "AbstractModelComponent", "AbstractModelComponent"]]: The
            lineage. Each tuple contains the recursion depth relative to the original
            calling parameter in the first position, the current parameter in the
            second position, and that parameter's relative (child if walking down,
            parent if walking up) in the second.
        """
        # Get the variables to loop over
        relatives = self.children if walk_down else self.parents

        # Recurse
        to_return = []
        for relative in relatives:

            # Add the current parameter and the relative parameter to the list
            to_return.append((_recursion_depth, self, relative))

            # Recurse on the relative parameter
            to_return.extend(
                relative.walk_tree(
                    walk_down=walk_down, _recursion_depth=_recursion_depth + 1
                )
            )

        return to_return

    def get_stan_level_compatibility(self, other: "AbstractModelComponent") -> int:
        """
        Determines the extent to which the shapes of this and the other components
        are compatible. This is the number of shared leading dimensions between
        the two components' shapes.
        """
        # Define the compatibility level
        compat_level = 0

        # Get the extent to which the shapes this and the other components are compatible.
        for i, (prev_dimsize, current_dimsize) in enumerate(
            zip(self.shape, other.shape)
        ):

            # If the dimensions are equal or at least one is 1, they are compatible
            # at this level of indentation
            if (
                prev_dimsize == current_dimsize
                or prev_dimsize == 1
                or current_dimsize == 1
            ):
                compat_level = i
                break

        return compat_level

    def get_supporting_functions(self) -> list[str]:
        """
        Gets the set of functions that need to be defined in the Stan model to support
        the model component.

        Each element of the list is a string that contains the function definition
        in Stan code.
        """
        # The default is no supporting functions
        return []

    def get_transformation_assignment(
        self, index_opts: tuple[str, ...]  # pylint: disable=unused-argument
    ) -> str:
        """
        Gets the transformation assignment operation for the model component. The
        default is to return an empty string, which means that the parameter has
        no transformation assignment.
        """
        return ""

    def get_target_incrementation(
        self, index_opts: tuple[str, ...]  # pylint: disable=unused-argument
    ) -> str:
        """
        Gets the target incrementation operation for the model component. The default
        is to return an empty string, which means that the target variable is not
        incremented by this component.
        """
        return ""

    @abstractmethod
    def get_right_side(self, index_opts: tuple[str, ...] | None) -> dict[str, str]:
        """
        Gets the right side of any statement (i.e., Stan code to the right of an
        assignment or distribution statement) for the parameter. Note that if index
        options are provided, then the right side will be indexed by those options.
        If index options are not provided, then we assume that this is the right-hand
        side in a partial summation function and that we will be operating over
        a slice of the final dimension.

        Note that the abstract method is not called directly, but defines the first
        few steps. The abstract method will return a dictionary that gives the Stan
        string representations for the model components that make up the right-hand-
        side of the statement (the parent parameters of this parameter). The keys
        of the dictionary are the names of the parent parameters and the values are
        the component name appropriately indexed if named, a constant, or a parameter,
        or the thread of operations that make up the appropriate transformation
        for an unnamed transformed parameter.

        The dictionary is then used to format the Stan code for the right-hand-side
        of the statement in the get_right_side method of the child class.
        """
        # Get variables that make up the right side of the statement. These will
        # be the parent parameters of the current parameter.
        components: dict[str, str] = {}
        for name, param in self._parents.items():

            # If the parameter is a constant or another parameter OR it is a named
            # transformed parameter, we get its indexed variable name
            if (
                isinstance(param, (constants.Constant, parameters.Parameter))
                or param.is_named
            ):
                components[name] = param.get_indexed_varname(index_opts)

            # Otherwise, we need to get the thread of operations that make up the
            # transformation for the parameter. This is equivalent to calling the
            # get_right_side method of the parameter.
            elif isinstance(param, transformations.TransformedParameter):
                components[name] = param.get_right_side(index_opts)

            # Otherwise, raise an error
            else:
                raise TypeError(f"Unknown model component type {type(param)}")

        return components

    def get_right_side_components(self) -> list["AbstractModelComponent"]:
        """
        Gets the components (i.e., Python objects) that make up the right-hand-side
        of this component's statement.
        """
        # Recurse over the parents up until we reach a named one or the top of the
        # tree
        components = []
        for component in self._parents.values():
            # If named, a parameter, or a constant, we can just add it to the list
            if (
                isinstance(component, (constants.Constant, parameters.Parameter))
                or component.is_named
            ):
                components.append(component)
                continue

            # Otherwise, this must be a transformed parameter that is not named
            # and we need to recurse up to the first named parameter
            assert isinstance(component, transformations.TransformedParameter)
            components.extend(component.get_right_side_components())

        return components

    def declare_stan_variable(self, varname: str) -> str:
        """Declares a variable in Stan code."""
        return f"{self.stan_dtype} {varname}"

    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of the component. This must be defined."""

    def __repr__(self) -> str:
        """This is identical to the string representation of the component."""
        return str(self)

    def __contains__(self, key: str) -> bool:
        """Check if the parameter has a parent with the given key"""
        return key in self._parents

    def __getitem__(self, key: str) -> "AbstractModelComponent":
        """Get the parent parameter with the given key"""
        return self._parents[key]

    def __getattr__(self, key: str) -> "AbstractModelComponent":
        """Get the parent parameter with the given key"""
        # Make sure we don't have a circular reference between `__getattr__` and
        # `__getitem__`
        if key == "_parents":
            raise AttributeError("No attribute '_parents' in this object")

        # If the key is not in the parents, then we raise an error
        try:
            return self[key]
        except KeyError as error:
            try:
                string_repr = repr(self)
            except Exception as error2:  # pylint: disable=broad-except
                string_repr = super().__repr__()
                raise AttributeError(
                    f"Attribute '{key}' not found in {string_repr}. "
                    "Encountered an error while trying to get the string representation."
                ) from error2

            # Otherwise, just raise the original error
            raise AttributeError(
                f"Attribute '{key}' not found in {string_repr}"
            ) from error

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
    def stan_bounds(self) -> str:
        """Return the Stan bounds for this parameter"""
        # Format the lower and upper bounds
        lower = "" if self.LOWER_BOUND is None else f"lower={self.LOWER_BOUND}"
        upper = "" if self.UPPER_BOUND is None else f"upper={self.UPPER_BOUND}"

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
        dtype = self.BASE_STAN_DTYPE

        # Base data type for 0-dimensional parameters. If the parameter is 0-dimensional,
        # then we can only have real or int as the data type.
        if self.ndim == 0:
            assert dtype in {"real", "int"}
            return f"{dtype}{self.stan_bounds}"

        # Convert shape to strings
        string_shape = [str(dim) for dim in self.shape]

        # Handle different data types for different dimensions
        check_array = False
        if dtype == "real":  # Becomes vector or array of vectors
            dtype = f"vector{self.stan_bounds}[{string_shape[-1]}]"
            check_array = True
        elif dtype == "int":  # Becomes array
            dtype = f"array[{','.join(string_shape)}] int{self.stan_bounds}"
        elif dtype == "simplex":  # Becomes array of simplexes
            dtype = f"simplex[{string_shape[-1]}]"
            check_array = True
        else:
            raise AssertionError(f"Unknown data type {dtype}")

        # Convert to an array of vectors or simplexes if necessary
        if check_array and self.ndim > 1:
            dtype = f"array[{','.join(string_shape[:-1])}] {dtype}"

        return dtype

    @property
    def stan_code_level(self) -> int:
        """
        The level (index of the for-loop block) at which the parameter is manipulated
        in the Stan code. This is the number of dimensions, excluding trailing singleton
        dimensions, of the parameter minus one, clipped at zero. This is because
        the last dimension is assumed to always be vectorized.
        """
        # Default is number of dimensions minus one. We subtract one because the
        # last dimension is always vectorized.
        level = self.ndim - 1

        # Subtract off trailing singleton dimensions ignoring the last dimension
        # (again, always vectorized)
        for dimsize in reversed(self.shape[:-1]):
            if dimsize > 1:
                break
            level -= 1

        # Clip at zero
        return max(level, 0)

    @property
    def stan_parameter_declaration(self) -> str:
        """Returns the Stan parameter declaration for this parameter."""
        return self.declare_stan_variable(self.stan_model_varname)

    @property
    def model_varname(self) -> str:
        """Return the DMS Stan variable name for this parameter"""
        # If the _model_varname variable is set, then we return it
        if self.is_named:
            return self._model_varname

        # Otherwise, we automatically create the name. This is the name of the
        # child components and the name of this component as defined in that child
        # component separated by underscores.
        return ".".join(
            [
                f"{child.model_varname}.{name}"
                for child, name in self.get_child_paramnames().items()
                if not isinstance(
                    child, transformations.transformed_data.TransformedData
                )
            ]
        )

    @model_varname.setter
    def model_varname(self, name: str) -> None:
        """Set the DMS Stan variable name for this parameter"""
        # If the name is not set, then we set it
        if self._model_varname == "":
            self._model_varname = name
        else:
            raise ValueError(
                "Cannot set model variable name more than once. Trying to rename "
                f"{self._model_varname} to {name}"
            )

    @property
    def stan_model_varname(self) -> str:
        """Return the Stan variable name for this parameter"""
        return self.model_varname.replace(".", "__")

    @property
    def constants(self):
        """Return the constants of the component."""
        return {
            name: component
            for name, component in self._parents.items()
            if isinstance(component, constants.Constant)
        }

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

    @property
    @abstractmethod
    def torch_parametrization(self) -> torch.Tensor:
        """Return the PyTorch parameters for this component, appropriately transformed."""

    @property
    def observable(self) -> bool:
        """
        Return whether the parameter is observable. By default, all parameters are
        not observable.
        """
        return False
