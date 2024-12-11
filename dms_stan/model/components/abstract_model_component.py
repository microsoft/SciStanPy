"""Holds the abstract classes for the core components of a DMS Stan model."""

from abc import ABC, abstractmethod
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
import torch

import dms_stan as dms
import dms_stan.model.components as dms_components


class AbstractModelComponent(ABC):
    """Base class for all core components of a DMS Stan model."""

    # Define allowed ranges for the parameters used to define this one
    POSITIVE_PARAMS: set[str] = set()
    NEGATIVE_PARAMS: set[str] = set()
    SIMPLEX_PARAMS: set[str] = set()

    # Define the stan data type
    BASE_STAN_DTYPE: Literal["real", "int", "simplex"] = "real"

    # Define the bounds for this parameter
    LOWER_BOUND: Optional[float | int] = None
    UPPER_BOUND: Optional[float | int] = None
    IS_SIMPLEX: bool = False

    def __init__(  # pylint: disable=unused-argument
        self,
        *,
        shape: tuple[int, ...] = (),
        **parameters: "dms.custom_types.CombinableParameterType",
    ):
        """Builds a parameter instance with the given shape."""

        # Define placeholder variables
        self._model_varname: str = ""  # DMS Stan Model variable name
        self.observable: bool = False  # Whether the parameter is observable
        self._parents: dict[str, AbstractModelComponent]
        self._component_to_paramname: dict[AbstractModelComponent, str]
        self._shape: tuple[int, ...] = shape  # Shape of the parameter
        self._children: list[AbstractModelComponent] = []  # Children of the component

        # Validate incoming parameters
        self._validate_parameters(parameters)

        # Set parents
        self._set_parents(parameters)

        # Link parent and child parameters
        for val in self._parents.values():
            val._record_child(self)

        # Set the shape
        self._set_shape()

    def _validate_parameters(
        self,
        parameters: dict[str, "dms.custom_types.CombinableParameterType"],
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
            if self.LOWER_BOUND is not None or self.LOWER_BOUND != 0.0:
                raise ValueError("Simplex parameters cannot have lower bounds")
            if self.UPPER_BOUND is not None or self.UPPER_BOUND != 1.0:
                raise ValueError("Simplex parameters cannot have upper bounds")

    def _set_parents(
        self,
        parameters: dict[str, "dms.custom_types.CombinableParameterType"],
    ) -> None:
        """Sets the parent parameters of the current parameter."""

        # Convert any non-model components to model components, making sure to
        # propagate any restrictions on
        self._parents = {}
        for name, val in parameters.items():

            # Just the value if an AbstractModelComponent
            if isinstance(val, AbstractModelComponent):
                self._parents[name] = val
                continue

            # Otherwise, convert to a constant model component with the appropriate
            # bounds
            if name in self.POSITIVE_PARAMS:
                lower_bound = 0
                upper_bound = None
            elif name in self.NEGATIVE_PARAMS:
                lower_bound = None
                upper_bound = 0
            elif name in self.SIMPLEX_PARAMS:
                lower_bound = 0
                upper_bound = 1
            else:
                lower_bound = None
                upper_bound = None
            self._parents[name] = dms_components.Constant(
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
            broadcasted_shape = np.broadcast_shapes(
                self._shape, *[param.shape for param in self.parents]
            )
        except ValueError as error:
            raise ValueError("Shape is not broadcastable to parent shapes") from error

        # The broadcasted shape must be the same as the shape of the parameter if
        # it is not 0-dimensional.
        if broadcasted_shape != self._shape and self._shape != ():
            raise ValueError(
                "Provided shape does not match broadcasted shapes if of parents"
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
        self, index_opts: tuple[str, ...], _name_override: str = ""
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
        base_name = _name_override or self.model_varname

        # If there are no indices, then we just return the variable name
        if self.ndim == 0:
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

        # Now draw from the current parameter and check the bounds
        draws = self._draw(n, level_draws, seed=seed)
        assert self.LOWER_BOUND is None or np.all(draws >= self.LOWER_BOUND), (
            draws,
            type(self),
            self.LOWER_BOUND,
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
    def get_transformation_assignment(self, index_opts: tuple[str, ...]) -> str:
        """Gets the transformation assignment operation for the model component."""

    @abstractmethod
    def get_target_incrementation(self, index_opts: tuple[str, ...]) -> str:
        """Gets the target incrementation operation for the model component."""

    @abstractmethod
    def _handle_transformation_code(
        self, param: "AbstractModelComponent", index_opts: tuple[str, ...]
    ) -> str:
        """Handles code formatting for a transformed parameter."""

    @abstractmethod
    def get_stan_code(self, index_opts: tuple[str, ...]) -> dict[str, str]:
        """Gets the Stan code for the parameter."""

        def get_formattables(param: "AbstractModelComponent") -> str:
            """Get the formattables for the parameter."""
            # If the parameter is a constant or another parameter, record
            if isinstance(param, (dms_components.Constant, dms_components.Parameter)):
                return param.get_indexed_varname(index_opts)

            # If the parameter is transformed and not named, the computation is
            # happening in the model. Otherwise, the computation has already happened
            # in the transformed parameters block. Note that THIS instance handles the
            # transformation code based on ITS type, not the type of the parameter.
            elif isinstance(param, dms_components.TransformedParameter):
                return self._handle_transformation_code(
                    param=param, index_opts=index_opts
                )

            # Otherwise, raise an error
            else:
                raise TypeError(f"Unknown model component type {type(param)}")

        # Get the formattables and return
        return {name: get_formattables(param) for name, param in self._parents.items()}

    def declare_stan_variable(self, varname: str) -> str:
        """Declares a variable in Stan code."""
        return f"{self.stan_dtype} {varname}"

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __contains__(self, key: str) -> bool:
        """Check if the parameter has a parent with the given key"""
        return key in self._parents

    def __getitem__(self, key: str) -> "AbstractModelComponent":
        """Get the parent parameter with the given key"""
        return self._parents[key]

    def __getattr__(self, key: str) -> "AbstractModelComponent":
        """Get the parent parameter with the given key"""
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(f"Attribute '{key}' not found in {self}") from error

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
        if dtype == "real":  # Becomes vector or array of vectors
            dtype = f"vector{self.stan_bounds}[{string_shape[-1]}]"
            if self.ndim > 1:
                dtype = f"array[{','.join(string_shape[:-1])}] {dtype}"

        elif dtype == "int":  # Becomes array
            dtype = f"array[{','.join(string_shape)}] int{self.stan_bounds}"

        elif dtype == "simplex":  # Becomes array of simplexes
            dtype = f"array[{','.join(string_shape[:-1])}] simplex[{string_shape[-1]}]"

        else:
            raise AssertionError(f"Unknown data type {dtype}")

        return dtype

    @property
    def stan_code_level(self) -> int:
        """
        The level (index of the for-loop block) at which the parameter is manipulated
        in the Stan code. This is the number of dimensions of the parameter minus
        one, clipped at zero. This is because the last dimension is assumed to always
        be vectorized.
        """
        return max(self.ndim - 1, 0)

    @property
    def stan_parameter_declaration(self) -> str:
        """Returns the Stan parameter declaration for this parameter."""
        return self.declare_stan_variable(self.model_varname)

    @property
    def model_varname(self) -> str:
        """Return the DMS Stan variable name for this parameter"""
        # If the _model_varname variable is set, then we return it
        if self.is_named:
            return self._model_varname

        # Otherwise, we automatically create the name. This is the name of the
        # child components and the name of this component as defined in that child
        # component separated by underscores.
        return "_".join(
            [
                f"{child.model_varname}_{name}"
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
