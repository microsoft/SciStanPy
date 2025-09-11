# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Abstract base classes for SciStanPy model components.

This module defines the foundational abstract class that forms the core
architecture of SciStanPy model components, including parameters, constants, and
transformations. Users typically do not interact with this module directly; instead,
they use the concrete implementations provided in the
:py:mod:`scistanpy.model.components.constants` and
:py:mod:`scistanpy.model.components.parameters` submodules.

The module establishes the common functionality that all model components must
implement.

Core Abstractions:

    - **Component Hierarchy**: Parent-child relationships between model elements
    - **Stan Code Generation**: Automatic translation to Stan programming language
    - **Shape Broadcasting**: Automatic handling of multi-dimensional parameters
    - **Dependency Management**: Tracking and validation of component relationships

Key Responsibilities:

    - Define abstract interfaces for model component behavior
    - Implement common functionality for shape handling and validation
    - Provide Stan code generation template
    - Manage component relationships and dependency graphs
    - Handle parameter bounds and constraints
    - Support sampling and drawing from component distributions

Stan Integration: The abstract base provides core Stan code generation capabilities
including:

    - Variable declarations with appropriate types and constraints
    - Index management for multi-dimensional arrays
    - Target increment and transformation assignment generation
    - Support function inclusion for custom distributions

Component Relationships: The hierarchy system enables complex model construction
through:

    - Parent-child linkage for dependency tracking
    - Parameter name resolution and validation
    - Automatic shape broadcasting across related components
    - Tree traversal for model analysis and code generation

This foundational layer enables the construction of sophisticated probabilistic
models while maintaining type safety and automatic Stan code generation.
"""

# pylint: disable=too-many-lines

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Optional, overload, TYPE_CHECKING, Union

import numpy as np
import numpy.typing as npt
import torch

from scistanpy.exceptions import NumpySampleError
from scistanpy import utils

# Lazy imports for performance and to avoid circular imports
constants_module = utils.lazy_import("scistanpy.model.components.constants")
parameters = utils.lazy_import("scistanpy.model.components.parameters")
transformed_data = utils.lazy_import(
    "scistanpy.model.components.transformations.transformed_data"
)
transformed_parameters = utils.lazy_import(
    "scistanpy.model.components.transformations.transformed_parameters"
)

if TYPE_CHECKING:
    from scistanpy import custom_types


class AbstractModelComponent(ABC):
    """Abstract base class for all SciStanPy model components.

    This class defines the fundamental interface and common functionality for
    all elements in a SciStanPy probabilistic model. It provides the foundation
    for :py:mod:`~scistanpy.model.components.parameters`,
    :py:mod:`~scistanpy.model.components.constants`,
    :py:mod:`~scistanpy.model.components.transformations.transformed_parameters`,
    and other model components.

    :param shape: Shape of the component array. Defaults to scalar ().
    :type shape: Union[tuple[custom_types.Integer, ...], custom_types.Integer]
    :param model_params: Named parameters that this component depends on
    :type model_params: custom_types.CombinableParameterType

    :cvar POSITIVE_PARAMS: Set of parameter names that must be positive
    :cvar NEGATIVE_PARAMS: Set of parameter names that must be negative
    :cvar SIMPLEX_PARAMS: Set of parameter names that must be simplexes
    :cvar LOG_SIMPLEX_PARAMS: Set of parameter names that must be log-simplexes
    :cvar BASE_STAN_DTYPE: Base Stan data type for this component
    :cvar LOWER_BOUND: Lower bound constraint for component values
    :cvar UPPER_BOUND: Upper bound constraint for component values
    :cvar IS_SIMPLEX: Whether this component represents a simplex
    :cvar IS_LOG_SIMPLEX: Whether this component represents a log-simplex
    :cvar FORCE_PARENT_NAME: Whether to force naming of parent variables in Stan code
    :cvar FORCE_LOOP_RESET: Whether to force loop reset in Stan code

    The class provides core functionality for:

    - Component relationship management (parents and children)
    - Shape validation and broadcasting
    - Stan code generation for variable declarations and operations
    - Sampling and drawing from component distributions
    - Tree traversal for model analysis

    All model components must implement the abstract methods for drawing samples
    and generating Stan code appropriate to their type.
    """

    # Define allowed ranges for the parameters used to define this one
    POSITIVE_PARAMS: set[str] = set()
    """Class variable giving the set of parent parameter names that must be positive."""

    NEGATIVE_PARAMS: set[str] = set()
    """Class variable giving the set of parent parameter names that must be negative."""

    SIMPLEX_PARAMS: set[str] = set()
    """
    Class variable giving the set of parent parameter names that must be simplexes.
    The sum over the last dimension for any parent parameter named here must equal 1.
    """

    LOG_SIMPLEX_PARAMS: set[str] = set()
    """
    Class variable giving the set of parent parameter names that must be log-simplexes.
    The sum of exponentials over the last dimension for any parent parameter named
    here must equal 1
    """

    # Define the stan data type
    BASE_STAN_DTYPE: Literal["real", "int", "simplex"] = "real"
    """Class variable giving the base Stan data type for this component."""

    # Define the bounds for this parameter
    LOWER_BOUND: Optional["custom_types.Float" | "custom_types.Integer"] = None
    """
    Class variable giving the lower bound constraint for component values. None
    if unbounded.
    """

    UPPER_BOUND: Optional["custom_types.Float" | "custom_types.Integer"] = None
    """
    Class variable giving the upper bound constraint for component values. None
    if unbounded.
    """

    IS_SIMPLEX: bool = False
    """
    Class variable giving whether this component represents a simplex (elements
    over last dim sum to 1).
    """

    IS_LOG_SIMPLEX: bool = False
    """
    Class variable giving whether this component represents a log-simplex (exponentials
    over last dim sum to 1).
    """

    # Do we force parents of this parameter to be named in Stan code?
    FORCE_PARENT_NAME: bool = False
    """
    Class variable noting whether to force naming of parent variables in Stan code.
    If ``True`` and not provided by the user, parent parameters will be assigned
    names automatically in the Stan code.
    """

    # Do we want the loop to be reset in Stan code?
    FORCE_LOOP_RESET: bool = False
    """
    Class variable noting whether to force loop reset in Stan code. This is useful
    where this parameter's shape makes it appear nestable with another inside the
    same loop, but it actually is not.
    """

    def __init__(  # pylint: disable=unused-argument
        self,
        *,
        shape: tuple["custom_types.Integer", ...] | "custom_types.Integer" = (),
        **model_params: "custom_types.CombinableParameterType",
    ):
        """Initialize a model component with specified shape and parameters.

        :param shape: Shape of the component.
        :type shape: Union[tuple[custom_types.Integer, ...], custom_types.Integer]
        :param model_params: Named parameters this component depends on
        :type model_params: custom_types.CombinableParameterType

        The initialization process:
        1. Normalizes shape specification to tuple format (i.e., integer to 1-element tuple)
        2. Validates parameter constraints and bounds
        3. Converts non-component parameters to constants
        4. Establishes parent-child relationships
        5. Validates and sets component shape through broadcasting
        """
        # Convert shape to the appropriate type
        try:
            len(shape)
        except TypeError:
            shape = (shape,)

        # Define placeholder variables
        self._model_varname: str = ""  # SciStanPy Model variable name
        self._parents: dict[str, AbstractModelComponent]
        self._component_to_paramname: dict[AbstractModelComponent, str]
        self._shape: tuple["custom_types.Integer", ...] = shape
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
        model_params: dict[str, "custom_types.CombinableParameterType"],
    ) -> None:
        """Validate parameter constraints and bounds.

        :param model_params: Dictionary of parameter names to values
        :type model_params: dict[str, custom_types.CombinableParameterType]

        :raises ValueError: If required bounded parameters are missing
        :raises ValueError: If bounds are invalid (lower >= upper)
        :raises ValueError: If simplex parameters have incompatible bounds
        :raises ValueError: If log-simplex parameters have incompatible bounds

        This method ensures that:
        - All bounded parameters are present in the parameter dictionary
        - Lower bounds are less than upper bounds
        - Simplex and log-simplex parameters have appropriate bound constraints
        """
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
        model_params: dict[str, "custom_types.CombinableParameterType"],
    ) -> None:
        """Establish parent component relationships with automatic constant creation.

        :param model_params: Dictionary of parameter names to values/components
        :type model_params: dict[str, custom_types.CombinableParameterType]

        This method processes the input parameters and:
        1. Preserves existing AbstractModelComponent instances as parents
        2. Converts non-component values to Constant instances with appropriate bounds
        3. Creates bidirectional mapping between components and parameter names
        4. Applies parameter type constraints (positive, negative, simplex, etc.)
        """
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
            self._parents[name] = constants_module.Constant(
                value=val,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )

        # Map components to param names
        self._component_to_paramname = {v: k for k, v in self._parents.items()}
        assert len(self._component_to_paramname) == len(self._parents)

    def _record_child(self, child: "AbstractModelComponent") -> None:
        """Record a child component in the dependency graph.

        :param child: Child component that depends on this component
        :type child: AbstractModelComponent

        :raises AssertionError: If child is already recorded

        This method maintains the bidirectional parent-child relationships
        that form the model dependency graph. Each child is recorded only
        once to prevent duplicate dependencies.
        """
        # If the child is already in the list of children, then we don't need to
        # add it again
        assert child not in self._children, "Child already recorded"

        # Record the child
        self._children.append(child)

    def _set_shape(self) -> None:
        """Validate and set component shape through broadcasting with parents.

        :raises ValueError: If shape is not broadcastable with parent shapes
        :raises ValueError: If provided shape conflicts with broadcasted shape

        This method:
        1. Collects shapes from all parent components
        2. Attempts to broadcast the component shape with parent shapes
        3. Validates that the final shape is consistent
        4. Sets the component's shape to the broadcasted result

        Shape broadcasting follows NumPy broadcasting rules, ensuring that
        multi-dimensional model components can interact properly.
        """
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
        """Get parameter names that this component defines in its children.

        :returns: Mapping from child components to parameter names they use for this component
        :rtype: dict[AbstractModelComponent, str]

        :raises AssertionError: If a child references this component with multiple parameter names

        This method analyzes the dependency graph to determine how child
        components reference this component. Each child should reference
        this component through exactly one parameter name.
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
        self,
        index_opts: tuple[str, ...] | None,
        offset: "custom_types.Integer" = 0,
        start_dim: "custom_types.Integer" = 0,
        end_dim: Optional["custom_types.Integer"] = -1,
        _name_override: str = "",
    ) -> str:
        """Generate Stan variable name with appropriate indexing.

        :param index_opts: Index variable names to choose from.
        :type index_opts: Optional[tuple[str, ...]]
        :param offset: Number of leading indices to skip. Defaults to 0.
        :type offset: custom_types.Integer
        :param start_dim: First dimension to include in indexing. Defaults to 0.
        :type start_dim: custom_types.Integer
        :param end_dim: Last dimension to include in indexing. Defaults to -1.
        :type end_dim: Optional[custom_types.Integer]
        :param _name_override: Override for base variable name. Defaults to "".
            Internal use only.
        :type _name_override: str

        :returns: Stan variable name with proper indexing
        :rtype: str

        This method generates proper Stan variable names for multi-dimensional
        components, handling:

        - Singleton dimension skipping
        - Index offset management for broadcasting
        - Dimension range selection
        - Automatic vectorization of the last dimension

        The offset parameter accounts for implicit singleton dimensions prepended
        during broadcasting between parent and child components.
        """
        # Offset must be >= 0
        assert offset >= 0, self.model_varname

        # If end dim is not provided, set it to the number of dimensions
        if end_dim is None:
            end_dim = self.ndim

        # If the name override is provided, then we use that name
        base_name = _name_override or self.stan_model_varname

        # If there are no indices, then we just return the variable name
        if self.ndim == 0 or index_opts is None:
            return base_name

        # First and last dim must be positive integers
        start_dim = start_dim if start_dim >= 0 else self.ndim + start_dim
        end_dim = end_dim if end_dim >= 0 else self.ndim + end_dim
        assert end_dim >= start_dim

        # Skip singleton dimensions. All others get the index options.
        indices = [
            index_opts[index_ind]
            for index_ind, dimsize in enumerate(self.shape[start_dim:end_dim], offset)
            if dimsize != 1
        ]

        # If there are no indices, then we just return the variable name
        if len(indices) == 0:
            return base_name

        # Build the indexed variable name
        indexed_varname = f"{base_name}[{','.join(indices)}]"

        return indexed_varname

    @abstractmethod
    def _draw(
        self,
        level_draws: dict[str, Union[npt.NDArray, "custom_types.Float"]],
        seed: Optional["custom_types.Integer"],
    ) -> Union[npt.NDArray, "custom_types.Float", "custom_types.Integer"]:
        """Draw a single sample from this component's distribution.

        :param level_draws: Samples from parent components for this draw
        :type level_draws: dict[str, Union[npt.NDArray, custom_types.Float]]
        :param seed: Random seed for reproducible sampling
        :type seed: Optional[custom_types.Integer]

        :returns: Sample(s) from this component's distribution
        :rtype: Union[npt.NDArray, custom_types.Float, custom_types.Integer]

        This abstract method must be implemented by all concrete component
        classes to define how samples are drawn from their specific
        distribution or deterministic function.
        """

    def draw(
        self,
        n: "custom_types.Integer",
        *,
        _drawn: Optional[dict["AbstractModelComponent", npt.NDArray]] = None,
        seed: Optional["custom_types.Integer"] = None,
    ) -> tuple[npt.NDArray, dict["AbstractModelComponent", npt.NDArray]]:
        """Recursively draw samples from this component and its dependency tree.

        :param n: Number of samples to draw
        :type n: custom_types.Integer
        :param _drawn: Cache of previously drawn samples. Auto-created if None. Defaults to None.
            Internal use only. Used to cache draws throughout recursion.
        :type _drawn: Optional[dict[AbstractModelComponent, npt.NDArray]]
        :param seed: Random seed for reproducible sampling. Defaults to None.
        :type seed: Optional[custom_types.Integer]

        :returns: Tuple of (samples_from_this_component, all_drawn_samples)
        :rtype: tuple[npt.NDArray, dict[AbstractModelComponent, npt.NDArray]]

        :raises NumpySampleError: If sampling fails due to parameter issues
        :raises AssertionError: If drawn values violate component bounds or constraints

        This method implements the complete sampling workflow:

        1. Recursively draws from parent components if not already drawn
        2. Collects parent samples for the current level
        3. Draws n samples from this component using parent values
        4. Validates drawn samples against bounds and constraints
        5. Returns samples and updates the global draw cache

        The method enforces constraint validation including:

        - Lower and upper bound checking
        - Simplex sum-to-one validation
        - Parameter type constraint validation
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

            # Record the parent draw for this level
            level_draws[paramname] = parent_draw

        # Now draw from the current parameter
        try:
            draws = np.stack(
                [
                    self._draw(
                        {k: v[i] for k, v in level_draws.items()},
                        seed=(None if seed is None else seed + i),
                    )
                    for i in range(n)
                ]
            )
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
        self, walk_down: bool = True, _recursion_depth: "custom_types.Integer" = 1
    ) -> list[tuple[int, "AbstractModelComponent", "AbstractModelComponent"]]:
        """Traverse the model component dependency tree.

        :param walk_down: Whether to walk toward children (True) or parents (False).
            Defaults to True.
        :type walk_down: bool
        :param _recursion_depth: Current recursion depth (internal parameter).
            Defaults to 1.
        :type _recursion_depth: custom_types.Integer

        :returns: List of (depth, current_component, relative_component) tuples
        :rtype: list[tuple[int, AbstractModelComponent, AbstractModelComponent]]

        This method enables systematic traversal of the model dependency graph
        in either direction. Each tuple contains:

        - Recursion depth relative to the starting component
        - The current component in the traversal
        - The relative component (child if walking down, parent if walking up)

        Tree traversal is useful for:

        - Model structure analysis and visualization
        - Dependency validation and cycle detection
        - Code generation ordering
        - Model component discovery
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

    def get_shared_leading(
        self, other: "AbstractModelComponent"
    ) -> "custom_types.Integer":
        """Determine the number of compatible leading dimensions with another component.

        :param other: Component to compare shapes with
        :type other: AbstractModelComponent

        :returns: Number of shared leading dimensions
        :rtype: custom_types.Integer

        This method analyzes shape compatibility between components by counting
        the number of leading dimensions that are compatible according to
        broadcasting rules. Dimensions are compatible if they are equal or
        if at least one of them is 1 (singleton).

        Shape compatibility is important for:

        - Determining indexing strategies
        - Validating broadcasting operations
        - Optimizing Stan code generation
        """
        # Define the compatibility level
        compat_level = 0

        # Get the extent to which the shapes this and the other components are compatible.
        for i, (prev_dimsize, current_dimsize) in enumerate(
            zip(self.shape, other.shape)
        ):

            # If the dimensions are equal or at least one is 1, they are compatible
            # at this level of indentation. Otherwise, we break the loop, as we
            # have reached a position where the shapes are not compatible.
            if (
                prev_dimsize == current_dimsize
                or prev_dimsize == 1
                or current_dimsize == 1
            ):
                compat_level = i
            else:
                break

        return compat_level

    def get_supporting_functions(self) -> list[str]:
        """Get Stan function definitions required by this component.

        :returns: List of Stan function definition strings
        :rtype: list[str]

        This method returns Stan function definitions or include statements
        that must be added to the Stan program to support this component.
        The default implementation returns an empty list.

        Custom components may override this method to include:

        - Custom distribution definitions
        - Helper function implementations
        - Include statements for external function libraries
        """
        # The default is no supporting functions
        return []

    def get_transformation_assignment(
        self, index_opts: tuple[str, ...]  # pylint: disable=unused-argument
    ) -> str:
        """Generate Stan code for parameter transformation assignments. This is
        the statement that appears in the ``transformed parameters`` block of the
        Stan program.

        :param index_opts: Index variable names for multi-dimensional access
        :type index_opts: tuple[str, ...]

        :returns: Stan code for transformation assignment (empty by default)
        :rtype: str

        This method generates Stan code for the transformed parameters block,
        where deterministic transformations of parameters are computed.
        The default implementation returns an empty string.

        Components that require parameter transformations (such as non-centered
        parameterizations) should override this method to provide appropriate
        Stan transformation code.
        """
        return ""

    def get_target_incrementation(
        self, index_opts: tuple[str, ...]  # pylint: disable=unused-argument
    ) -> str:
        """Generate Stan code for log-probability target increments. This is the
        statement that appears in the ``model`` block of the Stan program.

        :param index_opts: Index variable names for multi-dimensional access
        :type index_opts: tuple[str, ...]

        :returns: Stan code for target increment (empty by default)
        :rtype: str

        This method generates Stan code for the model block, where
        log-probability contributions are added to the target density.
        The default implementation returns an empty string.

        Probabilistic components should override this method to provide
        appropriate target increment statements for their distributions.
        """
        return ""

    def get_index_offset(
        self, query: Union[str, "AbstractModelComponent"], offset_adjustment: int = 0
    ) -> int:
        """Calculate index offset for multi-dimensional variable access.

        :param query: Component or parameter name to calculate offset for
        :type query: Union[str, AbstractModelComponent]
        :param offset_adjustment: Additional offset to apply. Defaults to 0.
        :type offset_adjustment: int

        :returns: Number of leading indices to skip
        :rtype: int

        :raises KeyError: If query string doesn't match any parent parameter

        This method calculates the appropriate index offset when accessing
        parent components that have different numbers of dimensions. The
        offset accounts for implicit singleton dimensions that are added
        during broadcasting operations (e.g., the offset between ``x`` with shape
        (10,) and ``y`` with shape (2, 10) would be "1").

        Index offsets are essential for:

        - Proper multi-dimensional array indexing in Stan code
        - Handling broadcasting between components of different shapes
        - Maintaining correct dimension alignment in generated code
        """
        # Get the query if we need to
        if isinstance(query, str):
            query = self._parents[query]

        # Calculate offset
        return self.ndim - query.ndim + offset_adjustment

    @abstractmethod
    def get_right_side(
        self,
        index_opts: tuple[str, ...] | None,
        start_dims: dict[str, "custom_types.Integer"] | None = None,
        end_dims: dict[str, "custom_types.Integer"] | None = None,
        offset_adjustment: int = 0,
    ) -> dict[str, str]:
        """Generate Stan code for the right-hand side of statements.

        :param index_opts: Index variable names for multi-dimensional access
        :type index_opts: Optional[tuple[str, ...]]
        :param start_dims: First indexable dimension for each parent parameter.
            Defaults to None.
        :type start_dims: Optional[dict[str, custom_types.Integer]]
        :param end_dims: Last indexable dimension for each parent parameter. Defaults
            to None.
        :type end_dims: Optional[dict[str, custom_types.Integer]]
        :param offset_adjustment: Index offset adjustment. Defaults to 0.
        :type offset_adjustment: int

        :returns: Dictionary mapping parameter names to Stan code strings
        :rtype: dict[str, str]

        This abstract method must be implemented by all model components
        to generate appropriate Stan code for probability statements, transformations,
        and assignments. The method processes parent components and returns
        properly formatted Stan expressions.

        The base implementation in this abstract class provides common
        functionality for:

        - Processing parent component relationships
        - Handling index offsets and dimension slicing
        - Determining when to use variable names vs. inline expressions
        - Managing transformed parameter code generation

        Subclasses extend this foundation to generate component-specific
        Stan code patterns.
        """
        # Get default values for start and end dims
        start_dims = start_dims or {}
        end_dims = end_dims or {}

        # Get variables that make up the right side of the statement. These will
        # be the parent parameters of the current parameter.
        model_components: dict[str, str] = {}
        for name, param in self._parents.items():

            # What's the offset between the current parameter in the loop and THIS
            # parameter
            current_offset = self.get_index_offset(
                param, offset_adjustment=offset_adjustment
            )

            # If the parameter is a constant or another parameter OR it is a named
            # transformed parameter OR the assignment depth changes, we get its
            # indexed variable name offset by the appropriate amount to account
            # for implicit singleton dimensions
            if (
                isinstance(
                    param,
                    (constants_module.Constant, parameters.Parameter),
                )
                or param.is_named
                or param.force_name
            ):
                model_components[name] = param.get_indexed_varname(
                    index_opts,
                    offset=current_offset,
                    start_dim=start_dims.get(name, 0),
                    end_dim=end_dims.get(name, -1),
                )

            # Otherwise, we need to get the thread of operations that make up the
            # transformation for the parameter. This is equivalent to calling the
            # get_right_side method of the parameter.
            elif isinstance(param, transformed_parameters.TransformedParameter):

                # We need to propogate the offset of the current parameter in the
                # loop to ITS parents
                model_components[name] = param.get_right_side(
                    index_opts, offset_adjustment=current_offset
                )

            # Otherwise, raise an error
            else:
                raise TypeError(f"Unknown model component type {type(param)}")

        return model_components

    def declare_stan_variable(self, varname: str, force_basetype: bool = False) -> str:
        """Generate Stan variable declaration with appropriate type and bounds.

        :param varname: Variable name to declare
        :type varname: str
        :param force_basetype: Whether to force array[...] basetype format. Defaults to False.
        :type force_basetype: bool

        :returns: Complete Stan variable declaration
        :rtype: str

        This method combines the Stan data type (from
        :py:meth:`~scistanpy.model.components.abstract_model_component.AbstractModelComponent.get_stan_dtype`)
        with the variable name to create a complete variable declaration suitable
        for use in Stan data, parameters, or other blocks.
        """
        return f"{self.get_stan_dtype(force_basetype)} {varname}"

    def get_assign_depth(self) -> int:
        """Calculate the assignment depth for Stan loop structure.

        :returns: Loop nesting level for this component's assignment
        :rtype: int

        This method determines the appropriate loop nesting level for
        defining this component in Stan code. The depth is calculated as:

        - Number of dimensions minus one (last dimension is vectorized)
        - Minus trailing singleton dimensions (except the last)
        - Clipped to a minimum of zero

        Assignment depth affects:

        - Loop structure in generated Stan code
        - Index variable management
        - Vectorization opportunities
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

    def get_stan_dtype(self, force_basetype: bool = False) -> str:
        """Generate Stan data type declaration for this component.

        :param force_basetype: Whether to force array[...] format instead of vectors.
            Defaults to False.
        :type force_basetype: bool

        :returns: Stan data type string with bounds
        :rtype: str

        :raises AssertionError: If unknown data type is encountered

        This method generates appropriate Stan data type declarations based on:

        - Base data type (real, int, simplex)
        - Component dimensionality
        - Bound constraints
        - Whether vector/array format is preferred
        """
        # Get the base datatype
        dtype = self.BASE_STAN_DTYPE

        # Convert shape to strings
        string_shape = [str(dim) for dim in self.shape if dim > 1]
        ndim = len(string_shape)

        # Base data type for 0-dimensional parameters. If the parameter is 0-dimensional,
        # then we can only have real or int as the data type.
        if ndim == 0:
            assert dtype in {"real", "int"}
            return f"{dtype}{self.stan_bounds}"

        # Handle different data types for different dimensions
        if dtype == "int" or force_basetype:
            return (
                f"array[{','.join(string_shape)}] "
                + ("int" if dtype == "int" else "real")
                + self.stan_bounds
            )
        if dtype == "real":  # Becomes vector or array of vectors
            dtype = f"vector{self.stan_bounds}[{string_shape[-1]}]"
        elif dtype == "simplex":  # Becomes array of simplexes
            dtype = f"simplex[{string_shape[-1]}]"
        else:
            raise AssertionError(f"Unknown data type {dtype}")

        # Convert to an array of vectors or simplexes if necessary
        if ndim > 1:
            return f"array[{','.join(string_shape[:-1])}] {dtype}"

        return dtype

    def get_stan_parameter_declaration(self, force_basetype: bool = False) -> str:
        """Generate Stan parameter declaration for this component.

        :param force_basetype: Whether to force array[...] format. Defaults to False.
        :type force_basetype: bool

        :returns: Complete Stan parameter declaration
        :rtype: str

        This convenience method generates a parameter declaration using the
        component's Stan model variable name and appropriate data type.
        """
        return self.declare_stan_variable(
            self.stan_model_varname, force_basetype=force_basetype
        )

    @abstractmethod
    def __str__(self) -> str:
        """Return human-readable string representation of the component.

        :returns: String representation showing component structure
        :rtype: str

        This abstract method must be implemented by all concrete components
        to provide meaningful string representations for debugging and
        model inspection.
        """

    def __repr__(self) -> str:
        """Return detailed string representation (identical to __str__).

        :returns: String representation of the component
        :rtype: str
        """
        return str(self)

    def __contains__(self, key: str) -> bool:
        """Check if the component has a parent with the given parameter name.

        :param key: Parameter name to check
        :type key: str

        :returns: True if parameter name exists in parents
        :rtype: bool
        """
        return key in self._parents

    @overload
    def __getitem__(self, key: str) -> "AbstractModelParameter": ...

    @overload
    def __getitem__(self, key: "custom_types.IndexType") -> "IndexParameter": ...

    def __getitem__(self, key):
        """Access parent parameters by name or create indexed subcomponents.

        :param key: Parameter name (string) or array indexing specification
        :type key: Union[str, custom_types.IndexType]

        :returns: Parent component or indexed subcomponent
        :rtype: Union[AbstractModelComponent, IndexParameter]

        This method provides two access patterns:
        1. String keys return parent components by parameter name
        2. Index specifications create IndexParameter subcomponents for array slicing
        """
        # If a string, check the parents
        if isinstance(key, str):
            return self._parents[key]

        # Anything else, we must be slicing or selecting the underlying distribution
        if not isinstance(key, tuple):
            key = (key,)

        return transformed_parameters.IndexParameter(self, *key)

    def __getattr__(self, key: str) -> "AbstractModelComponent":
        """Access parent parameters as attributes.

        :param key: Parameter name to access
        :type key: str

        :returns: Parent component with the given parameter name
        :rtype: AbstractModelComponent

        :raises AttributeError: If parameter name not found in parents

        This method enables convenient attribute-style access to parent
        components using dot notation instead of bracket notation.
        """
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
        """Get the shape of this component.

        :returns: Shape tuple for this component
        :rtype: tuple[int, ...]
        """
        return self._shape

    @property
    def ndim(self) -> "custom_types.Integer":
        """Get the number of dimensions of this component.

        :returns: Number of dimensions
        :rtype: custom_types.Integer
        """
        return len(self.shape)

    @property
    def is_named(self) -> bool:
        """Check whether this component has an assigned variable name.

        :returns: True if component has been assigned a model variable name
        :rtype: bool

        Examples:
            .. code-block:: python

                # Example model with named and unnamed parameters
                class MyModel(Model):
                    def __init__(self):
                        super().__init__()
                        self.param1 = Parameter(...)  # is_named is True
                        param2 = Parameter(...)       # is_named is False

                # Outside of a model, is_named is always False
                param = Parameter(...)
                assert not param.is_named
        """
        return self._model_varname != ""

    @property
    def stan_bounds(self) -> str:
        """Generate Stan bounds specification string.

        :returns: Stan bounds specification (empty if no bounds)
        :rtype: str

        This property formats
        :py:attr:`~scistanpy.model.components.abstract_model_component.AbstractModelComponent.LOWER_BOUND`
        and :py:attr:`~scistanpy.model.components.abstract_model_component.AbstractModelComponent.UPPER_BOUND`
        into Stan's bound format. Returns an empty string if no bounds are specified.
        """
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
    def assign_depth(self) -> "custom_types.Integer":
        """Get the assignment depth for Stan loop nesting.

        :returns: Loop nesting level for this component
        :rtype: custom_types.Integer

        This is the property form of
        :py:meth:`~scistanpy.model.components.abstract_model_component.AbstractModelComponent.get_assign_depth`
        and provides convenient access to the assignment depth. It determines how
        deeply nested this component should be in Stan's loop structure.
        """
        return self.get_assign_depth()

    @property
    def model_varname(self) -> str:
        """Get or generate the SciStanPy variable name for this component. Only
        valid within a SciStanPy Model.

        :returns: Variable name for this component
        :rtype: str

        If a variable name has been explicitly assigned, returns that name.
        Otherwise, automatically generates a name based on child component
        relationships using dot notation for hierarchical names.

        Examples:
            .. code-block:: python

                class MyModel(Model):
                    def __init__(self):
                        super().__init__()

                        # Parameter without explicit name has auto-generated name
                        # Name is "param2.mu" based on child relationship
                        param1 = Parameter(...)  # model_varname is "param2.mu"

                        # Explicitly named parameter
                        param2 = Parameter(mu = self.param1) # model_varname is "param2"
        """
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
                if not isinstance(child, transformed_data.TransformedData)
            ]
        )

    @model_varname.setter
    def model_varname(self, name: str) -> None:
        """Set the SciStanPy variable name for this component.

        :param name: Variable name to assign
        :type name: str

        :raises ValueError: If attempting to rename an already-named component

        Variable names can only be set once to prevent accidental overwrites
        that could break model consistency. These are set automatically when parameters
        are defined inside the `__init__` method of a SciStanPy Model subclass.
        """
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
        """Get the Stan-compatible variable name for this component. This is identical
        to :py:attr:`~scistanpy.model.components.abstract_model_component.AbstractModelComponent.model_varname`,
        but with dots replaced by double underscores.

        :returns: Stan variable name with dots replaced by double underscores
        :rtype: str
        """
        return self.model_varname.replace(".", "__")

    @property
    def constants(self):
        """Get all constant-valued parent components.

        :returns: Dictionary mapping parameter names to constant components
        :rtype: dict[str, Constant]

        This property filters parent components to return only those that
        are :py:class:`~scistanpy.model.components.constants.Constant` instances,
        useful for identifying fixed values in the model hierarchy.
        """
        return {
            name: component
            for name, component in self._parents.items()
            if isinstance(component, constants_module.Constant)
        }

    @property
    def parents(self) -> list["AbstractModelComponent"]:
        """Get all parent components that this component depends on.

        :returns: List of parent components
        :rtype: list[AbstractModelComponent]
        """
        return list(self._parents.values())

    @property
    def children(self) -> list["AbstractModelComponent"]:
        """Get all child components that depend on this component.

        :returns: Copy of the children list
        :rtype: list[AbstractModelComponent]

        Returns a shallow copy to prevent external modification of the internal
        ``_children`` list while allowing iteration and inspection.
        """
        return self._children.copy()

    @property
    @abstractmethod
    def torch_parametrization(self) -> torch.Tensor:
        """Get PyTorch tensor representation with appropriate transformations.

        :returns: PyTorch tensor for this component
        :rtype: torch.Tensor

        This abstract property must be implemented by all concrete components
        to provide PyTorch tensor representations suitable for gradient-based
        computation and optimization.
        """

    @property
    def observable(self) -> bool:
        """Check whether this component represents observed data.

        :returns: False by default (most components are not observable)
        :rtype: bool

        Observable components represent known data values rather than
        parameters to be inferred. The default implementation returns
        ``False``; subclasses may override for specific behavior.
        """
        return False

    @property
    def force_name(self) -> bool:
        """Check whether this component should be explicitly named in Stan code.

        :returns: ``True`` if any child forces parent naming
        :rtype: bool

        This property returns ``True`` if any child component has
        :py:attr:`~scistanpy.model.components.abstract_model_component.AbstractModelComponent.FORCE_PARENT_NAME`
        set to ``True``, indicating that this component should be given an explicit
        variable name in the generated Stan code rather than being inlined.
        """
        return any(child.FORCE_PARENT_NAME for child in self._children)
