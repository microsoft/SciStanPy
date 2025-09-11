# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Stan probabilistic programming language integration and code generation.

This module provides the core functionality for translating SciStanPy models into
Stan probabilistic programming language code and managing the complete Stan
compilation and execution workflow. It handles automatic code generation,
compilation management, and provides enhanced interfaces to Stan's sampling
algorithms.

The module implements a code generation system that organizes SciStanPy model components
into proper Stan program structure, including automatic handling of dependency
relationships, loop optimization, and efficient Stan code patterns.

Users will not normally interact with this module directly. Instead, they will either
(1) use the :py:meth:`Model.to_stan() <scistanpy.model.model.Model.to_stan>` method
to convert a SciStanPy model to a :py:class:`~scistanpy.model.stan.stan_model.StanModel`
instance or (2) use this module implicitly when fitting a SciStanPy model via the
:py:meth:`Model.mcmc() <scistanpy.model.model.Model.mcmc>` method.
"""

from __future__ import annotations

import functools
import os.path
import warnings
import weakref

from abc import ABC, abstractmethod
from collections import Counter
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Callable,
    Generator,
    Literal,
    Optional,
    ParamSpec,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import numpy as np
import numpy.typing as npt

from cmdstanpy import CmdStanModel, format_stan_file

import scistanpy

from scistanpy import utils
from scistanpy.defaults import (
    DEFAULT_CPP_OPTIONS,
    DEFAULT_FORCE_COMPILE,
    DEFAULT_MODEL_NAME,
    DEFAULT_INDEX_ORDER,
    DEFAULT_STANC_OPTIONS,
    DEFAULT_USER_HEADER,
)
from scistanpy.model import stan
from scistanpy.model.components import abstract_model_component, constants, parameters
from scistanpy.model.components.transformations import (
    transformed_data,
    transformed_parameters,
)

if TYPE_CHECKING:
    from scistanpy import custom_types

results = utils.lazy_import("scistanpy.model.results")

# pylint: disable=too-many-lines

# Function for combining a list of Stan code lines
DEFAULT_INDENTATION = 4

# Parameter and return types for decorated functions
P = ParamSpec("P")
R = TypeVar("R")


class StanCodeBase(ABC, list):
    """Abstract base class for Stan code organization and generation.

    This class provides the foundational infrastructure for organizing and
    generating Stan code from SciStanPy model components. It implements a
    hierarchical structure that mirrors the nested scope organization
    required by Stan programs.

    :param parent_loop: Parent code block in the hierarchy, or `None` for root
    :type parent_loop: Optional[StanCodeBase]

    :ivar parent_loop: Reference to parent code block for hierarchical organization

    The class serves as both a container for model components and for-loop
    constructs, and as a code generator that can produce appropriate Stan
    syntax for different program blocks (model, transformed parameters, etc.).

    Key Features:

    - Hierarchical organization of code blocks and loops
    - Automatic Stan syntax generation for different program sections
    - Component filtering and organization based on block requirements
    - Proper indentation and formatting management
    - Loop optimization and combination capabilities
    """

    def __init__(self, parent_loop: Optional["StanCodeBase"]):
        """Initializes the Stan code base."""
        # Initialize the list
        super().__init__()

        # Record the parent loop
        self.parent_loop = parent_loop

    def recurse_for_loops(self) -> Generator["StanForLoop", None, None]:
        """Generate all for-loops in the program hierarchy.

        This method recursively traverses the code structure to yield all
        for-loop constructs, enabling comprehensive analysis and optimization
        of the loop structure.

        :yields: All StanForLoop instances in the hierarchy
        :rtype: Generator[StanForLoop, None, None]

        The traversal follows a depth-first pattern, yielding the current
        level before recursing into nested loops, ensuring proper processing
        order for loop optimization procedures.
        """
        # Yield this loop
        yield self

        # Loop over all nested loops and yield from them as well
        for loop in self.nested_loops:
            yield from loop.recurse_for_loops()

    def recurse_model_components(
        self,
    ) -> Generator[abstract_model_component.AbstractModelComponent, None, None]:
        """Recursively generate all model components in the program.

        This method traverses the code structure to yield all SciStanPy
        model components.

        :yields: All model components in the code hierarchy
        :rtype: Generator[AbstractModelComponent, None, None]

        This is essential for global analysis of model structure, dependency
        validation, and code generation across all program blocks.
        """
        for component in self:
            if isinstance(component, StanForLoop):
                yield from component.recurse_model_components()
            else:
                assert isinstance(
                    component,
                    abstract_model_component.AbstractModelComponent,
                )
                yield component

    @abstractmethod
    def get_parent_loop(self, n: custom_types.Integer) -> "StanCodeBase":
        """Get ancestor loop at specified level in the hierarchy.

        This method navigates the loop hierarchy to retrieve a specific
        ancestor loop.

        :param n: Level in ancestry (0=root, -1=current)
        :type n: custom_types.Integer

        :returns: Ancestor loop at the specified level
        :rtype: StanCodeBase

        The method supports both positive indices (from root) and negative
        indices (from current level) for flexible hierarchy navigation.
        """

    def _write_block(
        self,
        block_name: Literal[
            "model",
            "transformed_parameters",
            "generated_quantities",
            "transformed_data",
        ],
        declarations: Union[str, tuple[str, ...]] = (),
    ) -> str:
        """Generate Stan code for a specific program block.

        This method orchestrates the generation of complete Stan program
        blocks by filtering appropriate components, combining for-loops,
        and applying proper formatting and structure.

        :param block_name: Name of the Stan block to generate
        :type block_name: Literal["model", "transformed_parameters", "generated_quantities",
            "transformed_data"]
        :param declarations: Variable declarations for the block header. Defaults to empty.
        :type declarations: Union[str, tuple[str, ...]]

        :returns: Complete Stan code for the specified block
        :rtype: str

        The method handles:

        - Component filtering based on block requirements
        - For-loop combination and optimization
        - Proper Stan syntax generation and formatting
        - Empty block handling and omission
        """

        def filter_generated_quantities(
            nested_component: Union[
                StanCodeBase, abstract_model_component.AbstractModelComponent
            ],
        ) -> bool:
            """
            Filters hierarchy of loops for the generated quantities block. We take
            observables.
            """
            return (
                isinstance(nested_component, parameters.Parameter)
                and nested_component.observable
            ) or isinstance(nested_component, StanForLoop)

        # Functions for filtering the tree
        def filter_model_transformed_params(
            nested_component: Union[
                StanCodeBase, abstract_model_component.AbstractModelComponent
            ],
        ) -> bool:
            """
            Filters hierarchy of loops for the model and transformed parameters
            blocks. We take Parameters and named TransformedParameters.
            """
            return isinstance(
                nested_component, (parameters.Parameter, StanForLoop)
            ) or (
                isinstance(
                    nested_component, transformed_parameters.TransformedParameter
                )
                and (nested_component.is_named or nested_component.force_name)
            )

        def filter_transformed_data(
            nested_component: Union[
                StanCodeBase, abstract_model_component.AbstractModelComponent
            ],
        ) -> bool:
            """
            Filters hierarchy of loops for the transformed data block. We take
            TransformedData only.
            """
            return isinstance(
                nested_component,
                (transformed_data.TransformedData, StanForLoop),
            )

        def filter_components() -> (
            list[Union[StanCodeBase, abstract_model_component.AbstractModelComponent]]
        ):
            """Filters components to those we want and combines for-loops."""
            # First combine for-loops at this level that are identical. First we filter.
            # Then, we iterate over and check for for loops with the same start and end
            # values that are next to one another and share the same parent loop. If
            # they do, we combine them.
            filtered_components = []
            prev_loop: Optional[StanForLoop] = None
            for current_component in filter(filter_func, self):

                # If the current component has no assignments, skip it. It has
                # no impact on this block.
                if getattr(current_component, func)(self.allowed_index_names) == "":
                    continue

                # If this is a for loop, perform a series of checks to see if we can
                # combine it with the previous loop (if there was one).
                if isinstance(current_component, StanForLoop):

                    # Combine with the previous for-loop if it is compatible
                    if (
                        prev_loop is not None
                        and current_component.end == prev_loop.end
                        and current_component.depth == prev_loop.depth
                    ):

                        # Extend a copy of the previous loop with the current component
                        combined_loop = prev_loop.copy()
                        combined_loop.extend(current_component)

                        # Set the previous loop to the combined loop
                        filtered_components[-1] = combined_loop
                        prev_loop = combined_loop
                        continue

                    # If not compatible, just set the previous loop to this one
                    else:
                        prev_loop = current_component

                # If not a for-loop, there is no previous loop
                else:
                    prev_loop = None

                # Record the current component
                filtered_components.append(current_component)

            return filtered_components

        # We need a dictionary that will map from block name to the prefix for the
        # block we are writing,
        dispatcher = {
            "model": {
                "func": "get_target_incrementation",
                "prefix": self.target_inc_prefix,
                "filter": filter_model_transformed_params,
            },
            "transformed_parameters": {
                "func": "get_transformation_assignment",
                "prefix": self.transformation_assi_prefix,
                "filter": filter_model_transformed_params,
            },
            "generated_quantities": {
                "func": "get_generated_quantities",
                "prefix": self.generated_quantities_prefix,
                "filter": filter_generated_quantities,
            },
            "transformed_data": {
                "func": "get_transformed_data_assignment",
                "prefix": self.transformed_data_prefix,
                "filter": filter_transformed_data,
            },
        }

        # Get the function, prefix, and filter for the block
        func = dispatcher[block_name]["func"]
        prefix = dispatcher[block_name]["prefix"]
        filter_func = dispatcher[block_name]["filter"]

        # Get assignments and incrementations
        assignments = [
            assignment
            for component in filter_components()
            if (assignment := getattr(component, func)(self.allowed_index_names))
        ]

        # Null string if no assignments
        if len(assignments) == 0:
            return ""

        # Get the number of non-for-loops in the program.
        n_model_components = len(self.model_components)

        # Otherwise, combine lines, add a prefix, and finalize the line
        return (
            "\n"
            + prefix
            + ("\n" + declarations + "\n" if isinstance(declarations, str) else "")
            + ("\n" if n_model_components > 0 else "")
            + self.combine_lines(assignments, indentation_level=self.depth + 1)
            + "\n"
            + self.finalize_line("}")
        )

    def get_target_incrementation(
        self, _dummy: Optional[tuple[str, ...]] = None
    ) -> str:
        """Generate Stan code for model block `target` variable incrementation.

        :param _dummy: Unused parameter for interface compatibility. Defaults to None.
        :type _dummy: Optional[tuple[str, ...]]

        :returns: Stan code for model block.
        :rtype: str

        This method generates the model block containing all log-probability (target)
        increment statements for parameters and observables.
        """
        return self._write_block("model")

    def get_transformation_assignment(
        self, declarations: Union[str, tuple[str, ...]]
    ) -> str:
        """Generate Stan code for transformed parameters block.

        :param declarations: Variable declarations for the block
        :type declarations: Union[str, tuple[str, ...]]

        :returns: Stan code for transformed parameters block
        :rtype: str

        Creates the transformed parameters block containing all deterministic
        transformations.
        """
        return self._write_block("transformed_parameters", declarations=declarations)

    def get_generated_quantities(
        self, declarations: Union[str, tuple[str, ...]]
    ) -> str:
        """Generate Stan code for generated quantities block.

        :param declarations: Variable declarations for the block
        :type declarations: Union[str, tuple[str, ...]]

        :returns: Stan code for generated quantities block
        :rtype: str

        Creates the generated quantities block for posterior predictive
        sampling and derived quantity computation.
        """
        return self._write_block("generated_quantities", declarations=declarations)

    def get_transformed_data_assignment(
        self, declarations: Union[str, tuple[str, ...]]
    ) -> str:
        """Generate Stan code for transformed data block.

        :param declarations: Variable declarations for the block
        :type declarations: Union[str, tuple[str, ...]]

        :returns: Stan code for transformed data block
        :rtype: str

        Creates the transformed data block for preprocessing and
        deterministic data transformations.
        """
        return self._write_block("transformed_data", declarations=declarations)

    def finalize_line(
        self, text: str, indendation_level: Optional[custom_types.Integer] = None
    ) -> str:
        """Apply proper indentation and Stan syntax formatting to code lines.

        :param text: Raw code text to format
        :type text: str
        :param indendation_level: Indentation level (uses current depth if None).
                                 Defaults to None.
        :type indendation_level: Optional[custom_types.Integer]

        :returns: Properly formatted Stan code line
        :rtype: str

        This method handles:

        - Consistent indentation based on scope depth
        - Automatic semicolon insertion for statements
        - Proper formatting for control structures and comments
        """
        # Get the indentation level
        indendation_level = (
            self.depth if indendation_level is None else indendation_level
        )

        # Pad the input text with spaces
        formatted = f"{' ' * DEFAULT_INDENTATION * indendation_level}{text}"

        # Add a semicolon to the end if not a bracket or blank
        if text and text[-1] not in {"{", "}", ";"} and not text.startswith("#"):
            formatted += ";"

        return formatted

    def combine_lines(
        self, lines: list[str], indentation_level: Optional[custom_types.Integer] = None
    ) -> str:
        """Combine multiple Stan code lines with proper formatting.

        :param lines: List of code lines to combine
        :type lines: list[str]
        :param indentation_level: Indentation level for all lines. Defaults to None.
        :type indentation_level: Optional[custom_types.Integer]

        :returns: Combined and formatted Stan code block
        :rtype: str

        Applies consistent formatting and indentation to all provided lines,
        creating a properly structured Stan code block.
        """

        # Nothing if no lines
        if len(lines) == 0:
            return ""

        # Combine the lines
        return "\n".join(
            self.finalize_line(el, indendation_level=indentation_level) for el in lines
        )

    @property
    def nested_loops(self):
        """Get all nested for-loops at this level.

        :returns: List of direct child for-loops
        :rtype: list[StanForLoop]
        """
        return [component for component in self if isinstance(component, StanForLoop)]

    @property
    def model_components(
        self,
    ) -> list[abstract_model_component.AbstractModelComponent]:
        """Get all model components at this level.

        :returns: List of direct child model components
        :rtype: list[AbstractModelComponent]
        """
        return [
            component
            for component in self
            if isinstance(component, abstract_model_component.AbstractModelComponent)
        ]

    @property
    @abstractmethod
    def depth(self) -> "custom_types.Integer":
        """Get the nesting depth of this code block.

        :returns: Depth level in the code hierarchy
        :rtype: custom_types.Integer
        """

    @property
    @abstractmethod
    def allowed_index_names(self) -> tuple[str, ...]:
        """Get allowed index variable names for this scope.

        :returns: Tuple of allowed index variable names
        :rtype: tuple[str, ...]
        """

    @property
    @abstractmethod
    def target_inc_prefix(self) -> str:
        """Get the prefix for model block target incrementation.

        :returns: Stan code prefix for model block
        :rtype: str
        """

    @property
    @abstractmethod
    def transformation_assi_prefix(self) -> str:
        """Get the prefix for transformed parameters block.

        :returns: Stan code prefix for transformed parameters block
        :rtype: str
        """

    @property
    @abstractmethod
    def generated_quantities_prefix(self) -> str:
        """Get the prefix for generated quantities block.

        :returns: Stan code prefix for generated quantities block
        :rtype: str
        """

    @property
    @abstractmethod
    def transformed_data_prefix(self) -> str:
        """Get the prefix for transformed data block.

        :returns: Stan code prefix for transformed data block
        :rtype: str
        """


class StanForLoop(StanCodeBase):
    """Represents a Stan for-loop construct with optimization capabilities.

    This class manages individual for-loop constructs in Stan code, including
    automatic loop range determination, optimization through combination with
    compatible loops, and proper nesting within the code hierarchy.

    :param parent_loop: Parent code block containing this loop
    :type parent_loop: StanCodeBase
    :param record_in_parent: Whether to automatically add this loop to parent. Defaults to True.
    :type record_in_parent: bool

    :ivar parent_loop: Reference to parent code block

    Key Features:

    - Automatic loop range calculation based on component dimensions
    - Loop combination optimization for compatible adjacent loops
    - Singleton loop detection and elimination
    - Proper index variable management and scope resolution
    - Hierarchical ancestry tracking for scope navigation

    The class automatically determines loop ranges by analyzing the dimensions
    of nested model components and provides optimization features like loop
    combination to generate efficient Stan code.
    """

    def __init__(self, parent_loop: StanCodeBase, record_in_parent: bool = True):
        """Initialize the for-loop.

        Args:
            index: The index variable for the loop
            start: The starting value of the loop
            end: The ending value of the loop
        """
        # Initialize the list
        super().__init__(parent_loop)

        # Append this loop to the parent loop if requested
        if record_in_parent:
            self.parent_loop.append(self)

    def _get_ancestry(self) -> list[StanCodeBase]:
        """Retrieve the complete ancestry chain of this loop.

        :returns: List of ancestor code blocks from root to immediate parent
        :rtype: list[StanCodeBase]

        This method traces the hierarchy from the current loop back to the
        root program, enabling proper scope resolution and code organization.
        """

        # If the parent is the program, we are at the top level and have only the
        # program as our ancestor
        if isinstance(self.parent_loop, StanProgram):
            return [self.parent_loop]

        # Otherwise, we have the parent loop and all of its ancestors
        return self.parent_loop.ancestry + [self.parent_loop]

    def get_parent_loop(self, n: custom_types.Integer) -> StanCodeBase:
        """Get ancestor loop at specified level in the hierarchy.

        :param n: Ancestry level to retrieve (clipped to available range). For example,
            `0` returns the root loop; `-1` returns this loop.
        :type n: custom_types.Integer

        :returns: Ancestor loop at the specified level
        :rtype: StanCodeBase
        """
        # Clip n to the number of ancestors
        n = min(n, len(self.ancestry))

        # Get the appropriate loop
        return (self.ancestry + [self])[n]

    def append(
        self,
        component: Union[
            "StanForLoop", abstract_model_component.AbstractModelComponent
        ],
    ) -> None:
        """Add a component to this loop with depth validation.

        :param component: Component to add to the loop
        :type component: Union[StanForLoop, AbstractModelComponent]

        :raises AssertionError: If component depth doesn't match loop depth

        Ensures that only components with appropriate nesting depth are
        added to maintain proper Stan code structure.
        """
        # If a model component, make sure we are at the appropriate code level
        if isinstance(component, abstract_model_component.AbstractModelComponent):
            assert component.assign_depth == self.depth

        # Append the component
        super().append(component)

    def copy(self, record_in_parent: bool = False) -> "StanForLoop":
        """Create a shallow copy of this loop.

        :param record_in_parent: Whether to register copy with parent. Defaults to False.
        :type record_in_parent: bool

        :returns: New loop instance with same contents
        :rtype: StanForLoop

        Creates a copy with the same parent loop and components, useful for
        loop combination and optimization operations.
        """
        # Create a new loop with the same parent loop
        new_loop = StanForLoop(
            parent_loop=self.parent_loop, record_in_parent=record_in_parent
        )

        # Populate the new loop with the same components
        new_loop.extend(self)

        return new_loop

    def squash(self) -> None:
        """Remove singleton loops by moving contents to parent.

        This optimization method detects when a loop has only one iteration
        (singleton) and eliminates the unnecessary loop construct by moving
        its contents directly to the parent scope.
        """
        # If the end is 1, we are a singleton loop. Move the contents of this loop
        # to the parent loop and then remove this loop from the parent loop.
        if self.end == 1:
            self.parent_loop.extend(self)
            self.parent_loop.remove(self)

    @property
    def ancestry(self):
        """Get the complete ancestry chain of this loop.

        :returns: List of ancestor code blocks from root to immediate parent
        :rtype: list[StanCodeBase]
        """
        return self._get_ancestry()

    @property
    def end(self) -> "custom_types.Integer":
        """Calculate the end value (iteration count) for this loop.

        :returns: Number of iterations for this loop
        :rtype: custom_types.Integer

        :raises ValueError: If components have incompatible dimension sizes

        Automatically determines loop range by analyzing the dimensions of
        all nested model components at the appropriate depth level. Handles
        dimension compatibility checking and singleton dimension filtering.
        """
        # Get the size of the dimension at the index level for all model components
        # nested in the loop
        all_ends = {
            component.shape[self.depth - 1]
            for component in self.recurse_model_components()
        }

        # If there are multiple options, remove 1s
        if len(all_ends) > 1:
            all_ends.discard(1)

        # If there are still multiple options, raise an error
        if len(all_ends) > 1:
            raise ValueError(f"Invalid end values: {all_ends}")

        # The value must be 1 or greater
        val = all_ends.pop()
        assert val >= 1

        return val

    @property
    def program(self) -> "StanProgram":
        """Get the root Stan program containing this loop.

        :returns: Root StanProgram instance
        :rtype: StanProgram
        """
        return self.ancestry[0]

    @property
    def allowed_index_names(self) -> tuple[str, ...]:
        """Get allowed index variable names from the root program.

        :returns: Tuple of allowed index variable names
        :rtype: tuple[str, ...]
        """
        return self.program.allowed_index_names

    @property
    def depth(self) -> "custom_types.Integer":
        """Get the nesting depth of this loop. This is equivalent to the number
        of ancestor loops.

        :returns: Depth level based on number of ancestors
        :rtype: custom_types.Integer
        """
        n_ancestors = len(self.ancestry)
        assert n_ancestors > 0
        return n_ancestors

    @property
    def loop_index(self) -> str:
        """Get the index variable name for this loop (e.g., 'i' if loop construct
        is written as `for (i in 1:10) {`).

        :returns: Index variable name based on depth
        :rtype: str
        """
        return self.program.allowed_index_names[self.depth - 1]

    @property
    def target_inc_prefix(self) -> str:
        """Get Stan code prefix for model block loops.

        :returns: Stan for-loop syntax for model block
        :rtype: str
        """
        return self.finalize_line("") + f"for ({self.loop_index} in 1:{self.end}) {{"

    @property
    def transformation_assi_prefix(self) -> str:
        """Get Stan code prefix for transformed parameters block loops.

        :returns: Stan for-loop syntax for transformed parameters block
        :rtype: str
        """
        return self.target_inc_prefix

    @property
    def generated_quantities_prefix(self) -> str:
        """Get Stan code prefix for generated quantities block loops.

        :returns: Stan for-loop syntax for generated quantities block
        :rtype: str
        """
        return self.target_inc_prefix

    @property
    def transformed_data_prefix(self) -> str:
        """Get Stan code prefix for transformed data block loops.

        :returns: Stan for-loop syntax for transformed data block
        :rtype: str
        """
        return self.target_inc_prefix


class StanProgram(StanCodeBase):
    """Complete Stan program generation and management.

    This class orchestrates the generation of complete Stan programs from
    SciStanPy models, handling dependency analysis, component organization,
    and the generation of all required Stan program blocks.

    :param model: SciStanPy model to convert to Stan
    :type model: scistanpy.Model

    :ivar model: Reference to the source SciStanPy model
    :ivar node_to_depth: Mapping from components to their hierarchy depth
    :ivar all_varnames: Set of all variable names in the program
    :ivar all_paramnames: Set of all parameter names
    :ivar autogathered_varnames: Set of variables whose data will be automatically
        gathered from the SciStanPy model.
    :ivar user_provided_varnames: Set of variables whose data the user must provide
        (i.e., the names of observable parameters).

    The class performs comprehensive analysis of the SciStanPy model to:

    - Build dependency graphs and determine component ordering
    - Generate appropriate variable names avoiding conflicts
    - Organize components into proper Stan program structure
    - Create optimized loop constructs for multi-dimensional components
    - Generate all required Stan program blocks with proper syntax

    The code is generated following the below procedure:

    1. Dependency Analysis: Build component dependency graph
    2. Depth Assignment: Determine nesting levels for components
    3. Loop Organization: Create optimized for-loop structures
    4. Block Generation: Generate all Stan program blocks
    5. Code Formatting: Apply Stan canonical formatting
    """

    def __init__(self, model: "scistanpy.Model"):
        """Initializes and compiles the Stan program."""
        # Initialize the list
        super().__init__(None)

        # Note the model
        self.model = model

        # Build the map from model node to depth in the tree of nodes
        self.node_to_depth: dict[
            abstract_model_component.AbstractModelComponent, int
        ] = self.build_node_to_depth()

        # Get the names of all variables, parameters, and data inputs for the model
        (
            self.all_varnames,
            self.all_paramnames,
            self.autogathered_varnames,
            self.user_provided_varnames,
        ) = self.get_varnames()

        # Get allowed index variable names for each level
        self._allowed_index_names = tuple(
            char
            for char in DEFAULT_INDEX_ORDER
            if {char, char.upper()}.isdisjoint(self.all_varnames)
        )

        # Compile the program
        self.compile()

    def build_node_to_depth(
        self,
    ) -> dict[abstract_model_component.AbstractModelComponent, int]:
        """Build mapping from model components to their maximum hierarchy depth.

        :returns: Dictionary mapping components to their depth levels
        :rtype: dict[AbstractModelComponent, int]

        This method analyzes the SciStanPy model's dependency graph to determine
        the maximum depth of each component relative to the root constants.
        This information is crucial for proper code ordering and loop organization.
        """
        # We need a mapping from each node to its maximum depth in the tree.
        node_to_depth: dict[
            abstract_model_component.AbstractModelComponent, custom_types.Integer
        ] = {}

        # Get all constants, named or otherwise
        model_constants = list(
            filter(
                lambda x: isinstance(x, constants.Constant),
                self.model.all_model_components,
            )
        )

        # Starting from constants (which are all root nodes), walk down the tree
        # and record the depth of each node
        for root in model_constants:

            # Root nodes are at depth 0
            node_to_depth[root] = 0

            # Walk the tree rooted at this root and record the depth of each node
            # if it is higher than the current maximum depth
            for depth, _, child in root.walk_tree():

                # Record the maximum depth of this node
                node_to_depth[child] = max(depth, node_to_depth.get(child, 0))

        return node_to_depth

    def get_varnames(self) -> tuple[set[str], set[str], set[str], set[str]]:
        """Extract and categorize all variable names from the model.

        :returns: Tuple of (all_varnames, all_paramnames, autogathered_varnames,
            user_provided_varnames)
        :rtype: tuple[set[str], set[str], set[str], set[str]]

        :raises ValueError: If variable name collisions are detected

        Analyzes the model to categorize variables into:
        - All variable names (for conflict detection)
        - Parameter names (for Stan parameters block)
        - Auto-gathered names (constants from model)
        - User-provided names (observables requiring external data)
        """
        # Get the names of all the variables in the model
        name_counts = Counter(node.model_varname for node in self.node_to_depth)
        if collisions := {name for name, count in name_counts.items() if count > 1}:
            raise ValueError(f"Name collisions for variables: {collisions}")

        # Get all variable names and all parameter names
        all_varnames = {node.model_varname for node in self.node_to_depth}
        all_paramnames = {
            node.model_varname
            for node in self.node_to_depth
            if isinstance(node, parameters.Parameter) and not node.observable
        }

        # Get all data inputs
        auto_gathered_data = {
            node.model_varname
            for node in self.node_to_depth
            if isinstance(node, constants.Constant)
        }
        user_provided_varnames = {
            node.model_varname
            for node in self.node_to_depth
            if isinstance(node, parameters.Parameter) and node.observable
        }

        return all_varnames, all_paramnames, auto_gathered_data, user_provided_varnames

    def get_parent_loop(self, n: custom_types.Integer) -> StanCodeBase:
        """Get parent loop (only self is available for root program).

        :param n: Must be -1 or 0 for root program
        :type n: custom_types.Integer

        :returns: Self (root program)
        :rtype: StanCodeBase

        :raises AssertionError: If n is not -1 or 0
        """
        # Can only get self
        assert n == -1 or n == 0
        return self

    def compile(self) -> None:
        """Organize model components into proper Stan program structure.

        This method performs the core compilation process:
        1. Determines topological ordering of components based on dependencies
        2. Organizes components into appropriate loop structures
        3. Creates and optimizes for-loop constructs
        4. Eliminates singleton loops for efficiency

        The compilation process ensures that all dependencies are satisfied
        and that the resulting Stan code is properly structured and efficient.
        """
        # Get the order of operations. This is a list of nodes sorted by their maximum
        # depth in the tree
        order_of_operations: list[abstract_model_component.AbstractModelComponent] = (
            sorted(
                self.node_to_depth,
                key=lambda x: (self.node_to_depth[x], x.assign_depth),
            )
        )

        # The first component in the order of operations must be at definition depth 0
        assert order_of_operations[0].assign_depth == 0

        # Set up for compilation. We define the `target_loop`, which is the loop
        # to which components will be actively added. We also defined the `previous_component`,
        # which is the component one operation back.
        target_loop: StanCodeBase = self
        previous_component = None  # Item one operation back
        for component in order_of_operations:

            # If the current or previous component is defined a depth 0, then the
            # program is the target loop. We will need to add as many for-loops
            # as we have levels of indentation in the current component.
            if (
                component.assign_depth == 0
                or previous_component.assign_depth == 0
                or component.FORCE_LOOP_RESET
            ):
                target_loop = self
                n_loops = component.assign_depth

            # In any other case, we need to determine which dimensions of the current
            # component are compatible with the previous component and find the
            # appropriate parent loop.
            else:

                # We need to determine the extent to which the shapes of the current
                # and previous components are compatible. This is the number of
                # shared leading dimensions between the two components' shapes.
                compat_level = previous_component.get_shared_leading(component)
                assert compat_level >= 0

                # Get the parent loop of the current component that is compatible
                # with the the current component. Note that we also update our referenced
                # loop in this step. Note that the target loop must be a for-loop
                # at this point as the previous `if` block handles all cases where
                # the target loop is the program.
                assert isinstance(target_loop, StanForLoop)
                target_loop = target_loop.get_parent_loop(compat_level)

                # How many for-loops do we need to add? The definition depth of
                # the current component must be greater than or equal to the depth
                # of the current loop
                n_loops = component.assign_depth - target_loop.depth
                assert n_loops >= 0

            # Add for-loops if necessary
            for _ in range(n_loops):
                target_loop = StanForLoop(parent_loop=target_loop)

            # Add the component to the target loop and update the previous component
            target_loop.append(component)
            previous_component = component

        # Squash any singletons
        for loop in self.recurse_for_loops():
            loop.squash()

        # Remove all squashed loops
        self[:] = [
            component
            for component in self
            if not isinstance(component, StanForLoop)
            or component.parent_loop is not None
        ]

    def recurse_for_loops(self) -> Generator[StanForLoop, None, None]:
        """Generate all for-loops in the program (excluding self).

        :yields: All StanForLoop instances in the program
        :rtype: Generator[StanForLoop, None, None]

        This method yields only the nested for-loops, not the program itself, which
        is different from the `StanForLoop` sibling class.
        """
        for loop in self.nested_loops:
            yield from loop.recurse_for_loops()

    def append(
        self,
        component: Union[
            "StanForLoop", abstract_model_component.AbstractModelComponent
        ],
    ) -> None:
        """Add component to program with depth validation.

        :param component: Component to add to the program
        :type component: Union[StanForLoop, AbstractModelComponent]

        :raises AssertionError: If component depth is not 0 (root level)

        Ensures only root-level components are added directly to the program.
        """
        # If a model component, make sure we are at the appropriate depth
        if isinstance(component, abstract_model_component.AbstractModelComponent):
            assert component.assign_depth == 0

        # Append the component
        super().append(component)

    @property
    def depth(self) -> "custom_types.Integer":
        """Get the depth of the program (always 0 for root).

        :returns: Depth level (always 0)
        :rtype: custom_types.Integer
        """
        return 0

    @property
    def functions_block(self) -> str:
        """Generate Stan functions block with custom function includes.

        :returns: Stan functions block code or empty string if no functions needed
        :rtype: str

        Automatically determines required supporting functions from model
        components and generates appropriate include statements and function
        definitions for the Stan functions block.
        """
        # Get all model components.
        model_components = list(self.model.all_model_components)

        # If there is a MultinomialLogit component, remove all Multinomial components.
        # This is because the MultinomialLogit component will include the Multinomial
        # functions, so we don't need to include them again.
        if any(
            isinstance(component, parameters.MultinomialLogit)
            for component in model_components
        ):
            model_components = [
                component
                for component in model_components
                if not isinstance(component, parameters.Multinomial)
            ]

        # Get all supporting functions
        supporting_functions = []
        for component in model_components:
            supporting_functions.extend(component.get_supporting_functions())

        # There is no need to include a functions block if there are no functions
        if len(supporting_functions) == 0:
            return ""

        # No duplicates. Include statements first. Alphabetical after that.
        supporting_functions = sorted(
            set(supporting_functions),
            key=lambda x: (x, x.startswith("#include")),
        )

        # Otherwise, we need to combine the lines and add a functions block
        return (
            "functions {\n"
            + self.combine_lines(supporting_functions, indentation_level=1)
            + "\n}"
        )

    @property
    def data_block(self) -> str:
        """Generate Stan data block with observable and constant declarations.

        :returns: Stan data block code
        :rtype: str

        Creates the data block containing declarations for all observables
        (user-provided data) and constants (auto-gathered from model).
        """
        # Get the declarations for the data block. This is all observables and all
        # constants.
        declarations = [
            component.get_stan_parameter_declaration()
            for component in self.model.all_model_components
            if (isinstance(component, parameters.Parameter) and component.observable)
            or isinstance(component, constants.Constant)
        ]

        # Combine declarations and wrap in the data block
        return (
            "data {\n" + self.combine_lines(declarations, indentation_level=1) + "\n}"
        )

    @property
    def transformed_data_block(self) -> str:
        """Generate Stan transformed data block.

        :returns: Stan transformed data block code or empty string if not needed
        :rtype: str

        Creates the transformed data block for any deterministic data
        transformations required by the model.
        """
        # Check parameters for any transformed data.
        declarations = [
            component.get_stan_parameter_declaration()
            for component in self.model.all_model_components
            if isinstance(component, transformed_data.TransformedData)
        ]

        # No transformed data if no declarations
        if len(declarations) == 0:
            return ""

        # Combine declarations
        declaration_block = self.combine_lines(declarations, indentation_level=1)

        # Combine declarations and transformations
        return self.get_transformed_data_assignment(declarations=declaration_block)

    @property
    def parameters_block(self) -> str:
        """Generate Stan parameters block with parameter declarations.

        :returns: Stan parameters block code
        :rtype: str

        Creates the parameters block containing all model parameters that
        will be sampled during MCMC.
        """
        # Loop over all components recursively and define the parameters
        declarations: list[str] = []
        for component in self.model.parameters:

            # If the component is defined in raw form, we declare the raw variable.
            # The true variable will be defined in the transformed parameters block.
            if component.HAS_RAW_VARNAME:
                declarations.append(component.get_raw_stan_parameter_declaration())

            # Otherwise, add regular declarations for parameters.
            else:
                declarations.append(component.get_stan_parameter_declaration())

        # Combine the lines and enclose in the parameters block
        return (
            "\n"
            + "parameters {\n"
            + self.combine_lines(declarations, indentation_level=1)
            + "\n}"
        )

    @property
    def model_block(self) -> str:
        """Generate Stan model block with log-probability statements.

        :returns: Stan model block code
        :rtype: str

        Creates the model block containing all target increment statements
        for priors and likelihoods.
        """
        return self.get_target_incrementation()

    @property
    def transformed_parameters_block(self) -> str:
        """Generate Stan transformed parameters block.

        :returns: Stan transformed parameters block code
        :rtype: str

        Creates the transformed parameters block for deterministic parameter
        transformations and derived quantities.
        """
        # Get the declarations for transformed parameters. We take any named transformed
        # that are named or indexed. We also take any Parameter that transforms
        # a raw to a real parameter
        declarations = [
            component.get_stan_parameter_declaration()
            for component in self.recurse_model_components()
            if (
                isinstance(component, transformed_parameters.TransformedParameter)
                and (component.is_named or component.force_name)
            )
            or (
                isinstance(component, parameters.Parameter)
                and component.HAS_RAW_VARNAME
            )
        ]

        # Combine declarations
        declaration_block = self.combine_lines(declarations, indentation_level=1)

        # Combine declarations and transformations
        return self.get_transformation_assignment(declarations=declaration_block)

    @property
    def generated_quantities_block(self) -> str:
        """Generate Stan generated quantities block.

        :returns: Stan generated quantities block code
        :rtype: str

        Creates the generated quantities block for posterior predictive
        sampling and derived quantity computation.
        """
        # Get declarations for the generated quantities block. This is all observables
        # in the program.
        declarations = [
            component.get_generated_quantity_declaration()
            for component in self.recurse_model_components()
            if isinstance(component, parameters.Parameter) and component.observable
        ]

        # Combine declarations
        declaration_block = self.combine_lines(declarations, indentation_level=1)

        # Combine declarations and transformations
        return self.get_generated_quantities(declarations=declaration_block)

    @property
    def allowed_index_names(self) -> tuple[str, ...]:
        """Get allowed index variable names avoiding conflicts.

        :returns: Tuple of allowed index variable names
        :rtype: tuple[str, ...]
        """
        return self._allowed_index_names

    @property
    def target_inc_prefix(self) -> str:
        """Get prefix for model block.

        :returns: Stan model block opening syntax
        :rtype: str
        """
        return "model {"

    @property
    def transformation_assi_prefix(self) -> str:
        """Get prefix for transformed parameters block.

        :returns: Stan transformed parameters block opening syntax
        :rtype: str
        """
        return "transformed parameters {"

    @property
    def generated_quantities_prefix(self) -> str:
        """Get prefix for generated quantities block.

        :returns: Stan generated quantities block opening syntax
        :rtype: str
        """
        return "generated quantities {"

    @property
    def transformed_data_prefix(self) -> str:
        """Get prefix for transformed data block.

        :returns: Stan transformed data block opening syntax
        :rtype: str
        """
        return "transformed data {"

    @property
    def code(self) -> str:
        """Generate complete Stan program code.

        :returns: Complete Stan program as formatted string
        :rtype: str

        Assembles all program blocks into a complete Stan program with
        proper formatting and structure.
        """
        # Join steps that have contents
        return "\n".join(
            val
            for val in (
                self.functions_block,
                self.data_block,
                self.transformed_data_block,
                self.parameters_block,
                self.transformed_parameters_block,
                self.model_block,
                self.generated_quantities_block,
            )
            if len(val.strip()) > 0
        )


def _update_cmdstanpy_func(func: Callable[P, R], warn: bool = False) -> Callable[P, R]:
    """Decorator to enhance CmdStanModel functions with automatic data gathering.

    This decorator modifies CmdStanModel functions to automatically gather
    required data from the SciStanPy model while requiring users to provide
    values only for observable parameters.

    :param func: CmdStanModel function to enhance
    :type func: Callable[P, R]
    :param warn: Whether to warn about experimental status. Defaults to False.
    :type warn: bool

    :returns: Enhanced function with automatic data gathering
    :rtype: Callable[P, R]

    The inner function handles:
    - Automatic seed generation from global RNG if not provided
    - Data gathering and validation from SciStanPy model
    - Warning messages for experimental functions
    - Proper argument handling and forwarding
    """

    @functools.wraps(func)
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        """Wrapper function for gathering inputs."""
        # If warning the user about lack of testing, do so
        if warn:
            warnings.warn(
                f"{func.__name__} is experimental and has not been thoroughly tested"
                " in the context of SciStanPy models. Use with caution."
            )

        # Get the Stan model from the first argument
        stan_model = args[0]
        assert isinstance(stan_model, StanModel)

        # Combine args and kwargs into a single dictionary
        kwargs.update(dict(zip(func.__code__.co_varnames[1:], args[1:])))

        # If a seed is not provided, use the global random number generator to get
        # one
        if kwargs.get("seed") is None:
            kwargs["seed"] = scistanpy.RNG.integers(0, 2**32 - 1)

        # `data` must be a key in the kwargs
        if "data" not in kwargs:
            raise ValueError(
                f"The 'data' keyword argument must be provided to {func.__name__}"
            )

        # Gather the inputs for the Stan model. The user should have provided
        # values for the observables. We will get the rest of the inputs from the
        # SciStanPy model.
        kwargs["data"] = stan_model.gather_inputs(**kwargs["data"])

        # Run the wrapped function
        return func(stan_model, **kwargs)

    return inner


class StanModel(CmdStanModel):
    """Enhanced CmdStanModel with SciStanPy integration.

    This class extends CmdStanPy's CmdStanModel to provide integration
    with SciStanPy models, including automatic Stan code generation, enhanced
    sampling interfaces, and comprehensive result processing.

    :param model: SciStanPy model to compile to Stan
    :type model: scistanpy.Model
    :param output_dir: Directory for Stan files and compilation. Defaults to None (temporary).
    :type output_dir: Optional[str]
    :param force_compile: Whether to force recompilation. Defaults to False.
    :type force_compile: bool
    :param stanc_options: Options for Stan compiler. Defaults to None (uses defaults).
    :type stanc_options: Optional[dict[str, Any]]
    :param cpp_options: Options for C++ compilation. Defaults to None (uses defaults).
    :type cpp_options: Optional[dict[str, Any]]
    :param user_header: Custom C++ header code. Defaults to None.
    :type user_header: Optional[str]
    :param model_name: Name for compiled model. Defaults to 'model'.
    :type model_name: str

    :ivar model: Reference to source SciStanPy model
    :ivar program: Generated StanProgram instance
    :ivar output_dir: Directory containing Stan files
    :ivar stan_executable_path: Path to compiled Stan executable

    Key Features:
    - Automatic Stan code generation from SciStanPy models
    - Enhanced sampling with prior initialization and validation
    - Automatic data gathering for observables and constants
    - Comprehensive result processing and structuring
    - Integration with ArviZ for Bayesian workflow support

    The class handles the complete workflow from SciStanPy model to Stan
    execution, providing an interface that abstracts away the complexities of Stan
    code generation and compilation management.
    """

    # We initialize with a SciStanPy model instance
    def __init__(
        self,
        model: "scistanpy.Model",
        output_dir: Optional[str] = None,
        force_compile: bool = DEFAULT_FORCE_COMPILE,
        stanc_options: Optional[dict[str, Any]] = None,
        cpp_options: Optional[dict[str, Any]] = None,
        user_header: Optional[str] = DEFAULT_USER_HEADER,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        # Set default options
        self._stanc_options = stanc_options or DEFAULT_STANC_OPTIONS
        cpp_options = cpp_options or DEFAULT_CPP_OPTIONS

        # Add the "include_paths" kwarg
        self._stanc_options["include-paths"] = (
            self._stanc_options.get("include-paths", []) + stan.STAN_INCLUDE_PATHS
        )

        # Note the underlying SciStanPy model
        self.model = model

        # Compile the program
        self.program = StanProgram(model)

        # Set the output directory
        self._set_output_dir(output_dir)

        # Get the model name
        self.stan_executable_path = os.path.join(self.output_dir, model_name)

        # Write the Stan program
        self.write_stan_program()

        # Initialize the CmdStanModel
        super().__init__(
            stan_file=self.stan_program_path,
            exe_file=(
                self.stan_executable_path
                if os.path.exists(self.stan_executable_path) and not force_compile
                else None
            ),
            force_compile=force_compile,
            stanc_options=self._stanc_options,
            cpp_options=cpp_options,
            user_header=user_header,
        )

    def _set_output_dir(self, output_dir: Optional[str]) -> None:
        """Configure output directory with automatic cleanup for temporary directories.

        :param output_dir: Directory path or None for temporary directory
        :type output_dir: Optional[str]

        :raises FileNotFoundError: If specified directory doesn't exist

        Sets up the output directory for Stan files, creating a temporary
        directory with automatic cleanup if none is specified.
        """
        # Make a temporary directory if none is specified. Set up a weak reference
        # to clean up the temporary directory when the model is deleted.
        if output_dir is None:
            tempdir = TemporaryDirectory()
            weakref.finalize(self, tempdir.cleanup)
            output_dir = tempdir.name

        # Make sure the output directory exists
        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"Output directory {output_dir} does not exist.")

        # Set the output directory
        self.output_dir = output_dir

    def write_stan_program(self) -> None:
        """Write and format the generated Stan program to disk.

        This method writes the Stan program code to a .stan file in the
        output directory and applies Stan's canonical formatting for
        consistency and readability.

        The written file can be inspected, modified, or used independently
        of SciStanPy for debugging or optimization purposes.
        """
        # Write the raw code
        with open(self.stan_program_path, "w", encoding="utf-8") as f:
            f.write(self.code())

        # Format the code
        format_stan_file(
            self.stan_program_path,
            overwrite_file=True,
            canonicalize=True,
            stanc_options=self._stanc_options,
        )

    def gather_inputs(
        self, **observables: "custom_types.SampleType"
    ) -> dict[str, "custom_types.SampleType"]:
        """Gather complete input data for Stan sampling.

        :param observables: User-provided data for observable parameters
        :type observables: dict[str, custom_types.SampleType]

        :returns: Complete data dictionary for Stan sampling
        :rtype: dict[str, custom_types.SampleType]

        :raises ValueError: If required observables are missing or extra observables provided
        :raises ValueError: If observable shapes don't match model specifications

        This method combines user-provided observable data with automatically
        gathered constants and hyperparameters to create the complete data
        input required for Stan sampling.

        Data Processing:
        - Validates all required observables are provided
        - Checks shape compatibility between data and model specifications
        - Automatically gathers constants and hyperparameters from model
        - Handles variable name transformation (dots to double underscores)
        """
        # Make sure we have all the observables that the user must provide. Report
        # any missing or extra observables
        provided_observables = set(observables.keys())
        if missing := self.program.user_provided_varnames - provided_observables:
            raise ValueError(f"Missing observables: {', '.join(missing)}")
        elif extra := provided_observables - self.program.user_provided_varnames:
            raise ValueError(f"Extra observables: {', '.join(extra)}")

        # The shapes of the provided observables must match the shapes of the
        # observables in the model
        for name, obs in observables.items():
            if hasattr(obs, "shape"):
                if obs.shape != self.model[name].shape:
                    raise ValueError(
                        f"Shape mismatch for observable {name}: {obs.shape} != "
                        f"{self.model[name].shape}"
                    )
            elif self.model[name].shape != ():
                raise ValueError(
                    f"Shape mismatch for observable {name}: scalar != {self.model[name].shape}"
                )

        # Pull the hyperparameters from the model and add them to the inputs
        observables.update(self.autogathered_data)

        # Squeeze all numpy arrays
        observables = {
            name: obs.squeeze() if isinstance(obs, np.ndarray) else obs
            for name, obs in observables.items()
        }

        # All dots in the names must be replaced with double underscores
        return {name.replace(".", "__"): obs for name, obs in observables.items()}

    def code(self) -> str:
        """Get the complete Stan program code.

        :returns: Stan program code as formatted string
        :rtype: str

        Returns the complete, formatted Stan program generated from the
        SciStanPy model for inspection or external use.
        """
        return self.program.code

    def _get_sample_init(
        self, *, chains: custom_types.Integer, seed: Optional[custom_types.Integer]
    ) -> list[dict[str, Union[npt.NDArray[np.floating], np.floating]]]:
        """Generate prior-based initialization for MCMC chains.

        :param chains: Number of chains to initialize
        :type chains: custom_types.Integer
        :param seed: Random seed for reproducible initialization. Defaults to None.
        :type seed: Optional[custom_types.Integer]

        :returns: List of initialization dictionaries, one per chain
        :rtype: list[dict[str, Union[npt.NDArray[np.floating], np.floating]]]

        This method draws samples from the model's prior distribution to
        provide reasonable starting values for MCMC sampling, which can
        improve convergence and sampling efficiency.
        """
        # Draw from the prior distribution of the model, keeping only the draws
        # for the non-observable parameters
        draws = {
            component.model_varname: draw
            for component, draw in self.model.draw(
                n=chains, named_only=False, seed=seed
            ).items()
            if isinstance(component, parameters.Parameter) and not component.observable
        }

        # The draws should overlap perfectly with the parameters of the model
        assert set(draws.keys()) == self.program.all_paramnames

        # Separate the draws into one dictionary per chain
        return [{name: draw[i] for name, draw in draws.items()} for i in range(chains)]

    def get_varnames_to_dimnames(self) -> dict[str, list[str]]:
        """Create mapping from variable names to dimension names for ArviZ.

        :returns: Dictionary mapping variable names to their dimension names
        :rtype: dict[str, list[str]]

        This mapping is essential for creating properly structured ArviZ
        InferenceData objects with appropriate coordinate information.
        """
        # Get the mapping from level and size of a dimension to the name of the
        # dimension
        name_mapper = self.model.get_dimname_map()

        # Loop over all variables and create a mapping of dimension names
        return {
            varname.replace(".", "__"): tuple(
                reversed(
                    [
                        name_mapper[k]
                        for k in enumerate(self.model[varname].shape[::-1])
                        if k[-1] > 1
                    ]
                )
            )
            for varname in self.program.all_varnames
        }

    def sample(  # pylint: disable=arguments-differ
        self,
        *args,
        precision: Literal["double", "single", "half"] = "single",
        mib_per_chunk: custom_types.Integer | None = None,
        use_dask: bool = False,
        **kwargs,
    ) -> "results.SampleResults":
        """Execute MCMC sampling with enhanced SciStanPy integration.

        :param args: Positional arguments passed to CmdStanModel.sample
        :param precision: Numerical precision for result arrays. Defaults to 'single'.
        :type precision: Literal["double", "single", "half"]
        :param mib_per_chunk: When using dask, memory limit per chunk for large arrays.
            Defaults to None (uses Dask default).
        :type mib_per_chunk: Optional[custom_types.Integer]
        :param use_dask: Whether to use Dask for large array processing. Defaults to False.
        :type use_dask: bool
        :param kwargs: Keyword arguments passed to CmdStanModel.sample

        :returns: Comprehensive sampling results with SciStanPy integration
        :rtype: results.SampleResults

        Enhanced Features:
        - Automatic data gathering from SciStanPy model
        - Prior-based initialization when requested
        - Result processing and structuring
        - Memory-efficient handling of large result arrays
        - Integration with SciStanPy's result analysis tools

        The method provides an interface that handles all the complexities of data
        preparation and result processing while maintaining full compatibility with
        CmdStanPy's sampling options.
        """
        # Update the sample function from CmdStanModel to automatically pull the
        # data from the StanModel
        updated_parent_sample = _update_cmdstanpy_func(CmdStanModel.sample)

        # Combine args and kwargs into a single dictionary
        kwargs.update(dict(zip(CmdStanModel.sample.__code__.co_varnames[1:], args)))

        # Set the number of chains if not provided
        if "chains" not in kwargs or kwargs["chains"] is None:
            kwargs["chains"] = 4

        # If initializing from the prior, we need to draw from the prior
        if kwargs.get("inits") == "prior":
            kwargs["inits"] = self._get_sample_init(  # pylint: disable=protected-access
                chains=kwargs["chains"], seed=kwargs.get("seed")
            )

        # Run the sample function
        fit = updated_parent_sample(self, **kwargs)

        # Build the results object
        return results.SampleResults(
            model=self.model,
            fit=fit,
            precision=precision,
            mib_per_chunk=mib_per_chunk,
            use_dask=use_dask,
        )

    # Enhanced CmdStanModel methods with automatic data gathering
    generate_quantities = _update_cmdstanpy_func(
        CmdStanModel.generate_quantities, warn=True
    )
    """Enhanced generate_quantities with automatic data gathering.

    Automatically gathers required data from SciStanPy model. Experimental feature.

    :warning: This method is experimental and not thoroughly tested.
    """

    laplace_sample = _update_cmdstanpy_func(CmdStanModel.laplace_sample, warn=True)
    """Enhanced laplace_sample with automatic data gathering.

    Automatically gathers required data from SciStanPy model. Experimental feature.

    :warning: This method is experimental and not thoroughly tested.
    """

    log_prob = _update_cmdstanpy_func(CmdStanModel.log_prob, warn=True)
    """Enhanced log_prob with automatic data gathering.

    Automatically gathers required data from SciStanPy model. Experimental feature.

    :warning: This method is experimental and not thoroughly tested.
    """

    optimize = _update_cmdstanpy_func(CmdStanModel.optimize, warn=True)
    """Enhanced optimize with automatic data gathering.

    Automatically gathers required data from SciStanPy model. Experimental feature.

    :warning: This method is experimental and not thoroughly tested.
    """

    pathfinder = _update_cmdstanpy_func(CmdStanModel.pathfinder, warn=True)
    """Enhanced pathfinder with automatic data gathering.

    Automatically gathers required data from SciStanPy model. Experimental feature.

    :warning: This method is experimental and not thoroughly tested.
    """

    variational = _update_cmdstanpy_func(CmdStanModel.variational, warn=True)
    """Enhanced variational with automatic data gathering.

    Automatically gathers required data from SciStanPy model. Experimental feature.

    :warning: This method is experimental and not thoroughly tested.
    """

    @property
    def stan_program_path(self) -> str:
        """Get path to the generated Stan program file.

        :returns: Full path to .stan file
        :rtype: str
        """
        return self.stan_executable_path + ".stan"

    @property
    def autogathered_data(self) -> dict[str, npt.NDArray]:
        """Get automatically gathered data from model constants.

        :returns: Dictionary of constant values from SciStanPy model
        :rtype: dict[str, npt.NDArray]
        """
        return {
            name.replace(".", "__"): self.model[name].value
            for name in self.program.autogathered_varnames
        }

    @property
    def all_varnames(self) -> set[str]:
        """Get all variable names with Stan-compatible formatting.

        :returns: Set of all variable names (dots replaced with double underscores)
        :rtype: set[str]
        """
        return {name.replace(".", "__") for name in self.program.all_varnames}
