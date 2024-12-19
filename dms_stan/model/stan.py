"""Holds code for interfacing with the Stan modeling language."""

import copy
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
    Optional,
    ParamSpec,
    TypedDict,
    TypeVar,
    Union,
)

import numpy as np
import numpy.typing as npt

from cmdstanpy import CmdStanMCMC, CmdStanModel

import dms_stan as dms

from dms_stan.custom_types import SampleType
from dms_stan.defaults import (
    DEFAULT_CPP_OPTIONS,
    DEFAULT_FORCE_COMPILE,
    DEFAULT_STANC_OPTIONS,
    DEFAULT_USER_HEADER,
)

from .components import Constant, Normal, Parameter, TransformedParameter
from .components.abstract_model_component import AbstractModelComponent

# Function for combining a list of Stan code lines
DEFAULT_INDENTATION = 4

# Parameter and return types for decorated functions
P = ParamSpec("P")
R = TypeVar("R")


def combine_lines(lines: list[str], indentation: int = DEFAULT_INDENTATION) -> str:
    """Combine a list of Stan code lines into a single string."""

    # Nothing if no lines
    if len(lines) == 0:
        return ""

    # Combine the lines
    return "\n" + ";\n".join(f"{' ' * indentation}{el}" for el in lines) + ";"


def _update_cmdstanpy_func(func: Callable[P, R], warn: bool = False) -> Callable[P, R]:
    """
    Decorator that modifies CmdStanModel functions requiring data to automatically
    pull the data from the StanModel. The user must provide values for the observables.
    """

    @functools.wraps(func)
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        """Wrapper function for gathering inputs."""
        # If warning the user about lack of testing, do so
        if warn:
            warnings.warn(
                f"{func.__name__} is experimental and has not been thoroughly tested"
                " in the context of DMS Stan models. Use with caution."
            )

        # Get the Stan model from the first argument
        stan_model = args[0]
        assert isinstance(stan_model, StanModel)

        # Combine args and kwargs into a single dictionary
        kwargs.update(dict(zip(func.__code__.co_varnames[1:], args[1:])))

        # If a seed is not provided, use the global random number generator to get
        # one
        if kwargs.get("seed") is None:
            kwargs["seed"] = dms.RNG.integers(0, 2**32 - 1)

        # `data` must be a key in the kwargs
        if "data" not in kwargs:
            raise ValueError(
                f"The 'data' keyword argument must be provided to {func.__name__}"
            )

        # Gather the inputs for the Stan model. The user should have provided
        # values for the observables. We will get the rest of the inputs from the
        # DMS Stan model.
        kwargs["data"] = stan_model.gather_inputs(**kwargs["data"])

        # Call the wrapped function with the inputs
        return func(stan_model, **kwargs)

    return inner


class StanCodeBase(ABC, list):
    """Base object for Stan code."""

    def __init__(self, parent_loop: Optional["StanCodeBase"]):
        """Initializes the Stan code base."""
        # Initialize the list
        super().__init__()

        # Record the parent loop
        self.parent_loop = parent_loop

        # Descendents are an empty list for now
        self.descendents: list[StanCodeBase] = []

    def generate_for_loops(self) -> Generator["StanForLoop", None, None]:
        """Generates all for-loops in the program."""
        # Yield this loop
        yield self

        # Loop over all descendants and yield their descendents
        for descendent in self.descendents:
            yield from descendent.generate_for_loops()

    @abstractmethod
    def get_parent_loop(self, n: int) -> "StanCodeBase":
        """
        If rolling out the ancestry of the loop, starting from the progenitor and
        ending with this loop, return the nth loop in that list (i.e., if n = 0,
        return the progenitor, if n = -1, return this loop).
        """

    @property
    @abstractmethod
    def stan_code_level(self) -> int:
        """Returns the code level of the object."""


class StanForLoop(StanCodeBase):
    """Represents a Stan for-loop."""

    def __init__(self, parent_loop: StanCodeBase):
        """Initialize the for-loop.

        Args:
            index: The index variable for the loop
            start: The starting value of the loop
            end: The ending value of the loop
        """
        # Initialize the list
        super().__init__(parent_loop)

        # Record the loop parameters
        self.end: int = 1

        # Append this loop to the parent loop and its descendents
        self.parent_loop.append(self)
        self.parent_loop.descendents.append(self)

        # Get the ancestry of the loop
        self.ancestry = self._get_ancestry()
        self._stan_code_level = len(self.ancestry) - 1

    def _get_ancestry(self) -> list[StanCodeBase]:
        """Retrieves the ancestry of the loop."""

        # If the parent is the program, we are at the top level and have only the
        # program as our ancestor
        if isinstance(self.parent_loop, StanProgram):
            return [self.parent_loop]

        # Otherwise, we have the parent loop and all of its ancestors
        return self.parent_loop.ancestry + [self.parent_loop]

    def get_parent_loop(self, n: int) -> StanCodeBase:
        return (self.ancestry + self)[n]

    def set_end(self) -> None:
        """Sets the end value of the loop based on the components it contains."""
        # Get all end values of the components in the loop
        all_ends = {component.shape[self.stan_code_level] for component in self}
        n_ends = len(all_ends)

        # If two end values, one must be 1
        if n_ends == 2:
            assert 1 in all_ends
            all_ends.remove(1)

        # Otherwise, we must have only one end value
        elif n_ends != 1:
            raise ValueError(f"Invalid end values: {all_ends}")

        # Set the end value
        self.end = all_ends.pop()

    def squash(self) -> None:
        """
        If this is a singleton loop, it is removed from the lineage. This is done
        by moving the decendant loops to the parent loop and removing this loop.
        """
        # If the end is 1, we are a singleton loop
        if self.end == 1:
            self.parent_loop.remove(self)
            self.parent_loop.extend(self.descendents)
            self.descendents = []
            self.parent_loop = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the loop and any parent loops."""
        # If the parent loop is the program, we are at the top level and there is
        # no shape to return.
        if isinstance(self.parent_loop, StanProgram):
            return (0, self.end)

        # Otherwise, return the shape of the parent loop followed by the shape of
        # this loop
        return self.parent_loop.shape + (self.end,)

    @property
    def stan_code_level(self) -> int:
        return self._stan_code_level


class StanProgram(StanCodeBase):
    """Represents a Stan program"""

    def __init__(self, model: "dms.model.Model"):
        """Initializes and compiles the Stan program."""
        # Initialize the list
        super().__init__(None)

        # Note the model
        self.model = model

        # Build the map from model node to depth in the tree of nodes
        self.node_to_depth: dict[AbstractModelComponent, int] = (
            self.build_node_to_depth()
        )

        # Get the names of all variables and parameters in the model
        self.all_varnames, self.all_paramnames = self.get_varnames()

        # Get allowed index variable names for each level
        self.allowed_index_names = tuple(
            char
            for char in dms.defaults.DEFAULT_INDEX_ORDER
            if {char, char.upper()}.isdisjoint(self.all_varnames)
        )

        # Compile the program
        self.compile()

    def build_node_to_depth(self) -> dict[AbstractModelComponent, int]:
        """
        Builds a mapping from each node to its maximum depth in the tree relative
        to the tree roots.
        """
        # We need a mapping from each node to its maximum depth in the tree.
        node_to_depth: dict[AbstractModelComponent, int] = {}

        # Starting from hyperparameters, walk down the tree and record the depth of each
        # node
        for hyper in self.model.hyperparameters:

            # Hyperparameters should be at depth 0
            node_to_depth[hyper] = 0

            # We should never encounter a node twice for a single hyperparameter
            observed_nodes: set[AbstractModelComponent] = set()

            # Walk the tree rooted at this hyperparameter and record the depth of each node
            # if it is higher than the current maximum depth
            for depth, _, child in hyper.walk_tree():

                # Make sure we haven't seen this node before
                assert child not in observed_nodes, "Node encountered twice in tree"
                observed_nodes.add(child)

                # Record the maximum depth of this node
                node_to_depth[child] = max(depth, node_to_depth.get(child, 0))

        return node_to_depth

    def get_varnames(self) -> tuple[set[str], set[str]]:
        """Gets the names of all variables and parameters in the model."""

        # Get the names of all the variables in the model
        name_counts = Counter(node.model_varname for node in self.node_to_depth)
        if collisions := {name for name, count in name_counts.items() if count > 1}:
            raise ValueError(f"Name collisions for variables: {collisions}")

        # Get all variable names and all parameter names
        all_varnames = {node.model_varname for node in self.node_to_depth}
        all_paramnames = {
            node.model_varname
            for node in self.node_to_depth
            if isinstance(node, Parameter)
        }

        return all_varnames, all_paramnames

    def get_parent_loop(self, n: int) -> StanCodeBase:
        # Can only get self
        assert n == -1 or n == 0
        return self

    def compile(self) -> None:
        """Organizes the elements of the Stan program into the correct order."""
        # Get the order of operations. This is a list of nodes sorted by their maximum
        # depth in the tree
        order_of_operations: list[AbstractModelComponent] = sorted(
            self.node_to_depth, key=lambda x: (self.node_to_depth[x], x.stan_code_level)
        )

        # The first component in the order of operations must be at code level 0
        assert order_of_operations[0].stan_code_level == 0

        # Set up for compilation. We define the `target_loop`, which is the loop
        # to which components will be actively added. We also defined the `previous_component`,
        # which is the component one operation back.
        target_loop: StanCodeBase = self
        previous_component = None  # Item one operation back
        for component in order_of_operations:

            # If the current or previous component is at code level 0, then the
            # program is the target loop. We will need to add as many for-loops
            # as we have levels of indentation in the current component.
            if (
                component.stan_code_level == 0
                or previous_component.stan_code_level == 0
            ):
                target_loop = self
                n_loops = component.stan_code_level

            # In any other case, we need to determine which dimensions of the current
            # component are compatible with the previous component and find the
            # appropriate parent loop.
            else:

                # We need to determine the extent to which the shapes of the current
                # and previous components are compatible. This is the number of
                # shared leading dimensions between the two components' shapes.
                compat_level = previous_component.get_stan_level_compatibility(
                    other=component
                )

                # Get the parent loop of the current component that is compatible
                # with the the current component. Note that we also update our referenced
                # loop in this step. Note that the target loop must be a for-loop
                # at this point as the previous `if` block handles all cases where
                # the target loop is the program.
                assert isinstance(target_loop, StanForLoop)
                target_loop = target_loop.get_parent_loop(compat_level)

                # How many for-loops do we need to add? The code level of the current
                # component must be greater than or equal to the target now
                n_loops = component.stan_code_level - (
                    0 if target_loop is self else target_loop.stan_code_level
                )
                assert (
                    n_loops >= 0
                ), "Current component code level is less than target loop"

            # Add for-loops if necessary
            for _ in range(
                target_loop.stan_code_level, target_loop.stan_code_level + n_loops
            ):
                target_loop = StanForLoop(parent_loop=target_loop)

            # Add the component to the target loop and update the previous component
            target_loop.append(component)
            previous_component = component

        # Set end values for all for-loops
        for loop in self.generate_for_loops():
            loop.set_end()

        # Squash any singletons
        for loop in self.generate_for_loops():
            loop.squash()

        # Remove all squashed loops
        self[:] = [
            component
            for component in self
            if not isinstance(component, StanForLoop)
            or component.parent_loop is not None
        ]

    def generate_for_loops(self) -> Generator[StanForLoop, None, None]:
        """Yields from the descendents only."""
        for descendent in self.descendents:
            yield from descendent.generate_for_loops()

    @property
    def stan_code_level(self) -> int:
        """Returns the code level of the program, which is always 0."""
        return 0


# # We need a specific type for the steps of the Stan model
# class StanStepsType(TypedDict):
#     """Type for the steps of the Stan model."""

#     data: Union[str, list[str]]
#     parameters: Union[str, list[str]]
#     transformed_parameters: Union[str, list[str]]
#     model: str
#     generated_quantities: Union[str, list[str]]


# class CodeBlock(ABC):
#     """Used for organizing code blocks in the Stan model."""

#     def __init__(self, model: "dms.model.Model"):
#         """
#         Holds code blocks for the Stan model. This will never be directly instantiated.
#         """
#         # Build the block levels
#         self.levels: dict[Parameter : list[list[Parameter]]] = {
#             obs: [[] for _ in range(obs.stan_code_level + 1)]
#             for obs in model.observables
#         }

#     def add_component(
#         self, observable: Parameter, component: AbstractModelComponent
#     ) -> None:
#         """Adds a component to the appropriate level of the code block."""
#         self[observable][component.stan_code_level].append(component)

#     def finalize_levels(self) -> None:
#         """
#         Removes empty levels from the end of each observable branch and reverse
#         orders the components in each level to account for the fact that we built
#         the levels from the bottom up.
#         """
#         # New levels for each observable
#         new_levels: dict[Parameter, list[list[Parameter]]] = {}

#         # Reverse each level for each observable and remove empty levels from the
#         # end
#         for observable, levels in self.levels.items():

#             # Reverse the levels
#             reversed_levels = [level[::-1] for level in levels]

#             # Remove empty levels from the end
#             while reversed_levels and not reversed_levels[-1]:
#                 reversed_levels.pop()

#             # Add the reversed levels to the new levels
#             new_levels[observable] = reversed_levels

#         # Update the levels
#         self.levels = new_levels

#     def build_tree(self) -> dms.custom_types.StanTreeType:
#         """Combines the levels from the observables into a single tree."""
#         # A list giving the full tree
#         tree: dms.custom_types.StanTreeType = []

#         # A map for error checking to be sure we don't double-add components
#         observable_to_list: dict[
#             AbstractModelComponent, dms.custom_types.StanTreeType
#         ] = {}
#         # Loop over the different levels
#         for level_ind in range(
#             max(len(level_list) for level_list in self.levels.values())
#         ):

#             # Build the mapping from level size to list
#             level_size_to_list: dict[int, dms.custom_types.StanTreeType] = {}

#             # Get the dimsize of each branch in the level
#             for observable, branch in self.levels.items():

#                 # Skip if the level is not in the branch
#                 if level_ind >= len(branch):
#                     continue

#                 # Get level sizes
#                 level_sizes = {
#                     0 if len(component.shape) == 0 else component.shape[level_ind]
#                     for component in branch[level_ind]
#                 }

#                 # If on the first level, all sizes are valid, so we assign 0. If
#                 # there are no sizes, we assign 0.
#                 if level_ind == 0 or len(level_sizes) == 0:
#                     level_size = 0

#                 # Otherwise, we need to check the sizes. There can only be one size
#                 # and it cannot be 1.
#                 else:
#                     assert 1 not in level_sizes
#                     assert len(level_sizes) == 1, level_sizes
#                     level_size = level_sizes.pop()

#                 # If the first level, the reflist is the tree itself
#                 if level_ind == 0:
#                     reflist = tree

#                 # Otherwise, if we have not encountered this level size before,
#                 # we have a new branch in the list. Create the list, add it to the
#                 # mapping between level size and list, and append it as a new level
#                 # to the tree.
#                 elif level_size not in level_size_to_list:
#                     reflist = []
#                     level_size_to_list[level_size] = reflist
#                     observable_to_list[observable].append(reflist)

#                 # If the level size does exist, however, we can just retrieve the
#                 # list we have already built
#                 else:
#                     reflist = level_size_to_list[level_size]

#                 # Add the components of the branch to the list if they are not
#                 # already there
#                 for component in branch[level_ind]:
#                     if component not in reflist:
#                         reflist.append(component)

#                 # Update the mapping from observable to list
#                 observable_to_list[observable] = reflist

#         return tree

#     def write_tree_code(self, allowed_index_names: tuple[str, ...]) -> str:
#         """Builds Stan code for the tree of observables."""

#         # We need a function to recursively build the code for the tree
#         def write_code(tree: dms.custom_types.StanTreeType, level: int) -> str:
#             """Recursively builds the code for the tree."""
#             # Loop over the tree
#             stan_code: list[str] = []
#             sizes: list[int] = []
#             for element in tree:

#                 # If the element is a list, call the function recursively
#                 if isinstance(element, list):
#                     stan_code.append(write_code(element, level + 1))

#                 # If the element is a component, write the code for the component
#                 elif isinstance(element, AbstractModelComponent):
#                     stan_code.append(
#                         self.write_component_code(
#                             component=element, allowed_index_names=allowed_index_names
#                         )
#                     )
#                     sizes.append(0 if element.ndim == 0 else element.shape[level])

#                 # Otherwise, raise an error
#                 else:
#                     raise ValueError(f"Unknown element type {type(element)}")

#             # Determine the indendation
#             indentation = DEFAULT_INDENTATION * (level + 1)

#             # Combine the lines
#             combined_lines = combine_lines(stan_code, indentation=indentation)

#             # Get the size of the level
#             unique_sizes = {size for size in sizes if size != 1}

#             # If there are no sizes OR we are at the first level, just return.
#             if level == 0 or len(unique_sizes) == 0:
#                 return combined_lines

#             # There should only be one size
#             assert len(unique_sizes) == 1

#             # Wrap the code in a for loop
#             index_var = allowed_index_names[level - 1]
#             return (
#                 f"\n{' ' * indentation}for ({index_var} in 1:{unique_sizes.pop()}) "
#                 + "{\n"
#                 + combined_lines
#                 + f"\n{' ' * indentation}}}"
#             )

#         # Build the tree and write the code
#         tree = self.build_tree()
#         return write_code(tree, 0)

#     def __getitem__(self, key: Parameter) -> list[list[Parameter]]:
#         """Get the code block for a parameter."""
#         return self.levels[key]

#     @abstractmethod
#     def write_component_code(
#         self,
#         component: Union[Parameter, TransformedParameter],
#         allowed_index_names: tuple[str, ...],
#     ) -> str:
#         """Writes a component to the appropriate level of the code block."""


# class CodeBlockWithObservable(CodeBlock):
#     """
#     Identical to `CodeBlock`, but automatically handles the placement of the observable.
#     """

#     def __init__(self, model: "dms.model.Model"):

#         # Run the initializer for the parent class
#         super().__init__(model)

#         # Populate the observable levels
#         for obs in model.observables:
#             self[obs][-1].append(obs)


# class ModelCodeBlock(CodeBlockWithObservable):
#     """Used for organizing the 'model' code block in the Stan model."""

#     def write_component_code(
#         self,
#         component: Union[Parameter, TransformedParameter],
#         allowed_index_names: tuple[str, ...],
#     ) -> str:
#         """Writes a component to the appropriate level of the code block."""
#         return component.get_target_incrementation(allowed_index_names)


# class TransformedParametersCodeBlock(CodeBlock):
#     """Used for organizing the 'transformed parameters' code block in the Stan model."""

#     def write_component_code(
#         self,
#         component: Union[Parameter, TransformedParameter],
#         allowed_index_names: tuple[str, ...],
#     ) -> str:
#         """Writes a component to the appropriate level of the code block."""
#         return component.get_transformation_assignment(allowed_index_names)


# class GeneratedQuantitiesCodeBlock(CodeBlockWithObservable):
#     """Used for organizing the 'generated quantities' code block in the Stan model."""

#     def write_component_code(
#         self, component: Parameter, allowed_index_names: tuple[str, ...]
#     ) -> str:
#         """Writes a component to the appropriate level of the code block."""
#         return component.get_generated_quantities(allowed_index_names)


class StanModel(CmdStanModel):
    """
    Expands the CmdStanModel class to allow for interfacing with the rest of
    DMSStan.
    """

    # We initialize with a DMSStan model instance
    def __init__(
        self,
        model: "dms.model.Model",
        output_dir: Optional[str] = None,
        force_compile: bool = DEFAULT_FORCE_COMPILE,
        stanc_options: Optional[dict[str, Any]] = DEFAULT_STANC_OPTIONS,
        cpp_options: Optional[dict[str, Any]] = DEFAULT_CPP_OPTIONS,
        user_header: Optional[str] = DEFAULT_USER_HEADER,
    ):
        # Set default options
        stanc_options = stanc_options or {}
        cpp_options = cpp_options or {}

        # Note the underlying DMSStan model
        self.model = model

        # Compile the program
        self.program = StanProgram(model)

        # # A dictionary stores the steps of the program
        # self.steps: StanStepsType = {
        #     "data": [],
        #     "parameters": [],
        #     "transformed parameters": [],
        #     "model": "",
        #     "generated quantities": [],
        # }

        # Build the program elements
        # self.data_inputs = self._declare_variables()
        # self._build_code_blocks()

        # # Set the output directory
        # self._set_output_dir(output_dir)

        # # Write the Stan program
        # self.write_stan_program()

        # # Initialize the CmdStanModel
        # super().__init__(
        #     stan_file=self.stan_program_path,
        #     exe_file=(
        #         self.stan_executable_path
        #         if os.path.exists(self.stan_executable_path) and not force_compile
        #         else None
        #     ),
        #     force_compile=force_compile,
        #     stanc_options=stanc_options,
        #     cpp_options=cpp_options,
        #     user_header=user_header,
        # )

    # def _set_output_dir(self, output_dir: Optional[str]) -> None:
    #     """Set the output directory for the model."""
    #     # Make a temporary directory if none is specified. Set up a weak reference
    #     # to clean up the temporary directory when the model is deleted.
    #     if output_dir is None:
    #         tempdir = TemporaryDirectory()
    #         weakref.finalize(self, tempdir.cleanup)
    #         output_dir = tempdir.name

    #     # Make sure the output directory exists
    #     if not os.path.exists(output_dir):
    #         raise FileNotFoundError(f"Output directory {output_dir} does not exist.")

    #     # Set the output directory
    #     self.output_dir = output_dir

    # def _declare_variables(self) -> tuple[tuple[str, bool], ...]:
    #     """Write the parameters, transformed parameters, and data of the model."""

    #     def record_data(data: Union[Parameter, Constant]) -> None:
    #         """Record a data object in the steps."""
    #         self.steps["data"].append(data.stan_parameter_declaration)
    #         data_inputs.append((data.model_varname, data.observable))

    #     def check_model_varname(name: str) -> None:
    #         """
    #         Records the variable's model name in the set of all variable names and
    #         makes sure it is not already in the set.
    #         """
    #         # Check for name collisions
    #         if name in self.all_varnames:
    #             raise ValueError(f"Name collision for {name}")

    #         # Add the variable name to the set of all variable names
    #         self.all_varnames.add(name)

    #     def write_variable_code(variable: AbstractModelComponent) -> None:
    #         """Writes the Stan code for a variable."""

    #         # Check the model variable name
    #         check_model_varname(variable.model_varname)

    #         # Note that the node has been observed
    #         observed_nodes.add(variable)

    #         # If an observable, we need a data and generated quantity entry only
    #         if variable.observable:
    #             record_data(observable)
    #             check_model_varname(variable.stan_generated_quantity_declaration)
    #             self.steps["generated quantities"].append(
    #                 variable.stan_generated_quantity_declaration
    #             )
    #             return

    #         # If the variable is a Normal distribution that is not a hyperparameter,
    #         # we will be non-centering. Thus, the true parameter is defined as a
    #         # transformed parameter and a dummy parameter is defined as a parameter.
    #         elif isinstance(variable, Normal) and not variable.is_hyperparameter:
    #             self.steps["parameters"].append(
    #                 f"{variable.stan_dtype} {variable.noncentered_varname}"
    #             )
    #             self.steps["transformed parameters"].append(
    #                 variable.stan_parameter_declaration
    #             )

    #         # If the variable is a parameter, add it to the parameters list
    #         elif isinstance(variable, Parameter):
    #             self.all_paramnames.add(variable.model_varname)
    #             self.steps["parameters"].append(variable.stan_parameter_declaration)

    #         # If the variable is a constant, add it to the data block
    #         elif isinstance(variable, Constant):
    #             record_data(variable)

    #         # If it is a transformed parameter AND named, add it to the transformed
    #         # parameters list
    #         elif isinstance(variable, TransformedParameter):
    #             if variable.is_named:
    #                 self.steps["transformed parameters"].append(
    #                     variable.stan_parameter_declaration
    #                 )

    #         # Otherwise, raise an error
    #         else:
    #             raise ValueError(f"Unknown node type {type(variable)}")

    #     # We only have one observable
    #     assert len(self.model.observables) == 1
    #     observable = self.model.observables[0]

    #     # Define a set for recording observed nodes and a list for recording data
    #     # inputs
    #     observed_nodes: set[AbstractModelComponent] = set()
    #     data_inputs: list[tuple[str, bool]] = []

    #     # Record the observable as a data input and write the variable code for
    #     # the observable
    #     write_variable_code(observable)

    #     # Now loop over the parents of the observable and add them to the appropriate
    #     # blocks
    #     for *_, parent in observable.walk_tree(False):

    #         # The parent cannot be an observable
    #         assert not parent.observable, "Parent cannot be an observable."

    #         # If the parent has been handled already, skip it.
    #         if parent in observed_nodes:
    #             continue

    #         # Write the variable code for the parent
    #         write_variable_code(parent)

    #     # Combine lines for the blocks
    #     self.steps["parameters"] = combine_lines(self.steps["parameters"])
    #     self.steps["data"] = combine_lines(self.steps["data"])
    #     self.steps["transformed parameters"] = combine_lines(
    #         self.steps["transformed parameters"]
    #     )
    #     self.steps["generated quantities"] = combine_lines(
    #         self.steps["generated quantities"]
    #     )

    #     return tuple(data_inputs)

    # def _build_code_blocks(self):
    #     """Builds the 'model' and 'transformed data' blocks of the Stan model."""

    #     def format_code_block(levels: list[list[str]]) -> str:
    #         """
    #         Builds a block of code from a list of levels. Each level is a list of
    #         strings giving the code for that level. Each element of each list will
    #         be joined by a semi-colon and a newline. The first level is the top
    #         level while the other levels are nested within the previous level using
    #         for loops.
    #         """
    #         # Invert the contents of each level. The inversion is needed because
    #         # we built the levels from the bottom up.
    #         levels = [level[::-1] for level in levels]

    #         # Now build the nested 'for' loops
    #         code_block = combine_lines(levels[0])
    #         indentation = DEFAULT_INDENTATION
    #         for code_level, level in enumerate(levels[1:], 1):

    #             # Build the prefix for the 'for' loop
    #             index_var = allowed_index_names[code_level - 1]
    #             max_size = level_to_size[code_level]
    #             for_prefix = (
    #                 f"\n{' ' * indentation}for ({index_var} in 1:{max_size}) " + "{\n"
    #             )

    #             # Update the indentation
    #             indentation += DEFAULT_INDENTATION

    #             # Add the code, making sure to include the appropriate number of spaces
    #             code_block += for_prefix + combine_lines(level, indentation=indentation)

    #         # Now close the 'for' loops
    #         for _ in range(len(levels) - 1):
    #             indentation -= DEFAULT_INDENTATION
    #             code_block += f"\n{' ' * indentation}}}"
    #         assert indentation == DEFAULT_INDENTATION

    #         return code_block

    #     # TODO: We need to be able to handle multiple observables. We can do this
    #     # by identifying the tree that each observable is part of and then building
    #     # different for-loops for each tree.
    #     # Items can be in the same for-loop if all parents have the same size!

    #     # Get allowed index variable names for each level
    #     allowed_index_names = tuple(
    #         char
    #         for char in dms.defaults.DEFAULT_INDEX_ORDER
    #         if {char, char.upper()}.isdisjoint(self.all_varnames)
    #     )

    #     # Get the number of for-loop levels. This is given by the dimensionality
    #     # of the observables. We create lists for each level. Different variables
    #     # will be defined and accessed depending on the level to which they belong.
    #     # Note that we assume the last level is vectorized
    #     model_code_block = ModelCodeBlock(self.model)
    #     transformed_code_block = TransformedParametersCodeBlock(self.model)
    #     generated_code_block = GeneratedQuantitiesCodeBlock(self.model)

    #     # Loop over all observables
    #     for observable in self.model.observables:

    #         # There should be enough index names to cover all levels
    #         if len(allowed_index_names) < (observable.ndim - 1):
    #             raise ValueError(
    #                 f"Not enough index names ({len(allowed_index_names)}) to cover "
    #                 f"{observable.ndim} levels"
    #             )

    #         # There should never be a transformation for the observable
    #         assert observable.get_transformation_assignment(allowed_index_names) == ""

    #         # Walk up the tree from the observable to the top level
    #         for *_, parent in observable.walk_tree(walk_down=False):

    #             # If a parameter or a named transformed parameter, add target incrementation
    #             # and any transformations to the appropriate level
    #             if isinstance(parent, Parameter) or (
    #                 isinstance(parent, TransformedParameter) and parent.is_named
    #             ):
    #                 if parent.get_target_incrementation(allowed_index_names):
    #                     model_code_block.add_component(
    #                         observable=observable, component=parent
    #                     )
    #                 if parent.get_transformation_assignment(allowed_index_names):
    #                     transformed_code_block.add_component(
    #                         observable=observable, component=parent
    #                     )

    #             # Otherwise the parent must be a constant or an unnamed transformed
    #             # parameter
    #             else:
    #                 assert isinstance(parent, Constant) or (
    #                     isinstance(parent, TransformedParameter) and not parent.is_named
    #                 )

    #     # Format code blocks
    #     for code_block in (
    #         model_code_block,
    #         transformed_code_block,
    #         generated_code_block,
    #     ):
    #         # Finalize levels and write the code
    #         code_block.finalize_levels()
    #         print(code_block.write_tree_code(allowed_index_names=allowed_index_names))

    #     # # Format the code blocks
    #     # self.steps["model"] = format_code_block(model_levels)
    #     # self.steps["transformed parameters"] = (
    #     #     self.steps["transformed parameters"]
    #     #     + "\n"
    #     #     + format_code_block(transformed_param_levels)
    #     # )
    #     # self.steps["generated quantities"] = (
    #     #     self.steps["generated quantities"]
    #     #     + "\n"
    #     #     + format_code_block(generated_quantities_levels)
    #     # )

    # def write_stan_program(self) -> None:
    #     """
    #     Write the Stan model to the output directory. This will overwrite any model
    #     in that directory.
    #     """
    #     with open(self.stan_program_path, "w", encoding="utf-8") as f:
    #         f.write(self.stan_program)

    # def gather_inputs(self, **observables: SampleType) -> dict[str, SampleType]:
    #     """
    #     Gathers the inputs for the Stan model. Values for observables must be provided
    #     by the user. All other inputs will be drawn from the DMS Stan model itself.

    #     Returns:
    #         dict[str, SampleType]: A dictionary of inputs for the Stan model.
    #     """
    #     # Split out the observable values from the other inputs
    #     required_observables = {
    #         name for name, indicator in self.data_inputs if indicator
    #     }

    #     # Make sure we have all the observables in the inputs and report any missing
    #     # or extra observables
    #     provided_observables = set(observables.keys())
    #     if missing := required_observables - provided_observables:
    #         raise ValueError(f"Missing observables: {', '.join(missing)}")
    #     elif extra := provided_observables - required_observables:
    #         raise ValueError(f"Extra observables: {', '.join(extra)}")

    #     # The shapes of the provided observables must match the shapes of the
    #     # observables in the model
    #     for name, obs in observables.items():
    #         if hasattr(obs, "shape"):
    #             if obs.shape != self.model[name].shape:
    #                 raise ValueError(
    #                     f"Shape mismatch for observable {name}: {obs.shape} != "
    #                     f"{self.model[name].shape}"
    #                 )
    #         elif self.model[name].shape != ():
    #             raise ValueError(
    #                 f"Shape mismatch for observable {name}: scalar != {self.model[name].shape}"
    #             )

    #     # Pull the hyperparameters from the model and add them to the inputs
    #     observables.update(
    #         {
    #             name: self.model[name].value
    #             for name, indicator in self.data_inputs
    #             if not indicator
    #         }
    #     )

    #     return observables

    # def code(self) -> str:
    #     """Just an alias for the `stan_program` property."""
    #     return self.stan_program

    # def _get_sample_init(
    #     self, *, chains: int, seed: Optional[int]
    # ) -> list[dict[str, Union[npt.NDArray[np.floating], np.floating]]]:
    #     """
    #     Draws from the prior distribution of the model to initialize the MCMC sampler.

    #     Args:
    #         chains (int): The number of chains.
    #         seed (Optional[int]): The seed for the random number generator.

    #     Returns:
    #         list[dict[str, npt.NDArray]]: A list of dictionaries where each dictionary
    #             contains the initial values for the parameters of the model in one
    #             chain.
    #     """
    #     # Draw from the prior distribution of the model, keeping only the draws
    #     # for the non-observable parameters
    #     draws = {
    #         component.model_varname: draw
    #         for component, draw in self.model.draw(
    #             n=chains, named_only=False, seed=seed
    #         ).items()
    #         if isinstance(component, Parameter) and not component.observable
    #     }

    #     # The draws should overlap perfectly with the parameters of the model
    #     assert set(draws.keys()) == self.all_paramnames

    #     # Separate the draws into one dictionary per chain
    #     return [{name: draw[i] for name, draw in draws.items()} for i in range(chains)]

    # def sample(self, *args, **kwargs) -> CmdStanMCMC:

    #     # Update the sample function from CmdStanModel to automatically pull the
    #     # data from the StanModel
    #     updated_parent_sample = _update_cmdstanpy_func(CmdStanModel.sample)

    #     # Combine args and kwargs into a single dictionary
    #     kwargs.update(dict(zip(CmdStanModel.sample.__code__.co_varnames[1:], args)))

    #     # Set the number of chains if not provided
    #     if "chains" not in kwargs or kwargs["chains"] is None:
    #         kwargs["chains"] = 4

    #     # If initializing from the prior, we need to draw from the prior
    #     if kwargs.get("inits") == "prior":
    #         kwargs["inits"] = self._get_sample_init(  # pylint: disable=protected-access
    #             chains=kwargs["chains"], seed=kwargs.get("seed")
    #         )

    #     # Call the parent sample function
    #     return updated_parent_sample(self, **kwargs)

    # # Update the CmdStanModel functions that require data
    # generate_quantities = _update_cmdstanpy_func(
    #     CmdStanModel.generate_quantities, warn=True
    # )
    # laplace_sample = _update_cmdstanpy_func(CmdStanModel.laplace_sample, warn=True)
    # log_prob = _update_cmdstanpy_func(CmdStanModel.log_prob, warn=True)
    # optimize = _update_cmdstanpy_func(CmdStanModel.optimize, warn=True)
    # pathfinder = _update_cmdstanpy_func(CmdStanModel.pathfinder, warn=True)
    # variational = _update_cmdstanpy_func(CmdStanModel.variational, warn=True)

    # @property
    # def stan_executable_path(self) -> str:
    #     """Get the path to the executable for this model."""
    #     return os.path.join(self.output_dir, "model")

    # @property
    # def stan_program_path(self) -> str:
    #     """Get the path to the Stan program for this model."""
    #     return self.stan_executable_path + ".stan"

    # @property
    # def stan_program(self) -> str:
    #     """Get the Stan program for this model."""
    #     # Join steps that have contents
    #     return "\n".join(
    #         f"{key} {{{self.steps[key]}\n}}"
    #         for key, val in self.steps.items()
    #         if len(val.strip()) > 0
    #     )
