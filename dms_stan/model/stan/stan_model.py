"""Holds code for interfacing with the Stan modeling language."""

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
    TypeVar,
    Union,
)

import numpy as np
import numpy.typing as npt

from cmdstanpy import CmdStanMCMC, CmdStanModel, format_stan_file

import dms_stan as dms

from dms_stan.custom_types import SampleType
from dms_stan.defaults import (
    DEFAULT_CPP_OPTIONS,
    DEFAULT_FORCE_COMPILE,
    DEFAULT_STANC_OPTIONS,
    DEFAULT_USER_HEADER,
)

from dms_stan.model.components import Constant, Normal, Parameter, TransformedParameter
from dms_stan.model.components.abstract_model_component import AbstractModelComponent
from dms_stan.model.stan.stan_results import SampleResults

# Function for combining a list of Stan code lines
DEFAULT_INDENTATION = 4

# Parameter and return types for decorated functions
P = ParamSpec("P")
R = TypeVar("R")


class StanCodeBase(ABC, list):
    """Base object for Stan code."""

    def __init__(self, parent_loop: Optional["StanCodeBase"]):
        """Initializes the Stan code base."""
        # Initialize the list
        super().__init__()

        # Record the parent loop
        self.parent_loop = parent_loop

    def recurse_for_loops(self) -> Generator["StanForLoop", None, None]:
        """Generates all for-loops in the program."""
        # Yield this loop
        yield self

        # Loop over all nested loops and yield from them as well
        for loop in self.nested_loops:
            yield from loop.recurse_for_loops()

    def recurse_model_components(
        self,
    ) -> Generator[AbstractModelComponent, None, None]:
        """Recursively generates all model components in the program."""
        for component in self:
            if isinstance(component, StanForLoop):
                yield from component.recurse_model_components()
            else:
                assert isinstance(component, AbstractModelComponent)
                yield component

    @abstractmethod
    def get_parent_loop(self, n: int) -> "StanCodeBase":
        """
        If rolling out the ancestry of the loop, starting from the progenitor and
        ending with this loop, return the nth loop in that list (i.e., if n = 0,
        return the progenitor, if n = -1, return this loop).
        """

    def _write_block(
        self,
        block_name: Literal["model", "transformed_parameters", "generated_quantities"],
        declarations: Union[str, tuple[str, ...]] = (),
    ) -> str:

        def filter_generated_quantities(
            nested_component: Union[StanCodeBase, AbstractModelComponent]
        ) -> bool:
            """
            Filters hierarchy of loops for the generated quantities block. We take
            observables.
            """
            return (
                isinstance(nested_component, Parameter) and nested_component.observable
            ) or isinstance(nested_component, StanForLoop)

        # Functions for filtering the tree
        def filter_model_transformed_params(
            nested_component: Union[StanCodeBase, AbstractModelComponent]
        ) -> bool:
            """
            Filters hierarchy of loops for the model and transformed parameters
            blocks. We take Parameters and named TransformedParameters.
            """
            return isinstance(nested_component, (Parameter, StanForLoop)) or (
                isinstance(nested_component, TransformedParameter)
                and nested_component.is_named
            )

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
        }

        # Get the function, prefix, and filter for the block
        func = dispatcher[block_name]["func"]
        prefix = dispatcher[block_name]["prefix"]
        filter_func = dispatcher[block_name]["filter"]

        # Get assignments and incrementations
        assignments = [
            addition
            for component in self
            if filter_func(component)
            and (addition := getattr(component, func)(self.allowed_index_names))
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
            + self.combine_lines(
                assignments, indentation_level=self.stan_code_level + 1
            )
            + "\n"
            + self.finalize_line("}")
        )

    def get_target_incrementation(
        self, _dummy: Optional[tuple[str, ...]] = None
    ) -> str:
        """Returns the Stan code for the target incrementation."""
        return self._write_block("model")

    def get_transformation_assignment(
        self, declarations: Union[str, tuple[str, ...]]
    ) -> str:
        """Returns the Stan code for the transformation assignment."""

        return self._write_block("transformed_parameters", declarations=declarations)

    def get_generated_quantities(
        self, declarations: Union[str, tuple[str, ...]]
    ) -> str:
        """Returns the Stan code for the generated quantities."""
        return self._write_block("generated_quantities", declarations=declarations)

    def finalize_line(self, text: str, indendation_level: Optional[int] = None) -> str:
        """Indents a block of text by a specified number of spaces and adds semicolons."""

        # Get the indentation level
        indendation_level = (
            self.stan_code_level if indendation_level is None else indendation_level
        )

        # Pad the input text with spaces
        formatted = f"{' ' * DEFAULT_INDENTATION * indendation_level}{text}"

        # Add a semicolon to the end if not a bracket or blank
        if text and text[-1] not in {"{", "}", ";"}:
            formatted += ";"

        return formatted

    def combine_lines(
        self, lines: list[str], indentation_level: Optional[int] = None
    ) -> str:
        """Combine a list of Stan code lines into a single string."""

        # Nothing if no lines
        if len(lines) == 0:
            return ""

        # Combine the lines
        return "\n".join(
            self.finalize_line(el, indendation_level=indentation_level) for el in lines
        )

    @property
    def nested_loops(self):
        """Returns the nested loops."""
        return [component for component in self if isinstance(component, StanForLoop)]

    @property
    def model_components(self) -> list[AbstractModelComponent]:
        """Returns the model components in the program."""
        return [
            component
            for component in self
            if isinstance(component, AbstractModelComponent)
        ]

    @property
    @abstractmethod
    def stan_code_level(self) -> int:
        """Returns the code level of the object."""

    @property
    @abstractmethod
    def allowed_index_names(self) -> tuple[str, ...]:
        """Returns the allowed index names for the loop."""

    @property
    @abstractmethod
    def target_inc_prefix(self) -> str:
        """Returns the prefix for the target incrementation."""

    @property
    @abstractmethod
    def transformation_assi_prefix(self) -> str:
        """Returns the prefix for the transformation assignment."""

    @property
    @abstractmethod
    def generated_quantities_prefix(self) -> str:
        """Returns the prefix for the generated quantities."""


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

        # Append this loop to the parent loop
        self.parent_loop.append(self)

        # Get the ancestry of the loop
        self.ancestry = self._get_ancestry()

        # The shape index is the initial code level of the loop minus 1. In other
        # words, if the loop is at code level "1", then we are indexing the first
        # dimension of the model components (index 0).
        self._shape_index = self.stan_code_level - 1

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

    def squash(self) -> None:
        """
        If this is a singleton loop, it is removed from the lineage. This is done
        by moving the nested loops to the parent loop and removing this loop.
        """
        # If the end is 1, we are a singleton loop. Move the contents of this loop
        # to the parent loop and then remove this loop from the parent loop.
        if self.end == 1:
            self.parent_loop.extend(self)
            self.parent_loop.remove(self)
            self.parent_loop = None

    def append(self, component: Union["StanForLoop", AbstractModelComponent]) -> None:
        """Appends a component to the list."""
        # If a model component, make sure we are at the appropriate code level
        if isinstance(component, AbstractModelComponent):
            assert component.stan_code_level == (self.shape_index + 1)

        # Append the component
        super().append(component)

    @property
    def end(self) -> int:
        """
        The end value of the loop. This is the size of all non-loop contents at
        the index level.
        """
        # Get the size of the dimension at the index level for all model components
        # nested in the loop
        all_ends = {
            component.shape[self.shape_index]
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
        """Returns the program to which this loop belongs"""
        return self.ancestry[0]

    @property
    def allowed_index_names(self) -> tuple[str, ...]:
        """Returns the allowed index names for the loop."""
        return self.program.allowed_index_names

    @property
    def stan_code_level(self) -> int:
        """Returns the code level of the loop, which is the number of ancestors."""
        n_ancestors = len(self.ancestry)
        assert n_ancestors > 0
        return n_ancestors

    @property
    def loop_index(self) -> str:
        """Returns the index of the loop in the program."""
        return self.program.allowed_index_names[self.shape_index]

    @property
    def target_inc_prefix(self) -> str:
        return self.finalize_line("") + f"for ({self.loop_index} in 1:{self.end}) {{"

    @property
    def transformation_assi_prefix(self) -> str:
        return self.target_inc_prefix

    @property
    def generated_quantities_prefix(self) -> str:
        return self.target_inc_prefix

    @property
    def shape_index(self) -> int:
        """Returns the index level of the loop."""
        return self._shape_index


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

        # Get the names of all variables, parameters, and data inputs for the model
        (
            self.all_varnames,
            self.all_paramnames,
            self.autogathered_data,
            self.user_provided_data,
        ) = self.get_varnames()

        # Get allowed index variable names for each level
        self._allowed_index_names = tuple(
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

        # Get all constants, named or otherwise
        constants = list(
            filter(lambda x: isinstance(x, Constant), self.model.all_model_components)
        )

        # Starting from constants (which are all root nodes), walk down the tree
        # and record the depth of each node
        for root in constants:

            # Root nodes are at depth 0
            node_to_depth[root] = 0

            # We should never encounter a node twice from a single root
            observed_nodes: set[AbstractModelComponent] = set()

            # Walk the tree rooted at this root and record the depth of each node
            # if it is higher than the current maximum depth
            for depth, _, child in root.walk_tree():

                # Make sure we haven't seen this node before
                assert child not in observed_nodes, "Node encountered twice in tree"
                observed_nodes.add(child)

                # Record the maximum depth of this node
                node_to_depth[child] = max(depth, node_to_depth.get(child, 0))

        return node_to_depth

    def get_varnames(self) -> tuple[set[str], set[str], set[str], set[str]]:
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
            if isinstance(node, Parameter) and not node.observable
        }

        # Get all data inputs
        auto_gathered_data = {
            node.model_varname
            for node in self.node_to_depth
            if isinstance(node, Constant)
        }
        user_provided_data = {
            node.model_varname
            for node in self.node_to_depth
            if isinstance(node, Parameter) and node.observable
        }

        return all_varnames, all_paramnames, auto_gathered_data, user_provided_data

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
                n_loops = component.stan_code_level - target_loop.stan_code_level
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
        """Yields from the nested loops only."""
        for loop in self.nested_loops:
            yield from loop.recurse_for_loops()

    def append(self, component: Union["StanForLoop", AbstractModelComponent]) -> None:
        """Appends a component to the list."""
        # If a model component, make sure we are at the appropriate code level
        if isinstance(component, AbstractModelComponent):
            assert component.stan_code_level == 0

        # Append the component
        super().append(component)

    @property
    def stan_code_level(self) -> int:
        """Returns the code level of the program, which is always 0."""
        return 0

    @property
    def data_block(self) -> str:
        """Returns the Stan code for the data block."""
        # Get the declarations for the data block. This is all observables and all
        # constants.
        declarations = [
            component.stan_parameter_declaration
            for component in self.recurse_model_components()
            if (isinstance(component, Parameter) and component.observable)
            or isinstance(component, Constant)
        ]

        # Combine declarations and wrap in the data block
        return (
            "data {\n" + self.combine_lines(declarations, indentation_level=1) + "\n}"
        )

    @property
    def parameters_block(self) -> str:
        """Returns the Stan code for the parameters block."""
        # Loop over all components recursively and define the parameters
        declarations: list[str] = []
        for component in self.recurse_model_components():

            # Normal distributions are non-centered if they are not hyperparameters
            if isinstance(component, Normal) and not component.is_hyperparameter:
                declarations.append(
                    f"{component.stan_dtype} {component.noncentered_varname}"
                )

            # Otherwise, add all parameters that are not observables
            elif isinstance(component, Parameter) and not component.observable:
                declarations.append(component.stan_parameter_declaration)

        # Combine the lines and enclose in the parameters block
        return (
            "\n"
            + "parameters {\n"
            + self.combine_lines(declarations, indentation_level=1)
            + "\n}"
        )

    @property
    def model_block(self) -> str:
        """Returns the Stan code for the model block."""
        return self.get_target_incrementation()

    @property
    def transformed_parameters_block(self) -> str:
        """Returns the Stan code for the transformed parameters block."""
        # Get the declarations for transformed parameters. We take any named transformed
        # parameters. We also take non-centered normal distributions.
        declarations = [
            component.stan_parameter_declaration
            for component in self.recurse_model_components()
            if (isinstance(component, TransformedParameter) and component.is_named)
            or (
                isinstance(component, Normal)
                and not component.is_hyperparameter
                and not component.observable
            )
        ]

        # Combine declarations
        declaration_block = self.combine_lines(declarations, indentation_level=1)

        # Combine declarations and transformations
        return self.get_transformation_assignment(declarations=declaration_block)

    @property
    def generated_quantities_block(self) -> str:
        """Returns the Stan code for the generated quantities block."""
        # Get declarations for the generated quantities block. This is all observables
        # in the program.
        declarations = [
            component.stan_generated_quantity_declaration
            for component in self.recurse_model_components()
            if isinstance(component, Parameter) and component.observable
        ]

        # Combine declarations
        declaration_block = self.combine_lines(declarations, indentation_level=1)

        # Combine declarations and transformations
        return self.get_generated_quantities(declarations=declaration_block)

    @property
    def allowed_index_names(self) -> tuple[str, ...]:
        """Returns the allowed index names for the loop."""
        return self._allowed_index_names

    @property
    def target_inc_prefix(self) -> str:
        return "model {"

    @property
    def transformation_assi_prefix(self) -> str:
        return "transformed parameters {"

    @property
    def generated_quantities_prefix(self) -> str:
        return "generated quantities {"

    @property
    def code(self) -> str:
        """Get the program code for this model."""
        # Join steps that have contents
        return "\n".join(
            val
            for val in (
                self.data_block,
                self.parameters_block,
                self.transformed_parameters_block,
                self.model_block,
                self.generated_quantities_block,
            )
            if len(val.strip()) > 0
        )


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

        # Run the wrapped function
        cmdstanmcmc = func(stan_model, **kwargs)
        assert isinstance(cmdstanmcmc, CmdStanMCMC)  # For type checking

        # Build the results object
        return SampleResults(fit=cmdstanmcmc, data=kwargs["data"])

    return inner


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

        # Set the output directory
        self._set_output_dir(output_dir)

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
            stanc_options=stanc_options,
            cpp_options=cpp_options,
            user_header=user_header,
        )

    def _set_output_dir(self, output_dir: Optional[str]) -> None:
        """Set the output directory for the model."""
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
        """
        Write the Stan model to the output directory. This will overwrite any model
        in that directory.
        """
        # Write the raw code
        with open(self.stan_program_path, "w", encoding="utf-8") as f:
            f.write(self.code())

        # Format the code
        format_stan_file(self.stan_program_path, overwrite_file=True, canonicalize=True)

    def gather_inputs(self, **observables: SampleType) -> dict[str, SampleType]:
        """
        Gathers the inputs for the Stan model. Values for observables must be provided
        by the user. All other inputs will be drawn from the DMS Stan model itself.

        Returns:
            dict[str, SampleType]: A dictionary of inputs for the Stan model.
        """
        # Make sure we have all the observables that the user must provide. Report
        # any missing or extra observables
        provided_observables = set(observables.keys())
        if missing := self.program.user_provided_data - provided_observables:
            raise ValueError(f"Missing observables: {', '.join(missing)}")
        elif extra := provided_observables - self.program.user_provided_data:
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
        observables.update(
            {name: self.model[name].value for name in self.program.autogathered_data}
        )

        # All dots in the names must be replaced with double underscores
        return {name.replace(".", "__"): obs for name, obs in observables.items()}

    def code(self) -> str:
        """Returns Stan code for the model."""
        return self.program.code

    def _get_sample_init(
        self, *, chains: int, seed: Optional[int]
    ) -> list[dict[str, Union[npt.NDArray[np.floating], np.floating]]]:
        """
        Draws from the prior distribution of the model to initialize the MCMC sampler.

        Args:
            chains (int): The number of chains.
            seed (Optional[int]): The seed for the random number generator.

        Returns:
            list[dict[str, npt.NDArray]]: A list of dictionaries where each dictionary
                contains the initial values for the parameters of the model in one
                chain.
        """
        # Draw from the prior distribution of the model, keeping only the draws
        # for the non-observable parameters
        draws = {
            component.model_varname: draw
            for component, draw in self.model.draw(
                n=chains, named_only=False, seed=seed
            ).items()
            if isinstance(component, Parameter) and not component.observable
        }

        # The draws should overlap perfectly with the parameters of the model
        assert set(draws.keys()) == self.program.all_paramnames

        # Separate the draws into one dictionary per chain
        return [{name: draw[i] for name, draw in draws.items()} for i in range(chains)]

    def sample(self, *args, **kwargs) -> tuple[CmdStanMCMC, dict[str, npt.NDArray]]:

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

        # Get the other arguments needed to build the ArViz object

        # Call the parent sample function
        return updated_parent_sample(self, **kwargs)

    # Update the CmdStanModel functions that require data
    generate_quantities = _update_cmdstanpy_func(
        CmdStanModel.generate_quantities, warn=True
    )
    laplace_sample = _update_cmdstanpy_func(CmdStanModel.laplace_sample, warn=True)
    log_prob = _update_cmdstanpy_func(CmdStanModel.log_prob, warn=True)
    optimize = _update_cmdstanpy_func(CmdStanModel.optimize, warn=True)
    pathfinder = _update_cmdstanpy_func(CmdStanModel.pathfinder, warn=True)
    variational = _update_cmdstanpy_func(CmdStanModel.variational, warn=True)

    @property
    def stan_executable_path(self) -> str:
        """Get the path to the executable for this model."""
        return os.path.join(self.output_dir, "model")

    @property
    def stan_program_path(self) -> str:
        """Get the path to the Stan program for this model."""
        return self.stan_executable_path + ".stan"
