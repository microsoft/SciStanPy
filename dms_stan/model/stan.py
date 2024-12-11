"""Holds code for interfacing with the Stan modeling language."""

import copy
import functools
import os.path
import warnings
import weakref

from tempfile import TemporaryDirectory
from typing import Any, Callable, Optional, ParamSpec, TypedDict, TypeVar, Union

from cmdstanpy import CmdStanModel

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


# TODO: Need generated quantities for observables


# We need a specific type for the steps of the Stan model
class StanStepsType(TypedDict):
    """Type for the steps of the Stan model."""

    data: Union[str, list[str]]
    parameters: Union[str, list[str]]
    transformed_parameters: Union[str, list[str]]
    model: str


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

        # There can only be one observable
        if len(model.observables) > 1:
            raise ValueError(
                "Compilation to Stan currently only supports one observable."
            )

        # Note the underlying DMSStan model
        self.model = model

        # A dictionary stores the steps of the program
        self.steps: StanStepsType = {
            "data": [],
            "parameters": [],
            "transformed parameters": [],
            "model": "",
        }

        # All variable names in the model
        self.all_varnames: set[str] = set()

        # Build the program elements
        self.data_inputs = self._declare_variables()
        self._build_code_blocks()

        # Set the output directory
        self._set_output_dir(output_dir)

        # Write the Stan program
        self.write_stan_program()

        # Initialize the CmdStanModel
        super().__init__(
            stan_file=self.stan_program_path,
            exe_file=(
                self.stan_executable_path
                if os.path.exists(self.stan_executable_path)
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

    def _declare_variables(self) -> tuple[tuple[str, bool], ...]:
        """Write the parameters, transformed parameters, and data of the model."""

        def record_data(data: Union[Parameter, Constant]) -> None:
            """Record a data object in the steps."""
            self.steps["data"].append(data.stan_parameter_declaration)
            data_inputs.append((data.model_varname, data.observable))

        def write_variable_code(variable: AbstractModelComponent) -> None:
            """Writes the Stan code for a variable."""
            # Check for name collisions
            if variable.model_varname in self.all_varnames:
                raise ValueError(f"Name collision for {variable.model_varname}")

            # Add the variable name to the set of all variable names
            self.all_varnames.add(variable.model_varname)
            observed_nodes.add(variable)

            # If an observable, do nothing
            if variable.observable:
                return

            # If the variable is a Normal distribution that is not a hyperparameter,
            # we will be non-centering. Thus, the true parameter is defined as a
            # transformed parameter and a dummy parameter is defined as a parameter.
            elif isinstance(variable, Normal) and not variable.is_hyperparameter:
                self.steps["parameters"].append(
                    f"{variable.stan_dtype} {variable.noncentered_varname}"
                )
                self.steps["transformed parameters"].append(
                    variable.stan_parameter_declaration
                )

            # If the variable is a parameter, add it to the parameters list
            elif isinstance(variable, Parameter):
                self.steps["parameters"].append(variable.stan_parameter_declaration)

            # If the variable is a constant, add it to the data block
            elif isinstance(variable, Constant):
                record_data(variable)

            # If it is a transformed parameter AND named, add it to the transformed
            # parameters list
            elif isinstance(variable, TransformedParameter):
                if variable.is_named:
                    self.steps["transformed parameters"].append(
                        variable.stan_parameter_declaration
                    )

            # Otherwise, raise an error
            else:
                raise ValueError(f"Unknown node type {type(variable)}")

        # We only have one observable
        assert len(self.model.observables) == 1
        observable = self.model.observables[0]

        # Define a set for recording observed nodes and a list for recording data
        # inputs
        observed_nodes: set[AbstractModelComponent] = set()
        data_inputs: list[tuple[str, bool]] = []

        # Record the observable as a data input and write the variable code for
        # the observable
        record_data(observable)
        write_variable_code(observable)

        # Now loop over the parents of the observable and add them to the appropriate
        # blocks
        for _, parent in observable.walk_tree(False):

            # The parent cannot be an observable
            assert not parent.observable, "Parent cannot be an observable."

            # If the parent has been handled already, skip it.
            if parent in observed_nodes:
                continue

            # Write the variable code for the parent
            write_variable_code(parent)

        # Combine lines for the blocks
        self.steps["parameters"] = combine_lines(self.steps["parameters"])
        self.steps["data"] = combine_lines(self.steps["data"])
        self.steps["transformed parameters"] = combine_lines(
            self.steps["transformed parameters"]
        )

        return tuple(data_inputs)

    def _build_code_blocks(self):
        """Builds the 'model' and 'transformed data' blocks of the Stan model."""
        # We need a way to record the level to expected index. This is handled by
        # the child function
        level_to_size: dict[int, int] = {}

        def check_level_to_size(model_component: AbstractModelComponent) -> None:

            # Do nothing if level 0. We should not need to check these as they
            # are not indexed
            if model_component.stan_code_level == 0:
                return

            # Get the dimension whose size we need to check
            corresponding_dim = obs.ndim - model_component.ndim
            assert corresponding_dim >= 0

            # Get the size of the component at the corresponding dimension and any
            # recorded size
            component_size = model_component.shape[corresponding_dim]
            retrieved_size = level_to_size.get(model_component.stan_code_level)

            # Make sure the recorded size matches the size of the component at the
            # corresponding dimension. Record the size if it has not been recorded.
            if retrieved_size is None:
                level_to_size[model_component.stan_code_level] = component_size
            elif retrieved_size != component_size:

                # If one of the two sizes is 1, we can ignore the error and set
                # the size to the larger of the two
                if retrieved_size == 1 or component_size == 1:
                    level_to_size[model_component.stan_code_level] = max(
                        retrieved_size, component_size
                    )

                # Otherwise, raise an error
                else:
                    raise AssertionError(
                        "Size mismatch at dimensions indexed by same variable: "
                        + f"{retrieved_size} != {component_size}"
                    )

        def format_code_block(levels: list[list[str]]) -> str:
            """
            Builds a block of code from a list of levels. Each level is a list of
            strings giving the code for that level. Each element of each list will
            be joined by a semi-colon and a newline. The first level is the top
            level while the other levels are nested within the previous level using
            for loops.
            """
            # Invert the contents of each level. The inversion is needed because
            # we built the levels from the bottom up.
            levels = [level[::-1] for level in levels]

            # Now build the nested 'for' loops
            code_block = combine_lines(levels[0])
            indentation = DEFAULT_INDENTATION
            for code_level, level in enumerate(levels[1:], 1):

                # Build the prefix for the 'for' loop
                index_var = allowed_index_names[code_level - 1]
                max_size = level_to_size[code_level]
                for_prefix = (
                    f"\n{' ' * indentation}for ({index_var} in 1:{max_size}) " + "{\n"
                )

                # Update the indentation
                indentation += DEFAULT_INDENTATION

                # Add the code, making sure to include the appropriate number of spaces
                code_block += for_prefix + combine_lines(level, indentation=indentation)

            # Now close the 'for' loops
            for _ in range(len(levels) - 1):
                indentation -= DEFAULT_INDENTATION
                code_block += f"\n{' ' * indentation}}}"
            assert indentation == DEFAULT_INDENTATION

            return code_block

        # Get the number of for-loop levels. This is given by the dimensionality
        # of the observable. We create lists for each level. Different variables
        # will be defined and accessed depending on the level to which they belong.
        # Note that we assume the last level is vectorized
        obs = self.model.observables[0]  # Shorthand for the observable
        model_levels = [[] for _ in range(obs.ndim)]
        transformed_param_levels = copy.deepcopy(model_levels)

        # Get allowed index variable names for each level
        allowed_index_names = tuple(
            char
            for char in dms.defaults.DEFAULT_INDEX_ORDER
            if {char, char.upper()}.isdisjoint(self.all_varnames)
        )

        # There should be enough index names to cover all levels
        if len(allowed_index_names) < (obs.ndim - 1):
            raise ValueError(
                f"Not enough index names ({len(allowed_index_names)}) to cover {obs.ndim} levels"
            )

        # Observable is automatically the last level
        assert obs.get_transformation_assignment(allowed_index_names) == ""
        model_levels[-1].append(obs.get_target_incrementation(allowed_index_names))
        check_level_to_size(obs)

        # Walk up the tree from the observable to the top level, adding variables
        # and transformations to the appropriate level
        max_level = obs.stan_code_level
        for _, parent in obs.walk_tree(walk_down=False):

            # We must always be going down a level or staying at the same level
            if parent.stan_code_level > max_level:
                raise ValueError(
                    f"Cannot go up a level from {max_level} to {parent.stan_code_level}"
                )
            max_level = parent.stan_code_level

            # If a parameter or a named transformed parameter, add target incrementation
            # and any transformations to the appropriate level
            if isinstance(parent, Parameter) or (
                isinstance(parent, TransformedParameter) and parent.is_named
            ):
                if target_incrementation := parent.get_target_incrementation(
                    allowed_index_names
                ):
                    model_levels[parent.stan_code_level].append(target_incrementation)
                if transformation_assignment := parent.get_transformation_assignment(
                    allowed_index_names
                ):
                    transformed_param_levels[parent.stan_code_level].append(
                        transformation_assignment
                    )

            # Otherwise the parent must be a constant or an unnamed transformed
            # parameter
            else:
                assert isinstance(parent, Constant) or (
                    isinstance(parent, TransformedParameter) and not parent.is_named
                )

            # Check the level to size mapping
            check_level_to_size(parent)

        # Format the code blocks
        self.steps["model"] = format_code_block(model_levels)
        self.steps["transformed parameters"] = (
            self.steps["transformed parameters"]
            + "\n"
            + format_code_block(transformed_param_levels)
        )

    def write_stan_program(self) -> None:
        """
        Write the Stan model to the output directory. This will overwrite any model
        in that directory.
        """
        with open(self.stan_program_path, "w", encoding="utf-8") as f:
            f.write(self.stan_program)

    def gather_inputs(self, **observables: SampleType) -> dict[str, SampleType]:
        """
        Gathers the inputs for the Stan model. Values for observables must be provided
        by the user. All other inputs will be drawn from the DMS Stan model itself.

        Returns:
            dict[str, SampleType]: A dictionary of inputs for the Stan model.
        """
        # Split out the observable values from the other inputs
        required_observables = {
            name for name, indicator in self.data_inputs if indicator
        }

        # Make sure we have all the observables in the inputs and report any missing
        # or extra observables
        provided_observables = set(observables.keys())
        if missing := required_observables - provided_observables:
            raise ValueError(f"Missing observables: {', '.join(missing)}")
        elif extra := provided_observables - required_observables:
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
            {
                name: self.model[name].value
                for name, indicator in self.data_inputs
                if not indicator
            }
        )

        return observables

    def code(self) -> str:
        """Just an alias for the `stan_program` property."""
        return self.stan_program

    # Update the CmdStanModel functions that require data
    generate_quantities = _update_cmdstanpy_func(
        CmdStanModel.generate_quantities, warn=True
    )
    laplace_sample = _update_cmdstanpy_func(CmdStanModel.laplace_sample, warn=True)
    log_prob = _update_cmdstanpy_func(CmdStanModel.log_prob, warn=True)
    optimize = _update_cmdstanpy_func(CmdStanModel.optimize, warn=True)
    pathfinder = _update_cmdstanpy_func(CmdStanModel.pathfinder, warn=True)
    sample = _update_cmdstanpy_func(CmdStanModel.sample)
    variational = _update_cmdstanpy_func(CmdStanModel.variational, warn=True)

    @property
    def stan_executable_path(self) -> str:
        """Get the path to the executable for this model."""
        return os.path.join(self.output_dir, "model")

    @property
    def stan_program_path(self) -> str:
        """Get the path to the Stan program for this model."""
        return self.stan_executable_path + ".stan"

    @property
    def stan_program(self) -> str:
        """Get the Stan program for this model."""
        # Join steps that have contents
        return "\n".join(
            f"{key} {{{self.steps[key]}\n}}"
            for key, val in self.steps.items()
            if len(val.strip()) > 0
        )
