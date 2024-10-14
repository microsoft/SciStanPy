"""Holds code for interfacing with the Stan modeling language."""

import os.path
import weakref

from tempfile import TemporaryDirectory
from typing import Any, Optional, TypedDict

from cmdstanpy import CmdStanModel

import dms_stan as dms

from dms_stan.model.components.parameters import Parameter
from dms_stan.model.components.transformed_parameters import TransformedParameter


# We need a specific type for the steps of the Stan model
class StanStepsType(TypedDict):
    """Type for the steps of the Stan model."""

    data: list[str]
    parameters: list[str]
    transformed_parameters: list[str]
    model: str
    generated_quantities: list[str]


class StanModel(CmdStanModel):

    # We initialize with a DMSStan model instance
    def __init__(
        self,
        model: "dms.model.Model",
        output_dir: Optional[str] = None,
        stanc_options: Optional[dict[str, Any]] = None,
        cpp_options: Optional[dict[str, Any]] = None,
        user_header: Optional[str] = None,
    ):
        # Set default options
        stanc_options = stanc_options or {}
        cpp_options = cpp_options or {}

        # Note the underlying DMSStan model
        self.dms_stan_model = model

        # A dictionary stores the steps of the program
        self.steps: StanStepsType = {
            "data": [],
            "parameters": [],
            "transformed parameters": [],
            "model": "",
            "generated quantities": [],
        }

        # All variable names in the model
        self.all_varnames: set[str] = set()

        # Required data for the Stan model
        self.data_inputs: list[str] = []

        # Build the steps
        self._build_steps()

        # Set the output directory
        self._set_output_dir(output_dir)

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

    def _record_parameter(self, param: Parameter):
        """Record a parameter in the steps."""
        self.steps["parameters"].append(param.stan_parameter_declaration)

    def _record_transformed_parameter(self, param: TransformedParameter):
        """Record a transformed parameter in the steps."""
        self.steps["transformed parameters"].append(param.stan_parameter_declaration)

    def _write_parameters(self):
        """Write the parameters of the model."""

        # Loop over the togglable parameters. This takes us from hyperparameters
        # to observables (top to bottom)
        for togglable in self.dms_stan_model.togglable_params.values():

            # Make sure this parameter has not been written yet
            assert togglable not in self.all_varnames

            # Record the parameter
            self._record_parameter(togglable)
            self.all_varnames.add(togglable.model_varname)

            # Recurse through the children of the togglable parameter and add them
            # to the steps as well
            for _, child, _ in togglable.recurse_children():

                # If the child has been handled already or is an observable skip it
                if child.model_varname in self.all_varnames or child.observable:
                    continue

                # If the child is a parameter, add it to the parameters list
                if isinstance(child, Parameter):
                    self._record_parameter(child)

                # If it is a transformed parameter, add it to the transformed parameters
                # list
                elif isinstance(child, TransformedParameter):
                    self._record_transformed_parameter(child)

                # Otherwise, raise an error
                else:
                    raise ValueError(f"Unknown child type {type(child)}")

                # Note that the child has been written
                self.all_varnames.add(child.model_varname)

    def _write_data(self):
        """Write the data of the model."""
        # Define types
        param_declarations: tuple[str, ...]
        varnames: tuple[str, ...]

        # There can only be one observable
        if len(self.dms_stan_model.observables) > 1:
            raise ValueError(
                "Compilation to Stan currently only supports one observable."
            )

        # Get variable names and parameter declarations for the observables and
        # constants
        param_declarations, varnames = zip(
            *[
                (obj.stan_parameter_declaration, obj.model_varname)
                for obj in (
                    self.dms_stan_model.observables + self.dms_stan_model.constants
                )
            ]
        )

        # Add the parameter declarations to the data steps
        self.steps["data"].extend(param_declarations)

        # Update the data inputs
        self.data_inputs.extend(varnames)

        # Any data input is also a reserved word. There should be no overlap between
        # the two sets
        if (di_set := set(self.data_inputs)).intersection(self.all_varnames):
            raise AssertionError(
                "Name collision between data inputs and variable names."
            )

        # Add the data inputs to the reserved words
        self.all_varnames.update(di_set)

    def _write_model(self):
        """Write the model of the Stan code."""
        # We need one less nested for-loops than the number of dimensions in the
        # observable
        observable: Parameter = self.dms_stan_model.observables[0]
        n_for_loops = observable.ndim - 1

        # We begin the model as a list in the inner loop
        model = []

        # Add the observable to the model. Note that we assume the last dimension
        # is vectorized.
        # TODO: Any 1D arrays, vectors, or scalars can be directly defined in the
        # model outside of for-loops. We can go top-down to get these.
        # TODO: Any ND arrays can be defined in the model inside of for-loops. We
        # can go top-down to get these too.

    def _build_steps(self):

        # Record the parameters first
        self._write_parameters()

        # Now data
        self._write_data()

    def write_stan_code(self) -> str:
        """Get the Stan code for this model."""
