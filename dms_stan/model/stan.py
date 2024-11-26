"""Holds code for interfacing with the Stan modeling language."""

import os.path
import weakref

from tempfile import TemporaryDirectory
from typing import Any, Optional, TypedDict, Union

from cmdstanpy import CmdStanModel

import dms_stan as dms

from .components import Constant, Parameter, TransformedParameter
from .components.abstract_model_component import AbstractModelComponent


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

        # There can only be one observable
        if len(model.observables) > 1:
            raise ValueError(
                "Compilation to Stan currently only supports one observable."
            )

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
            weakref.finalize(self, TemporaryDirectory.cleanup)
            output_dir = tempdir.name

        # Make sure the output directory exists
        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"Output directory {output_dir} does not exist.")

        # Set the output directory
        self.output_dir = output_dir

    def _record_data(self, data: Union[Parameter, Constant]):
        """Record a data object in the steps."""
        self.steps["data"].append(data.stan_parameter_declaration)
        self.data_inputs.append(data.model_varname)

    def _record_parameter(self, param: Parameter):
        """Record a parameter in the steps."""
        self.steps["parameters"].append(param.stan_parameter_declaration)

    def _record_transformed_parameter(self, param: TransformedParameter):
        """Record a transformed parameter in the steps."""
        self.steps["transformed parameters"].append(param.stan_parameter_declaration)

    def _declare_variables(self):
        """Write the parameters, transformed parameters, and data of the model."""

        # We only have one observable
        assert len(self.dms_stan_model.observables) == 1
        observable = self.dms_stan_model.observables[0]

        # Record the observable in the data block
        self._record_data(observable)
        self.all_varnames.add(observable.model_varname)
        observed_nodes: set[AbstractModelComponent] = {observable}

        # Now loop over the parents of the observable and add them to the appropriate
        # blocks
        for child, parent in observable.walk_tree(False):

            # The parent cannot be an observable
            assert not parent.observable, "Parent cannot be an observable."

            # If the parent has been handled already, skip it.
            if parent in observed_nodes:
                continue

            # Check for name collisions
            if parent.model_varname in self.all_varnames:
                raise ValueError(f"Name collision for {parent.model_varname}")

            # If the parent is a parameter, add it to the parameters list
            if isinstance(parent, Parameter):
                self._record_parameter(parent)

            # If the parent is a constant and the child is a parameter, add it to
            # the data block
            elif isinstance(parent, Constant):
                if isinstance(child, Parameter):
                    self._record_data(parent)
                else:
                    continue

            # If it is a transformed parameter, add it to the transformed parameters
            # list
            elif isinstance(parent, TransformedParameter):
                self._record_transformed_parameter(parent)

            # Otherwise, raise an error
            else:
                raise ValueError(f"Unknown node type {type(parent)}")

            # Note that the child has been written
            self.all_varnames.add(parent.model_varname)
            observed_nodes.add(parent)

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
        self._declare_variables()

    def write_stan_code(self) -> str:
        """Get the Stan code for this model."""
