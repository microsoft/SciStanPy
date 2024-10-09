"""Holds code for interfacing with the Stan modeling language."""

import os.path
import weakref

from tempfile import TemporaryDirectory
from typing import Any, Optional

import numpy as np

from cmdstanpy import CmdStanModel

import dms_stan as dms


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
        self.steps = {
            "data": [],
            "parameters": [],
            "transformed parameters": [],
            "model": "",
            "generated quantities": [],
        }

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

    def _record_parameter(self, param: dms.param.Parameter):
        """Record a parameter in the steps."""
        self.steps["parameters"].append(param.stan_parameter_declaration)

    def _record_transformed_parameter(self, param: dms.param.TransformedParameter):
        """Record a transformed parameter in the steps."""
        self.steps["transformed parameters"].append(param.stan_parameter_declaration)

    def _write_parameters(self):
        """Write the parameters of the model."""

        # We need a set for the parameters that have been written
        written_parameters = set()

        # Loop over the togglable parameters. This takes us from hyperparameters
        # to observables (top to bottom)
        for togglable in self.dms_stan_model.togglable_params.values():

            # Make sure this parameter has not been written yet
            assert togglable not in written_parameters

            # Record the parameter
            self._record_parameter(togglable)
            written_parameters.add(togglable)

            # Recurse through the children of the togglable parameter and add them
            # to the steps as well
            for _, child, _ in togglable.recurse_children():

                # If the child has been handled already or it is an observable, skip it
                if child in written_parameters or child.observable:
                    continue

                # If the child is a parameter, add it to the parameters list
                if isinstance(child, dms.param.Parameter):
                    self._record_parameter(child)

                # If it is a transformed parameter, add it to the transformed parameters
                # list
                elif isinstance(child, dms.param.TransformedParameter):
                    self._record_transformed_parameter(child)

                # Otherwise, raise an error
                else:
                    raise ValueError(f"Unknown child type {type(child)}")

                # Note that the child has been written
                written_parameters.add(child)

    def _write_data(self):
        """Write the data of the model."""
        # There can only be one observable
        if len(self.dms_stan_model.observables) > 1:
            raise ValueError(
                "Compilation to Stan currently only supports one observable."
            )

        # Add observables to the data
        param: "dms.param.Parameter" = self.dms_stan_model.observables[0]
        self.steps["data"].append(param.stan_parameter_declaration)
        self.data_inputs.append(param.model_varname)

        # Now add the constants
        for name, constant in self.dms_stan_model.constant_dict.items():

            # Get the data type
            dtype = "real" if isinstance(constant.value.dtype, np.floating) else "int"

            # Get the declaration
            declaration = f"{dtype} {name}"

            # If a scalar, the declaration needs no modification
            if constant.value.ndim == 0:
                self.steps["data"].append(declaration)

            # Otherwise, it is part of an array
            else:
                str_shape = [str(s) for s in constant.value.shape]
                self.steps["data"].append(f"array[{','.join(str_shape)}] {declaration}")

            # Add the constant to the data inputs
            self.data_inputs.append(name)

    def _write_model(self):
        """Write the model of the Stan code."""
        # We need one less nested for-loops than the number of dimensions in the
        # observable
        observable: dms.param.Parameter = self.dms_stan_model.observables[0]
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
