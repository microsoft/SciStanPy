"""This module is used for building and displaying prior predictive checks."""

import re

from copy import deepcopy
from typing import Optional

import hvplot.interactive
import hvplot.pandas
import numpy as np
import numpy.typing as npt
import pandas as pd
import panel.widgets as pnw

import dms_stan as dms

from .components.constants import Constant
from .dms_stan_model import Model

# We need a regular expression for separating the variable name from its indices
_INDEX_EXTRACTOR = re.compile(r"([A-Za-z0-9_]+)\[?([0-9, ]*)\]?")


class PriorPredictiveCheck:
    """Base class for prior predictive checks."""

    def __init__(self, model: Model, copy_model: bool = False):

        # Copy the model if requested. If we don't copy, then we can modify our
        # values on the model directly.
        self.model = deepcopy(model) if copy_model else model

    def build_plotting_df(
        self,
        displayed_param: str,
        n_experiments: int,
        independent_dim: Optional[int] = None,
        independent_labels: Optional[npt.NDArray] = None,
    ) -> pd.DataFrame:
        """
        Builds the dataframes that will be used for plotting the prior predictive
        check. The dataframes are built from the samples drawn from the model.
        """
        # Sampling prepends a dimension, so we need to adjust the independent dimension
        if independent_dim is not None and independent_dim >= 0:
            independent_dim += 1

        # Get the samples from the model and build the dataframe
        return dms.plotting.build_plotting_df(
            samples=getattr(self.model, displayed_param).draw(n_experiments)[0],
            paramname=displayed_param,
            independent_dim=independent_dim,
            independent_labels=independent_labels,
        )

    def _process_display_inputs(
        self,
        initial_view: Optional[str],
        independent_dim: Optional[int],
        independent_labels: Optional[npt.NDArray],
    ) -> tuple[str, list[str]]:
        """Processes the inputs for the display method."""
        # If independent labels are provided, make sure that the independent dimension
        # is also provided
        if independent_labels is not None and independent_dim is None:
            raise ValueError("Independent labels require an independent dimension")

        # Get the list of parameters and observables in the model
        legal_targets = self.model.parameter_dict.copy()
        error_message = "The model has no parameters or observables"

        # If there is an independent dimension provided, filter down to just those
        # parameters that have that dimension
        if independent_dim is not None:

            # Determine the number of dimensions needed for the independent dimension
            dims_needed = (
                abs(independent_dim) if independent_dim < 0 else independent_dim + 1
            )
            legal_targets = {
                k: v for k, v in legal_targets.items() if v.ndim >= dims_needed
            }
            error_message += f" with an independent dimension of {independent_dim}"

            # If there are independent labels provided, filter down to just those parameters
            # that match the length of the independent labels
            if independent_labels is not None:
                legal_targets = {
                    k: v
                    for k, v in legal_targets.items()
                    if v.shape[independent_dim] == len(independent_labels)
                }
                error_message += f" and {len(independent_labels)} independent labels"

        # Raise an error if there are no legal parameters
        if not legal_targets:
            raise ValueError(error_message)

        # If the initial dim is provided, make sure that it is a legal parameter
        # Otherwise, just take the first observable if it exists
        if initial_view is not None and initial_view not in legal_targets:
            raise ValueError(
                error_message.replace(
                    "observables", f"observables named {initial_view}"
                )
            )
        else:
            opts = [
                observable.model_varname
                for observable in self.model.observables
                if observable.model_varname in legal_targets
            ]
            initial_view = (
                opts[0] if len(opts) > 0 else legal_targets[legal_targets.keys()[0]]
            )

        return initial_view, list(legal_targets)

    def _init_float_sliders(self) -> dict[str, pnw.EditableFloatSlider]:
        """Gets the float sliders for the togglable parameters in the model."""
        # Each togglable parameter gets its own float slider
        sliders = {}

        # Process all constants in the model
        for hyperparam_name, hyperparam_val in self.model.constants_dict.items():
            # Skip non-togglable parameters
            if not hyperparam_val.is_togglable:
                continue

            # If no dimensions, just create a slider
            if hyperparam_val.ndim == 0:
                sliders[hyperparam_name] = pnw.EditableFloatSlider(
                    name=hyperparam_name, value=hyperparam_val.value.item()
                )
                continue

            # Otherwise, create a slider for each entry
            target_shape = hyperparam_val.shape
            remaining_elements = np.prod(target_shape)
            current_index = [0] * len(target_shape)
            current_dim = len(target_shape) - 1
            while remaining_elements > 0:

                # Build the slider name
                name = f"{hyperparam_name}[{', '.join(map(str, current_index))}]"

                # Get the slider value
                slider_val = hyperparam_val.value[tuple(current_index)]

                # Add the slider
                sliders[name] = pnw.EditableFloatSlider(name=name, value=slider_val)

                # Increment the current index. If we have reached the end of the
                # current dimension, move to the next dimension
                if current_index[current_dim] == target_shape[current_dim] - 1:
                    current_index[current_dim:] = [0] * (
                        len(target_shape) - current_dim
                    )
                    current_dim -= 1

                # Decrement the remaining elements. Increment the current index
                remaining_elements -= 1
                current_index[current_dim] += 1

        return sliders

    def _init_target_dropdown(
        self, initial_view: str, legal_targets: list[str]
    ) -> pnw.Select:
        """
        Gets the dropdown for selecting the target parameter. The target parameter
        is any named parameter or observable in the model.
        """

        # Build the dropdown
        return pnw.Select(
            name="Viewed Parameter", options=legal_targets, value=initial_view
        )

    def _init_draw_slider(self) -> pnw.EditableIntSlider:
        """Gets the slider for the number of draws to use in the prior predictive check."""
        return pnw.EditableIntSlider(
            name="Number of Experiments", value=1, start=1, end=100
        )

    # We need a function that updates the model with new parameters
    def _viewer_backend(
        self,
        displayed_param: str,
        n_experiments: int,
        independent_dim: Optional[int],
        independent_labels: Optional[npt.NDArray],
        **kwargs: float,
    ):
        """
        Updates the model with the new constant values and rebuilds the plotting
        dataframe. The key of each kwarg gives the name of the parameter to update,
        and the value is a dictionary that links the constant names within that
        parameter to the new values for those constants.
        """
        # Update the model with the new parameters
        for paramname, paramval in kwargs.items():

            # Get the parameter name and the indices
            paramname, indices = _INDEX_EXTRACTOR.match(paramname).groups()
            indices = tuple(map(int, indices.split(","))) if indices else ()

            # The parameter must be a constant
            assert isinstance(self.model[paramname], Constant)

            # Update the value of the constant
            self.model[paramname].value[indices] = paramval

        # Build the dataframes for plotting
        return self.build_plotting_df(
            displayed_param=displayed_param,
            n_experiments=n_experiments,
            independent_dim=independent_dim,
            independent_labels=independent_labels,
        )

    def display(
        self,
        initial_view: Optional[str] = None,
        independent_dim: Optional[int] = None,
        independent_labels: Optional[npt.NDArray] = None,
    ) -> hvplot.interactive.Interactive:
        """
        Renders a display of samples drawn from the parameter given by `initial_view`.

        Args:
            initial_view (Optional[str]): The name of the parameter to display when
                initializing the display. If not provided, the observable in the
                model is displayed.

            independent_dim (Optional[int]): The dimension of the data that defines
                the independent variable, if any. If not provided, an ECDF and KDE
                plot is displayed. If provided without `independent_labels`, then
                a series of ECDFs and a series of violin plots are displayed, grouped
                by the independent dimension. If provided with `independent_labels`,
                then a series of violin plots are displayed, again grouped by the
                independent dimension, but now spaced according to the labels. This
                is a way of looking at, e.g., how distributions change over time.

            independent_labels (Optional[npt.NDArray]): The labels for the independent
                dimension. This must be the same length as the size of the independent
                dimension. If not provided, the independent dimension is treated
                as a simple index.
        """
        # Choose the appropriate plotting function
        plotting_func = dms.plotting.choose_plotting_function(
            independent_dim=independent_dim, independent_labels=independent_labels
        )

        # Process the input parameters
        initial_view, legal_targets = self._process_display_inputs(
            initial_view=initial_view,
            independent_dim=independent_dim,
            independent_labels=independent_labels,
        )

        # Build widgets for the display
        float_sliders = self._init_float_sliders()
        target_dropdown = self._init_target_dropdown(
            initial_view=initial_view, legal_targets=legal_targets
        )
        draw_slider = self._init_draw_slider()

        # Bind the widgets to the viewer backend
        plot_df = hvplot.bind(
            self._viewer_backend,
            displayed_param=target_dropdown,
            n_experiments=draw_slider,
            independent_dim=independent_dim,
            independent_labels=independent_labels,
            **float_sliders,
        ).interactive()

        # Make the plot
        return plotting_func(plot_df, paramname=target_dropdown)
