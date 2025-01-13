"""This module is used for building and displaying prior predictive checks."""

import re

from copy import deepcopy
from typing import Optional

import hvplot.interactive
import hvplot.xarray
import numpy as np
import numpy.typing as npt
import pandas as pd
import panel as pn
import panel.widgets as pnw
import xarray as xr

from param.parameterized import Event

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

        # Initialize widgets.
        self.float_sliders = self._init_float_sliders()
        self.target_dropdown = pnw.Select(
            name="Viewed Parameter",
            options=list(self.model.named_model_components_dict),
            value=self.model.observables[0].model_varname,
        )
        self.dependent_dim_dropdown = pnw.Select(
            name="Dependent Dimension", options=[], value=None
        )
        self.independent_var_dropdown = pnw.Select(
            name="Independent Variable", options=[], value=""
        )
        self.independent_dim_dropdown = pnw.Select(
            name="Independent Dimension", options=[], value=""
        )
        self.plot_type_dropdown = pnw.Select(name="Plot Type", options=[], value="")
        self.draw_seed_entry = pnw.IntInput(name="Seed", value=1025)
        self.draw_entry = pnw.IntInput(name="Number of Experiments", value=1)
        self.update_button = pnw.Button(name="Update Model", button_type="primary")

        # Build the data generation function
        self.plotting_data = hvplot.bind(
            self._generate_data,
            update=self.update_button,
            n_draws=self.draw_entry,
            seed=self.draw_seed_entry,
        ).interactive()

        # Define callbacks needed for updating reactive components. The options
        # for the independent variable depend on the target selected. The options
        # for the independent dimension depend on the target and independent variable
        # selected. The options for the dependent dimension depend on the independent
        # dimension and independent variable. The options for the plot type depend
        # on whether there are valid independent and dependent dimensions.
        self.target_dropdown.param.watch(self.set_dependent_dim_options, "value")
        self.dependent_dim_dropdown.param.watch(
            self.set_independent_var_options, ["value", "options"]
        )
        self.independent_var_dropdown.param.watch(
            self.set_independent_dim_options, ["value", "options"]
        )
        self.dependent_dim_dropdown.param.watch(
            self.set_plot_type_options, ["value", "options"]
        )
        self.independent_var_dropdown.param.watch(
            self.set_plot_type_options, ["value", "options"]
        )

        # Set initial values for reactive components
        self.target_dropdown.param.trigger("value")

    def _init_float_sliders(self) -> dict[str, pnw.EditableFloatSlider]:
        """Gets the float sliders for the togglable parameters in the model."""
        # Each togglable parameter gets its own float slider
        sliders = {}

        # Process all constants in the model
        for (
            hyperparam_name,
            hyperparam_val,
        ) in self.model.all_model_components_dict.items():

            # Skip non-constants and non-togglable parameters
            if (
                not isinstance(hyperparam_val, Constant)
                or not hyperparam_val.is_togglable
            ):
                continue

            # If no dimensions, just create a slider
            if hyperparam_val.ndim == 0:
                sliders[hyperparam_name] = pnw.EditableFloatSlider(
                    name=hyperparam_name,
                    value=hyperparam_val.value.item(),
                    start=hyperparam_val.slider_start,
                    end=hyperparam_val.slider_end,
                    step=hyperparam_val.slider_step_size,
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
                sliders[name] = pnw.EditableFloatSlider(
                    name=name,
                    value=slider_val,
                    start=hyperparam_val.slider_start,
                    end=hyperparam_val.slider_end,
                    step=hyperparam_val.slider_step_size,
                )

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

    def _update_model(self) -> None:
        """Updates the underlying model with the new constant values."""
        for paramname, slider in self.float_sliders.items():

            # Get the parameter name and the indices
            paramname, indices = _INDEX_EXTRACTOR.match(paramname).groups()
            indices = tuple(map(int, indices.split(","))) if indices else ()

            # The parameter must be a constant
            assert isinstance(self.model[paramname], Constant)

            # Update the value of the constant
            self.model[paramname].value[indices] = slider.value

    def _generate_data(self, update: bool, n_draws: int, seed: int) -> xr.Dataset:
        """
        Updates the model with the new constant values and rebuilds the plotting
        dataframe. The key of each kwarg gives the name of the parameter to update,
        and the value is a dictionary that links the constant names within that
        parameter to the new values for those constants.
        """
        # Set button to loading mode
        self.update_button.loading = True

        # Update the model if the function was triggered by the "update" button
        if update:
            self._update_model()

        # Redraw the data
        draw = self.model.draw(n=n_draws, named_only=True, as_xarray=True, seed=seed)

        # No more loading
        self.update_button.loading = False

        return draw

    def _get_dims(self, component_name: str) -> set[str]:
        """Gets the dimensions of the component variable."""
        return set(self.plotting_data.eval()[component_name].dims[1:])

    def set_dependent_dim_options(self, event: Event) -> None:
        """
        Sets the dependent dimension options based on the selected target. This
        is just all dimensions available to the target variable.
        """
        # Get the dimensions of the target variable
        target_dim_opts = [""] + list(self._get_dims(event.new))

        # If the previous dependent dimension is not in the options, reset it
        if self.dependent_dim_dropdown.value not in target_dim_opts:
            self.dependent_dim_dropdown.value = target_dim_opts[-1]

        # Update the dropdown options
        self.dependent_dim_dropdown.options = target_dim_opts

    def set_independent_var_options(self, event: Event) -> None:
        """Sets the independent variable options based on the selected target."""
        # Get the dimensions of the target variable
        target_dims = self._get_dims(self.target_dropdown.value)

        # Remove the dependent dimension from the target dimensions if it is set
        target_dims.discard(self.dependent_dim_dropdown.value)

        # The independent variable must be a coordinate with at least one dimension
        # overlapping with the target variable that is not selected as the current
        # dependent dimension.
        independent_var_opts = [""] + [
            coord
            for coord, arr in self.plotting_data.coords.eval().items()
            if len(target_dims.intersection(arr.dims[1:])) >= 1
        ]

        # If the previous independent variable is not in the options, reset it
        if self.independent_var_dropdown.value not in independent_var_opts:
            self.independent_var_dropdown.value = ""

        # Update the dropdown options
        self.independent_var_dropdown.options = independent_var_opts

    def set_independent_dim_options(self, event: Event) -> None:
        """
        Sets the independent dimension options based on the selected independent
        variable and target. Specifically, the independent dimension must exist
        in both the selected variable and the target variable.
        """
        # Get the dimensions of the target variable
        target_dims = self._get_dims(self.target_dropdown.value)

        # If the independent variable is not set, there are no options. Otherwise,
        # it is all dimensions that are in both the target and independent variable
        # but that are not the dependent dimension.
        if self.independent_var_dropdown.value == "":
            independent_dim_opts = [""]
        else:
            independent_dim_opts = list(
                target_dims.intersection(
                    self._get_dims(self.independent_var_dropdown.value)
                )
                - {self.dependent_dim_dropdown.value}
            )
            independent_dim_opts = (
                independent_dim_opts if len(independent_dim_opts) > 0 else [""]
            )

        # If the previous independent dimension is not in the options, reset it
        if self.independent_dim_dropdown.value not in independent_dim_opts:
            self.independent_dim_dropdown.value = independent_dim_opts[-1]

        # Update the dropdown options
        self.independent_dim_dropdown.options = independent_dim_opts

    def set_plot_type_options(self, event: Event) -> None:
        """
        Sets the options for the plot types depending on the selected dimensions.
        ECDF is always an option, as is KDE. Violin plots are allowed if a dependent
        dimension is selected. Relationship plots are allowed if both an independent
        variable and dimension are selected along with a dependent dimension.
        """
        # We can always have an ECDF and KDE plot
        plot_type_opts = ["ECDF", "KDE"]
        default_plot = "ECDF"

        # If a dependent dimension is set, then we can also have violin plots
        if self.dependent_dim_dropdown.value != "":
            plot_type_opts.append("Violin")
            default_plot = "Violin"

        # If an independent variable and dimension are set, then we can also have
        # relationship plots
        if (self.independent_dim_dropdown.value != "") and (
            self.independent_var_dropdown.value != ""
        ):
            plot_type_opts.append("Relationship")
            default_plot = "Relationship"

        # Update to the default plot type
        self.plot_type_dropdown.value = default_plot
        self.plot_type_dropdown.options = plot_type_opts

    def display(self) -> pn.WidgetBox:
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
        # Organize widgets
        widgets = pn.WidgetBox(
            pn.WidgetBox(
                "# Model Hyperparameters",
                *self.float_sliders.values(),
            ),
            pn.WidgetBox(
                "# Viewing Options",
                self.target_dropdown,
                self.dependent_dim_dropdown,
                self.independent_var_dropdown,
                self.independent_dim_dropdown,
                self.plot_type_dropdown,
                self.draw_seed_entry,
                self.draw_entry,
                self.update_button,
            ),
        )

        # Build the plot
        # plot = plotting_func(plot_df, paramname=plot_df.columns[0])

        # Organize the display
        return widgets
