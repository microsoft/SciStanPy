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

        # Initialize widgets used to generate data
        self.float_sliders = self._init_float_sliders()
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

        # Set the initial view
        self.target_dropdown = pnw.Select(
            name="Viewed Parameter",
            options=list(self.model.named_model_components_dict),
            value=self.model.observables[0].model_varname,
        )

        # Define reactive components for widgets that update state
        self._plot_options = pn.rx([])  # Plot types possible for target
        self._previous_plot_type = pn.rx()  # Last-selected plot type
        self._independent_dim_options = pn.rx([])  # Independent dimension options
        self._previous_independent_dim = pn.rx()  # Last-selected independent dimension

        # Set the initial values for the reactive components
        self.set_plot_options()
        self.set_independent_dim_options()

        # Create widgets whose values update dynamically.
        self.plot_type_dropdown = pnw.Select(
            name="Plot Type", options=self._plot_options, value=self._previous_plot_type
        )
        self.independent_dim_dropdown = pnw.Select(
            name="Independent Dimension",
            options=self._independent_dim_options,
            value=self._previous_independent_dim,
        )

        # Define callbacks needed for updating reactive components
        self.plot_type_dropdown.rx.watch(self.set_plot_options)
        self.independent_dim_dropdown.rx.watch(self.set_independent_dim_options)

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

    def set_plot_options(self) -> None:
        """Sets the plot options based on the selected target."""
        self._plot_options.rx.value[:] = ["KDE", "ECDF", "Violin", "Relational"]
        self._previous_plot_type.rx.value = "KDE"

    def set_independent_dim_options(self) -> None:
        """Sets the independent dimension options based on the selected target."""
        self._independent_dim_options.rx.value[:] = ["a"]
        self._previous_independent_dim.rx.value = "a"

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
                pn.pane.Markdown(
                    "**Togglable Parameters**", styles={"font-size": "1.5em"}
                ),
                *self.float_sliders.values(),
            ),
            self.target_dropdown,
            self.plot_type_dropdown,
            self.independent_dim_dropdown,
            self.draw_seed_entry,
            self.draw_entry,
            self.update_button,
        )

        # Build the plot
        # plot = plotting_func(plot_df, paramname=plot_df.columns[0])

        # Organize the display
        return widgets
