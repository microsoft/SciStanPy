# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Prior predictive check functionality for SciStanPy models.

This module provides interactive widgets and visualizations for conducting
prior predictive checks on SciStanPy models. Prior predictive checks allow
users to examine the behavior of their models before fitting to data by
sampling from prior distributions and visualizing the resulting predictions.

The module centers around the PriorPredictiveCheck class, which creates
an interactive dashboard with sliders for model hyperparameters and
dropdown menus for selecting visualization options. This enables rapid
exploration of how different prior specifications affect model behavior.

Key Features:
    - Interactive hyperparameter adjustment with sliders
    - Multiple visualization types (ECDF, KDE, violin, relationship plots)
    - Automatic widget configuration based on model structure
    - Real-time plot updates with efficient data processing
    - Support for multi-dimensional parameter exploration

The interface automatically adapts to the structure of the provided model,
exposing only relevant parameters and visualization options based on the
data dimensions and available coordinates.
"""

from __future__ import annotations

import itertools
import re

from copy import deepcopy
from typing import Optional, TYPE_CHECKING

import holoviews as hv
import hvplot.pandas  # pylint: disable=unused-import
import numpy as np
import pandas as pd
import panel as pn
import panel.widgets as pnw
import xarray as xr

from param.parameterized import Event

from scistanpy.model.components import constants

if TYPE_CHECKING:
    from scistanpy import model as ssp_model

# We need a regular expression for separating the variable name from its indices
_INDEX_EXTRACTOR = re.compile(r"([A-Za-z0-9_\.]+)\[?([0-9, ]*)\]?")


class PriorPredictiveCheck:
    """Interactive dashboard for conducting prior predictive checks.

    This class creates a comprehensive interface for exploring model behavior
    through prior predictive sampling. It automatically generates appropriate
    widgets based on the model structure and provides multiple visualization
    modes for examining parameter distributions and relationships.

    :param model: SciStanPy model to analyze
    :type model: ssp_model.Model
    :param copy_model: Whether to create a copy of the model to avoid modifying the original.
                      Defaults to False, meaning updates to model parameters in the
                      interactive plot will be reflected in the original model object.
    :type copy_model: bool

    :ivar model: The model being analyzed (copy or reference)
    :ivar float_sliders: Dictionary of parameter adjustment sliders
    :ivar target_dropdown: Widget for selecting which parameter to visualize
    :ivar group_dim_dropdown: Widget for selecting grouping dimension
    :ivar independent_var_dropdown: Widget for selecting independent variable
    :ivar plot_type_dropdown: Widget for selecting visualization type
    :ivar draw_seed_entry: Widget for setting random seed
    :ivar draw_entry: Widget for setting number of experiments
    :ivar update_model_button: Button to update model and redraw data
    :ivar update_plot_button: Button to update plot without redrawing data
    :ivar fig: HoloViews pane containing the current plot

    The dashboard provides real-time updates as users adjust parameters.

    Example:
        >>> # Add parameters and observables to model
        >>> model = ssp.ModelSubclass(*args, **kwargs)
        >>> # Display dashboard in notebook or web application
        >>> check = PriorPredictiveCheck(model, copy_model=True)
        >>> dashboard = check.display()
    """

    def __init__(self, model: "ssp_model.Model", copy_model: bool = False):

        # Copy the model if requested. If we don't copy, then we can modify our
        # values on the model directly.
        self.model = deepcopy(model) if copy_model else model

        # Initialize widgets.
        self.float_sliders = self._init_float_sliders()
        self.target_dropdown = pnw.Select(
            name="Viewed Parameter",
            options=[
                k
                for k, v in self.model.named_model_components_dict.items()
                if not isinstance(v, constants.Constant)
            ],
            value=self.model.observables[0].model_varname,
        )
        self.group_dim_dropdown = pnw.Select(name="Group By", options=[], value="")
        self.independent_var_dropdown = pnw.Select(
            name="Independent Variable", options=[], value=""
        )
        self.plot_type_dropdown = pnw.Select(name="Plot Type", options=[], value="ECDF")
        self.draw_seed_entry = pnw.IntInput(name="Seed", value=1025)
        self.draw_entry = pnw.IntInput(name="Number of Experiments", value=1)
        self.update_model_button = pnw.Button(
            name="Update Model", button_type="primary"
        )
        self.update_plot_button = pnw.Button(name="Update Plot", button_type="primary")

        # We need additional components for the plotting data
        self.fig = pn.pane.HoloViews(
            hv.Curve([]),
            name="Plot",
            align="center",
            sizing_mode="stretch_both",
        )

        # The update model button will run the full pipeline, including drawing new
        # data and updating the plot. The update plot button will only update the
        # plot, without redrawing the data.
        self.update_model_button.on_click(self._full_pipeline)
        self.update_plot_button.on_click(self._update_plot)

        # We need to store the data and the plotting data
        self._xarray_data: xr.Dataset = xr.Dataset()
        self._processed_data: pd.DataFrame = pd.DataFrame()

        # Draw initial data. We need this for setting up the remaining reactive
        # components. This will set the `_xarray_data` attribute.
        self._draw_data()

        # Create reactive components whose values depend on the data. These components
        # have no effect on the data that is drawn, only what is shown.
        self.target_dropdown.param.watch(self.set_group_dim_options, "value")
        self.group_dim_dropdown.param.watch(self.set_independent_var_options, "value")
        self.group_dim_dropdown.param.watch(self.set_plot_type_options, "value")
        self.independent_var_dropdown.param.watch(self.set_plot_type_options, "value")

        # Set initial values for the components relevant to showing data
        self.target_dropdown.param.trigger("value")
        self.group_dim_dropdown.param.trigger("value")
        self.independent_var_dropdown.param.trigger("value")

        # Run the full pipeline to set up the plotting data
        self._full_pipeline()

    def _init_float_sliders(self) -> dict[str, pnw.EditableFloatSlider]:
        """Initialize float sliders for toggleable model parameters.

        This method scans the model for constant parameters marked as toggleable
        and creates appropriate slider widgets for interactive adjustment. It
        handles both scalar and multi-dimensional parameters appropriately.

        :returns: Dictionary mapping parameter names to slider widgets
        :rtype: dict[str, pnw.EditableFloatSlider]

        For scalar parameters or those with enforced uniformity, a single slider
        is created. For multi-dimensional parameters, individual sliders are
        created for each array element with indexed naming conventions.

        Slider ranges and step sizes are determined by the parameter's configured
        slider properties (slider_start, slider_end, slider_step_size).
        """
        # Each togglable parameter gets its own float slider
        sliders = {}

        # Process all constants in the model
        for (
            hyperparam_name,
            hyperparam_val,
        ) in self.model.all_model_components_dict.items():

            # Skip non-constants and non-togglable parameters
            if (
                not isinstance(hyperparam_val, constants.Constant)
                or not hyperparam_val.is_togglable
            ):
                continue

            # If no dimensions OR if we are forcing uniformity across a multidimensional
            # array, just create a slider
            if hyperparam_val.ndim == 0 or hyperparam_val.enforce_uniformity:
                sliders[hyperparam_name] = pnw.EditableFloatSlider(
                    name=hyperparam_name,
                    value=np.unique(hyperparam_val.value).item(),
                    start=hyperparam_val.slider_start,
                    end=hyperparam_val.slider_end,
                    step=hyperparam_val.slider_step_size,
                )
                continue

            # Otherwise, create a slider for each entry
            for arr_ind in np.ndindex(hyperparam_val.shape):

                # Build the slider name
                name = f"{hyperparam_name}[{', '.join(map(str, arr_ind))}]"

                # Get the slider value
                slider_val = hyperparam_val.value[arr_ind]

                # Add the slider
                sliders[name] = pnw.EditableFloatSlider(
                    name=name,
                    value=slider_val,
                    start=hyperparam_val.slider_start,
                    end=hyperparam_val.slider_end,
                    step=hyperparam_val.slider_step_size,
                )

        return sliders

    def _update_model(self) -> None:
        """Update the underlying model with current slider values.

        This method reads the current values from all parameter sliders and
        updates the corresponding constant values in the model. It handles
        both scalar parameters and indexed multi-dimensional parameters.

        For indexed parameters, the method parses the slider name to extract
        the parameter name and array indices, then updates the appropriate
        array element.
        """
        for paramname, slider in self.float_sliders.items():

            # Get the parameter name and the indices
            paramname, indices = _INDEX_EXTRACTOR.match(paramname).groups()
            if indices:
                tuple(map(int, indices.split(",")))
            else:
                indices = ...

            # Update the value of the constant
            self.model[paramname].value[indices] = slider.value

    def _draw_data(self) -> None:
        """Draw new data from the model using current parameter values.

        This method generates samples from the model's prior predictive
        distribution using the current slider values and stores the results
        as an xarray Dataset for efficient manipulation and visualization.

        The number of samples and random seed are controlled by the
        corresponding widget values (draw_entry.value and draw_seed_entry.value).
        """
        self._xarray_data = self.model.draw(
            n=self.draw_entry.value,
            named_only=True,
            as_xarray=True,
            seed=self.draw_seed_entry.value,
        )

    def _process_data(self) -> None:
        """Process raw xarray data into DataFrame format appropriate for plotting.

        This method transforms the xarray Dataset into a pandas DataFrame
        with appropriate structure for the selected visualization type. It
        handles data reshaping, grouping, and computation of derived quantities
        like cumulative probabilities for ECDF plots.

        The processing logic adapts based on the selected plot type:
        - ECDF plots: Computes cumulative probabilities and sorts appropriately
        - Relationship plots: Sorts by independent variable and adds separators
        - Other plots: Basic reshaping and filtering

        The method respects the current widget selections for target parameter,
        grouping dimension, and independent variable.
        """

        # We need to define aggregation functions for the different plot types
        def build_ecdfs(group):

            # We need to record the original values of the dependent variable
            new_df = group[[self.target_dropdown.value]]

            # Rank the target variable
            new_df["Cumulative Probability"] = group[[self.target_dropdown.value]].rank(
                method="max", pct=True
            )

            # Add the independent variable values
            sort_keys = ["Cumulative Probability"]
            if self._independent_label is not None:
                sort_keys.insert(0, self._independent_label)
                new_df[self._independent_label] = group[self._independent_label]

            # Sort and return
            return new_df.sort_values(by=sort_keys)

        def build_relations(group):
            # Sort the data by the independent variable and add a NaN row to separate
            # the data by the appropriate dependent variable
            return pd.concat(
                [
                    group.sort_values(by=self.independent_var_dropdown.value),
                    pd.DataFrame({self.independent_var_dropdown.value: [np.nan]}),
                ]
            )

        # Gather the target data
        selected_data = self._xarray_data[
            [self.target_dropdown.value]
            + (
                []
                if self.independent_var_dropdown.value == ""
                else [self.independent_var_dropdown.value]
            )
        ]

        # Reshape the data as appropriate and convert the extracted data to a DataFrame.
        # We keep the grouping dimension separate from the stacked results.
        df = (
            selected_data.stack(
                stacked=[
                    dim
                    for dim in selected_data.dims
                    if dim != self.group_dim_dropdown.value
                ],
                create_index=False,
            )
            .to_dataframe()
            .reset_index()
        )

        # Filter to just the columns needed. These are the grouping dimensions and
        # independent variables, if any.
        target_cols = [self.target_dropdown.value]
        if self.group_dim_dropdown.value != "":
            target_cols.append(self.group_dim_dropdown.value)
        if self.independent_var_dropdown.value != "":
            target_cols.extend([self.independent_var_dropdown.value, "stacked"])
        df = df[target_cols]

        # Final processing for certain plots.
        if self.plot_type_dropdown.value == "ECDF":
            if self.group_dim_dropdown.value != "":
                df = df.groupby(self.group_dim_dropdown.value, group_keys=False).apply(
                    build_ecdfs
                )
            else:
                df = build_ecdfs(df)
        elif self.plot_type_dropdown.value == "Relationship":
            df = df.groupby("stacked").apply(build_relations)

        # Store the processed data
        self._processed_data = df

    def _update_plot(  # pylint: disable=unused-argument
        self, event: Optional[Event] = None
    ) -> None:
        """Update the plot display using current data and widget settings.

        This method reprocesses the current xarray data according to the
        selected visualization options and updates the plot display. It does
        not redraw data from the model, making it efficient for exploring
        different visualization modes.

        :param event: Panel event object (unused, for callback compatibility).
                     Defaults to None.
        :type event: Optional[Event]

        The method automatically selects appropriate plotting functions and
        styling based on the plot type dropdown selection, handling both
        hvplot-based plots and specialized HoloViews elements like violin plots.
        """
        # Update plot button to loading
        self.update_plot_button.loading = True

        # Reformat the data
        self._process_data()

        # Update the plot kwargs
        plot_kwargs = {
            "ECDF": self.get_ecdf_kwargs,
            "KDE": self.get_kde_kwargs,
            "Violin": self.get_violin_kwargs,
            "Relationship": self.get_relationship_kwargs,
        }[self.plot_type_dropdown.value]()

        # Update the plot
        if self.plot_type_dropdown.value == "Violin":
            self.fig.object = hv.Violin(plot_kwargs.pop("args"), **plot_kwargs).opts(
                show_legend=False
            )
        else:
            self.fig.object = self._processed_data.hvplot(**plot_kwargs)

        # Update plot button to not be loading
        self.update_plot_button.loading = False

    def _full_pipeline(  # pylint: disable=unused-argument
        self, event: Optional[Event] = None
    ) -> None:
        """Execute complete model update and visualization pipeline.

        This method performs the full sequence of operations: updating model
        parameters from sliders, drawing new data from the updated model,
        and refreshing the plot display. It provides a complete refresh of
        the analysis when model parameters change.

        :param event: Panel event object (unused, for callback compatibility).
                     Defaults to None.
        :type event: Optional[Event]

        The method sets loading states on relevant buttons to provide user
        feedback during potentially time-consuming operations like data
        generation for complex models.
        """
        # Buttons to loading mode
        self.update_model_button.loading = True
        self.update_plot_button.loading = True

        # Update the model
        self._update_model()

        # Draw new data
        self._draw_data()

        # Update the plot
        self._update_plot()

        # Buttons to not be loading
        self.update_model_button.loading = False
        self.update_plot_button.loading = False

    def set_group_dim_options(self, event: Event) -> None:
        """Update grouping dimension options based on selected target parameter.

        This method configures the group dimension dropdown with appropriate
        options based on the dimensionality of the currently selected target
        parameter. It ensures that grouping options are only available for
        multi-dimensional parameters.

        :param event: Panel event containing the new target parameter selection
        :type event: Event

        The method updates both the dropdown options and the descriptive text
        showing dimension sizes to help users understand the data structure.
        If the previously selected grouping dimension is no longer valid,
        it automatically resets to a sensible default.
        """
        # Get the dimensions of the target variable
        partial_opts = list(self._xarray_data[event.new].dims[1:])

        # Grouping can only be performed when we have more than one dimension
        target_dim_opts = [""]
        if len(partial_opts) > 1:
            target_dim_opts += partial_opts

        # If the previous dependent dimension is not in the options, reset it
        if self.group_dim_dropdown.value not in target_dim_opts:
            self.group_dim_dropdown.value = target_dim_opts[-1]

        # Update the description of the grouping dimension
        description = ", ".join(
            f"[{dim}: {self._xarray_data.sizes[dim]}]" for dim in target_dim_opts[1:]
        )
        self.group_dim_dropdown.name = f"Group By ({description})"

        # Update the dropdown options
        self.group_dim_dropdown.options = target_dim_opts

    def set_independent_var_options(self, event: Event) -> None:
        """Update independent variable options based on grouping dimension.

        This method configures the independent variable dropdown with coordinates
        and data variables that are compatible with the selected grouping
        dimension. Only variables that vary along the grouping dimension are
        included as options.

        :param event: Panel event containing the new grouping dimension selection
        :type event: Event

        The method scans both coordinates and data variables in the xarray
        Dataset to find suitable independent variables, ensuring compatibility
        with the selected visualization approach.
        """
        # The independent variable must be a coordinate or data variable that contains
        # the `Group By` dimension.
        independent_var_opts = [""] + [
            varname
            for varname, arr in itertools.chain(
                self._xarray_data.coords.items(), self._xarray_data.data_vars.items()
            )
            if arr.sizes.get(event.new, 0) > 1
        ]

        # If the previous independent variable is not in the options, reset it
        if self.independent_var_dropdown.value not in independent_var_opts:
            self.independent_var_dropdown.value = ""

        # Update the dropdown options
        self.independent_var_dropdown.options = independent_var_opts

    def set_plot_type_options(  # pylint: disable=unused-argument
        self, event: Event
    ) -> None:
        """Update available plot types based on current dimension selections.

        This method determines which visualization types are appropriate
        given the current selections for target parameter, grouping dimension,
        and independent variable. It enables more sophisticated plot types
        as more structure is specified.

        :param event: Panel event (used for callback compatibility is not used)
        :type event: Event

        Plot Type Logic:
        - ECDF and KDE: Always available for any parameter
        - Violin: Available when grouping dimension is selected
        - Relationship: Available when both grouping and independent variable are selected

        The method automatically selects the most sophisticated available plot
        type as the default when options change.
        """
        # We can always have an ECDF and KDE plot
        plot_type_opts = ["ECDF", "KDE"]
        default_plot = "ECDF"

        # If a grouping dimension is set, then we can also have violin plots
        if self.group_dim_dropdown.value != "":
            plot_type_opts.append("Violin")
            default_plot = "Violin"

        # If an independent variable is set, then we can also have relationship
        # plots
        if self.independent_var_dropdown.value != "":
            plot_type_opts.append("Relationship")
            default_plot = "Relationship"

        # If the previous plot type is not in the options, reset it
        if self.plot_type_dropdown.value not in plot_type_opts:
            self.plot_type_dropdown.value = default_plot

        # Update to the plot type options
        self.plot_type_dropdown.options = plot_type_opts

    def display(self) -> pn.Row:
        """Create and return the complete interactive dashboard layout.

        This method assembles all widgets and the plot display into a
        comprehensive dashboard layout suitable for display in Jupyter
        notebooks, Panel applications, or web interfaces.

        :returns: Panel layout containing all interface elements
        :rtype: pn.Row

        The layout consists of:
        - Left panel: Model hyperparameter sliders and viewing options
        - Right panel: Interactive plot display that updates based on selections

        All widgets are organized into logical groups with appropriate headers
        for intuitive navigation and use.

        Example:
            >>> check = PriorPredictiveCheck(model)
            >>> dashboard = check.display()
            >>> dashboard.servable()  # For web deployment
        """
        # Organize widgets and plot
        return pn.Row(
            pn.WidgetBox(
                pn.WidgetBox(
                    "# Model Hyperparameters",
                    *self.float_sliders.values(),
                    self.draw_seed_entry,
                    self.draw_entry,
                    self.update_model_button,
                ),
                pn.WidgetBox(
                    "# Viewing Options",
                    self.target_dropdown,
                    self.group_dim_dropdown,
                    self.independent_var_dropdown,
                    self.plot_type_dropdown,
                    self.update_plot_button,
                ),
            ),
            self.fig,
        )

    def get_kde_kwargs(self) -> dict:
        """Generate keyword arguments for kernel density estimate plots.

        This method constructs the parameter dictionary needed for creating
        KDE visualizations using hvplot, including appropriate grouping
        and styling options.

        :returns: Dictionary of hvplot parameters for KDE plotting
        :rtype: dict

        The returned dictionary includes:
        - kind: Set to 'kde' for kernel density estimation
        - x: `None` for KDE plots.
        - y: Target parameter name for the y-axis
        - by: Independent label for grouping (if applicable)
        - datashade: Disabled (False) for KDE plots to maintain clarity
        """
        return {
            "kind": "kde",
            "x": None,
            "y": self.target_dropdown.value,
            "by": self._independent_label,
            "datashade": False,
        }

    def get_ecdf_kwargs(self) -> dict:
        """Generate keyword arguments for empirical CDF plots.

        This method constructs the parameter dictionary needed for creating
        ECDF visualizations using hvplot, including cumulative probability
        calculations and appropriate hover interactions.

        :returns: Dictionary of hvplot parameters for ECDF plotting
        :rtype: dict

        The returned dictionary includes:
        - kind: Set to 'line' for step-like ECDF appearance
        - x: Target parameter values
        - y: 'Cumulative Probability' (computed during data processing)
        - by: Independent label for grouping multiple ECDFs
        - datashade: Disabled (False) for ECDF plots to maintain clarity
        - hover: Set to 'hline' for horizontal hover lines
        """
        return {
            "kind": "line",
            "x": self.target_dropdown.value,
            "y": "Cumulative Probability",
            "by": self._independent_label,
            "datashade": False,
            "hover": "hline",
        }

    def get_violin_kwargs(self) -> dict:
        """Generate arguments for violin plot creation.

        This method constructs the arguments needed for creating violin plots
        using HoloViews, handling complex grouping scenarios and determining
        appropriate categorization based on data structure.

        :returns: Dictionary containing plot arguments and dimensions
        :rtype: dict

        The method handles multiple grouping scenarios:
        - Single grouping by dimension index
        - Grouping by both dimension and independent variable
        - Automatic determination of primary vs. secondary grouping

        The returned dictionary includes:
        - args: Tuple of (groups..., values) for HoloViews Violin constructor
        - kdims: List of key dimension names
        - vdims: Value dimension name (target parameter)
        """
        # This is only an option if we have a grouping dimension
        group_indices = self._processed_data[self.group_dim_dropdown.value].to_numpy()

        # If the independent label is provided, then it is the grouping variable
        # and the group index is the category variable IF there are more unique
        # combinations of indices and independent labels than there are group indices
        if self.independent_var_dropdown.value != "":
            independent_labels = self._processed_data[
                self.independent_var_dropdown.value
            ].to_numpy()
            if len(np.unique(group_indices)) < len(
                self._processed_data[
                    [self.group_dim_dropdown.value, self.independent_var_dropdown.value]
                ].drop_duplicates()
            ):
                groups = [group_indices, independent_labels]
                kdims = [
                    self.independent_var_dropdown.value,
                    self.group_dim_dropdown.value,
                ]
            else:
                groups = [independent_labels]
                kdims = [self.independent_var_dropdown.value]

        # Otherwise, we just have the group indices
        else:
            groups = [group_indices]
            kdims = [self.group_dim_dropdown.value]

        return {
            "args": tuple(
                groups + [self._processed_data[self.target_dropdown.value].to_numpy()]
            ),
            "kdims": kdims,
            "vdims": self.target_dropdown.value,
        }

    def get_relationship_kwargs(self) -> dict:
        """Generate keyword arguments for relationship plots.

        This method constructs the parameter dictionary for creating
        relationship visualizations that show how parameters vary with
        respect to independent variables, using datashading for performance.

        :returns: Dictionary of hvplot parameters for relationship plotting
        :rtype: dict

        The returned dictionary includes:
        - kind: Set to 'line' for continuous relationships
        - x: Independent variable name
        - y: Target parameter name
        - datashade: Enabled (True) for performance with large datasets
        - dynamic: Disabled (False), resulting in all data being embedded in output
        - aggregator: Set to 'count' for density-based coloring
        - cmap: 'inferno' colormap
        """
        return {
            "kind": "line",
            "x": self.independent_var_dropdown.value,
            "y": self.target_dropdown.value,
            "by": None,
            "datashade": True,
            "dynamic": False,
            "aggregator": "count",
            "cmap": "inferno",
        }

    @property
    def _independent_label(self) -> Optional[str]:
        """Determine the effective independent label for plotting.

        This property provides a unified interface for determining which
        variable should be used as the independent label in plots, following
        a priority hierarchy based on user selections.

        :returns: Name of the independent variable, or None if not applicable
        :rtype: Optional[str]

        Priority order:
        1. Independent variable dropdown selection (if not empty)
        2. Group dimension dropdown selection (if not empty)
        3. None (for simple univariate plots)

        This property enables consistent handling of grouping and independent
        variables across different visualization types.
        """
        if self.independent_var_dropdown.value != "":
            return self.independent_var_dropdown.value
        elif self.group_dim_dropdown.value != "":
            return self.group_dim_dropdown.value
