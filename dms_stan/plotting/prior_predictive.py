"""This module is used for building and displaying prior predictive checks."""

import itertools
import re

from copy import deepcopy
from typing import Optional

import holoviews as hv
import hvplot.pandas  # pylint: disable=unused-import
import numpy as np
import pandas as pd
import panel as pn
import panel.widgets as pnw
import xarray as xr

from param.parameterized import Event

from dms_stan import model as dms_model
from dms_stan.model.components import constants

# We need a regular expression for separating the variable name from its indices
_INDEX_EXTRACTOR = re.compile(r"([A-Za-z0-9_\.]+)\[?([0-9, ]*)\]?")


class PriorPredictiveCheck:
    """Base class for prior predictive checks."""

    def __init__(self, model: "dms_model.Model", copy_model: bool = False):

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
        """Updates the underlying model with the new constant values."""
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
        """Draws data from the model and stores it as an xarray Dataset."""
        self._xarray_data = self.model.draw(
            n=self.draw_entry.value,
            named_only=True,
            as_xarray=True,
            seed=self.draw_seed_entry.value,
        )

    def _process_data(self) -> None:
        """Processes the currently drawn data into a DataFrame for plotting."""

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

        # # We are assuming that, at this point, the independent variable and grouping
        # # dimension can be used interchangeably. This is because the independent
        # # variable values at this dimension should be coordinates that are indexed
        # # by the grouping variable. Check this assumption here.
        # if self.independent_var_dropdown.value != "":
        #     assert (
        #         len(df[self.independent_var_dropdown.value].unique())
        #         == len(df[self.group_dim_dropdown.value].unique())
        #         == len(
        #             df[
        #                 [
        #                     self.independent_var_dropdown.value,
        #                     self.group_dim_dropdown.value,
        #                 ]
        #             ].drop_duplicates()
        #         )
        #     )

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
        """Formats the xarray data to a DataFrame and updates the plot."""
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
        """Updates the model, draws new data, and updates the plot."""
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
        """
        Sets the dimension by which the currently viewed paramter will be grouped
        for plotting. This is just all dimensions available to the target variable.
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
        """Sets the independent variable options based on the selected target."""
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
        """
        Sets the options for the plot types depending on the selected dimensions.
        ECDF is always an option, as is KDE. Violin plots are allowed if a dependent
        dimension is selected. Relationship plots are allowed if both an independent
        variable and dimension are selected along with a dependent dimension.
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
        """
        Builds kwargs needed for plotting a kernel density estimate, optionally grouped
        by the independent variable.
        """
        return {
            "kind": "kde",
            "x": None,
            "y": self.target_dropdown.value,
            "by": self._independent_label,
            "datashade": False,
        }

    def get_ecdf_kwargs(self) -> dict:
        """
        Builds kwargs needed for plotting an empirical cumulative distribution function,
        optionally grouped by the independent variable.
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
        """
        Builds kwargs needed for plotting a violin plot, optionally grouped by the
        independent variable.
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
        """
        Builds kwargs needed for plotting a relationship plot, optionally grouped by
        the independent variable.
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
        """
        String representation of the independent variable. This is the independent
        variable name if it is provided; otherwise, it is the group dimension;
        otherwise, it is not defined.
        """
        if self.independent_var_dropdown.value != "":
            return self.independent_var_dropdown.value
        elif self.group_dim_dropdown.value != "":
            return self.group_dim_dropdown.value
