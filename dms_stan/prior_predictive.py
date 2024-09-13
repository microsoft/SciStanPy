"""This module is used for building and displaying prior predictive checks."""

from copy import deepcopy
from typing import Optional

import hvplot.interactive
import hvplot.pandas
import numpy as np
import numpy.typing as npt
import pandas as pd
import panel.widgets as pnw

import dms_stan as dms


class PriorPredictiveCheck:
    """Base class for prior predictive checks."""

    def __init__(self, model: dms.model.Model, copy_model: bool = False):

        # Copy the model if requested. If we don't copy, then we can modify our
        # values on the model directly.
        self.model = deepcopy(model) if copy_model else model

    def build_plotting_df(
        self,
        paramname: str,
        n_experiments: int,
        independent_dim: Optional[int] = None,
        independent_labels: Optional[npt.NDArray] = None,
    ) -> pd.DataFrame:
        """
        Builds the dataframes that will be used for plotting the prior predictive
        check. The dataframes are built from the samples drawn from the model.
        """
        # Get the samples from the model and build the dataframe
        return dms.plotting.build_plotting_df(
            samples=self.model.draw_from(paramname, n_experiments),
            paramname=paramname,
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
        # Get the list of parameters and observables in the model
        legal_targets = list(self.model)
        error_message = "The model has no parameters or observables"

        # If there is an independent dimension provided, filter down to just those
        # parameters that have that dimension
        if independent_dim is not None:
            legal_targets = list(
                filter(lambda x: x[1].ndim > independent_dim, legal_targets)
            )
            error_message += f" with an independent dimension of {independent_dim}"

        # If there are independent labels provided, filter down to just those parameters
        # that match the length of the independent labels
        if independent_labels is not None:
            legal_targets = list(
                filter(
                    lambda x: x[1].shape[independent_dim] == len(independent_labels),
                    legal_targets,
                )
            )
            error_message += f" with a size of {len(independent_labels)}"

        # Raise an error if there are no legal parameters
        if not legal_targets:
            raise ValueError(error_message)

        # Legal targets down to just the names
        legal_targets = [paramname for paramname, _ in legal_targets]

        # If the initial dim is provided, make sure that it is a legal parameter
        # Otherwise, just take the first legal parameter
        if initial_view is not None:
            if initial_view not in legal_targets:
                raise ValueError(
                    error_message.replace(
                        "observables", f"observables named {initial_view}"
                    )
                )
        else:
            initial_view = legal_targets[-1]

        return initial_view, legal_targets

    def _init_float_sliders(self) -> dict[str, pnw.EditableFloatSlider]:
        """Gets the float sliders for the togglable parameters in the model."""
        # Each togglable parameter gets its own float slider
        sliders = {}
        for paramname, paramdict in self.model.togglable_param_values.items():
            for constant_name, constant_value in paramdict.items():
                combined_name = f"{paramname}.{constant_name}"
                sliders[combined_name] = pnw.EditableFloatSlider(
                    name=combined_name, value=constant_value.item()
                )

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
        paramname: str,
        n_experiments: int,
        independent_dim: Optional[int],
        independent_labels: Optional[npt.NDArray],
        **kwargs: float,
    ):
        """
        The key of each kwarg gives the name of the parameter to update, and the
        value is a dictionary that links the constant names within that parameter
        to the new values for those constants.
        """

        # Define helper functions. This is just for scoping and readability.
        def process_kwargs() -> dict[str, dict[str, npt.NDArray]]:
            """
            Kwargs passed in to the parent function are formatted as `paramname.constantname`
            mapped to floats. This function processes those kwargs into a dictionary
            of dictionaries, where the outer dictionary maps parameter names to
            dictionaries that map constant names to new values. The new values are
            also converted to numpy arrays.
            """
            processed_kwargs = {}
            for key, val in kwargs.items():
                paramname, constantname = key.split(".")
                if paramname not in processed_kwargs:
                    processed_kwargs[paramname] = {}
                processed_kwargs[paramname][constantname] = np.array(val)
            return processed_kwargs

        def update_model(processed_kwargs: dict[str, dict[str, npt.NDArray]]):
            """
            Changes the values of the constants in the model according to the processed
            kwargs.
            """
            for paramname, constant_dict in processed_kwargs.items():
                assert set(constant_dict) == set(self.model[paramname].parameters)
                self.model[paramname].parameters.update(constant_dict)

        # Update the model with the new parameters
        update_model(process_kwargs())

        # Return the dataframes for plotting
        return self.build_plotting_df(
            paramname=paramname,
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
            paramname=target_dropdown,
            n_experiments=draw_slider,
            independent_dim=independent_dim,
            independent_labels=independent_labels,
            **float_sliders,
        ).interactive()

        # Make the plot
        return plotting_func(plotting_df=plot_df, paramname=target_dropdown)
