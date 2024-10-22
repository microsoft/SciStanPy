"""Holds objects used for plotting."""

from functools import partial, wraps
from typing import Callable, Optional, overload, ParamSpec, TypeVar, Union

import holoviews as hv
import hvplot.interactive
import hvplot.pandas
import numpy as np
import numpy.typing as npt
import pandas as pd
import panel.widgets as pnw
import torch

# Types
P = ParamSpec("P")
T = TypeVar("T")
HVType = Union[hv.element.raster.RGB, hv.element.raster.Overlay]


def aggregate_data(
    data: npt.NDArray, independent_dim: Optional[int] = None
) -> npt.NDArray:
    """
    Aggregates data from a numpy array. Here are the rules:

    1.  If the independent dimension is not provided, the data array is flattened.
        In this case, the `independent_labels` parameter is ignored.
    2.  If the independent dimension is provided and the independent labels
        are not provided, then the data array is flattened along all dimensions
        except for the independent dimension. That is, a 2D array is returned
        with shape (-1, n_independent), where "-1" indicates the product of
        all other dimensions.
    """
    # Flatten the data if the independent dimension is not provided.
    if independent_dim is None:
        return data.flatten()

    # If the independent dimension is provided, first move that dimension to
    # the end (reshape is C-major), then reshape the data to flatten all other dimensions
    else:
        independent_dim = (
            independent_dim + 1 if independent_dim >= 0 else independent_dim
        )
        n_independent = data.shape[independent_dim]
        return np.moveaxis(data, independent_dim, -1).reshape((-1, n_independent))


def allow_interactive(plotting_func: Callable[P, T]) -> Callable[P, T]:
    """
    A decorator that allows the plotting function to be interactive. This assumes
    that the first argument is a dataframe or a hvplot interactive object.
    """

    @wraps(plotting_func)
    def interactive_plotting_func(*args, **kwargs):

        # Run the plotting function
        plot = plotting_func(*args, **kwargs)

        # If the first argument is a dataframe, then we return the plot
        if isinstance(args[0], pd.DataFrame):
            return plot

        # If a list, then we combine the plots
        if isinstance(plot, list):
            interactive = plot[0]
            for p in plot[1:]:
                interactive = interactive + p
            return interactive.cols(1)

        # Otherwise, autorange the y-axis and return
        return plot.opts(autorange="y")

    return interactive_plotting_func


@overload
def plot_ecdf_kde(plotting_df: pd.DataFrame, /, paramname: str) -> list[HVType]: ...


@overload
def plot_ecdf_kde(
    plotting_df: hvplot.interactive.Interactive, /, paramname: pnw.Select
) -> hvplot.interactive.Interactive: ...


@allow_interactive
def plot_ecdf_kde(plotting_df, /, paramname):
    """Renders the plots for 1D data."""

    # Build the plots, combine, and return
    ecdf_plot = plotting_df.hvplot.line(
        x=paramname, y="Cumulative Probability", title="ECDF", width=600, height=400
    )
    kde_plot = plotting_df.hvplot.kde(
        y=paramname, title="KDE", width=600, height=400, cut=0, autorange="y"
    )

    return [kde_plot, ecdf_plot]


@overload
def plot_ecdf_violin(plotting_df: pd.DataFrame, /, paramname: str) -> list[HVType]: ...


@overload
def plot_ecdf_violin(
    plotting_df: hvplot.interactive.Interactive, /, paramname: pnw.Select
) -> hvplot.interactive.Interactive: ...


@allow_interactive
def plot_ecdf_violin(plotting_df, /, paramname):
    """
    Renders the plots for 2D data treating the second dimension elements as independent.
    """
    ecdfplot = plotting_df.hvplot.line(
        x=paramname,
        y="Cumulative Probability",
        by="Independent Label",
        color=hv.Palette("Inferno"),
        title="ECDF",
        width=600,
        height=400,
    )

    violinplot = plotting_df.hvplot.violin(
        y=paramname,
        by="Independent Label",
        title="Violin Plot",
        c="Independent Label",
        cmap="inferno",
        width=600,
        height=400,
        invert=True,
        colorbar=True,
    )

    return [ecdfplot + violinplot]


# TODO: Add ability to slice out specific dimensions


@overload
def plot_relationship(
    plotting_df: pd.DataFrame, /, paramname: str, datashade: bool
) -> HVType: ...


@overload
def plot_relationship(
    plotting_df: hvplot.interactive.Interactive,
    /,
    paramname: pnw.Select,
    datashade: bool,
) -> hvplot.interactive.Interactive: ...


@allow_interactive
def plot_relationship(plotting_df, /, paramname, datashade=True):
    """
    Renders the plots for 2D data treating the second dimension elements as dependent
    on a provided label.
    """
    # Different kwargs for datashade
    if datashade:
        extra_kwargs = {
            "datashade": True,
            "dynamic": False,
            "aggregator": "count",
            "cmap": "inferno",
        }
    else:
        extra_kwargs = {"datashade": False, "dynamic": True, "line_color": "lime"}

    return plotting_df.hvplot.line(
        x="Independent Label",
        y=paramname,
        title="Relationship",
        width=600,
        height=400,
        **extra_kwargs,
    )


def choose_plotting_function(
    independent_dim: Optional[int],
    independent_labels: Optional[npt.NDArray],
    datashade: bool = True,
) -> Callable:
    """
    Chooses the plotting function based on the independent dimension and labels.
    If we just have the initial view, then we will plot an ECDF and KDE. If we have
    an independent dimension but no labels, then we will plot a series of ECDFs
    and violin plots. If we have an independent dimension and labels, then we will
    plot lines describing those relationships.
    """
    if independent_dim is None:
        return plot_ecdf_kde
    elif independent_labels is None:
        return plot_ecdf_violin
    else:
        return partial(plot_relationship, datashade=datashade)


def build_plotting_df(
    samples: npt.NDArray,
    paramname: str = "param",
    independent_dim: Optional[int] = None,
    independent_labels: Optional[npt.NDArray] = None,
) -> pd.DataFrame:
    """
    Builds the dataframes that will be used for plotting the prior predictive
    check. The dataframes are built from the samples drawn from the model.
    """
    # Aggregate the data
    data = aggregate_data(data=samples, independent_dim=independent_dim)

    # If the independent dimension is provided, one path
    if independent_dim is not None:

        # The data must be a 2D array
        assert data.ndim == 2

        # Build the independent labels if they are not provided. If they are
        # provided, make sure they are the right length.
        if no_labels := independent_labels is None:
            independent_labels = np.arange(data.shape[1])
        assert len(independent_labels) == data.shape[1]

        # Add the data to a dataframe, separating each trace with a row of NaNs
        sub_dfs = [None] * len(data)
        for i, data_row in enumerate(data):
            # Combine arrays and add a row of NaNs
            combined = np.vstack([data_row, independent_labels]).T
            combined = np.vstack([combined, np.full(2, np.nan)])

            # Build the dataframe
            temp_df = pd.DataFrame(combined, columns=[paramname, "Independent Label"])
            temp_df["Trace"] = i
            sub_dfs[i] = temp_df

        # Combine all dataframes
        plotting_df = pd.concat(sub_dfs, ignore_index=True)

        # If no labels were provided, drop the NaN rows and add an ECDF column
        if no_labels:

            # Drop the NaN rows
            plotting_df = plotting_df.dropna()

            # Add an ECDF column
            plotting_df["Cumulative Probability"] = plotting_df.groupby(
                by="Independent Label"
            )[paramname].rank(method="max", pct=True)

            plotting_df = plotting_df.sort_values(
                by=["Independent Label", "Cumulative Probability"]
            )

        return plotting_df

    # If the independent dimension is not provided, we need to add an ECDF
    # column
    else:
        # Get the dataframe
        plotting_df = pd.DataFrame({paramname: data})

        # Add an ECDF to the dataframe
        plotting_df["Cumulative Probability"] = plotting_df[paramname].rank(
            method="max", pct=True
        )

        return plotting_df.sort_values(by="Cumulative Probability")


def plot_distribution(
    samples: Union[npt.NDArray, torch.Tensor],
    overlay: Optional[npt.NDArray] = None,
    paramname: str = "param",
    independent_dim: Optional[int] = None,
    independent_labels: Optional[npt.NDArray] = None,
) -> Union[HVType, list[HVType]]:
    """Plots a distribution of samples with an optional ground truth overlay."""

    # Samples must be a numpy array
    samples = (
        samples.detach().cpu().numpy()
        if not isinstance(samples, np.ndarray)
        else samples
    )

    # Build the plotting dataframe for the distribution
    plotting_df = build_plotting_df(
        samples=samples,
        paramname=paramname,
        independent_dim=independent_dim,
        independent_labels=independent_labels,
    )

    # Get the plotting function
    plotting_func = choose_plotting_function(
        independent_dim=independent_dim, independent_labels=independent_labels
    )

    # Get the figure
    fig = plotting_func(plotting_df, paramname=paramname)

    # If no overlay, just return the plot
    if overlay is None:
        return fig

    # We are working with a 2D overlay if the independent dimension is provided
    expected_ndim = 2 if independent_dim is not None else 1
    if overlay.shape[-expected_ndim:] != samples.shape[-expected_ndim:]:
        raise ValueError(
            f"The last {expected_ndim} dimensions of the overlay must be the same "
            f"shape as the last {expected_ndim} dimensions of the samples"
        )

    # Build the plotting dataframe for the overlay
    overlay_df = build_plotting_df(
        samples=overlay,
        paramname=paramname,
        independent_dim=independent_dim,
        independent_labels=independent_labels,
    )

    # Plot the overlay. No data shading for the overlay.
    overfig = choose_plotting_function(
        independent_dim=independent_dim,
        independent_labels=independent_labels,
        datashade=False,
    )(overlay_df, paramname=paramname)

    # If a list, then we combine the plots by index
    if isinstance(fig, list):
        assert isinstance(overfig, list)
        assert len(fig) == len(overfig)
        for i, p in enumerate(fig):
            fig[i] = p * overfig[i]

        return fig

    # If not a list, then we combine the plots as is
    return fig * overfig
