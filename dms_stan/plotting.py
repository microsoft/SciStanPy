"""Holds objects used for plotting."""

from functools import partial, wraps
from typing import Any, Callable, Optional, overload, Literal, ParamSpec, TypeVar, Union

import holoviews as hv
import hvplot.interactive
import hvplot.pandas
import numpy as np
import numpy.typing as npt
import pandas as pd
import panel.widgets as pnw
import torch

from scipy import stats

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
        return np.moveaxis(data, independent_dim, -1).reshape(
            (-1, data.shape[independent_dim])
        )


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

        # Otherwise, set the framewise option and return
        return plot.opts(framewise=True)

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
        samples.detach().cpu().numpy() if isinstance(samples, torch.Tensor) else samples
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


def calculate_relative_quantiles(
    reference: npt.NDArray, observed: npt.NDArray
) -> npt.NDArray:
    """
    Given a set of reference observations and a set of observed values, this function
    calculates the quantiles of the observed values relative to the reference.
    More specifically, for each value in "observed," we find the number of values
    in an equivalent position in the reference that are less than or equal to the
    observed value. This is then returned as a numpy array.
    Args:
        reference (npt.NDArray): The reference observations. This must have shape
            (n_samples, feat1, feat2, ..., featn). The first dimensions is the number
            of samples and the rest are the feature dimensions.
        observed (npt.NDArray): The observed values. This must have the same number
            of dimensions as the reference. Each dimension must be the same size
            as the corresponding dimension of the reference except for the first
            dimension, which gives the number of observations.

    Returns:
        npt.NDArray: The quantiles of the observed values relative to the reference.
            This has shape (n_observations, feat1, feat2, ..., featn).
    """
    # Check shapes
    if reference.ndim < 2:
        raise ValueError("Reference must be at least 2D.")
    if observed.ndim < 2:
        raise ValueError("Observed must be at least 2D.")
    if reference.shape[1:] != observed.shape[1:]:
        raise ValueError(
            "The shape of the reference and observed must match except for the "
            "first dimension."
        )

    # Now we calculate the quantiles that the observations fall into relative to
    # the reference. The produced array has shape (n_observations, n_features)
    return (reference[None] <= observed[:, None]).mean(axis=1)


def _set_defaults(
    kwargs: dict[str, Any] | None, default_values: tuple[tuple[str, Any], ...]
) -> dict[str, Any]:
    """
    Provides a set of default values for kwargs that are different from the function
    defaults.
    """

    # Convert none to empty dict if needed
    kwargs = kwargs or {}
    for k, v in default_values:
        # If the key is not in the kwargs, add it
        if k not in kwargs:
            kwargs[k] = v

    # Update the kwargs with default values
    return kwargs


def plot_calibration(
    reference: npt.NDArray,
    observed: npt.NDArray,
    **kwargs,
) -> tuple[hv.Overlay, npt.NDArray[np.floating]]:
    """
    Given a set of reference observations and a set of observed values, this function
    plots an ECDF of the quantiles of the observed values relative to the reference.
    More specifically, for each value in "observed," we find the number of values
    in an equivalent position in the reference that are less than or equal to the
    observed value. This is then plotted as a cumulative distribution function.

    Args:
        reference (npt.NDArray): The reference observations. See `calculate_relative_quantiles`
            for details.
        observed (npt.NDArray): The observed values. See `calculate_relative_quantiles`
            for details.
        return_deviance (bool): If True, the function will return the deviance
            statistics for each observation, where the deviance is defined as the
            absolute difference between the observed ECDF and the idealized ECDF,
            which is a straight line from (0, 0) to (1, 1). Default is False.
        **kwargs: Additional keyword arguments to pass to the plotting function.
            These will be passed to the `hvplot.Curve` function.

    Returns:
        hv.Overlay: The ECDF plot.
        npt.NDArray (Optional): The deviance statistics for each observation.
    """

    # pylint: disable=line-too-long
    def calculate_deviance(
        x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]
    ) -> float:
        r"""
        Calculates the absolute difference in area between the observed ECDF and the
        ideal ECDF. We can calculate this by subtracting the area under the curve
        of the ideal ECDF from the area under the curve of the observed ECDF, calculated
        using the trapezoidal rule:

        \begin{align}
        AUC_{obs} = \sum_{i=1}^{n} (x_{i+1} - x_{i}) * (y_{i+1} + y_{i}) / 2
        AUC_{ideal} = \sum_{i=1}^{n} (x_{i+1} - x_{i}) * (x_{i+1} + x_{i}) / 2
        AUC_{diff} = \sum_{i=1}^{n} (x_{i+1} - x_{i}) * abs((y_{i+1} + y_{i}) - (x_{i+1} + x_{i})) / 2
        \end{align}

        where $x$ are the quantiles and $y$ are the cumulative probabilities and we
        take the absolute value of the difference between the two AUCs at each step
        to get the absolute difference.
        """
        # Get the widths of the intervals
        dx = np.diff(x)

        # Get the total heights of the trapezoids over intervals for the observed
        # and ideal ECDFs
        h_obs = y[1:] + y[:-1]
        h_ideal = x[1:] + x[:-1]

        # Calculate the absolute difference in areas under the curves
        return np.sum(dx * np.abs(h_obs - h_ideal) / 2).item()

    # Now we calculate the quantiles that the observations fall into relative to
    # the reference. The produced array has shape (n_observations, n_features)
    quantiles = calculate_relative_quantiles(reference, observed)

    # Add to plots
    deviances = np.empty(quantiles.shape[0])
    plots = [None] * (quantiles.shape[0] + 1)
    for obs_ind, obs_quantiles in enumerate(quantiles):

        # Get the ECDF coordinates of the observed quantiles
        ecdf = stats.ecdf(obs_quantiles)
        x, y = ecdf.cdf.quantiles, ecdf.cdf.probabilities

        # Calculate the absolute deviance
        deviances[obs_ind] = calculate_deviance(x, y)

        # Build the plot
        plots[obs_ind] = hv.Curve(
            (x, y), kdims=["Quantiles"], vdims=["Cumulative Probability"]
        ).opts(**kwargs)

    # One final plot giving the idealized ECDF
    plots[-1] = hv.Curve(
        ((0, 1), (0, 1)),
        kdims=["Quantiles"],
        vdims=["Cumulative Probability"],
    ).opts(line_color="black", line_dash="dashed", show_legend=False)

    return hv.Overlay(plots), deviances


@overload
def quantile_plot(
    x: npt.NDArray,
    reference: npt.NDArray,
    quantiles: npt.ArrayLike,
    *,
    observed: npt.ArrayLike | None,
    labels: dict[str, npt.ArrayLike] | None,
    include_median: bool,
    overwrite_input: bool,
    return_quantiles: Literal[False],
    observed_type: Literal["line", "scatter"],
    area_kwargs: dict[str, Any] | None,
    median_kwargs: dict[str, Any] | None,
    observed_kwargs: dict[str, Any] | None,
    allow_nan: bool,
) -> hv.Overlay: ...


@overload
def quantile_plot(
    x: npt.NDArray,
    reference: npt.NDArray,
    quantiles: npt.ArrayLike,
    *,
    observed: npt.ArrayLike | None,
    labels: dict[str, npt.ArrayLike] | None,
    include_median: bool,
    overwrite_input: bool,
    return_quantiles: Literal[True],
    observed_type: Literal["line", "scatter"],
    area_kwargs: dict[str, Any] | None,
    median_kwargs: dict[str, Any] | None,
    observed_kwargs: dict[str, Any] | None,
    allow_nan: bool,
) -> tuple[hv.Overlay, npt.NDArray[np.floating]]: ...


def quantile_plot(
    x,
    reference,
    quantiles,
    *,
    observed=None,
    labels=None,
    include_median=True,
    overwrite_input=False,
    return_quantiles=False,
    observed_type="line",
    area_kwargs=None,
    median_kwargs=None,
    observed_kwargs=None,
    allow_nan=False,
):
    """
    Given a 2D array of data, calculates the quantiles over the first axis and plots
    the results as an area chart. Optionally, the median is also plotted as a line.

    Args:
        x: The x-axis labels.
        reference: The data to plot. Must be 2D.
        quantiles: The quantiles to calculate. Note that these will be symmetrized
            and the median will be added if it is not already in the list. For example,
            if [0.025, 0.25] is passed in, the following will be calculated and
            returned: [0.025, 0.25, 0.5, 0.75, 0.975].
        observed: An optional observed to plot on top of the quantile plot. The last
            dimension of the observed must match the last dimension of the plot data.
        include_median: Whether to include the median line.
        overwrite_input: Passed to `np.quantile`. If True, memory will be saved
            by overwriting components of the `reference` array during quantile
            calculation. The array contents will be undefined after such an operation.
            Default is False.
        return_quantiles: Whether to return the calculated quantiles. If True, a tuple
            is returned with the quantile plot as the first element and the quantiles
            as the second element.
        sort_by_range: Whether to sort the quantiles by range. If True, the quantiles
            will be sorted by the difference between the upper and lower bounds.
            In this case, the x-axis labels will be ignored and replaced with a
            range of integers.
        observed_type: The type of observed to plot. Either "line" or "scatter".
        area_kwargs: Additional plotting options for the area plots.
        ovrlay_kwargs: Additional plotting options for the observed.

    Returns:
        hv.Overlay: The quantile plot.
        npt.NDArray (Optional): The calculated quantiles.
    """

    # Set the default kwargs
    area_kwargs = _set_defaults(
        area_kwargs,
        (
            ("color", "black"),
            ("alpha", 0.2),
            ("line_width", 1),
            ("line_color", "black"),
            ("fill_alpha", 0.2),
            ("show_legend", False),
        ),
    )
    median_kwargs = _set_defaults(
        median_kwargs,
        (
            ("color", "black"),
            ("line_width", 1),
            ("line_color", "black"),
            ("show_legend", False),
        ),
    )
    observed_kwargs = _set_defaults(
        observed_kwargs,
        (
            ("color", "gold"),
            ("line_width", 1 if observed_type == "line" else 0),
            ("alpha", 0.5),
            ("show_legend", False),
        ),
    )
    labels = _set_defaults(labels, ())

    # The plot data must be 2D.
    if reference.ndim != 2:
        raise ValueError("The plot data must be 2D.")

    # If provided, the observed must be 1D or 2D and the last dimension must match
    # that of the plot data
    if observed is not None:

        # Check the observed shape
        if observed.ndim == 1:
            observed = observed[None]
        elif observed.ndim != 2:
            raise ValueError("The observed must be 1D or 2D.")
        if observed.shape[-1] != reference.shape[-1]:
            raise ValueError(
                "The last dimension of the observed must match the last "
                "dimension of the plot data."
            )

        # Get the type of observed plot
        observed_plot = hv.Scatter if observed_type == "scatter" else hv.Curve

    # Get the quantiles
    max_digits = max(len(str(q).split(".")[1]) for q in quantiles)
    quantiles = sorted(
        set(quantiles) | {np.round(1 - q, max_digits) for q in quantiles} | {0.5}
    )

    # Check that the quantiles are between 0 and 1
    if not all(0 < q < 1 for q in quantiles):
        raise ValueError(
            "Quantiles must be between 0 and 1. Please provide a valid list of "
            "quantiles."
        )

    # Check that the quantiles are odd in number and include 0.5
    assert len(quantiles) % 2 == 1, "Quantiles must be odd in number"
    median_ind = len(quantiles) // 2
    assert quantiles[median_ind] == 0.5, "Quantiles must include 0.5"

    # Calculate the quantiles
    area_bounds = (np.nanquantile if allow_nan else np.quantile)(
        reference, quantiles, axis=0, overwrite_input=overwrite_input
    )

    # Only include hover tools if we have labels
    def add_hover_tools(kwargset):
        kwargset.update(
            {
                "hover_mode": kwargset.get("hover_mode", "vline"),
                "tools": list(set(kwargset.get("tools", []) + ["hover"])),
            }
        )

    if labels:
        add_hover_tools(median_kwargs)
        add_hover_tools(observed_kwargs)

    # Build the plots
    plots = [
        hv.Area(
            (x, area_bounds[i], area_bounds[-i - 1]),
            vdims=["lower", "upper"],
        ).opts(**area_kwargs)
        for i in range(len(quantiles) // 2)
    ]
    if include_median:
        plots.append(
            hv.Curve(
                (x, area_bounds[median_ind], *labels.values()),
                kdims=["x"],
                vdims=["y", *labels.keys()],
            ).opts(**median_kwargs)
        )
    if observed is not None:
        plots.extend(
            observed_plot(
                (x, observed_data, *labels.values()),
                kdims=["x"],
                vdims=["y", *labels.keys()],
            ).opts(**observed_kwargs)
            for observed_data in observed
        )

    # Return the quantiles if requested
    plots = hv.Overlay(plots)
    if return_quantiles:
        return plots, area_bounds

    return plots


def hexgrid_with_mean(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    *,
    mean_windowsize: int | None = None,
    hex_kwargs: dict[str, Any] | None = None,
    mean_kwargs: dict[str, Any] | None = None,
) -> hv.Overlay:
    """Creates a hexgrid plot out of the provided x and y data. Additionally, plots
    a rolling mean of the data across the x-axis.

    Args:
        x (npt.NDArray[np.floating]): x-axis data.
        y (npt.NDArray[np.floating]): y-axis data.
        mean_windowsize (int | None, optional): The windowsize to use when calculating
            the rolling average. Defaults to None, in which case it is set to `x.size // 100`.
        hex_kwargs (dict[str, Any] | None, optional): Additional keyword arguments
            to pass to the options of `hv.HexTiles`. Defaults to None.
        mean_kwargs (dict[str, Any] | None, optional): Additional keyword arguments
            to pass to the options of `hv.Curve`. Defaults to None.

    Returns:
        hv.Overlay: The hexgrid plot with the rolling mean.
    """
    # x and y must be 1D arrays
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    # Set the default kwargs
    hex_kwargs = _set_defaults(
        hex_kwargs,
        (("cmap", "viridis"), ("colorbar", True)),
    )
    mean_kwargs = _set_defaults(
        mean_kwargs,
        (("color", "slategray"), ("line_width", 1)),
    )
    windowsize = mean_windowsize or max(1, x.size // 100)

    # Build the plot
    return hv.HexTiles((x, y)).opts(**hex_kwargs) * hv.Curve(
        pd.DataFrame({"x": x, "y": y})
        .sort_values("x")
        .rolling(window=windowsize)
        .mean()
        .dropna(),
        "x",
        "y",
        label="Rolling Mean",
    ).opts(**mean_kwargs)
