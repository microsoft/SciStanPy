"""Core plotting functions for SciStanPy visualization and analysis.

This module implements the primary plotting functionality for SciStanPy,
providing specialized visualization tools for Bayesian analysis, model
diagnostics, and statistical relationships.

The module leverages HoloViews and hvplot for flexible, interactive
visualizations that can be easily customized and extended. All plotting
functions support both standard NumPy arrays and interactive widgets
for dynamic exploration of model results.

Key Features:
    - ECDF and KDE plots for distribution visualization
    - Quantile plots with confidence intervals
    - Model calibration diagnostics
    - Hexagonal binning for large datasets
    - Interactive plotting with widget support
    - Customizable styling and overlays

Functions are organized by visualization type and complexity, from simple
distribution plots to sophisticated multi-panel diagnostic displays.
"""

# pylint: disable=too-many-lines

from __future__ import annotations

from functools import partial, wraps
from typing import (
    Any,
    Callable,
    Optional,
    overload,
    Literal,
    ParamSpec,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import holoviews as hv
import hvplot.interactive
import hvplot.pandas
import numpy as np
import numpy.typing as npt
import pandas as pd
import panel.widgets as pnw
import torch

from scipy import stats

if TYPE_CHECKING:
    from scistanpy import custom_types

# Types
P = ParamSpec("P")
T = TypeVar("T")
HVType = Union[hv.element.raster.RGB, hv.element.raster.Overlay]


def aggregate_data(
    data: npt.NDArray, independent_dim: Optional[int] = None
) -> npt.NDArray:
    """Aggregate multi-dimensional data for plotting purposes.

    This function reshapes multi-dimensional arrays according to specified
    aggregation rules, preparing data for visualization functions that
    expect specific array structures.

    :param data: Input data array to aggregate
    :type data: npt.NDArray
    :param independent_dim: Dimension to preserve during aggregation.
                           If None, flattens entire array. (Default: None)
    :type independent_dim: Optional[int]

    :returns: Aggregated data array
    :rtype: npt.NDArray

    Aggregation Rules:
        - If independent_dim is None: Returns flattened 1D array
        - If independent_dim is specified: Returns 2D array with shape
          (-1, n_independent) where -1 represents the product of all
          other dimensions

    Example:
        >>> data = np.random.randn(10, 5, 3)
        >>> # Flatten completely
        >>> flat = aggregate_data(data)  # Shape: (150,)
        >>> # Preserve last dimension
        >>> agg = aggregate_data(data, independent_dim=2)  # Shape: (50, 3)
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
    """Decorator to enable interactive plotting capabilities.

    This decorator modifies plotting functions to handle both static
    DataFrames and interactive hvplot objects, automatically configuring
    the appropriate display options for each case.

    :param plotting_func: The plotting function to make interactive
    :type plotting_func: Callable[P, T]

    :returns: Enhanced function with interactive capabilities
    :rtype: Callable[P, T]

    The decorator handles:
        - Static DataFrames: Returns plot directly
        - Interactive objects: Configures framewise options
        - Plot lists: Combines multiple plots into column layout

    Example:
        >>> @allow_interactive
        ... def my_plot(df, param):
        ...     return df.hvplot.line(y=param)
        >>> # Works with both static and interactive data
        >>> plot = my_plot(dataframe, 'column_name')
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
    """Create empirical CDF and kernel density estimate plots.

    This function generates complementary ECDF and KDE visualizations
    for univariate data, providing both cumulative and density perspectives
    on the data distribution.

    :param plotting_df: DataFrame containing the data to plot
    :type plotting_df: Union[pd.DataFrame, hvplot.interactive.Interactive]
    :param paramname: Name of the parameter/column to visualize
    :type paramname: Union[str, pnw.Select]

    :returns: List containing KDE and ECDF plots, or interactive plot
    :rtype: Union[list[HVType], hvplot.interactive.Interactive]

    The function creates:
        - KDE plot: Smooth density estimate with automatic bandwidth
        - ECDF plot: Step function showing cumulative probabilities

    Both plots are configured with consistent styling and appropriate
    axis labels for scientific presentation.

    Example:
        >>> df = pd.DataFrame({'param': np.random.normal(0, 1, 1000)})
        >>> plots = plot_ecdf_kde(df, 'param')
        >>> # plots[0] is KDE, plots[1] is ECDF
    """
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
    """Create ECDF and violin plots for multi-group data comparison.

    This function visualizes distributions across multiple groups or
    categories, combining empirical CDFs with violin plots to show
    both cumulative and density information simultaneously.

    :param plotting_df: DataFrame with grouped data including 'Independent Label' column
    :type plotting_df: Union[pd.DataFrame, hvplot.interactive.Interactive]
    :param paramname: Name of the parameter/column to visualize
    :type paramname: Union[str, pnw.Select]

    :returns: Combined ECDF and violin plot overlay
    :rtype: Union[list[HVType], hvplot.interactive.Interactive]

    The visualization includes:
        - Multi-line ECDF plot: One curve per group with color coding
        - Violin plot: Density distributions by group with colorbar

    Groups are automatically colored using the Inferno colormap.

    Example:
        >>> # DataFrame with 'param' values and 'Independent Label' grouping
        >>> plots = plot_ecdf_violin(grouped_df, 'param')
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
    """Visualize relationships between parameters and independent variables.

    This function creates line plots showing how parameters vary with
    respect to independent variables, with optional datashading for
    large datasets to improve performance and readability.

    :param plotting_df: DataFrame with 'Independent Label' and parameter columns
    :type plotting_df: Union[pd.DataFrame, hvplot.interactive.Interactive]
    :param paramname: Name of the dependent parameter to plot
    :type paramname: Union[str, pnw.Select]
    :param datashade: Whether to use datashading for large datasets (Default: True)
    :type datashade: bool

    :returns: Line plot showing parameter relationships
    :rtype: Union[HVType, hvplot.interactive.Interactive]

    Datashading options:
        - True: Uses count aggregation with Inferno colormap (large data)
        - False: Uses dynamic line plotting with lime color (small data)

    The function automatically optimizes visualization based on data size
    and user preferences for performance versus detail.

    Example:
        >>> # Plot parameter evolution over time/conditions
        >>> plot = plot_relationship(time_series_df, 'param', datashade=True)
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
    independent_dim: Optional["custom_types.Integer"],
    independent_labels: Optional[npt.NDArray],
    datashade: bool = True,
) -> Callable:
    """Select appropriate plotting function based on data characteristics.

    This function implements intelligent plot type selection based on
    the structure of the data and available metadata.

    :param independent_dim: Dimension index for independent variable, if any
    :type independent_dim: Optional[custom_types.Integer]
    :param independent_labels: Labels for independent variable values
    :type independent_labels: Optional[npt.NDArray]
    :param datashade: Whether to enable datashading for large datasets (Default: True)
    :type datashade: bool

    :returns: Appropriate plotting function for the data structure
    :rtype: Callable

    Selection Logic:
        - No independent_dim: Returns plot_ecdf_kde (univariate analysis)
        - Independent_dim but no labels: Returns plot_ecdf_violin (multi-group)
        - Both independent_dim and labels: Returns plot_relationship (dependency)

    Example:
        >>> plotter = choose_plotting_function(None, None)  # ECDF/KDE
        >>> plotter = choose_plotting_function(1, time_labels)  # Relationship
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
    independent_dim: Optional["custom_types.Integer"] = None,
    independent_labels: Optional[npt.NDArray] = None,
) -> pd.DataFrame:
    """Construct DataFrame optimized for plotting functions.

    This function transforms raw sample arrays into structured DataFrames
    with appropriate columns and formatting for visualization functions.
    It handles various data structures and automatically generates
    necessary metadata for plotting.

    :param samples: Raw sample data to structure for plotting
    :type samples: npt.NDArray
    :param paramname: Column name to assign for the parameter values (Default: "param")
    :type paramname: str
    :param independent_dim: Dimension representing independent variable (Default: None)
    :type independent_dim: Optional[custom_types.Integer]
    :param independent_labels: Labels for independent variable values (Default: None)
    :type independent_labels: Optional[npt.NDArray]

    :returns: Structured DataFrame ready for plotting functions
    :rtype: pd.DataFrame

    The function handles:
        - Data aggregation according to independent dimension
        - Automatic label generation when not provided
        - ECDF calculation for cumulative plots
        - Trace separation with NaN boundaries for line plots
        - Proper sorting for visualization functions

    Example:
        >>> samples = np.random.randn(100, 50, 10)  # 100 traces, 50 time points, 10 params
        >>> df = build_plotting_df(samples, 'measurement', independent_dim=1)
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
    independent_dim: Optional["custom_types.Integer"] = None,
    independent_labels: Optional[npt.NDArray] = None,
) -> Union[HVType, list[HVType]]:
    """Create comprehensive distribution plots with optional overlays.

    This is the main distribution plotting function that automatically
    selects appropriate visualization types based on data structure
    and provides optional ground truth or reference overlays.

    :param samples: Sample data from model simulations or posterior draws
    :type samples: Union[npt.NDArray, torch.Tensor]
    :param overlay: Optional reference data to overlay on the plot (Default: None)
    :type overlay: Optional[npt.NDArray]
    :param paramname: Name to assign for the parameter being plotted (Default: "param")
    :type paramname: str
    :param independent_dim: Dimension index for independent variable (Default: None)
    :type independent_dim: Optional[custom_types.Integer]
    :param independent_labels: Labels for independent variable values (Default: None)
    :type independent_labels: Optional[npt.NDArray]

    :returns: Plot or list of plots showing data distribution
    :rtype: Union[HVType, list[HVType]]

    :raises ValueError: If overlay dimensions don't match sample dimensions

    The function automatically:
        - Converts PyTorch tensors to NumPy arrays
        - Selects appropriate plot type based on data structure
        - Overlays reference data with distinct styling
        - Handles both single plots and multi-panel layouts

    Example:
        >>> # Simple distribution plot
        >>> plot = plot_distribution(posterior_samples, paramname='mu')
        >>> # With ground truth overlay
        >>> plot = plot_distribution(samples, overlay=true_values, paramname='sigma')
    """
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
    """Calculate quantiles of observed values relative to reference distribution.

    For each observed value, this function computes the quantile (percentile)
    it would occupy within the corresponding reference distribution. This is
    essential for calibration analysis and model validation.

    :param reference: Reference observations with shape (n_samples, feat1, ..., featN).
                     First dimension is samples, remaining are feature dimensions.
    :type reference: npt.NDArray
    :param observed: Observed values with shape (n_obs, feat1, ..., featN).
                    Feature dimensions must match reference.
    :type observed: npt.NDArray

    :returns: Quantiles of observed values relative to reference
    :rtype: npt.NDArray

    :raises ValueError: If arrays have incompatible dimensions

    The calculation determines, for each observed value, what fraction of
    reference values in the corresponding position are less than or equal
    to the observed value. This produces values between 0 and 1.

    Mathematical Definition:
        For observed value x_ij and reference distribution R_j:
        quantile_ij = P(R_j <= x_ij) = (1/n) * sum(R_kj <= x_ij for k=1..n)

    Example:
        >>> ref = np.random.normal(0, 1, (1000, 10))  # 1000 samples, 10 features
        >>> obs = np.random.normal(0.5, 1, (5, 10))   # 5 observations, 10 features
        >>> quantiles = calculate_relative_quantiles(ref, obs)
        >>> # quantiles.shape == (5, 10), values between 0 and 1
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
    # the reference. The produced array has shape (n_observations, ...)
    return (reference[None] <= observed[:, None]).mean(axis=1)


def _set_defaults(
    kwargs: dict[str, Any] | None, default_values: tuple[tuple[str, Any], ...]
) -> dict[str, Any]:
    """Apply default values to kwargs dictionary without overwriting existing keys.

    This utility function provides a clean way to set default plotting
    parameters while respecting user-provided customizations.

    :param kwargs: User-provided keyword arguments (may be None)
    :type kwargs: Union[dict[str, Any], None]
    :param default_values: Tuple of (key, value) pairs for defaults
    :type default_values: tuple[tuple[str, Any], ...]

    :returns: Dictionary with defaults applied for missing keys
    :rtype: dict[str, Any]

    Example:
        >>> defaults = (('color', 'blue'), ('alpha', 0.5))
        >>> user_kwargs = {'color': 'red'}
        >>> final = _set_defaults(user_kwargs, defaults)
        >>> # final == {'color': 'red', 'alpha': 0.5}
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
    """Generate calibration plots for model validation.

    This function creates empirical cumulative distribution plots of
    relative quantiles to assess model calibration. Well-calibrated
    models should produce observed values that are uniformly distributed
    across quantiles of the reference distribution.

    :param reference: Reference observations for calibration assessment
    :type reference: npt.NDArray
    :param observed: Observed values to assess against reference
    :type observed: npt.NDArray
    :param kwargs: Additional styling options passed to hvplot.Curve

    :returns: Tuple of (calibration plot overlay, deviance statistics)
    :rtype: tuple[hv.Overlay, npt.NDArray[np.floating]]

    The calibration plot shows:
        - ECDF curves for each observation set
        - Ideal calibration line (diagonal from (0,0) to (1,1))
        - Deviation areas highlighting calibration errors

    Deviance Calculation:
        For each observation, computes the absolute difference in area
        between the observed ECDF and the ideal uniform ECDF using
        the trapezoidal rule for numerical integration.

    Interpretation:
        - Points near diagonal: Well-calibrated
        - Points above diagonal: Model underconfident OR bad fit
        - Points below diagonal: Model overconfident OR bad fit

    Example:
        >>> ref_data = posterior_predictive_samples  # Shape: (1000, 100)
        >>> obs_data = actual_observations          # Shape: (10, 100)
        >>> plot, deviances = plot_calibration(ref_data, obs_data)
        >>> print(f"Mean deviance: {deviances.mean():.3f}")
    """

    # pylint: disable=line-too-long
    def calculate_deviance(
        x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]
    ) -> "custom_types.Float":
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
    """Create quantile plots with confidence intervals and optional overlays.

    This function generates area plots showing quantile ranges of reference
    data along with optional median lines and observed data overlays.
    It's particularly useful for visualizing uncertainty bands around
    model predictions.

    :param x: X-axis values (independent variable)
    :type x: npt.NDArray
    :param reference: Reference data with shape (n_samples, n_points)
    :type reference: npt.NDArray
    :param quantiles: Quantile values to calculate and plot (0 < q < 1)
    :type quantiles: npt.ArrayLike
    :param observed: Optional observed data to overlay. Must be 1D or 2D with last
        dimension matching that of the reference data (Default: None).
    :type observed: Optional[npt.ArrayLike]
    :param labels: Optional labels for hover tooltips (Default: None).
    :type labels: Optional[dict[str, npt.ArrayLike]]
    :param include_median: Whether to include median line (Default: True)
    :type include_median: bool
    :param overwrite_input: Whether to overwrite reference array during calculations.
        This can help save memory by avoiding the creation of intermediate copies.
        (Default: False)
    :type overwrite_input: bool
    :param return_quantiles: Whether to return calculated quantiles along with plot.
        (Default: False)
    :type return_quantiles: bool
    :param observed_type: Type of overlay plot ('line' or 'scatter') (Default: 'line')
    :type observed_type: Literal["line", "scatter"]
    :param area_kwargs: Styling options for quantile areas. See `hv.opts.Area`.
    :type area_kwargs: Optional[dict[str, Any]]
    :param median_kwargs: Styling options for median line. See `hv.opts.Line`.
    :type median_kwargs: Optional[dict[str, Any]]
    :param observed_kwargs: Styling options for observed overlay. See `hv.opts.Curve`
        or `hv.opts.Scatter` depending on choice of `observed_type`.
    :type observed_kwargs: Optional[dict[str, Any]]
    :param allow_nan: If True, uses `np.nanquantile` for quantile calculation. Otherwise,
        uses `np.quantile` (Default: False).
    :type allow_nan: bool

    :returns: Quantile plot overlay, optionally with calculated quantiles
    :rtype: Union[hv.Overlay, tuple[hv.Overlay, npt.NDArray[np.floating]]]

    :raises ValueError: If quantiles are not between 0 and 1, or if array dimensions
        are invalid

    Features:
        - Automatic quantile symmetrization (adds complement quantiles)
        - Nested confidence intervals with graduated transparency
        - Customizable styling for all plot components
        - Optional hover labels for interactive exploration

    Example:
        >>> x = np.linspace(0, 10, 100)
        >>> ref = np.random.normal(np.sin(x), 0.1, (1000, 100))
        >>> obs = np.sin(x) + 0.05 * np.random.randn(100)
        >>> plot = quantile_plot(x, ref, [0.025, 0.25], observed=obs)
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
    mean_windowsize: "custom_types.Integer" | None = None,
    hex_kwargs: dict[str, Any] | None = None,
    mean_kwargs: dict[str, Any] | None = None,
) -> hv.Overlay:
    """Create hexagonal binning plot with rolling mean overlay.

    This function generates a hexagonal heatmap showing data density
    combined with a rolling mean trend line, useful for visualizing
    large datasets with underlying trends.

    :param x: X-axis data values
    :type x: npt.NDArray[np.floating]
    :param y: Y-axis data values
    :type y: npt.NDArray[np.floating]
    :param mean_windowsize: Window size for rolling mean calculation.
                           Defaults to x.size // 100 if not specified.
    :type mean_windowsize: Optional[custom_types.Integer]
    :param hex_kwargs: Styling options for hexagonal tiles. See `hv.opts.HexTiles`.
    :type hex_kwargs: Optional[dict[str, Any]]
    :param mean_kwargs: Styling options for rolling mean line. See `hv.opts.Line`.
    :type mean_kwargs: Optional[dict[str, Any]]

    :returns: Overlay combining hexagonal heatmap and rolling mean
    :rtype: hv.Overlay

    :raises ValueError: If x and y arrays have different shapes or are not 1D

    The hexagonal binning:
        - Aggregates points into hexagonal cells
        - Colors cells by point density using viridis colormap
        - Includes colorbar for density interpretation

    The rolling mean:
        - Computed over sorted x values to show trend
        - Window size automatically scaled to data size
        - Styled for clear visibility over density plot

    Example:
        >>> # Large dataset with trend
        >>> x = np.random.randn(10000)
        >>> y = 2*x + 0.5*np.random.randn(10000)
        >>> plot = hexgrid_with_mean(x, y, mean_windowsize=200)
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
