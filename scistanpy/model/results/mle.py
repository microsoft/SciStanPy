"""Maximum likelihood estimation results analysis and visualization.

This module provides analysis tools for maximum likelihood estimation
results from SciStanPy models. It offers diagnostic plots, calibration
checks, and posterior predictive analysis tools designed specifically for MLE-based
inference workflows.

The module centers around the MLEInferenceRes class, which wraps ArviZ InferenceData
objects with specialized methods for MLE result analysis. It provides both individual
diagnostic tools and analysis workflows that combine multiple checks
into unified reporting interfaces.

Key Features:
    - Posterior predictive checking workflows
    - Model calibration analysis with quantitative metrics
    - Interactive visualization with customizable display options
    - Integration with ArviZ for standardized Bayesian workflows
    - Memory-efficient handling of large posterior predictive samples
    - Flexible output formats for different analysis needs

Visualization Capabilities:
    - Posterior predictive sample plotting with confidence intervals
    - Calibration plots with deviation metrics
    - Quantile-quantile plots for model validation
    - Interactive layouts with customizable dimensions

The module is designed to work with SciStanPy's MLE estimation workflow,
providing immediate access to model diagnostics and validation tools
once MLE fitting is complete.
"""

from __future__ import annotations

from typing import (
    Generator,
    Literal,
    Optional,
    overload,
    Sequence,
    TYPE_CHECKING,
    Union,
)

import arviz as az
import holoviews as hv
import hvplot.pandas  # pylint: disable=unused-import
import numpy as np
import numpy.typing as npt
import panel as pn
import xarray as xr

from scipy import stats

from scistanpy import plotting

if TYPE_CHECKING:
    from scistanpy import custom_types


def _log10_shift(*args: npt.NDArray) -> tuple[npt.NDArray, ...]:
    """Apply log10 transformation with automatic shifting for non-positive values.

    This utility function handles logarithmic transformation of arrays that may
    contain non-positive values by finding the global minimum across all arrays
    and shifting them to ensure all values are positive before applying log10.

    :param args: Arrays to transform with log10 after shifting
    :type args: npt.NDArray

    :returns: Tuple of log10-transformed arrays after appropriate shifting
    :rtype: tuple[npt.NDArray, ...]

    The function:
    1. Finds the absolute minimum value across all input arrays
    2. Shifts all arrays by (1 - min_value) to ensure minimum becomes 1
    3. Applies log10 transformation to all shifted arrays

    This ensures logarithmic scaling is possible even when data contains
    zero or negative values, which is common in certain statistical contexts.

    Example:
        >>> arr1 = np.array([-5, 0, 5])
        >>> arr2 = np.array([-2, 3, 8])
        >>> log_arr1, log_arr2 = _log10_shift(arr1, arr2)
        >>> # All values are now log10-transformed with proper scaling
    """
    # Get the minimum value across all arrays
    min_val = min(np.min(arg) for arg in args)

    # Shift the arrays and apply log10
    return tuple(np.log10(arg - min_val + 1) for arg in args)


class MLEInferenceRes:
    """Analysis interface for maximum likelihood estimation results.

    This class provides tools for analyzing and visualizing MLE
    results from SciStanPy models. It wraps ArviZ InferenceData objects with
    specialized methods for posterior predictive checking, calibration analysis,
    and model validation.

    :param inference_obj: ArviZ InferenceData object or path to saved results
    :type inference_obj: Union[az.InferenceData, str]

    :ivar inference_obj: Stored ArviZ InferenceData object with all results

    :raises ValueError: If inference_obj is neither string nor InferenceData
    :raises ValueError: If required groups (posterior, posterior_predictive) are missing

    The class expects the InferenceData object to contain:
    - **posterior**: Samples from fitted parameter distributions
    - **posterior_predictive**: Samples from observable distributions
    - **observed_data**: Original observed data used for fitting

    Key Capabilities:
    - Posterior predictive checking with multiple visualization modes
    - Quantitative model calibration assessment
    - Interactive diagnostic dashboards
    - Summary statistics computation and caching

    Example:
        >>> # Load from MLE result
        >>> mle_analysis = mle_result.get_inference_obj()
        >>> # Run comprehensive diagnostics
        >>> dashboard = mle_analysis.run_ppc()
    """

    def __init__(self, inference_obj: az.InferenceData | str):
        """Base class just initializes the ArviZ object."""
        # If the ArviZ object is a string, we assume it is a path to a netcdf file
        # and load it from there
        if isinstance(inference_obj, str):
            self.inference_obj = az.from_netcdf(inference_obj)

        # If the ArviZ object is an inference data object, we assume it is already
        # built and just assign it to the class
        elif isinstance(inference_obj, az.InferenceData):
            self.inference_obj = inference_obj

        # Otherwise, we raise an error
        else:
            raise ValueError(
                "inference_obj must be either a string or an InferenceData object"
            )

        # The arviz object must have a posterior, a posterior_predictive, and
        # an observed_data group
        if missing_groups := (
            {"posterior", "posterior_predictive"} - set(self.inference_obj.groups())
        ):
            raise ValueError(
                f"ArviZ object is missing the following groups: {', '.join(missing_groups)}"
            )

    def save_netcdf(self, filename: str) -> None:
        """Save the ArviZ InferenceData object to NetCDF format.

        :param filename: Path where to save the NetCDF file
        :type filename: str

        This method provides persistent storage of analysis results.

        Example:
            >>> mle_analysis.save_netcdf('my_mle_results.nc')
            >>> # Later: reload with MLEInferenceRes('my_mle_results.nc')
        """
        self.inference_obj.to_netcdf(filename)

    def _update_group(
        self, attrname: str, new_group: xr.Dataset, force_del: bool = False
    ) -> None:
        """Update or add a group to the ArviZ InferenceData object.

        :param attrname: Name of the group to update or create
        :type attrname: str
        :param new_group: New dataset to add or use for updating
        :type new_group: xr.Dataset
        :param force_del: Whether to force deletion before adding. Defaults to False.
        :type force_del: bool

        This internal method manages the ArviZ object structure, enabling
        addition of computed statistics and derived quantities while
        maintaining data integrity.
        """
        # If the group already exists and we are not forcing a delete, we just update
        # the group.
        if hasattr(self.inference_obj, attrname) and not force_del:
            getattr(self.inference_obj, attrname).update(new_group)
            return

        # Otherwise, if we are forcing a delete, we delete the group before adding
        # the new one
        if force_del:
            delattr(self.inference_obj, attrname)
        self.inference_obj.add_groups({attrname: new_group})

    def calculate_summaries(
        self,
        var_names: list[str] | None = None,
        filter_vars: Literal[None, "like", "regex"] = None,
        kind: Literal["all", "stats", "diagnostics"] = "stats",
        round_to: "custom_types.Integer" = 2,
        circ_var_names: list[str] | None = None,
        stat_focus: str = "mean",
        stat_funcs: Optional[Union[dict[str, callable], callable]] = None,
        extend: bool = True,
        hdi_prob: "custom_types.Float" = 0.94,
        skipna: bool = False,
    ) -> xr.Dataset:
        """Compute summary statistics for MLE results.

        This method wraps ArviZ's summary functionality while adding the computed
        statistics to the InferenceData object for persistence and reuse. See
        `az.summary` for detailed descriptions of arguments.

        :param var_names: Variable names to include in summary. Defaults to None (all variables).
        :type var_names: Optional[list[str]]
        :param filter_vars: Variable filtering method. Defaults to None.
        :type filter_vars: Optional[Literal[None, "like", "regex"]]
        :param kind: Type of statistics to compute. Defaults to "stats".
        :type kind: Literal["all", "stats", "diagnostics"]
        :param round_to: Number of decimal places for rounding. Defaults to 2.
        :type round_to: custom_types.Integer
        :param circ_var_names: Names of circular variables. Defaults to None.
        :type circ_var_names: Optional[list[str]]
        :param stat_focus: Primary statistic for focus. Defaults to "mean".
        :type stat_focus: str
        :param stat_funcs: Custom statistic functions. Defaults to None.
        :type stat_funcs: Optional[Union[dict[str, callable], callable]]
        :param extend: Use functions provided by `stat_funcs`. Defaults to True.
            Only meaningful when `stat_funcs` is provided.
        :type extend: bool
        :param hdi_prob: Probability for highest density interval. Defaults to 0.94.
        :type hdi_prob: custom_types.Float
        :param skipna: Whether to skip NaN values. Defaults to False.
        :type skipna: bool

        :returns: Dataset containing computed summary statistics
        :rtype: xr.Dataset

        :raises ValueError: If diagnostics requested without chain dimension existing
            in `self.inference_obj.posterior.dims`
        :raises ValueError: If diagnostics requested with single chain.

        The computed statistics are automatically added to the InferenceData
        object under the 'variable_summary_stats' group for persistence.

        Example:
            >>> # Compute basic statistics
            >>> stats = mle_analysis.calculate_summaries()
            >>> # Compute diagnostics for multi-chain results
            >>> diag = mle_analysis.calculate_summaries(kind="diagnostics")
        """
        # If there is no chain and draw dimension, we cannot run diagnostics
        if "chain" not in self.inference_obj.posterior.dims:
            raise ValueError(
                "Cannot run diagnostics on a dataset without chain and draw dimensions."
            )

        # If there is only one chain, we cannot run diagnostics
        if kind != "stats" and self.inference_obj.posterior.sizes["chain"] <= 1:
            raise ValueError(
                "Cannot run diagnostics on a dataset run using a single chain"
            )

        # Get the summary statistics
        summaries = az.summary(
            data=self.inference_obj,
            var_names=var_names,
            filter_vars=filter_vars,
            fmt="xarray",
            kind=kind,
            round_to=round_to,
            circ_var_names=circ_var_names,
            stat_focus=stat_focus,
            stat_funcs=stat_funcs,
            extend=extend,
            hdi_prob=hdi_prob,
            skipna=skipna,
        )

        # Build or update the group
        self._update_group("variable_summary_stats", summaries)

        return summaries

    def _iter_pp_obs(
        self,
    ) -> Generator[tuple[str, npt.NDArray, npt.NDArray], None, None]:
        """Iterate over posterior predictive samples and corresponding observations.

        :yields: Tuples of (variable_name, reference_samples, observed_data)
        :rtype: Generator[tuple[str, npt.NDArray, npt.NDArray], None, None]

        This internal method provides a standardized interface for accessing
        posterior predictive samples and observed data, handling dimension
        reshaping and alignment automatically.

        The yielded arrays are formatted as:
        - reference_samples: 2D array (n_samples, n_features)
        - observed_data: 1D array (n_features,)

        This standardization enables consistent processing across all
        diagnostic and visualization methods.
        """
        # Loop over the posterior predictive samples
        for varname, reference in self.inference_obj.posterior_predictive.items():

            # Get the observed data and convert reference and observed to numpy
            # arrays.
            observed = self.inference_obj.observed_data[  # pylint: disable=no-member
                varname
            ].to_numpy()
            reference = np.moveaxis(
                reference.stack(
                    samples=["chain", "draw"], features=[], create_index=False
                ).to_numpy(),
                -1,
                0,
            )

            # Dims must align
            assert observed.shape == reference.shape[1:]

            yield varname, reference.reshape(reference.shape[0], -1), observed.reshape(
                -1
            )

    @overload
    def check_calibration(
        self,
        *,
        return_deviance: Literal[False],
        display: Literal[True],
        width: "custom_types.Integer",
        height: "custom_types.Integer",
    ) -> hv.Layout: ...

    @overload
    def check_calibration(
        self,
        *,
        return_deviance: Literal[False],
        display: Literal[False],
        width: "custom_types.Integer",
        height: "custom_types.Integer",
    ) -> dict[str, hv.Overlay]: ...

    @overload
    def check_calibration(
        self,
        *,
        return_deviance: Literal[True],
        display: Literal[False],
        width: "custom_types.Integer",
        height: "custom_types.Integer",
    ) -> tuple[dict[str, hv.Overlay], dict[str, float]]: ...

    def check_calibration(
        self, *, return_deviance=False, display=True, width=600, height=600
    ):
        """Assess model calibration through posterior predictive quantile analysis.

        This method evaluates how well the model's posterior predictive distribution
        matches the observed data by analyzing the distribution of quantiles. Well-
        calibrated models should produce observed data that are uniformly distributed
        across the quantiles of the posterior predictive distribution.

        :param return_deviance: Whether to return quantitative deviance metrics.
            Defaults to False.
        :type return_deviance: bool
        :param display: Whether to return formatted layout for display. Defaults to True.
        :type display: bool
        :param width: Width of individual plots in pixels. Defaults to 600.
        :type width: custom_types.Integer
        :param height: Height of individual plots in pixels. Defaults to 600.
        :type height: custom_types.Integer

        :returns: Calibration plots and optionally deviance metrics
        :rtype: Union[hv.Layout, dict[str, hv.Overlay], tuple[dict[str, hv.Overlay],
            dict[str, float]]]

        :raises ValueError: If both display and return_deviance are True

        Calibration Assessment Process:
        1. Calculate quantiles of observed data relative to posterior predictive samples
        2. Plot empirical CDF of these quantiles
        3. Compare to ideal diagonal line (perfect calibration)
        4. Compute absolute deviance as area difference between curves

        Interpretation:
        - Diagonal line indicates perfect calibration
        - Curves above diagonal suggest model overconfidence or poor fit
        - Curves below diagonal suggest model underconfidence or poor fit
        - Deviance score of 0 indicates perfect calibration

        Example:
            >>> # Visual assessment
            >>> cal_layout = mle_analysis.check_calibration()
            >>> # Quantitative assessment
            >>> plots, deviances = mle_analysis.check_calibration(
            ...     return_deviance=True, display=False
            ... )
            >>> print(f"Mean deviance: {np.mean(list(deviances.values())):.3f}")
        """
        # We cannot have both `display` and `return_deviance` set to True
        if display and return_deviance:
            raise ValueError(
                "Cannot have both `display` and `return_deviance` set to True."
            )

        # Loop over the posterior predictive samples
        plots: dict[str, hv.Overlay] = {}
        deviances: dict[str, "custom_types.Float"] = {}
        for varname, reference, observed in self._iter_pp_obs():

            # Build calibration plots and record deviance
            plot, dev = plotting.plot_calibration(reference, observed[None])
            dev = dev.item()
            deviances[varname] = dev

            # Finalize the plot with a text annotation and updates to the axes
            plots[varname] = (
                plot
                * hv.Text(
                    0.95,
                    0.0,
                    f"Absolute Deviance: {dev:.2f}",
                    halign="right",
                    valign="bottom",
                )
            ).opts(
                title=f"ECDF of Quantiles: {varname}",
                xlabel="Quantiles",
                ylabel="Cumulative Probability",
                width=width,
                height=height,
            )

        # If requested, display the plots
        if display:
            return hv.Layout(plots.values()).cols(1)

        # If requested, return the plots and the deviance
        if return_deviance:
            return plots, deviances

        # Otherwise, just return the plots
        return plots

    @overload
    def plot_posterior_predictive_samples(
        self,
        *,
        quantiles: Sequence["custom_types.Float"],
        use_ranks: bool,
        logy: bool,
        display: Literal[True],
        width: "custom_types.Integer",
        height: "custom_types.Integer",
    ) -> hv.Layout: ...

    @overload
    def plot_posterior_predictive_samples(
        self,
        *,
        quantiles: Sequence["custom_types.Float"],
        use_ranks: bool,
        logy: bool,
        display: Literal[False],
        width: "custom_types.Integer",
        height: "custom_types.Integer",
    ) -> dict[str, hv.Overlay]: ...

    def plot_posterior_predictive_samples(
        self,
        *,
        quantiles=(0.025, 0.25, 0.5),
        use_ranks=True,
        logy=False,
        display=True,
        width=600,
        height=400,
    ):
        """Visualize observed data against posterior predictive uncertainty intervals.

        This method creates plots showing how observed data relates to the uncertainty
        quantified by posterior predictive samples. The posterior predictive samples
        are displayed as confidence intervals, with observed data overlaid as points.

        :param quantiles: Quantiles defining confidence intervals. Defaults to
            (0.025, 0.25, 0.5). Note: quantiles are automatically symmetrized and
            median is always included.
        :type quantiles: Sequence[custom_types.Float]
        :param use_ranks: Whether to use ranks instead of raw values for x-axis.
            Defaults to True.
        :type use_ranks: bool
        :param logy: Whether to use logarithmic y-axis scaling. Defaults to False.
        :type logy: bool
        :param display: Whether to return formatted layout for display. Defaults to True.
        :type display: bool
        :param width: Width of individual plots in pixels. Defaults to 600.
        :type width: custom_types.Integer
        :param height: Height of individual plots in pixels. Defaults to 400.
        :type height: custom_types.Integer

        :returns: Posterior predictive plots in requested format
        :rtype: Union[hv.Layout, dict[str, hv.Overlay]]

        Visualization Features:
        - Confidence intervals shown as nested colored regions
        - Observed data displayed as scatter points
        - Optional rank transformation for better visualization of skewed data
        - Logarithmic scaling with automatic shifting for non-positive values
        - Interactive hover labels showing data point identifiers

        The rank transformation is particularly useful when observed values have
        highly skewed distributions, as it emphasizes the ordering rather than
        the absolute magnitudes.

        Example:
            >>> # Standard posterior predictive plot
            >>> pp_layout = mle_analysis.plot_posterior_predictive_samples()
            >>> # Custom quantiles with logarithmic scaling
            >>> pp_plots = mle_analysis.plot_posterior_predictive_samples(
            ...     quantiles=(0.05, 0.5, 0.95), logy=True, display=False
            ... )
        """
        # Process each observed variable
        plots: dict[str, hv.Overlay] = {}
        for varname, reference, observed in self._iter_pp_obs():

            # Get the x-axis data
            x = stats.rankdata(observed, method="ordinal") if use_ranks else observed

            # If using a log-y axis, shift the y-data
            if logy:
                reference, observed = _log10_shift(reference, observed)

            # Get labels
            labels = np.array(
                [
                    ".".join(map(str, indices))
                    for indices in np.ndindex(
                        self.inference_obj.observed_data[  # pylint: disable=no-member
                            varname
                        ].shape
                    )
                ]
            )

            # Sort data for plotting the areas and lines
            sorted_inds = np.argsort(x)
            x, reference, observed, labels = (
                x[sorted_inds],
                reference[:, sorted_inds],
                observed[sorted_inds],
                labels[sorted_inds],
            )

            # Build the plot
            plots[varname] = plotting.quantile_plot(
                x=x,
                reference=reference,
                quantiles=quantiles,
                observed=observed,
                labels={varname: labels},
                include_median=False,
                overwrite_input=True,
                observed_type="scatter",
            ).opts(
                xlabel=f"Observed Value {'Rank' if use_ranks else ''}: {varname}",
                ylabel=f"Value{' log10' if logy else ''}: {varname}",
                title=f"Posterior Predictive Samples: {varname}",
                width=width,
                height=height,
            )

        # If requested, display the plots
        if display:
            return hv.Layout(plots.values()).cols(1).opts(shared_axes=False)

        return plots

    @overload
    def plot_observed_quantiles(
        self,
        *,
        use_ranks: bool,
        display: Literal[True],
        width: "custom_types.Integer",
        height: "custom_types.Integer",
        windowsize: Optional["custom_types.Integer"],
    ) -> hv.Layout: ...

    @overload
    def plot_observed_quantiles(
        self,
        *,
        use_ranks: bool,
        display: Literal[False],
        width: "custom_types.Integer",
        height: "custom_types.Integer",
        windowsize: Optional["custom_types.Integer"],
    ) -> dict[str, hv.Overlay]: ...

    def plot_observed_quantiles(
        self, *, use_ranks=True, display=True, width=600, height=400, windowsize=None
    ):
        """Visualize systematic patterns in observed data quantiles.

        This method creates hexagonal density plots showing the relationship between
        observed data values (or their ranks) and their corresponding quantiles
        within the posterior predictive distribution. A rolling mean overlay
        highlights systematic trends.

        :param use_ranks: Whether to use ranks instead of raw values for x-axis. Defaults to True.
        :type use_ranks: bool
        :param display: Whether to return formatted layout for display. Defaults to True.
        :type display: bool
        :param width: Width of individual plots in pixels. Defaults to 600.
        :type width: custom_types.Integer
        :param height: Height of individual plots in pixels. Defaults to 400.
        :type height: custom_types.Integer
        :param windowsize: Size of rolling window for trend line. Defaults to None (automatic).
        :type windowsize: Optional[custom_types.Integer]

        :returns: Quantile plots in requested format
        :rtype: Union[hv.Layout, dict[str, hv.Overlay]]

        Visualization Components:
        - Hexagonal binning showing density of (value, quantile) pairs
        - Rolling mean trend line highlighting systematic patterns
        - Colormap indicating point density for pattern identification

        Pattern Interpretation:
        - Horizontal trend line around 0.5 with uniformly distributed points indicates
          good calibration
        - Systematic deviations suggest model bias or miscalibration

        The hexagonal binning is particularly effective for visualizing large
        datasets where individual points would create overplotting issues.

        Example:
            >>> # Standard quantile analysis
            >>> quant_layout = mle_analysis.plot_observed_quantiles()
            >>> # Custom window size for trend analysis
            >>> quant_plots = mle_analysis.plot_observed_quantiles(
            ...     windowsize=50, use_ranks=False, display=False
            ... )
        """
        # Loop over quantiles for different observed variables
        plots: dict[str, hv.Overlay] = {}
        for varname, reference, observed in self._iter_pp_obs():

            # Get the quantiles of the observed data relative to the reference
            y = plotting.calculate_relative_quantiles(
                reference, observed[None] if observed.ndim == 1 else observed
            )

            # Flatten the data and update x to use rankings if requested
            x, y = observed.ravel(), y.ravel()
            x = stats.rankdata(x, method="ordinal") if use_ranks else x

            # Build the plot
            plots[varname] = plotting.hexgrid_with_mean(
                x=x, y=y, mean_windowsize=windowsize
            ).opts(
                xlabel=f"Observed Value {'Rank' if use_ranks else ''}: {varname}",
                ylabel=f"Observed Quantile: {varname}",
                title=f"Observed Quantiles: {varname}",
                width=width,
                height=height,
            )

        # If requested, display the plots
        if display:
            return hv.Layout(plots.values()).cols(1).opts(shared_axes=False)

        return plots

    @overload
    def run_ppc(
        self,
        *,
        use_ranks: bool,
        display: Literal[True],
        square_ecdf: bool,
        windowsize: Optional["custom_types.Integer"],
        quantiles: Sequence["custom_types.Float"],
        logy_ppc_samples: bool,
        subplot_width: "custom_types.Integer",
        subplot_height: "custom_types.Integer",
    ) -> pn.Column: ...

    @overload
    def run_ppc(
        self,
        *,
        use_ranks: bool,
        display: Literal[False],
        square_ecdf: bool,
        windowsize: Optional["custom_types.Integer"],
        quantiles: Sequence["custom_types.Float"],
        logy_ppc_samples: bool,
        subplot_width: "custom_types.Integer",
        subplot_height: "custom_types.Integer",
    ) -> list[dict[str, hv.Overlay]]: ...

    def run_ppc(
        self,
        *,
        use_ranks=True,
        display=True,
        square_ecdf=True,
        windowsize=None,
        quantiles=(0.025, 0.25, 0.5),
        logy_ppc_samples=False,
        subplot_width=600,
        subplot_height=400,
    ):
        """Execute comprehensive posterior predictive checking analysis.

        This method provides a complete posterior predictive checking workflow by
        combining multiple diagnostic approaches into a unified analysis. It runs
        three complementary diagnostic procedures and presents them in an organized,
        interactive dashboard.

        :param use_ranks: Whether to use ranks instead of raw values for x-axes.
            Defaults to True.
        :type use_ranks: bool
        :param display: Whether to return interactive dashboard layout. Defaults to True.
        :type display: bool
        :param square_ecdf: Whether to make ECDF plots square (width=height). Defaults
            to True.
        :type square_ecdf: bool
        :param windowsize: Size of rolling window for trend analysis. Defaults to
            None (automatic).
        :type windowsize: Optional[custom_types.Integer]
        :param quantiles: Quantiles for confidence intervals. Defaults to (0.025,
            0.25, 0.5).
        :type quantiles: Sequence[custom_types.Float]
        :param logy_ppc_samples: Whether to use log scale for posterior predictive
            plots. Defaults to False.
        :type logy_ppc_samples: bool
        :param subplot_width: Width of individual subplots in pixels. Defaults to 600.
        :type subplot_width: custom_types.Integer
        :param subplot_height: Height of individual subplots in pixels. Defaults to 400.
        :type subplot_height: custom_types.Integer

        :returns: Interactive dashboard or list of plot dictionaries
        :rtype: Union[pn.Column, list[dict[str, hv.Overlay]]]

        Comprehensive Analysis Components:
        1. **Posterior Predictive Samples**: Shows observed data against uncertainty intervals
        2. **Observed Quantiles**: Reveals systematic patterns in model calibration
        3. **Calibration Assessment**: Quantifies overall model calibration quality

        Dashboard Features:
        - Interactive variable selection across all diagnostic types
        - Consistent formatting and scaling across related plots
        - Automatic layout optimization for comparison and analysis
        - Widget-based navigation for multi-variable models

        Analysis Workflow:
        The method integrates three diagnostic perspectives:
        - **Predictive accuracy**: How well do predictions match observations?
        - **Calibration quality**: Are prediction intervals properly calibrated?
        - **Systematic bias**: Are there patterns indicating model inadequacy?

        Example:
            >>> # Complete interactive analysis
            >>> dashboard = mle_analysis.run_ppc()
            >>> dashboard  # Display in notebook
            >>>
            >>> # Programmatic access to individual components
            >>> ppc_plots, quant_plots, cal_plots = mle_analysis.run_ppc(display=False)
        """
        # Get ecdf widths and heights
        if square_ecdf:
            ecdf_width = subplot_width
            ecdf_height = ecdf_width
        else:
            ecdf_width = subplot_width
            ecdf_height = subplot_height

        # Get the different plots
        plots = [
            self.plot_posterior_predictive_samples(
                quantiles=quantiles,
                use_ranks=use_ranks,
                logy=logy_ppc_samples,
                display=False,
                width=subplot_width,
                height=subplot_height,
            ),
            self.plot_observed_quantiles(
                use_ranks=use_ranks,
                display=False,
                width=subplot_width,
                height=subplot_height,
                windowsize=windowsize,
            ),
            self.check_calibration(
                return_deviance=False,
                display=False,
                width=ecdf_width,
                height=ecdf_height,
            ),
        ]

        # If not displaying, return the plots
        if not display:
            return plots

        # Otherwise, display the plots
        plots, widget = pn.panel(
            hv.Layout(
                [
                    hv.HoloMap(plots[0], kdims="Variable").opts(
                        hv.opts.Scatter(framewise=True),
                        hv.opts.Area(framewise=True),
                    ),
                    hv.HoloMap(plots[1], kdims="Variable").opts(
                        hv.opts.HexTiles(framewise=True, axiswise=True, min_count=0),
                        hv.opts.Curve(framewise=True, color="darkgray"),
                    ),
                    hv.HoloMap(plots[2], kdims="Variable").opts(
                        hv.opts.Curve(framewise=True),
                    ),
                ]
            )
            .opts(shared_axes=False)
            .cols(1)
        )
        widget.align = ("start", "start")

        return pn.Column(widget, plots)

    @classmethod
    def from_disk(cls, path: str) -> "MLEInferenceRes":
        """Load MLEInferenceRes object from saved NetCDF file.

        :param path: Path to NetCDF file containing saved InferenceData
        :type path: str

        :returns: Reconstructed MLEInferenceRes object with all analysis capabilities
        :rtype: MLEInferenceRes

        This class method enables loading of previously saved analysis results,
        preserving all computed statistics and enabling continued analysis from
        where previous sessions left off.

        Example:
            >>> # Load previously saved results
            >>> mle_analysis = MLEInferenceRes.from_disk('saved_results.nc')
            >>> # Continue analysis with full functionality
            >>> dashboard = mle_analysis.run_ppc()
        """
        return cls(az.from_netcdf(path))
