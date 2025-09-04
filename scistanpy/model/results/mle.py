# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Maximum likelihood estimation results analysis and visualization for SciStanPy
models.

This module provides analysis tools for maximum likelihood estimation
results from SciStanPy models. It offers diagnostic plots, calibration
checks, and posterior predictive analysis tools designed specifically for MLE-based
inference workflows.

The module centers around three main classes: MLEParam for individual parameter
estimates, MLE for complete model results, and MLEInferenceRes class, which wraps
ArviZ InferenceData objects with specialized methods for MLE result analysis. Together,
these classes provide the estimated parameter values and the fitted probability
distributions resulting from MLE analysis, and allow for downstream analysis including
uncertainty quantification and posterior predictive sampling. It provides both individual
diagnostic tools and analysis workflows that combine multiple checks
into unified reporting interfaces.

Key Features:
    - Individual parameter MLE estimates with associated distributions
    - Complete model MLE results with loss tracking and diagnostics
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


Performance Considerations:
    - Batch sampling prevents memory overflow for large sample requests
    - GPU acceleration is preserved through PyTorch distribution objects

The module is designed to work with SciStanPy's MLE estimation workflow,
providing immediate access to model diagnostics and validation tools
once MLE fitting is complete. The MLE results can be used for various purposes
including model comparison, uncertainty quantification, and as initialization for
more sophisticated inference procedures like MCMC sampling.
"""

# pylint: disable=too-many-lines

from __future__ import annotations

import warnings

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
import pandas as pd
import panel as pn
import torch
import xarray as xr

from scipy import stats

from scistanpy import plotting

if TYPE_CHECKING:
    from scistanpy import custom_types
    from scistanpy import model as ssp_model


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


class MLEParam:
    """Container for maximum likelihood estimate of a single model parameter.

    This class encapsulates the MLE result for an individual parameter,
    including the estimated value and the corresponding fitted probability
    distribution. It provides methods for sampling from the fitted distribution
    and accessing parameter properties.

    :param name: Name of the parameter in the model
    :type name: str
    :param value: Maximum likelihood estimate of the parameter value.
                 Can be None for some distribution types.
    :type value: Optional[npt.NDArray]
    :param distribution: Fitted probability distribution object
    :type distribution: custom_types.SciStanPyDistribution

    :ivar name: Parameter name identifier
    :ivar mle: Stored maximum likelihood estimate
    :ivar distribution: Fitted distribution for sampling and analysis

    The class maintains both point estimates and distributional representations,
    enabling both point-based analysis and uncertainty quantification through
    sampling from the fitted distribution.

    Example:
        >>> param = MLEParam('mu', np.array([2.5]), fitted_normal_dist)
        >>> samples = param.draw(1000, seed=42)
        >>> print(f"MLE estimate: {param.mle}")
    """

    def __init__(
        self,
        name: str,
        value: Optional[npt.NDArray],
        distribution: "custom_types.SciStanPyDistribution",
    ):

        # Store the inputs
        self.name = name
        self.mle = value
        self.distribution = distribution

    def draw(
        self,
        n: int,
        *,
        seed: Optional[custom_types.Integer] = None,
        batch_size: Optional[custom_types.Integer] = None,
    ) -> npt.NDArray:
        """Sample from the fitted parameter distribution.

        This method generates samples from the parameter's fitted probability
        distribution using batch processing to handle large sample requests.

        :param n: Total number of samples to generate
        :type n: int
        :param seed: Random seed for reproducible sampling. Defaults to None.
        :type seed: Optional[custom_types.Integer]
        :param batch_size: Size of batches for memory-efficient sampling.
                          Defaults to None (uses n as batch size).
        :type batch_size: Optional[custom_types.Integer]

        :returns: Array of samples from the fitted distribution
        :rtype: npt.NDArray

        Batch processing prevents memory overflow when requesting large numbers
        of samples from complex distributions, particularly important when
        working with GPU-based computations.

        Example:
            >>> # Generate 10000 samples in batches of 1000
            >>> samples = param.draw(10000, batch_size=1000, seed=42)
            >>> print(f"Sample mean: {samples.mean()}")
        """
        # Set the seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # If the batch size is not provided, we set it to `n`
        batch_size = batch_size or n

        # Calculate the batch sizes for each sampling iteration
        batch_sizes = [batch_size] * (n // batch_size)
        if (n_remaining := n % batch_size) > 0:
            batch_sizes.append(n_remaining)

        # Sample from the distribution
        return np.concatenate(
            [
                self.distribution.sample((batch_size,)).detach().cpu().numpy()
                for batch_size in batch_sizes
            ]
        )


class MLE:
    """Complete maximum likelihood estimation results for a SciStanPy model.

    This class encapsulates the full results of MLE parameter estimation,
    including parameter estimates, fitted distributions, optimization
    diagnostics, and utilities for further analysis. It provides a
    comprehensive interface for working with MLE results.

    :param model: Original SciStanPy model
    :type model: ssp_model.Model
    :param mle_estimate: Dictionary of parameter names to their MLE values
    :type mle_estimate: dict[str, npt.NDArray]
    :param distributions: Dictionary of parameter names to fitted distributions
    :type distributions: dict[str, torch.distributions.Distribution]
    :param losses: Array of loss values throughout optimization
    :type losses: npt.NDArray
    :param data: Observed data used for parameter estimation
    :type data: dict[str, npt.NDArray]

    :ivar model: Reference to the original model
    :ivar data: Observed data used for fitting
    :ivar model_varname_to_mle: Mapping from parameter names to MLEParam objects
    :ivar losses: DataFrame containing loss trajectory and diagnostics

    :raises ValueError: If MLE estimate keys are not subset of distribution keys
    :raises ValueError: If parameter names conflict with existing attributes

    The class automatically creates attributes for each parameter, allowing
    direct access like `mle_result.mu` for a parameter named 'mu'. It also
    provides comprehensive utilities for visualization, sampling, and
    integration with Bayesian analysis workflows.

    Key Features:
    - Direct attribute access to individual parameter results
    - Comprehensive loss trajectory tracking and visualization
    - Efficient sampling from fitted parameter distributions
    - Integration with ArviZ for Bayesian workflow compatibility
    - Memory-efficient batch processing for large sample requests

    Example:
        >>> mle_result = model.mle(data=observed_data)
        >>> mu_samples = mle_result.mu.draw(1000)  # Direct parameter access
        >>> loss_plot = mle_result.plot_loss_curve()
        >>> inference_data = mle_result.get_inference_obj()
    """

    def __init__(
        self,
        model: "ssp_model.Model",
        mle_estimate: dict[str, npt.NDArray],
        distributions: dict[str, torch.distributions.Distribution],
        losses: npt.NDArray,
        data: dict[str, npt.NDArray],
    ):

        # The keys of the mle estimate should be a subset of the keys of the distributions
        if not set(mle_estimate.keys()).issubset(distributions.keys()):
            raise ValueError(
                "Keys of mle estimate should be a subset of the keys of the distributions"
            )

        # Record the model and data
        self.model = model
        self.data = data

        # Store inputs. Each key in the mle estimate will be mapped to an instance
        # variable
        self.model_varname_to_mle: dict[str, MLEParam] = {
            key: MLEParam(name=key, value=mle_estimate.get(key), distribution=value)
            for key, value in distributions.items()
        }

        # Set an attribute for all MLE parameters
        for k, v in self.model_varname_to_mle.items():
            if hasattr(self, k):
                raise ValueError(
                    f"MLE parameter {k} already exists in the model. Please rename it."
                )
            setattr(self, k, v)

        # Record the loss trajectory as a pandas dataframe
        self.losses = pd.DataFrame(
            {
                "-log pdf/pmf": losses,
                "iteration": np.arange(len(losses)),
                "shifted log(-log pdf/pmf)": losses - losses.min() + 1,
            },
        )

    def plot_loss_curve(self, logy: bool = True):
        """Generate interactive plot of the optimization loss trajectory.

        This method creates a visualization of how the loss function evolved
        during the optimization process, providing insights into convergence
        behavior and optimization effectiveness.

        :param logy: Whether to use logarithmic y-axis scaling. Defaults to True.
        :type logy: bool

        :returns: Interactive HoloViews plot of the loss curve

        The plot automatically handles:
        - Logarithmic scaling with proper handling of negative/zero values
        - Appropriate axis labels and titles based on scaling choice
        - Interactive features for detailed examination of convergence
        - Warning messages for problematic loss trajectories

        For logarithmic scaling with non-positive loss values, the method
        automatically switches to a shifted logarithmic scale to maintain
        visualization quality while issuing appropriate warnings.

        Example:
            >>> # Standard logarithmic loss plot
            >>> loss_plot = mle_result.plot_loss_curve()
            >>> # Linear scale loss plot
            >>> linear_plot = mle_result.plot_loss_curve(logy=False)
        """
        # Get y-label and title
        y = "-log pdf/pmf"
        if logy:
            if self.losses["-log pdf/pmf"].min() <= 0:
                warnings.warn("Negative values in loss curve. Using shifted log10.")
                y = "shifted log(-log pdf/pmf)"
                ylabel = y
            else:
                ylabel = "log(-log pdf/pmf)"
            title = "Log Loss Curve"
        else:
            ylabel = "-log pdf/pmf"
            title = "Loss Curve"

        return self.losses.hvplot.line(
            x="iteration", y=y, title=title, logy=logy, ylabel=ylabel
        )

    @overload
    def draw(
        self,
        n: custom_types.Integer,
        *,
        seed: Optional[custom_types.Integer],
        as_xarray: Literal[True],
        as_inference_data: Literal[False],
        batch_size: Optional[custom_types.Integer] = None,
    ) -> xr.Dataset: ...

    @overload
    def draw(
        self,
        n: custom_types.Integer,
        *,
        seed: Optional[custom_types.Integer],
        as_xarray: Literal[False],
        batch_size: Optional[custom_types.Integer] = None,
    ) -> dict[str, npt.NDArray]: ...

    def draw(self, n, *, seed=None, as_xarray=False, batch_size=None):
        """Generate samples from all fitted parameter distributions.

        This method draws samples from the fitted distributions of all model
        parameters. It supports multiple output formats for integration with
        different analysis workflows.

        :param n: Number of samples to draw from each parameter distribution
        :type n: custom_types.Integer
        :param seed: Random seed for reproducible sampling. Defaults to None.
        :type seed: Optional[custom_types.Integer]
        :param as_xarray: Whether to return results as xarray Dataset. Defaults to False.
        :type as_xarray: bool
        :param batch_size: Batch size for memory-efficient sampling. Defaults to None.
        :type batch_size: Optional[custom_types.Integer]

        :returns: Sampled parameter values in requested format
        :rtype: Union[dict[str, npt.NDArray], xr.Dataset]

        Output Formats:
        - Dictionary (default): Keys are parameter names, values are sample arrays
        - xarray Dataset: Structured dataset with proper dimension labels and coordinates

        This is particularly useful for:
        - Uncertainty propagation through model predictions
        - Bayesian model comparison and validation
        - Posterior predictive checking with MLE-based approximations
        - Sensitivity analysis of parameter estimates

        Example:
            >>> # Draw samples as dictionary
            >>> samples = mle_result.draw(1000, seed=42)
            >>> # Draw as structured xarray Dataset
            >>> dataset = mle_result.draw(1000, as_xarray=True, batch_size=100)
        """
        # Set the seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Draw samples
        draws = {
            self.model.all_model_components_dict[k]: v.draw(n, batch_size=batch_size)
            for k, v in self.model_varname_to_mle.items()
        }

        # If returning as an xarray or InferenceData object, convert the draws to
        # an xarray format.
        if as_xarray:
            return self.model._dict_to_xarray(draws)  # pylint: disable=protected-access

        # If we make it here, we are not returning as an xarray or InferenceData
        # object, so we need to convert the parameters to their original names
        # and return them as a dictionary
        return {k.model_varname: v for k, v in draws.items()}

    def get_inference_obj(
        self,
        n: custom_types.Integer = 1000,
        *,
        seed: Optional[custom_types.Integer] = None,
        batch_size: Optional[custom_types.Integer] = None,
    ) -> MLEInferenceRes:
        """Create ArviZ-compatible inference data object from MLE results.

        This method constructs a comprehensive inference data structure that
        integrates MLE results with the ArviZ ecosystem for Bayesian analysis.
        It organizes parameter samples, observed data, and posterior predictive
        samples into a standardized format.

        :param n: Number of samples to generate for the inference object. Defaults to 1000.
        :type n: custom_types.Integer
        :param seed: Random seed for reproducible sample generation. Defaults to None.
        :type seed: Optional[custom_types.Integer]
        :param batch_size: Batch size for memory-efficient sampling. Defaults to None.
        :type batch_size: Optional[custom_types.Integer]

        :returns: Structured inference data object with all MLE results
        :rtype: results.MLEInferenceRes

        The resulting inference object contains:
        - **Posterior samples**: Draws from fitted parameter distributions
        - **Observed data**: Original data used for parameter estimation
        - **Posterior predictive**: Samples from observable distributions

        Data Organization:
        - Latent parameters are stored in the main posterior group
        - Observable parameters become posterior predictive samples
        - Observed data is stored separately for comparison
        - All data maintains proper dimensional structure and labeling

        This enables:
        - Integration with ArviZ plotting and diagnostic functions
        - Model comparison
        - Posterior predictive checking workflows
        - Standardized reporting and visualization

        Example:
            >>> # Create inference object with default settings
            >>> inference_obj = mle_result.get_inference_obj()
            >>> # Generate larger sample with custom batch size
            >>> inference_obj = mle_result.get_inference_obj(
            ...     n=5000, batch_size=500, seed=42
            ... )
        """
        # Get the samples from the posterior
        draws = self.draw(n, seed=seed, as_xarray=True, batch_size=batch_size)

        # Otherwise, we also are going to want to attach the observed data
        # to the InferenceData object. First, rename the "n" dimension to "sample"
        # and add a dummy "chain" dimension
        draws = draws.rename_dims({"n": "draw"})
        draws = draws.expand_dims("chain", 0)

        # Now separate out the observables from the latent variables. Build
        # the initial inference data object with the latent variables
        inference_data = az.convert_to_inference_data(
            draws[
                [
                    varname
                    for varname, mle_param in self.model_varname_to_mle.items()
                    if not self.model.all_model_components_dict[varname].observable
                ]
            ]
        )

        # Add the observables and the observed data to the inference data object
        # pylint: disable=protected-access
        inference_data.add_groups(
            observed_data=xr.Dataset(
                data_vars={
                    k: self.model._compress_for_xarray(v)[0]
                    for k, v in self.data.items()
                }
            ),
            posterior_predictive=draws[
                [
                    varname
                    for varname, mle_param in self.model_varname_to_mle.items()
                    if self.model.all_model_components_dict[varname].observable
                ]
            ],
        )
        return MLEInferenceRes(inference_data)
