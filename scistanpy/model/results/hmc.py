# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Hamiltonian Monte Carlo (HMC) sampling results analysis and diagnostics.

This module provides tools for analyzing and diagnosing HMC sampling results from
Stan models. It offers specialized classes and functions for processing
MCMC output, conducting diagnostic tests, and creating interactive visualizations
for model validation and troubleshooting.

The module centers around the :py:class:`~scistanpy.model.results.hmc.SampleResults`
class, which extends :py:class:`~scistanpy.model.results.MLEInferenceRes` to provide
HMC-specific functionality including convergence diagnostics, sample quality assessment,
and specialized visualization tools for identifying problematic parameters and sampling
behavior.

Key Features:

    - MCMC diagnostic test suites
    - Interactive visualization tools for failed diagnostics
    - Efficient CSV to NetCDF conversion for large datasets
    - Dask-enabled processing for memory-intensive operations
    - Specialized trace plot analysis for problematic variables
    - Automated detection and reporting of sampling issues

Diagnostic Capabilities:

    - R-hat convergence assessment
    - Effective sample size (ESS) evaluation
    - Energy fraction of missing information (E-BFMI) analysis
    - Divergence detection and analysis
    - Tree depth saturation monitoring
    - Variable-specific failure pattern identification

The module is designed to handle both small-scale interactive analysis and
large-scale batch processing of MCMC results, with particular attention to
memory efficiency and computational performance for complex models.
"""

from __future__ import annotations

import functools
import inspect
import itertools
import os.path
import re
import warnings
from glob import glob
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Literal,
    Sequence,
    Union,
    overload,
)

import arviz as az
import h5netcdf
import holoviews as hv
import numpy as np
import numpy.typing as npt
import panel as pn
import xarray as xr
from cmdstanpy.cmdstan_args import CmdStanArgs, SamplerArgs
from cmdstanpy.stanfit import CmdStanMCMC, RunSet
from cmdstanpy.utils import check_sampler_csv, scan_config
from tqdm import tqdm

from scistanpy import plotting
from scistanpy.defaults import (
    DEFAULT_EBFMI_THRESH,
    DEFAULT_ESS_THRESH,
    DEFAULT_RHAT_THRESH,
)
from scistanpy.model.results.base_classes import (
    InferenceRes,
    SciStanPyToNetCDFConverter,
)

if TYPE_CHECKING:
    from scistanpy import Model, custom_types

# pylint: disable=too-many-lines


def _symmetrize_quantiles(
    quantiles: Sequence[custom_types.Float],
) -> list[custom_types.Float]:
    """Symmetrize and validate quantile sequences for plotting.

    This utility function takes a sequence of quantiles and creates a symmetric
    set by adding complementary quantiles and ensuring the median is included.
    It also validates that all quantiles are properly bounded.

    :param quantiles: Sequence of quantile values between 0 and 1
    :type quantiles: Sequence[custom_types.Float]

    :returns: Symmetrized and sorted list of quantiles including median
    :rtype: list[custom_types.Float]

    :raises ValueError: If quantiles are not between 0 and 1
    :raises AssertionError: If result doesn't have odd length or include median

    The function ensures:
    - All quantiles are between 0 and 1 (exclusive)
    - Complementary quantiles are added (e.g., 0.1 → 0.1, 0.9)
    - Median (0.5) is always included
    - Result has odd length for symmetric confidence intervals

    Example:
        >>> quantiles = _symmetrize_quantiles([0.1, 0.2])
        >>> # Returns [0.1, 0.2, 0.5, 0.8, 0.9]
    """
    # Get the quantiles
    quantiles = sorted(set(quantiles) | {1 - q for q in quantiles} | {0.5})

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

    return quantiles


class VariableAnalyzer:
    """Interactive analysis tool for variables that fail MCMC diagnostic tests.

    This class provides an interactive interface for analyzing individual variables
    that have failed diagnostic tests during MCMC sampling. It creates a dashboard
    with widgets for selecting variables, metrics, and specific array indices,
    along with trace plots showing the problematic sampling behavior.

    :param sample_results: SampleResults object containing MCMC diagnostics
    :type sample_results: SampleResults
    :param plot_width: Width of plots in pixels. Defaults to 800.
    :type plot_width: custom_types.Integer
    :param plot_height: Height of plots in pixels. Defaults to 400.
    :type plot_height: custom_types.Integer
    :param plot_quantiles: Whether to plot quantiles vs raw values. Defaults to False.
    :type plot_quantiles: bool

    :ivar sample_results: Reference to source sampling results
    :ivar plot_quantiles: Flag controlling plot content type
    :ivar n_chains: Number of MCMC chains in the results
    :ivar x: Array of step indices for x-axis
    :ivar failed_vars: Dictionary mapping variable names to failure information
    :ivar varchoice: Widget for selecting variables to analyze
    :ivar metricchoice: Widget for selecting diagnostic metrics
    :ivar indexchoice: Widget for selecting array indices
    :ivar plot_width: Recorded width of plots
    :ivar plot_height: Recorded height of plots
    :ivar fig: HoloViews pane containing the current plot
    :ivar layout: Panel layout containing all interface elements

    The analyzer automatically identifies variables that have failed diagnostic
    tests and organizes them by failure type. It provides trace plots that can
    show either raw parameter values or their quantiles relative to passing
    samples, helping identify the nature of sampling problems.

    Key Features:

    - Automatic identification of failed variables and metrics
    - Interactive widget-based navigation
    - Trace plots with chain-specific coloring
    - Quantile-based analysis for identifying sampling bias
    - Real-time plot updates based on widget selections

    .. note::
        This class should not be instantiated directly. Use the
        :py:meth:`~scistanpy.model.results.hmc.SampleResults.plot_variable_failure_quantile_traces`
        method of :py:class:`~scistanpy.model.results.hmc.SampleResults` instead.
    """

    # pylint: disable=attribute-defined-outside-init

    def __init__(
        self,
        sample_results: "SampleResults",
        plot_width: custom_types.Integer = 800,
        plot_height: custom_types.Integer = 400,
        plot_quantiles: bool = False,
    ):

        # Hold a reference to the sample results
        self.sample_results = sample_results
        self.plot_quantiles = plot_quantiles

        # Some placeholders for whether or not we both updating the plot
        self._previous_vals = None

        # Record the number of chains and an array for the steps
        self.n_chains = sample_results.inference_obj.posterior.sizes["chain"]
        self.x = np.arange(sample_results.inference_obj.posterior.sizes["draw"])

        # Identify failed variables
        self.failed_vars = {}
        self._identify_failed_vars()

        # Set up the holoviews plot
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.fig = pn.pane.HoloViews(
            hv.Curve([]).opts(width=self.plot_width, height=self.plot_height),
            name="Plot",
            align="center",
        )

        # Set up widgets
        self.varchoice = pn.widgets.Select(
            name="Variable", options=list(self.failed_vars.keys())
        )
        self.metricchoice = pn.widgets.Select(name="Metric", options=[])
        self.indexchoice = pn.widgets.Select(name="Index", options=[])

        self.varchoice.param.watch(self._get_var_data, "value")
        self.varchoice.param.watch(self._update_metric_selector, "value")
        self.varchoice.param.watch(self._get_metric_data, "value")
        self.varchoice.param.watch(self._update_index_selector, "value")
        self.varchoice.param.watch(self._update_plot, "value")

        self.metricchoice.param.watch(self._get_metric_data, "value")
        self.metricchoice.param.watch(self._update_index_selector, "value")
        self.metricchoice.param.watch(self._update_plot, "value")

        self.indexchoice.param.watch(self._update_plot, "value")

        # Package widgets and figure into a layout
        self.layout = pn.Column(
            self.varchoice,
            self.metricchoice,
            self.indexchoice,
            self.fig,
        )

        # Trigger the initial data retrieval and plotting
        self.varchoice.param.trigger("value")

    def _identify_failed_vars(self):
        """Identify variables that failed diagnostic tests.

        This method analyzes the variable diagnostic tests in the SampleResults
        object to identify which variables failed which tests, organizing the
        information for easy access by the widget interface.

        The method populates the failed_vars dictionary with variable names as
        keys and tuples containing dimension information and failure details
        as values.
        """

        # Identify both the variables that fail and their indices
        for (
            varname,
            vartests,
        ) in self.sample_results.inference_obj.variable_diagnostic_tests.items():

            # The first variable name is the metric
            assert vartests.dims[0] == "metric"

            # Process each metric for this variable
            metric_test_summaries = {}
            for metric in vartests.metric:

                # Get the test results for the metric
                metrictests = vartests.sel(metric=metric).to_numpy()

                # Get the indices of the failing tests
                if metrictests.ndim > 0:
                    failing_inds = [
                        ".".join(map(str, indices))
                        for indices in zip(*np.nonzero(metrictests))
                    ]
                else:
                    failing_inds = [""] if metrictests else []  # Just variable name

                # Record the failing tests
                if len(failing_inds) > 0:
                    metric_test_summaries[metric.item()] = failing_inds

            # If there are any failing tests, add them to the dictionary
            if len(metric_test_summaries) > 0:
                self.failed_vars[varname] = (vartests.dims[1:], metric_test_summaries)

    def _update_metric_selector(self, event):  # pylint: disable=unused-argument
        """Update metric selection options based on selected variable.

        :param event: Panel event object (required for callback interface)

        This callback method updates the available metric options when a new
        variable is selected, ensuring only relevant diagnostic metrics are
        shown for the current variable.
        """

        # Get the current variable name
        current_name = self.metricchoice.value

        # Update the metric choice options
        self.metricchoice.options = list(
            self.failed_vars[self.varchoice.value][1].keys()
        )

        # If the currently selected metric is not in the new options, set it to
        # the first one
        if current_name not in self.metricchoice.options:
            self.metricchoice.value = self.metricchoice.options[0]

    def _update_index_selector(self, event):  # pylint: disable=unused-argument
        """Update index selection options based on selected variable and metric.

        :param event: Panel event object (required for callback interface)

        This callback method updates the available index options when a new
        variable or metric is selected, showing only the array indices that
        failed the selected diagnostic test.
        """

        # Get the current variable name
        current_name = self.indexchoice.value

        # Update the index choice options
        opts = self.failed_vars[self.varchoice.value][1]
        if len(opts) == 0:
            self.indexchoice.options = []
            self.indexchoice.value = None
            return
        self.indexchoice.options = opts[self.metricchoice.value]

        # If the currently selected index is not in the new options, set it to
        # the first one
        if current_name not in self.indexchoice.options:
            self.indexchoice.value = self.indexchoice.options[0]

    def _get_var_data(self, event):  # pylint: disable=unused-argument
        """Retrieve and prepare data for the selected variable.

        :param event: Panel event object (required for callback interface)

        This callback method loads the sample data for the currently selected
        variable and prepares it for analysis and visualization, including
        dimension validation and array reshaping.
        """

        # Get the samples for the selected variable
        self._samples = getattr(
            self.sample_results.inference_obj.posterior, self.varchoice.value
        )

        # Check the dimensions of the variable
        assert self._samples.dims[:2] == ("chain", "draw")
        assert self._samples.dims[2:] == self.failed_vars[self.varchoice.value][0]

        # We also want the samples as a numpy array with the draw and chain dimensions
        # last
        self._np_samples = np.moveaxis(self._samples.to_numpy(), [0, 1], [-2, -1])

    def _get_metric_data(self, event):  # pylint: disable=unused-argument
        """Retrieve and prepare diagnostic test data for the selected metric.

        :param event: Panel event object (required for callback interface)

        This callback method processes the diagnostic test results for the
        currently selected metric, separating failing and passing samples
        for comparative analysis.
        """

        # Get the tests for the selected variable and metric
        tests = self.sample_results.inference_obj.variable_diagnostic_tests[
            self.varchoice.value
        ].sel(metric=self.metricchoice.value)

        # Make sure the dimensions are correct
        assert tests.dims == self._samples.dims[2:]

        # Tests as numpy array
        tests = tests.to_numpy()

        # Map between index name and failing index. Separate the failing and passing
        # samples. Note the different approach for scalar variables (no indices)
        if tests.ndim == 0:
            self._index_map = {"": ...}  # No indices, no map
            if tests:
                self._failing_samples = self._np_samples.copy()
                self._passing_samples = np.array([], dtype=self._np_samples.dtype)
            else:
                self._failing_samples = np.array([], dtype=self._np_samples.dtype)
                self._passing_samples = self._np_samples.copy()
        else:
            self._index_map = {
                ".".join(map(str, indices)): i
                for i, indices in enumerate(zip(*np.nonzero(tests)))
            }
            self._failing_samples, self._passing_samples = (
                self._np_samples[tests],
                self._np_samples[~tests],
            )

    def _update_plot(self, event):  # pylint: disable=unused-argument
        """Update the trace plot based on current widget selections.

        :param event: Panel event object (required for callback interface)

        This callback method generates new trace plots when widget selections
        change, creating overlays that show sampling traces for each chain
        with appropriate styling and hover information.
        """

        # Skip the update if the values haven't changed
        if self._previous_vals == (
            new_vals := (
                self.varchoice.value,
                self.metricchoice.value,
                self.indexchoice.value,
            )
        ):
            return
        self._previous_vals = new_vals

        # Get the variable name
        varname = self.varchoice.value
        if self.indexchoice.value is not None:
            varname += "." + self.indexchoice.value

        # Calculate the relative quantiles for the selected failing index relative
        # to the passing samples
        if self.indexchoice.value == "":  # For scalars
            failing_samples = self._failing_samples
        else:
            failing_samples = self._failing_samples[
                self._index_map[self.indexchoice.value]
            ]
        n_failing, n_passing = len(self._failing_samples), len(self._passing_samples)

        # We always calculate quantiles. If there are no reference samples, however,
        # quantiles are undefined. We raise a warning if there are more failing
        # samples than passing samples and the user is trying to plot quantiles.
        if n_passing == 0:
            if self.plot_quantiles:
                raise ValueError(
                    f"No passing samples found for {self.varchoice.value}. Cannot "
                    "calculate quantiles."
                )
            failing_quantiles = np.full_like(failing_samples, np.nan)
        else:
            if n_failing > n_passing and self.plot_quantiles:
                warnings.warn(
                    "There are more failing samples than passing samples for "
                    f"{self.varchoice.values}. Consider plotting true values instead."
                )
            failing_quantiles = plotting.calculate_relative_quantiles(
                reference=self._passing_samples,
                observed=failing_samples[None],
            )[0]

        # We should have shape (n_chains, n_draws)
        assert (
            failing_samples.shape
            == failing_quantiles.shape
            == (self.n_chains, self.x.size)
        )

        # Build an overlay for the failing quantiles and use it to update the plot
        overlay_dict = {}
        for i in range(self.n_chains):
            if self.plot_quantiles:
                order = (failing_quantiles[i], failing_samples[i])
                order_vdim = ["Sample Quantile", "Sample Value"]
            else:
                order = (failing_samples[i], failing_quantiles[i])
                order_vdim = ["Sample Value", "Sample Quantile"]
            overlay_dict[i] = hv.Curve(
                (self.x, *order, n_passing),
                kdims=["Step"],
                vdims=[*order_vdim, "N Reference Variables"],
            ).opts(line_alpha=0.5, **({"ylim": (0, 1)} if self.plot_quantiles else {}))

        # Create the overlay
        self.fig.object = hv.NdOverlay(overlay_dict, kdims="Chain").opts(
            title=f"{self.metricchoice.value}: {varname}",
            tools=["hover"],
            width=self.plot_width,
            height=self.plot_height,
        )

    def display(self):
        """Display the complete interactive analysis interface.

        :returns: Panel layout containing all widgets and plots
        :rtype: pn.Layout

        This method returns the complete interactive interface for display
        in Jupyter notebooks or Panel applications.
        """
        return self.layout


class CmdStanMCMCToNetCDFConverter(SciStanPyToNetCDFConverter):
    """Object responsible for converting CmdStan CSV output to NetCDF format. This
    class is used internally and should not be instantiated directly in most use
    cases.

    This class handles the conversion of CmdStan CSV output files to NetCDF
    format, providing efficient storage and access for large MCMC datasets.
    It properly organizes data into appropriate groups and handles dimension
    naming and chunking strategies.

    :param results: CmdStanMCMC object or path to CSV files
    :type results: Union[CmdStanMCMC, str, list[str], os.PathLike]
    :param model: SciStanPy model object for metadata extraction
    :type model: Model
    :param data: Optional observed data dictionary. Defaults to None.
    :type data: Optional[dict[str, Any]]

    The converter handles:

    - Automatic detection of variable types and dimensions
    - Proper NetCDF group organization
    - Chunking strategies for large datasets
    - Data type optimization based on precision requirements
    """

    def __init__(
        self,
        results: CmdStanMCMC | str | list[str] | os.PathLike,
        model: "Model",
        data: dict[str, Any] | None = None,
    ):
        """
        Initialization involves collecting information about the different variables
        in the fit object. This includes the names of the variables, their shapes,
        and their types. This information is used to create the HDF5 file.
        """

        # If `results` is a string, we assume we need to load it from disk
        if isinstance(results, str):
            results = fit_from_csv_noload(results)

        # Init parent class
        super().__init__(results=results, model=model, data=data)

        # Additional placeholder for method data types
        self.method_var_dtypes: dict[str, Union[type[np.floating], type[np.integer]]]

        # Record the config object
        self.config = results.metadata.cmdstan_config

        # The number of chains is per thread. We want the number of chains total
        self.num_chains = len(results.runset.csv_files)

        # How many samples are we expecting?
        self.num_draws = self.config["num_samples"] + (
            self.config["num_warmup"] if self.config["save_warmup"] else 0
        )

        # Argsort the columns for each variable such that the columns are in row-major
        # order. This is important for efficiently saving to the HDF5 file.
        self.varname_to_column_order = self._get_c_order()

    def _get_c_order(self) -> dict[str, npt.NDArray[np.int64]]:
        """Determine optimal column ordering for efficient NetCDF storage.

        :returns: Dictionary mapping variable names to column order arrays
        :rtype: dict[str, npt.NDArray[np.int64]]

        This method analyzes the CSV column structure to determine the optimal
        ordering for writing multi-dimensional arrays to NetCDF format, ensuring
        efficient memory access patterns and proper array reconstruction.
        """
        # We need a regular expression for parsing indices out of variable names
        ind_re = re.compile(r"\[([0-9,]+)\]")

        # Get the indices of each column in the csv files. If there is no match,
        # then there are no indices and we return an empty tuple.
        column_indices = [
            (
                tuple(int(ind) - 1 for ind in match_obj.group(1).split(","))
                if (match_obj := ind_re.search(col))
                else ()
            )
            for col in self.results.column_names
        ]

        # Now we assign the row-major argsort of indices to each variable
        varname_to_column_order = {}
        for varname, var in itertools.chain(
            self.results.metadata.method_vars.items(),
            self.results.metadata.stan_vars.items(),
        ):

            # Slice out the indices for this variable
            var_inds = column_indices[var.start_idx : var.end_idx]

            # All indices must be unique
            assert len(set(var_inds)) == len(var_inds)

            # All indices should have the appropriate number of dimensions
            assert all(len(ind) == len(var.dimensions) for ind in var_inds)

            # All indices should fit within the dimensions of the variable
            for dimind, dimsize in enumerate(var.dimensions):
                assert all(ind[dimind] < dimsize for ind in var_inds)

            # Argsort the indices such that the last dimension changes fastest (c-major)
            varname_to_column_order[varname] = np.array(
                sorted(range(len(var_inds)), key=var_inds.__getitem__)
            )

        return varname_to_column_order

    def _write_attributes(self, netcdf_file: h5netcdf.File) -> None:

        # Run inherited method
        super()._write_attributes(netcdf_file)

        # Write attributes to the file
        for attr in (
            "stan_version_major",
            "stan_version_minor",
            "stan_version_patch",
            "model",
            "start_datetime",
            "method",
            "num_samples",
            "num_warmup",
            "save_warmup",
            "max_depth",
            "num_chains",
            "data_file",
            "diagnostic_file",
            "seed",
            "sig_figs",
            "num_threads",
            "stanc_version",
        ):
            netcdf_file.attrs[attr] = self.results.metadata.cmdstan_config[attr]

    def _create_netcdf_groups(
        self, netcdf_file: h5netcdf.File
    ) -> dict[str, h5netcdf.Group]:

        # Run inherited method to get the basic groups
        groups = super()._create_netcdf_groups(netcdf_file)

        # Build the meta data group
        groups["metadata_group"] = netcdf_file.create_group("sample_stats")

        # Get the data types for the method and stan variables
        # pylint: disable=protected-access
        self.method_var_dtypes = {
            "lp__": self.__class__._NP_TYPE_MAP[self.precision]["float"],
            "accept_stat__": self.__class__._NP_TYPE_MAP[self.precision]["float"],
            "stepsize__": self.__class__._NP_TYPE_MAP[self.precision]["float"],
            "treedepth__": self.__class__._NP_TYPE_MAP[self.precision]["int"],
            "n_leapfrog__": self.__class__._NP_TYPE_MAP[self.precision]["int"],
            "divergent__": self.__class__._NP_TYPE_MAP[self.precision]["int"],
            "energy__": self.__class__._NP_TYPE_MAP[self.precision]["float"],
        }

        # Create variables for each of the method variables. Build a mapping
        # from the variable name to the dataset object. We store all of the
        # method variables with a single chunk.
        self.varname_to_dset.update(
            {
                varname: groups["metadata_group"].create_variable(
                    name=varname,
                    dimensions=("chain", "draw"),
                    dtype=self.method_var_dtypes[varname],
                    chunks=(self.num_chains, self.num_draws),
                )
                for varname in self.results.metadata.method_vars.keys()
            }
        )

        return groups

    def _stream_draws(
        self,
    ) -> Generator[tuple[int, int, dict[str, npt.NDArray]], None, None]:

        # Now we populate the datasets with the data from the csv files
        for chain_ind, csv_file in enumerate(
            tqdm(sorted(self.results.runset.csv_files), desc="Converting CSV to NetCDF")
        ):
            for draw_ind, draw in enumerate(
                tqdm(
                    self._parse_csv(filename=csv_file),
                    total=self.num_draws,
                    desc=f"Processing chain {chain_ind + 1}",
                    leave=False,
                    position=1,
                )
            ):
                yield chain_ind, draw_ind, draw

            # We must have all the draws for this chain
            assert draw_ind == self.num_draws - 1  # pylint: disable=W0631

    def _parse_csv(
        self, filename: str
    ) -> Generator[dict[str, npt.NDArray], None, None]:
        """Parse CSV file and yield properly formatted arrays.

        :param filename: Path to CSV file to parse
        :type filename: str

        :yields: Dictionary of variable names to properly shaped arrays for each draw
        :rtype: Generator[dict[str, npt.NDArray], None, None]

        This generator function parses CSV files line by line, converting each
        row into properly typed and shaped NumPy arrays according to the
        variable specifications determined during initialization.
        """

        # Start parsing the file line by line
        with open(filename, "r", encoding="utf-8") as csv_file:
            for line in csv_file:

                # Skip the header information. This was parsed by the fit object.
                if line.startswith("#") or line.startswith("lp__"):
                    continue

                # Split the components of the line
                vals = line.strip().split(",")

                # Build the arrays for each variable
                processed_vals = {}
                for varname, dtype in itertools.chain(
                    self.method_var_dtypes.items(), self.var_dtypes.items()
                ):

                    # Get the variable object from the metadata
                    var = getattr(
                        self.results.metadata,
                        "stan_vars" if varname in self.var_dtypes else "method_vars",
                    )[varname]

                    # Using that variable object, slice out the data, convert to
                    # an appropriately typed numpy array, reorder it such that it
                    # is a flattened row-major array, then reshape it to the final
                    # shape. Note that we convert to a float first followed by a
                    # conversion to the final type. This is because some numbers
                    # are stored in the CSV as scientific notation.
                    processed_val = np.array(
                        [float(val) for val in vals[var.start_idx : var.end_idx]]
                    )[self.varname_to_column_order[varname]].reshape(var.dimensions)
                    if issubclass(dtype, np.integer):
                        processed_val = np.rint(processed_val)
                    processed_val = processed_val.astype(dtype, order="C")

                    # Confirm that the array is c-contiguous and record
                    assert processed_val.flags["C_CONTIGUOUS"]
                    processed_vals[varname] = processed_val

                # Yield the processed values for this line
                yield processed_vals


class SampleResults(InferenceRes):
    """Comprehensive analysis interface for HMC sampling results. This class should
    never be instantiated directly. Instead, use the `from_disk` method to load the
    appropriate results object from disk.

    This class extends MLEInferenceRes to provide specialized functionality
    for analyzing Hamiltonian Monte Carlo sampling results from Stan. It offers
    comprehensive diagnostic capabilities, interactive visualization tools,
    and efficient data management for large MCMC datasets.

    :param model: SciStanPy model used for sampling. Defaults to None.
    :type model: Optional[Model]
    :param results: CmdStanMCMC object or path to CSV files. Defaults to None.
    :type results: Optional[Union[str, list[str], os.PathLike, CmdStanMCMC]]
    :param data: Observed data dictionary. Defaults to None.
    :type data: Optional[dict[str, npt.NDArray]]
    :param precision: Numerical precision for arrays. Defaults to "single".
    :type precision: Literal["double", "single", "half"]
    :param inference_obj: Pre-existing InferenceData or NetCDF path. Defaults to None.
    :type inference_obj: Optional[Union[az.InferenceData, str]]
    :param mib_per_chunk: Memory limit per chunk in MiB. Defaults to None.
    :type mib_per_chunk: Optional[custom_types.Integer]
    :param use_dask: Whether to use Dask for computation. Defaults to False.
    :type use_dask: bool
    :param output_filename: Output NetCDF filename. Defaults to None.
    :type output_filename: Optional[Union[str, os.PathLike]]

    The class provides comprehensive functionality for:

    - MCMC convergence diagnostics and reporting
    - Sample quality assessment and visualization
    - Interactive analysis of problematic variables
    - Efficient handling of large datasets with Dask integration
    - Automated detection and reporting of sampling issues

    Key Diagnostic Features:

    - R-hat convergence assessment
    - Effective sample size evaluation
    - Energy-based diagnostics (E-BFMI)
    - Divergence detection and analysis
    - Tree depth saturation monitoring

    The class automatically handles NetCDF conversion for efficient storage
    and supports both in-memory and out-of-core computation depending on
    dataset size and available memory.

    Example:
        .. code-block:: python

            import scistanpy as ssp

            # Get MCMC results
            mcmc_results = model.mcmc(data=observed_data, chains=4, iter_sampling=2000)

            # Run full diagnostics
            diagnostics = mcmc_results.diagnose()

            # Posterior predictive check (interactive in notebook)
            mcmc_results.run_ppc()

            # Evaluate problematic samples (interactive in notebook)
            mcmc_results.plot_sample_failure_quantile_traces()

            # Evaluate problematic variables (interactive in notebook)
            mcmc_results.plot_variable_failure_quantile_traces()
    """

    RESULTS_TO_NETCDF_CONVERTER = CmdStanMCMCToNetCDFConverter

    def __init__(
        self,
        *,
        model: Union["Model", None] = None,
        results: str | list[str] | os.PathLike | CmdStanMCMC | None = None,
        data: Union[dict[str, npt.NDArray], None] = None,
        precision: Literal["double", "single", "half"] = "single",
        inference_obj: Union[az.InferenceData, str, None] = None,
        mib_per_chunk: Union[custom_types.Integer, None] = None,
        use_dask: bool = False,
        output_filename: Union[str, os.PathLike, None] = None,
    ):
        """Initializes the SampleResults object by setting up the necessary parameters."""
        # If no filename is provided, we create one based on the csv files
        if output_filename is None:
            csv_files = None
            if isinstance(results, CmdStanMCMC):
                csv_files = results.runset.csv_files
            elif isinstance(results, list[str]):
                csv_files = results
            if csv_files is not None:
                output_filename = os.path.commonprefix(csv_files).rstrip("_") + ".nc"

        # Run parent init
        super().__init__(
            model=model,
            results=results,
            data=data,
            precision=precision,
            inference_obj=inference_obj,
            mib_per_chunk=mib_per_chunk,
            use_dask=use_dask,
            output_filename=output_filename,
        )

    def calculate_diagnostics(self) -> xr.Dataset:
        """Shortcut to running
        :py:meth:`~scistanpy.model.results.mle.MLEInferenceRes.calculate_summaries`
        with ``kind="diagnostics"`` and no other arguments.

        :returns: Dataset containing diagnostic metrics
        :rtype: xr.Dataset

        The method is designed as a simple interface for users who only need
        diagnostic information without summary statistics.
        """
        return self.calculate_summaries(kind="diagnostics")

    def evaluate_sample_stats(
        self,
        max_tree_depth: custom_types.Integer | None = None,
        ebfmi_thresh: custom_types.Float = DEFAULT_EBFMI_THRESH,
    ) -> xr.Dataset:
        """Evaluate sample-level diagnostic statistics for MCMC quality assessment.

        :param max_tree_depth: Maximum tree depth threshold. Uses model default if None.
            Defaults to None.
        :type max_tree_depth: Optional[custom_types.Integer]
        :param ebfmi_thresh: E-BFMI threshold for energy diagnostics. Defaults to 0.2.
        :type ebfmi_thresh: custom_types.Float

        :returns: Dataset with boolean arrays indicating test failures
        :rtype: xr.Dataset

        This method evaluates sample-level diagnostic statistics to identify
        problematic samples in the MCMC chains. Tests are considered failures
        when samples exhibit the following characteristics:

        - **Tree Depth**: Sample reached maximum tree depth (saturation)
        - **E-BFMI**: Energy-based fraction of missing information below threshold
        - **Divergence**: Sample diverged during Hamiltonian dynamics

        The resulting boolean arrays have ``True`` values indicating failed samples
        and ``False`` values indicating successful samples. This information is
        stored in the 'sample_diagnostic_tests' group of the InferenceData object.

        Example:
            >>> sample_tests = results.evaluate_sample_stats(ebfmi_thresh=0.15)
            >>> n_diverged = sample_tests.diverged.sum().item()
            >>> print(f"Number of divergent samples: {n_diverged}")
        """

        # If not provided, extract the maximum tree depth from the attributes
        if max_tree_depth is None:
            max_tree_depth = self.inference_obj.attrs["max_depth"].item()

        # Run all tests and build a dataset
        sample_tests = xr.Dataset(
            {
                "low_ebfmi": self.inference_obj.sample_stats.energy__ < ebfmi_thresh,
                "max_tree_depth_reached": self.inference_obj.sample_stats.treedepth__
                == max_tree_depth,
                "diverged": self.inference_obj.sample_stats.divergent__ == 1,
            }
        )
        # pylint: enable=no-member

        # Add the new group to the ArviZ object
        self._update_group("sample_diagnostic_tests", sample_tests)

        return sample_tests

    def evaluate_variable_diagnostic_stats(
        self,
        r_hat_thresh: custom_types.Float = DEFAULT_RHAT_THRESH,
        ess_thresh=DEFAULT_ESS_THRESH,
    ) -> xr.Dataset:
        """Evaluate variable-level diagnostic statistics for convergence assessment.

        :param r_hat_thresh: R-hat threshold for convergence. Defaults to 1.01.
        :type r_hat_thresh: custom_types.Float
        :param ess_thresh: ESS threshold *per chain*. Defaults to 100.
        :type ess_thresh: custom_types.Integer

        :returns: Dataset with boolean arrays indicating variable-level test failures
        :rtype: xr.Dataset

        :raises ValueError: If variable_diagnostic_stats group doesn't exist
        :raises ValueError: If required metrics are missing

        This method evaluates variable-level diagnostic statistics to identify
        parameters that exhibit poor sampling behavior. Tests are considered
        failures when variables meet the following criteria:

        Failure Conditions:

        - **R-hat**: Split R-hat statistic >= threshold (poor convergence)
        - **ESS Bulk**: Bulk effective sample size / n_chains <= threshold per chain
        - **ESS Tail**: Tail effective sample size / n_chains <= threshold per chain

        Results are stored in the 'variable_diagnostic_tests' group with boolean
        arrays indicating which variables failed which tests.

        Example:
            >>> var_tests = results.evaluate_variable_diagnostic_stats(r_hat_thresh=1.02)
            >>> failed_convergence = var_tests.sel(metric='r_hat').sum()
            >>> print(f"Variables with poor convergence: {failed_convergence.sum().item()}")
        """

        # We need to check if the `variable_diagnostic_stats` group exists. If it doesn't,
        # we need to run `calculate_diagnostics` first.
        if not hasattr(self.inference_obj, "variable_diagnostic_stats"):
            raise ValueError(
                "The `variable_diagnostic_stats` group does not exist. Please run "
                "`calculate_diagnostics` first."
            )

        # All metrics should be present in the `variable_diagnostic_stats` group.
        # pylint: disable=no-member
        if missing_metrics := (
            {"r_hat", "ess_bulk", "ess_tail"}
            - set(self.inference_obj.variable_diagnostic_stats.metric.values.tolist())
        ):
            raise ValueError(
                "The following metrics are missing from the `variable_diagnostic_stats` "
                f"group: {missing_metrics}."
            )

        # Update the ess threshold based on the number of chains
        ess_thresh *= self.inference_obj.posterior.sizes["chain"]

        # Run all tests and build a dataset
        variable_tests = xr.concat(
            [
                self.inference_obj.variable_diagnostic_stats.sel(metric="r_hat")
                >= r_hat_thresh,
                self.inference_obj.variable_diagnostic_stats.sel(metric="ess_bulk")
                <= ess_thresh,
                self.inference_obj.variable_diagnostic_stats.sel(metric="ess_tail")
                <= ess_thresh,
            ],
            dim="metric",
        )
        # pylint: enable=no-member

        # Add the new group to the ArviZ object
        self._update_group("variable_diagnostic_tests", variable_tests)

        return variable_tests

    def identify_failed_diagnostics(self, silent: bool = False) -> tuple[
        "custom_types.StrippedTestRes",
        dict[str, "custom_types.StrippedTestRes"],
    ]:
        """Identify and report diagnostic test failures with comprehensive summary.

        :param silent: Whether to suppress printed output. Defaults to False.
        :type silent: bool

        :returns: Tuple of (sample_failures, variable_failures) dictionaries
        :rtype: tuple[custom_types.StrippedTestRes, dict[str, custom_types.StrippedTestRes]]

        This method analyzes the results of diagnostic tests and provides both
        programmatic access to failure information and human-readable summaries.
        It requires that diagnostic evaluation methods have been run previously.

        Return Structure:

        - **sample_failures**: Dictionary mapping test names to arrays of failed
          sample indices
        - **variable_failures**: Dictionary mapping metric names to dictionaries
          of failed variables

        The method processes test results to extract:

        - Indices of samples that failed each diagnostic test
        - Names of variables that failed each diagnostic metric
        - Summary statistics showing failure rates and percentages

        When not silent, provides detailed reporting including:

        - Failure counts and percentages for each test type
        - Variable-specific failure information organized by metric
        - Clear categorization of sample vs. variable-level issues
        """

        def process_test_results(
            test_res_dataarray: xr.Dataset,
        ) -> "custom_types.ProcessedTestRes":
            """
            Process the test results from a DataArray into a dictionary of test results.

            Parameters
            ----------
            test_res_dataarray : xr.Dataset
                The DataArray containing the test results.

            Returns
            -------
            custom_types.ProcessedTestRes
                A dictionary where the keys are the variable names and the values are tuples
                containing the indices of the failed tests and the total number of tests.
            """
            return {
                varname: (np.atleast_1d(tests.values).nonzero(), tests.values.size)
                for varname, tests in test_res_dataarray.items()
            }

        def strip_totals(
            processed_test_results: "custom_types.ProcessedTestRes",
        ) -> "custom_types.StrippedTestRes":
            """
            Strip the totals from the test results.

            Parameters
            ----------
            processed_test_results : custom_types.ProcessedTestRes
                The processed test results from the `process_test_results` function.

            Returns
            -------
            custom_types.StrippedTestRes
                The processed test results with the totals stripped.
            """
            return {k: v[0] for k, v in processed_test_results.items()}

        def report_test_summary(
            processed_test_results: "custom_types.ProcessedTestRes",
            type_: str,
            prepend_newline: bool = True,
        ) -> None:
            """
            Report the summary of the test results.

            Parameters
            ----------
            processed_test_results : dict[str, tuple[tuple[npt.NDArray, ...], int]]
                The processed test results from the `process_test_results` function.
            type_ : str
                The type of test results (e.g., "sample", "variable").
            prepend_newline : bool, optional
                Whether to prepend a newline before the summary, by default True.
            """
            if prepend_newline:
                print()
            header = f"{type_.capitalize()} diagnostic tests results' summaries:"
            print(header)
            print("-" * len(header))
            for varname, (
                failed_indices,
                total_tests,
            ) in processed_test_results.items():
                n_failures = len(failed_indices[0])
                assert all(n_failures == len(failed) for failed in failed_indices[1:])
                print(
                    f"{n_failures} of {total_tests} ({n_failures / total_tests:.2%}) {type_}s "
                    f"{message_map.get(varname, f'tests failed for {varname}')}."
                )

        # Different messages for different test types
        message_map = {
            "low_ebfmi": "had a low energy",
            "max_tree_depth_reached": "reached the maximum tree depth",
            "diverged": "diverged",
        }

        # Get the indices of the sampling tests that failed and the total number of tests
        # performed
        # pylint: disable=no-member
        sample_test_failures = process_test_results(
            self.inference_obj.sample_diagnostic_tests
        )

        # Get the indices of the variable diagnostic tests that failed and the total number
        # of tests performed
        variable_test_failures = {
            metric.item(): process_test_results(
                self.inference_obj.variable_diagnostic_tests.sel(metric=metric.item())
            )
            for metric in self.inference_obj.variable_diagnostic_tests.metric
        }
        # pylint: enable=no-member

        # Strip the totals from the test results and package as return values
        res = (
            strip_totals(sample_test_failures),
            {
                metric: strip_totals(failures)
                for metric, failures in variable_test_failures.items()
            },
        )

        # If silent, return the test results now
        if silent:
            return res

        # Report sample test failures
        report_test_summary(sample_test_failures, "sample", prepend_newline=False)

        # Report variable test failures
        for metric, failures in variable_test_failures.items():
            report_test_summary(failures, metric)

        return res

    def diagnose(
        self,
        max_tree_depth: custom_types.Integer | None = None,
        ebfmi_thresh: custom_types.Float = DEFAULT_EBFMI_THRESH,
        r_hat_thresh: custom_types.Float = DEFAULT_RHAT_THRESH,
        ess_thresh: custom_types.Float = DEFAULT_ESS_THRESH,
        silent: bool = False,
    ) -> tuple[
        "custom_types.StrippedTestRes", dict[str, "custom_types.StrippedTestRes"]
    ]:
        """Runs the complete MCMC diagnostic pipeline. This involves running, in order:

        1. :py:meth:`~scistanpy.model.results.hmc.SampleResults.calculate_diagnostics`
        2. :py:meth:`~scistanpy.model.results.hmc.SampleResults.evaluate_sample_stats`
        3. :py:meth:`~scistanpy.model.results.hmc.SampleResults.evaluate_variable_diagnostic_stats`
        4. :py:meth:`~scistanpy.model.results.hmc.SampleResults.identify_failed_diagnostics`

        Typically, users will want to use this method rather than calling the individual
        methods themselves.

        :param max_tree_depth: Maximum tree depth threshold. Uses model default if None.
            Defaults to None.
        :type max_tree_depth: Optional[custom_types.Integer]
        :param ebfmi_thresh: E-BFMI threshold for energy diagnostics. Defaults to 0.2.
        :type ebfmi_thresh: custom_types.Float
        :param r_hat_thresh: R-hat threshold for convergence assessment. Defaults to 1.01.
        :type r_hat_thresh: custom_types.Float
        :param ess_thresh: ESS threshold per chain. Defaults to 100.
        :type ess_thresh: custom_types.Float
        :param silent: Whether to suppress diagnostic output. Defaults to False.
        :type silent: bool

        :returns: Tuple of (sample_failures, variable_failures) as returned by
            identify_failed_diagnostics
        :rtype: tuple[custom_types.StrippedTestRes, dict[str, custom_types.StrippedTestRes]]

        The method provides comprehensive assessment of MCMC sampling quality,
        identifying both immediate issues (e.g., divergences, energy problems) and
        convergence concerns (e.g., R-hat, effective sample size).

        All intermediate results are stored in the ``inference_obj`` attribute
        for later access and further analysis.
        """
        # Run the diagnostics
        self.calculate_diagnostics()
        self.evaluate_sample_stats(
            max_tree_depth=max_tree_depth, ebfmi_thresh=ebfmi_thresh
        )
        self.evaluate_variable_diagnostic_stats(
            r_hat_thresh=r_hat_thresh, ess_thresh=ess_thresh
        )

        # Identify the failed diagnostics
        return self.identify_failed_diagnostics(silent=silent)

    @overload
    def plot_sample_failure_quantile_traces(
        self,
        display: Literal[True],
        width: custom_types.Integer,
        height: custom_types.Integer,
    ) -> hv.HoloMap: ...

    @overload
    def plot_sample_failure_quantile_traces(
        self,
        display: Literal[False],
        width: custom_types.Integer,
        height: custom_types.Integer,
    ) -> dict[str, hv.Overlay]: ...

    def plot_sample_failure_quantile_traces(
        self, *, display=True, width=600, height=600
    ):
        """Visualize quantile traces for samples that failed diagnostic tests.

        :param display: Whether to return formatted layout for display. Defaults to True.
        :type display: bool
        :param width: Width of plots in pixels. Defaults to 600.
        :type width: custom_types.Integer
        :param height: Height of plots in pixels. Defaults to 600.
        :type height: custom_types.Integer

        :returns: Quantile trace plots in requested format
        :rtype: Union[hv.HoloMap, dict[str, hv.Overlay]]

        :raises ValueError: If no samples failed diagnostic tests

        This method creates specialized trace plots showing how samples that
        failed diagnostic tests compare to those that passed. The visualization
        helps identify systematic patterns in sampling failures.

        Plot Structure:

        - **X-axis**: Cumulative fraction of parameters (0 to 1, sorted by typical
          quantile of failed samples)
        - **Y-axis**: Quantiles of failed samples relative to passing samples
        - **Individual traces**: Semi-transparent lines for each failed sample
        - **Typical trace**: Bold line showing median behavior across failures
        - **Reference line**: Diagonal indicating perfect calibration

        The plots reveal:

        - Whether failures are systematic across parameters
        - Patterns in how failed samples deviate from typical behavior
        - The severity and consistency of sampling problems

        Example:
            >>> # Display interactive traces
            >>> results.plot_sample_failure_quantile_traces()
        """

        # x-axis labels are meaningless, so we will use a hook to hide them
        def hook(plot, element):  # pylint: disable=unused-argument
            plot.state.xaxis.major_tick_line_color = None
            plot.state.xaxis.minor_tick_line_color = None
            plot.state.xaxis.major_label_text_font_size = "0pt"

        # If there are no failed samples, raise an error
        # pylint: disable=no-member
        if not any(
            self.inference_obj.sample_diagnostic_tests.apply(lambda x: x.any()).values()
        ):
            raise ValueError(
                "No samples failed the diagnostic tests. This error is a good thing!"
            )

        # First, we need to get all samples and the diagnostic tests. We will reshape
        # both to be 2D, with the first dimension being the samples and the second
        # dimension being the parameter values or the test results, respectively.
        sample_arr = (
            self.inference_obj.posterior.to_stacked_array(
                new_dim="vals", sample_dims=["chain", "draw"], variable_dim="parameter"
            )
            .stack(samples=("chain", "draw"))
            .T
        )
        sample_test_arr = self.inference_obj.sample_diagnostic_tests.apply(
            lambda x: x.stack(samples=("chain", "draw"))
        )
        # pylint: enable=no-member

        # Now we need some metadata about the samples, such as the chain and draw
        # indices and the parameter names
        varnames = np.array(
            [
                ".".join([str(i) for i in name_tuple if isinstance(i, (str, int))])
                for name_tuple in sample_arr.vals.to_numpy()
            ]
        )
        chains = sample_arr.coords["chain"].to_numpy()
        draws = sample_arr.coords["draw"].to_numpy()

        # We will need the sample array as a numpy array from here
        sample_arr = sample_arr.to_numpy()

        # x-values are just incrementally increasing from 0 to 1 for the number of
        # parameters
        x = np.linspace(0, 1, sample_arr.shape[1])

        # Now we process each of the diagnostic tests
        plots = {}
        for testname, testmask in sample_test_arr.items():

            # Get the failed samples. If there are no failed samples, skip this
            # test
            testmask = testmask.to_numpy()
            if not testmask.any():
                continue
            failed_samples, failed_chains, failed_draws, passed_samples = (
                sample_arr[testmask],
                chains[testmask],
                draws[testmask],
                sample_arr[~testmask],
            )

            # Get the quantiles of the  failed samples relative to the passed ones
            failed_quantiles = plotting.calculate_relative_quantiles(
                passed_samples, failed_samples
            )

            # Get the typical quantiles of the failed samples
            typical_failed_quantiles = np.median(failed_quantiles, axis=0)

            # Sort samples by the values of the typical failed samples
            sorted_inds = np.argsort(typical_failed_quantiles)
            (
                failed_samples,
                failed_quantiles,
                typical_failed_quantiles,
                resorted_varnames,
            ) = (
                failed_samples[:, sorted_inds],
                failed_quantiles[:, sorted_inds],
                typical_failed_quantiles[sorted_inds],
                varnames[sorted_inds],
            )

            # Build the traces
            plots[testname] = hv.Overlay(
                [
                    hv.Curve(
                        (
                            x,
                            quantile,
                            resorted_varnames,
                            failed_chains[i],
                            failed_draws[i],
                            failed_samples[i],
                        ),
                        kdims=["Fraction of Parameters"],
                        vdims=["Quantile", "Parameter", "Chain", "Draw", "Value"],
                    ).opts(line_color="blue", line_alpha=0.1, tools=["hover"])
                    for i, quantile in enumerate(failed_quantiles)
                ]
                + [
                    hv.Curve(
                        (x, typical_failed_quantiles),
                        kdims=["Fraction of Parameters"],
                        vdims=["Quantile"],
                        label="Typical Failed Quantiles",
                    ).opts(line_color="black", line_width=1),
                    hv.Curve(
                        ((0, 1), (0, 1)),
                        kdims=["Fraction of Parameters"],
                        vdims=["Quantile"],
                        label="Idealized Quantiles",
                    ).opts(line_color="black", line_width=1, line_dash="dashed"),
                ]
            ).opts(
                hooks=[hook],
                title=f"Quantiles of Samples Failing: {testname}",
                width=width,
                height=height,
            )

        # If requested, display the plots
        if display:
            return hv.Layout(plots.values()).cols(1).opts(shared_axes=False)

        return plots

    @overload
    def plot_variable_failure_quantile_traces(
        self,
        *,
        display: Literal[True],
        width: custom_types.Integer,
        height: custom_types.Integer,
        plot_quantiles: bool,
    ) -> VariableAnalyzer: ...

    @overload
    def plot_variable_failure_quantile_traces(
        self,
        *,
        display: Literal[False],
        width: custom_types.Integer,
        height: custom_types.Integer,
        plot_quantiles: bool,
    ) -> pn.pane.HoloViews: ...

    def plot_variable_failure_quantile_traces(
        self,
        display=True,
        width=800,
        height=400,
        plot_quantiles=False,
    ):
        """Create interactive analyzer for variables that failed diagnostic tests.

        :param display: Whether to return display-ready analyzer. Defaults to True.
        :type display: bool
        :param width: Width of plots in pixels. Defaults to 800.
        :type width: custom_types.Integer
        :param height: Height of plots in pixels. Defaults to 400.
        :type height: custom_types.Integer
        :param plot_quantiles: Whether to plot quantiles vs raw values. Defaults to False.
        :type plot_quantiles: bool

        :returns: Interactive analyzer or Panel layout
        :rtype: Union[VariableAnalyzer, pn.pane.HoloViews]

        This method creates an interactive analysis tool for examining individual
        variables that failed diagnostic tests. The analyzer provides widgets for
        selecting specific variables, diagnostic metrics, and array indices.

        Interactive Features:

        - **Variable Selection**: Choose from variables that failed any test
        - **Metric Selection**: Focus on specific diagnostic failures
        - **Index Selection**: Examine individual array elements for multi-dimensional parameters

        The resulting trace plots show:

        - Sample trajectories across MCMC chains with distinct colors
        - Quantile analysis relative to parameters that passed tests
        - Hover information with detailed sample metadata
        - Chain-specific behavior identification

        This tool is particularly valuable for:

        - Understanding the nature of convergence problems
        - Identifying problematic parameter regions
        - Diagnosing systematic vs. sporadic sampling issues
        - Planning model reparameterization strategies

        Example:
            >>> # Interactive analysis in notebook
            >>> analyzer = results.plot_variable_failure_quantile_traces()
            >>> analyzer  # Display widget interface
        """
        # Build the analyzer object
        analyzer = VariableAnalyzer(
            self, plot_width=width, plot_height=height, plot_quantiles=plot_quantiles
        )

        # Return the analyzer object if not displaying
        if not display:
            return analyzer

        # Otherwise, display the plots
        return analyzer.display()

    @classmethod
    def from_disk(  # pylint: disable=arguments-renamed
        cls,
        path: str,
        csv_files: list[str] | str | None = None,
        skip_fit: bool = False,
        use_dask: bool = False,
    ) -> "SampleResults":
        """Load SampleResults from saved NetCDF file with optional CSV metadata.

        :param path: Path to NetCDF file containing inference data
        :type path: str
        :param csv_files: Paths to CSV files output by Stan. Can also be a glob
            pattern in place of a list. Defaults to None (auto-detect based on
            ``path`` value).
        :type csv_files: Optional[Union[list[str], str]]
        :param skip_fit: Whether to skip loading CSV metadata. Defaults to False.
        :type skip_fit: bool
        :param use_dask: Whether to enable Dask for computation. Defaults to False.
        :type use_dask: bool

        :returns: Loaded SampleResults object ready for analysis
        :rtype: SampleResults

        :raises FileNotFoundError: If the specified NetCDF file doesn't exist

        This class method enables loading of previously saved MCMC results from
        NetCDF format, with optional access to original CSV metadata for complete
        functionality.

        Loading Modes:

        - **Full loading**: NetCDF + CSV metadata (complete functionality)
        - **NetCDF only**: Fast loading without CSV metadata (limited functionality)
        - **Auto-detection**: Automatically finds CSV files based on NetCDF path

        When use_dask=True, the loaded data supports out-of-core computation
        for memory-efficient analysis of large datasets. Management of Dask happens
        internally, so users do not need to be familiar with Dask to take advantage
        of it.

        Example:
            >>> # Load with auto-detected CSV files (csvs must have same basename)
            >>> results = SampleResults.from_disk('model_results.nc')
            >>>
            >>> # Load with explicit CSV files
            >>> results = SampleResults.from_disk(
            ...     'results.nc', csv_files=['chain_1.csv', 'chain_2.csv']
            ... )
            >>>
            >>> # Fast loading without CSV metadata
            >>> results = SampleResults.from_disk('results.nc', skip_fit=True)
        """
        # The path to the netcdf file must exist
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"The file {path} does not exist. Please provide a valid path."
            )

        # If csv files are not provided and we are not skipping the fit, we need
        # to get them from the netcdf file
        if skip_fit:
            csv_files = None
        elif csv_files is None:
            # If the path to the netcdf file does not end with ".nc", raise a warning
            # that we cannot automatically find the csv files. If it does, find
            # the csv files
            if path.endswith(".nc"):
                csv_files = list(glob(path.removesuffix(".nc") + "*.csv"))
            else:
                warnings.warn(
                    "Could not identify csv files automatically. Loading without."
                    "To be auto-detected, csv files must be named according to the"
                    "following pattern: <extensionless_netcdf_filename>*.csv"
                )

        # Initialize the object
        return cls(
            model=None,
            results=None if csv_files is None else fit_from_csv_noload(csv_files),
            inference_obj=path,
            use_dask=use_dask,
        )


def identify_csv_files(
    path: str | list[str] | os.PathLike, allow_missing: bool = False
) -> list[str]:
    """Identifies CSV files from the given path."""
    csvfiles = []
    if isinstance(path, list):
        csvfiles = path
    elif isinstance(path, str) and "*" in path:
        splits = os.path.split(path)
        if splits[0] is not None:
            if not (os.path.exists(splits[0]) and os.path.isdir(splits[0])):
                raise ValueError(
                    f"Invalid path specification, {path} unknown directory: {splits[0]}"
                )
        csvfiles = glob(path)
    elif isinstance(path, (str, os.PathLike)):
        if os.path.exists(path) and os.path.isdir(path):
            for file in os.listdir(path):
                if os.path.splitext(file)[1] == ".csv":
                    csvfiles.append(os.path.join(path, file))
        elif os.path.exists(path):
            csvfiles.append(str(path))
        else:
            raise ValueError(f"Invalid path specification: {path}")
    else:
        raise ValueError(f"Invalid path specification: {path}")

    if len(csvfiles) == 0:
        if allow_missing:
            return []
        raise ValueError(f"No CSV files found in directory {path}")
    for file in csvfiles:
        if not (os.path.exists(file) and os.path.splitext(file)[1] == ".csv"):
            raise ValueError(f"Bad CSV file path spec, includes non-csv file: {file}")

    return csvfiles


def fit_from_csv_noload(path: str | list[str] | os.PathLike) -> CmdStanMCMC:
    """Create CmdStanMCMC object from CSV files without loading data into memory.
    This function is adapted from ``cmdstanpy.from_csv``.

    :param path: Path specification for CSV files (single file, list, or glob pattern)
    :type path: Union[str, list[str], os.PathLike]

    :returns: CmdStanMCMC object with metadata but no loaded sample data
    :rtype: CmdStanMCMC

    :raises ValueError: If path specification is invalid or no CSV files found
    :raises ValueError: If CSV files are not valid Stan output

    This function provides a memory-efficient way to create CmdStanMCMC objects
    by parsing only the metadata from CSV files without loading the actual
    sample data. This is particularly useful for large datasets where memory
    usage is a concern.

    Path Specifications:

    - **Single file**: Direct path to one CSV file
    - **File list**: List of paths to multiple CSV files
    - **Glob pattern**: Wildcard pattern for automatic file discovery
    - **Directory**: Directory containing CSV files (loads all .csv files)

    The function performs validation to ensure:

    - All specified files exist and are readable
    - Files contain valid Stan CSV output
    - Sampling method is compatible (only 'sample' method supported)
    - Configuration is consistent across files

    This approach enables efficient processing workflows where sample data
    is converted to more efficient formats (like NetCDF) without requiring
    full memory loading of the original CSV files.

    Example:
        >>> # Load from glob pattern
        >>> fit = fit_from_csv_noload('model_output_*.csv')
        >>>
        >>> # Load from explicit list
        >>> fit = fit_from_csv_noload(['chain1.csv', 'chain2.csv'])
    """

    def get_config_dict() -> dict[str, Any]:
        """Reads the first CSV file and returns the configuration dictionary."""
        config_dict: dict[str, Any] = {}
        try:
            with open(csvfiles[0], "r", encoding="utf-8") as fd:
                scan_config(fd, config_dict, 0)
        except (IOError, OSError, PermissionError) as e:
            raise ValueError(f"Cannot read CSV file: {csvfiles[0]}") from e
        if "model" not in config_dict or "method" not in config_dict:
            raise ValueError(f"File {csvfiles[0]} is not a Stan CSV file.")
        if config_dict["method"] != "sample":
            raise ValueError(
                "Expecting Stan CSV output files from method sample, "
                f" found outputs from method {config_dict['method']}"
            )

        return config_dict

    def build_sampler_args() -> SamplerArgs:
        """Builds the sampler arguments"""
        sampler_args = SamplerArgs(
            iter_sampling=config_dict["num_samples"],
            iter_warmup=config_dict["num_warmup"],
            thin=config_dict["thin"],
            save_warmup=config_dict["save_warmup"],
        )
        # bugfix 425, check for fixed_params output
        try:
            check_sampler_csv(
                csvfiles[0],
                iter_sampling=config_dict["num_samples"],
                iter_warmup=config_dict["num_warmup"],
                thin=config_dict["thin"],
                save_warmup=config_dict["save_warmup"],
            )
        except ValueError:
            try:
                check_sampler_csv(
                    csvfiles[0],
                    is_fixed_param=True,
                    iter_sampling=config_dict["num_samples"],
                    iter_warmup=config_dict["num_warmup"],
                    thin=config_dict["thin"],
                    save_warmup=config_dict["save_warmup"],
                )
                sampler_args = SamplerArgs(
                    iter_sampling=config_dict["num_samples"],
                    iter_warmup=config_dict["num_warmup"],
                    thin=config_dict["thin"],
                    save_warmup=config_dict["save_warmup"],
                    fixed_param=True,
                )
            except ValueError as e:
                raise ValueError("Invalid or corrupt Stan CSV output file, ") from e

        return sampler_args

    def build_fit() -> CmdStanMCMC:
        """Builds the CmdStanMCMC object"""
        chains = len(csvfiles)
        cmdstan_args = CmdStanArgs(
            model_name=config_dict["model"],
            model_exe=config_dict["model"],
            chain_ids=[x + 1 for x in range(chains)],
            method_args=sampler_args,
        )
        runset = RunSet(args=cmdstan_args, chains=chains)
        # pylint: disable=protected-access
        runset._csv_files = csvfiles
        for i in range(len(runset._retcodes)):
            runset._set_retcode(i, 0)
        # pylint: enable=protected-access
        fit = CmdStanMCMC(runset)
        return fit

    # Run the functions to parse the CSV files
    csvfiles = identify_csv_files(path)
    config_dict = get_config_dict()
    sampler_args = build_sampler_args()
    return build_fit()


def get_n_samples(csv_file: str) -> int:
    """
    Counts the number of samples in a csv file output by Stan. Specifically, this
    returns the number of rows in a csv file that do not start with a '#' minus
    one (to account for the header row).

    :param csv_file: Path to the csv file.
    :type csv_file: str

    :returns: Number of samples in the csv file
    :rtype: int
    """
    with open(csv_file, "r", encoding="utf-8") as f:
        return sum(1 for row in f if not row.startswith("#")) - 1


def _check_outdir_exists[T, **P](f: Callable[P, T]) -> Callable[P, T]:
    """
    Function to check if a directory exists. If it does, returns the result of the
    decorated function. If it does not and we are allowing missing, returns None.
    If it does not and we are not allowing missing, raises an error.
    """

    @functools.wraps(f)
    def inner(*args: P.args, **kwargs: P.kwargs) -> T | None:

        # The first argument should be the output directory
        paramspec = inspect.signature(f).parameters
        if len(paramspec) == 0:
            raise ValueError(
                "The decorated function must have at least one argument for the output directory."
            )
        if next(iter(paramspec)) != "outdir":
            raise ValueError(
                "The first argument of the decorated function must be `outdir`."
            )

        # Combine args and kwargs
        combined = {**dict(zip(paramspec.keys(), args)), **kwargs}

        # `allow_missing` should be a keyword argument
        if "allow_missing" not in paramspec:
            raise ValueError(
                "The decorated function must have an `allow_missing` keyword argument."
            )

        # If the directory exists, we can run the function
        if os.path.exists(combined["outdir"]):
            return f(**combined)

        # If the directory does not exist and we are allowing missing, return `None`
        if combined["allow_missing"]:
            return None

        # If the directory does not exist and we are not allowing missing, raise
        # an error
        raise ValueError(f"Output directory {combined['outdir']} does not exist.")

    return inner


@_check_outdir_exists
def sampling_complete(
    outdir: str,
    expected_samples: int,
    expected_chains: int,
    allow_missing: bool = False,  # pylint: disable=unused-argument
) -> bool:
    """
    Determines if HMC sampling is complete. Sampling is complete if we have the
    expected number of csv files and each csv file has the expected number of samples.

    :param outdir: Path to the directory containing the csv files.
    :type outdir: str
    :param expected_samples: The expected number of samples in each csv file.
    :type expected_samples: int
    :param expected_chains: The expected number of csv files (chains).
    :type expected_chains: int
    :param allow_missing: Whether to allow a missing `outdir` directory. If True,
        a missing `outdir` will cause the function to return False rather than raise
        an error. Defaults to False.
    :type allow_missing: bool

    :returns: True if sampling is complete, False otherwise.
    :rtype: bool

    :raises ValueError: If we find more csv files than expected or if any csv file
        has more samples than expected.
    """

    def isolate_timestamps() -> tuple[dict[int, int], dict[int, list[str]], list[int]]:
        """
        Counts the number of unique timestamps in the csv file names.

        :returns: A tuple of (timestamp_counts, timestamp_to_files, unique_timestamps),
            where timestamp_counts is a dictionary mapping timestamps to counts,
            timestamp_to_files is a dictionary mapping timestamps to lists of csv
            files, and unique_timestamps is a list of unique timestamps sorted from
            earliest to latest.
        :rtype: tuple[dict[int, int], dict[int, list[str]], list[int]]
        """
        # Build the regex
        csv_timestamp_parser = re.compile(r".+-([0-9]{14})_([0-9]+)\.csv")

        # Identify timestamps and counts for each csv file
        timestamp_counts: dict[int, int] = {}
        timestamp_to_files: dict[int, list[str]] = {}
        for csv_file in csv_files:
            if match := csv_timestamp_parser.match(os.path.basename(csv_file)):
                timestamp = int(match.group(1))
                if timestamp not in timestamp_counts:
                    timestamp_counts[timestamp] = 0
                    timestamp_to_files[timestamp] = []
                timestamp_counts[timestamp] += 1
                timestamp_to_files[timestamp].append(csv_file)
            else:
                warnings.warn(f"Could not extract timestamp from csv file: {csv_file}.")

        # Get the unique timestamps sorted from earliest to latest
        unique_timestamps = sorted(timestamp_counts.keys())

        return timestamp_counts, timestamp_to_files, unique_timestamps

    # Identify csv files
    csv_files = identify_csv_files(outdir, allow_missing=True)

    # If we get more csv files than expected, warn the user and take the latest
    # files. If we get fewer, sampling is not complete.
    if (n_files := len(csv_files)) > expected_chains:

        # Isolate timestamps from the csv file names
        timestamp_counts, timestamp_to_files, unique_timestamps = isolate_timestamps()

        # The latest timestamp should correspond to the current sampling run, so
        # we isolate the csv files with that timestamp. If there are not exactly
        # the number of expected chains with that timestamp, we have an error.
        if len(unique_timestamps) == 0:
            raise ValueError(
                f"Found more CSV files ({n_files}) than expected ({expected_chains}), "
                "but could not isolate files from the current sampling run based on "
                "timestamps in the file names."
            )
        if timestamp_counts[unique_timestamps[-1]] != expected_chains:
            raise ValueError(
                f"Found more CSV files ({n_files}) than expected ({expected_chains}), "
                "and while we were able to isolate the files from the current sampling "
                "run based on timestamps in the file names, we found a different number "
                f"of files with the latest timestamp ({timestamp_counts[unique_timestamps[-1]]}) "
                f"than expected ({expected_chains})."
            )
        csv_files = timestamp_to_files[unique_timestamps[-1]]
        warnings.warn(
            f"Found more CSV files ({n_files}) than expected ({expected_chains}). "
            f"Isolated {len(csv_files)} files with the latest timestamp ({unique_timestamps[-1]}). "
        )
    elif n_files < expected_chains:
        return False

    # If any csv file has less samples than expected, sampling is not complete.
    # If any csv file has more samples than expected, we have an error
    over, under = {}, {}
    for csv_file in csv_files:
        n_samples = get_n_samples(csv_file)
        if n_samples > expected_samples:
            over[csv_file] = n_samples
        elif n_samples < expected_samples:
            under[csv_file] = n_samples
    if over:
        raise ValueError(
            f"Found more samples than expected ({expected_samples}) in the following "
            f"files: {over}"
        )
    elif under:
        return False

    # If we get here, sampling is complete
    return True


def isolate_sample_res(
    outdir: str, diagnosed: bool = False
) -> Union[SampleResults, None]:
    """
    Returns the SampleResults found at `outdir` corresponding to either the diagnosed
    or non-diagnosed NetCDF file in the given directory, if it exists. If there
    is none or the file is corrupted, returns None. If there is more than one,
    raises a ValueError.
    """
    # Get netcdf files
    filter_func = (
        (lambda f: f.endswith("diagnosed.nc"))
        if diagnosed
        else (lambda f: not f.endswith("diagnosed.nc"))
    )
    sample_file = [f for f in glob(os.path.join(outdir, "*.nc")) if filter_func(f)]

    # If we have more than 1, we have an error. If we have 0, we do not have NetCDF
    # files.
    if (n_files := len(sample_file)) > 1:
        raise ValueError(
            f"Found more {'diagnosed' if diagnosed else 'non-diagnosed'} NetCDF "
            f"files ({n_files}) than expected (1) in {outdir}."
        )
    elif n_files == 0:
        return None

    # If we have 1, but we cannot load it, we need to rebuild the file, so we also
    # return None in that case. If we can load it, return the results.
    try:
        return SampleResults.from_disk(sample_file[0], skip_fit=True)
    except (KeyError, OSError, AttributeError):
        return None


@_check_outdir_exists
def load_converted(
    outdir: str,
    expected_samples: int,
    expected_chains: int,
    allow_missing: bool = False,  # pylint: disable=unused-argument
    diagnosed: bool = False,
) -> Union[SampleResults, None]:
    """
    Determines if the csv-to-nc conversion is complete. The conversion is complete
    if we have exactly one non-diagnosed NetCDF file and that file has the expected
    number of samples and chains.

    :param outdir: Path to the directory containing the NetCDF files.
    :type outdir: str
    :param expected_samples: The expected number of samples in the NetCDF file.
    :type expected_samples: int
    :param expected_chains: The expected number of chains in the NetCDF file.
    :type expected_chains: int
    :param allow_missing: Whether to allow a missing `outdir` directory. If True,
        a missing `outdir` will cause the function to return False rather than raise
        an error. Defaults to False.
    :type allow_missing: bool
    :param diagnosed: Whether to look for the diagnosed NetCDF file rather than the
        non-diagnosed one. Defaults to False.
    :type diagnosed: bool

    :returns: The SampleResults object if the conversion is complete, None otherwise.
    :rtype: Union[SampleResults, None]

    :raises ValueError: If we find more than one non-diagnosed NetCDF file or if the
        NetCDF file has more samples or chains than expected.
    """
    # Do we have NetCDF files? If not, conversion is not complete.
    if (res := isolate_sample_res(outdir, diagnosed=diagnosed)) is None:
        return None

    # Do we have the expected number of samples and chains? If not, this is an error.
    if (
        found_chains := res.inference_obj.sample_stats.sizes["chain"]
    ) != expected_chains:
        raise ValueError(
            f"Found more chains ({found_chains}) than expected ({expected_chains}) "
            "in converted NetCDF."
        )
    if (
        found_samples := res.inference_obj.sample_stats.sizes["draw"]
    ) != expected_samples:
        raise ValueError(
            f"Found more samples ({found_samples}) than expected ({expected_samples}) "
            "in converted NetCDF."
        )

    # If we get here, the conversion is complete
    return res
