"""Handles results from a `dms_stan.model.stan.StanModel` object."""

import itertools
import os.path
import re
import warnings

from glob import glob
from typing import Any, Generator, Literal, Optional, overload, Sequence, Union

import arviz as az
import dask
import holoviews as hv
import h5netcdf
import numpy as np
import numpy.typing as npt
import panel as pn
import xarray as xr

from cmdstanpy.cmdstan_args import CmdStanArgs, SamplerArgs
from cmdstanpy.stanfit import CmdStanMCMC, RunSet
from cmdstanpy.utils import check_sampler_csv, scan_config
from tqdm import tqdm

import dms_stan.model  # pylint: disable=unused-import

from dms_stan.custom_types import ProcessedTestRes, StrippedTestRes
from dms_stan.defaults import (
    DEFAULT_EBFMI_THRESH,
    DEFAULT_ESS_THRESH,
    DEFAULT_RHAT_THRESH,
)
from dms_stan.model.components import (
    DiscreteDistribution,
    Parameter,
    TransformedParameter,
)
from dms_stan.model.pytorch.mle import MLEInferenceRes
from dms_stan.plotting import calculate_relative_quantiles
from dms_stan.utils import az_dask, get_chunk_shape

# pylint: disable=too-many-lines

# Maps between the precision of the data and the numpy types
_NP_TYPE_MAP = {
    "double": {"float": np.float64, "int": np.int64},
    "single": {"float": np.float32, "int": np.int32},
    "half": {"float": np.float16, "int": np.int16},
}


def _symmetrize_quantiles(quantiles: Sequence[float]) -> list[float]:
    """
    Symmetrizes a list of quantiles by adding the complementary quantiles and the
    median. For example, if the input is [0.1, 0.2], the output will be
    [0.1, 0.2, 0.5, 0.8, 0.9].
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
    """
    Used for analysis of the variables that fail diagnostic tests during sampling.
    This should never be instantiated directly. Instead, use the `SampleResults`
    class to evaluate the traceplots for the variables that failed the tests.
    """

    # pylint: disable=attribute-defined-outside-init

    def __init__(
        self,
        sample_results: "SampleResults",
        plot_width: int = 800,
        plot_height: int = 400,
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
        """Identify the variables that failed the diagnostic tests"""

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
        """Updates the metric options based on the selected variable"""

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
        """Updates the index options based on the selected variable"""

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
        """Gets the data for the currently selected variable"""
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
        """Gets the data for the currently selected metric"""
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
        """Updates the panel plot based on the selected variable, metric, and index"""
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
            failing_quantiles = calculate_relative_quantiles(
                reference=self._passing_samples,
                observed=failing_samples[None],
            )[0]

        # We should have shape (n_chains, n_draws)
        assert (
            failing_samples.shape
            == failing_quantiles.shape
            == (self.n_chains, self.x.size)
        ), (
            failing_samples.shape,
            failing_quantiles.shape,
            self.n_chains,
            self.x.size,
            self._failing_samples.shape,
            self.indexchoice.value,
            self._np_samples.shape,
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
        """Display the panel layout"""
        return self.layout


class CmdStanMCMCToNetCDFConverter:
    """
    This class is used within `cmdstanmcmc_to_hdf5` to convert a set of cmdstan
    csv files to a single hdf5 file.
    """

    def __init__(
        self,
        fit: CmdStanMCMC | str | list[str] | os.PathLike,
        model: "dms_stan.model.Model",
        data: dict[str, Any] | None = None,
    ):
        """
        Initialization involves collecting information about the different variables
        in the fit object. This includes the names of the variables, their shapes,
        and their types. This information is used to create the HDF5 file.
        """
        # If `fit` is a string, we assume we need to load it from disk
        if isinstance(fit, str):
            fit = fit_from_csv_noload(fit)

        # The fit and model are stored as attributes
        self.fit = fit
        self.model = model
        self.data = data

        # Record the config object
        self.config = fit.metadata.cmdstan_config

        # The number of chains is per thread. We want the number of chains total
        self.config["total_chains"] = len(fit.runset.csv_files)

        # How many samples are we expecting?
        self.num_draws = self.config["num_samples"] + (
            self.config["num_warmup"] if self.config["save_warmup"] else 0
        )

        # Argsort the columns for each variable such that the columns are in row-major
        # order. This is important for efficiently saving to the HDF5 file.
        self.varname_to_column_order = self._get_c_order()

    def _get_c_order(self) -> dict[str, npt.NDArray[np.int64]]:
        """
        This function argsorts the columns such that they are in row-major order.
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
            for col in self.fit.column_names
        ]

        # Now we assign the row-major argsort of indices to each variable
        varname_to_column_order = {}
        for varname, var in itertools.chain(
            self.fit.metadata.method_vars.items(), self.fit.metadata.stan_vars.items()
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

            # Argsort the indices
            varname_to_column_order[varname] = np.array(
                sorted(range(len(var_inds)), key=var_inds.__getitem__)
            )

        return varname_to_column_order

    def write_netcdf(
        self,
        filename: str | None = None,
        precision: Literal["double", "single", "half"] = "single",
        mib_per_chunk: int | None = None,
    ) -> str:
        """
        Write the HDF5 file to disk.
        """
        # If no filename is provided, we create one based on the csv files
        filename = (
            filename
            or os.path.commonprefix(self.fit.runset.csv_files).rstrip("_") + ".nc"
        )

        # Get the data types for the method and stan variables
        method_var_dtypes = {
            "lp__": _NP_TYPE_MAP[precision]["float"],
            "accept_stat__": _NP_TYPE_MAP[precision]["float"],
            "stepsize__": _NP_TYPE_MAP[precision]["float"],
            "treedepth__": _NP_TYPE_MAP[precision]["int"],
            "n_leapfrog__": _NP_TYPE_MAP[precision]["int"],
            "divergent__": _NP_TYPE_MAP[precision]["int"],
            "energy__": _NP_TYPE_MAP[precision]["float"],
        }
        stan_var_dtypes, stan_var_dimnames = self._get_stan_var_dtypes_dimnames(
            precision
        )
        assert not set(stan_var_dtypes.keys()).intersection(
            set(method_var_dtypes.keys())
        ), "Stan variable names should not overlap with method variable names."

        # Create the HDF5 file
        with h5netcdf.File(filename, "w") as netcdf_file:

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
                netcdf_file.attrs[attr] = self.fit.metadata.cmdstan_config[attr]

            # Set dimensions
            netcdf_file.dimensions = {
                "chain": self.config["total_chains"],
                "draw": self.num_draws,
                **{
                    dimname: dimsize
                    for varinfo in filter(
                        lambda x: len(x) > 0, stan_var_dimnames.values()
                    )
                    for dimname, dimsize in varinfo
                },
            }

            # We need a group for metadata, samples, posterior predictive checks,
            # observations, and transformed parameters.
            metadata_group = netcdf_file.create_group("sample_stats")
            sample_group = netcdf_file.create_group("posterior")
            ppc_group = netcdf_file.create_group("posterior_predictive")
            observed_group = netcdf_file.create_group("observed_data")

            # Create variables for each of the method variables. Build a mapping
            # from the variable name to the dataset object. We store all of the
            # method variables with a single chunk.
            varname_to_dset = {
                varname: metadata_group.create_variable(
                    name=varname,
                    dimensions=("chain", "draw"),
                    dtype=method_var_dtypes[varname],
                    chunks=(self.config["total_chains"], self.num_draws),
                )
                for varname in self.fit.metadata.method_vars.keys()
            }

            # Now we can create a dataset for each stan variable. We update the
            # mapping from the variable name to the dataset object
            for varname, stan_dtype in stan_var_dtypes.items():

                # Get the shape of the variable
                if len(shape_info := stan_var_dimnames[varname]) == 0:
                    named_shape, true_shape = (), ()
                else:
                    named_shape, true_shape = zip(*shape_info)

                # Calculate the chunk shape. We always hold the first two dimensions
                # frozen. This is because the first two dimensions are what we
                # are typically performing operations over.
                chunk_shape = get_chunk_shape(
                    array_shape=(
                        self.config["total_chains"],
                        self.num_draws,
                        *true_shape,
                    ),
                    array_precision=precision,
                    mib_per_chunk=mib_per_chunk,
                    frozen_dims=(0, 1),
                )

                # We record without the '_ppc' suffix
                recorded_varname = varname.removesuffix("_ppc")

                # Build the group
                group = ppc_group if varname.endswith("_ppc") else sample_group
                varname_to_dset[varname] = group.create_variable(
                    name=recorded_varname,
                    dimensions=("chain", "draw", *named_shape),
                    dtype=stan_dtype,
                    chunks=chunk_shape,
                )

                # If an observable, also create a dataset in the observed group
                # and populate it with the data
                if varname.endswith("_ppc") and self.data is not None:
                    observed_group.create_variable(
                        name=recorded_varname,
                        data=self.data[recorded_varname].squeeze(),
                        dimensions=named_shape,
                        dtype=stan_dtype,
                        chunks=chunk_shape[2:],
                    )

            # Now we populate the datasets with the data from the csv files
            for chain_ind, csv_file in enumerate(
                tqdm(sorted(self.fit.runset.csv_files), desc="Converting CSV to NetCDF")
            ):
                for draw_ind, draw in enumerate(
                    tqdm(
                        self._parse_csv(
                            filename=csv_file,
                            method_var_dtypes=method_var_dtypes,
                            stan_var_dtypes=stan_var_dtypes,
                        ),
                        total=self.num_draws,
                        desc=f"Processing chain {chain_ind + 1}",
                        leave=False,
                        position=1,
                    )
                ):
                    for varname, varvals in draw.items():
                        varname_to_dset[varname][
                            chain_ind, draw_ind
                        ] = varvals.squeeze()

                # We must have all the draws for this chain
                assert draw_ind == self.num_draws - 1  # pylint: disable=W0631

        return filename

    def _get_stan_var_dtypes_dimnames(
        self, precision: Literal["double", "single", "half"]
    ) -> tuple[
        dict[str, Union[type[np.floating], type[np.integer]]],
        dict[str, tuple[tuple[str, int], ...]],
    ]:
        """Retrieves the datatypes and dimension names for the stan variables."""

        def get_dimname() -> tuple[tuple[str, int], ...] | tuple[()]:
            """Retrieves the dimension names for the current component."""
            # Get the name of the dimensions
            named_shape = []
            for dimind, dimsize in enumerate(component.shape[::-1]):

                # See if we can get the name of the dimension. If we cannot, this must
                # be a singleton dimension
                if (dimname := dim_map.get((dimind, dimsize))) is None:
                    assert dimsize == 1
                    continue

                # If we have a name, record
                named_shape.append((dimname, dimsize))

            # If we have no dimensions, we return an empty tuple
            if len(named_shape) == 0:
                return ()

            # We have our named shape
            return tuple(named_shape[::-1])

        # We will need the map from dimension depth and size to dimension name
        dim_map = self.model.get_dimname_map()

        # Datatypes for the stan variables
        stan_var_dtypes = {}
        stan_var_dimnames = {}
        for varname, component in self.model.named_model_components_dict.items():

            # We only take parameters and transformed parameters
            if not isinstance(component, (Parameter, TransformedParameter)):
                continue

            # Update the varname if needed
            if isinstance(component, Parameter) and component.observable:
                varname = f"{varname}_ppc"

            # Record the datatype
            stan_var_dtypes[varname] = _NP_TYPE_MAP[precision][
                "int" if isinstance(component, DiscreteDistribution) else "float"
            ]

            # Record the dimension names
            stan_var_dimnames[varname] = get_dimname()

        return stan_var_dtypes, stan_var_dimnames

    def _parse_csv(
        self,
        filename: str,
        method_var_dtypes: dict[str, Union[type[np.floating], type[np.integer]]],
        stan_var_dtypes: dict[str, Union[type[np.floating], type[np.integer]]],
    ) -> Generator[dict[str, npt.NDArray], None, None]:
        """
        Parses a csv file and returns a generator of dictionaries. Each dictionary
        contains a numpy array with the correct shape and datatype for each variable
        in the model associated with this instance of the class.
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
                    method_var_dtypes.items(), stan_var_dtypes.items()
                ):

                    # Get the variable object from the metadata
                    var = getattr(
                        self.fit.metadata,
                        "stan_vars" if varname in stan_var_dtypes else "method_vars",
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


def cmdstan_csv_to_netcdf(
    path: str | list[str] | os.PathLike | CmdStanMCMC,
    model: "dms_stan.model.Model",
    data: dict[str, Any] | None = None,
    output_filename: str | None = None,
    precision: Literal["double", "single", "half"] = "single",
    mib_per_chunk: int | None = None,
) -> str:
    """
    Converts a set of cmdstan csv files to a single hdf5 file. This is particularly
    useful for large datasets that need to be processed in chunks with Dask.
    """
    # If no data, check for default data in the model. Otherwise, data provided
    # takes priority
    if data is None and model.has_default_data:
        data = model.default_data

    # Build the converter
    converter = CmdStanMCMCToNetCDFConverter(fit=path, model=model, data=data)

    # Run conversion
    return converter.write_netcdf(
        filename=output_filename,
        precision=precision,
        mib_per_chunk=mib_per_chunk,
    )


def dask_enabled_summary_stats(inference_obj: az.InferenceData) -> xr.Dataset:
    """Calculates summary statistics for the inference object using Dask."""
    # Queue up the delayed computations
    with az_dask():
        delayed_summaries = [
            inference_obj.posterior.mean(dim=("chain", "draw")),
            inference_obj.posterior.std(dim=("chain", "draw")),
            az.hdi(
                inference_obj,
                hdi_prob=0.94,
                dask_gufunc_kwargs={"output_sizes": {"hdi": 2}},
            ),
        ]

        # Compute the results
        mean, std, hdi = dask.compute(*delayed_summaries)

    # Concatenate the results
    return xr.concat(
        [
            mean.assign_coords(metric=["mean"]),
            std.assign_coords(metric=["sd"]),
            hdi.assign_coords(hdi=["hdi_3%", "hdi_97%"]).rename(hdi="metric"),
        ],
        dim="metric",
    )


def dask_enabled_diagnostics(inference_obj: az.InferenceData) -> xr.Dataset:
    """Calculates diagnostics for the inference object using Dask."""
    # Run computations
    with az_dask():
        diagnostics = dask.compute(
            az.mcse(inference_obj.posterior, method="mean"),
            az.mcse(inference_obj.posterior, method="sd"),
            az.ess(inference_obj.posterior, method="bulk"),
            az.ess(inference_obj.posterior, method="tail"),
            az.rhat(inference_obj.posterior),
        )

    # Concatenate the results and return
    return xr.concat(
        [
            dset.assign_coords(metric=[metric])
            for metric, dset in zip(
                ["mcse_mean", "mcse_sd", "ess_bulk", "ess_tail", "r_hat"], diagnostics
            )
        ],
        dim="metric",
    )


class SampleResults(MLEInferenceRes):
    """
    Holds results from a CmdStanMCMC object and an ArviZ object. This should never
    be instantiated directly. Instead, use the `from_disk` method to load the object.
    """

    def __init__(
        self,
        model: Union["dms_stan.model.Model", None] = None,
        fit: str | list[str] | os.PathLike | CmdStanMCMC | None = None,
        data: dict[str, npt.NDArray] | None = None,
        precision: Literal["double", "single", "half"] = "single",
        inference_obj: Optional[az.InferenceData | str] = None,
        mib_per_chunk: int | None = None,
        use_dask: bool = False,
    ):
        # Store the CmdStanMCMC object
        self.fit = fit

        # Note whether we are using dask
        self.use_dask = use_dask

        # If the inference object is None, we assume that we need to create a NETCDF
        # file from the CmdStanMCMC object.
        if inference_obj is None:

            # Compile results to a NetCDF file
            inference_obj = cmdstan_csv_to_netcdf(
                path=fit,
                model=model,
                data=data,
                precision=precision,
                mib_per_chunk=mib_per_chunk,
            )

        # If the inference object is a string, we assume that it is a NetCDF file
        # to be loaded from disk
        if isinstance(inference_obj, str):

            # Load the inference object. Ignore warnings about chunking.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="The specified chunks separate the stored chunks along dimension",
                )
                inference_obj = az.from_netcdf(
                    filename=inference_obj,
                    engine="h5netcdf",
                    group_kwargs={
                        k: {"chunks": "auto" if use_dask else None}
                        for k in ("posterior", "posterior_predictive", "sample_stats")
                    },
                )

        # Initialize the parent class
        super().__init__(inference_obj)

    def calculate_summaries(
        self,
        var_names: list[str] | None = None,
        filter_vars: Literal[None, "like", "regex"] = None,
        kind: Literal["all", "stats", "diagnostics"] = "all",
        round_to: int = 2,
        circ_var_names: list[str] | None = None,
        stat_focus: str = "mean",
        stat_funcs: Optional[Union[dict[str, callable], callable]] = None,
        extend: bool = True,
        hdi_prob: float = 0.94,
        skipna: bool = False,
        diagnostic_varnames: Sequence[str] = (
            "mcse_mean",
            "mcse_sd",
            "ess_bulk",
            "ess_tail",
            "r_hat",
        ),
    ) -> xr.Dataset:
        """
        This is a wrapper around `az.summary`. See that function for details. There
        is one important difference: This function will two new groups to the ArviZ
        InferenceData object. The first is 'variable_diagnostic_stats', which contains any
        metrics that are diagnostic in nature; the second is 'variable_summary_stats', which
        contains summary statistics for the samples. The `diagnostic_varnames` argument
        is used to specify which metrics are considered diagnostic.

        Note that a full `xr.DataSet` is returned containing all metrics, including
        both diagnostics and summary statistics.

        This function will update any existing groups in the ArviZ object with
        the same name.
        """
        # We use custom functions if we are using dask
        if self.use_dask:

            # Calculate the two datasets
            summary_stats = dask_enabled_summary_stats(self.inference_obj)
            diagnostics = dask_enabled_diagnostics(self.inference_obj)

            # Combine datasets to get the summaries
            if kind == "all":
                summaries = xr.concat([summary_stats, diagnostics], dim="metric")
            elif kind == "stats":
                summaries = summary_stats
            elif kind == "diagnostics":
                summaries = diagnostics

        # Otherwise, we use the default ArviZ functions
        else:
            # Run the inherited method to get the summary statistics
            summaries = super().calculate_summaries(
                var_names=var_names,
                filter_vars=filter_vars,
                kind=kind,
                round_to=round_to,
                circ_var_names=circ_var_names,
                stat_focus=stat_focus,
                stat_funcs=stat_funcs,
                extend=extend,
                hdi_prob=hdi_prob,
                skipna=skipna,
            )

            # Identify the diagnostic and summary statistics
            noted_diagnostics = set(diagnostic_varnames)
            calculated_metrics = set(summaries.metric.values.tolist())

            diagnostic_metrics = list(noted_diagnostics & calculated_metrics)
            stat_metrics = list(calculated_metrics - noted_diagnostics)

            summary_stats = summaries.sel(metric=stat_metrics)
            diagnostics = summaries.sel(metric=diagnostic_metrics)

        # Update the groups
        if kind == "all" or kind == "diagnostics":
            self._update_group("variable_diagnostic_stats", diagnostics)
        if kind == "all" or kind == "stats":
            self._update_group("variable_summary_stats", summary_stats)
        return summaries

    def calculate_diagnostics(self) -> xr.Dataset:
        """
        This method performs a few actions:

        1.  It will calculate the diagnostics for the samples. This is done by
            calling `self.calculate_summaries` with the `kind` argument set to "diagnostics".
            All side effects of `self.calculate_summaries` are preserved. No default
            arguments of the `self.calculate_summaries` method can be changed.
        2.  It will print out a summary of the diagnostic results.
        3.  It will return the diagnostics as an `xr.Dataset` and the diagnostic
            summary as a dictionary.
        """
        return self.calculate_summaries(kind="diagnostics")

    def evaluate_sample_stats(
        self,
        max_tree_depth: int | None = None,
        ebfmi_thresh: float = DEFAULT_EBFMI_THRESH,
    ) -> xr.Dataset:
        """
        This evaluates the sample statistics for the samples. This is done by
        checking the following conditions:

            1. The maximum tree depth is less than or equal to `max_tree_depth`
            2. The E-BFMI is greater than or equal to `ebfmi_thresh`
            3. The samples did not diverge

        The output dataset contains boolean arrays for each test, where those that
        fail are `True` and those that pass are `False`. Specifically, tests are
        considered to fail if the following conditions are met:

            1. Tree Depth == `max_tree_depth`
            2. E-BFMI < `ebfmi_thresh`
            3. Samples diverged (i.e., `diverging` of `sample_stats` is `True`)

        After evaluation, the `inference_obj` will have `sample_diagnostic_tests`
        as a new group containing boolean arrays for each test at the sample level
        (i.e., each step in each MCMC chain). The dataset of boolean arrays is returned.
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
        self, r_hat_thresh: float = DEFAULT_RHAT_THRESH, ess_thresh=DEFAULT_ESS_THRESH
    ) -> xr.Dataset:
        """
        This identifies variables that fail the diagnostic tests. The output dataset
        contains boolean arrays for each test, where those that fail are `True`
        and those that pass are `False`. Specifically, tests are considered to fail
        if the following conditions are met:

            1. R-hat >= `r_hat_thresh`
            2. Effective Sample Size - Bulk <= `ess_thresh`
            3. Effective Sample Size - Tail <= `ess_thresh`

        After evaluation, the `inference_obj` will have `variable_diagnostic_tests`
        as an additional group. This contains boolean arrays for each test at the
        variable level (i.e., each variable in the model). The dataset of boolean
        arrays is returned.
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
        StrippedTestRes,
        dict[str, StrippedTestRes],
    ]:
        """
        Evaluates diagnostic tests and prints a summary of the results. This method
        also returns dictionaries that map from test/variable names to numpy arrays
        of indices at which tests failed. This can only be run if the `calculate_diagnostics`,
        `evaluate_sample_stats`, and `evaluate_variable_diagnostic_stats` methods
        have been run first.
        """

        def process_test_results(test_res_dataarray: xr.Dataset) -> ProcessedTestRes:
            """
            Process the test results from a DataArray into a dictionary of test results.

            Parameters
            ----------
            test_res_dataarray : xr.Dataset
                The DataArray containing the test results.

            Returns
            -------
            ProcessedTestRes
                A dictionary where the keys are the variable names and the values are tuples
                containing the indices of the failed tests and the total number of tests.
            """
            return {
                varname: (np.atleast_1d(tests.values).nonzero(), tests.values.size)
                for varname, tests in test_res_dataarray.items()
            }

        def strip_totals(processed_test_results: ProcessedTestRes) -> StrippedTestRes:
            """
            Strip the totals from the test results.

            Parameters
            ----------
            processed_test_results : ProcessedTestRes
                The processed test results from the `process_test_results` function.

            Returns
            -------
            StrippedTestRes
                The processed test results with the totals stripped.
            """
            return {k: v[0] for k, v in processed_test_results.items()}

        def report_test_summary(
            processed_test_results: ProcessedTestRes,
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
        max_tree_depth: int | None = None,
        ebfmi_thresh: float = DEFAULT_EBFMI_THRESH,
        r_hat_thresh: float = DEFAULT_RHAT_THRESH,
        ess_thresh: float = DEFAULT_ESS_THRESH,
        silent: bool = False,
    ) -> tuple[StrippedTestRes, dict[str, StrippedTestRes]]:
        """
        Runs the full diagnostics pipeline. Under the hood, this calls, in order:

        1. `calculate_diagnostics`
        2. `evaluate_sample_stats`
        3. `evaluate_variable_diagnostic_stats`
        4. `identify_failed_diagnostics`

        The results of the last step are returned.
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
        self, display: Literal[True], width: int, height: int
    ) -> hv.HoloMap: ...

    @overload
    def plot_sample_failure_quantile_traces(
        self, display: Literal[False], width: int, height: int
    ) -> dict[str, hv.Overlay]: ...

    def plot_sample_failure_quantile_traces(
        self, *, display=True, width=600, height=600
    ):
        """
        Plots the quantiles of sampled values that failed the diagnostic tests relative
        to the samples that passed the tests. The x-axis is a percentage of total
        parameters, passing from the parameter whose failed samples were in the
        lowest percentile to the parameter whose failed samples were in the highest
        percentile relative to the samples that passed the tests. The y-axis is the
        quantiles of the failed samples relative to the passing samples. The traces
        for each individual failure are plotted, as is the median trace over all
        failures.
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
            failed_quantiles = calculate_relative_quantiles(
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
        self, *, display: Literal[True], width: int, height: int, plot_quantiles: bool
    ) -> VariableAnalyzer: ...

    @overload
    def plot_variable_failure_quantile_traces(
        self, *, display: Literal[False], width: int, height: int, plot_quantiles: bool
    ) -> pn.pane.HoloViews: ...

    def plot_variable_failure_quantile_traces(
        self,
        display=True,
        width=800,
        height=400,
        plot_quantiles=False,
    ):
        """
        Plots the quantiles of variables that failed the diagnostic tests. The x-axis
        is the step along the sampling process and the y-axis is the quantiles of
        the failed samples relative to others in the same family that passed the
        tests.
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
    def from_disk(
        cls,
        path: str,
        csv_files: list[str] | str | None = None,
        skip_fit: bool = False,
        use_dask: bool = False,
    ) -> "SampleResults":
        """
        Loads the object from disk. The path should be the path to the net-cdf file.
            If this is provided and `csv_files` is not, then the path to the csv
            files will be assumed to have the same prefix as the net-cdf file (i.e.,
            the path that results from removing "_arviz.nc" from the end of the
            provided path). If `csv_files` is provided, it will be used instead
            and passed as the `path` argument to `cmdstanpy.from_csv`.
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
                    "The path to the netcdf file must end with '_arviz.nc' in order "
                    "to automatically find the csv files."
                )

        # Initialize the object
        return cls(
            model=None,
            fit=None if csv_files is None else fit_from_csv_noload(csv_files),
            inference_obj=path,
            use_dask=use_dask,
        )


def fit_from_csv_noload(path: str | list[str] | os.PathLike) -> CmdStanMCMC:
    """
    Parses the output files from Stan. This is derived from `cmdstanpy.from_csv`,
    and performs the same function, but stops short of loading the data into memory.
    This is useful for large datasets that need to be processed in chunks.
    """

    def identify_files() -> list[str]:
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
            raise ValueError(f"No CSV files found in directory {path}")
        for file in csvfiles:
            if not (os.path.exists(file) and os.path.splitext(file)[1] == ".csv"):
                raise ValueError(
                    f"Bad CSV file path spec, includes non-csv file: {file}"
                )

        return csvfiles

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
                f" found outputs from method {config_dict["method"]}"
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
    csvfiles = identify_files()
    config_dict = get_config_dict()
    sampler_args = build_sampler_args()
    return build_fit()
