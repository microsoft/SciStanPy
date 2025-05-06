"""Handles results from a `dms_stan.model.stan.StanModel` object."""

import os.path
import warnings

from glob import glob
from typing import Literal, Optional, overload, Sequence, Union

import arviz as az
import cmdstanpy
import holoviews as hv
import numpy as np
import numpy.typing as npt
import panel as pn
import xarray as xr

import dms_stan.model.stan as stan_module

from dms_stan.custom_types import ProcessedTestRes, StrippedTestRes
from dms_stan.defaults import (
    DEFAULT_EBFMI_THRESH,
    DEFAULT_ESS_THRESH,
    DEFAULT_MAX_TREE_DEPTH,
    DEFAULT_RHAT_THRESH,
)
from dms_stan.model.components import Normal, TransformedParameter
from dms_stan.model.pytorch.map import MAPInferenceRes
from dms_stan.plotting import calculate_relative_quantiles

# pylint: disable=too-many-lines


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
                failing_inds = [
                    ".".join(map(str, indices))
                    for indices in zip(*np.nonzero(metrictests))
                ]

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

        # Split into passing and failing tests
        self._failing_samples, self._passing_samples = (
            self._np_samples[tests],
            self._np_samples[~tests],
        )

        # Map between index name and failing index
        self._index_map = {
            ".".join(map(str, indices)): i
            for i, indices in enumerate(zip(*np.nonzero(tests)))
        }

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
        failing_samples = self._failing_samples[self._index_map[self.indexchoice.value]]
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


class SampleResults(MAPInferenceRes):
    """
    Holds results from a CmdStanMCMC object and an ArviZ object. This should never
    be instantiated directly. Instead, use the `from_disk` method to load the object.
    """

    # TODO: Set a flag for excluding transformed parameters. This should be True
    # by default. They can be exlucded because they can be recalculated easily.
    def __init__(
        self,
        stan_model: Union["stan_module.StanModel", None] = None,
        fit: cmdstanpy.CmdStanMCMC | None = None,
        data: dict[str, npt.NDArray] | None = None,
        precision: Literal["double", "single", "half"] = "single",
        inference_obj: Optional[az.InferenceData | str] = None,
    ):
        # Store the CmdStanMCMC object
        self.fit = fit

        # If the inference object passed in is `None`, we build one using the
        # other arguments. Otherwise, we use the one passed in.
        super().__init__(
            inference_obj=(
                self._build_arviz_object(
                    stan_model=stan_model, data=data, precision=precision
                )
                if inference_obj is None
                else inference_obj
            ),
        )

        # Save the arviz object to disk if it was not loaded from disk
        if inference_obj is None:
            self.save_netcdf()

    def _build_arviz_object(
        self,
        stan_model: "stan_module.StanModel",
        data: dict[str, npt.NDArray],
        precision: Literal["double", "single", "half"] = "single",
    ) -> az.InferenceData:

        # Decide on the precision of the data
        if precision == "double":
            float_dtype, int_dtype = np.float64, np.int64
        elif precision == "single":
            float_dtype, int_dtype = np.float32, np.int32
        elif precision == "half":
            float_dtype, int_dtype = np.float16, np.int16
        else:
            raise ValueError("precision must be one of 'double', 'single', or 'half'.")

        # Get the additional arguments needed for building the arviz object
        posterior_predictive = self._get_ppc(data)
        coords, dims = self._get_coords_dims(stan_model)

        # Build the arviz object
        inference_obj = az.from_cmdstanpy(
            posterior=self.fit,
            posterior_predictive=posterior_predictive,
            observed_data=data,
            constant_data=stan_model.autogathered_data,
            coords=coords,
            dims=dims,
            dtypes=stan_model,
        )

        # Identify the names of transformed and non-transformed parameters
        transformed_params = set(
            k.model_varname
            for k, v in stan_model.model.all_model_components.items()
            if isinstance(v, TransformedParameter)
        )

        # Squeeze the dummy dimensions out of the ArviZ object and cooerce data
        # to the precision requested. Ideally, precision would be set during creation
        # of the original ArviZ object, but there is no option to do this right
        # now. Also assign the type of parameter as an attribute.
        for group, dataset in inference_obj.items():

            # Squeeze the dummy dimensions out of the dataset
            dataset = dataset.squeeze(drop=True)

            # Coerce the data to the requested precision
            for varname, dataarray in dataset.data_vars.items():
                if np.issubdtype(dataarray.dtype, np.floating):
                    dataset[varname] = dataarray.astype(
                        float_dtype, casting="same_kind"
                    )
                elif np.issubdtype(dataarray.dtype, np.integer):
                    dataset[varname] = dataarray.astype(int_dtype, casting="same_kind")

            # Add the transformed parameters as an attribute
            dataset = dataset.assign_attrs(transformed_params=transformed_params)

            # Update the group in the ArviZ object with the modified dataset
            setattr(inference_obj, group, dataset)

        # Rename the posterior predictive variables to not have the "_ppc" suffix
        inference_obj.posterior_predictive = inference_obj.posterior_predictive.rename(
            {name: name.removesuffix("_ppc") for name in posterior_predictive}
        )

        return inference_obj

    def _get_ppc(self, data: dict[str, npt.NDArray]) -> list[str]:

        # Note the difference between the provided observed data and the known
        # observed data
        expected_observations = {
            name for name in self.fit.stan_variables().keys() if name.endswith("_ppc")
        }
        actual_observations = {k + "_ppc" for k in data.keys()}
        if additional_observations := actual_observations - expected_observations:
            raise ValueError(
                "The following observations were provided as data, but there were "
                "no samples generated for them by the Stan model: "
                + ", ".join(additional_observations)
            )
        if missing_observations := expected_observations - actual_observations:
            raise ValueError(
                "The following observations were expected to be provided as data, "
                "but were not: " + ", ".join(missing_observations)
            )

        return list(expected_observations)

    def _get_coords_dims(
        self, stan_model: "stan_module.StanModel"
    ) -> tuple[dict[str, npt.NDArray[np.int64]], dict[str, list[str]]]:
        """Get the coordinates and dimensions for the ArviZ object"""
        # Set up variables for recording
        varname_to_named_shape: dict[str, list[str]] = {}  # Named shapes
        dummies: set[str] = set()  # Dummy dimension names for singletons

        # Get a map from dimension depth and size to dimension name
        dim_map = stan_model.model.get_dimname_map()

        # We also need the variables for which there are samples generated by Stan
        sampled_varnames = set(self.fit.stan_variables().keys())

        # Process all variables
        for varname in stan_model.program.all_varnames:

            # Get the stan-friendly variable name
            stan_varname = varname.replace(".", "__")

            # Get the model component
            model_component = stan_model.model[varname]

            # Get the name of the dimensions
            named_shape = [None] * model_component.ndim
            for dimind, dimsize in enumerate(model_component.shape[::-1]):

                # See if we can get the name of the dimension. If we cannot, this must
                # be a singleton dimension
                if (dimname := dim_map.get((dimind, dimsize))) is None:
                    assert dimsize == 1
                    dimname = f"dummy_{dimind}"
                named_shape[dimind] = dimname

            # Scalars are a special case unless they are sampled
            if model_component.ndim == 0 and stan_varname not in sampled_varnames:
                named_shape = ["dummy_0"]

            # Update the set of dummies
            dummies.update(name for name in named_shape if name.startswith("dummy_"))

            # Update the mapping
            named_shape = named_shape[::-1]
            varname_to_named_shape[stan_varname] = named_shape

            # If an observable, also add the posterior predictive samples
            if model_component.observable:
                varname_to_named_shape[f"{stan_varname}_ppc"] = named_shape

            # If non-centered, also add the "raw" version
            if (
                isinstance(model_component, Normal)
                and not model_component.is_hyperparameter
            ):
                varname_to_named_shape[f"{stan_varname}_raw"] = named_shape

        # Get the coordinates
        coords: dict[str, npt.NDArray[np.int64]] = {
            name: np.arange(dimsize) for (_, dimsize), name in dim_map.items()
        } | {dummy: np.array([0]) for dummy in dummies}

        return coords, varname_to_named_shape

    def save_netcdf(  # pylint: disable=arguments-renamed
        self, file_prefix: str | None = None
    ) -> None:
        """
        Saves the ArViz object to a netcdf file. This is performed on initialization
        when not loading from disk
        """
        # If a file prefix is provided, use it. Otherwise, extract from the csv
        # files attached to the CmdStanMCMC object
        if file_prefix is not None:
            prefix = file_prefix
        elif self.fit is None:
            raise ValueError(
                "No CmdStanMCMC object is attached to this SampleResults object, "
                "so a file prefix cannot be automatically generated. Please provide "
                "a file prefix."
            )
        else:
            prefix = os.path.commonprefix(self.fit.runset.csv_files)

        # Save the ArViZ object to a netcdf file
        super().save_netcdf(prefix.rstrip("_") + "_" + "arviz.nc")

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

        # Build new or update old groups
        self._update_group(
            "variable_diagnostic_stats", summaries.sel(metric=diagnostic_metrics)
        )
        self._update_group(
            "variable_summary_stats", summaries.sel(metric=stat_metrics), force_del=True
        )

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
        max_tree_depth: int = DEFAULT_MAX_TREE_DEPTH,
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
        # The maximum tree depth should be less than or equal to the `max_tree_depth`
        # argument. If it isn't, we have an error.
        # pylint: disable=no-member
        if (
            found_max := self.inference_obj.sample_stats.tree_depth.max().item()
        ) > max_tree_depth:
            raise ValueError(
                f"The maximum tree depth found in the sample stats was {found_max},"
                f"which is greater than the provided `max_tree_depth` of {max_tree_depth}."
                "Did you run sampling with a non-default value for the tree depth?"
            )

        # Run all tests and build a dataset
        sample_tests = xr.Dataset(
            {
                "low_ebfmi": self.inference_obj.sample_stats.energy < ebfmi_thresh,
                "max_tree_depth_reached": self.inference_obj.sample_stats.tree_depth
                == max_tree_depth,
                "diverged": self.inference_obj.sample_stats.diverging,
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
        max_tree_depth: int = DEFAULT_MAX_TREE_DEPTH,
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
        cls, path: str, csv_files: list[str] | str | None = None, skip_fit: bool = False
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
            # If the path to the netcdf file ends with "_arviz.nc", raise a warning
            # that we cannot automatically find the csv files. If it does, find
            # the csv files
            if path.endswith("_arviz.nc"):
                csv_files = list(glob(path.removesuffix("_arviz.nc") + "*.csv"))
            else:
                warnings.warn(
                    "The path to the netcdf file must end with '_arviz.nc' in order "
                    "to automatically find the csv files."
                )

        # Initialize the object
        return cls(
            stan_model=None,
            fit=None if csv_files is None else cmdstanpy.from_csv(csv_files),
            data=None,
            inference_obj=az.from_netcdf(path),
        )
