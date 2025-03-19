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
import pandas as pd
import panel as pn
import xarray as xr

from scipy import stats

import dms_stan.model.stan as stan_module

from dms_stan.custom_types import ProcessedTestRes, StrippedTestRes
from dms_stan.defaults import (
    DEFAULT_EBFMI_THRESH,
    DEFAULT_ESS_THRESH,
    DEFAULT_MAX_TREE_DEPTH,
    DEFAULT_RHAT_THRESH,
)
from dms_stan.model.components import Normal

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


def _log10_shift(*args: npt.NDArray) -> tuple[npt.NDArray, ...]:
    """
    Identify the minimum value across all arrays. Then, use that single value (i.e.,
    even if 30 arrays were passed, we find the absolute minimum across all 30 to
    get a single value) to shift the arrays such that the absolute minimum is 1.
    Finally, apply log10 to the shifted arrays.
    """
    # Get the minimum value across all arrays
    min_val = min(np.min(arg) for arg in args)

    # Shift the arrays and apply log10
    return tuple(np.log10(arg - min_val + 1) for arg in args)


class SampleResults:
    """
    Holds results from a CmdStanMCMC object and an ArviZ object. This should never
    be instantiated directly. Instead, use the `from_disk` method to load the object.
    """

    def __init__(
        self,
        stan_model: Union["stan_module.StanModel", None] = None,
        fit: cmdstanpy.CmdStanMCMC | None = None,
        data: dict[str, npt.NDArray] | None = None,
        precision: Literal["double", "single", "half"] = "single",
        _from_disk: bool = False,
    ):
        # If loading from disk, skip the rest of the initialization
        if _from_disk:
            return

        # `stan_model`, `fit`, and `data` are all required
        if any(arg is None for arg in (stan_model, fit, data)):
            raise ValueError(
                "stan_model, fit, and data are all required when not loading from "
                "disk."
            )

        # Store the CmdStanMCMC object
        self.fit = fit

        # Build the ArviZ object
        self.inference_obj = self._build_arviz_object(
            stan_model=stan_model, data=data, precision=precision
        )

        # Save the arviz object to disk
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

        # Squeeze the dummy dimensions out of the ArviZ object and cooerce data
        # to the precision requested. Ideally, precision would be set during creation
        # of the original ArviZ object, but there is no option to do this right
        # now.
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

    def save_netcdf(self, file_prefix: str | None = None) -> None:
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
        self.inference_obj.to_netcdf(prefix + "arviz.nc")

    def _update_group(self, attrname: str, new_group: xr.Dataset) -> None:
        """Either adds or updates a group in the ArviZ object"""
        if hasattr(self.inference_obj, attrname):
            getattr(self.inference_obj, attrname).update(new_group)
        else:
            self.inference_obj.add_groups({attrname: new_group})

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

        def update_group(metric_names: list[str], attrname: str) -> None:
            """
            Replace the group in the ArviZ object with the new group. This is done
            by removing the old group and adding the new one.
            """
            # Do nothing if the group is empty
            if not metric_names:
                return

            # Get the group
            new_group = summaries.sel(metric=diagnostic_metrics)

            # Add the new group or update the old one
            self._update_group(attrname, new_group)

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

        # Identify the diagnostic and summary statistics
        noted_diagnostics = set(diagnostic_varnames)
        calculated_metrics = set(summaries.metric.values.tolist())

        diagnostic_metrics = list(noted_diagnostics.union(calculated_metrics))
        stat_metrics = list(calculated_metrics - noted_diagnostics)

        # Build new or update old groups
        update_group(diagnostic_metrics, "variable_diagnostic_stats")
        update_group(stat_metrics, "variable_summary_stats")

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
                varname: (np.nonzero(tests.values), tests.values.size)
                for varname, tests in test_res_dataarray.items()
            }

        def strip_totals(process_test_results: ProcessedTestRes) -> StrippedTestRes:
            """
            Strip the totals from the test results.

            Parameters
            ----------
            process_test_results : ProcessedTestRes
                The processed test results from the `process_test_results` function.

            Returns
            -------
            StrippedTestRes
                The processed test results with the totals stripped.
            """
            return {k: v[0] for k, v in process_test_results.items()}

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

    def _get_quantiles(self) -> xr.Dataset:
        """
        Calculates the quantiles of the observed data relative to the posterior
        predictive samples.
        """
        # pylint: disable=no-member
        return (
            (self.inference_obj.posterior_predictive - self.inference_obj.observed_data)
            < 0
        ).mean(dim=["chain", "draw"])

    @overload
    def check_calibration(
        self,
        *,
        return_deviance: Literal[False],
        display: Literal[True],
        width: int,
        height: int,
    ) -> hv.Layout: ...

    @overload
    def check_calibration(
        self,
        *,
        return_deviance: Literal[False],
        display: Literal[False],
        width: int,
        height: int,
    ) -> dict[str, hv.Overlay]: ...

    @overload
    def check_calibration(
        self,
        *,
        return_deviance: Literal[True],
        display: Literal[False],
        width: int,
        height: int,
    ) -> tuple[dict[str, hv.Overlay], dict[str, float]]: ...

    def check_calibration(
        self, *, return_deviance=False, display=True, width=600, height=600
    ):
        """
        This method checks how well the observed data matches the sampled posterior
        predictive samples. The procedure is as follows:

        1.  Calculate the (exclusive) quantiles of the observed data relative to
            the posterior predictive samples.
        2.  Plot an ECDF of the observed quantiles. A perfectly calibrated model
            will produce a straight line from (0, 0) to (1, 1). This is because
            the at the xth percentile, x% of samples should be less than the observed
            value.
        3.  Calculate the absolute difference in area between the observed ECDF
            and the idealized ECDF. This is the calibration score. A perfectly calibrated
            model will have a calibration score of 0. A perfectly miscalibrated
            model will have a calibration score of 0.5.

        Returns:
            If `display` is `True`, a holoviews.Layout object containing the ECDF
            plots for each observed variable. The plots will be displayed in a single
            column.

            If `display` is `False`, a list of holoviews.Overlay objects containing
            the ECDF plots for each observed variable.

            If `return_deviance` is `True`, a tuple containing the list of
            holoviews.Overlay objects and a dictionary mapping from the variable
            names to the calibration scores. The calibration scores are the absolute
            difference in area between the observed ECDF and the idealized ECDF.
            Note that `display` cannot be `True` if `return_deviance` is `True`.
        """

        # pylint: disable=line-too-long
        def calculate_deviance(
            x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]
        ) -> float:
            r"""
            Calculates the absolute difference in area between the observed ECDF and the
            ideal ECDF were the model fit perfectly. We can calculate this by subtracting
            the area under the curve of the ideal ECDF from the area under the curve of
            the observed ECDF, calculated using the trapezoidal rule. Formally:

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

        # pylint: enable=line-too-long

        # We cannot have both `display` and `return_deviance` set to True
        if display and return_deviance:
            raise ValueError(
                "Cannot have both `display` and `return_deviance` set to True."
            )

        # Build ECDFs for the quantiles. The ideal ECDF would be a straight line
        plots: dict[str, hv.Overlay] = {}
        deviances: dict[str, float] = {}
        for varname, data in self._get_quantiles().items():

            # Get the ECDF coordinates of the observed data
            ecdf = stats.ecdf(data.values.flatten())
            x, y = ecdf.cdf.quantiles, ecdf.cdf.probabilities

            # Calculate the absolute deviance
            dev = calculate_deviance(x, y)
            deviances[varname] = dev

            # Create the plot
            plots[varname] = (
                hv.Curve(
                    (x, y),
                    kdims=["Quantiles"],
                    vdims=["Cumulative Probability"],
                )
                * hv.Curve(
                    ((0, 1), (0, 1)),
                    kdims=["Quantiles"],
                    vdims=["Cumulative Probability"],
                ).opts(
                    line_color="black",
                    line_dash="dashed",
                )
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
        quantiles: Sequence[float],
        use_ranks: bool,
        logy: bool,
        display: Literal[True],
        width: int,
        height: int,
    ) -> hv.Layout: ...

    @overload
    def plot_posterior_predictive_samples(
        self,
        *,
        quantiles: Sequence[float],
        use_ranks: bool,
        logy: bool,
        display: Literal[False],
        width: int,
        height: int,
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
        """
        Plots observed data against the corresponding posterior predictive samples.
        The posterior predictive samples are plotted as a series of confidence intervals.

        Args:
            quantiles (Sequence[float]): The quantiles defining the plotted confidence
                intervals. Note that the median will always be included and the
                quantiles will be symmetrized (e.g., if passing in 0.025 as a quantile,
                0.975 will be added automatically to the list). Defaults to
                (0.025, 0.25, 0.5).
            use_ranks (bool): If `True`, the ranks of the observed values will be
                plotted on the x-axis instead of their raw values. This is useful
                when the observed values are not symmetrically distributed. Defaults
                to `True`.
            logy (bool): If `True`, the y-axis will be plotted on a logarithmic
                scale. Note that, due to a bug in the underlying holoviews library,
                y-values will be shifted to have a minimum of 1 before the log is
                applied and the log will be applied *before* plotting. This means
                that, from the perspective of the holoviews library, the y-axis
                will be plotted on a linear scale. Defaults to `False`.

        Returns:
            If `display` is `True`, a holoviews.Layout object containing the plots
            for each observed variable. The plots will be displayed in a single
            column.
            If `display` is `False`, a list of holoviews.Overlay objects containing
            the plots for each observed variable.
        """

        # Symmetrize the quantiles
        quantiles = _symmetrize_quantiles(quantiles)

        # Get the confidence intervals for the posterior predictive samples
        ppc_cis = self.inference_obj.posterior_predictive.apply(
            lambda x: x.stack(samples=("chain", "draw")).quantile(
                quantiles, dim="samples"
            )
        )

        # Process each observed variable
        plots: dict[str, hv.Overlay] = {}
        for varname, y_data_sampled in ppc_cis.data_vars.items():

            # Get the observed data
            y_data_obs = self.inference_obj.observed_data[  # pylint: disable=no-member
                varname
            ]
            assert y_data_obs.dims == y_data_sampled.dims[1:]
            assert y_data_sampled.dims[0] == "quantile"

            # Convert data to numpy arrays and reshape
            y_data_sampled = y_data_sampled.to_numpy().reshape(len(quantiles), -1)
            y_data_obs = y_data_obs.to_numpy().ravel()

            # If using a log-y axis, shift the y-data
            if logy:
                y_data_sampled, y_data_obs = _log10_shift(y_data_sampled, y_data_obs)

            # Get the x-axis data
            x_data = (
                stats.rankdata(y_data_obs, method="ordinal")
                if use_ranks
                else y_data_obs
            )

            # Sort data for plotting the areas and lines
            sorted_inds = np.argsort(x_data)
            x_data, y_data_obs, y_data_sampled = (
                x_data[sorted_inds],
                y_data_obs[sorted_inds],
                y_data_sampled[:, sorted_inds],
            )

            # Plot the observed data on top of the posterior predictive samples'
            # confidence intervals
            plots[varname] = hv.Overlay(
                [
                    hv.Area(
                        (
                            x_data,
                            y_data_sampled[quantile_ind],
                            y_data_sampled[-quantile_ind - 1],
                        ),
                        vdims=["Lower", "Upper"],
                    ).opts(
                        color="black",
                        alpha=0.2,
                        line_width=1,
                        line_color="black",
                        line_alpha=0.3,
                        show_legend=False,
                    )
                    for quantile_ind in range(len(quantiles) // 2)
                ]
                + [
                    hv.Scatter(
                        (x_data, y_data_obs),
                        label="Observed Data",
                    ).opts(color="gold", line_width=0, alpha=0.8, size=1)
                ]
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
        width: int,
        height: int,
        windowsize: Optional[int],
    ) -> hv.Layout: ...

    @overload
    def plot_observed_quantiles(
        self,
        *,
        use_ranks: bool,
        display: Literal[False],
        width: int,
        height: int,
        windowsize: Optional[int],
    ) -> dict[str, hv.Overlay]: ...

    def plot_observed_quantiles(
        self, *, use_ranks=True, display=True, width=600, height=400, windowsize=None
    ):
        """
        Plots the quantiles of the observed data relative to the posterior predictive
        samples. The x-axis is either the values of the observed data or their ranks.
        The y-axis is the quantiles of the observed data relative to the posterior
        predictive samples. A sliding window is used to calculate a rolling mean
        of the quantiles.

        Args:
            use_ranks (bool): If `True`, the ranks of the observed values will be
                plotted on the x-axis instead of their raw values. This is useful
                when the observed values are not symmetrically distributed. Defaults
                to `True`.
            display (bool): If `True`, the plots will be displayed. Defaults to `True`.
            width (int): The width of the plots. Defaults to 600.
            height (int): The height of the plots. Defaults to 400.
        Returns:
            If `display` is `True`, a holoviews.Layout object containing the plots
            for each observed variable. The plots will be displayed in a single
            column.
            If `display` is `False`, a list of holoviews.Overlay objects containing
            the plots for each observed variable.
        """
        # Loop over quantiles for different observed variables
        plots: dict[str, hv.Overlay] = {}
        for varname, y_data in self._get_quantiles().data_vars.items():

            # Get the observed data. This is the x-axis data.
            x_data = self.inference_obj.observed_data[  # pylint: disable=no-member
                varname
            ]
            assert y_data.dims == x_data.dims

            # Convert data to numpy arrays
            x_data = x_data.to_numpy().ravel()
            y_data = y_data.to_numpy().ravel()

            # Convert x-data to ranks if requested
            if use_ranks:
                x_data = stats.rankdata(x_data, method="ordinal")

            # Build the plot
            plots[varname] = (
                hv.HexTiles((x_data, y_data)).opts(
                    cmap="viridis",
                    colorbar=True,
                )
                * hv.Curve(
                    pd.DataFrame({"x": x_data, "y": y_data})
                    .sort_values(by="x")
                    .rolling(
                        window=max(
                            1,
                            (x_data.size // 100 if windowsize is None else windowsize),
                        )
                    )
                    .mean()
                    .dropna(),
                    "x",
                    "y",
                    label="Rolling Mean",
                ).opts(color="black", line_width=3)
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
        windowsize: Optional[int],
        quantiles: Sequence[float],
        logy_ppc_samples: bool,
        subplot_width: int,
        subplot_height: int,
    ) -> pn.Column: ...

    @overload
    def run_ppc(
        self,
        *,
        use_ranks: bool,
        display: Literal[False],
        square_ecdf: bool,
        windowsize: Optional[int],
        quantiles: Sequence[float],
        logy_ppc_samples: bool,
        subplot_width: int,
        subplot_height: int,
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
        """
        Runs all posterior predictive checks. This includes running the following
        methods:

            1. `plot_posterior_predictive_samples`
            2. `plot_observed_quantiles`
            3. `check_calibration`

        Args:
            use_ranks (bool): If `True`, the ranks of the observed values will be
                plotted on the x-axis instead of their raw values. This is useful
                when the observed values are not symmetrically distributed. Defaults
                to `True`.
            display (bool): If `True`, the plots will be displayed. Otherwise, a
                list of outputs from each of the called subfunctions (in the order
                listed above) will be returned. Defaults to `True`.
            square_ecdf (bool): If `True`, the ECDF plots will be made square by
                using the smaller of the width and height for both dimensions. Defaults
                to `True`.
            windowsize (int): The size of the rolling window for the ECDF plots.
                Defaults to None.
            quantiles (Sequence[float]): The quantiles defining the plotted confidence
                intervals. Note that the median will always be included and the
                quantiles will be symmetrized (e.g., if passing in 0.025 as a quantile,
                0.975 will be added automatically to the list). Defaults to
                (0.025, 0.25, 0.5).
            logy_ppc_samples (bool): If `True`, the y-axis of the posterior predictive
                samples plot will be logarithmic. Defaults to False.
        """
        # Get ecdf widths and heights
        if square_ecdf:
            ecdf_width = min(subplot_width, subplot_height)
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
        self = cls(_from_disk=True)

        # If the path to the csv files is provided, build the CmdStanMCMC object.
        # Otherwise, `fit` is `None`
        if csv_files is None:
            self.fit = None
        else:
            self.fit = cmdstanpy.from_csv(csv_files)

        # Load the arviz object from the netcdf file
        self.inference_obj = az.from_netcdf(path)

        return self
