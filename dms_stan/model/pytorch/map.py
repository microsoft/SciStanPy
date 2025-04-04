"""Holds code for the maximum a posteriori (MAP) estimation of the model parameters."""

from typing import Generator, Literal, Optional, overload, Sequence, Union

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

import dms_stan as dms

from dms_stan.plotting import (
    calculate_relative_quantiles,
    hexgrid_with_mean,
    plot_calibration,
    quantile_plot,
)


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


class MAPInferenceRes:
    """
    Holds results from a CmdStanMCMC object and an ArviZ object. This should never
    be instantiated directly. Instead, use the `from_disk` method to load the object.
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
            {"posterior", "posterior_predictive", "observed_data"}
            - set(self.inference_obj.groups())
        ):
            raise ValueError(
                f"ArviZ object is missing the following groups: {', '.join(missing_groups)}"
            )

    def save_netcdf(self, filename: str) -> None:
        """
        Saves the ArViz object to a netcdf file.
        """
        self.inference_obj.to_netcdf(filename)

    def _update_group(
        self, attrname: str, new_group: xr.Dataset, force_del: bool = False
    ) -> None:
        """Either adds or updates a group in the ArviZ object"""
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
        round_to: int = 2,
        circ_var_names: list[str] | None = None,
        stat_focus: str = "mean",
        stat_funcs: Optional[Union[dict[str, callable], callable]] = None,
        extend: bool = True,
        hdi_prob: float = 0.94,
        skipna: bool = False,
    ) -> xr.Dataset:
        """
        This is a wrapper around `az.summary`. See that function for details. There
        is one important difference: This function will add the group 'variable_summary_stats',
        which contains summary statistics for the samples.

        Note that a full `xr.DataSet` is returned containing all metrics.

        This function will update any existing groups in the ArviZ object with
        the same name.
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
        """
        Iterates over the posterior predictive samples and observed variables, converting
        the samples to 2D arrays and the observations to 1D arrays.
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

        1.  Calculate the (inclusive) quantiles of the observed data relative to
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
        # We cannot have both `display` and `return_deviance` set to True
        if display and return_deviance:
            raise ValueError(
                "Cannot have both `display` and `return_deviance` set to True."
            )

        # Loop over the posterior predictive samples
        plots: dict[str, hv.Overlay] = {}
        deviances: dict[str, float] = {}
        for varname, reference, observed in self._iter_pp_obs():

            # Build calibration plots and record deviance
            plot, dev = plot_calibration(reference, observed[None])
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
            plots[varname] = quantile_plot(
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
        for varname, reference, observed in self._iter_pp_obs():

            # Get the quantiles of the observed data relative to the reference
            y = calculate_relative_quantiles(
                reference, observed[None] if observed.ndim == 1 else observed
            )

            # Flatten the data and update x to use rankings if requested
            x, y = observed.ravel(), y.ravel()
            x = stats.rankdata(x, method="ordinal") if use_ranks else x

            # Build the plot
            plots[varname] = hexgrid_with_mean(
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
                using the width for both width and height dimensions of the plot.
                Defaults to `True`.
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


class MAPParam:
    """Holds the MAP estimate for a single parameter."""

    def __init__(
        self,
        name: str,
        value: Optional[npt.NDArray],
        distribution: dms.custom_types.DMSStanDistribution,
    ):

        # Store the inputs
        self.name = name
        self.map = value
        self.distribution = distribution

    def draw(
        self, n: int, *, seed: Optional[int] = None, batch_size: Optional[int] = None
    ) -> npt.NDArray:
        """
        Sample from the MAP estimate.
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


class MAP:
    """Holds the MAP estimate for all parameters of a model."""

    def __init__(
        self,
        model: "dms.model.Model",
        map_estimate: dict[str, npt.NDArray],
        distributions: dict[str, torch.distributions.Distribution],
        losses: npt.NDArray,
        data: dict[str, npt.NDArray],
    ):

        # The keys of the map estimate should be a subset of the keys of the distributions
        if not set(map_estimate.keys()).issubset(distributions.keys()):
            raise ValueError(
                "Keys of map estimate should be a subset of the keys of the distributions"
            )

        # Record the model and data
        self.model = model
        self.data = data

        # Store inputs. Each key in the map estimate will be mapped to an instance
        # variable
        self.parameters: str = []
        for key, value in distributions.items():
            self.parameters.append(key)
            setattr(
                self,
                key,
                MAPParam(name=key, value=map_estimate.get(key), distribution=value),
            )

        # Record the loss trajectory as a pandas dataframe
        self.losses = pd.DataFrame(
            {"-log pdf/pmf": losses, "iteration": np.arange(len(losses))}
        )

    def plot_loss_curve(self, logy: bool = True):
        """Plots the loss curve of the MAP estimation."""
        # Get y-label and title
        if logy:
            ylabel = "log(-log pdf/pmf)"
            title = "Log Loss Curve"
        else:
            ylabel = "-log pdf/pmf"
            title = "Loss Curve"

        return self.losses.hvplot.line(
            x="iteration", y="-log pdf/pmf", title=title, logy=logy, ylabel=ylabel
        )

    @overload
    def draw(
        self,
        n: int,
        *,
        seed: Optional[int],
        as_xarray: Literal[True],
        as_inference_data: Literal[False],
        batch_size: Optional[int] = None,
    ) -> xr.Dataset: ...

    @overload
    def draw(
        self,
        n: int,
        *,
        seed: Optional[int],
        as_xarray: Literal[False],
        batch_size: Optional[int] = None,
    ) -> dict[str, npt.NDArray]: ...

    def draw(self, n: int, *, seed=None, as_xarray=False, batch_size=None):
        """Draws samples from the MAP estimate.

        Args:
            n (int): The number of samples to draw.
            seed (int, optional): Sets the random seed. Defaults to None.
            as_xarray (bool, optional): If `True`, results are returned as an xarray
                DataSet. Defaults to `False`, meaning results are returned as a
                dictionary of numpy arrays. This and `as_inference_data` are mutually
                exclusive.

        Returns:
            dict[str, npt.NDArray] | xr.DataSet: The samples drawn from the MAP
                estimate. If `as_xarray` is `True`, returns an xarray DataSet.
                Otherwise, returns a dictionary of numpy arrays.
        """
        # Set the seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Draw samples
        draws = {
            self.model.all_model_components_dict[param]: getattr(self, param).draw(
                n, batch_size=batch_size
            )
            for param in self.parameters
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
        n: int = 1000,
        *,
        seed: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> MAPInferenceRes:
        """Builds an inference data object from the MAP estimate."""
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
                    p
                    for p in self.parameters
                    if not self.model.all_model_components_dict[p].observable
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
                    p
                    for p in self.parameters
                    if self.model.all_model_components_dict[p].observable
                ]
            ],
        )
        return MAPInferenceRes(inference_data)
