import os.path
import warnings

from typing import Literal

import holoviews as hv
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from scipy import stats

from dms_stan.flip_dsets import load_trpb_dataset
from dms_stan.model.pytorch.map import MAPInferenceRes
from dms_stan.model.stan.stan_results import SampleResults
from dms_stan.plotting import quantile_plot
from dms_stan.utils import faster_autocorrelation


LIB_TO_OD600 = {
    "libA": xr.DataArray(np.array([0.1, 0.75, 2.55]), dims=["b"]),
    "libB": xr.DataArray(np.array([0.1, 0.75, 3.3]), dims=["b"]),
    "libC": xr.DataArray(np.array([0.1, 0.74, 1.95]), dims=["b"]),
    "libD": xr.DataArray(
        np.array(
            [[0.05, 0.19, 0.29, 0.51, 0.85, 1.42], [0.05, 0.18, 0.28, 0.49, 0.97, 1.81]]
        ),
        dims=["c", "b"],
    ),
    "libE": xr.DataArray(
        np.array(
            [
                [0.05, 0.20, 0.27, 0.47, 0.91, 1.41],
                [0.05, 0.20, 0.26, 0.44, 0.94, 1.54],
            ]
        ),
        dims=["c", "b"],
    ),
    "libF": xr.DataArray(
        np.array(
            [
                [0.05, 0.17, 0.20, 0.23, 0.27, 0.79],
                [0.05, 0.17, 0.20, 0.24, 0.27, 0.79],
            ]
        ),
        dims=["c", "b"],
    ),
    "libG": xr.DataArray(
        np.array(
            [
                [0.05, 0.14, 0.18, 0.23, 0.44, 2.0],
                [0.05, 0.14, 0.18, 0.23, 0.44, 1.95],
            ]
        ),
        dims=["c", "b"],
    ),
    "libH": xr.DataArray(
        np.array(
            [[0.05, 0.15, 0.19, 0.26, 0.67, 2.9], [0.05, 0.14, 0.18, 0.26, 0.58, 1.85]]
        ),
        dims=["c", "b"],
    ),
    "libI": xr.DataArray(
        np.array(
            [[0.05, 0.36, 0.83, 1.24, 1.7, 1.95], [0.05, 0.39, 0.87, 1.36, 2.1, 2.25]]
        ),
        dims=["c", "b"],
    ),
    "four-site": xr.DataArray(
        np.array(
            [
                [0.025, 0.19, 0.51, 1.26, 1.50, 1.675, 1.75],
                [0.025, 0.19, 0.52, 1.34, 1.625, 1.75, 1.875],
            ]
        ),
        dims=["c", "b"],
    ),
}


class TrpBDataAnalyzer:
    def __init__(
        self,
        *,
        lib_name: str,
        flip_data_loc: str,
        map_res_prefix: str,
        hmc_res: str,
        hmc_use_dask: bool = False,
    ):

        # Note the name
        self.lib_name = lib_name

        # Load the counts and fitness data
        count_data = load_trpb_dataset(
            os.path.join(flip_data_loc, "counts", "trpb", f"{lib_name}.csv")
        )
        self.reported_fitness = pd.read_csv(
            os.path.join(flip_data_loc, "fitnesses", "trpb", f"{lib_name}.csv")
        )

        # Load the map and hmc samples
        self.map_samples = MAPInferenceRes(map_res_prefix + "_samples.nc")
        self.hmc_samples = SampleResults.from_disk(
            hmc_res, skip_fit=True, use_dask=hmc_use_dask
        )

        # Load the map
        self.map = xr.Dataset(
            data_vars={
                varname: (
                    tuple(
                        dim
                        for dim in self.map_samples.inference_obj.posterior[
                            varname
                        ].dims
                        if dim not in {"chain", "draw"}
                    ),
                    vals.squeeze(),
                )
                for varname, vals in np.load(
                    map_res_prefix + "_map.npz", allow_pickle=True
                ).items()
            }
        )

        # Convert relevant data to xarrays
        self.times = xr.DataArray(count_data["times"], dims=["b"])
        self.reported_starting_counts = (
            self.map_samples.inference_obj.observed_data.starting_counts
        )
        self.reported_timepoint_counts = (
            self.map_samples.inference_obj.observed_data.timepoint_counts
        )

        # Build a mask for variants containing a stop codon
        self.variants = count_data["variants"]
        self.stop_mask = xr.DataArray(["*" in var for var in self.variants], dims=["a"])

        # Make sure the indices of the loaded and sampled datasets are identical
        assert np.array_equal(
            count_data["starting_counts"], self.reported_starting_counts.values
        )
        assert np.array_equal(
            count_data["timepoint_counts"], self.reported_timepoint_counts.values
        )
        assert np.array_equal(
            self.map_samples.inference_obj.observed_data.starting_counts.values,
            self.reported_starting_counts.values,
        )
        assert np.array_equal(
            self.map_samples.inference_obj.observed_data.timepoint_counts.values,
            self.reported_timepoint_counts.values,
        )

    def recalculate_fitness(self, count_source: Literal["reported", "map", "hmc"]):

        # Get the appropriate counts
        c0, cf = {
            "reported": (self.reported_starting_counts, self.reported_timepoint_counts),
            "map": (
                self.map_samples.inference_obj.posterior_predictive.starting_counts,
                self.map_samples.inference_obj.posterior_predictive.timepoint_counts,
            ),
            "hmc": (
                self.hmc_samples.inference_obj.posterior_predictive.starting_counts,
                self.hmc_samples.inference_obj.posterior_predictive.timepoint_counts,
            ),
        }[count_source]

        # Normalize by OD600 to get 'x' values
        od600_0 = LIB_TO_OD600[self.lib_name].isel(b=0)
        assert (od600_0 := np.unique(od600_0.values)).size == 1
        x0 = c0 / od600_0
        xf = cf / LIB_TO_OD600[self.lib_name].isel(b=slice(1, None))

        # Calculate the specific growth rate for each timepoint
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="divide by zero encountered in log"
            )
            mu = np.log(xf / x0) / self.times

        # Replace infs with NaNs. Do the same for any variants that had x0 < 10
        # on average
        mu = mu.where((~np.isinf(mu)) & (c0 >= 10))

        # We need to know the average growth rate for stop codon variants at each timepoint
        stop_mu = mu.sel(a=["*" in var for var in self.variants]).mean(
            dim="a", skipna=True
        )

        # We need the maximum growth rate for each timepoint
        max_mu = mu.max(dim="a")

        # Calculate the fitness values. Ignore NaN in the calculation.
        return ((mu - stop_mu) / (max_mu - stop_mu)).mean(
            dim=[dim for dim in mu.dims if dim not in {"chain", "draw", "a"}],
            skipna=True,
        )

    def fitness_sanity_check(self) -> np.floating:
        """Makes sure the recalculated fitness is close to the reported fitness values"""
        # Get the recalculated fitness values
        recalculated_fitness = self.recalculate_fitness("reported")

        # Make sure the combinations are in the correct order
        reported_aas = self.reported_fitness.AAs.to_list()
        reported_aas_set = set(reported_aas)
        variant_mask = [var in reported_aas_set for var in self.variants]
        assert reported_aas == [
            var for var, mask in zip(self.variants, variant_mask) if mask
        ]

        # Get the correlations
        return stats.spearmanr(
            recalculated_fitness.sel(a=variant_mask).values,
            self.reported_fitness.fitness.to_numpy(),
            nan_policy="raise",
        ).statistic

    def get_fitness_correlations(
        self, fitness: xr.DataArray | npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:

        # The input fitness must be an xarray DataArray
        if not isinstance(fitness, xr.DataArray):
            raise ValueError("The input fitness must be an xarray DataArray")

        # The provided fitness values must have a "chain" and "draw" dimension
        if "chain" not in fitness.dims or "draw" not in fitness.dims:
            raise ValueError(
                "The provided fitness values must have a 'chain' and 'draw' dimension"
            )

        # Stack the chain and draw dimensions
        fitness = fitness.stack(draws=("chain", "draw"))
        assert fitness.dims == ("a", "draws")
        fitness = fitness.values.T

        # Get the recalculated fitness values
        recalculated_fitness = self.recalculate_fitness("reported").values
        assert recalculated_fitness.ndim == 1

        # Get spearman correlation between reported and recalculated fitness
        recalculated_vs_sampled = np.array(
            [
                stats.spearmanr(
                    sample, recalculated_fitness, nan_policy="omit"
                ).statistic
                for sample in fitness
            ]
        )

        return recalculated_vs_sampled
        # TODO: Figure out if we want autocorrelation here or not
        # Now the autocorrelation of the sampled fitness values
        autocorr = faster_autocorrelation(fitness)

        # Package fitness values and correlations
        return recalculated_vs_sampled, autocorr

    def get_stdev_vs_counts(self, fitness: xr.DataArray) -> hv.Scatter:
        """
        Plot the standard deviation of the fitness values against the total number
        of counts for each variant.
        """
        # Get the standard deviation of the fitness values
        reduction_dims = [dim for dim in fitness.dims if dim != "a"]
        stdevs = fitness.std(dim=reduction_dims, skipna=True)
        n_nonna = fitness.notnull().sum(dim=reduction_dims)

        # Get the total number of counts for each variant
        total_counts = self.reported_starting_counts.sum(
            dim=[dim for dim in self.reported_starting_counts.dims if dim != "a"]
        ) + self.reported_timepoint_counts.sum(
            dim=[dim for dim in self.reported_timepoint_counts.dims if dim != "a"]
        )

        # Build the plot
        return hv.Scatter(
            (total_counts.values, stdevs.values, self.variants, n_nonna.values),
            kdims=["Total counts"],
            vdims=["Standard deviation", "Variant", "N Fitness Values"],
        ).opts(
            title="Standard Deviation of Fitness Values vs Total Counts",
            xlabel="Total Counts",
            ylabel="Standard Deviation",
            tools=["hover"],
            width=600,
            height=600,
        )

    def plot_fitness_distribution(
        self,
        fitness_samples: npt.NDArray[np.floating],
        reference: npt.NDArray[np.floating] | None = None,
    ) -> hv.Overlay:
        """Builds a quantile plot of the fitness values provided by the samples"""
        # The fitness samples must be at least 2D
        if fitness_samples.ndim < 2:
            raise ValueError("The fitness samples must be at least 2D")

        # The reference is the median of the fitness values if not provided. The
        # median is calculated along the last axis of the fitness samples.
        if reference is None:
            reference = np.nanmedian(
                fitness_samples, axis=list(range(fitness_samples.ndim - 1))
            )

        # The reference must be 1D
        if reference.ndim != 1:
            raise ValueError("The reference must be 1D")

        # The reference must be the same length as the fitness samples
        if reference.shape[0] != fitness_samples.shape[-1]:
            raise ValueError(
                "The reference must be the same length as the fitness samples"
            )

        # Drop values that are `NaN` in the reference
        mask = ~np.isnan(reference)
        reference, fitness_samples = reference[mask], fitness_samples[..., mask]

        # Sort by the reference values
        sorted_indices = np.argsort(reference)
        reference, fitness_samples = (
            reference[sorted_indices],
            fitness_samples[..., sorted_indices],
        )

        # Get the ranks of the samples and reference values
        ref_rank = stats.rankdata(reference, nan_policy="raise")
        fit_ranks = stats.rankdata(fitness_samples, axis=-1, nan_policy="omit")

        # Build the plot
        return quantile_plot(
            x=ref_rank,
            reference=fit_ranks,
            quantiles=[0.025, 0.25],
            observed_type="scatter",
            allow_nan=True,
        ).opts(
            title="Quantile Plot of Fitness Values",
            xlabel="Variant Index",
            ylabel="Variant Rank",
            height=600,
            width=600,
        )
