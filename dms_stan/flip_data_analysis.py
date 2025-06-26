import os.path
import warnings

from abc import ABC, abstractmethod
from typing import Literal

import holoviews as hv
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from scipy import stats

from dms_stan.model.pytorch.mle import MLEInferenceRes
from dms_stan.model.stan.stan_results import SampleResults
from dms_stan.pipelines.mcmc_flip import LOAD_DATASET_MAP
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


def reshape_fitness(fitness: xr.DataArray) -> npt.NDArray[np.floating]:
    """Flattens the chain and draw dimensions of the fitness DataArray."""
    # If already 1-dimensional, add a singleton dimension
    if fitness.ndim == 1:
        return fitness.values[None]

    # Otherwise, stack the chain and draw dimensions into a single 'draws' dimension
    fitness = fitness.stack(draws=("chain", "draw"))
    assert fitness.dims == ("a", "draws")

    return fitness.values.T


class BaseDataAnalyzer(ABC):

    # Class variable for dataset type
    dataset_type: str = ""
    growth_rate_variable: str = ""

    def __init__(
        self,
        *,
        lib_name: str,
        flip_data_loc: str,
        mle_res_prefix: str,
        hmc_res: str,
        hmc_use_dask: bool = False,
    ):
        # We must have set the dataset type
        if not self.dataset_type:
            raise ValueError(
                "The dataset_type class variable must be set in the subclass."
            )

        # We must have set the growth rate variable
        if not self.growth_rate_variable:
            raise ValueError(
                "The growth_rate_variable class variable must be set in the subclass."
            )

        # Note the name
        self.lib_name = lib_name

        # Load the counts and fitness data
        load_func, file_ext = LOAD_DATASET_MAP[self.dataset_type]
        self.raw_count_data = load_func(
            os.path.join(
                flip_data_loc, "counts", self.dataset_type, f"{lib_name}{file_ext}"
            )
        )
        self.reported_fitness = pd.read_csv(
            os.path.join(
                flip_data_loc, "fitnesses", self.dataset_type, f"{lib_name}.csv"
            )
        )

        # Load the mle and hmc samples
        self.mle_samples = MLEInferenceRes(mle_res_prefix + "_samples.nc")
        self.hmc_samples = SampleResults.from_disk(
            hmc_res, skip_fit=True, use_dask=hmc_use_dask
        )

        # Load the mle
        self.mle = xr.Dataset(
            data_vars={
                varname: (
                    tuple(
                        dim
                        for dim in self.mle_samples.inference_obj.posterior[
                            varname
                        ].dims
                        if dim not in {"chain", "draw"}
                    ),
                    vals.squeeze(),
                )
                for varname, vals in np.load(
                    mle_res_prefix + "_mle.npz", allow_pickle=True
                ).items()
            }
        )

        # Convert relevant data to xarrays
        self.reported_starting_counts = (
            self.mle_samples.inference_obj.observed_data.starting_counts
        )
        self.reported_timepoint_counts = (
            self.mle_samples.inference_obj.observed_data.timepoint_counts
        )

        # Record variants
        self.variants = self.raw_count_data["variants"]

        # Make sure the indices of the loaded and sampled datasets are identical
        assert np.array_equal(
            self.raw_count_data["starting_counts"], self.reported_starting_counts.values
        )
        assert np.array_equal(
            self.raw_count_data["timepoint_counts"],
            self.reported_timepoint_counts.values,
        )
        assert np.array_equal(
            self.mle_samples.inference_obj.observed_data.starting_counts.values,
            self.reported_starting_counts.values,
        )
        assert np.array_equal(
            self.mle_samples.inference_obj.observed_data.timepoint_counts.values,
            self.reported_timepoint_counts.values,
        )

    @abstractmethod
    def recalculate_fitness(
        self, count_source: Literal["reported", "mle", "hmc"]
    ) -> tuple[xr.DataArray, xr.DataArray] | xr.DataArray:
        """
        Recalculates the fitness values based on the counts from the specified source.
        The source can be 'reported', 'mle', or 'hmc'.

        This base method should be overridden in subclasses to implement the specific
        logic for recalculating fitness based on the counts from the specified source.
        This method will return the starting and timepoint counts for the specified
        source.
        """
        # Get the appropriate counts
        return {
            "reported": (self.reported_starting_counts, self.reported_timepoint_counts),
            "mle": (
                self.mle_samples.inference_obj.posterior_predictive.starting_counts,
                self.mle_samples.inference_obj.posterior_predictive.timepoint_counts,
            ),
            "hmc": (
                self.hmc_samples.inference_obj.posterior_predictive.starting_counts,
                self.hmc_samples.inference_obj.posterior_predictive.timepoint_counts,
            ),
        }[count_source]

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
            recalculated_fitness.sel(a=variant_mask).values,  # pylint: disable=E1101
            self.reported_fitness.fitness.to_numpy(),
            nan_policy="raise",
        ).statistic

    def get_fitness_correlations(
        self,
        fitness: xr.DataArray,
        skip_autocorr: bool = False,
    ) -> (
        npt.NDArray[np.floating]
        | tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
    ):
        # Flatten the fitness values
        np_fitness = reshape_fitness(fitness)

        # Get the recalculated fitness values
        recalculated_fitness = self.recalculate_fitness(  # pylint: disable=E1101
            "reported"
        ).values
        assert recalculated_fitness.ndim == 1

        # Get spearman correlation between reported and recalculated fitness
        recalculated_vs_sampled = np.array(
            [
                stats.spearmanr(
                    sample, recalculated_fitness, nan_policy="omit"
                ).statistic
                for sample in np_fitness
            ]
        )

        # Go no further if there are any NaN values in the provided fitness values
        # and we are not forcing autocorrelation calculation
        if skip_autocorr:
            return recalculated_vs_sampled

        # Otherwise, calculate autocorrelation of the sampled fitness values
        autocorr = (
            faster_autocorrelation(np_fitness)
            if np.isnan(np_fitness).any()
            else stats.spearmanr(np_fitness, axis=1, nan_policy="raise").statistic
        )

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
        fitness: xr.DataArray,
        reference: npt.NDArray[np.floating] | None = None,
    ) -> hv.Overlay:
        """Builds a quantile plot of the fitness values provided by the samples"""
        # Flatten the fitness values
        fitness_samples = reshape_fitness(fitness)

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

    def get_growth_rate(self, source: Literal["mle", "hmc"]) -> xr.DataArray:
        """Returns the growth rate from the specified source and variable"""
        # Get the specified source
        if source == "mle":
            dset = self.mle
        elif source == "hmc":
            dset = self.hmc_samples.inference_obj.posterior
        else:
            raise ValueError(f"Invalid source: {source}. Must be 'mle' or 'hmc'.")

        # Get the growth rate variable
        base_rate = dset[self.growth_rate_variable]

        # Invert the growth rate variable if needed
        if self.growth_rate_variable.startswith("inv_"):
            return 1 / base_rate
        return base_rate

    def run_analysis(self):
        """
        Runs the full analysis for the dataset. This includes...

        1. Running the fitness sanity check.
        2. Calculating the fitness correlations between the recalculated and MLE/HMC
        fitness values.
        3. Calculating the correlations between the recalculated fitness and growth
        rate values. Also calculates the autocorrelation of the growth rate values.
        4. Plotting the quantile plot of the fitness values for the MLE and HMC.
        5. Plotting the quantile plot of the growth rate values for the MLE and HMC.
        6. Plotting the standard deviation of the fitness values against the total
        number of counts for each variant.
        7. Plotting the growth rate values against the total number of counts for
        each variant.
        8. Creating a dataframe that gives the sampled rate values.
        """
        # Run the fitness sanity check
        print(
            f"The correlation between the recalculated and reported fitness values is: "
            f"{self.fitness_sanity_check():.4f}"
        )

        # Recalculate the fitness values for the MLE and HMC sources
        print("Recalculating fitness values...")
        recalculated_mle = self.recalculate_fitness("mle")
        recalculated_hmc = self.recalculate_fitness("hmc")
        r_mle = self.get_growth_rate("mle")
        assert r_mle.ndim == 1, "Growth rate for MLE should be 1D"
        r_hmc = self.get_growth_rate("hmc")

        # Get the fitness correlations
        print("Calculating fitness correlations...")
        mle_vs_recalc = self.get_fitness_correlations(
            recalculated_mle, skip_autocorr=True
        )
        hmc_vs_recalc = self.get_fitness_correlations(
            recalculated_hmc, skip_autocorr=True
        )
        r_mle_vs_recalc = self.get_fitness_correlations(r_mle, skip_autocorr=True)
        r_hmc_vs_recalc, r_hmc_autocorr = self.get_fitness_correlations(r_hmc)

        # Get quantile plots
        print("Building quantile plots...")
        quantile_plots = (
            self.plot_fitness_distribution(recalculated_mle).opts(
                title="MLE: Recalculated Fitness Quantiles"
            )
            + self.plot_fitness_distribution(recalculated_hmc).opts(
                title="HMC: Recalculated Fitness Quantiles"
            )
            + self.plot_fitness_distribution(r_hmc).opts(
                title="HMC: Growth Rate Quantiles"
            )
        ).opts(hv.opts.Area(line_width=0.02))

        # Get the standard deviation vs counts plot
        print("Calculating standard deviation vs counts...")
        stdev_vs_counts = (
            self.get_stdev_vs_counts(recalculated_mle).opts(
                title="MLE: Recalculated Fitness σ vs Total Counts"
            )
            + self.get_stdev_vs_counts(recalculated_hmc).opts(
                title="HMC: Recalculated Fitness σ vs Total Counts"
            )
            + self.get_stdev_vs_counts(r_hmc).opts(
                title="HMC: Growth Rate σ vs Total Counts"
            )
        )

        # Build a dataframe of growth rate values
        print("Building dataframe of sampled growth rates...")
        sample_df = pd.DataFrame(
            {
                "variant": self.variants,
                "MLE": r_mle.values,
                **{
                    f"sample_{i}": sample
                    for i, sample in enumerate(reshape_fitness(r_hmc))
                },
            }
        )

        # Package the results
        return {
            "Correlations": {
                "MLE Recalculated vs Recalculated Reported": mle_vs_recalc,
                "HMC Recalculated vs Recalculated Reported": hmc_vs_recalc,
                "MLE Growth Rate vs Recalculated Reported": r_mle_vs_recalc,
                "HMC Growth Rate vs Recalculated Reported": r_hmc_vs_recalc,
                "HMC Growth Rate Autocorrelation": r_hmc_autocorr,
            },
            "Quantile Plots": quantile_plots,
            "Standard Deviation vs Counts": stdev_vs_counts,
            "Sampled Growth Rates": sample_df,
        }


class TrpBDataAnalyzer(BaseDataAnalyzer):

    # Set class variables
    dataset_type = "trpb"

    def __init__(self, **kwargs):
        """See BaseDataAnalyzer.__init__ for parameters."""
        # Run parent
        super().__init__(**kwargs)

        # We also need data on the variants that contain stop codons and the times
        self.times = xr.DataArray(self.raw_count_data["times"], dims=["b"])
        self.stop_mask = xr.DataArray(["*" in var for var in self.variants], dims=["a"])

    def recalculate_fitness(self, count_source: Literal["reported", "mle", "hmc"]):

        # Get the starting and timepoint counts from the specified source
        c0, cf = super().recalculate_fitness(count_source)

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


class TrpBHierarchicalDataAnalyzer(TrpBDataAnalyzer):
    growth_rate_variable = "inv_r_mean"


class TrpBNonHierarchicalDataAnalyzer(TrpBDataAnalyzer):
    growth_rate_variable = "r"


TRPB_DATA_ANALYZER_MAP = {
    "libA": TrpBNonHierarchicalDataAnalyzer,
    "libB": TrpBNonHierarchicalDataAnalyzer,
    "libC": TrpBNonHierarchicalDataAnalyzer,
    "libD": TrpBHierarchicalDataAnalyzer,
    "libE": TrpBHierarchicalDataAnalyzer,
    "libF": TrpBHierarchicalDataAnalyzer,
    "libG": TrpBHierarchicalDataAnalyzer,
    "libH": TrpBHierarchicalDataAnalyzer,
    "libI": TrpBHierarchicalDataAnalyzer,
    "four-site": TrpBHierarchicalDataAnalyzer,
}
