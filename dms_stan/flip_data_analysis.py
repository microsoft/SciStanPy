import os.path
import warnings

from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from scipy import stats

from dms_stan.flip_dsets import load_trpb_dataset
from dms_stan.model.pytorch.map import MAPInferenceRes
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
    def __init__(self, lib_name: str, flip_data_loc: str, map_res_prefix: str):

        # Note the name
        self.lib_name = lib_name

        # Load the counts and fitness data
        count_data = load_trpb_dataset(
            os.path.join(flip_data_loc, "counts", "trpb", f"{lib_name}.csv")
        )
        self.reported_fitness = pd.read_csv(
            os.path.join(flip_data_loc, "fitnesses", "trpb", f"{lib_name}.csv")
        )

        # Load the map samples
        self.map_samples = MAPInferenceRes(map_res_prefix + "_samples.nc")

        # TODO: Remove filter. We don't need it now that we don't save 'None' values
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
                if vals.size != 1 or vals.item() is not None
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

    def recalculate_fitness(
        self, count_source: Literal["reported", "map_samples", "mcmc"]
    ):

        # Get the appropriate counts
        c0, cf = {
            "reported": (self.reported_starting_counts, self.reported_timepoint_counts),
            "map_samples": (
                self.map_samples.inference_obj.posterior_predictive.starting_counts,
                self.map_samples.inference_obj.posterior_predictive.timepoint_counts,
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

        # Replace infs with NaNs
        mu = mu.where(~np.isinf(mu))

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

    def get_fitness_correlations(self):

        # Calculate the relevant fitness values
        recalculated_fitness = self.recalculate_fitness("reported")
        sampled_fitness = self.recalculate_fitness("map_samples").stack(
            draws=("chain", "draw")
        )

        # Make sure the combinations are in the correct order
        reported_aas = self.reported_fitness.AAs.to_list()
        reported_aas_set = set(reported_aas)
        variant_mask = [var in reported_aas_set for var in self.variants]
        assert reported_aas == [
            var for var, mask in zip(self.variants, variant_mask) if mask
        ]

        # Get spearman correlation between reported and recalculated fitness
        reported_vs_recalculated = stats.spearmanr(
            recalculated_fitness.sel(a=variant_mask).values,
            self.reported_fitness.fitness.to_numpy(),
            nan_policy="raise",
        ).statistic

        # Now get spearman correlation between the recalculated and sampled fitness values
        assert sampled_fitness.dims == ("a", "draws")
        transposed_sampled_fitness = sampled_fitness.values.T
        recalculated_vs_sampled = np.array(
            [
                stats.spearmanr(
                    sample, recalculated_fitness, nan_policy="omit"
                ).statistic
                for sample in transposed_sampled_fitness
            ]
        )

        # Now the autocorrelation of the sampled fitness values
        autocorr = faster_autocorrelation(transposed_sampled_fitness)

        # Package fitness values and correlations
        return (
            {
                "reported": self.reported_fitness.fitness.to_numpy(),
                "recalculated": recalculated_fitness,
                "map_samples": sampled_fitness,
            },
            {
                "reported_vs_recalculated": reported_vs_recalculated,
                "recalculated_vs_sampled": recalculated_vs_sampled,
                "sampled_vs_sampled": autocorr,
            },
        )
