"""Holds code relevant for the TrpB datasets."""

import numpy as np
import numpy.typing as npt
import pandas as pd

import dms_stan.model as dms
import dms_stan.model.components as dms_components
import dms_stan.operations as dms_ops


def load_trpb_dataset(filepath: str) -> dict[str, npt.NDArray]:
    """
    Load a TrpB dataset from Johnston et al.
    """
    # Load in the data
    data = pd.read_csv(filepath)

    # Get the output columns
    output_cols = sorted(
        (col for col in data.columns if col.startswith("OutputCount")),
        key=lambda x: int(x.split("_")[1]),
    )

    # Get unique combo to input counts
    t0_data = data[["AAs", "InputCount_1"]].drop_duplicates()
    assert (t0_data.AAs.value_counts() == 1).all()
    combo_order = t0_data.AAs.tolist()
    t0_counts = t0_data.InputCount_1.to_numpy(dtype=int)

    # Get the timepoint counts
    times = data["Time (h)"].unique().astype(float)
    times.sort()
    tg0_counts = np.zeros([len(output_cols), len(times), len(combo_order)], dtype=int)
    for timeind, time in enumerate(times):

        # Filter down to just the data for this time
        time_data = data[data["Time (h)"] == time]

        # Make sure the data is in the right order
        assert time_data.AAs.tolist() == combo_order

        # Get the counts
        tg0_counts[:, timeind, :] = time_data[output_cols].to_numpy(dtype=int).T

    return {
        "times": np.concatenate([[0], times]),
        "starting_counts": t0_counts,
        "timepoint_counts": tg0_counts,
    }


class TrpBGrowthModel(dms.Model):
    """
    Models the TrpB count data from Johnston et al. using an exponential growth
    function to model the time-dependent increase in counts and a multinomial distribution
    to model the counts at each timepoint.
    """

    def __init__(
        self,
        times: npt.NDArray[np.floating],
        starting_counts: npt.NDArray[np.integer],
        timepoint_counts: npt.NDArray[np.integer],
    ):

        # Check shapes. Times and starting counts should be 1D arrays. Timepoint
        # counts should be a 3D array with shape (n_replicates, n_timepoints - 1, n_variants)
        if times.ndim != 1:
            raise ValueError("Times should be a 1D array")
        if starting_counts.ndim != 1:
            raise ValueError("Starting counts should be a 1D array")
        if timepoint_counts.ndim != 3:
            raise ValueError("Timepoint counts should be a 3D array")

        # Get the number of timepoints, replicates, and variants
        n_timepoints = len(times)
        n_replicates = timepoint_counts.shape[0]
        n_variants = timepoint_counts.shape[2]

        # Check that the shapes of the arrays are consistent
        if n_timepoints != timepoint_counts.shape[1] + 1:
            raise ValueError(
                "Timepoint counts should have one fewer timepoint than the number "
                "of times"
            )
        if starting_counts.shape[0] != n_variants:
            raise ValueError(
                "Starting counts and timepoint counts should have the same number "
                "of replicates"
            )

        # Normalize the times such that the maximum timepoint is 1
        times = times / times.max()
        if times[0] != 0.0:
            raise ValueError("Times should start at 0")

        # The starting distributions at t = 0 are the same for all replicates. Because
        # the relative abundances need to add up to 1, our prior is a Dirichlet
        # distribution with an alpha value > 1 to enforce the belief that the inputs
        # are roughly equally abundant.
        self.theta_t0 = dms_components.Dirichlet(alpha=10.0, shape=(n_variants,))

        # We will be modeling the log of exponential growth, so our starting values
        # for `log_A` are the log of the starting counts.
        self.log_A = dms_ops.log(self.theta_t0)  # pylint: disable=invalid-name

        # Now hyperparameters on the growth rate. We assume that the noise in the
        # growth rate is the same for all variants, so we model a single standard
        # deviation for all variants. The mean growth rate is allowed to vary between
        # variants, however.
        self.r_mean = dms_components.Exponential(beta=10.0, shape=(n_variants,))
        self.r_std = dms_components.HalfNormal(sigma=0.25)  # Shared across variants

        # Now the next layer of the model. Both technical replicates start from
        # the same culture, so the starting counts (and hence log_A) are the same.
        # The growth rate might vary between replicates, however, so we model a
        # separate one for each replicate.
        self.r = dms_components.Normal(
            mu=self.r_mean, sigma=self.r_std, shape=(n_replicates, 1, n_variants)
        )

        # Calculate the thetas at t > 0.
        self.theta_tg0 = dms_ops.exp(
            dms_ops.normalize_log(
                dms_components.LogExponentialGrowth(
                    log_A=self.log_A,
                    r=self.r,
                    t=dms_components.Constant(times[None, 1:, None], togglable=False),
                    shape=(n_replicates, n_timepoints - 1, n_variants),
                )
            )
        )

        # Model the counts data.
        self.starting_counts = dms_components.Multinomial(
            theta=self.theta_t0,
            N=dms_components.Constant(starting_counts.sum(), togglable=False),
            shape=(n_variants,),
        )
        self.timepoint_counts = dms_components.Multinomial(
            theta=self.theta_tg0,
            N=dms_components.Constant(
                timepoint_counts.sum(axis=-1, keepdims=True), togglable=False
            ),
            shape=timepoint_counts.shape,
        )

        # Record the data
        self.times = times
        self.starting_counts_data = starting_counts
        self.timepoint_counts_data = timepoint_counts

    def approximate_map(self, *args, **kwargs):
        """Approximates the MAP estimate of the model."""
        return super().approximate_map(
            *args,
            data={
                "starting_counts": self.starting_counts_data,
                "timepoint_counts": self.timepoint_counts_data,
            },
            **kwargs,
        )

    def mcmc(self, *args, **kwargs):
        """Runs MCMC on the model."""
        return super().mcmc(
            *args,
            data={
                "starting_counts": self.starting_counts_data,
                "timepoint_counts": self.timepoint_counts_data,
            },
            **kwargs,
        )

    @classmethod
    def from_data_file(cls, filepath: str):
        """
        Load a TrpB dataset from Johnston et al. and create a model from it.
        """
        return cls(**load_trpb_dataset(filepath))
