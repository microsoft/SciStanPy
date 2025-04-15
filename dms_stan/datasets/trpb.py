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


class TrpBBaseGrowthModel(dms.Model):
    """Base class for all TrpB growth models."""

    def __init__(
        self,
        times: npt.NDArray[np.floating],
        starting_counts: npt.NDArray[np.integer],
        timepoint_counts: npt.NDArray[np.integer],
        r_mean_beta: float = 1.0,
        r_std_sigma: float = 0.05,
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
        self._n_timepoints = len(times)
        self._n_replicates = timepoint_counts.shape[0]
        self._n_variants = timepoint_counts.shape[2]

        # Check that the shapes of the arrays are consistent
        if self.n_timepoints != timepoint_counts.shape[1] + 1:
            raise ValueError(
                "Timepoint counts should have one fewer timepoint than the number "
                "of times"
            )
        if starting_counts.shape[0] != self.n_variants:
            raise ValueError(
                "Starting counts and timepoint counts should have the same number "
                "of replicates"
            )

        # Normalize the times such that the maximum timepoint is 1
        times = times / times.max()
        if times[0] != 0.0:
            raise ValueError("Times should start at 0")

        # Record the data
        self.starting_counts_data = starting_counts
        self.timepoint_counts_data = timepoint_counts

        # Total number of counts is always the same
        self.starting_counts_total = dms_components.Constant(
            starting_counts.sum(), togglable=False
        )
        self.timepoint_counts_total = dms_components.Constant(
            timepoint_counts.sum(axis=-1, keepdims=True), togglable=False
        )

        # Also define time as a constant
        self.tg0 = dms_components.Constant(times[None, 1:, None], togglable=False)

        # Every model has a "rate" parameter, which gives the growth rate of each
        # variant in each replicate. Each variant gets its own `r`` which we model
        # as being drawn from a normal distribution with mean `r_mean` and standard
        # deviation `r_std`. The mean and standard deviation are different by variant,
        self.r_mean = dms_components.Exponential(
            beta=r_mean_beta, shape=(self.n_variants,)
        )
        self.r_std = dms_components.HalfNormal(
            sigma=r_std_sigma, shape=(self.n_variants,)
        )
        self.r = dms_components.Normal(
            mu=self.r_mean,
            sigma=self.r_std,
            shape=(self.n_replicates, 1, self.n_variants),
        )

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
    def from_data_file(cls, filepath: str, **kwargs):
        """
        Load a TrpB dataset from Johnston et al. and create a model from it.
        """
        return cls(**load_trpb_dataset(filepath), **kwargs)

    @property
    def n_timepoints(self) -> int:
        """Number of timepoints in the dataset."""
        return self._n_timepoints

    @property
    def n_replicates(self) -> int:
        """Number of replicates in the dataset."""
        return self._n_replicates

    @property
    def n_variants(self) -> int:
        """Number of variants in the dataset."""
        return self._n_variants


class _TrpBInitParam(TrpBBaseGrowthModel):
    def __init__(
        self,
        times: npt.NDArray[np.floating],
        starting_counts: npt.NDArray[np.integer],
        timepoint_counts: npt.NDArray[np.integer],
        r_mean_beta: float = 1.0,
        r_std_sigma: float = 0.05,
        alpha: float = 2.0,
    ):
        # Run inherited init
        super().__init__(
            times=times,
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            r_mean_beta=r_mean_beta,
            r_std_sigma=r_std_sigma,
        )

        # Every model has a theta_t0 parameter, which gives the starting proportions
        # at the beginning of the experiment. We assume that its possible values
        # are a fundamental property of the variant; hence, we define one per variant.
        self.theta_t0 = dms_components.Dirichlet(alpha=alpha, shape=(self.n_variants,))

        # The starting counts are modeled as a multinomial distribution
        self.starting_counts = dms_components.Multinomial(
            theta=self.theta_t0,
            N=self.starting_counts_total,
            shape=(self.n_variants,),
        )


class TrpBExponentialGrowth(_TrpBInitParam):
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
        r_mean_beta: float = 1.0,
        r_std_sigma: float = 0.05,
        alpha: float = 2.0,
    ):
        # Run inherited init
        super().__init__(
            times=times,
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            r_mean_beta=r_mean_beta,
            r_std_sigma=r_std_sigma,
            alpha=alpha,
        )

        # What are our proportions at t > 0?
        self.raw_abundances = dms_components.ExponentialGrowth(
            A=self.theta_t0,
            r=self.r,
            t=self.tg0,
            shape=(self.n_replicates, self.n_timepoints - 1, self.n_variants),
        )
        self.theta_tg0 = dms_ops.normalize(self.raw_abundances)

        # Model the counts data after growth
        self.timepoint_counts = dms_components.Multinomial(
            theta=self.theta_tg0,
            N=self.timepoint_counts_total,
            shape=timepoint_counts.shape,
        )


class TrpBSigmoidGrowthInitParam(_TrpBInitParam):
    """
    Models the TrpB count data from Johnston et al. using a sigmoid growth
    function to model the time-dependent increase in counts and a multinomial distribution
    to model the counts at each timepoint. This model uses the initial proportions
    as a parameter, which is then used to calculate the proportions at t > 0. The
    initial proportions are assumed to capture the `c` parameter implicitly.
    """

    def __init__(
        self,
        times: npt.NDArray[np.floating],
        starting_counts: npt.NDArray[np.integer],
        timepoint_counts: npt.NDArray[np.integer],
        r_mean_beta: float = 1.0,
        r_std_sigma: float = 0.05,
        alpha: float = 2.0,
        foldchange_beta: float = 0.1,
    ):
        # Run inherited init
        super().__init__(
            times=times,
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            r_mean_beta=r_mean_beta,
            r_std_sigma=r_std_sigma,
            alpha=alpha,
        )

        # We model the maximum foldchange in growth of the culture as an inverse
        # gamma distribution plus 1.
        self.max_foldchange = dms_components.Exponential(
            beta=foldchange_beta, shape=(self.n_variants,)
        ) + dms_components.Constant(1.0, togglable=False)

        # Get the final abundances. The final abundances are the initial abundances
        # scaled by the maximum foldchange for that variant.
        self.A = self.theta_t0 * self.max_foldchange  # pylint: disable=invalid-name

        # What are our proportions at t > 0?
        self.abundances_tg0 = dms_components.SigmoidGrowthInitParametrization(
            t=self.tg0,
            A=self.A,
            r=self.r,
            x0=self.theta_t0,
            shape=(self.n_replicates, self.n_timepoints - 1, self.n_variants),
        )
        self.theta_tg0 = dms_ops.normalize(self.abundances_tg0)

        # Model the counts data.
        self.timepoint_counts = dms_components.Multinomial(
            theta=self.theta_tg0,
            N=self.timepoint_counts_total,
            shape=timepoint_counts.shape,
        )


class TrpBSigmoidGrowth(TrpBBaseGrowthModel):
    """
    Identical to `TrpBSigmoidGrowthUniformA`, but with the `A` parameter modeled
    as a Dirichlet distribution. This allows the maximum carrying capacity to be
    different for each variant, but still assumes that the maximum carrying
    capacity is the same across replicates.
    """

    def __init__(
        self,
        times: npt.NDArray[np.floating],
        starting_counts: npt.NDArray[np.integer],
        timepoint_counts: npt.NDArray[np.integer],
        r_mean_beta: float = 1.0,
        r_std_sigma: float = 0.05,
        c_mean_alpha: float = 4.0,
        c_mean_beta: float = 8.0,
        A_alpha: float = 0.25,  # pylint: disable=invalid-name
        c_std_sigma: float = 0.05,
    ):
        # Run inherited init
        super().__init__(
            times=times,
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            r_mean_beta=r_mean_beta,
            r_std_sigma=r_std_sigma,
        )
        # We have an additional parameter for the sigmoid growth model, `c`, which
        # defines the time at which the growth rate is half of its maximum value.
        # We know that this has to be greater than 0, but is unlikely to be close
        # to 0. Also, we're going to assume that differences in growth conditions
        # could allow for differences in the time at which the maximum growth rate
        # is reached, so we will model different values of `c` for each replicate
        # using the Gamma distribution.
        self.c_mean = dms_components.Gamma(alpha=c_mean_alpha, beta=c_mean_beta)
        self.c_std = dms_components.HalfNormal(sigma=c_std_sigma)
        self.c = dms_components.Normal(
            mu=self.c_mean,
            sigma=self.c_std,
            shape=(self.n_replicates, 1, 1),
        )

        # Get the A values values from the dirichlet distribution. We scale the
        # values by the total number of variants for numerical stability.
        self.A = (  # pylint: disable=invalid-name
            dms_components.Dirichlet(alpha=A_alpha, shape=(self.n_variants,))
            * self.n_variants
        )

        # What are the proportions at t = 0? Everything starts from the same culture
        # at t = 0, so the proportions should be the same regardless of replicate.
        # We will thus backcalculate to the proportions at t = 0 using the MEAN
        # rate and MEAN c rather than the replicate-specific values.
        self.raw_abundances_t0 = dms_components.SigmoidGrowth(
            A=self.A,
            r=self.r_mean,
            t=dms_components.Constant(0.0, togglable=False),
            c=self.c_mean,
            shape=(self.n_variants,),
        )
        self.theta_t0 = dms_ops.normalize(self.raw_abundances_t0)

        # For t > 0, we can have values of `c` that are different for each replicate.
        # and `r` that are different for each variant. We will use the replicate-specific
        self.raw_abundances_tg0 = dms_components.SigmoidGrowth(
            A=self.A,
            r=self.r,
            t=self.tg0,
            c=self.c,
            shape=(self.n_replicates, self.n_timepoints - 1, self.n_variants),
        )
        self.theta_tg0 = dms_ops.normalize(self.raw_abundances_tg0)

        # Model the counts data.
        self.starting_counts = dms_components.Multinomial(
            theta=self.theta_t0,
            N=self.starting_counts_total,
            shape=(self.n_variants,),
        )
        self.timepoint_counts = dms_components.Multinomial(
            theta=self.theta_tg0,
            N=self.timepoint_counts_total,
            shape=timepoint_counts.shape,
        )
