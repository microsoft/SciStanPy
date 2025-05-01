"""Holds code relevant for the TrpB datasets."""

from abc import abstractmethod

import numpy as np
import numpy.typing as npt
import pandas as pd

import dms_stan.model as dms
import dms_stan.model.components as dms_components
import dms_stan.operations as dms_ops


def load_trpb_dataset(filepath: str) -> dict[str, npt.NDArray | list[str]]:
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
        "variants": combo_order,
    }


class TrpBBaseGrowthModel(dms.Model):
    """Base class for all TrpB growth models."""

    def __init__(
        self,
        times: npt.NDArray[np.floating],
        starting_counts: npt.NDArray[np.integer],
        timepoint_counts: npt.NDArray[np.integer],
        alpha: float = 0.75,
        inv_r_alpha: float = 2.5,
        inv_r_beta: float = 0.5,
    ):

        # Set the default data
        super().__init__(
            default_data={
                "starting_counts": starting_counts,
                "timepoint_counts": timepoint_counts,
            }
        )

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

        # Total number of counts is always the same
        self.starting_counts_total = dms_components.Constant(
            starting_counts.sum(), togglable=False
        )
        self.timepoint_counts_total = dms_components.Constant(
            timepoint_counts.sum(axis=-1, keepdims=True), togglable=False
        )

        # Also define time as a constant
        self.tg0 = dms_components.Constant(times[None, 1:, None], togglable=False)

        # Every variant has a fundamental "rate" which will be the same across
        # experiments. This rate is modeled by the Gamma distribution and is the
        # inverse of the growth rate (so that we can use it as the exponential rate
        # parameter).
        self.inv_r_mean = dms_components.Gamma(
            alpha=inv_r_alpha,
            beta=inv_r_beta,
            shape=(self.n_variants,),
        )

        # The inverse rate is the beta parameter for the exponential distributions
        # describing the growth rates in each experiment.
        self.r = dms_components.Exponential(
            beta=self.inv_r_mean,
            shape=(self.n_replicates, 1, self.n_variants),
        )

        # Every model has a theta_t0 parameter, which gives the starting proportions
        # at the beginning of the experiment. We assume that its possible values
        # are a fundamental property of the variant; hence, we define one per variant.
        self.theta_t0 = dms_components.Dirichlet(alpha=alpha, shape=(self.n_variants,))

        # Define the growth function
        self.raw_abundances = self._define_growth_function()
        self.theta_tg0 = dms_ops.normalize(self.raw_abundances)

        # The counts are modeled as multinomial distributions
        self.starting_counts = dms_components.Multinomial(
            theta=self.theta_t0,
            N=self.starting_counts_total,
            shape=(self.n_variants,),
        )
        self.timepoint_counts = dms_components.Multinomial(
            theta=self.theta_tg0,
            N=self.timepoint_counts_total,
            shape=(self.n_replicates, self.n_timepoints - 1, self.n_variants),
        )

    @abstractmethod
    def _define_growth_function(self) -> dms_components.TransformedParameter:
        """
        Define the growth function for the model. This should return a
        TransformedParameter object that defines the growth function.
        """

    @classmethod
    def from_data_file(cls, filepath: str, **kwargs):
        """
        Load a TrpB dataset from Johnston et al. and create a model from it.
        """
        # Load the dataset and remove the 'variants'
        dataset = load_trpb_dataset(filepath)
        dataset.pop("variants")

        # Build the model
        return cls(**dataset, **kwargs)

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


class TrpBExponentialGrowth(TrpBBaseGrowthModel):
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
        alpha: float = 0.75,
        inv_r_alpha: float = 7.0,
        inv_r_beta: float = 1.0,
    ):
        # Run inherited init
        super().__init__(
            times=times,
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            alpha=alpha,
            inv_r_alpha=inv_r_alpha,
            inv_r_beta=inv_r_beta,
        )

    def _define_growth_function(self) -> dms_components.ExponentialGrowth:
        """
        Defines the growth function for the model. This should return a
        TransformedParameter object that defines the growth function.
        """
        return dms_components.ExponentialGrowth(
            t=self.tg0,
            A=self.theta_t0,
            r=self.r,
            shape=(self.n_replicates, self.n_timepoints - 1, self.n_variants),
        )


class TrpBSigmoidGrowth(TrpBBaseGrowthModel):
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
        alpha: float = 0.75,
        inv_r_alpha: float = 2.5,
        inv_r_beta: float = 0.5,
        c_mean_alpha: float = 4.0,
        c_mean_beta: float = 8.0,
        c_sigma_sigma: float = 0.05,
    ):
        # We have an additional parameter for the sigmoid growth model, `c`, which
        # defines the time at which the growth rate is half of its maximum value.
        # We know that this has to be greater than 0, but is unlikely to be close
        # to 0. Also, we're going to assume that differences in growth conditions
        # could allow for differences in the time at which the maximum growth rate
        # is reached, so we will model different values of `c` for each replicate
        # using the Gamma distribution.
        self.c_mean = dms_components.Gamma(alpha=c_mean_alpha, beta=c_mean_beta)
        self.c_std = dms_components.HalfNormal(sigma=c_sigma_sigma)
        self.c = dms_components.Normal(
            mu=self.c_mean,
            sigma=self.c_std,
            shape=(timepoint_counts.shape[0], 1, 1),
        )

        # Run inherited init
        super().__init__(
            times=times,
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            alpha=alpha,
            inv_r_alpha=inv_r_alpha,
            inv_r_beta=inv_r_beta,
        )

    def _define_growth_function(
        self,
    ) -> dms_components.SigmoidGrowthInitParametrization:
        return dms_components.SigmoidGrowthInitParametrization(
            t=self.tg0,
            x0=self.theta_t0,
            r=self.r,
            c=self.c,
            shape=(self.n_replicates, self.n_timepoints - 1, self.n_variants),
        )
