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


class BaseGrowthModel(dms.Model):
    """Base class for all TrpB growth models."""

    def __init__(
        self,
        times: npt.NDArray[np.floating],
        starting_counts: npt.NDArray[np.integer],
        timepoint_counts: npt.NDArray[np.integer],
        alpha: float = 0.75,
        **hyperparameters,
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

        # Every model has a theta_t0 parameter, which gives the starting proportions
        # at the beginning of the experiment. We assume that its possible values
        # are a fundamental property of the variant; hence, we define one per variant.
        self.theta_t0 = dms_components.Dirichlet(alpha=alpha, shape=(self.n_variants,))

        # Define non-base parameters
        self._define_non_base_parameters(**hyperparameters)

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
    def _define_non_base_parameters(self, **hyperparameters):
        """
        Define any non-base parameters for the model. This should be implemented
        in subclasses to define any additional parameters needed for the model.
        """

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
        return cls(**dataset, **kwargs)  # pylint: disable=unexpected-keyword-arg

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


class ExponentialMixIn:
    """Mix-in class for models that use an exponential growth function."""

    def _define_growth_function(self) -> dms_components.ExponentialGrowth:
        """
        Defines the growth function for the model. This should return a
        TransformedParameter object that defines the growth function.
        """
        # pylint: disable=no-member
        return dms_components.ExponentialGrowth(
            t=self.tg0,
            A=self.theta_t0,
            r=self.r,
            shape=(self.n_replicates, self.n_timepoints - 1, self.n_variants),
        )


class SigmoidMixIn:
    """Mix-in class for models that use a sigmoid growth function."""

    def __init__(
        self,
        times: npt.NDArray[np.floating],
        starting_counts: npt.NDArray[np.integer],
        timepoint_counts: npt.NDArray[np.integer],
        alpha: float = 0.75,
        c_alpha: float = 4.0,
        c_beta: float = 8.0,
        **hyperparameters,
    ):
        # Define the inflection point of the sigmoid function
        self.c = dms_components.Gamma(alpha=c_alpha, beta=c_beta)

        # Run inherited init
        super().__init__(
            times=times,
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            alpha=alpha,
            **hyperparameters,
        )

    def _define_growth_function(
        self,
    ) -> dms_components.SigmoidGrowthInitParametrization:
        """
        Defines the growth function for the model. This should return a
        TransformedParameter object that defines the growth function.
        """
        # pylint: disable=no-member
        return dms_components.SigmoidGrowthInitParametrization(
            t=self.tg0,
            x0=self.theta_t0,
            r=self.r,
            c=self.c,
            shape=(self.n_replicates, self.n_timepoints - 1, self.n_variants),
        )


class BaseGammaInvRate(BaseGrowthModel):
    """
    Base model for all TrpB growth models that parametrize the inverse of the mean
    growth rate as gamma distributed and the individual growth rates as exponential.
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

    def _define_non_base_parameters(  # pylint: disable=arguments-differ
        self, inv_r_alpha: float, inv_r_beta: float
    ):
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


class BaseFoldChangeRate(BaseGrowthModel):
    """
    Base model for all TrpB growth models that parametrize the growth rate as a
    as a fold-change of some mean growth rate.
    """

    def __init__(
        self,
        times: npt.NDArray[np.floating],
        starting_counts: npt.NDArray[np.integer],
        timepoint_counts: npt.NDArray[np.integer],
        alpha: float = 0.75,
        log_foldchange_sigma_sigma: float = 0.1,
        **hyperparameters,
    ):
        # Run inherited init
        super().__init__(
            times=times,
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            alpha=alpha,
            log_foldchange_sigma_sigma=log_foldchange_sigma_sigma,
            **hyperparameters,
        )

    @abstractmethod
    def _define_growth_distribution(self, **hyperparameters):
        """
        Define the growth rate distribution for the model. This should return a
        Parameter object.
        """

    def _define_non_base_parameters(  # pylint: disable=arguments-differ
        self, log_foldchange_sigma_sigma: float, **hyperparameters
    ):

        # Define the typical growth rate
        self.r_raw = self._define_growth_distribution(**hyperparameters)

        # The error on the log of the fold-change is modeled as a half-normal distribution.
        # We assume homoscedasticity in fold-change error
        self.log_foldchange_sigma = dms_components.HalfNormal(
            sigma=log_foldchange_sigma_sigma
        )

        # The fold-change is modeled as a log-normal distribution (i.e., the log
        # of the fold-change is normally distributed) with a mean of 0 (i.e., the
        # typical fold-change is 1).
        self.fold_change = dms_components.LogNormal(
            mu=dms_components.Constant(0.0, togglable=False),
            sigma=self.log_foldchange_sigma,
            shape=(self.n_replicates, 1, self.n_variants),
        )

        # The rate is then the product of the mean rate and the fold-change
        self.r = self.r_raw * self.fold_change


class BaseExponentialRate(BaseFoldChangeRate):
    """
    Base model for all TrpB growth models that parametrize the mean growth rate
    as exponential distributed and the individual growth rates as being derived
    from some fold-change of the mean.
    """

    def __init__(
        self,
        times: npt.NDArray[np.floating],
        starting_counts: npt.NDArray[np.integer],
        timepoint_counts: npt.NDArray[np.integer],
        alpha: float = 0.75,
        log_foldchange_sigma_sigma: float = 0.1,
        beta: float = 1.0,
    ):
        # Run inherited init
        super().__init__(
            times=times,
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            alpha=alpha,
            log_foldchange_sigma_sigma=log_foldchange_sigma_sigma,
            beta=beta,
        )

    def _define_growth_distribution(  # pylint: disable=arguments-differ
        self, beta: float
    ):
        # The mean growth rate is modeled as an exponential distribution
        return dms_components.Exponential(beta=beta, shape=(self.n_variants,))


class BaseLomaxRate(BaseFoldChangeRate):
    """
    Base model for all TrpB growth models that parametrize the mean growth rate
    as Lomax (Paretto Type II) distributed and the individual growth rates as being
    derived from some fold-change of the mean. This is similar to the exponential
    model, but allows for a heavier tail.
    """

    def __init__(
        self,
        times: npt.NDArray[np.floating],
        starting_counts: npt.NDArray[np.integer],
        timepoint_counts: npt.NDArray[np.integer],
        alpha: float = 0.75,
        log_foldchange_sigma_sigma: float = 0.1,
        lambda_: float = 1.0,
        lomax_alpha: float = 2.5,
    ):
        # Run inherited init
        super().__init__(
            times=times,
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            alpha=alpha,
            log_foldchange_sigma_sigma=log_foldchange_sigma_sigma,
            lambda_=lambda_,
            lomax_alpha=lomax_alpha,
        )

    def _define_growth_distribution(  # pylint: disable=arguments-differ
        self, lambda_: float, lomax_alpha: float
    ):
        return dms_components.Lomax(
            lambda_=lambda_, alpha=lomax_alpha, shape=(self.n_variants,)
        )


class GammaInvRateExponGrowth(ExponentialMixIn, BaseGammaInvRate):
    """
    Models the TrpB count data from Johnston et al. using an exponential growth
    function to model the time-dependent increase in counts and a multinomial distribution
    to model the counts at each timepoint. The prior on the inverse growth rate is
    gamma distributed.
    """


class GammaInvRateSigmoidGrowth(SigmoidMixIn, BaseGammaInvRate):
    """Gamma-distributed inverse rate parameter with a sigmoid growth function."""


class ExponRateExponGrowth(ExponentialMixIn, BaseExponentialRate):
    """
    Models the TrpB count data from Johnston et al. using an exponential growth
    function to model the time-dependent increase in counts and a multinomial distribution
    to model the counts at each timepoint. The prior on the growth rate is
    exponential.
    """


class ExponRateSigmoidGrowth(SigmoidMixIn, BaseExponentialRate):
    """Exponential-distributed rate parameter with a sigmoid growth function."""


class LomaxRateExponGrowth(ExponentialMixIn, BaseLomaxRate):
    """
    Models the TrpB count data from Johnston et al. using an exponential growth
    function to model the time-dependent increase in counts and a multinomial distribution
    to model the counts at each timepoint. The prior on the growth rate is
    Lomax-distributed.
    """


class LomaxRateSigmoidGrowth(SigmoidMixIn, BaseLomaxRate):
    """Lomax-distributed rate parameter with a sigmoid growth function."""
