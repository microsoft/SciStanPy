"""Holds DMS Stan models used for modeling enrichment assays."""

from abc import abstractmethod

import numpy as np
import numpy.typing as npt

import dms_stan.model as dms
import dms_stan.model.components as dms_components
import dms_stan.operations as dms_ops


class BaseEnrichmentModel(dms.Model):
    """Base class for all enrichment models."""

    def __init__(
        self,
        starting_counts: npt.NDArray[np.int64],
        timepoint_counts: npt.NDArray[np.int64],
        times: npt.NDArray[np.floating] | None = None,
        alpha: float = 0.75,
        **hyperparameters,
    ):
        """Initializes the PDZ3 model with starting and ending counts.

        Args:
            starting_counts (npt.NDArray[np.int64]): Starting counts for the PDZ3 dataset.
            timepoint_counts (npt.NDArray[np.int64]): Ending counts for the PDZ3 dataset.
        """
        # Register default data
        super().__init__(
            default_data={
                "starting_counts": starting_counts,
                "timepoint_counts": timepoint_counts,
            }
        )

        # The last dimension of starting and ending counts should be equivalent
        if starting_counts.shape[-1] != timepoint_counts.shape[-1]:
            raise ValueError(
                "The last dimension of starting and ending counts should be equivalent."
            )

        # Record times if they are present and normalize them to have a maximum
        # of 1
        self.times = (
            times
            if times is None
            else dms_components.Constant(times / times.max(), togglable=False)
        )

        # Set the total number of starting and ending counts by replicate
        self.total_starting_counts = dms_components.Constant(
            starting_counts.sum(axis=-1, keepdims=True), togglable=False
        )
        self.total_timepoint_counts = dms_components.Constant(
            timepoint_counts.sum(axis=-1, keepdims=True), togglable=False
        )

        # Starting proportions are Dirichlet distributed
        self.theta_t0 = dms_components.Dirichlet(
            alpha=alpha, shape=starting_counts.shape
        )

        # Define non-base parameters
        self._define_additional_parameters(**hyperparameters)

        # Calculate ending proportions
        self.raw_abundances_tg0 = self._define_growth_function()
        self.theta_tg0 = dms_ops.normalize(self.raw_abundances_tg0)

        # The counts are modeled as multinomial distributions
        self.starting_counts = dms_components.Multinomial(
            theta=self.theta_t0,
            N=self.total_starting_counts,
            shape=starting_counts.shape,
        )
        self.timepoint_counts = dms_components.Multinomial(
            theta=self.theta_tg0,
            N=self.total_timepoint_counts,
            shape=timepoint_counts.shape,
        )

    @abstractmethod
    def _define_additional_parameters(self, **hyperparameters):
        """
        Defines additional parameters for the model. This must be overridden in
        child classes and is used to define any additional parameters needed for
        the model.
        """

    @abstractmethod
    def _define_growth_function(self) -> dms_components.TransformedParameter:
        """
        Defines the transformation for the model. This must be overridden in
        child classes and is used to define the transformation between starting
        and ending abundances.
        """


class ExponentialMixIn:
    """Mixin class for exponential growth models."""

    def _define_growth_function(
        self,
    ) -> dms_components.BinaryExponentialGrowth | dms_components.ExponentialGrowth:
        # pylint: disable=no-member
        if self.times is None:
            return dms_components.BinaryExponentialGrowth(
                A=self.theta_t0,
                r=self.r,
                shape=self.default_data["timepoint_counts"].shape,
            )
        return dms_components.ExponentialGrowth(
            t=self.times,
            A=self.theta_t0,
            r=self.r,
            shape=self.default_data["timepoint_counts"].shape,
        )


class SigmoidMixIn:
    """Mixin class for sigmoid growth models."""

    def __init__(
        self,
        starting_counts: npt.NDArray[np.integer],
        timepoint_counts: npt.NDArray[np.integer],
        times: npt.NDArray[np.floating] | None = None,
        alpha: float = 0.75,
        c_alpha: float = 4.0,
        c_beta: float = 8.0,
        **hyperparameters,
    ):
        # pylint: disable=no-member
        # Initialize a `c` parameter for the sigmoid growth function. We assume
        # separate 'c' for each replicate. `c` will be a scalar if there are no
        # replicates.
        if times is None:

            # No times provided and 2D timepoint counts means we have replicates
            if timepoint_counts.ndim == 2:
                shape = (timepoint_counts.shape[0], 1)

            # No times provided and 1D timepoint counts means we have no replicates
            elif timepoint_counts.ndim == 1:
                shape = ()

            # Otherwise, we have an error
            else:
                raise ValueError(
                    "If times are not provided, the timepoint counts should be 1D "
                    "for non-hierarchical models or 2D for hierarchical models."
                )

        else:
            # Times provided and 3D timepoint counts means we have replicates
            if timepoint_counts.ndim == 3:
                shape = (timepoint_counts.shape[0], 1, 1)

            # Times provided and 2D timepoint counts means we have no replicates
            elif timepoint_counts.ndim == 2:
                shape = ()

            # Otherwise, we have an error
            else:
                raise ValueError(
                    "If times are provided, the timepoint counts should be 2D for "
                    "non-hierarchical models or 3D for hierarchical models."
                )

        self.c = dms_components.Gamma(alpha=c_alpha, beta=c_beta, shape=shape)

        # Initialize the base class
        super().__init__(
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            times=times,
            alpha=alpha,
            **hyperparameters,
        )

    def _define_growth_function(
        self,
    ) -> dms_components.SigmoidGrowthInitParametrization:
        # pylint: disable=no-member
        # Get the ending abundances
        return dms_components.SigmoidGrowthInitParametrization(
            t=(
                dms_components.Constant(1.0, togglable=False)
                if self.times is None
                else self.times
            ),
            x0=self.theta_t0,
            r=self.r,
            c=self.c,
            shape=self.default_data["timepoint_counts"].shape,
        )


class HierarchicalModel(BaseEnrichmentModel):
    """Used when we have replicates and want to model them hierarchically."""

    def __init__(
        self,
        starting_counts: npt.NDArray[np.int64],
        timepoint_counts: npt.NDArray[np.int64],
        times: npt.NDArray[np.floating] | None = None,
        alpha: float = 0.75,
        **hyperparameters,
    ):

        # If times are provided, then the timepoint counts should be 3D and have
        # the same middle dimension as the times. If times are not provided, then
        # the timepoint counts should be 2D.
        if times is not None:
            # The timepoint counts should be 3D. There must be as many timepoints
            # as the middle dimension of timepoint counts
            if timepoint_counts.ndim != 3:
                raise ValueError(
                    "If times are provided, the timepoint counts should be 3D."
                )
            if timepoint_counts.shape[1] != len(times):
                raise ValueError(
                    "The middle dimension of timepoint counts should match the length"
                    "of times."
                )

            # Correct the shape of the times array
            assert times.ndim == 1, "Times should be a 1D array."
            times = times[None, :, None]  # Add two new dimensions to times
        else:
            # The timepoint counts should be 2D. There must be as many timepoints
            # as the last dimension of timepoint counts
            if timepoint_counts.ndim != 2:
                raise ValueError(
                    "If times are not provided, the timepoint counts should be 2D."
                )

        # This model is not appropriate if there are no replicates
        if timepoint_counts.shape[0] == 1:
            raise ValueError(
                "This model is not appropriate if there are no replicates. Use the "
                "non-hierarchical model instead."
            )

        # Run inherited init
        super().__init__(
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            times=times,
            alpha=alpha,
            **hyperparameters,
        )


class BaseGammaInvRate(HierarchicalModel):
    """
    Base model for all enrichment models that parametrize the inverse of the mean
    growth rate as gamma distributed and the individual growth rates as exponential.
    """

    def __init__(
        self,
        starting_counts: npt.NDArray[np.integer],
        timepoint_counts: npt.NDArray[np.integer],
        times: npt.NDArray[np.floating] | None = None,
        alpha: float = 0.75,
        inv_r_alpha: float = 7.0,
        inv_r_beta: float = 1.0,
    ):
        # Run inherited init
        super().__init__(
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            alpha=alpha,
            times=times,
            inv_r_alpha=inv_r_alpha,
            inv_r_beta=inv_r_beta,
        )

    def _define_additional_parameters(  # pylint: disable=arguments-differ
        self, inv_r_alpha: float, inv_r_beta: float
    ):

        # Every variant has a fundamental "rate" which will be the same across
        # experiments. This rate is modeled by the Gamma distribution and is the
        # inverse of the growth rate (so that we can use it as the exponential rate
        # parameter).
        self.inv_r_mean = dms_components.Gamma(
            alpha=inv_r_alpha,
            beta=inv_r_beta,
            shape=(self.default_data["timepoint_counts"].shape[-1],),
        )

        # Get the shape of r. We need to account for possible additional dimensions
        # needed because of the times parameter.
        r_shape = list(self.default_data["timepoint_counts"].shape)
        if self.times is not None:
            r_shape[1] = 1

        # The inverse rate is the beta parameter for the exponential distributions
        # describing the growth rates in each experiment.
        self.r = dms_components.Exponential(
            beta=self.inv_r_mean,
            shape=tuple(r_shape),
        )


class BaseFoldChangeRate(HierarchicalModel):
    """
    Base model for all TrpB growth models that parametrize the growth rate as a
    as a fold-change of some mean growth rate.
    """

    def __init__(
        self,
        starting_counts: npt.NDArray[np.integer],
        timepoint_counts: npt.NDArray[np.integer],
        times: npt.NDArray[np.floating] | None = None,
        alpha: float = 0.75,
        log_foldchange_sigma_sigma: float = 0.5,
        **hyperparameters,
    ):
        # Run inherited init
        super().__init__(
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            times=times,
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

    def _define_additional_parameters(  # pylint: disable=arguments-differ
        self, log_foldchange_sigma_sigma: float, **hyperparameters
    ):

        # Define the log of typical growth rate
        self.log_r = self._define_growth_distribution(**hyperparameters)

        # The error on the log of the fold-change is modeled as a half-normal distribution.
        # We assume homoscedasticity in fold-change error
        self.log_foldchange_sigma = dms_components.HalfNormal(
            sigma=log_foldchange_sigma_sigma
        )

        # The shape of the fold-change depends on whether we are including times
        shape = list(self.default_data["timepoint_counts"].shape)
        if self.times is not None:
            shape[1] = 1

        # The fold-change is modeled as a log-normal distribution (i.e., the log
        # of the fold-change is normally distributed) with a mean of 0 (i.e., the
        # typical fold-change is 1).
        self.r = dms_components.LogNormal(
            mu=self.log_r,
            sigma=self.log_foldchange_sigma,
            shape=tuple(shape),
        )


class BaseExponentialRate(BaseFoldChangeRate):
    """
    Base model for all TrpB growth models that parametrize the mean growth rate
    as exponential distributed and the individual growth rates as being derived
    from some fold-change of the mean.
    """

    def __init__(
        self,
        starting_counts: npt.NDArray[np.integer],
        timepoint_counts: npt.NDArray[np.integer],
        times: npt.NDArray[np.floating] | None = None,
        alpha: float = 0.75,
        log_foldchange_sigma_sigma: float = 0.5,
        beta: float = 1.0,
    ):
        # Run inherited init
        super().__init__(
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            times=times,
            alpha=alpha,
            log_foldchange_sigma_sigma=log_foldchange_sigma_sigma,
            beta=beta,
        )

    def _define_growth_distribution(  # pylint: disable=arguments-differ
        self, beta: float
    ):
        # The mean growth rate is modeled as an exponential distribution
        return dms_components.ExpExponential(
            beta=beta, shape=(self.default_data["timepoint_counts"].shape[-1],)
        )


class BaseLomaxRate(BaseFoldChangeRate):
    """
    Base model for all TrpB growth models that parametrize the mean growth rate
    as Lomax (Paretto Type II) distributed and the individual growth rates as being
    derived from some fold-change of the mean. This is similar to the exponential
    model, but allows for a heavier tail.
    """

    def __init__(
        self,
        starting_counts: npt.NDArray[np.integer],
        timepoint_counts: npt.NDArray[np.integer],
        times: npt.NDArray[np.floating] | None = None,
        alpha: float = 0.75,
        log_foldchange_sigma_sigma: float = 0.5,
        lambda_: float = 1.0,
        lomax_alpha: float = 2.5,
    ):
        # Run inherited init
        super().__init__(
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            times=times,
            alpha=alpha,
            log_foldchange_sigma_sigma=log_foldchange_sigma_sigma,
            lambda_=lambda_,
            lomax_alpha=lomax_alpha,
        )

    def _define_growth_distribution(  # pylint: disable=arguments-differ
        self, lambda_: float, lomax_alpha: float
    ):
        return dms_components.ExpLomax(
            lambda_=lambda_,
            alpha=lomax_alpha,
            shape=(self.default_data["timepoint_counts"].shape[-1],),
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

    def __init__(  # Updating default values
        self,
        starting_counts: npt.NDArray[np.int64],
        timepoint_counts: npt.NDArray[np.int64],
        times: npt.NDArray[np.floating] | None = None,
        alpha: float = 0.75,
        inv_r_alpha: float = 2.5,
        inv_r_beta: float = 0.5,
    ):
        super().__init__(
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            times=times,
            alpha=alpha,
            inv_r_alpha=inv_r_alpha,
            inv_r_beta=inv_r_beta,
        )


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


class NonHierarchicalModel(BaseEnrichmentModel):
    """Used when we have no replicates."""

    def __init__(
        self,
        starting_counts: npt.NDArray[np.int64],
        timepoint_counts: npt.NDArray[np.int64],
        times: npt.NDArray[np.floating] | None = None,
        alpha: float = 0.75,
        **hyperparameters,
    ):
        # If times are provided, then the timepoint counts should be 2D and have
        # the same first dimension as the times. If times are not provided, then
        # the timepoint counts should be 1D.
        if times is not None:
            # The timepoint counts should be 2D. There must be as many timepoints
            # as the middle dimension of timepoint counts
            if timepoint_counts.ndim != 2:
                raise ValueError(
                    "If times are provided, the timepoint counts should be 2D."
                )
            if timepoint_counts.shape[0] != len(times):
                raise ValueError(
                    "The middle dimension of timepoint counts should match the length"
                    "of times."
                )

            # Correct the shape of the times array
            assert times.ndim == 1, "Times should be a 1D array."
            times = times[:, None]  # Append dimension to times
        else:
            # The timepoint counts should be 1D.
            if timepoint_counts.ndim != 1:
                raise ValueError(
                    "If times are not provided, the timepoint counts should be 1D."
                )

        # Run inherited init
        super().__init__(
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            alpha=alpha,
            times=times,
            **hyperparameters,
        )


class NonHierarchicalBaseExponentialRate(NonHierarchicalModel):
    def __init__(
        self,
        starting_counts: npt.NDArray[np.integer],
        timepoint_counts: npt.NDArray[np.integer],
        times: npt.NDArray[np.floating] | None = None,
        alpha: float = 0.75,
        beta: float = 1.0,
    ):
        # Run inherited init
        super().__init__(
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            times=times,
            alpha=alpha,
            beta=beta,
        )

    def _define_additional_parameters(  # pylint: disable=arguments-differ
        self, beta: float
    ):
        # The mean growth rate is modeled as an exponential distribution
        self.r = dms_components.Exponential(
            beta=beta, shape=(self.default_data["timepoint_counts"].shape[-1],)
        )


class NonHierarchicalBaseLomaxRate(NonHierarchicalModel):
    def __init__(
        self,
        starting_counts: npt.NDArray[np.integer],
        timepoint_counts: npt.NDArray[np.integer],
        times: npt.NDArray[np.floating] | None = None,
        alpha: float = 0.75,
        lambda_: float = 1.0,
        lomax_alpha: float = 2.5,
    ):
        # Run inherited init
        super().__init__(
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            times=times,
            alpha=alpha,
            lambda_=lambda_,
            lomax_alpha=lomax_alpha,
        )

    def _define_additional_parameters(  # pylint: disable=arguments-differ
        self, lambda_: float, lomax_alpha: float
    ):
        # The mean growth rate is modeled as a Lomax distribution
        self.r = dms_components.Lomax(
            lambda_=lambda_,
            alpha=lomax_alpha,
            shape=(self.default_data["timepoint_counts"].shape[-1],),
        )


class NonHierarchicalExponRateExponGrowth(
    ExponentialMixIn, NonHierarchicalBaseExponentialRate
):
    """
    Models the TrpB count data from Johnston et al. using an exponential growth
    function to model the time-dependent increase in counts and a multinomial distribution
    to model the counts at each timepoint. The prior on the growth rate is
    exponential.
    """


class NonHierarchicalExponRateSigmoidGrowth(
    SigmoidMixIn, NonHierarchicalBaseExponentialRate
):
    """Exponential-distributed rate parameter with a sigmoid growth function."""


class NonHierarchicalLomaxRateExponGrowth(
    ExponentialMixIn, NonHierarchicalBaseLomaxRate
):
    """
    Models the TrpB count data from Johnston et al. using an exponential growth
    function to model the time-dependent increase in counts and a multinomial distribution
    to model the counts at each timepoint. The prior on the growth rate is
    Lomax-distributed.
    """


class NonHierarchicalLomaxRateSigmoidGrowth(SigmoidMixIn, NonHierarchicalBaseLomaxRate):
    """Lomax-distributed rate parameter with a sigmoid growth function."""
