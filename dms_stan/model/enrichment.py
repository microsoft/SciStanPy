"""Holds DMS Stan models used for modeling enrichment assays."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt

import dms_stan.operations as dms_ops

from dms_stan.model.components import constants, parameters
from dms_stan.model.model import Model


def is_broadcastable(*shapes):
    """True if inputs are broadcastable. False otherwise."""
    try:
        _ = np.broadcast_shapes(*shapes)
        return True
    except ValueError:
        return False


class MetaEnrichment(type):
    """Metaclass for the enrichment models. This is responsible for assigning the
    appropriate growth function and shape checks to the model classes."""

    def __new__(
        mcs,
        name,
        bases,
        namespace: dict[str, Any],
        growth_func: Literal["exponential", "logistic"],
        rate_dist: Literal["gamma", "exponential", "lomax"],
        include_times: bool,
        biological_replicates: bool,
        sequence_replicates: bool,
        include_od: bool,
    ) -> type[Model]:
        """
        Prepare the namespace for the model class. This is where we build the appropriate
        __init__ method for the model class based on the provided parameters.
        """
        # The only allowed base class is Model
        if not bases or bases[0] is not Model:
            raise TypeError(
                "Enrichment models must inherit from Model. "
                f"Got {bases[0].__name__ if bases else 'None'} instead."
            )
        if len(bases) > 1:
            raise TypeError(
                "Enrichment models must inherit from Model only. "
                f"Got {', '.join(base.__name__ for base in bases)} instead."
            )

        # Get the base __init__ method for the model class
        namespace["base_init"] = mcs.set_base_init(
            include_times=include_times,
            biological_replicates=biological_replicates,
            sequence_replicates=sequence_replicates,
        )

        # We need to set the times
        namespace["init_times"] = mcs.set_times(
            include_times=include_times,
            biological_replicates=biological_replicates,
            sequence_replicates=sequence_replicates,
        )

        # Now the rate distribution, fold change, and growth function
        namespace["init_rate_distribution"] = mcs.set_rate_distribution(
            rate_distribution=rate_dist,
            biological_replicates=biological_replicates,
        )
        namespace["init_growth_noise"] = mcs.set_growth_noise(
            rate_distribution=rate_dist, biological_replicates=biological_replicates
        )
        namespace["init_growth_function"] = mcs.set_growth_function(
            growth_function=growth_func,
            include_times=include_times,
            biological_replicates=biological_replicates,
        )

        # Next, set the counts. This involves normalizing the raw abundances first
        namespace["init_abundance_normalization"] = mcs.set_normalization(
            include_od=include_od
        )
        namespace["init_counts"] = mcs.set_counts()

        # Next, set the OD distribution if needed
        namespace["init_od"] = mcs.set_od_distribution(include_od=include_od)

        # Finally, set the __init__ method for the model class
        namespace["__init__"] = mcs.set_init(
            growth_func=growth_func,
            rate_dist=rate_dist,
            include_times=include_times,
            biological_replicates=biological_replicates,
            include_od=include_od,
        )

        # Run the base new method to create the class
        return super().__new__(mcs, name, bases, namespace)

    @classmethod
    def set_init(
        mcs,
        growth_func: Literal["exponential", "logistic"],
        rate_dist: Literal["gamma", "exponential", "lomax"],
        include_times: bool,
        biological_replicates: bool,
        include_od: bool,
    ):
        """
        Runs all other initialization methods to complete object initialization.
        """
        # Determine the allowed and required keywords for the model class
        required_keywords = ["starting_counts", "timepoint_counts"]
        allowed_keywords = []

        # Different keywords for different growth functions (no keywords for exponential)
        if growth_func == "logistic":
            allowed_keywords.extend(["c_alpha", "c_beta"])

        # Different keywords for different rate distributions
        if rate_dist == "gamma":
            allowed_keywords.extend(["inv_r_alpha", "inv_r_beta"])
        elif rate_dist == "exponential":
            allowed_keywords.append("beta")
        elif rate_dist == "lomax":
            allowed_keywords.extend(["lambda_", "lomax_alpha"])

        # If we have times, we need to include keywords for them
        if include_times:
            required_keywords.append("times")

        # If biological replicates and either the exponential or lomax rate distribution
        # is used, we add the `r_sigma_sigma` keyword.
        if biological_replicates and rate_dist in {"exponential", "lomax"}:
            allowed_keywords.append("r_sigma_sigma")

        # If we have ODs, we need to include keywords for them
        if include_od:
            required_keywords.extend(["starting_od", "timepoint_od"])
            allowed_keywords.extend(
                [
                    "conversion_factor_mean",
                    "conversion_factor_std",
                    "od_measurement_error",
                ]
            )

        # Convert allowed and required keywords to sets
        required_keywords = frozenset(required_keywords)
        allowed_keywords = frozenset(allowed_keywords) | required_keywords

        # Build the __init__ method for the model class
        def __init__(self, **kwargs):
            """Initializes the enrichment model with the provided parameters."""
            # Check that all required keywords are present
            if missing_keywords := required_keywords - kwargs.keys():
                raise ValueError(
                    f"Missing required keyword arguments: {', '.join(missing_keywords)}"
                )

            # Make sure we don't have any unexpected keywords
            if unexpected_keywords := kwargs.keys() - allowed_keywords:
                raise ValueError(
                    f"Unexpected keyword arguments: {', '.join(unexpected_keywords)}"
                )

            # Run the various initialization methods
            self.base_init(**kwargs)
            self.init_times(**kwargs)
            self.init_rate_distribution(**kwargs)
            self.init_growth_noise(**kwargs)
            self.init_growth_function(**kwargs)
            self.init_abundance_normalization()
            self.init_counts()
            self.init_od(**kwargs)

        return __init__

    @classmethod
    def set_base_init(
        mcs, include_times: bool, biological_replicates: bool, sequence_replicates: bool
    ):
        """Defines the portions of the init method that is shared by all enrichment models."""

        def validate_count_arrays(
            starting_counts: npt.NDArray[np.int64],
            timepoint_counts: npt.NDArray[np.int64],
            kwargs: dict[str, Any],
        ) -> None:
            """Confirms the shape of the starting and timepoint counts arrays depending
            on whether we are including times and biological replicates."""
            # The starting counts must be broadcastable to the timepoint counts
            if not is_broadcastable(starting_counts.shape, timepoint_counts.shape):
                raise ValueError(
                    "The starting counts must be broadcastable to the timepoint "
                    f"counts. Got {starting_counts.shape} and {timepoint_counts.shape}."
                )

            # The number of dimensions of timepoint counts depends on whether we
            # are including times and whether we have replicates
            if timepoint_counts.ndim != (
                expected_dims := sum(
                    [
                        include_times,
                        biological_replicates,
                        sequence_replicates,
                        1,
                    ]  # +1 for variants
                )
            ):
                raise ValueError(
                    f"With include_times={include_times}, biological_replicates="
                    f"{biological_replicates}, sequence_replicates={sequence_replicates}, "
                    f"the timepoint counts must be {expected_dims}D. Got "
                    f"{timepoint_counts.ndim}D."
                )

            # If we have times, the second-to-last dimension of the timepoint counts
            # should be the number of times
            if include_times:
                if timepoint_counts.shape[-2] != len(kwargs["times"]):
                    raise ValueError(
                        "The second-to-last dimension of the timepoint counts should "
                        f"match the length of times. Got {timepoint_counts.shape[-2]} "
                        f"and {len(kwargs['times'])}."
                    )

        def set_shapes(
            timepoint_counts: npt.NDArray[np.floating],
        ) -> tuple[tuple[int, ...], tuple[int, ...]]:
            """
            Sets the shape of the r and log_theta_tg0 parameters based on the timepoint
            counts.
            """
            # Base shape matches that of the timepoint counts.
            r_shape = list(timepoint_counts.shape)
            tg0_shape = r_shape.copy()

            # If we have times, then we need a singleton dimension at the times
            # index for the rate
            if include_times:
                r_shape[-2] = 1

            # If we have sequence replicates, we need a singleton dimension at the
            # sequence replicates index for both r and log_theta_tg0
            if sequence_replicates:
                r_shape[-3] = 1
                tg0_shape[-3] = 1

            return tuple(r_shape), tuple(tg0_shape)

        # Define the init code shared by all enrichment models.
        def base_init(
            self,
            *,
            starting_counts: npt.NDArray[np.int64],
            timepoint_counts: npt.NDArray[np.int64],
            alpha: float = 0.1,
            **kwargs,
        ):
            # Validate the shape of the starting and timepoint counts arrays
            validate_count_arrays(
                starting_counts=starting_counts,
                timepoint_counts=timepoint_counts,
                kwargs=kwargs,
            )

            # Initialize the base class.
            Model.__init__(
                self,
                default_data={
                    "starting_counts": starting_counts,
                    "timepoint_counts": timepoint_counts,
                },
            )

            # Get the number of variants and the shapes of `r` and `log_theta_tg0`
            self.n_variants = starting_counts.shape[-1]
            self.r_shape, self.log_theta_tg0_shape = set_shapes(timepoint_counts)

            # Set the total number of starting and ending counts by replicate
            self.total_starting_counts = constants.Constant(
                starting_counts.sum(axis=-1, keepdims=True), togglable=False
            )
            self.total_timepoint_counts = constants.Constant(
                timepoint_counts.sum(axis=-1, keepdims=True), togglable=False
            )

            # Get the shape of the proportions. This is identical to the shape of
            # the counts EXCEPT for the sequence replicates dimension, if present
            theta_t0_shape = list(starting_counts.shape)
            if sequence_replicates and len(theta_t0_shape) > 2:
                theta_t0_shape[-3] = 1

            # Starting proportions are Exp-Dirichlet distributed.
            self.log_theta_t0 = parameters.ExpDirichlet(
                alpha=alpha, shape=tuple(theta_t0_shape)
            )

        return base_init

    @classmethod
    def set_times(
        mcs,
        include_times: bool,
        biological_replicates: bool,
        sequence_replicates: bool = False,
    ):
        """Sets the `time` attribute if needed"""

        # pylint: disable=unused-argument
        def null_init_times(self, **kwargs) -> None:
            """Does nothing"""

        def init_times(self, times, **kwargs) -> None:

            # If times are provided, they should be a 1D array.
            if times.ndim != 1:
                raise ValueError("Expected 'times' to be a 1D array.")

            # Add the appropriate number of dimensions depending on whether we have
            # replicates.
            times = times[
                (None,) * sum([biological_replicates, sequence_replicates])
                + (slice(None), None)
            ]

            # Normalize and record times if they are provided
            self.times = constants.Constant(times / times.max(), togglable=False)

        # Use the appropriate function for initializing times
        if include_times:
            return init_times
        return null_init_times

    @classmethod
    def set_rate_distribution(
        mcs,
        rate_distribution: Literal["gamma", "exponential", "lomax"],
        biological_replicates: bool,
    ):
        """Sets the rate distribution for the model."""

        def init_gamma_inv_rate(
            self,
            *,
            inv_r_alpha: float = 7.0,
            inv_r_beta: float = 1.0,
            **kwargs,  # pylint: disable=unused-argument
        ) -> None:

            # Every variant has a fundamental "rate" which will be the same across
            # experiments. This rate is modeled by the Gamma distribution and is the
            # inverse of the growth rate (so that we can use it as the exponential rate
            # parameter).
            if biological_replicates:
                varname = "inv_r_mean"
                distclass = parameters.Gamma
            else:
                varname = "r"
                distclass = parameters.InverseGamma

            # Set the inverse growth rate distribution
            setattr(
                self,
                varname,
                distclass(alpha=inv_r_alpha, beta=inv_r_beta, shape=(self.n_variants,)),
            )

        def init_exponential_rate(
            self, *, beta: float = 1.0, **kwargs  # pylint: disable=unused-argument
        ) -> None:

            # We stay in the log space for hierarchical models, but not for non-hierarchical
            # models
            if biological_replicates:
                varname = "log_r_mean"
                distclass = parameters.ExpExponential
            else:
                varname = "r"
                distclass = parameters.Exponential

            # Set the growth rate distribution
            setattr(self, varname, distclass(beta=beta, shape=(self.n_variants,)))

        def init_lomax_rate(
            self,
            *,
            lambda_: float = 1.0,
            lomax_alpha: float = 2.5,
            **kwargs,  # pylint: disable=unused-argument
        ) -> None:

            # We stay in the log space for hierarchical models, but not for non-hierarchical
            # models
            if biological_replicates:
                varname = "log_r_mean"
                distclass = parameters.ExpLomax
            else:
                varname = "r"
                distclass = parameters.Lomax

            # Set the growth rate distribution
            setattr(
                self,
                varname,
                distclass(lambda_=lambda_, alpha=lomax_alpha, shape=(self.n_variants,)),
            )

        # Return the appropriate initialization method based on the rate distribution
        if rate_distribution == "gamma":
            return init_gamma_inv_rate
        if rate_distribution == "exponential":
            return init_exponential_rate
        if rate_distribution == "lomax":
            return init_lomax_rate
        raise ValueError("Unsupported rate distribution.")

    @classmethod
    def set_growth_noise(
        mcs,
        rate_distribution: Literal["gamma", "exponential", "lomax"],
        biological_replicates: bool,
    ):
        """Sets the fold-change that we observe"""

        def null_init_growth_noise(
            self, **kwargs  # pylint: disable=unused-argument
        ) -> None:
            """Does nothing for non-hierarchical models."""
            return

        def init_gamma_inv_rate_growth_noise(
            self, **kwargs  # pylint: disable=unused-argument
        ) -> None:
            """Builds 'r' from 'inv_r_mean'"""
            self.r = parameters.Exponential(beta=self.inv_r_mean, shape=self.r_shape)

        def init_exp_lomax_rate_growth_noise(
            self,
            r_sigma_sigma: float = 0.01,
            **kwargs,  # pylint: disable=unused-argument
        ) -> None:
            """Builds 'r' from 'log_r_mean'"""

            # Get the sigma for the log fold-change
            self.r_sigma = parameters.HalfNormal(sigma=r_sigma_sigma)

            # Calculate r
            self.r = parameters.Normal(
                mu=dms_ops.exp(self.log_r_mean), sigma=self.r_sigma, shape=self.r_shape
            )

        # Do nothing for non-hierarchical models
        if not biological_replicates:
            return null_init_growth_noise

        # Return the appropriate initialization method based on the rate distribution
        if rate_distribution == "gamma":
            return init_gamma_inv_rate_growth_noise
        elif rate_distribution in {"exponential", "lomax"}:
            return init_exp_lomax_rate_growth_noise
        raise ValueError("Unsupported rate distribution.")

    @classmethod
    def set_growth_function(
        mcs,
        growth_function: Literal["exponential", "logistic"],
        include_times: bool,
        biological_replicates: bool,
    ):
        """Sets the growth function for the model."""

        # Define the different init functions
        def init_exponential_growth(
            self, **kwargs  # pylint: disable=unused-argument
        ) -> None:
            """Initializes the binary exponential growth function."""
            # Different growth functions depending on whether we have times or not
            if include_times:
                self.raw_abundances_tg0 = dms_ops.log_exponential_growth(
                    t=self.times,
                    log_A=self.log_theta_t0,
                    r=self.r,
                    shape=self.log_theta_tg0_shape,
                )
            else:
                self.raw_abundances_tg0 = dms_ops.binary_log_exponential_growth(
                    log_A=self.log_theta_t0,
                    r=self.r,
                    shape=self.log_theta_tg0_shape,
                )

        def init_logistic_growth(
            self,
            c_alpha: float = 4.0,
            c_beta: float = 8.0,
            **kwargs,  # pylint: disable=unused-argument
        ) -> None:
            """Initializes the binary logistic growth function."""
            # Get the shape of `c`. If we have biological replicates, We have as
            # many values for `c` as we do biological replicates.
            if biological_replicates:
                c_shape = [1] * self.default_data["timepoint_counts"].ndim
                c_shape[0] = self.default_data["timepoint_counts"].shape[0]
                if all(dim == 1 for dim in c_shape):
                    c_shape = ()
                else:
                    c_shape = tuple(c_shape)
            else:
                c_shape = ()

            # Define c.
            self.c = parameters.Gamma(alpha=c_alpha, beta=c_beta, shape=c_shape)

            # Growth is different depending on whether we have times or not
            self.raw_abundances_tg0 = dms_ops.log_sigmoid_growth_init_parametrization(
                t=(
                    self.times
                    if include_times
                    else constants.Constant(1.0, togglable=False)
                ),
                log_x0=self.log_theta_t0,
                r=self.r,
                c=self.c,
                shape=self.log_theta_tg0_shape,
            )

        # Return the appropriate initialization method based on the growth function
        if growth_function == "exponential":
            return init_exponential_growth
        if growth_function == "logistic":
            return init_logistic_growth
        raise ValueError("Unsupported growth function.")

    @classmethod
    def set_normalization(mcs, include_od: bool):
        """Sets the normalization of the raw abundances."""

        # If we are including ODs, we have an additional transformed parameter in
        # the normalization coefficient
        def init_od_normalization(self) -> None:
            """Initializes the normalization of the raw abundances when ODs are included."""
            # Sum over the last dimension to get the total raw abundances, keeping
            # the last dimension
            self.total_raw_abundance_tg0 = dms_ops.logsumexp(
                self.raw_abundances_tg0, keepdims=True
            )

            # Normalize the raw abundances to get proportions at t > 0
            self.log_theta_tg0 = self.raw_abundances_tg0 - self.total_raw_abundance_tg0

        # Otherwise, we wrap everything in the log normalization function
        def init_normalization(self) -> None:
            """Initializes the normalization of the raw abundances."""
            self.log_theta_tg0 = dms_ops.normalize_log(self.raw_abundances_tg0)

        if include_od:
            return init_od_normalization
        return init_normalization

    @classmethod
    def set_counts(mcs):
        """
        Assigns the model for counts. We use the multinomial distribution parametrized
        with log proportions.
        """

        def init_counts(self) -> None:
            """Initializes modeling of counts."""
            # Counts are modeled as a multinomial distribution
            self.starting_counts = parameters.MultinomialLogTheta(
                log_theta=self.log_theta_t0,
                N=self.total_starting_counts,
                shape=self.default_data["starting_counts"].shape,
            )
            self.timepoint_counts = parameters.MultinomialLogTheta(
                log_theta=self.log_theta_tg0,
                N=self.total_timepoint_counts,
                shape=self.default_data["timepoint_counts"].shape,
            )

        return init_counts

    @classmethod
    def set_od_distribution(mcs, include_od: bool):
        """Sets the OD distribution based on whether we are including ODs."""

        # Null operation for models that do not include ODs
        def init_od_null(self, **kwargs) -> None:  # pylint: disable=unused-argument
            """Does nothing for models that do not include ODs."""
            return

        # OD distribution for models that include ODs
        def init_od(
            self,
            starting_od: npt.NDArray[np.floating],
            timepoint_od: npt.NDArray[np.floating],
            log_slope_mean: float = 0.0,
            log_slope_std: float = 1.0,
            intercept_mean: float = 0.0,
            intercept_std: float = 1.0,
            od_measurement_error: float = 0.01,
            **kwargs,  # pylint: disable=unused-argument
        ) -> None:
            """Sets the OD distribution for models that include ODs."""
            # Check OD shapes. They must be broadcastable with the first dimensions
            # of the starting and timepoint counts.
            if not is_broadcastable(
                starting_od.shape, self.starting_counts.shape[: starting_od.ndim]
            ):
                raise ValueError(
                    f"The starting OD must be broadcastable to the first {starting_od.ndim} "
                    f"dimensions of the starting counts. Got {starting_od.shape} and "
                    f"{self.starting_counts.shape}."
                )
            if not is_broadcastable(
                timepoint_od.shape, self.timepoint_counts.shape[: timepoint_od.ndim]
            ):
                raise ValueError(
                    f"The timepoint OD must be broadcastable to the first {timepoint_od.ndim} "
                    f"dimensions of the timepoint counts. Got {timepoint_od.shape} and "
                    f"{self.timepoint_counts.shape}."
                )

            # Add the appropriate number of dimensions to the ODs
            if starting_od.ndim > 0 and starting_od.ndim < self.starting_counts.ndim:
                starting_od = starting_od[
                    (...,) + (None,) * (self.starting_counts.ndim - starting_od.ndim)
                ]
            if timepoint_od.ndim < self.timepoint_counts.ndim:
                timepoint_od = timepoint_od[
                    (...,) + (None,) * (self.timepoint_counts.ndim - timepoint_od.ndim)
                ]

            # Record as default data
            self.default_data["starting_od"] = starting_od
            self.default_data["timepoint_od"] = timepoint_od

            # Get the abundance to OD slope and intercept. We don't know much about
            # the slope or the intercept, only that the slope is positive.
            self.log_abundance_to_od_slope = parameters.Normal(
                mu=log_slope_mean,
                sigma=log_slope_std,
            )
            self.od_intercept = parameters.Normal(
                mu=intercept_mean,
                sigma=intercept_std,
            )

            # Now get the stdev of the measurement error
            self.od_measurement_error = parameters.HalfNormal(
                sigma=od_measurement_error
            )

            # Model the OD at t=0. The raw abundance is always "1" at this point,
            # so we ignore it
            self.starting_od = parameters.Normal(
                mu=dms_ops.exp(self.log_abundance_to_od_slope) + self.od_intercept,
                sigma=self.od_measurement_error,
                shape=starting_od.shape,
            )

            # Model at t > 0. This is the total abundance plus the correction factor
            self.timepoint_od = parameters.Normal(
                mu=dms_ops.exp(
                    self.log_abundance_to_od_slope + self.total_raw_abundance_tg0
                )
                + self.od_intercept,
                sigma=self.od_measurement_error,
                shape=timepoint_od.shape,
            )

        # Return the appropriate initialization method based on whether we are
        # including ODs
        if include_od:
            return init_od
        return init_od_null


def hierarchical_class_factory(
    name: str,
    growth_func: Literal["exponential", "logistic"],
    rate_dist: Literal["gamma", "exponential", "lomax"],
    include_times: bool = False,
    sequence_replicates: bool = False,
    include_od: bool = False,
) -> type[Model]:
    """
    Factory function to create a hierarchical enrichment model class with the specified
    parameters.
    """
    # Create a new class with the specified parameters
    return MetaEnrichment(
        name,
        (Model,),
        {},
        growth_func=growth_func,
        rate_dist=rate_dist,
        include_times=include_times,
        biological_replicates=True,
        sequence_replicates=sequence_replicates,
        include_od=include_od,
    )


def non_hierarchical_class_factory(
    name: str,
    growth_func: Literal["exponential", "logistic"],
    rate_dist: Literal["gamma", "exponential", "lomax"],
    include_times: bool = False,
    sequence_replicates: bool = False,
    include_od: bool = False,
) -> type[Model]:
    """
    Factory function to create a non-hierarchical enrichment model class with the
    specified parameters.
    """
    # Create a new class with the specified parameters
    return MetaEnrichment(
        name,
        (Model,),
        {},
        growth_func=growth_func,
        rate_dist=rate_dist,
        include_times=include_times,
        biological_replicates=False,
        sequence_replicates=sequence_replicates,
        include_od=include_od,
    )
