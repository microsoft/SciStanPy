"""Holds DMS Stan models used for modeling enrichment assays."""

from typing import Any, Literal

import numpy as np
import numpy.typing as npt

import dms_stan.model as dms
import dms_stan.model.components as dms_components
import dms_stan.operations as dms_ops


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
        include_od: bool,
    ) -> type[dms.Model]:
        """
        Prepare the namespace for the model class. This is where we build the appropriate
        __init__ method for the model class based on the provided parameters.
        """
        # The only allowed base class is dms.Model
        if not bases or bases[0] is not dms.Model:
            raise TypeError(
                "Enrichment models must inherit from dms.Model. "
                f"Got {bases[0].__name__ if bases else 'None'} instead."
            )
        if len(bases) > 1:
            raise TypeError(
                "Enrichment models must inherit from dms.Model only. "
                f"Got {', '.join(base.__name__ for base in bases)} instead."
            )

        # Get the base __init__ method for the model class
        namespace["base_init"] = mcs.set_base_init(
            include_times=include_times,
            biological_replicates=biological_replicates,
        )

        # We need to set the times
        namespace["init_times"] = mcs.set_times(include_times=include_times)

        # Now the rate distribution, fold change, and growth function
        namespace["init_rate_distribution"] = mcs.set_rate_distribution(
            rate_distribution=rate_dist,
            biological_replicates=biological_replicates,
        )
        namespace["init_foldchange"] = mcs.set_foldchange(
            rate_distribution=rate_dist, biological_replicates=biological_replicates
        )
        namespace["init_growth_function"] = mcs.set_growth_function(
            growth_function=growth_func,
            include_times=include_times,
            biological_replicates=biological_replicates,
        )

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
        # is used, we add the `log_foldchange_sigma_sigma` keyword.
        if biological_replicates and rate_dist in {"exponential", "lomax"}:
            allowed_keywords.append("log_foldchange_sigma_sigma")

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
            self.init_foldchange(**kwargs)
            self.init_growth_function(**kwargs)
            self.init_od(**kwargs)

        return __init__

    @classmethod
    def set_base_init(mcs, include_times: bool, biological_replicates: bool):
        """Defines the portion of the init method that is shared by all enrichment models."""

        def validate_count_arrays(
            starting_counts: npt.NDArray[np.int64],
            timepoint_counts: npt.NDArray[np.int64],
            kwargs: dict[str, Any],
        ) -> None:
            """Confirms the shape of the starting and timepoint counts arrays depending
            on whether we are including times and biological replicates."""
            # The last dimension of starting and ending counts should be equivalent
            if starting_counts.shape[-1] != timepoint_counts.shape[-1]:
                raise ValueError(
                    "The last dimension of starting and ending counts should be equivalent."
                )

            # If no times and no biological replicates, both sets of counts should
            # be 1D.
            if not include_times and not biological_replicates:
                if starting_counts.ndim != 1 or timepoint_counts.ndim != 1:
                    raise ValueError(
                        "If no times and no biological replicates, both sets of counts "
                        "should be 1D."
                    )

            # If we have times but no biological replicates, the starting counts
            # should be 1D and the timepoint counts should be 2D. The first dimension
            # is the timepoint, and the second dimension is the variant.
            elif include_times and not biological_replicates:
                if starting_counts.ndim != 1 or timepoint_counts.ndim != 2:
                    raise ValueError(
                        "If times are provided but no biological replicates, the "
                        "starting counts should be 1D and the timepoint counts should "
                        "be 2D."
                    )
                if len(kwargs["times"]) != timepoint_counts.shape[0]:
                    raise ValueError(
                        "The number of timepoints should match the first dimension "
                        "of the timepoint counts."
                    )

            # If we have biological replicates but not times, both sets of counts
            # should be 2D. Shapes should be (n_replicates, n_variants).
            elif biological_replicates and not include_times:
                if starting_counts.ndim != 2 or timepoint_counts.ndim != 2:
                    raise ValueError(
                        "If biological replicates are provided but no times, both sets "
                        "of counts should be 2D."
                    )
                if starting_counts.shape[0] != timepoint_counts.shape[0]:
                    raise ValueError(
                        "The first dimension of starting and timepoint counts should "
                        "be the same if biological replicates are provided."
                    )

            # If we have both times and biological replicates, the starting counts
            # and timepoint counts should be 3D. Shapes should be (n_replicates,
            # n_times, n_variants). This means that there will be a singleton dimension
            # for the times in the starting counts.
            elif include_times and biological_replicates:
                if starting_counts.ndim != 3 or timepoint_counts.ndim != 3:
                    raise ValueError(
                        "If both times and biological replicates are provided, both sets "
                        "of counts should be 3D."
                    )
                if starting_counts.shape[1] != 1:
                    raise ValueError(
                        "If both times and biological replicates are provided, the "
                        "starting counts should have a singleton dimension for the times."
                    )
                if timepoint_counts.shape[1] != len(kwargs["times"]):
                    raise ValueError(
                        "The second dimension of timepoint counts should match the "
                        "length of times."
                    )

        def set_r_shape(timepoint_counts: npt.NDArray[np.floating]) -> tuple[int, ...]:
            """Sets the shape of the r parameter based on the timepoint counts."""
            # Base shape matches that of the timepoint counts
            r_shape = list(timepoint_counts.shape)

            # If we have times, then we need a singleton dimension at the times
            # index
            if include_times:
                r_shape[-1] = 1

            return tuple(r_shape)

        # Define the init code shared by all enrichment models.
        # Define the base __init__ method used by all enrichment models.
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
            dms.Model.__init__(
                self,
                default_data={
                    "starting_counts": starting_counts,
                    "timepoint_counts": timepoint_counts,
                },
            )

            # Get the number of variants and the shape of `r`
            self.n_variants = starting_counts.shape[-1]
            self.r_shape = set_r_shape(timepoint_counts)

            # Set the total number of starting and ending counts by replicate
            self.total_starting_counts = dms_components.Constant(
                starting_counts.sum(axis=-1, keepdims=True), togglable=False
            )
            self.total_timepoint_counts = dms_components.Constant(
                timepoint_counts.sum(axis=-1, keepdims=True), togglable=False
            )

            # Starting proportions are Exp-Dirichlet distributed
            self.log_theta_t0 = dms_components.ExpDirichlet(
                alpha=alpha, shape=starting_counts.shape
            )

        return base_init

    @classmethod
    def set_times(mcs, include_times: bool):
        """Sets the `time` attribute if needed"""

        def null_init_times(self, **kwargs) -> None:  # pylint: disable=unused-argument
            """Does nothing"""

        def init_times(self, times, **kwargs) -> None:

            # If times are provided, they should be a 1D array.
            if times.ndim != 1:
                raise ValueError("Expected 'times' to be a 1D array.")

            # Normalize and record times if they are provided
            self.times = dms_components.Constant(times / times.max(), togglable=False)

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
                distclass = dms_components.Gamma
            else:
                varname = "r"
                distclass = dms_components.InverseGamma

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
                varname = "log_r"
                distclass = dms_components.ExpExponential
            else:
                varname = "r"
                distclass = dms_components.Exponential

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
                varname = "log_r"
                distclass = dms_components.ExpLomax
            else:
                varname = "r"
                distclass = dms_components.Lomax

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
    def set_foldchange(
        mcs,
        rate_distribution: Literal["gamma", "exponential", "lomax"],
        biological_replicates: bool,
    ):
        """Sets the fold-change that we observe"""

        def null_init_foldchange(
            self, **kwargs  # pylint: disable=unused-argument
        ) -> None:
            """Does nothing for non-hierarchical models."""
            return

        def init_gamma_inv_rate_foldchange(
            self, **kwargs  # pylint: disable=unused-argument
        ) -> None:
            """Builds 'r' from 'inv_r_mean'"""
            self.r = dms_components.Exponential(
                beta=self.inv_r_mean, shape=self.r_shape
            )

        def init_exp_lomax_rate_foldchange(
            self,
            log_foldchange_sigma_sigma: float = 0.1,
            **kwargs,  # pylint: disable=unused-argument
        ) -> None:
            """Builds 'r' from 'log_r'"""

            # Get the sigma for the log fold-change
            self.log_foldchange_sigma = dms_components.HalfNormal(
                sigma=log_foldchange_sigma_sigma
            )

            # Calculate r
            self.r = dms_ops.exp(
                dms_components.Normal(
                    mu=self.log_r,
                    sigma=log_foldchange_sigma_sigma,
                    shape=self.r_shape,
                )
            )

        # Do nothing for non-hierarchical models
        if not biological_replicates:
            return null_init_foldchange

        # Return the appropriate initialization method based on the rate distribution
        if rate_distribution == "gamma":
            return init_gamma_inv_rate_foldchange
        elif rate_distribution in {"exponential", "lomax"}:
            return init_exp_lomax_rate_foldchange
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
                self.growth_function = dms_components.LogExponentialGrowth(
                    t=self.times,
                    log_A=self.log_theta_t0,
                    r=self.r,
                    shape=self.default_data["timepoint_counts"].shape,
                )
            else:
                self.growth_function = dms_components.BinaryLogExponentialGrowth(
                    log_A=self.log_theta_t0,
                    r=self.r,
                    shape=self.default_data["timepoint_counts"].shape,
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
                c_shape = ()

            # Define c.
            self.c = dms_components.Gamma(alpha=c_alpha, beta=c_beta, shape=c_shape)

            # Growth is different depending on whether we have times or not
            self.growth_function = dms_components.LogSigmoidGrowthInitParametrization(
                t=(
                    self.times
                    if include_times
                    else dms_components.Constant(1.0, togglable=False)
                ),
                log_x0=self.log_theta_t0,
                r=self.r,
                c=self.c,
                shape=self.default_data["timepoint_counts"].shape,
            )

        # Return the appropriate initialization method based on the growth function
        if growth_function == "exponential":
            return init_exponential_growth
        if growth_function == "logistic":
            return init_logistic_growth
        raise ValueError("Unsupported growth function.")

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
            conversion_factor_mean: float = -2.3,
            conversion_factor_std: float = 1.0,
            od_measurement_error: float = 0.1,
            **kwargs,  # pylint: disable=unused-argument
        ) -> None:
            """Sets the OD distribution for models that include ODs."""
            # Check OD shapes. They should match the shapes of their respective
            # counts except in the final dimension, which should be a singleton
            if (
                starting_od.shape[:-1]
                != self.default_data["starting_counts"].shape[:-1]
            ):
                raise ValueError(
                    "The shape of starting_od should match the shape of "
                    "starting_counts except in the last dimension."
                )
            if (
                timepoint_od.shape[:-1]
                != self.default_data["timepoint_counts"].shape[:-1]
            ):
                raise ValueError(
                    "The shape of timepoint_od should match the shape of "
                    "timepoint_counts except in the last dimension."
                )
            if starting_od.shape[-1] != 1 or timepoint_od.shape[-1] != 1:
                raise ValueError(
                    "The last dimension of starting_od and timepoint_od should be a "
                    "singleton dimension."
                )

            # Get the conversion factor
            self.log_abundance_to_od = dms_components.Normal(
                mu=conversion_factor_mean,
                sigma=conversion_factor_std,
            )

            # Now get the stdev of the measurement error
            self.od_measurement_error = dms_components.HalfNormal(
                sigma=od_measurement_error
            )

            # Model the OD at t=0. This is just the correction factor exponentiated
            self.od_t0 = dms_components.Normal(
                mu=dms_ops.exp(self.log_abundance_to_od),
                sigma=self.od_measurement_error,
                shape=starting_od.shape,
            )

            # Model at t > 0. This is the abundance plus the correction factor
            self.od_tg0 = dms_components.Normal(
                mu=dms_ops.exp(self.log_abundance_to_od + self.raw_abundances_tg0),
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
    include_od: bool = False,
) -> type[dms.Model]:
    """Factory function to create a hierarchical enrichment model class with the specified parameters."""
    # Create a new class with the specified parameters
    return MetaEnrichment(
        name,
        (dms.Model,),
        {},
        growth_func=growth_func,
        rate_dist=rate_dist,
        include_times=include_times,
        biological_replicates=True,
        include_od=include_od,
    )


def non_hierarchical_class_factory(
    name: str,
    growth_func: Literal["exponential", "logistic"],
    rate_dist: Literal["gamma", "exponential", "lomax"],
    include_times: bool = False,
    include_od: bool = False,
) -> type[dms.Model]:
    """Factory function to create a non-hierarchical enrichment model class with the specified parameters."""
    # Create a new class with the specified parameters
    return MetaEnrichment(
        name,
        (dms.Model,),
        {},
        growth_func=growth_func,
        rate_dist=rate_dist,
        include_times=include_times,
        biological_replicates=False,
        include_od=include_od,
    )
