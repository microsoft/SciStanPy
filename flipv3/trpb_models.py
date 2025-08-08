"""Code for building TrpB models for the three- and four-site libraries."""

from abc import ABCMeta, abstractmethod

import numpy as np
import numpy.typing as npt

from dms_stan import Constant, Model, operations, parameters


DEFAULT_HYPERPARAMS = {
    "alpha": 0.1,
    "exp_beta": 1.0,
    "lambda_": 1.0,
    "lomax_alpha": 1.0,
    "c_alpha": 4.0,
    "c_beta": 8.0,
    "r_sigma_sigma": 0.01,
    "inv_r_alpha": 7.0,
    "inv_r_beta": 1.0,
}


class BaseMetaTrpBModel(ABCMeta):
    """Base metaclass for TrpB models. Children of this set the growth curve and
    growth rate methods."""

    def __new__(mcs, name, bases, attrs):
        # Update the attributes based on the growth curve and growth rate
        assert "GROWTH_CURVE" in attrs, "GROWTH_CURVE must be defined"
        assert "GROWTH_RATE" in attrs, "GROWTH_RATE must be defined"

        # Create the class
        return super().__new__(mcs, name, bases, attrs)

    def __init__(cls, name, bases, attrs, **kwargs):  # pylint: disable=unused-argument

        # Run `type.__init__` to ensure the class is properly initialized
        super().__init__(name, bases, attrs)

        # Set the growth rate and growth methods
        # pylint: disable=no-value-for-parameter
        cls._set_growth_rate = cls._def_growth_rate()
        cls._grow = cls._def_growth_method()

    @abstractmethod
    def _def_growth_rate(cls):
        """Define the growth rate method for the model class."""

    @abstractmethod
    def _def_growth_method(cls):
        """Define the growth method for the model class."""


class MetaTrpBFlatModel(BaseMetaTrpBModel):
    """
    Metaclass for TrpB models. This sets the `_set_growth_rate` and `_grow`
    methods
    """

    def _def_growth_rate(cls):
        """Sets the growth rate for the model based on the growth rate type."""

        def exponential_growth_rate(  # pylint: disable=unused-argument
            self, beta: float = DEFAULT_HYPERPARAMS["exp_beta"], **kwargs
        ):
            """Set the growth rate to exponential."""
            return parameters.Exponential(rate=beta, shape=self.n_variants)

        def lomax_growth_rate(  # pylint: disable=unused-argument
            self,
            lambda_: float = DEFAULT_HYPERPARAMS["lambda_"],
            lomax_alpha: float = DEFAULT_HYPERPARAMS["lomax_alpha"],
            **kwargs,
        ):
            """Set the growth rate to Lomax."""
            return parameters.Lomax(
                lambda_=lambda_, alpha=lomax_alpha, shape=self.n_variants
            )

        # Different function for different growth rates
        if cls.GROWTH_RATE == "exponential":
            return exponential_growth_rate
        elif cls.GROWTH_RATE == "lomax":
            return lomax_growth_rate
        elif cls.GROWTH_RATE == "gamma":
            return None
        else:
            raise ValueError(f"Unknown growth rate: {cls.GROWTH_RATE}")

    def _def_growth_method(cls):
        """Sets the growth method for the model based on the growth curve type."""

        def exponential_growth(self, **kwargs):  # pylint: disable=unused-argument
            """Grow the initial proportions to the time tg0 using exponential growth."""
            return operations.log_exponential_growth(
                t=self.times,
                r=self.r,
                log_A=self.log_theta_t0,
                shape=self.log_theta_tg0.shape,
            )

        def sigmoid_growth(  # pylint: disable=unused-argument
            self,
            c_alpha: float = DEFAULT_HYPERPARAMS["c_alpha"],
            c_beta: float = DEFAULT_HYPERPARAMS["c_beta"],
            **kwargs,
        ):
            """Grow the initial proportions to the time tg0 using sigmoid growth."""
            self.c = parameters.Gamma(alpha=c_alpha, beta=c_beta)
            return operations.log_sigmoid_growth_init_param(
                t=self.times,
                log_x0=self.log_theta_t0,
                c=self.c,
                r=self.r,
                shape=self.log_theta_tg0.shape,
            )

        # Different function for different growth curves
        if cls.GROWTH_CURVE == "exponential":
            return exponential_growth
        elif cls.GROWTH_CURVE == "sigmoid":
            return sigmoid_growth
        else:
            raise ValueError(f"Unknown growth curve: {cls.GROWTH_CURVE}")


class MetaTrpBHierarchicalModel(MetaTrpBFlatModel):
    """
    Metaclass for TrpB hierarchical models. This sets the `_set_growth_rate` and
    `_grow` methods.
    """

    def _def_growth_rate(cls):
        """Sets the growth rate for the model based on the growth rate type."""
        # Run the parent method to get the mean growth rate function
        gr_func = super()._def_growth_rate()

        def lomax_exp_growth_rate(
            self, r_sigma_sigma: float = DEFAULT_HYPERPARAMS["r_sigma_sigma"], **kwargs
        ):
            """
            The returned function is the mean growth rate. We add noise for the
            hierarchical models.
            """
            # Assign the mean growth rate
            self.r_mean = gr_func(self, **kwargs)

            # Set the standard deviation for the growth rate
            self.r_sigma = parameters.HalfNormal(sigma=r_sigma_sigma)
            return parameters.Normal(
                mu=self.r_mean,
                sigma=self.r_sigma,
                shape=(self.n_replicates, self.n_variants),
            )

        def gamma_inv_growth_rate(  # pylint: disable=unused-argument
            self,
            inv_r_alpha: float = DEFAULT_HYPERPARAMS["inv_r_alpha"],
            inv_r_beta: float = DEFAULT_HYPERPARAMS["inv_r_beta"],
            **kwargs,
        ):
            """
            A gamma distribution defines the inverse of the growth rate.
            """
            # Assign the inverse growth rate
            assert gr_func is None
            self.inv_r_mean = parameters.Gamma(
                alpha=inv_r_alpha, beta=inv_r_beta, shape=(self.n_variants,)
            )

            # The growth rates are given by the exponential distribution with inverse
            # rate given by the gamma distribution
            return parameters.Exponential(
                rate=self.inv_r_mean, shape=(self.n_replicates, self.n_variants)
            )

        # Different function for different growth rates
        if cls.GROWTH_RATE in {
            "lomax",
            "exponential",
        }:
            return lomax_exp_growth_rate
        elif cls.GROWTH_RATE == "gamma":
            return gamma_inv_growth_rate
        else:
            raise ValueError(f"Unknown growth rate: {cls.GROWTH_RATE}")


class BaseTrpBTemplate(Model):
    """Functionality shared by all TrpB models."""

    def __init__(
        self,
        starting_counts: npt.NDArray[np.int64],
        timepoint_counts: npt.NDArray[np.int64],
        times: npt.NDArray[np.float64],
        alpha: float = DEFAULT_HYPERPARAMS["alpha"],
        **kwargs,
    ):
        """
        Initializes a model for the non-hierarchical TrpB three-site libraries
        (libA, B, and C)
        """
        # Confirm input shapes
        assert times.ndim == 1
        assert timepoint_counts.shape[-2] == times.shape[0]
        assert timepoint_counts.shape[-1] == starting_counts.shape[-1]

        # Note the number of variants
        self.n_variants = starting_counts.shape[-1]

        # Set the total number of starting and ending counts by replicate
        self.total_starting_counts = Constant(
            starting_counts.sum(axis=-1, keepdims=True), togglable=False
        )
        self.total_timepoint_counts = Constant(
            timepoint_counts.sum(axis=-1, keepdims=True), togglable=False
        )

        # We keep the times as a constant. Dimensions must be added to take it to
        # the size of the timepoint counts. We also normalize the times to have
        # a max of 1.0.
        self.times = Constant(
            times[(...,) + (None,) * (timepoint_counts.ndim - 1)] / times.max(),
            togglable=False,
        )

        # Record default data
        super().__init__(
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
        )

        # Set the initial proportions
        self.log_theta_t0 = parameters.ExpDirichlet(
            alpha=alpha, shape=(starting_counts.shape[-1],)
        )

        # Set the growth rate
        self.r = self._set_growth_rate(**kwargs)

        # Calculate the proportions at time tg0 and renormalized
        self.raw_abundances_tg0 = self._grow(**kwargs)
        self.log_theta_tg0 = operations.normalize_log(self.raw_abundances_tg0)

        # Model counts
        self.starting_counts = parameters.MultinomialLogTheta(
            log_theta=self.log_theta_t0,
            N=self.total_starting_counts,
            shape=starting_counts.shape,
        )
        self.timepoint_counts = parameters.MultinomialLogTheta(
            log_theta=self.log_theta_tg0,
            N=self.total_timepoint_counts,
            shape=timepoint_counts.shape,
        )

    @abstractmethod
    def _set_growth_rate(self, **kwargs):
        """
        Set the growth rate for the model. This should be implemented in subclasses.
        """

    @abstractmethod
    def _grow(self, **kwargs):
        """
        Grow the initial proportions to the time tg0. This should be implemented in
        subclasses.
        """


class TemplateThreeSiteModelFlat(BaseTrpBTemplate):  # pylint: disable=abstract-method
    """
    Template class for TrpB three-site models that are not hierarchical (libA, libB,
    and libC).
    """

    def __init__(
        self,
        starting_counts: npt.NDArray[np.int64],
        timepoint_counts: npt.NDArray[np.int64],
        times: npt.NDArray[np.float64],
        alpha: float = DEFAULT_HYPERPARAMS["alpha"],
        **kwargs,
    ):
        # Make sure the starting counts are 1D and the timepoint counts are 2D
        assert starting_counts.ndim == 1
        assert timepoint_counts.ndim == 2

        super().__init__(
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            times=times,
            alpha=alpha,
            **kwargs,
        )


class TemplateThreeSiteModelHierarchical(
    BaseTrpBTemplate
):  # pylint: disable=abstract-method
    """
    Template class for TrpB three-site models that are hierarchical (libD, libE, libF,
    libG, libH, and libI).
    """

    def __init__(
        self,
        starting_counts: npt.NDArray[np.int64],
        timepoint_counts: npt.NDArray[np.int64],
        times: npt.NDArray[np.float64],
        alpha: float = DEFAULT_HYPERPARAMS["alpha"],
        **kwargs,
    ):

        # Make sure the starting counts are 1D and the timepoint counts are 3D
        assert starting_counts.ndim == 1
        assert timepoint_counts.ndim == 3

        # The number of replicates is the first dimension of the timepoint counts
        self.n_replicates = timepoint_counts.shape[0]

        super().__init__(
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            times=times,
            alpha=alpha,
            **kwargs,
        )


# Define the four-site TrpB model
class TemplateFourSiteModel(BaseTrpBTemplate):  # pylint: disable=abstract-method
    """Template for the four-site TrpB models."""

    def __init__(
        self,
        starting_counts: npt.NDArray[np.int64],
        timepoint_counts: npt.NDArray[np.int64],
        times: npt.NDArray[np.float64],
        alpha: float = DEFAULT_HYPERPARAMS["alpha"],
        **kwargs,
    ):
        """
        Initializes a model for the four-site TrpB libraries.
        """
        # Make sure that the starting counts are 2D and the timepoint counts are 3D
        assert starting_counts.ndim == 2
        assert timepoint_counts.ndim == 3

        # The number of replicates is the first dimension of the timepoint counts
        self.n_replicates = timepoint_counts.shape[0]

        super().__init__(
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            times=times,
            alpha=alpha,
            **kwargs,
        )


# Define TrpB models for the flat three-site libraries
TrpBFlatEgLr = MetaTrpBFlatModel(
    "TrpBFlatEgLr",
    (TemplateThreeSiteModelFlat,),
    {"GROWTH_CURVE": "exponential", "GROWTH_RATE": "lomax"},
)
TrpBFlatEgEr = MetaTrpBFlatModel(
    "TrpBFlatEgEr",
    (TemplateThreeSiteModelFlat,),
    {"GROWTH_CURVE": "exponential", "GROWTH_RATE": "exponential"},
)
TrpBFlatSgLr = MetaTrpBFlatModel(
    "TrpBFlatSgLr",
    (TemplateThreeSiteModelFlat,),
    {"GROWTH_CURVE": "sigmoid", "GROWTH_RATE": "lomax"},
)
TrpBFlatSgEr = MetaTrpBFlatModel(
    "TrpBFlatSgEr",
    (TemplateThreeSiteModelFlat,),
    {"GROWTH_CURVE": "sigmoid", "GROWTH_RATE": "exponential"},
)

# Define TrpB models for the hierarchical three-site libraries
TrpBHierarchicalEgLr = MetaTrpBHierarchicalModel(
    "TrpBHierarchicalEgLr",
    (TemplateThreeSiteModelHierarchical,),
    {"GROWTH_CURVE": "exponential", "GROWTH_RATE": "lomax"},
)
TrpBHierarchicalEgEr = MetaTrpBHierarchicalModel(
    "TrpBHierarchicalEgEr",
    (TemplateThreeSiteModelHierarchical,),
    {"GROWTH_CURVE": "exponential", "GROWTH_RATE": "exponential"},
)
TrpBHierarchicalEgGr = MetaTrpBHierarchicalModel(
    "TrpBHierarchicalEgGr",
    (TemplateThreeSiteModelHierarchical,),
    {"GROWTH_CURVE": "exponential", "GROWTH_RATE": "gamma"},
)
TrpBHierarchicalSgLr = MetaTrpBHierarchicalModel(
    "TrpBHierarchicalSgLr",
    (TemplateThreeSiteModelHierarchical,),
    {"GROWTH_CURVE": "sigmoid", "GROWTH_RATE": "lomax"},
)
TrpBHierarchicalSgEr = MetaTrpBHierarchicalModel(
    "TrpBHierarchicalSgEr",
    (TemplateThreeSiteModelHierarchical,),
    {"GROWTH_CURVE": "sigmoid", "GROWTH_RATE": "exponential"},
)
TrpBHierarchicalSgGr = MetaTrpBHierarchicalModel(
    "TrpBHierarchicalSgGr",
    (TemplateThreeSiteModelHierarchical,),
    {"GROWTH_CURVE": "sigmoid", "GROWTH_RATE": "gamma"},
)

# Define the four-site TrpB models
TrpBFourSiteEgLr = MetaTrpBHierarchicalModel(
    "TrpBFourSiteEgLr",
    (TemplateFourSiteModel,),
    {"GROWTH_CURVE": "exponential", "GROWTH_RATE": "lomax"},
)
TrpBFourSiteEgEr = MetaTrpBHierarchicalModel(
    "TrpBFourSiteEgEr",
    (TemplateFourSiteModel,),
    {"GROWTH_CURVE": "exponential", "GROWTH_RATE": "exponential"},
)
TrpBFourSiteEgGr = MetaTrpBHierarchicalModel(
    "TrpBFourSiteEgGr",
    (TemplateFourSiteModel,),
    {"GROWTH_CURVE": "exponential", "GROWTH_RATE": "gamma"},
)
TrpBFourSiteSgLr = MetaTrpBHierarchicalModel(
    "TrpBFourSiteSgLr",
    (TemplateFourSiteModel,),
    {"GROWTH_CURVE": "sigmoid", "GROWTH_RATE": "lomax"},
)
TrpBFourSiteSgEr = MetaTrpBHierarchicalModel(
    "TrpBFourSiteSgEr",
    (TemplateFourSiteModel,),
    {"GROWTH_CURVE": "sigmoid", "GROWTH_RATE": "exponential"},
)
TrpBFourSiteSgGr = MetaTrpBHierarchicalModel(
    "TrpBFourSiteSgGr",
    (TemplateFourSiteModel,),
    {"GROWTH_CURVE": "sigmoid", "GROWTH_RATE": "gamma"},
)
