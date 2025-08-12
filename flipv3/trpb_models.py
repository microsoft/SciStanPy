"""Code for building TrpB models for the three- and four-site libraries."""

from typing import Literal

import numpy as np
import numpy.typing as npt

from scistanpy import Constant, parameters
from .base_models import (
    BaseEnrichmentTemplate,
    FlatEnrichmentMeta,
    HierarchicalEnrichmentMeta,
)
from .constants import DEFAULT_HYPERPARAMS, GrowthCurve, GrowthRate
from .flip_dsets import load_trpb_dataset

TrpBLibrary = Literal[
    "libA", "libB", "libC", "libD", "libE", "libF", "libG", "libH", "libI", "four-site"
]


class BaseTrpBTemplate(BaseEnrichmentTemplate):  # pylint: disable=abstract-method
    """Base template model for all TrpB models."""

    def __init__(
        self,
        starting_counts: npt.NDArray[np.int64],
        timepoint_counts: npt.NDArray[np.int64],
        times: npt.NDArray[np.float64],
        alpha: float = DEFAULT_HYPERPARAMS["alpha"],
        **kwargs,
    ):

        # Confirm input shapes
        assert times.ndim == 1
        assert timepoint_counts.shape[0] == times.shape[0]

        # We keep the times as a constant. Dimensions must be added to take it to
        # the size of the timepoint counts. We also normalize the times to have
        # a max of 1.0.
        self.times = Constant(
            times[(...,) + (None,) * (timepoint_counts.ndim - 1)] / times.max(),
            togglable=False,
        )

        # Run parent init
        super().__init__(
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            alpha=alpha,
            **kwargs,
        )

    def _set_starting_props(
        self, alpha: float = DEFAULT_HYPERPARAMS["alpha"], **kwargs
    ):
        """Sets the starting proportions using an exponential Dirichlet prior."""
        return parameters.ExpDirichlet(
            alpha=alpha, shape=(self.default_data["starting_counts"].shape[-1],)
        )


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
        self.n_replicates = timepoint_counts.shape[1]

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
        self.n_replicates = timepoint_counts.shape[1]

        super().__init__(
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            times=times,
            alpha=alpha,
            **kwargs,
        )


# Define TrpB models for the flat three-site libraries
TrpBFlatEgLr = FlatEnrichmentMeta(
    "TrpBFlatEgLr",
    (TemplateThreeSiteModelFlat,),
    {"GROWTH_CURVE": "exponential", "GROWTH_RATE": "lomax"},
)
TrpBFlatEgEr = FlatEnrichmentMeta(
    "TrpBFlatEgEr",
    (TemplateThreeSiteModelFlat,),
    {"GROWTH_CURVE": "exponential", "GROWTH_RATE": "exponential"},
)
TrpBFlatSgLr = FlatEnrichmentMeta(
    "TrpBFlatSgLr",
    (TemplateThreeSiteModelFlat,),
    {"GROWTH_CURVE": "sigmoid", "GROWTH_RATE": "lomax"},
)
TrpBFlatSgEr = FlatEnrichmentMeta(
    "TrpBFlatSgEr",
    (TemplateThreeSiteModelFlat,),
    {"GROWTH_CURVE": "sigmoid", "GROWTH_RATE": "exponential"},
)

# Define TrpB models for the hierarchical three-site libraries
TrpBHierarchicalEgLr = HierarchicalEnrichmentMeta(
    "TrpBHierarchicalEgLr",
    (TemplateThreeSiteModelHierarchical,),
    {"GROWTH_CURVE": "exponential", "GROWTH_RATE": "lomax"},
)
TrpBHierarchicalEgEr = HierarchicalEnrichmentMeta(
    "TrpBHierarchicalEgEr",
    (TemplateThreeSiteModelHierarchical,),
    {"GROWTH_CURVE": "exponential", "GROWTH_RATE": "exponential"},
)
TrpBHierarchicalEgGr = HierarchicalEnrichmentMeta(
    "TrpBHierarchicalEgGr",
    (TemplateThreeSiteModelHierarchical,),
    {"GROWTH_CURVE": "exponential", "GROWTH_RATE": "gamma"},
)
TrpBHierarchicalSgLr = HierarchicalEnrichmentMeta(
    "TrpBHierarchicalSgLr",
    (TemplateThreeSiteModelHierarchical,),
    {"GROWTH_CURVE": "sigmoid", "GROWTH_RATE": "lomax"},
)
TrpBHierarchicalSgEr = HierarchicalEnrichmentMeta(
    "TrpBHierarchicalSgEr",
    (TemplateThreeSiteModelHierarchical,),
    {"GROWTH_CURVE": "sigmoid", "GROWTH_RATE": "exponential"},
)
TrpBHierarchicalSgGr = HierarchicalEnrichmentMeta(
    "TrpBHierarchicalSgGr",
    (TemplateThreeSiteModelHierarchical,),
    {"GROWTH_CURVE": "sigmoid", "GROWTH_RATE": "gamma"},
)

# Define the four-site TrpB models
TrpBFourSiteEgLr = HierarchicalEnrichmentMeta(
    "TrpBFourSiteEgLr",
    (TemplateFourSiteModel,),
    {"GROWTH_CURVE": "exponential", "GROWTH_RATE": "lomax"},
)
TrpBFourSiteEgEr = HierarchicalEnrichmentMeta(
    "TrpBFourSiteEgEr",
    (TemplateFourSiteModel,),
    {"GROWTH_CURVE": "exponential", "GROWTH_RATE": "exponential"},
)
TrpBFourSiteEgGr = HierarchicalEnrichmentMeta(
    "TrpBFourSiteEgGr",
    (TemplateFourSiteModel,),
    {"GROWTH_CURVE": "exponential", "GROWTH_RATE": "gamma"},
)
TrpBFourSiteSgLr = HierarchicalEnrichmentMeta(
    "TrpBFourSiteSgLr",
    (TemplateFourSiteModel,),
    {"GROWTH_CURVE": "sigmoid", "GROWTH_RATE": "lomax"},
)
TrpBFourSiteSgEr = HierarchicalEnrichmentMeta(
    "TrpBFourSiteSgEr",
    (TemplateFourSiteModel,),
    {"GROWTH_CURVE": "sigmoid", "GROWTH_RATE": "exponential"},
)
TrpBFourSiteSgGr = HierarchicalEnrichmentMeta(
    "TrpBFourSiteSgGr",
    (TemplateFourSiteModel,),
    {"GROWTH_CURVE": "sigmoid", "GROWTH_RATE": "gamma"},
)

# Define a mapping from library, growth curves, and growth rates, to classes
TRPB_MODELS = {
    ("flat", "exponential", "lomax"): TrpBFlatEgLr,
    ("flat", "exponential", "exponential"): TrpBFlatEgEr,
    ("flat", "sigmoid", "lomax"): TrpBFlatSgLr,
    ("flat", "sigmoid", "exponential"): TrpBFlatSgEr,
    ("hierarchical", "exponential", "lomax"): TrpBHierarchicalEgLr,
    ("hierarchical", "exponential", "exponential"): TrpBHierarchicalEgEr,
    ("hierarchical", "exponential", "gamma"): TrpBHierarchicalEgGr,
    ("hierarchical", "sigmoid", "lomax"): TrpBHierarchicalSgLr,
    ("hierarchical", "sigmoid", "exponential"): TrpBHierarchicalSgEr,
    ("hierarchical", "sigmoid", "gamma"): TrpBHierarchicalSgGr,
    ("four-site", "exponential", "lomax"): TrpBFourSiteEgLr,
    ("four-site", "exponential", "exponential"): TrpBFourSiteEgEr,
    ("four-site", "exponential", "gamma"): TrpBFourSiteEgGr,
    ("four-site", "sigmoid", "lomax"): TrpBFourSiteSgLr,
    ("four-site", "sigmoid", "exponential"): TrpBFourSiteSgEr,
    ("four-site", "sigmoid", "gamma"): TrpBFourSiteSgGr,
}
TRPB_LIBRARIES = {
    "libA": "flat",
    "libB": "flat",
    "libC": "flat",
    "libD": "hierarchical",
    "libE": "hierarchical",
    "libF": "hierarchical",
    "libG": "hierarchical",
    "libH": "hierarchical",
    "libI": "hierarchical",
    "four-site": "four-site",
}


# Define a function that takes us from library, growth curve, and growth rate, to
# class
def get_trpb_model(
    lib: TrpBLibrary,
    growth_curve: GrowthCurve,
    growth_rate: GrowthRate,
) -> type[BaseTrpBTemplate]:
    """Gets the appropriate class for the given library, growth curve, and growth rate."""
    return TRPB_MODELS[(TRPB_LIBRARIES[lib], growth_curve, growth_rate)]


def get_trpb_instance(
    filepath: str, lib: TrpBLibrary, growth_curve: GrowthCurve, growth_rate: GrowthRate
) -> BaseTrpBTemplate:
    """Gets an instance of the appropriate class for the given library and growth curve."""
    # Load the data and remove fields we do not need
    dataset = load_trpb_dataset(filepath=filepath, libname=lib)
    dataset.pop("variants")
    dataset.pop("starting_od")
    dataset.pop("timepoint_od")

    # Build the model
    return get_trpb_model(lib=lib, growth_curve=growth_curve, growth_rate=growth_rate)(
        **dataset
    )
