"""Holds models for the PDZ datasets"""

import numpy as np
import numpy.typing as npt

from scistanpy import parameters
from .base_models import BaseEnrichmentTemplate, HierarchicalEnrichmentMeta
from .constants import DEFAULT_HYPERPARAMS, GrowthCurve, GrowthRate
from .flip_dsets import load_pdz_dataset


class BasePDZ3(BaseEnrichmentTemplate):  # pylint: disable=abstract-method
    """Base class for all PDZ3 models"""

    def __init__(
        self,
        starting_counts: npt.NDArray[np.int64],
        timepoint_counts: npt.NDArray[np.int64],
        alpha_alpha: float = DEFAULT_HYPERPARAMS["alpha_alpha"],
        alpha_beta: float = DEFAULT_HYPERPARAMS["alpha_beta"],
        **kwargs
    ):

        # Confirm input shapes
        assert starting_counts.shape == timepoint_counts.shape

        # We always have three replicates
        self.n_replicates = 3

        # Run parent init
        super().__init__(
            starting_counts=starting_counts,
            timepoint_counts=timepoint_counts,
            alpha_alpha=alpha_alpha,
            alpha_beta=alpha_beta,
            **kwargs
        )

    def _set_starting_props(
        self,
        alpha_alpha: float = DEFAULT_HYPERPARAMS["alpha_alpha"],
        alpha_beta: float = DEFAULT_HYPERPARAMS["alpha_beta"],
        **kwargs
    ):

        # Set a prior on the Dirichlet alpha parameters. We expect the starting
        # proportions from different experiments to be similar
        self.alpha = parameters.Gamma(
            alpha=alpha_alpha, beta=alpha_beta, shape=(self.n_variants,)
        )

        # Set the initial proportions
        return parameters.ExpDirichlet(
            alpha=self.alpha, shape=self.default_data["starting_counts"].shape
        )


# Build PDZ models
PDZEgLr = HierarchicalEnrichmentMeta(
    "PDZEgLr",
    (BasePDZ3,),
    {"GROWTH_CURVE": "exponential", "GROWTH_RATE": "lomax"},
)
PDZEgEr = HierarchicalEnrichmentMeta(
    "PDZEgEr",
    (BasePDZ3,),
    {"GROWTH_CURVE": "exponential", "GROWTH_RATE": "exponential"},
)
PDZEgGr = HierarchicalEnrichmentMeta(
    "PDZEgGr",
    (BasePDZ3,),
    {"GROWTH_CURVE": "exponential", "GROWTH_RATE": "gamma"},
)
PDZSgLr = HierarchicalEnrichmentMeta(
    "PDZSgLr",
    (BasePDZ3,),
    {"GROWTH_CURVE": "sigmoid", "GROWTH_RATE": "lomax"},
)
PDZSgEr = HierarchicalEnrichmentMeta(
    "PDZSgEr",
    (BasePDZ3,),
    {"GROWTH_CURVE": "sigmoid", "GROWTH_RATE": "exponential"},
)
PDZSgGr = HierarchicalEnrichmentMeta(
    "PDZSgGr",
    (BasePDZ3,),
    {"GROWTH_CURVE": "sigmoid", "GROWTH_RATE": "gamma"},
)

# Define a mapping from growth curves and rates to classes
PDZ3_MODELS = {
    ("exponential", "lomax"): PDZEgLr,
    ("exponential", "exponential"): PDZEgEr,
    ("exponential", "gamma"): PDZEgGr,
    ("sigmoid", "lomax"): PDZSgLr,
    ("sigmoid", "exponential"): PDZSgEr,
    ("sigmoid", "gamma"): PDZSgGr,
}


# Define a function that maps from growth curves and rates to classes
def get_pdz3_model(
    growth_curve: GrowthCurve, growth_rate: GrowthRate
) -> type[BasePDZ3]:
    """Gets the appropriate class for the given growth curve and rate."""
    return PDZ3_MODELS[(growth_curve, growth_rate)]


def get_pdz3_instance(
    filepath: str, growth_curve: GrowthCurve, growth_rate: GrowthRate
) -> BasePDZ3:
    """Gets an instance of the appropriate class for the given library and growth curve."""
    # Load the data and remove fields we do not need
    dataset = load_pdz_dataset(filepath)
    dataset.pop("variants")

    # Build the instance
    return get_pdz3_model(growth_curve=growth_curve, growth_rate=growth_rate)(**dataset)
