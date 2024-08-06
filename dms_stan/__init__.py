"""
Defines code needed for building models in Stan for evaluating deep mutational scanning
data.
"""

from typing import Optional

from typeguard import install_import_hook

import numpy as np

# Set up type checking
install_import_hook("dms_stan")

# pylint: disable=wrong-import-position
from dms_stan import constant, model, operations, param, prior_predictive

# Define the global random number generator
RNG: np.random.Generator


def manual_seed(seed: Optional[int] = None):
    """Set the seed for the global random number generator."""
    global RNG  # pylint: disable=global-statement
    RNG = np.random.default_rng(seed)


manual_seed()  # Set the seed for the global random number generator
