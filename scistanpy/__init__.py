"""
Defines code needed for building models in Stan for evaluating deep mutational scanning
data.
"""

from typing import Optional

from typeguard import install_import_hook

import numpy as np
import torch

# Define the version
__version__ = "0.0.7"

# Set up type checking
install_import_hook("scistanpy")

# Define the global random number generator
RNG: np.random.Generator


def manual_seed(seed: Optional[int] = None):
    """Set the seed for the global random number generator."""
    global RNG  # pylint: disable=global-statement
    RNG = np.random.default_rng(seed)
    if seed is not None:
        torch.manual_seed(seed)


manual_seed()  # Set the seed for the global random number generator

# Import objects that should be easily accessible from the package level
# pylint: disable=wrong-import-position
from scistanpy import utils

# Lazy imports for performance and to avoid circular imports
from scistanpy.model.components.constants import Constant
from scistanpy.model.model import Model

parameters = utils.lazy_import("scistanpyy.model.components.parameters")
results = utils.lazy_import("scistanpy.model.results")
