# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
SciStanPy: Probabilistic modeling for scientific applications with Stan and PyTorch.

SciStanPy is a Python package for building and evaluating Bayesian models,
particularly suited for analyzing deep mutational scanning data and other
scientific applications. The package combines the probabilistic modeling
capabilities of Stan with the flexibility of PyTorch for neural network
components.

Key Features:
    - Integration between Stan and PyTorch
    - Inbuilt model evaluation tools
    - Inbuilt prior selection/evaluation
    - Type-safe model construction with comprehensive type checking
    - Flexible parameter and distribution definitions
    - Built-in support for common scientific modeling patterns

Global Variables:
    RNG: Global random number generator for reproducible computations
    __version__: Package version string

Example:
    >>> import scistanpy as ssp
    >>> # Set global seed for reproducibility
    >>> ssp.manual_seed(42)
    >>> # Create a simple model
    >>> model = ssp.Model()
"""

from typing import Optional, TYPE_CHECKING

from typeguard import install_import_hook

import numpy as np
import torch

# Define the version
__version__ = "0.0.7"

# Set up type checking
install_import_hook("scistanpy")

# Define the global random number generator
RNG: np.random.Generator
"""Global random number generator for SciStanPy.

This generator is used throughout the package to ensure reproducible
random number generation. It can be seeded using the manual_seed()
function to ensure consistent results across runs.

:type: np.random.Generator

Example:
    >>> import scistanpy as ssp
    >>> # Use the global RNG
    >>> random_values = ssp.RNG.normal(0, 1, size=10)
"""

# Get custom types if TYPE_CHECKING is True
if TYPE_CHECKING:
    from scistanpy import custom_types


def manual_seed(seed: Optional["custom_types.Integer"] = None):
    """Set the seed for global random number generators.

    This function sets the seed for both NumPy and PyTorch random number
    generators to ensure reproducible results across the entire SciStanPy
    workflow. It updates the global RNG variable and synchronizes PyTorch's
    random state.

    :param seed: Seed value for random number generation. If None, uses
                system entropy to generate a random seed.
    :type seed: Union[custom_types.Integer, None]

    :raises TypeError: If seed is not an integer type when provided

    Example:
        >>> import scistanpy as ssp
        >>> # Set seed for reproducible results
        >>> ssp.manual_seed(42)
        >>> # Now all random operations will be reproducible
        >>> random_data = ssp.RNG.normal(0, 1, size=100)

    Note:
        This function modifies global state and should typically be called
        once at the beginning of a script or analysis for reproducibility. Note
        also that it sets the PyTorch global random seed by calling `torch.manual_seed`,
        which means that any random processes in PyTorch will also become deterministic.
    """
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

parameters = utils.lazy_import("scistanpy.model.components.parameters")
results = utils.lazy_import("scistanpy.model.results")
