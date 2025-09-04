# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Default configuration values for SciStanPy package components.

This module centralizes default values used across various components of the
SciStanPy package, including fitting parameters, naming conventions, and
Stan model configuration options. These defaults can be overridden by users
as needed for specific use cases.

The module is organized into logical groups covering:
- Model fitting and optimization defaults
- Variable and dimension naming conventions
- Stan model compilation and execution settings
- Diagnostic thresholds for model validation
"""

import string

# Fitting defaults
DEFAULT_N_EPOCHS = 100000
"""Default number of training epochs for model fitting.

:type: int
:value: 100000
"""

DEFAULT_EARLY_STOP = 10
"""Default number of epochs to wait before early stopping.

Controls how many epochs without improvement to wait before terminating
training early to prevent overfitting.

:type: int
:value: 10
"""

DEFAULT_LR = 0.001
"""Default learning rate for optimization algorithms.

:type: float
:value: 0.001
"""

# Default order for index variable names
DEFAULT_INDEX_ORDER = ("i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t")
"""Default ordering for index variable names in mathematical expressions.

Provides a consistent sequence of variable names for indexing operations,
following mathematical conventions starting with 'i', 'j', 'k', etc.

:type: tuple[str, ...]
"""

# Default names for the dimensions of the data
DEFAULT_DIM_NAMES = tuple(l for l in string.ascii_lowercase if l != "n")
"""Default names for data dimensions, excluding 'n'.

Generates dimension names from lowercase ASCII letters, excluding 'n' which
is typically reserved for sample size notation.

:type: tuple[str, ...]
"""

# Defaults for the Stan model
DEFAULT_FORCE_COMPILE = False
"""Default setting for forcing Stan model recompilation.

When False, uses cached compiled models when available. When True,
forces recompilation even if a cached version exists.

:type: bool
:value: False
"""

DEFAULT_STANC_OPTIONS = {"warn-pedantic": True, "O1": True}
"""Default options passed to the Stan compiler (stanc).

Configures Stan compiler behavior with pedantic warnings enabled
and first-level optimization enabled.

:type: dict[str, bool]
"""

DEFAULT_CPP_OPTIONS = {"STAN_THREADS": True}
"""Default C++ compilation options for Stan models.

Enables threading support in compiled Stan models for improved
performance on multi-core systems.

:type: dict[str, bool]
"""

DEFAULT_USER_HEADER = None
"""Default user header content for Stan models.

Custom C++ code that can be included in Stan model compilation.
None indicates no custom header by default.

:type: None
:value: None
"""

DEFAULT_MODEL_NAME = "model"
"""Default name for generated Stan models.

:type: str
:value: "model"
"""

# Defaults for Stan diagnostics
DEFAULT_EBFMI_THRESH = 0.2
"""Default threshold for Energy Bayesian Fraction of Missing Information (E-BFMI).

Values below this threshold may indicate inefficient sampling and
potential bias in MCMC results.

:type: float
:value: 0.2
"""

DEFAULT_ESS_THRESH = 100  # Per chain
"""Default threshold for Effective Sample Size (ESS) per chain.

Minimum effective sample size considered adequate for reliable
posterior inference from each MCMC chain.

:type: int
:value: 100
"""

DEFAULT_RHAT_THRESH = 1.01
"""Default threshold for R-hat convergence diagnostic.

Values above this threshold indicate potential convergence issues
in MCMC sampling across chains.

:type: float
:value: 1.01
"""
