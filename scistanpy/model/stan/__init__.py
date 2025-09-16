# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Stan probabilistic programming language integration for SciStanPy.

This submodule provides comprehensive integration between SciStanPy models and
the Stan probabilistic programming language, enabling high-performance Bayesian
inference through Hamiltonian Monte Carlo (HMC).

The integration handles the complete workflow from SciStanPy model specifications
to Stan code generation, compilation, and execution. It automatically translates
Python model definitions into Stan code while preserving the probabilistic structure
and mathematical relationships.

**Key Components:**
    - Automatic Code Generation: Converts SciStanPy models to Stan language
    - Compilation Management: Handles Stan-to-C++ compilation with caching
    - Sampling Interface: Provides high-level interface to Stan sampling algorithms
    - Custom Functions: Extends Stan with SciStanPy-specific mathematical functions
    - Diagnostics Integration: Incorporates Stan's comprehensive MCMC diagnostics

**Custom Extensions:**
    The submodule includes custom Stan functions that extend the base language
    with SciStanPy-specific mathematical operations. These functions are
    automatically included in generated Stan programs and provide:

        - Specialized probability distributions
        - Custom transformation functions
        - Optimized mathematical operations
        - SciStanPy-specific utility functions
"""

import os.path

# We need the path of the directory of the current file. This is used to include
# the custom stan functions.
STAN_INCLUDE_PATHS = [os.path.abspath(os.path.dirname(__file__))]
"""
A list of absolute paths used by the Stan code assembly step to locate bundled Stan
function snippets and headers.
"""
