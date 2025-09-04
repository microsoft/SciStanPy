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

Key Components:
    - **Automatic Code Generation**: Converts SciStanPy models to Stan language
    - **Compilation Management**: Handles Stan-to-C++ compilation with caching
    - **Sampling Interface**: Provides high-level interface to Stan sampling algorithms
    - **Custom Functions**: Extends Stan with SciStanPy-specific mathematical functions
    - **Diagnostics Integration**: Incorporates Stan's comprehensive MCMC diagnostics

Architecture:
    The submodule follows a layered architecture:

    1. **Model Translation Layer**: Converts Python objects to Stan syntax
    2. **Code Generation Layer**: Assembles complete Stan programs
    3. **Compilation Layer**: Manages C++ compilation and executable creation.
       This stage relies heavily on the `CmdStanPy` project.
    4. **Execution Layer**: Interfaces with Stan sampling algorithms. This stage
       also relies heavily on the `CmdStanPy` project.
    5. **Results Layer**: Processes and structures Stan output

Stan Language Features:
    - Support for Stan's probabilistic modeling constructs
    - Custom function definitions for SciStanPy-specific operations
    - Comprehensive convergence diagnostics and model checking

Custom Extensions:
    The submodule includes custom Stan functions that extend the base language
    with SciStanPy-specific mathematical operations. These functions are
    automatically included in generated Stan programs and provide:

    - Specialized probability distributions
    - Custom transformation functions
    - Optimized mathematical operations
    - SciStanPy-specific utility functions

Global Configuration:
    STAN_INCLUDE_PATHS: List of directories containing custom Stan function definitions.
                       These paths are automatically included during Stan compilation
                       to provide access to SciStanPy-specific extensions.

Example:
    >>> # Compile SciStanPy model to Stan
    >>> stan_model = model.to_stan()
    >>> # Run MCMC sampling
    >>> results = stan_model.sample(data=observed_data, chains=4)

See Also:
    - Stan User's Guide: https://mc-stan.org/docs/
    - CmdStanPy Documentation: https://cmdstanpy.readthedocs.io/
    - :mod:`scistanpy.model.results`: For working with Stan sampling results
"""

import os.path

# We need the path of the directory of the current file. This is used to include
# the custom stan functions.
STAN_INCLUDE_PATHS = [os.path.abspath(os.path.dirname(__file__))]
