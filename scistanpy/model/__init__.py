# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Model construction and management for SciStanPy.

This module provides the core infrastructure for building, compiling, and
executing Bayesian models using the Stan probabilistic programming language.
It serves as the primary interface between user-defined model specifications
and the underlying Stan compilation and sampling machinery.

The module is organized into several key submodules:

Components Submodule (:mod:`scistanpy.model.components`):
    Contains the building blocks for model construction, including:

    - **Constants**: Fixed values and hyperparameters
    - **Parameters**: Random variables and distributions
    - **Transformations**: Mathematical operations on parameters
    - **Custom Distributions**: Extended `torch` distribution library
    - **Abstract Base Classes**: Base class for all model components.

Model Core (:mod:`scistanpy.model.model`):
    Provides the main Model class that orchestrates:

    - Model component registration and validation
    - Stan code generation from Python specifications
    - Compilation management with caching
    - Prior and posterior sampling interfaces
    - Integration with external fitting libraries (i.e., PyTorch)

Results Submodule (:mod:`scistanpy.model.results`):
    Handles post-processing and analysis of model outputs:

    - MCMC diagnostics and convergence assessment
    - Posterior summary statistics
    - Visualization integration
    - Export utilities for downstream analysis

Key Features:
    - **Declarative Model Specification**: Build models using Python syntax
      that automatically translates to Stan code
    - **Type Safety**: Comprehensive type checking ensures model correctness
    - **Lazy Compilation**: Models are compiled only when needed, with
      intelligent caching to avoid redundant compilation
    - **Extensible Architecture**: Easy addition of new distributions,
      transformations, and model components

The module design emphasizes both ease of use for common modeling tasks
and extensibility for advanced applications. Models are constructed by
composing reusable components, promoting code reuse and maintainability.

Workflow Overview:
    1. **Model Definition**: Instantiate Model class and add components
    2. **Component Registration**: Add parameters, constants, and observables
    3. **Compilation**: Automatic Stan code generation and compilation
    4. **Sampling**: Execute prior predictive checks or posterior inference
    5. **Analysis**: Process results using built-in diagnostic tools

Example:
    >>> import scistanpy as ssp
    >>> # Create a simple regression model
    >>> class Regressor(ssp.Model):
    >>>     def __init__(self):
    >>>         super().__init__()
    >>>         self.mu = ssp.parameters.Normal(0, 1)
    >>>         self.sigma = ssp.parameters.HalfNormal(1)
    >>>         self.y = ssp.parameters.Normal(self.mu, self.sigma)
    >>> # Create a model instance
    >>> model = Regressor()
    >>> # Sample from prior
    >>> prior_samples = model.draw(n=1000)
    >>> # Run mcmc using Stan
    >>> res = model.mcmc(data = {"y": data})
    >>> # Run maximum likelihood estimation with PyTorch
    >>> mle = model.mle(data = {"y": data})


Performance Considerations:
    - Model compilation can be time-intensive; use caching for repeated runs
    - Large models benefit from Stan's threading capabilities (enabled by default)
    - Memory usage scales with model complexity and sample size

See Also:
    - :mod:`scistanpy.operations`: Mathematical operations for model building
    - :mod:`scistanpy.plotting`: Visualization tools for model diagnostics
"""
