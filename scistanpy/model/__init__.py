# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Model construction and management for SciStanPy.

This module provides the core infrastructure for building, compiling, and
executing Bayesian models. It serves as the primary interface between user-defined
model specifications and the underlying Stan and PyTorch compilation, as well as
the sampling and optimization machinery.

The primary interface is the :py:class:`~scistanpy.model.model.Model` class,
which orchestrates model definition, compilation, and execution. Key features include:

    - Model component registration and validation
    - Stan code generation from Python specifications
    - Compilation management with caching
    - Prior and posterior sampling interfaces
    - Integration with external fitting libraries (i.e., PyTorch)

Instances of the :py:class:`~scistanpy.model.model.Model` class expose a number
of methods useful for Bayesian inference, including methods to facilitate:

    - MCMC diagnostics and convergence assessment
    - Posterior summary statistics
    - Visualization integration
    - Export utilities for downstream analysis

Models are constructed from building blocks called components, which fall under
three main categories:

    - :py:class:`Constants <scistanpy.model.components.constants.Constant>`,
      which represent fixed values and hyperparameters in a SciStanPy model.
    - :py:class:`Parameters <scistanpy.model.components.parameters.Parameter>`,
      which represent random variables. These are either inferred (i.e., latent
      parameters) or directly modeled (i.e., observed variables).
    - :py:class:`Transformed Parameters <scistanpy.model.components.transformations.transformed_parameters.TransformedParameter>`,
      which, as the name suggests, are the result of deterministic transformations
      of Parameters. These result from the :py:mod:`scistanpy.operations` module.

A typical workflow using the :py:class:`~scistanpy.model.model.Model` class looks
like this:

    1. **Model Definition**: Instantiate Model class and add components. On initialization,
       the model will automatically register its components.
    2. **Prior Predictive Checks**: Use the
       :py:meth:`Model.prior_predictive() <scistanpy.model.model.Model.prior_predictive>`
       method to evaluate and set values for model hyperparameters.
    3. **Compilation**: Compilation to either Stan code or a PyTorch ``nn.Module``.
    4. **Sampling**: Fit a model using either Hamiltonian Monte Carlo (HMC) via Stan
       or Maximum Likelihood Estimation (MLE) via PyTorch. Draw samples from the
       resulting likelihood/posterior.
    5. **Analysis**: Process results using built-in diagnostic tools.

Example:
    >>> import scistanpy as ssp
    >>> # Create a simple regression model
    >>> class Regressor(ssp.Model):
    >>>     def __init__(self):
    >>>         super().__init__()
    >>>         self.mu = ssp.parameters.Normal(0.0, 1.0)
    >>>         self.sigma = ssp.parameters.HalfNormal(1.0)
    >>>         self.y = ssp.parameters.Normal(self.mu, self.sigma)
    >>> # Create a model instance
    >>> model = Regressor()
    >>> # Prior predictive checks in Jupyter
    >>> model.prior_predictive()
    >>> # Run mcmc using Stan
    >>> res = model.mcmc(data = {"y": data})
    >>> # Run maximum likelihood estimation with PyTorch
    >>> mle = model.mle(data = {"y": data})
"""
