Model API Reference
===================

This reference covers the Model implementation found in `scistanpy.model.model` and the model-related submodules used to build and run probabilistic models in SciStanPy.

The model submodule provides the primary interface for constructing models from composable components (parameters, constants, and transformations) and for running inference using the supported backends.

Model Submodule Overview
------------------------

.. toctree::
   :maxdepth: 4

   components/index
   model
   nn_module
   results/index
   stan/index

Core Module and Class
---------------------

The implementation of the Model class lives in the module
`scistanpy.model.model`.

How Models are Constructed
--------------------------

The Model class is designed to be subclassed. Model components (for example parameters and constants) are declared as instance attributes in __init__ and are discovered and registered automatically by the class machinery. This makes model definitions concise and declarative.

Example:

.. code-block:: python

   import scistanpy as ssp

   class SimpleModel(ssp.Model):
       def __init__(self, x):
           super().__init__()
           # register a prior parameter and an observable
           self.mu = ssp.parameters.Normal(mu=0.0, sigma=1.0)
           self.sigma = ssp.parameters.HalfNormal(scale=1.0)
           self.y = ssp.parameters.Normal(mu=self.mu, sigma=self.sigma, observable=True)

   model = SimpleModel(x_data)

Key Behaviors and API Surface
-----------------------------

- Automatic component registration: attributes that are `AbstractModelComponent` instances are assigned model variable names and collected during initialization.

- Introspection properties: the Model exposes typed properties such as `parameters`, `parameter_dict`, `transformed_parameters`, `observables`, `constant_dict`, and helpers like `named_model_components_dict` for programmatic inspection.

- Default data: models may carry `default_data` which is used by inference methods when explicit data is not provided.

- Core methods:
  - draw(n, ...): sample from priors for model components and return samples (supports returning xarray when requested).
  - to_pytorch(seed=None): convert the model to a PyTorch-backed `nn_module.PyTorchModel` for optimization or variational methods.
  - to_stan(...): compile or produce a `stan_model.StanModel` representation of the SciStanPy model.
  - mle(...): run maximum-likelihood (or optimization) using the PyTorch backend; returns an `mle_module.MLE` result object.
  - mcmc(...): generate Stan code, compile (with caching) and run MCMC sampling via the Stan backend; accepts options for compilation and sampling and returns a Stan-backed results object.

Notes on Backends
-----------------

- PyTorch is used for optimization / MLE and for converting models to a differentiable form via `to_pytorch`.

- Stan is used for MCMC sampling; the `mcmc` method produces Stan code from the model, compiles it (with configurable options), and runs sampling.

Documentation Pages
-------------------

See the subpages linked in the toctree for detailed information about model components, the PyTorch interface (`nn_module`), and Stan integration (the `stan` submodule).

Guidance
--------

- Build models incrementally and validate with prior draws before running expensive samplers.
- Use `mle` to obtain point estimates that can be used as sensible initializations for Stan in `mcmc` (for example by transforming MLE results into Stan inits when appropriate).
- Prefer explicit `data` arguments to `mle` / `mcmc` for reproducibility; `default_data` is a convenience but may hide sources of variation.

This page intentionally limits claims to behaviors implemented in the `scistanpy.model.model` implementation and refers readers to the component and backend pages for the complete API details.
