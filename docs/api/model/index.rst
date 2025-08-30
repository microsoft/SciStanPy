Model API Reference
===================

This reference covers the core Model class and related model components in SciStanPy.

The model submodule provides the primary interface for Bayesian model construction, compilation, and execution across multiple computational backends.

Model Submodule Overview
------------------------

The model submodule consists of several key components:

.. toctree::
   :maxdepth: 2

   components/index
   model
   nn_module
   results/index
   stan/index

Core Model Class
---------------

.. automodule:: scistanpy.model
   :members:
   :undoc-members:
   :show-inheritance:

Model Construction Framework
---------------------------

**Declarative Model Building:**

SciStanPy models are built using a declarative approach where model components are defined as class attributes:

.. code-block:: python

   import scistanpy as ssp
   import numpy as np

   class LinearRegressionModel(ssp.Model):
       def __init__(self, x_data):
           super().__init__()

           # Prior distributions for parameters
           self.intercept = ssp.parameters.Normal(mu=0, sigma=5)
           self.slope = ssp.parameters.Normal(mu=0, sigma=5)
           self.sigma = ssp.parameters.LogNormal(mu=0, sigma=1)

           # Linear predictor
           self.predictions = self.intercept + self.slope * x_data

           # Likelihood
           self.y = ssp.parameters.Normal(
               mu=self.predictions,
               sigma=self.sigma,
               observable=True
           )

**Multi-Backend Execution:**

.. code-block:: python

   # Create model instance
   x_data = np.linspace(0, 10, 50)
   model = LinearRegressionModel(x_data)

   # PyTorch backend for MLE
   mle_result = model.mle(data={'y': observed_y})

   # Stan backend for MCMC
   mcmc_result = model.mcmc(data={'y': observed_y})

   # Prior predictive checks
   prior_samples = model.draw(n=1000)

Key Features
-----------

**Automatic Component Registration**
   Model components are automatically discovered and registered through metaclass magic

**Multiple Computational Backends**
   Seamless integration with PyTorch (MLE/VI) and Stan (MCMC) backends

**Interactive Prior Exploration**
   Built-in dashboard for exploring prior distributions and model behavior

**Comprehensive Validation**
   Extensive error checking and validation throughout the modeling pipeline

**Efficient Compilation**
   Smart caching and compilation strategies for optimal performance

Model Component Types
--------------------

**Parameters**
   Probability distributions representing unknown quantities to be inferred

**Constants**
   Fixed values and hyperparameters that remain unchanged during inference

**Transformed Parameters**
   Deterministic functions of other model components

**Observables**
   Parameters that will be connected to observed data during inference

**Example Component Usage:**

.. code-block:: python

   class HierarchicalModel(ssp.Model):
       def __init__(self, group_sizes):
           super().__init__()

           # Constants
           self.n_groups = ssp.constants.Constant(len(group_sizes))

           # Hyperparameters
           self.global_mu = ssp.parameters.Normal(mu=0, sigma=10)
           self.global_sigma = ssp.parameters.LogNormal(mu=0, sigma=1)

           # Group-level parameters
           self.group_means = ssp.parameters.Normal(
               mu=self.global_mu,
               sigma=self.global_sigma,
               shape=(len(group_sizes),)
           )

           # Transformed parameter
           self.total_effect = ssp.operations.sum_(self.group_means)

           # Observables
           self.observations = ssp.parameters.Normal(
               mu=self.group_means,
               sigma=0.5,
               observable=True
           )

Workflow Integration
-------------------

**Complete Modeling Workflow:**

.. code-block:: python

   def complete_modeling_workflow(data):
       """Demonstrate complete SciStanPy workflow."""

       # 1. Model specification
       model = MyModel()

       # 2. Prior predictive checks
       dashboard = model.prior_predictive()
       # (Interactive exploration in Jupyter)

       # 3. Quick MLE for initial estimates
       mle_result = model.mle(data=data, epochs=5000)
       print(f"MLE estimates: {mle_result.estimates}")

       # 4. Full Bayesian inference
       mcmc_result = model.mcmc(
           data=data,
           chains=4,
           iter_sampling=2000
       )

       # 5. Model diagnostics
       mcmc_result.diagnose()

       # 6. Posterior analysis
       posterior_summary = mcmc_result.summary()

       return {
           'mle': mle_result,
           'mcmc': mcmc_result,
           'summary': posterior_summary
       }

**Simulation and Validation:**

.. code-block:: python

   # Simulate data and validate model
   sim_data, sim_mcmc = model.simulate_mcmc(chains=4, iter_sampling=1000)

   # Check parameter recovery
   true_params = sim_data  # True values used for simulation
   estimated_params = sim_mcmc.summary()

   for param in true_params:
       true_val = true_params[param]
       est_mean = estimated_params[param]['mean']
       est_ci = [estimated_params[param]['2.5%'], estimated_params[param]['97.5%']]

       print(f"{param}: true={true_val:.3f}, est={est_mean:.3f}, CI={est_ci}")

Advanced Features
----------------

**Custom Distributions:**

.. code-block:: python

   # Extend with domain-specific distributions
   class CustomModel(ssp.Model):
       def __init__(self):
           super().__init__()
           # Use custom distributions from SciStanPy
           self.shape_param = ssp.parameters.Gamma(alpha=2, beta=1)
           self.scale_param = ssp.parameters.InverseGamma(alpha=3, beta=2)

**Model Caching and Persistence:**

.. code-block:: python

   # Compile with caching for expensive models
   model = MyModel()

   # First compilation (slow)
   result1 = model.mcmc(
       output_dir='./model_cache',
       force_compile=False  # Use cache if available
   )

   # Subsequent runs (fast)
   result2 = model.mcmc(
       output_dir='./model_cache',
       force_compile=False  # Reuses compiled model
   )

**Delayed Execution:**

.. code-block:: python

   # Prepare MCMC for batch execution
   model.mcmc(
       data=data,
       delay_run=True,
       output_dir='./batch_jobs',
       chains=8,
       iter_sampling=5000
   )

   # Later, execute on compute cluster
   # results = run_delayed_mcmc('./batch_jobs/model-delay.pkl')

Performance Considerations
-------------------------

**Backend Selection Guidelines:**

- **PyTorch**: Fast MLE/VI, GPU acceleration, automatic differentiation
- **Stan**: Gold-standard MCMC, advanced samplers, robust diagnostics
- **NumPy**: Rapid prototyping, simple computations, debugging

**Optimization Tips:**

1. **Use MLE for initialization** to improve MCMC convergence
2. **Cache compiled models** to avoid recompilation overhead
3. **Choose appropriate device** (CPU vs GPU) based on model size
4. **Use simulation studies** to validate complex models
5. **Leverage prior predictive checks** before fitting real data

Error Handling and Debugging
----------------------------

**Common Issues and Solutions:**

.. code-block:: python

   # Handle naming conflicts
   try:
       model = MyModel()
   except ValueError as e:
       if "double underscores" in str(e):
           print("Fix parameter names to avoid double underscores")
       elif "starts with underscore" in str(e):
           print("Parameter names cannot start with underscore")

   # Validate data before inference
   try:
       model.default_data = data
   except ValueError as e:
       print(f"Data validation failed: {e}")
       # Check that all observables have corresponding data

Best Practices
-------------

1. **Start simple** and build complexity incrementally
2. **Use descriptive names** for model components
3. **Validate with simulation** before applying to real data
4. **Check prior implications** using prior predictive checks
5. **Compare multiple models** using information criteria
6. **Document model assumptions** and scientific interpretation
7. **Use version control** for model specifications

The Model API provides a unified interface for Bayesian modeling that scales from simple parameter estimation to complex hierarchical models across scientific domains.
   :noindex:

   Get a summary of the model structure.

   :returns: Human-readable model summary
   :rtype: str

.. automethod:: scistanpy.Model.plot_dag
   :noindex:

   Plot the directed acyclic graph of model dependencies.

   :param filename: Optional filename to save plot. Defaults to None.
   :type filename: Optional[str]

Model Serialization
------------------

.. automethod:: scistanpy.Model.save
   :noindex:

   Save model to disk.

   :param filename: Path to save model
   :type filename: str

.. automethod:: scistanpy.Model.load
   :noindex:

   Load model from disk.

   :param filename: Path to saved model
   :type filename: str

   :returns: Loaded model instance
   :rtype: Model

Utility Methods
--------------

.. automethod:: scistanpy.Model.copy
   :noindex:

   Create a deep copy of the model.

   :returns: Independent copy of the model
   :rtype: Model

.. automethod:: scistanpy.Model.sample_prior
   :noindex:

   Sample from parameter priors only.

   :param n_samples: Number of samples. Defaults to 1000.
   :type n_samples: int

   :returns: Prior samples for all parameters
   :rtype: dict[str, np.ndarray]

Performance Options
------------------

.. autoattribute:: scistanpy.Model.compile_options
   :annotation:

   Compilation options for Stan model.

.. autoattribute:: scistanpy.Model.parallel_chains
   :annotation:

   Whether to run chains in parallel.

This comprehensive API reference covers all aspects of the Model class, from basic usage to advanced features for model development and validation.
