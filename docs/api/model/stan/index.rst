Stan Integration API Reference
===============================

This reference covers the Stan probabilistic programming language integration submodule in SciStanPy.

The Stan submodule provides comprehensive integration with the Stan probabilistic programming language, including automatic code generation, compilation management, and enhanced sampling interfaces.

Stan Submodule Overview
-----------------------

The Stan integration consists of several key components:

.. toctree::
   :maxdepth: 2

   stan_functions
   stan_model


Core Stan Integration
--------------------

.. automodule:: scistanpy.model.stan
   :members:
   :undoc-members:
   :show-inheritance:

Key Features
-----------

**Automatic Code Generation**
   Complete translation from SciStanPy models to optimized Stan programs

**Enhanced CmdStanPy Integration**
   Extended CmdStanModel with automatic data gathering and result processing

**Specialized Function Libraries**
   Comprehensive Stan functions for scientific distributions and operations

**Performance Optimization**
   Intelligent loop organization, singleton elimination, and memory management

**Comprehensive Validation**
   Data validation, shape checking, and error reporting throughout the workflow

Stan Workflow Integration
------------------------

**Complete Modeling Pipeline:**

.. code-block:: python

   import scistanpy as ssp
   import numpy as np

   # 1. Define SciStanPy model
   class ScientificModel(ssp.Model):
       def __init__(self, predictors, outcomes):
           super().__init__(default_data={'outcomes': outcomes})

           # Model parameters
           self.intercept = ssp.parameters.Normal(mu=0, sigma=5)
           self.slope = ssp.parameters.Normal(mu=0, sigma=2)
           self.sigma = ssp.parameters.LogNormal(mu=0, sigma=1)

           # Linear predictor
           self.mu = self.intercept + self.slope * predictors

           # Likelihood
           self.outcomes = ssp.parameters.Normal(
               mu=self.mu, sigma=self.sigma, observable=True
           )

   # 2. Instantiate model
   model = ScientificModel(x_data, y_data)

   # 3. Compile to Stan
   stan_model = model.to_stan(
       output_dir='./stan_cache',
       force_compile=False,
       model_name='scientific_regression'
   )

   # 4. Enhanced sampling
   results = stan_model.sample(
       chains=4,
       iter_sampling=2000,
       iter_warmup=1000,
       inits='prior',  # Use prior-based initialization
       precision='single'  # Memory optimization
   )

   # 5. Comprehensive analysis
   summary = results.summary()
   diagnostics = results.diagnose()

Code Generation Process
----------------------

**Automatic Stan Program Assembly:**

The Stan integration automatically generates complete Stan programs through a sophisticated compilation process:

1. **Dependency Analysis**: Builds component dependency graphs
2. **Depth Assignment**: Determines hierarchical nesting levels
3. **Loop Organization**: Creates optimized for-loop structures
4. **Block Generation**: Generates all required Stan program blocks
5. **Code Formatting**: Applies Stan canonical formatting

**Generated Stan Structure:**

.. code-block:: stan

   functions {
       // Automatically included based on model components
       #include <multinomial.stanfunctions>
       // Additional function libraries as needed
   }

   data {
       // Observable parameters (user-provided)
       // Constants (auto-gathered from model)
   }

   transformed data {
       // Data preprocessing and transformations
   }

   parameters {
       // Model parameters for MCMC sampling
   }

   transformed parameters {
       // Derived quantities and transformations
   }

   model {
       // Prior and likelihood statements
       // Optimized loop structures
   }

   generated quantities {
       // Posterior predictive sampling
       // Additional derived quantities
   }

Advanced Stan Features
---------------------

**Loop Optimization:**

.. code-block:: python

   # SciStanPy automatically optimizes Stan loops

   # Multi-dimensional parameters generate efficient loops:
   group_effects = ssp.parameters.Normal(
       mu=global_mean,
       sigma=global_sigma,
       shape=(n_groups, n_timepoints)
   )

   # Generates optimized nested for-loops in Stan:
   # for (i in 1:n_groups) {
   #     for (j in 1:n_timepoints) {
   #         group_effects[i, j] ~ normal(global_mean, global_sigma);
   #     }
   # }

**Memory Management:**

.. code-block:: python

   # Large dataset handling with chunking
   results = stan_model.sample(
       data=large_dataset,
       precision='single',    # Use single precision
       mib_per_chunk=128,    # Control memory usage
       use_dask=True         # Enable distributed processing
   )

**Prior Initialization:**

.. code-block:: python

   # Automatic prior-based initialization for better convergence
   results = stan_model.sample(
       inits='prior',  # Draw initial values from prior
       chains=8,
       adapt_delta=0.95
   )

Enhanced CmdStanPy Integration
-----------------------------

**Automatic Data Gathering:**

.. code-block:: python

   # SciStanPy automatically gathers required data

   # User provides only observable data:
   results = stan_model.sample(data={'y': observed_values})

   # Constants and hyperparameters automatically included:
   # - Model constants from SciStanPy model
   # - Hyperparameter values
   # - Proper shape validation

**Enhanced Methods:**

All CmdStanPy methods are enhanced with SciStanPy integration:

.. code-block:: python

   # Enhanced sampling
   mcmc_results = stan_model.sample(data={'y': observations})

   # Enhanced optimization (experimental)
   mle_results = stan_model.optimize(data={'y': observations})

   # Enhanced variational inference (experimental)
   vi_results = stan_model.variational(data={'y': observations})

Stan Function Libraries
----------------------

**Comprehensive Function Support:**

SciStanPy includes specialized Stan function libraries:

- **Multinomial Distributions**: Multiple parameterizations for count data
- **Exp-Transformed Distributions**: Log-space distributions for numerical stability
- **Growth Models**: Exponential and logistic growth functions
- **Sequence Operations**: Convolution and pattern matching

**Automatic Function Inclusion:**

.. code-block:: python

   # Functions automatically included based on model components

   # Using MultinomialLogit automatically includes multinomial functions
   counts = ssp.parameters.MultinomialLogit(
       n=100, logits=log_probs, observable=True
   )

   # Using ExpExponential includes exp-exponential functions
   log_lifetime = ssp.parameters.ExpExponential(rate=1.0, observable=True)

Integration with Scientific Workflows
------------------------------------

**Model Comparison:**

.. code-block:: python

   # Compare multiple models using Stan backend
   models = {
       'linear': LinearModel(x, y),
       'quadratic': QuadraticModel(x, y),
       'cubic': CubicModel(x, y)
   }

   results = {}
   for name, model in models.items():
       stan_model = model.to_stan(model_name=name)
       results[name] = stan_model.sample(chains=4, iter_sampling=1000)

   # Model comparison using information criteria
   for name, result in results.items():
       print(f"{name}: LOO = {result.loo().loo}")

**Simulation Studies:**

.. code-block:: python

   # Simulate and recover parameters
   def simulation_study(true_params, n_sims=100):
       recovery_results = []

       for sim in range(n_sims):
           # Generate synthetic data
           model = create_model(true_params)
           sim_data = model.simulate()

           # Fit model to synthetic data
           stan_model = model.to_stan()
           fit_results = stan_model.sample(data=sim_data)

           # Check parameter recovery
           recovery_results.append(fit_results.summary())

       return recovery_results

**Sensitivity Analysis:**

.. code-block:: python

   # Analyze sensitivity to prior specifications
   def prior_sensitivity_analysis(base_model, prior_variants):
       results = {}

       for variant_name, prior_config in prior_variants.items():
           # Modify model priors
           model = base_model.copy()
           model.update_priors(prior_config)

           # Compile and sample
           stan_model = model.to_stan(model_name=f'variant_{variant_name}')
           results[variant_name] = stan_model.sample()

       return results

Performance and Optimization
---------------------------

**Compilation Caching:**

.. code-block:: python

   # Efficient model reuse with caching
   stan_model = model.to_stan(
       output_dir='./model_cache',
       force_compile=False  # Reuse cached compilation
   )

**Parallel Processing:**

.. code-block:: python

   # Multi-chain parallel sampling
   results = stan_model.sample(
       chains=8,
       parallel_chains=True,
       threads_per_chain=2
   )

**Memory Optimization:**

.. code-block:: python

   # Memory-efficient sampling for large models
   results = stan_model.sample(
       precision='single',     # Reduce memory footprint
       mib_per_chunk=64,      # Control chunk sizes
       save_warmup=False      # Don't save warmup samples
   )

Error Handling and Debugging
---------------------------

**Comprehensive Validation:**

.. code-block:: python

   # Model validation during compilation
   try:
       stan_model = model.to_stan()
   except ValueError as e:
       print(f"Model compilation failed: {e}")
       print("Check model specification and dependencies")

**Stan Code Inspection:**

.. code-block:: python

   # Inspect generated Stan code for debugging
   print("Generated Stan program:")
   print(stan_model.code())

   # Access Stan file for external inspection
   print(f"Stan file location: {stan_model.stan_program_path}")

**Data Validation:**

.. code-block:: python

   # Comprehensive data validation
   try:
       results = stan_model.sample(data={'y': observations})
   except ValueError as e:
       print(f"Data validation failed: {e}")
       # Detailed error messages for shape mismatches, missing data, etc.

Best Practices
-------------

1. **Use output directories** for Stan file management and debugging
2. **Enable prior initialization** for improved MCMC convergence
3. **Cache compiled models** for repeated use and development
4. **Monitor memory usage** with appropriate precision settings
5. **Validate data thoroughly** before large sampling runs
6. **Inspect generated Stan code** for understanding and optimization
7. **Use parallel processing** for computationally intensive models
8. **Implement proper error handling** for robust scientific workflows

The Stan integration provides a comprehensive bridge between SciStanPy's intuitive modeling interface and Stan's powerful probabilistic programming capabilities, enabling sophisticated scientific modeling with minimal complexity.
