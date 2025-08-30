Model API Reference
===================

This reference covers the core Model class for building and executing Bayesian models in SciStanPy.

Model Module
------------

.. automodule:: scistanpy.model.model
   :members:
   :undoc-members:
   :show-inheritance:

Core Model Class
---------------

.. autoclass:: scistanpy.model.model.Model
   :members:
   :undoc-members:
   :show-inheritance:

   **Model Construction Pattern:**

   The Model class uses a metaclass pattern for automatic component registration:

   .. code-block:: python

      import scistanpy as ssp
      import numpy as np

      class MyModel(ssp.Model):
          def __init__(self):
              super().__init__()

              # Components are automatically registered
              self.mu = ssp.parameters.Normal(mu=0, sigma=1)
              self.sigma = ssp.parameters.LogNormal(mu=0, sigma=0.5)
              self.y = ssp.parameters.Normal(mu=self.mu, sigma=self.sigma, observable=True)

      # Create and use the model
      model = MyModel()
      prior_samples = model.draw(n=1000)

   **Multiple Backend Support:**

   .. code-block:: python

      # PyTorch backend for MLE
      mle_result = model.mle(data=observed_data, device='cuda')

      # Stan backend for MCMC
      mcmc_result = model.mcmc(data=observed_data, chains=4, iter_sampling=2000)

   **Interactive Prior Exploration:**

   .. code-block:: python

      # Create interactive dashboard
      dashboard = model.prior_predictive()
      # Display in Jupyter notebook or serve as web app

Model Utilities
--------------

.. autofunction:: scistanpy.model.model.model_comps_to_dict
   :noindex:

   **Example Usage:**

   .. code-block:: python

      # Convert model components to dictionary
      components = [param1, param2, observable1]
      comp_dict = model_comps_to_dict(components)
      # Access by name: comp_dict['param1']

.. autofunction:: scistanpy.model.model.run_delayed_mcmc
   :noindex:

   **Delayed Execution Workflow:**

   .. code-block:: python

      # First, create delayed run configuration
      model.mcmc(
          output_dir="./mcmc_output",
          delay_run=True,
          chains=4,
          iter_sampling=2000
      )

      # Later, execute the run (possibly on different machine/process)
      results = run_delayed_mcmc("./mcmc_output/model-delay.pkl")

Model Properties and Methods
---------------------------

**Component Access Properties:**

The Model class provides several properties for accessing different types of components:

.. code-block:: python

   # Access different component types
   print(f"Parameters: {list(model.parameter_dict.keys())}")
   print(f"Observables: {list(model.observable_dict.keys())}")
   print(f"Constants: {list(model.constant_dict.keys())}")
   print(f"Hyperparameters: {list(model.hyperparameter_dict.keys())}")

**Sampling and Inference Methods:**

.. code-block:: python

   # Prior sampling
   prior_samples = model.draw(n=1000, as_xarray=True)

   # Maximum likelihood estimation
   mle_fit = model.mle(
       data=observed_data,
       epochs=10000,
       lr=0.01,
       device='cuda'
   )

   # MCMC sampling
   posterior_samples = model.mcmc(
       data=observed_data,
       chains=4,
       iter_sampling=2000,
       iter_warmup=1000
   )

**Simulation Methods:**

.. code-block:: python

   # Simulate data and fit with MLE
   sim_data, mle_result = model.simulate_mle(epochs=5000)

   # Simulate data and fit with MCMC
   sim_data, mcmc_result = model.simulate_mcmc(chains=4, iter_sampling=1000)

Model Introspection
------------------

**Component Inspection:**

.. code-block:: python

   # Check if parameter exists
   if 'mu' in model:
       mu_param = model['mu']
       print(f"Parameter type: {type(mu_param)}")

   # Get all component names
   all_names = list(model.all_model_components_dict.keys())
   named_only = list(model.named_model_components_dict.keys())

**Model Structure:**

.. code-block:: python

   # Print comprehensive model summary
   print(model)

   # Access specific component types
   for param in model.parameters:
       print(f"Parameter: {param.model_varname}")

   for observable in model.observables:
       print(f"Observable: {observable.model_varname}")

**Default Data Management:**

.. code-block:: python

   # Set default data for convenience
   model.default_data = {
       'y': observed_values,
       'x': predictor_values
   }

   # Now methods can use default data automatically
   mle_result = model.mle()  # Uses default_data
   mcmc_result = model.mcmc()  # Uses default_data

Advanced Features
----------------

**Custom Model Classes:**

.. code-block:: python

   class HierarchicalModel(ssp.Model):
       def __init__(self, groups, group_data):
           super().__init__()

           # Global parameters
           self.global_mu = ssp.parameters.Normal(mu=0, sigma=5)
           self.global_sigma = ssp.parameters.LogNormal(mu=0, sigma=1)

           # Group-specific parameters
           self.group_mus = ssp.parameters.Normal(
               mu=self.global_mu,
               sigma=self.global_sigma,
               shape=(len(groups),)
           )

           # Observations
           self.observations = ssp.parameters.Normal(
               mu=self.group_mus[groups],
               sigma=0.1,
               observable=True
           )

**Model Compilation and Caching:**

.. code-block:: python

   # Compile to PyTorch for optimization
   pytorch_model = model.to_pytorch(seed=42)

   # Compile to Stan for MCMC (with caching)
   stan_model = model.to_stan(
       output_dir='./compiled_models',
       force_compile=False  # Use cached version if available
   )

**Mixed Backend Workflows:**

.. code-block:: python

   # Use VI for initialization, then MCMC for final inference
   mle_result = model.mle(data=data, epochs=5000)

   # Use MLE results to initialize MCMC
   mcmc_result = model.mcmc(
       data=data,
       inits=mle_result.to_stan_inits(),
       chains=4,
       iter_sampling=2000
   )

Best Practices
-------------

1. **Use descriptive component names** that reflect their scientific meaning
2. **Set default data** for models with fixed datasets to streamline workflows
3. **Start with prior predictive checks** before fitting to real data
4. **Use simulation methods** to validate model implementation
5. **Leverage caching** for expensive Stan compilation with consistent output directories
6. **Choose appropriate backends** for different tasks (PyTorch for MLE, Stan for MCMC)
7. **Validate models incrementally** by building from simple to complex

The Model class provides the foundation for all Bayesian modeling workflows in SciStanPy, enabling seamless integration between model specification and computational backends.
