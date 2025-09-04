Model Components API Reference
===============================

This reference covers the comprehensive model component framework in SciStanPy.

The model components submodule provides the foundational building blocks for constructing probabilistic models. It includes abstract interfaces, concrete parameter implementations, mathematical transformations, and specialized probability distributions.

Model Components Submodule Overview
------------------------------------

The model components submodule consists of several key areas:

.. toctree::
   :maxdepth: 1

   abstract_model_component
   constants
   custom_distributions/index
   parameters
   transformations/index


Component Architecture
---------------------

.. automodule:: scistanpy.model.components
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Component Hierarchy
-------------------

**Foundation Layer**
   Abstract base classes that define the core component interface and behavior patterns

**Parameter Layer**
   Concrete implementations of probability distributions and statistical parameters

**Transformation Layer**
   Mathematical operations and transformations that can be applied to parameters

**Specialization Layer**
   Custom distributions and domain-specific components for advanced modeling

Key Design Principles
--------------------

**Compositional Design:**

All components follow a compositional design that enables complex model construction through combination of simple elements:

.. code-block:: python

   import scistanpy as ssp
   import numpy as np

   # Basic components
   intercept = ssp.parameters.Normal(mu=0, sigma=5)
   slope = ssp.parameters.Normal(mu=0, sigma=2)
   noise = ssp.parameters.LogNormal(mu=0, sigma=1)

   # Composition through mathematical operations
   x_data = np.linspace(0, 10, 50)
   linear_predictor = intercept + slope * x_data

   # Further composition
   likelihood = ssp.parameters.Normal(mu=linear_predictor, sigma=noise)

**Automatic Dependency Tracking:**

Components automatically track their relationships to enable proper Stan code generation and sampling:

.. code-block:: python

   # Dependencies are tracked automatically
   print(f"Linear predictor depends on: {[p.model_varname for p in linear_predictor.parents]}")
   print(f"Intercept is used by: {[c.model_varname for c in intercept.children]}")

**Multi-Backend Support:**

All components work consistently across NumPy, PyTorch, and Stan backends:

.. code-block:: python

   # Same model, different backends
   samples_numpy, _ = likelihood.draw(n=100)           # NumPy backend
   likelihood.init_pytorch()                           # Switch to PyTorch
   tensor_samples = likelihood.torch_parametrization   # PyTorch tensors
   stan_model = ssp.Model(likelihood).to_stan()        # Stan backend

Component Types
--------------

**Parameters:**
   Probability distributions representing unknown quantities to be inferred

.. code-block:: python

   # Various parameter types
   continuous_param = ssp.parameters.Normal(mu=0, sigma=1)
   discrete_param = ssp.parameters.Poisson(lambda_=5)
   simplex_param = ssp.parameters.Dirichlet(alpha=[1, 1, 1])
   positive_param = ssp.parameters.LogNormal(mu=0, sigma=1)

**Constants:**
   Fixed values that provide structure and constraints to models

.. code-block:: python

   # Constants with different characteristics
   data_matrix = ssp.constants.Constant(X_data)
   hyperparameter = ssp.constants.Constant(2.5, lower_bound=0, upper_bound=10)
   design_point = ssp.constants.Constant([1, 0, 1], enforce_uniformity=False)

**Transformations:**
   Mathematical operations that create derived quantities

.. code-block:: python

   # Mathematical transformations
   base_param = ssp.parameters.Normal(mu=0, sigma=1)
   exp_transform = ssp.operations.exp(base_param)      # Exponential transformation
   standardized = (base_param - mu_hat) / sigma_hat    # Arithmetic operations
   indexed = matrix_param[1:3, :]                     # Array indexing

**Custom Distributions:**
   Specialized probability distributions for domain-specific modeling

.. code-block:: python

   # Custom distributions for advanced modeling
   log_wealth = ssp.parameters.ExpLomax(lambda_=1.0, alpha=1.5)    # Heavy-tailed
   log_proportions = ssp.parameters.ExpDirichlet(alpha=[1, 2, 3])  # Log-simplex
   counts = ssp.parameters.MultinomialLogit(gamma=logits, N=100)   # Unconstrained

Component Relationships
----------------------

**Parent-Child Dependencies:**

Components form directed acyclic graphs representing dependency relationships:

.. code-block:: python

   # Build dependency relationships
   global_mean = ssp.parameters.Normal(mu=0, sigma=5)
   group_effects = ssp.parameters.Normal(
       mu=global_mean,  # Parent relationship
       sigma=1.0,
       shape=(10,)
   )
   observations = ssp.parameters.Normal(
       mu=group_effects,  # Another parent relationship
       sigma=0.5
   )

   # Analyze relationships
   def show_dependencies(component, level=0):
       indent = "  " * level
       print(f"{indent}{component.model_varname}")
       for parent in component.parents:
           show_dependencies(parent, level + 1)

   show_dependencies(observations)

**Shape Broadcasting:**

Components automatically handle shape broadcasting following NumPy conventions:

.. code-block:: python

   # Automatic shape inference and broadcasting
   scalar_param = ssp.parameters.Normal(mu=0, sigma=1)          # Shape: ()
   vector_param = ssp.parameters.Normal(mu=0, sigma=1, shape=(5,)) # Shape: (5,)

   # Broadcasting in operations
   broadcasted = scalar_param + vector_param  # Result shape: (5,)

   # Multi-dimensional broadcasting
   matrix_param = ssp.parameters.Normal(mu=0, sigma=1, shape=(3, 5))
   result = vector_param + matrix_param  # Result shape: (3, 5)

Stan Code Generation
-------------------

**Automatic Code Generation:**

Components automatically generate appropriate Stan code for all program blocks:

.. code-block:: python

   # Components generate Stan code automatically
   model = ssp.Model(observations)
   stan_model = model.to_stan()

   # Inspect generated code
   print("Generated Stan program:")
   print(stan_model.code())

**Block Organization:**

Components are automatically organized into appropriate Stan program blocks:

.. code-block:: python

   # Example of generated Stan structure:
   """
   data {
       // Observable parameters and constants
   }

   parameters {
       // Unobserved parameters for inference
   }

   transformed parameters {
       // Derived quantities and transformations
   }

   model {
       // Prior and likelihood statements
   }

   generated quantities {
       // Posterior predictive samples
   }
   """

**Loop Optimization:**

Multi-dimensional components generate optimized Stan loops:

.. code-block:: python

   # Multi-dimensional parameter
   param_3d = ssp.parameters.Normal(mu=0, sigma=1, shape=(4, 5, 3))

   # Generates optimized Stan loops:
   # for (i in 1:4) {
   #     for (j in 1:5) {
   #         param_3d[i,j] ~ normal(0, 1);  // Last dimension vectorized
   #     }
   # }

Advanced Component Features
--------------------------

**Observable Parameters:**

Parameters can be marked as observable to represent known data:

.. code-block:: python

   # Observable parameters represent data
   observed_data = np.array([1.2, 2.1, 1.8, 2.3])
   likelihood = ssp.parameters.Normal(mu=predicted_mean, sigma=error_std)
   likelihood.observe(observed_data)  # Mark as observable

   # Or use the .as_observable() method
   observations = ssp.parameters.Normal(mu=mu, sigma=sigma).as_observable()

**Non-Centered Parameterization:**

Hierarchical parameters automatically use non-centered parameterization for better MCMC performance:

.. code-block:: python

   # Automatic non-centered parameterization
   mu_global = ssp.parameters.Normal(mu=0, sigma=5)
   sigma_global = ssp.parameters.LogNormal(mu=0, sigma=1)

   # Automatically non-centered in Stan:
   # z_raw ~ std_normal();
   # z = mu_global + sigma_global * z_raw;
   group_effects = ssp.parameters.Normal(
       mu=mu_global,
       sigma=sigma_global,
       shape=(10,)
   )

**Interactive Support:**

Constants can be made interactive for model exploration:

.. code-block:: python

   # Interactive constants for exploration
   interactive_prior = ssp.constants.Constant(
       1.0,
       lower_bound=0.1,
       upper_bound=5.0,
       togglable=True  # Enable interactive widgets
   )

Component Validation
-------------------

**Automatic Validation:**

Components perform comprehensive validation during construction:

.. code-block:: python

   # Automatic parameter validation
   try:
       invalid_param = ssp.parameters.Beta(
           alpha=-1,  # Must be positive
           beta=2
       )
   except ValueError as e:
       print(f"Parameter validation failed: {e}")

   # Shape compatibility validation
   try:
       incompatible = ssp.parameters.Normal(
           mu=np.zeros((3, 4)),
           sigma=np.ones((5, 2)),  # Incompatible shape
       )
   except ValueError as e:
       print(f"Shape compatibility failed: {e}")

**Constraint Enforcement:**

Components automatically enforce distributional constraints:

.. code-block:: python

   # Constraint checking during sampling
   positive_param = ssp.parameters.Gamma(alpha=2, beta=1)
   samples, _ = positive_param.draw(n=100)
   assert np.all(samples >= 0)  # Automatic constraint enforcement

   # Simplex constraint enforcement
   simplex_param = ssp.parameters.Dirichlet(alpha=[1, 1, 1])
   simplex_samples, _ = simplex_param.draw(n=100)
   assert np.allclose(simplex_samples.sum(axis=-1), 1)

Performance Considerations
-------------------------

**Efficient Construction:**

.. code-block:: python

   # Efficient: Single multi-dimensional parameter
   efficient = ssp.parameters.Normal(mu=0, sigma=1, shape=(100, 50))

   # Less efficient: Many individual parameters
   # inefficient = [[ssp.parameters.Normal(mu=0, sigma=1)
   #                 for j in range(50)] for i in range(100)]

**Memory Management:**

.. code-block:: python

   # Share parent components to reduce memory usage
   shared_hyperprior = ssp.parameters.LogNormal(mu=0, sigma=1)

   # Multiple parameters sharing the same parent
   group_variances = [
       ssp.parameters.Normal(mu=0, sigma=shared_hyperprior)
       for _ in range(10)
   ]  # shared_hyperprior referenced, not copied

**Computation Optimization:**

.. code-block:: python

   # Components are optimized for different backends
   param = ssp.parameters.Normal(mu=0, sigma=1, shape=(1000, 100))

   # NumPy backend: CPU-optimized sampling
   samples, _ = param.draw(n=100)

   # PyTorch backend: GPU acceleration available
   param.init_pytorch()
   gpu_samples = param.torch_parametrization

   # Stan backend: Compiled C++ performance
   model = ssp.Model(param)
   stan_results = model.mcmc()

Best Practices
-------------

1. **Use appropriate component types** for your modeling needs
2. **Leverage automatic shape broadcasting** instead of manual management
3. **Build models compositionally** from simple to complex
4. **Validate component relationships** before running inference
5. **Use descriptive variable names** for model interpretability
6. **Take advantage of automatic Stan optimizations** like non-centering
7. **Monitor memory usage** in large hierarchical models
8. **Test component behavior** with prior predictive simulation

The model components framework provides a comprehensive foundation for probabilistic modeling that scales from simple parameter estimation to complex hierarchical models while maintaining mathematical rigor and computational efficiency.
