Operations API Reference
========================

This reference covers the mathematical operations and transformations available in SciStanPy.

Operations Module
-----------------

.. automodule:: scistanpy.operations
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Operation Framework
-------------------

The operations module provides a framework for creating mathematical operations that work with both SciStanPy model components and raw numerical data.

**Core Classes:**

.. autoclass:: scistanpy.operations.MetaOperation
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.operations.Operation
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

**Operation Builder:**

.. autofunction:: scistanpy.operations.build_operation
   :noindex:

Available Operations
--------------------

Mathematical Functions
~~~~~~~~~~~~~~~~~~~~~~

.. autodata:: scistanpy.operations.abs_
   :annotation:

   **Usage:**

   .. code-block:: python

      # With model components
      param = ssp.parameters.Normal(mu=0, sigma=1)
      abs_param = ssp.operations.abs_(param)

      # With numerical data
      result = ssp.operations.abs_([-1, -2, 3])  # Returns [1, 2, 3]

.. autodata:: scistanpy.operations.exp
   :annotation:

   **Usage:**

   .. code-block:: python

      # Exponential transformation
      log_rate = ssp.parameters.Normal(mu=0, sigma=1)
      rate = ssp.operations.exp(log_rate)  # Ensures positive values

.. autodata:: scistanpy.operations.log
   :annotation:

   **Usage:**

   .. code-block:: python

      # Natural logarithm
      positive_param = ssp.parameters.LogNormal(mu=0, sigma=1)
      log_param = ssp.operations.log(positive_param)

.. autodata:: scistanpy.operations.log1p_exp
   :annotation:

   Numerically stable computation of log(1 + exp(x)).

.. autodata:: scistanpy.operations.sigmoid
   :annotation:

   **Usage:**

   .. code-block:: python

      # Convert logits to probabilities
      logits = ssp.parameters.Normal(mu=0, sigma=1)
      probabilities = ssp.operations.sigmoid(logits)

.. autodata:: scistanpy.operations.log_sigmoid
   :annotation:

   Numerically stable computation of log(sigmoid(x)).

Statistical Operations
~~~~~~~~~~~~~~~~~~~~~~

.. autodata:: scistanpy.operations.normalize
   :annotation:

   **Usage:**

   .. code-block:: python

      # Normalize to unit sum (over last dimension)
      weights = ssp.parameters.LogNormal(mu=0, sigma=1, shape=(3,))
      probabilities = ssp.operations.normalize(weights)

.. autodata:: scistanpy.operations.normalize_log
   :annotation:

   Log-space normalization ensuring log-sum-exp equals 0.

.. autodata:: scistanpy.operations.sum_
   :annotation:

   **Usage:**

   .. code-block:: python

      # Sum over last dimension
      values = ssp.parameters.Normal(mu=0, sigma=1, shape=(5,))
      total = ssp.operations.sum_(values)

.. autodata:: scistanpy.operations.logsumexp
   :annotation:

   Numerically stable computation of log(sum(exp(x))).

Growth Model Operations
~~~~~~~~~~~~~~~~~~~~~~~

.. autodata:: scistanpy.operations.exponential_growth
   :annotation:

   **Usage:**

   .. code-block:: python

      # Model exponential population growth
      time_points = np.array([0, 1, 2, 3, 4])
      initial_pop = ssp.parameters.LogNormal(mu=np.log(100), sigma=0.1)
      growth_rate = ssp.parameters.Normal(mu=0.1, sigma=0.05)

      population = ssp.operations.exponential_growth(
          t=time_points, A=initial_pop, r=growth_rate
      )

.. autodata:: scistanpy.operations.log_exponential_growth
   :annotation:

   Log-space version of exponential growth modeling.

.. autodata:: scistanpy.operations.binary_exponential_growth
   :annotation:

   **Usage:**

   .. code-block:: python

      # Two-timepoint exponential growth
      initial_size = ssp.parameters.LogNormal(mu=np.log(50), sigma=0.2)
      growth_rate = ssp.parameters.Normal(mu=0.2, sigma=0.1)

      final_size = ssp.operations.binary_exponential_growth(
          A=initial_size, r=growth_rate
      )

.. autodata:: scistanpy.operations.binary_log_exponential_growth
   :annotation:

   Log-space version of binary exponential growth.

.. autodata:: scistanpy.operations.sigmoid_growth
   :annotation:

   **Usage:**

   .. code-block:: python

      # Logistic growth with carrying capacity
      time_points = np.array([0, 5, 10, 15, 20])
      carrying_capacity = ssp.parameters.LogNormal(mu=np.log(1000), sigma=0.1)
      growth_rate = ssp.parameters.Normal(mu=0.3, sigma=0.1)
      inflection_time = ssp.parameters.Normal(mu=10, sigma=2)

      population = ssp.operations.sigmoid_growth(
          t=time_points, A=carrying_capacity, r=growth_rate, c=inflection_time
      )

.. autodata:: scistanpy.operations.log_sigmoid_growth
   :annotation:

   Log-space version of sigmoid growth modeling.

.. autodata:: scistanpy.operations.sigmoid_growth_init_param
   :annotation:

   Alternative sigmoid growth parameterization using initial population.

.. autodata:: scistanpy.operations.log_sigmoid_growth_init_param
   :annotation:

   Log-space version of initial-population-parameterized sigmoid growth.

Specialized Operations
~~~~~~~~~~~~~~~~~~~~~~

.. autodata:: scistanpy.operations.convolve_sequence
   :annotation:

   **Usage:**

   .. code-block:: python

      # Sequence convolution for pattern matching
      sequence = np.array([0, 1, 2, 1, 0])  # Encoded sequence
      weights = ssp.parameters.Normal(mu=0, sigma=1, shape=(3, 3))

      convolved = ssp.operations.convolve_sequence(
          weights=weights, ordinals=sequence
      )

Operation Usage Patterns
------------------------

**With Model Components:**

.. code-block:: python

   # Operations with parameters create transformed parameters
   base_param = ssp.parameters.Normal(mu=0, sigma=1)
   transformed = ssp.operations.exp(base_param)

   # Use in model definitions
   likelihood = ssp.parameters.Normal(mu=transformed, sigma=0.1)

**With Numerical Data:**

.. code-block:: python

   # Operations with numerical data compute immediately
   import numpy as np

   data = np.array([1, 2, 3])
   result = ssp.operations.exp(data)  # Returns exp([1, 2, 3])

**Intelligent Dispatching:**

.. code-block:: python

   # Operations automatically detect input types
   def my_function(x):
       return ssp.operations.log(ssp.operations.exp(x) + 1)

   # Works with both parameters and data
   param_result = my_function(ssp.parameters.Normal(mu=0, sigma=1))
   data_result = my_function(np.array([1, 2, 3]))

Creating Custom Operations
--------------------------

**Using build_operation:**

.. code-block:: python

   from scistanpy.operations import build_operation
   from scistanpy.model.components.transformations.transformed_parameters import UnaryTransformedParameter

   class MyTransformation(UnaryTransformedParameter):
       """Custom mathematical transformation."""

       def run_np_torch_op(self, x):
           # Implementation for numerical data
           return x**2 + 1

       def write_stan_operation(self, x: str) -> str:
           # Stan code generation
           return f"square({x}) + 1"

   # Create operation
   my_operation = build_operation(MyTransformation)

   # Use the operation
   param = ssp.parameters.Normal(mu=0, sigma=1)
   transformed = my_operation(param)

Best Practices
--------------

1. **Use descriptive variable names** when chaining operations
2. **Prefer log-space operations** for numerical stability when dealing with extreme values
3. **Validate inputs** when creating custom operations
4. **Document custom operations** with clear mathematical descriptions
5. **Test operations** with both model components and numerical data
6. **Use growth model operations** for temporal modeling applications
7. **Combine operations naturally** to build complex mathematical expressions

The operations framework provides the foundation for building sophisticated mathematical models while maintaining computational efficiency across different backends.
