Transformed Data API Reference
==============================

This reference covers the transformed data functionality for efficient data preprocessing in SciStanPy.

Transformed Data Module
-----------------------

.. automodule:: scistanpy.model.components.transformations.transformed_data
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

TransformedData Class
---------------------

.. autoclass:: scistanpy.model.components.transformations.transformed_data.TransformedData
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Efficient Data Preprocessing:**

   The TransformedData class enables computationally expensive operations to be performed once in Stan's transformed data block rather than repeatedly during sampling:

   .. code-block:: python

      import scistanpy as ssp
      import numpy as np

      # Expensive computation that should be done once
      large_matrix = ssp.constants.Constant(np.random.randn(1000, 1000))

      # This computation will be moved to transformed data block
      matrix_inverse = ssp.transformations.TransformedData(
          operation=lambda x: np.linalg.inv(x),
          operand=large_matrix,
          stan_operation="inverse({operand})"
      )

   **Stan Code Generation:**

   .. code-block:: python

      # The transformed data generates:
      # transformed data {
      #     matrix[1000, 1000] matrix_inverse = inverse(large_matrix);
      # }
      #
      # Instead of computing the inverse at every MCMC iteration

Key Features
------------

**One-Time Computation:**

TransformedData components are evaluated once during Stan's data processing phase, making them ideal for:

- Matrix decompositions and inverses
- Expensive mathematical constants
- Data preprocessing and normalization
- Multinomial coefficient calculations

**Example Applications:**

.. code-block:: python

   # Matrix decomposition for efficient sampling
   covariance_matrix = ssp.constants.Constant(empirical_covariance)
   cholesky_decomp = ssp.transformations.TransformedData(
       operation=lambda cov: np.linalg.cholesky(cov),
       operand=covariance_matrix,
       stan_operation="cholesky_decompose({operand})"
   )

   # Multinomial normalization constants
   observed_counts = ssp.parameters.MultinomialLogTheta(
       log_theta=log_probabilities,
       N=100,
       observable=True
   )
   # Multinomial coefficient automatically computed in transformed data

**Stan Integration:**

.. code-block:: python

   # Automatic transformed data block generation
   model = ssp.Model(parameters_using_transformed_data)
   stan_model = model.to_stan()

   # Generated Stan includes:
   # transformed data {
   #     // Expensive computations here
   #     matrix[n, n] inv_cov = inverse(cov_matrix);
   #     real log_multinomial_coeff = lgamma(sum(counts) + 1) - sum(lgamma(counts + 1));
   # }

Automatic Usage in SciStanPy
----------------------------

**MultinomialLogTheta Optimization:**

.. code-block:: python

   # When using MultinomialLogTheta with observable=True:
   counts = ssp.parameters.MultinomialLogTheta(
       log_theta=log_probs,
       N=100,
       observable=True  # Triggers transformed data optimization
   )

   # SciStanPy automatically:
   # 1. Computes multinomial coefficient in transformed data
   # 2. Uses precomputed value in model block
   # 3. Avoids recomputing expensive lgamma functions

**Matrix Operations:**

.. code-block:: python

   # Large matrix operations are candidates for transformed data
   design_matrix = ssp.constants.Constant(X_data)  # Large design matrix

   # If used in computationally expensive ways:
   precision_matrix = design_matrix.T @ design_matrix
   # Could be moved to transformed data for efficiency

Custom Transformed Data
-----------------------

**Creating Custom Transformed Data:**

.. code-block:: python

   class CustomTransformedData(ssp.transformations.TransformedData):
       """Custom transformed data operation."""

       def __init__(self, input_data, **kwargs):
           def my_operation(data):
               # Expensive computation here
               return expensive_computation(data)

           super().__init__(
               operation=my_operation,
               operand=input_data,
               stan_operation="my_stan_function({operand})",
               **kwargs
           )

**Manual Transformed Data Creation:**

.. code-block:: python

   # For operations requiring custom Stan code
   complex_preprocessing = ssp.transformations.TransformedData(
       operation=lambda data: preprocess(data),
       operand=raw_data,
       stan_operation="""
       {
           // Custom Stan preprocessing code
           vector[N] processed_data;
           for (i in 1:N) {
               processed_data[i] = custom_transform(raw_data[i]);
           }
       }
       """
   )

Performance Benefits
--------------------

**Computational Efficiency:**

.. code-block:: python

   # Without transformed data: computed every MCMC iteration
   def slow_model():
       expensive_result = expensive_computation(data)  # Computed 4000+ times
       return likelihood_using(expensive_result)

   # With transformed data: computed once
   def fast_model():
       expensive_result = ssp.transformations.TransformedData(
           operation=expensive_computation,
           operand=data
       )  # Computed once in transformed data block
       return likelihood_using(expensive_result)

**Memory Efficiency:**

TransformedData operations can also help with memory management by preprocessing data into more efficient formats:

.. code-block:: python

   # Convert sparse data to dense format once
   sparse_matrix = ssp.constants.Constant(scipy_sparse_matrix)
   dense_matrix = ssp.transformations.TransformedData(
       operation=lambda x: x.toarray(),
       operand=sparse_matrix,
       stan_operation="to_matrix({operand})"
   )

Stan Code Generation Details
----------------------------

**Variable Declaration:**

.. code-block:: python

   # TransformedData automatically generates appropriate Stan types
   transformed_component = ssp.transformations.TransformedData(
       operation=matrix_operation,
       operand=matrix_input
   )

   # Generates Stan declaration based on operation result:
   # matrix[rows, cols] transformed_component = operation(matrix_input);

**Block Organization:**

.. code-block:: python

   # SciStanPy organizes transformed data block efficiently:
   # transformed data {
   #     // Declarations first
   #     matrix[n, n] inv_cov;
   #     real log_coeff;
   #
   #     // Computations in dependency order
   #     inv_cov = inverse(covariance_matrix);
   #     log_coeff = lgamma(sum(counts) + 1) - sum(lgamma(counts + 1));
   # }

Integration Patterns
--------------------

**With Model Components:**

.. code-block:: python

   # Transformed data integrates seamlessly with other components
   preprocessed_data = ssp.transformations.TransformedData(
       operation=preprocessing_function,
       operand=raw_data
   )

   # Use in parameter definitions
   regression_mean = design_matrix @ coefficients + preprocessed_data

**With Custom Distributions:**

.. code-block:: python

   # Use transformed data with custom distributions requiring expensive setup
   normalization_constant = ssp.transformations.TransformedData(
       operation=compute_normalization,
       operand=parameters
   )

   # Custom distribution using precomputed constant
   custom_likelihood = CustomDistribution(
       params=parameters,
       normalization=normalization_constant
   )

Best Practices
--------------

1. **Identify expensive operations** that don't depend on parameters
2. **Use for preprocessing** that can be done once before sampling
3. **Consider matrix operations** as candidates for transformed data
4. **Validate Stan code generation** for custom operations
5. **Profile models** to identify transformation opportunities
6. **Use automatic features** like MultinomialLogTheta optimization
7. **Document custom transformed data** operations clearly

The transformed data framework provides essential performance optimization capabilities while maintaining the clean separation between data preprocessing and statistical modeling in SciStanPy.
