Transformed Parameters API Reference
====================================

This reference covers the transformed parameter framework that enables mathematical operations and operator overloading in SciStanPy.

Transformed Parameters Module
-----------------------------

.. automodule:: scistanpy.model.components.transformations.transformed_parameters

   :members:
   :undoc-members:
   :show-inheritance:

Core Transformation Classes
---------------------------

Abstract Base Class
~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.TransformedParameter

   :members:
   :undoc-members:
   :show-inheritance:

   **Unified Transformation Interface:**

   .. code-block:: python

      # All transformations are TransformedParameters
      transformation = x + y

      # Access parent components
      parents = transformation.parents

      # Generate Stan code
      stan_code = transformation.get_transformation_assignment(('i', 'j'))

      # Check if explicitly named
      is_named = transformation.is_named

Binary and Unary Operations
---------------------------

BinaryTransformedParameter
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.BinaryTransformedParameter

   :members:
   :undoc-members:
   :show-inheritance:

   **Two-Operand Transformations:**

   .. code-block:: python

      # Binary operations automatically handle broadcasting
      x = ssp.parameters.Normal(mu=0, sigma=1, shape=(5,))
      y = ssp.parameters.Normal(mu=0, sigma=1, shape=(3, 1))

      # Result has shape (3, 5) through broadcasting
      combined = x + y

UnaryTransformedParameter
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.UnaryTransformedParameter

   :members:
   :undoc-members:
   :show-inheritance:

   **Single-Operand Transformations:**

   .. code-block:: python

      # Unary operations preserve shape
      x = ssp.parameters.Normal(mu=0, sigma=1, shape=(3, 4))
      neg_x = -x                           # Shape remains (3, 4)
      abs_x = ssp.operations.abs(x)        # Shape remains (3, 4)

Arithmetic Operations
---------------------

Basic Arithmetic
~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.AddParameter

   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.SubtractParameter

   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.MultiplyParameter

   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.DivideParameter

   :members:
   :undoc-members:
   :show-inheritance:

   **Arithmetic Operation Examples:**

   .. code-block:: python

      # Basic arithmetic
      a = ssp.parameters.Normal(mu=0, sigma=1)
      b = ssp.parameters.Normal(mu=0, sigma=1)

      addition = a + b          # AddParameter
      subtraction = a - b       # SubtractParameter
      multiplication = a * b    # MultiplyParameter
      division = a / b          # DivideParameter

Power and Unary Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.PowerParameter

   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.NegateParameter

   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.AbsParameter

   :members:
   :undoc-members:
   :show-inheritance:

   **Advanced Arithmetic Examples:**

   .. code-block:: python

      # Power operations
      squared = x ** 2                     # PowerParameter
      cube_root = x ** (1/3)               # PowerParameter

      # Unary operations
      negated = -x                         # NegateParameter
      absolute = ssp.operations.abs(x)     # AbsParameter

Mathematical Functions
----------------------

The transformed parameters framework enables complex mathematical expressions:

**Function Composition:**

.. code-block:: python

   # Complex mathematical expressions
   mu = ssp.parameters.Normal(mu=0, sigma=1)
   sigma = ssp.parameters.LogNormal(mu=0, sigma=0.5)

   # Standardized normal
   standardized = (x - mu) / sigma

   # Log-normal parameterization
   log_normal_mean = ssp.operations.exp(mu + 0.5 * sigma**2)

**Growth Model Applications:**

.. code-block:: python

   # Exponential growth model
   time_points = ssp.constants.Constant([0, 1, 2, 3, 4])
   initial_pop = ssp.parameters.LogNormal(mu=np.log(100), sigma=0.1)
   growth_rate = ssp.parameters.Normal(mu=0.1, sigma=0.02)

   # Population over time
   population = initial_pop * ssp.operations.exp(growth_rate * time_points)

Advanced Transformation Features
--------------------------------

Shape Broadcasting
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Automatic shape broadcasting
   scalar_param = ssp.parameters.Normal(mu=0, sigma=1)           # Shape: ()
   vector_param = ssp.parameters.Normal(mu=0, sigma=1, shape=(5,))  # Shape: (5,)
   matrix_param = ssp.parameters.Normal(mu=0, sigma=1, shape=(3, 5))  # Shape: (3, 5)

   # Broadcasting follows NumPy rules
   result1 = scalar_param + vector_param    # Shape: (5,)
   result2 = vector_param + matrix_param    # Shape: (3, 5)

Stan Code Generation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Transformations generate optimized Stan code
   complex_expr = (x + y) * ssp.operations.exp(z)

   # Automatic variable naming and code generation
   stan_code = complex_expr.get_transformation_assignment(('i', 'j'))

   # Custom naming for complex expressions
   complex_expr.model_varname = "complex_transformation"

Custom Transformations
----------------------

**Creating Custom Binary Operations:**

.. code-block:: python

   class CustomBinaryOp(BinaryTransformedParameter):
       """Custom binary transformation."""

       def _draw(self, level_draws, seed):
           left = level_draws['left']
           right = level_draws['right']
           return np.custom_operation(left, right)

       def get_right_side(self, index_opts, **kwargs):
           components = super().get_right_side(index_opts, **kwargs)
           return f"custom_function({components['left']}, {components['right']})"

**Creating Custom Unary Operations:**

.. code-block:: python

   class CustomUnaryOp(UnaryTransformedParameter):
       """Custom unary transformation."""

       def _draw(self, level_draws, seed):
           operand = level_draws['operand']
           return np.custom_function(operand)

       def get_right_side(self, index_opts, **kwargs):
           components = super().get_right_side(index_opts, **kwargs)
           return f"custom_transform({components['operand']})"

Indexing and Array Operations
-----------------------------

**Array Indexing Support:**

.. code-block:: python

   # Multi-dimensional parameter indexing
   matrix_param = ssp.parameters.Normal(mu=0, sigma=1, shape=(10, 5))

   # Advanced indexing creates IndexParameter transformations
   row_slice = matrix_param[2, :]           # Third row
   column_slice = matrix_param[:, 1]        # Second column
   submatrix = matrix_param[1:4, 2:4]      # Submatrix selection

   # NumPy-style advanced indexing
   diagonal = matrix_param[np.arange(5), np.arange(5)]

**Reduction Operations:**

.. code-block:: python

   # Statistical reductions
   vector_param = ssp.parameters.Normal(mu=0, sigma=1, shape=(10,))

   # Built-in reductions
   total = ssp.operations.sum_(vector_param)
   average = ssp.operations.mean(vector_param)
   variance = ssp.operations.var(vector_param)

Integration with Stan
---------------------

**Automatic Stan Translation:**

.. code-block:: python

   # Complex expression
   result = (x + y) * ssp.operations.exp(z / w)

   # Generates efficient Stan code like:
   # result = (x + y) * exp(z / w);

**Vectorization Optimization:**

.. code-block:: python

   # Vector operations are automatically vectorized in Stan
   vector_x = ssp.parameters.Normal(mu=0, sigma=1, shape=(100,))
   vector_y = ssp.parameters.Normal(mu=0, sigma=1, shape=(100,))

   # Generates vectorized Stan code
   sum_vectors = vector_x + vector_y

Performance Considerations
--------------------------

**Efficient Expression Building:**

.. code-block:: python

   # Efficient: Direct expression building
   efficient_expr = x + y * z

   # Less efficient: Multiple intermediate variables
   temp1 = y * z
   temp2 = x + temp1

**Memory Management:**

.. code-block:: python

   # Share parent references rather than copying
   shared_base = ssp.parameters.Normal(mu=0, sigma=1)

   transform1 = shared_base + 1         # References shared_base
   transform2 = shared_base * 2         # References shared_base
   transform3 = shared_base ** 2        # References shared_base

**Stan Code Optimization:**

.. code-block:: python

   # Transformations are optimized for Stan execution:
   # - Automatic vectorization where possible
   # - Efficient loop structures for multi-dimensional operations
   # - Proper variable scoping and memory management

Best Practices
--------------

1. **Use natural mathematical syntax** for clear model expression
2. **Name complex transformations** for better model interpretation
3. **Leverage broadcasting** to avoid manual shape management
4. **Build expressions incrementally** for complex mathematical models
5. **Validate shapes early** to catch dimension mismatches
6. **Use operations module** for specialized mathematical functions
7. **Monitor expression complexity** to maintain model interpretability

The transformed parameters framework provides the mathematical foundation for expressing complex scientific models while maintaining computational efficiency and Stan code generation capabilities.
