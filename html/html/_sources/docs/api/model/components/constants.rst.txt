Constants API Reference
=======================

This reference covers the constant value components used in SciStanPy models.

Constants Module
----------------

.. automodule:: scistanpy.model.components.constants
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Constant Class
--------------

.. autoclass:: scistanpy.model.components.constants.Constant
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Key Features:**

   The Constant class provides essential functionality for representing fixed values in SciStanPy models:

   - **Automatic Type Inference**: Determines appropriate Stan data types from Python values
   - **Bound Checking**: Validates values against optional constraints
   - **Interactive Support**: Configures sliders for model exploration
   - **Shape Handling**: Automatically infers dimensions from array inputs

   **Basic Usage:**

   .. code-block:: python

      import scistanpy as ssp
      import numpy as np

      # Scalar constants
      learning_rate = ssp.constants.Constant(0.01)
      n_samples = ssp.constants.Constant(100)

      # Array constants
      design_matrix = ssp.constants.Constant(X_data)
      time_points = ssp.constants.Constant(np.linspace(0, 10, 11))

      # Constants with bounds
      probability = ssp.constants.Constant(0.5, lower_bound=0.0, upper_bound=1.0)

   **Advanced Features:**

   .. code-block:: python

      # Interactive constants for exploration
      interactive_param = ssp.constants.Constant(
          1.0,
          lower_bound=0.1,
          upper_bound=5.0,
          togglable=True  # Enable slider widgets
      )

      # Uniform array enforcement
      symmetric_prior = ssp.constants.Constant(
          [1.0, 1.0, 1.0],
          enforce_uniformity=True
      )

Constant Properties and Methods
-------------------------------

Type Inference and Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Constants automatically infer Stan data types and validate values:

.. code-block:: python

   # Integer constant
   count = ssp.constants.Constant(42)
   print(count.BASE_STAN_DTYPE)  # "int"

   # Float constant
   rate = ssp.constants.Constant(0.5)
   print(rate.BASE_STAN_DTYPE)  # "real"

   # Array constant
   matrix = ssp.constants.Constant(np.random.randn(3, 4))
   print(matrix.shape)  # (3, 4)

Interactive Slider Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Constants can be configured for interactive exploration:

.. code-block:: python

   # Configure slider properties
   param = ssp.constants.Constant(
       value=2.0,
       lower_bound=0.0,
       upper_bound=10.0
   )

   # Access slider configuration
   print(f"Start: {param.slider_start}")      # 0.0
   print(f"End: {param.slider_end}")          # 10.0
   print(f"Step: {param.slider_step_size}")   # 0.1
   print(f"Togglable: {param.is_togglable}")  # True (for floats)

Stan Code Integration
~~~~~~~~~~~~~~~~~~~~~

Constants integrate seamlessly with Stan code generation:

.. code-block:: python

   # Stan data type
   dtype = param.get_stan_dtype()
   print(dtype)  # "real<lower=0, upper=10>"

   # Stan parameter declaration
   declaration = param.get_stan_parameter_declaration()
   print(declaration)  # "real<lower=0, upper=10> param_name"

PyTorch Integration
~~~~~~~~~~~~~~~~~~~

Constants provide tensor representations for PyTorch compatibility:

.. code-block:: python

   import torch

   # Create constant
   weights = ssp.constants.Constant(np.array([0.2, 0.3, 0.5]))

   # Get PyTorch tensor
   tensor = weights.torch_parametrization
   print(type(tensor))  # <class 'torch.Tensor'>
   print(tensor.dtype)  # torch.float32

Usage Patterns
--------------

Model Hyperparameters
~~~~~~~~~~~~~~~~~~~~~

Constants are commonly used for model hyperparameters:

.. code-block:: python

   class BayesianRegression(ssp.Model):
       def __init__(self, X, y):
           # Hyperparameters as constants
           self.prior_mu = ssp.constants.Constant(0.0)
           self.prior_sigma = ssp.constants.Constant(1.0)
           self.noise_alpha = ssp.constants.Constant(1.0)
           self.noise_beta = ssp.constants.Constant(1.0)

           # Model parameters
           self.coefficients = ssp.parameters.Normal(
               mu=self.prior_mu,
               sigma=self.prior_sigma,
               shape=(X.shape[1],)
           )
           self.noise = ssp.parameters.Gamma(
               alpha=self.noise_alpha,
               beta=self.noise_beta
           )

           # Likelihood
           predictions = X @ self.coefficients
           self.observations = ssp.parameters.Normal(
               mu=predictions,
               sigma=1.0 / ssp.operations.sqrt(self.noise)
           )
           self.observations.observe(y)

           super().__init__(self.observations)

Design Matrices and Fixed Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Constants handle experimental design and fixed data:

.. code-block:: python

   # Experimental design
   treatment_matrix = ssp.constants.Constant(design_matrix)
   time_points = ssp.constants.Constant(measurement_times)

   # Fixed experimental conditions
   temperature = ssp.constants.Constant(298.15)  # Kelvin
   pressure = ssp.constants.Constant(1.013)      # bar

   # Use in model
   reaction_rate = ssp.parameters.LogNormal(mu=0, sigma=1)
   measured_rate = reaction_rate * ssp.operations.exp(
       -activation_energy / (gas_constant * temperature)
   )

Prior Predictive Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Constants enable prior predictive checks by providing fixed reference values:

.. code-block:: python

   # Define model with constant reference
   true_effect_size = ssp.constants.Constant(0.5)  # Known for simulation

   # Model parameters
   estimated_effect = ssp.parameters.Normal(mu=0, sigma=1)
   measurement_noise = ssp.parameters.LogNormal(mu=0, sigma=0.5)

   # Simulated observations
   simulated_data = ssp.parameters.Normal(
       mu=true_effect_size,  # Use constant for simulation
       sigma=measurement_noise
   )

   # Prior predictive sampling
   model = ssp.Model(simulated_data)
   prior_samples = model.prior_predictive(n=1000)

Interactive Model Exploration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Constants with slider support enable interactive model exploration:

.. code-block:: python

   # Interactive hyperparameters
   explore_sigma = ssp.constants.Constant(
       value=1.0,
       lower_bound=0.1,
       upper_bound=5.0,
       togglable=True
   )

   # Model with interactive parameter
   test_param = ssp.parameters.Normal(mu=0, sigma=explore_sigma)

   # In Jupyter notebooks, this enables interactive widgets
   # Users can adjust explore_sigma and see effects on model

Validation and Error Handling
-----------------------------

Comprehensive Validation
~~~~~~~~~~~~~~~~~~~~~~~~

Constants perform extensive validation during construction:

.. code-block:: python

   # Bound checking
   try:
       invalid = ssp.constants.Constant(
           value=5.0,
           lower_bound=0.0,
           upper_bound=3.0  # value exceeds upper bound
       )
   except ValueError as e:
       print(f"Validation error: {e}")

   # Uniformity enforcement
   try:
       non_uniform = ssp.constants.Constant(
           value=[1.0, 2.0, 3.0],
           enforce_uniformity=True  # All values must be identical
       )
   except ValueError as e:
       print(f"Uniformity error: {e}")

Type Safety
~~~~~~~~~~~

Constants ensure type safety in model construction:

.. code-block:: python

   # Automatic type detection prevents issues
   integer_constant = ssp.constants.Constant(42)
   float_constant = ssp.constants.Constant(42.0)

   print(integer_constant.BASE_STAN_DTYPE)  # "int"
   print(float_constant.BASE_STAN_DTYPE)   # "real"

   # This ensures proper Stan code generation
   int_declaration = integer_constant.get_stan_dtype()    # "int"
   real_declaration = float_constant.get_stan_dtype()     # "real"

Best Practices
--------------

1. **Use descriptive names** for constants to improve model readability
2. **Set appropriate bounds** for parameters that have natural constraints
3. **Enable interactivity** for constants you want to explore
4. **Enforce uniformity** when symmetric priors are required
5. **Validate inputs** by setting bounds on physical parameters
6. **Document units** and meanings of constant values
7. **Group related constants** in model construction for clarity

Constants provide the foundation for robust, interpretable Bayesian models by ensuring that fixed values are properly handled throughout the modeling workflow.

.. code-block:: python

   prior_precision = ssp.constants.Constant(
       1.0,
       lower_bound=0.1,
       upper_bound=10.0,
       togglable=True  # Allow interactive tuning
   )

   # Prior distributions using constants
   coefficients = ssp.parameters.Normal(
       mu=0.0,
       sigma=1.0 / ssp.operations.sqrt(prior_precision)
   )

**Design Matrix Integration:**

.. code-block:: python

   def build_design_matrix_model(formula, data):
       """Build model with design matrix constant."""

       # Create design matrix from formula and data
       # (This would use a formula parser in practice)
       X = np.column_stack([
           np.ones(len(data)),  # Intercept
           data['x1'],
           data['x2'],
           data['x1'] * data['x2']  # Interaction
       ])

       # Design matrix as constant
       design_matrix = ssp.constants.Constant(X)

       # Coefficients as parameters
       coefficients = ssp.parameters.Normal(
           mu=0, sigma=1, shape=(X.shape[1],)
       )

       # Linear predictor
       linear_pred = design_matrix @ coefficients

       return linear_pred, coefficients

Memory and Performance Considerations
-------------------------------------

**Efficient Constant Sharing:**

.. code-block:: python

   # Share constants across model components
   shared_design_matrix = ssp.constants.Constant(X_data)

   # Multiple parameters using the same constant
   beta1 = ssp.parameters.Normal(mu=shared_design_matrix @ theta1, sigma=1)
   beta2 = ssp.parameters.Normal(mu=shared_design_matrix @ theta2, sigma=1)
   # shared_design_matrix is referenced, not copied

**Large Array Handling:**

.. code-block:: python

   # Constants handle large arrays efficiently
   large_data = ssp.constants.Constant(np.random.randn(10000, 100))

   # NumPy array is stored directly, not copied
   print(f"Memory efficient: {large_data.value is large_data._torch_parametrization.numpy()}")

Error Handling and Validation
-----------------------------

**Comprehensive Input Validation:**

.. code-block:: python

   # Type validation
   try:
       invalid_type = ssp.constants.Constant("not_a_number")
   except (TypeError, ValueError) as e:
       print(f"Type validation: {e}")

   # Shape consistency validation
   try:
       # Explicit shape that doesn't match value
       inconsistent = ssp.constants.Constant(
           np.array([1, 2, 3]),
           shape=(2, 2)  # Doesn't match actual shape
       )
   except ValueError as e:
       print(f"Shape validation: {e}")

**Bound Checking:**

.. code-block:: python

   def validate_constant_bounds(value, lower=None, upper=None):
       """Validate constant bounds before creation."""

       try:
           const = ssp.constants.Constant(
               value,
               lower_bound=lower,
               upper_bound=upper
           )
           return const
       except ValueError as e:
           print(f"Bound validation failed: {e}")
           return None

Interactive Model Building
--------------------------

**Dynamic Constant Modification:**

.. code-block:: python

   def create_interactive_model():
       """Create model with interactive constants."""

       # Togglable hyperparameters
       prior_mean = ssp.constants.Constant(
           0.0,
           lower_bound=-5.0,
           upper_bound=5.0,
           togglable=True
       )

       prior_scale = ssp.constants.Constant(
           1.0,
           lower_bound=0.1,
           upper_bound=5.0,
           togglable=True
       )

       # Model components using interactive constants
       theta = ssp.parameters.Normal(mu=prior_mean, sigma=prior_scale)

       return {
           'parameter': theta,
           'interactive_constants': [prior_mean, prior_scale]
       }

**Slider Property Access:**

.. code-block:: python

   # Access slider configuration for UI building
   const = ssp.constants.Constant(
       2.5,
       lower_bound=0.0,
       upper_bound=10.0,
       togglable=True
   )

   slider_config = {
       'start': const.slider_start,      # 0.0
       'end': const.slider_end,          # 10.0
       'step': const.slider_step_size,   # 0.1
       'value': const.value.item()       # 2.5
   }

Best Practices
--------------

1. **Use appropriate bounds** to constrain constants to valid ranges
2. **Enable togglability** for hyperparameters you want to explore interactively
3. **Share constants** across model components to reduce memory usage
4. **Validate inputs** with bound checking for robust model specification
5. **Use descriptive names** for constants to improve model interpretability
6. **Enforce uniformity** when modeling symmetric priors or constraints
7. **Consider data types** (int vs float) for appropriate Stan code generation

The Constant class provides essential infrastructure for incorporating fixed values into probabilistic models while maintaining consistency with SciStanPy's component architecture and multi-backend support.
