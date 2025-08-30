Transformations Guide
=====================

This guide covers mathematical transformations in SciStanPy and how to use them to build complex models from simple components.

What Are Transformations?
-------------------------

Transformations are mathematical operations that combine parameters to create new derived quantities. They enable you to build sophisticated models while maintaining automatic differentiation and Stan code generation.

.. code-block:: python

   # Basic transformations
   sum_param = param1 + param2
   scaled_param = 2.5 * param1
   log_param = ssp.operations.log(positive_param)

   # Complex expressions
   kinetic_energy = 0.5 * mass * velocity**2
   concentration = initial_conc * ssp.operations.exp(-decay_rate * time)

Arithmetic Operations
--------------------

Basic mathematical operations work naturally with parameters:

Addition and Subtraction
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Combining effects
   total_effect = treatment_effect + baseline_effect

   # Computing differences
   temperature_change = final_temp - initial_temp

   # Multiple terms
   polynomial = intercept + slope * x + quadratic * x**2

Multiplication and Division
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Scaling
   scaled_measurement = calibration_factor * raw_measurement

   # Rates and ratios
   reaction_rate = frequency_factor * ssp.operations.exp(-activation_energy / temperature)
   efficiency = useful_output / total_input

   # Element-wise operations
   predicted_values = coefficients * predictors  # Broadcasting applies

Exponentiation
~~~~~~~~~~~~~~

.. code-block:: python

   # Power relationships
   gravitational_force = G * mass1 * mass2 / distance**2

   # Exponential processes
   population = initial_size * ssp.operations.exp(growth_rate * time)

   # Polynomial terms
   quadratic_term = coefficient * x**2

Mathematical Functions
---------------------

SciStanPy provides a comprehensive library of mathematical functions:

Exponential and Logarithmic
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Ensure positive values
   positive_param = ssp.operations.exp(unbounded_param)

   # Log-scale modeling
   log_concentration = ssp.operations.log(concentration)

   # Log-sum-exp for numerical stability
   log_partition = ssp.operations.log_sum_exp(log_weights)

Trigonometric Functions
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Periodic phenomena
   seasonal_component = amplitude * ssp.operations.sin(frequency * time + phase)

   # Angular transformations
   x_component = radius * ssp.operations.cos(angle)
   y_component = radius * ssp.operations.sin(angle)

Absolute Value and Sign
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Magnitude calculations
   distance = ssp.operations.abs(position1 - position2)

   # Rectified linear units
   relu_output = ssp.operations.maximum(0, linear_combination)

Sigmoid and Logistic Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Convert to probabilities
   probability = ssp.operations.sigmoid(log_odds)

   # Bounded parameters
   bounded_param = lower + (upper - lower) * ssp.operations.sigmoid(unbounded)

Statistical Operations
---------------------

Specialized operations for statistical modeling:

Normalization
~~~~~~~~~~~~~

.. code-block:: python

   # Create probability vectors
   probabilities = ssp.operations.normalize(positive_weights)

   # Log-space normalization
   log_probabilities = ssp.operations.normalize_log(log_weights)

Reductions
~~~~~~~~~~

.. code-block:: python

   # Sum over dimensions
   total = ssp.operations.sum(individual_contributions)

   # Mean calculations
   average = ssp.operations.mean(measurements)

   # Log-sum-exp for numerical stability
   log_sum = ssp.operations.log_sum_exp(log_values)

Growth Models
-------------

SciStanPy provides specialized transformations for common growth patterns:

Exponential Growth
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Standard exponential growth
   population = ssp.operations.exponential_growth(
       t=time_points,
       A=initial_population,
       r=growth_rate
   )

   # Log-scale version for stability
   log_population = ssp.operations.log_exponential_growth(
       t=time_points,
       log_A=log_initial_population,
       r=growth_rate
   )

Sigmoid Growth
~~~~~~~~~~~~~~

.. code-block:: python

   # Logistic growth with carrying capacity
   population = ssp.operations.sigmoid_growth(
       t=time_points,
       A=carrying_capacity,
       r=growth_rate,
       c=inflection_point
   )

   # Alternative parameterization by initial conditions
   population_alt = ssp.operations.sigmoid_growth_init(
       t=time_points,
       x0=initial_population,
       r=growth_rate,
       c=time_scale
   )

Array Operations
---------------

Advanced operations for multi-dimensional data:

Indexing and Slicing
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # NumPy-style indexing
   subset = ssp.operations.index(full_array, [0, 2, 4])

   # Slicing operations
   time_slice = ssp.operations.index(time_series, slice(10, 50))

   # Multi-dimensional indexing
   selected_elements = ssp.operations.index(matrix, row_indices, col_indices)

Convolution
~~~~~~~~~~~

.. code-block:: python

   # Sequence pattern matching
   motif_scores = ssp.operations.convolve_sequence(
       weights=position_weight_matrix,
       ordinals=encoded_sequence
   )

Transformation Composition
--------------------------

Complex models emerge from composing simple transformations:

Building Complex Relationships
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Enzyme kinetics: Michaelis-Menten equation
   V_max = ssp.parameters.LogNormal(mu=np.log(10), sigma=0.5)
   K_m = ssp.parameters.LogNormal(mu=np.log(1), sigma=0.5)
   substrate_conc = np.array([0.1, 0.5, 1.0, 2.0, 5.0])

   # Compose the relationship
   velocity = (V_max * substrate_conc) / (K_m + substrate_conc)

Multi-Step Processes
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Drug pharmacokinetics
   dose = 100  # mg
   absorption_rate = ssp.parameters.LogNormal(mu=np.log(1.0), sigma=0.3)
   elimination_rate = ssp.parameters.LogNormal(mu=np.log(0.1), sigma=0.3)
   volume = ssp.parameters.LogNormal(mu=np.log(50), sigma=0.2)

   # Multi-compartment model
   time_points = np.linspace(0, 24, 100)
   concentration = (dose / volume) * (
       (absorption_rate / (absorption_rate - elimination_rate)) *
       (ssp.operations.exp(-elimination_rate * time_points) -
        ssp.operations.exp(-absorption_rate * time_points))
   )

Hierarchical Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Population pharmacokinetics
   pop_clearance = ssp.parameters.LogNormal(mu=np.log(5), sigma=0.3)
   pop_volume = ssp.parameters.LogNormal(mu=np.log(70), sigma=0.2)

   # Individual-level parameters
   individual_clearance = pop_clearance * ssp.operations.exp(
       ssp.parameters.Normal(mu=0, sigma=0.2, shape=(n_individuals,))
   )
   individual_volume = pop_volume * ssp.operations.exp(
       ssp.parameters.Normal(mu=0, sigma=0.15, shape=(n_individuals,))
   )

Custom Transformations
---------------------

Create domain-specific transformations for your field:

Function-Based Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def michaelis_menten(substrate, V_max, K_m):
       """Michaelis-Menten enzyme kinetics."""
       return (V_max * substrate) / (K_m + substrate)

   def arrhenius_rate(temperature, A, Ea, R=8.314):
       """Arrhenius rate equation."""
       return A * ssp.operations.exp(-Ea / (R * temperature))

   # Use in models
   enzyme_velocity = michaelis_menten(substrate_conc, V_max_param, K_m_param)
   reaction_rate = arrhenius_rate(temp_param, frequency_factor, activation_energy)

Class-Based Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class PharmacokineticModel(ssp.operations.CustomTransformation):
       """One-compartment PK model."""

       def __init__(self, dose, clearance, volume, time_points):
           self.dose = dose
           self.clearance = clearance
           self.volume = volume
           self.time_points = time_points
           super().__init__(clearance=clearance, volume=volume)

       def transform(self, clearance, volume):
           ke = clearance / volume
           return (self.dose / volume) * ssp.operations.exp(-ke * self.time_points)

Numerical Considerations
-----------------------

Best Practices for Stable Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use Log-Space When Appropriate**:

.. code-block:: python

   # Instead of exp(large_number)
   log_result = log_param1 + log_param2  # Addition in log-space

   # Instead of very small probabilities
   log_prob = ssp.operations.log_sum_exp(log_components)

**Centering and Scaling**:

.. code-block:: python

   # Center predictors for numerical stability
   x_centered = x - x.mean()
   predictions = intercept + slope * x_centered

**Avoid Extreme Values**:

.. code-block:: python

   # Constrain parameters to reasonable ranges
   bounded_param = ssp.operations.sigmoid(unbounded) * scale + offset

**Use Stable Implementations**:

.. code-block:: python

   # Use built-in stable functions
   stable_sigmoid = ssp.operations.sigmoid(x)  # Instead of 1/(1+exp(-x))
   stable_logsumexp = ssp.operations.log_sum_exp(x)  # Instead of log(sum(exp(x)))

Debugging Transformations
------------------------

Tools for understanding your transformations:

**Check Shapes**:

.. code-block:: python

   print(f"Parameter shape: {param.shape}")
   print(f"Transformation result shape: {transformed.shape}")

**Examine Values**:

.. code-block:: python

   # Prior predictive checks
   prior_samples = model.prior_predictive()
   print(f"Typical values: {prior_samples['param_name'][:10]}")

**Visualize Relationships**:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Plot transformation
   x_range = np.linspace(-3, 3, 100)
   y_values = ssp.operations.sigmoid(x_range)
   plt.plot(x_range, y_values)
   plt.title("Sigmoid Transformation")

**Test Edge Cases**:

.. code-block:: python

   # Check behavior at extremes
   extreme_values = np.array([-1e6, -10, 0, 10, 1e6])
   result = your_transformation(extreme_values)
   print(f"Extreme case results: {result}")

Performance Optimization
------------------------

**Vectorization**: Use array operations instead of loops

.. code-block:: python

   # Vectorized (fast)
   results = coefficient * x_array

   # Avoid explicit loops (slow)
   # results = [coefficient * x for x in x_array]

**Batch Operations**: Process multiple cases simultaneously

.. code-block:: python

   # Batch processing
   batch_results = model_function(batch_parameters)

**Memory Efficiency**: Avoid unnecessary intermediate arrays

.. code-block:: python

   # Memory efficient
   result = a * b + c * d

   # Less efficient
   # temp1 = a * b
   # temp2 = c * d
   # result = temp1 + temp2

This comprehensive guide provides the tools and knowledge needed to build sophisticated scientific models using SciStanPy's transformation system.
