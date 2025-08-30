Modeling Basics
===============

This guide covers the fundamental principles of building probabilistic models in SciStanPy.

The Scientific Modeling Process
-------------------------------

Bayesian modeling follows a systematic approach that mirrors scientific thinking:

1. **Formulate Hypotheses**: What are your scientific questions?
2. **Express Prior Knowledge**: What do you know before seeing data?
3. **Define the Data Model**: How do parameters relate to observations?
4. **Fit the Model**: Update knowledge based on evidence
5. **Validate and Iterate**: Check model assumptions and improve

Building Blocks of Models
-------------------------

Every SciStanPy model consists of three main components:

Parameters
~~~~~~~~~~

Parameters represent unknown quantities with associated uncertainty:

.. code-block:: python

   # Physical constants
   diffusion_coefficient = ssp.parameters.LogNormal(mu=-2, sigma=0.5)

   # Model coefficients
   slope = ssp.parameters.Normal(mu=0, sigma=1)
   intercept = ssp.parameters.Normal(mu=0, sigma=1)

   # Noise parameters
   measurement_noise = ssp.parameters.LogNormal(mu=np.log(0.1), sigma=0.3)

Relationships
~~~~~~~~~~~~~

Express how parameters combine to make predictions:

.. code-block:: python

   # Linear relationship
   predictions = intercept + slope * x_values

   # Nonlinear dynamics
   concentration = initial_conc * ssp.operations.exp(-decay_rate * time)

   # Complex models
   signal = amplitude * ssp.operations.sin(frequency * time + phase)

Observations
~~~~~~~~~~~~

Connect model predictions to your data:

.. code-block:: python

   # Define likelihood
   likelihood = ssp.parameters.Normal(mu=predictions, sigma=measurement_noise)

   # Observe data
   likelihood.observe(your_measurements)

Common Modeling Patterns
------------------------

Linear Models
~~~~~~~~~~~~~

For relationships of the form y = β₀ + β₁x₁ + β₂x₂ + ...

.. code-block:: python

   # Multiple regression
   intercept = ssp.parameters.Normal(mu=0, sigma=1)
   coefficients = ssp.parameters.Normal(mu=0, sigma=1, shape=(n_predictors,))
   noise = ssp.parameters.LogNormal(mu=0, sigma=0.5)

   predictions = intercept + ssp.operations.sum(coefficients * predictors, axis=-1)
   likelihood = ssp.parameters.Normal(mu=predictions, sigma=noise)

Hierarchical Models
~~~~~~~~~~~~~~~~~~~

For grouped or nested data:

.. code-block:: python

   # Group-level parameters
   global_mean = ssp.parameters.Normal(mu=0, sigma=1)
   global_sigma = ssp.parameters.LogNormal(mu=0, sigma=0.5)

   # Individual group parameters
   group_means = ssp.parameters.Normal(
       mu=global_mean,
       sigma=global_sigma,
       shape=(n_groups,)
   )

   # Individual observations
   individual_noise = ssp.parameters.LogNormal(mu=0, sigma=0.3)
   predictions = group_means[group_indices]
   likelihood = ssp.parameters.Normal(mu=predictions, sigma=individual_noise)

Time Series Models
~~~~~~~~~~~~~~~~~~

For temporal data:

.. code-block:: python

   # Random walk
   initial_state = ssp.parameters.Normal(mu=0, sigma=1)
   innovation_noise = ssp.parameters.LogNormal(mu=0, sigma=0.3)

   states = [initial_state]
   for t in range(1, n_timepoints):
       next_state = ssp.parameters.Normal(mu=states[-1], sigma=innovation_noise)
       states.append(next_state)

   # Observations
   observation_noise = ssp.parameters.LogNormal(mu=0, sigma=0.2)
   for t, state in enumerate(states):
       obs = ssp.parameters.Normal(mu=state, sigma=observation_noise)
       obs.observe(time_series[t])

Mechanistic Models
~~~~~~~~~~~~~~~~~~

Based on physical/biological processes:

.. code-block:: python

   # Pharmacokinetic model
   dose = 100  # mg
   clearance = ssp.parameters.LogNormal(mu=np.log(5), sigma=0.3)  # L/h
   volume = ssp.parameters.LogNormal(mu=np.log(70), sigma=0.2)    # L

   # Concentration over time
   time_points = np.array([0.5, 1, 2, 4, 8, 12, 24])  # hours
   ke = clearance / volume  # elimination rate
   concentrations = (dose / volume) * ssp.operations.exp(-ke * time_points)

   # Measurement model
   cv = 0.15  # 15% coefficient of variation
   measurement_error = cv * concentrations
   likelihood = ssp.parameters.Normal(mu=concentrations, sigma=measurement_error)

Prior Specification Guidelines
-----------------------------

Choosing appropriate priors is crucial for reliable inference:

Weakly Informative Priors
~~~~~~~~~~~~~~~~~~~~~~~~~~

When you have limited prior knowledge:

.. code-block:: python

   # For standardized coefficients
   beta = ssp.parameters.Normal(mu=0, sigma=1)

   # For positive scale parameters
   sigma = ssp.parameters.LogNormal(mu=0, sigma=1)

   # For correlation parameters
   correlation = ssp.parameters.Beta(alpha=2, beta=2)  # Centered at 0.5

Informative Priors
~~~~~~~~~~~~~~~~~~~

When you have scientific knowledge:

.. code-block:: python

   # Based on literature values
   reaction_rate = ssp.parameters.LogNormal(
       mu=np.log(0.05),  # Literature mean: 0.05 s⁻¹
       sigma=0.5         # Allows 2-fold variation
   )

   # Physical constraints
   temperature = ssp.parameters.Normal(
       mu=25,     # Room temperature
       sigma=5    # ±10°C range
   )

Reference Priors
~~~~~~~~~~~~~~~~

For objective analysis:

.. code-block:: python

   # Jeffrey's prior for scale parameters
   sigma = ssp.parameters.InverseGamma(alpha=0.001, beta=0.001)

   # Uniform on constrained spaces
   proportion = ssp.parameters.Beta(alpha=1, beta=1)  # Uniform on [0,1]

Model Building Best Practices
-----------------------------

Start Simple
~~~~~~~~~~~~

Begin with the simplest model that captures your scientific question:

.. code-block:: python

   # Simple linear model first
   y = intercept + slope * x + noise

   # Then add complexity if needed
   y = intercept + slope * x + quadratic * x**2 + noise

Modular Construction
~~~~~~~~~~~~~~~~~~~~

Build models in components for easier debugging:

.. code-block:: python

   # Separate model components
   def linear_trend(time, intercept, slope):
       return intercept + slope * time

   def seasonal_component(time, amplitude, frequency, phase):
       return amplitude * ssp.operations.sin(frequency * time + phase)

   # Combine components
   predictions = linear_trend(time, int_param, slope_param) + \
                 seasonal_component(time, amp_param, freq_param, phase_param)

Parameter Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~

Use transformations to improve sampling:

.. code-block:: python

   # Log-transform for positive parameters
   log_sigma = ssp.parameters.Normal(mu=0, sigma=1)
   sigma = ssp.operations.exp(log_sigma)

   # Logit-transform for bounded parameters
   logit_p = ssp.parameters.Normal(mu=0, sigma=1)
   p = ssp.operations.sigmoid(logit_p)

Centering and Scaling
~~~~~~~~~~~~~~~~~~~~~

Improve numerical stability:

.. code-block:: python

   # Center predictors
   x_centered = x - x.mean()

   # Scale to unit variance
   x_scaled = x_centered / x.std()

   # Use in model
   predictions = intercept + slope * x_scaled

Model Validation Workflow
-------------------------
1. **Prior Predictive Checks**:

.. code-block:: python

   prior_samples = model.prior_predictive()
   # Inspect simulated values
2. **Fit (MCMC) & Diagnostics**:

.. code-block:: python

   results = model.mcmc(chains=4, iter_warmup=500, iter_sampling=1000)
   sample_failures, variable_failures = results.diagnose()
   print(sample_failures, variable_failures.keys())
3. (Future) Posterior predictive / cross‑validation (not yet implemented)
4. **Sensitivity (manual)**:
   Re‑run with modified priors and compare parameter summaries.

Accuracy Note
-------------

Replaced model.sample -> model.mcmc and model.diagnose -> results.diagnose.
Removed unsupported: posterior_predictive(), loo(), waic(), marginal likelihood.

This systematic approach ensures robust, reliable models that properly capture scientific uncertainty and provide meaningful insights.
