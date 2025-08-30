Key Concepts for Scientists
===========================

This guide explains the fundamental concepts in SciStanPy from a scientific perspective.

Parameters vs Data
------------------

**Parameters** are unknown quantities you want to learn about:

.. code-block:: python

   # Unknown physical constants
   diffusion_coefficient = ssp.parameters.LogNormal(mu=-2, sigma=1)

   # Model parameters with uncertainty
   growth_rate = ssp.parameters.Normal(mu=0.1, sigma=0.05)

**Data** are your observations and measurements:

.. code-block:: python

   # Experimental measurements
   concentration_data = np.array([1.2, 2.1, 3.8, 5.5])

   # Connect to model
   model_predictions = some_function(parameters)
   likelihood = ssp.parameters.Normal(mu=model_predictions, sigma=measurement_error)
   likelihood.observe(concentration_data)

Probability Distributions
-------------------------

Distributions encode uncertainty and variability:

**Prior Distributions** - Your knowledge before seeing data:

.. code-block:: python

   # "I think the temperature is around 25°C, give or take 5°C"
   temperature = ssp.parameters.Normal(mu=25, sigma=5)

   # "The reaction rate should be positive, probably around 0.1 s⁻¹"
   rate = ssp.parameters.LogNormal(mu=np.log(0.1), sigma=0.5)

**Likelihood Distributions** - How data relates to parameters:

.. code-block:: python

   # "My measurements follow the true value plus Gaussian noise"
   likelihood = ssp.parameters.Normal(mu=true_value, sigma=measurement_noise)

**Posterior Distributions** - Updated knowledge after seeing data:

.. code-block:: python

   # Automatically computed by SciStanPy
   results = model.sample()
   posterior_samples = results['parameter_name']

Common Distribution Choices
---------------------------

**Normal Distribution** - For symmetric, unbounded quantities:

.. code-block:: python

   # Temperature differences, measurement errors
   temp_change = ssp.parameters.Normal(mu=0, sigma=2)

**LogNormal Distribution** - For positive quantities:

.. code-block:: python

   # Concentrations, rates, sizes
   concentration = ssp.parameters.LogNormal(mu=0, sigma=1)

**Exponential Distribution** - For waiting times, lifetimes:

.. code-block:: python

   # Time until decay, failure times
   lifetime = ssp.parameters.Exponential(rate=0.1)

**Beta Distribution** - For proportions, probabilities:

.. code-block:: python

   # Success rates, fractions
   success_rate = ssp.parameters.Beta(alpha=2, beta=5)

Mathematical Operations
-----------------------

Combine parameters naturally using Python syntax:

.. code-block:: python

   # Linear relationships
   final_temp = initial_temp + heating_rate * time

   # Exponential processes
   population = initial_size * ssp.operations.exp(growth_rate * time)

   # Power laws
   intensity = amplitude * (distance ** -2)

   # Ratios and products
   efficiency = useful_output / total_input

Uncertainty Propagation
-----------------------

SciStanPy automatically propagates uncertainty through calculations:

.. code-block:: python

   # Input uncertainties
   length = ssp.parameters.Normal(mu=10, sigma=0.1)  # cm
   width = ssp.parameters.Normal(mu=5, sigma=0.05)   # cm

   # Derived quantity with propagated uncertainty
   area = length * width  # Uncertainty automatically propagated

   # Complex functions
   volume = length * width * height
   density = mass / volume
   pressure = force / area

Model Building Workflow
-----------------------

1. **Define Unknowns**: What parameters do you want to estimate?

.. code-block:: python

   reaction_rate = ssp.parameters.LogNormal(mu=0, sigma=1)
   activation_energy = ssp.parameters.Normal(mu=50, sigma=10)  # kJ/mol

2. **Express Relationships**: How do parameters relate to observations?

.. code-block:: python

   # Arrhenius equation
   rate_constant = reaction_rate * ssp.operations.exp(-activation_energy / (R * temperature))

3. **Model Observations**: How do predictions compare to data?

.. code-block:: python

   predicted_rates = rate_constant  # Some function of parameters
   likelihood = ssp.parameters.Normal(mu=predicted_rates, sigma=measurement_error)
   likelihood.observe(measured_rates)

4. **Run Inference**: Update knowledge based on data

.. code-block:: python

   model = ssp.Model(likelihood)
   results = model.sample()

Types of Inference
------------------

**Maximum Likelihood Estimation (MLE)**: Find most likely parameter values

.. code-block:: python

   mle_results = model.mle()  # Point estimates

**Bayesian Inference**: Full uncertainty quantification

.. code-block:: python

   mcmc_results = model.mcmc()  # Full posterior distributions

**Variational Inference**: Fast approximate inference

.. code-block:: python

   vi_results = model.variational()  # Approximate posteriors

Model Checking and Validation
-----------------------------

**Prior Predictive Checks**: Does your model make sense before seeing data?

.. code-block:: python

   prior_predictions = model.prior_predictive()
   # Check if predictions are reasonable

**Posterior Predictive Checks**: Does your fitted model reproduce the data?

.. code-block:: python

   posterior_predictions = model.posterior_predictive(results)
   # Compare predictions to actual data

**Convergence Diagnostics**: Did the inference work properly?

.. code-block:: python

   diagnostics = model.diagnose(results)
   # Check R-hat, effective sample size, etc.

Scientific Interpretation
-------------------------

**Point Estimates vs Uncertainty**: Always report both central tendency and spread

.. code-block:: python

   # Not just the mean
   mean_estimate = results['parameter'].mean()

   # Also the uncertainty
   std_estimate = results['parameter'].std()
   credible_interval = np.percentile(results['parameter'], [2.5, 97.5])

**Credible Intervals**: "Given the data, there's a 95% probability the true value lies in this range"

**Model Comparison**: Use information criteria to compare alternative hypotheses

.. code-block:: python

   model1_loo = model1.loo()  # Leave-one-out cross-validation
   model2_loo = model2.loo()
   # Lower is better

Remember: Bayesian modeling is about updating your scientific knowledge in light of new evidence, while properly accounting for uncertainty at every step.
