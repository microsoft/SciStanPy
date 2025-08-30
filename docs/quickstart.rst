Quick Start Guide
=================

Welcome to SciStanPy! This guide will get you up and running with Bayesian modeling in minutes.

Your First Model
----------------

Let's analyze some measurement data to estimate the true value with uncertainty:

.. code-block:: python

   import scistanpy as ssp
   import numpy as np

   # Your measurement data
   measurements = np.array([10.2, 9.8, 10.1, 9.9, 10.3])

   # Define the model
   true_value = ssp.parameters.Normal(mu=10, sigma=1)  # Prior belief
   likelihood = ssp.parameters.Normal(mu=true_value, sigma=0.2)
   likelihood.observe(measurements)

   # Run inference
   model = ssp.Model(likelihood)
   results = model.sample()

   # Analyze results
   print(f"Estimated value: {results['true_value'].mean():.2f}")
   print(f"95% credible interval: {np.percentile(results['true_value'], [2.5, 97.5])}")

Key Concepts
------------

**Parameters**: Unknown quantities you want to learn about

.. code-block:: python

   # Define a parameter with prior uncertainty
   reaction_rate = ssp.parameters.LogNormal(mu=0, sigma=1)

**Observations**: Your measured data

.. code-block:: python

   # Link model predictions to data
   likelihood = ssp.parameters.Normal(mu=predicted_values, sigma=noise_level)
   likelihood.observe(your_data)

**Models**: Collections of parameters and relationships

.. code-block:: python

   model = ssp.Model(likelihood)
   results = model.sample()  # Bayesian inference

**Transformations**: Mathematical operations on parameters

.. code-block:: python

   log_rate = ssp.parameters.Normal(mu=0, sigma=1)
   rate = ssp.operations.exp(log_rate)  # Ensures positive values

Next Steps
----------

1. **Learn the Basics**: Read the modeling fundamentals guide
2. **Explore Examples**: Check out the examples gallery
3. **Try Different Distributions**: See the distributions reference
4. **Advanced Topics**: Learn about custom distributions and performance optimization

Common Patterns
---------------

**Linear Regression**:

.. code-block:: python

   slope = ssp.parameters.Normal(mu=0, sigma=1)
   intercept = ssp.parameters.Normal(mu=0, sigma=1)
   sigma = ssp.parameters.LogNormal(mu=0, sigma=1)

   predictions = intercept + slope * x_data
   likelihood = ssp.parameters.Normal(mu=predictions, sigma=sigma)
   likelihood.observe(y_data)

**Time Series**:

.. code-block:: python

   initial_state = ssp.parameters.Normal(mu=0, sigma=1)
   innovation_noise = ssp.parameters.LogNormal(mu=0, sigma=0.5)

   states = [initial_state]
   for t in range(1, len(time_series)):
       next_state = ssp.parameters.Normal(mu=states[-1], sigma=innovation_noise)
       states.append(next_state)

**Hierarchical Models**:

.. code-block:: python

   # Group-level parameters
   global_mean = ssp.parameters.Normal(mu=0, sigma=1)
   global_sigma = ssp.parameters.LogNormal(mu=0, sigma=1)

   # Individual-level parameters
   individual_means = ssp.parameters.Normal(mu=global_mean, sigma=global_sigma, shape=(n_groups,))
