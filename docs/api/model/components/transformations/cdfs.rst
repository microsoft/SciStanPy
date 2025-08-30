CDFs API Reference
==================

This reference covers the cumulative distribution function transformations in SciStanPy.

CDFs Module
-----------

.. automodule:: scistanpy.model.components.transformations.cdfs
   :members:
   :undoc-members:
   :show-inheritance:

CDF Transformation Framework
---------------------------

The CDF module provides automatic generation of cumulative distribution function transformations for all SciStanPy parameters, enabling probability transforms and model validation techniques.

Base CDF Classes
~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.transformations.cdfs.CDF
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.transformations.cdfs.CCDF
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.transformations.cdfs.LogCDF
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.transformations.cdfs.LogCCDF
   :members:
   :undoc-members:
   :show-inheritance:

   **Automatic CDF Generation:**

   All SciStanPy parameters automatically receive CDF methods through metaclass magic:

   .. code-block:: python

      import scistanpy as ssp
      import numpy as np

      # Define a normal parameter
      normal_param = ssp.parameters.Normal(mu=0, sigma=1)

      # CDF transformations are automatically available
      data_points = np.array([-2, -1, 0, 1, 2])

      # Standard CDF: P(X ≤ x)
      cdf_vals = normal_param.cdf(x=data_points)

      # Complementary CDF: P(X > x) = 1 - P(X ≤ x)
      survival_vals = normal_param.ccdf(x=data_points)

      # Log-space for numerical stability
      log_cdf_vals = normal_param.log_cdf(x=data_points)
      log_survival_vals = normal_param.log_ccdf(x=data_points)

Probability Transforms
---------------------

**Model Validation Applications:**

.. code-block:: python

   # Probability integral transform for model checking
   def probability_integral_transform(observed_data, fitted_param):
       """Transform data to uniform distribution if model is correct."""
       return fitted_param.cdf(x=observed_data)

   # Well-calibrated models produce uniform PIT values
   pit_values = probability_integral_transform(observations, fitted_normal)

   # Check uniformity with Kolmogorov-Smirnov test
   from scipy.stats import kstest
   ks_stat, p_value = kstest(pit_values, 'uniform')

**Quantile-Quantile Plots:**

.. code-block:: python

   # Create Q-Q plots using CDF transformations
   theoretical_quantiles = fitted_param.cdf(x=observed_data)
   empirical_quantiles = np.linspace(0, 1, len(observed_data))

   # Perfect model fit shows diagonal line
   import matplotlib.pyplot as plt
   plt.scatter(empirical_quantiles, theoretical_quantiles)
   plt.plot([0, 1], [0, 1], 'r--')  # Reference line
   plt.xlabel('Empirical Quantiles')
   plt.ylabel('Theoretical Quantiles')

Numerical Stability Features
---------------------------

**Log-Space Computations:**

.. code-block:: python

   # For extreme values, use log-space CDFs
   extreme_data = np.array([-100, -10, 0, 10, 100])

   # Standard CDF may underflow/overflow
   cdf_vals = normal_param.cdf(x=extreme_data)

   # Log-space provides numerical stability
   log_cdf_vals = normal_param.log_cdf(x=extreme_data)
   stable_cdf_vals = np.exp(log_cdf_vals)

**Survival Function Computations:**

.. code-block:: python

   # For tail probabilities, use complementary CDF
   large_values = np.array([5, 10, 15, 20])

   # Direct computation: 1 - CDF(x) loses precision for large x
   naive_survival = 1 - normal_param.cdf(x=large_values)

   # Complementary CDF provides better precision
   stable_survival = normal_param.ccdf(x=large_values)

   # Log-space for extreme tails
   log_survival = normal_param.log_ccdf(x=large_values)

Integration with Stan
--------------------

**Stan CDF Functions:**

.. code-block:: python

   # CDF transformations generate appropriate Stan code
   normal_param = ssp.parameters.Normal(mu=0, sigma=1)
   data_points = ssp.constants.Constant([0, 1, 2])

   # Generates Stan: normal_cdf(data_points | 0, 1)
   cdf_transform = normal_param.cdf(x=data_points)

**Custom Distribution CDFs:**

.. code-block:: python

   # Custom distributions automatically get CDF methods
   custom_param = ssp.parameters.ExpExponential(beta=1.0)

   # Uses custom Stan functions for CDF computation
   custom_cdf = custom_param.cdf(x=data_points)
   # Generates: expexponential_cdf(data_points | 1.0)

Advanced CDF Applications
------------------------

**Truncated Distributions:**

.. code-block:: python

   # Create truncated distributions using CDFs
   def truncated_normal(mu, sigma, lower, upper):
       """Create truncated normal using CDF normalization."""
       base_param = ssp.parameters.Normal(mu=mu, sigma=sigma)

       # Normalization constants
       lower_cdf = base_param.cdf(x=lower)
       upper_cdf = base_param.cdf(x=upper)
       normalization = upper_cdf - lower_cdf

       return base_param, normalization

**Mixture Model Components:**

.. code-block:: python

   # Use CDFs for mixture model construction
   def mixture_cdf(components, weights, x):
       """Compute CDF of mixture distribution."""
       mixture_cdf_vals = ssp.operations.sum_(
           weights * ssp.operations.stack([
               comp.cdf(x=x) for comp in components
           ])
       )
       return mixture_cdf_vals

**Change Point Detection:**

.. code-block:: python

   # Use CDFs for change point analysis
   def change_point_likelihood(data, change_point, params_before, params_after):
       """Likelihood function for change point model."""
       before_data = data[:change_point]
       after_data = data[change_point:]

       # Use log-CDFs for numerical stability
       before_loglik = params_before.log_cdf(x=before_data)
       after_loglik = params_after.log_cdf(x=after_data)

       return ssp.operations.sum_(before_loglik) + ssp.operations.sum_(after_loglik)

Model Checking Workflows
-----------------------

**Posterior Predictive Checking:**

.. code-block:: python

   # Complete posterior predictive checking workflow
   def posterior_predictive_check(model_result, observed_data):
       """Comprehensive model validation using CDFs."""

       # Get posterior predictive samples
       pp_samples = model_result.posterior_predictive_samples

       # For each posterior sample, compute CDF values
       cdf_values = []
       for param_sample in pp_samples:
           cdf_vals = param_sample.cdf(x=observed_data)
           cdf_values.append(cdf_vals)

       # Analyze distribution of CDF values
       cdf_array = np.array(cdf_values)
       return {
           'mean_cdf': cdf_array.mean(axis=0),
           'cdf_std': cdf_array.std(axis=0),
           'cdf_quantiles': np.percentile(cdf_array, [25, 50, 75], axis=0)
       }

**Calibration Assessment:**

.. code-block:: python

   # Assess model calibration using CDF transforms
   def assess_calibration(fitted_params, validation_data):
       """Assess model calibration quality."""

       # Transform validation data using fitted model
       pit_values = fitted_params.cdf(x=validation_data)

       # Well-calibrated models produce uniform PIT values
       # Test for uniformity
       from scipy.stats import kstest, anderson

       # Kolmogorov-Smirnov test
       ks_stat, ks_p = kstest(pit_values, 'uniform')

       # Anderson-Darling test
       ad_stat, ad_critical, ad_significance = anderson(pit_values, 'uniform')

       return {
           'ks_statistic': ks_stat,
           'ks_p_value': ks_p,
           'ad_statistic': ad_stat,
           'calibration_quality': 'good' if ks_p > 0.05 else 'poor'
       }

Computational Considerations
---------------------------

**Performance Optimization:**

.. code-block:: python

   # Vectorized CDF computations
   large_dataset = np.random.randn(10000)

   # Efficient: Single vectorized call
   cdf_values = normal_param.cdf(x=large_dataset)

   # Inefficient: Loop over individual values
   # cdf_values = [normal_param.cdf(x=val) for val in large_dataset]

**Memory Management:**

.. code-block:: python

   # For very large datasets, process in chunks
   def chunked_cdf_computation(param, data, chunk_size=1000):
       """Compute CDFs in memory-efficient chunks."""
       results = []
       for i in range(0, len(data), chunk_size):
           chunk = data[i:i+chunk_size]
           chunk_cdf = param.cdf(x=chunk)
           results.append(chunk_cdf)
       return np.concatenate(results)

Integration Examples
-------------------

**Survival Analysis:**

.. code-block:: python

   # Survival analysis using CDFs
   survival_times = ssp.parameters.Exponential(beta=failure_rate)
   observed_times = np.array([1.2, 2.5, 3.1, 4.8, 6.2])

   # Survival function: P(T > t)
   survival_probs = survival_times.ccdf(x=observed_times)

   # Hazard function using CDFs and PDFs
   cdf_vals = survival_times.cdf(x=observed_times)
   pdf_vals = survival_times.log_prob(observed_times)  # In log space
   hazard = pdf_vals - survival_times.log_ccdf(x=observed_times)

**Financial Risk Modeling:**

.. code-block:: python

   # Value at Risk (VaR) using CDFs
   returns = ssp.parameters.Normal(mu=expected_return, sigma=volatility)
   confidence_level = 0.05  # 5% VaR

   # Find quantile corresponding to confidence level
   # This would typically require inverse CDF
   # But can be approximated using CDF evaluations
   var_estimate = estimate_quantile(returns, confidence_level)

Best Practices
-------------

1. **Use log-space CDFs** for numerical stability with extreme values
2. **Leverage vectorization** for efficient computation on large datasets
3. **Apply CDFs for model validation** through probability transforms
4. **Use complementary CDFs** for accurate tail probability computation
5. **Implement calibration checks** as standard model validation practice
6. **Process large datasets in chunks** to manage memory usage
7. **Validate CDF implementations** against known analytical results

The CDF framework provides essential tools for model validation, probability transforms, and statistical analysis while maintaining numerical stability and computational efficiency.
