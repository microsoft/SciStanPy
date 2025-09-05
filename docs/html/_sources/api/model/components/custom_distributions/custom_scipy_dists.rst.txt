Custom SciPy Distributions API Reference
========================================

This reference covers the custom SciPy-based probability distributions in SciStanPy.

Custom SciPy Distributions Module
---------------------------------

.. automodule:: scistanpy.model.components.custom_distributions.custom_scipy_dists

   :members:
   :undoc-members:
   :show-inheritance:

expdirichlet Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.custom_distributions.custom_scipy_dists.expdirichlet

   :members:
   :undoc-members:
   :show-inheritance:

   **Log-Simplex Distributed Random Variables (SciPy):**

   .. code-block:: python

      # expdirichlet distribution with SciPy backend (symmetric to PyTorch implementation)
      alpha = np.ones(4)
      exp_dirichlet_scipy = ssp.expdirichlet(alpha=alpha)

      # Generate samples and evaluate densities
      samples = exp_dirichlet_scipy.rvs(size=100)
      log_probs = exp_dirichlet_scipy.logpdf(samples)

      # Verify log-simplex constraint
      recovered_simplex = np.exp(samples)
      assert np.allclose(recovered_simplex.sum(axis=-1), 1.0)

   **Mathematical Properties:**

   The SciPy implementation maintains consistency with PyTorch:
   - Support: R^(K-1) where K is the dimension of the original simplex
   - Constraint: sum(exp(y)) = 1
   - Dense evaluation using SciPy's statistical framework

explomax Implementation
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.custom_distributions.custom_scipy_dists.explomax

   :members:
   :undoc-members:
   :show-inheritance:

   **Heavy-Tailed Log-Scale Distribution (SciPy):**

   .. code-block:: python

      # SciPy backend for exp-Lomax distribution
      lambda_param = 1.0
      alpha = 1.5
      exp_lomax_scipy = ssp.explomax(lambda_=lambda_param, alpha=alpha)

      # Statistical analysis using SciPy
      samples = exp_lomax_scipy.rvs(size=1000)

      # Distribution properties
      mean_val = exp_lomax_scipy.mean()
      var_val = exp_lomax_scipy.var()

      # Tail analysis
      large_values = np.array([5, 10, 15])
      tail_probs = exp_lomax_scipy.sf(large_values)  # Survival function

   **Applications in Scientific Computing:**

   .. code-block:: python

      # Financial modeling: log-returns with heavy tails
      log_returns = exp_lomax_scipy.rvs(size=252)  # One year of daily returns

      # Extreme value analysis
      threshold = exp_lomax_scipy.ppf(0.95)  # 95th percentile
      extreme_events = samples[samples > threshold]

multinomial_logit Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.custom_distributions.custom_scipy_dists.multinomial_logit

   :members:
   :undoc-members:
   :show-inheritance:

   **Unconstrained Multinomial (SciPy):**

   .. code-block:: python

      # Multinomial with logit parameterization
      logits = np.array([0.5, -0.2, 0.8, -0.1])
      total_trials = 100
      multinomial_logit_scipy = ssp.multinomial_logit(
          gamma=logits, N=total_trials
      )

      # Sample category counts
      counts = multinomial_logit_scipy.rvs()
      assert counts.sum() == total_trials

      # Compute probabilities
      probabilities = multinomial_logit_scipy.get_probabilities()
      assert np.allclose(probabilities.sum(), 1.0)

multinomial_log_theta Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.custom_distributions.custom_scipy_dists.multinomial_log_theta

   :members:
   :undoc-members:
   :show-inheritance:

   **Log-Probability Multinomial (SciPy):**

   .. code-block:: python

      # Multinomial with log-probability parameterization
      log_probs = np.log([0.2, 0.3, 0.1, 0.4])
      total_trials = 500
      multinomial_logtheta_scipy = ssp.multinomial_log_theta(
          log_theta=log_probs, N=total_trials
      )

      # Efficient sampling for large N
      large_sample = multinomial_logtheta_scipy.rvs()

      # Log-probability evaluation (numerically stable)
      observed_counts = np.array([98, 152, 48, 202])
      log_likelihood = multinomial_logtheta_scipy.logpmf(observed_counts)

Backend Comparison and Integration
----------------------------------

**SciPy vs PyTorch Implementations:**

.. code-block:: python

   # Compare implementations for consistency
   alpha = np.array([2.0, 3.0, 1.5])

   # SciPy backend
   exp_dir_scipy = ssp.expdirichlet(alpha=alpha)
   scipy_samples = exp_dir_scipy.rvs(size=1000)

   # PyTorch backend
   exp_dir_torch = ssp.expdirichlet(alpha=torch.tensor(alpha))
   torch_samples = exp_dir_torch.sample((1000,)).detach().numpy()

   # Statistical comparison
   scipy_mean = scipy_samples.mean(axis=0)
   torch_mean = torch_samples.mean(axis=0)
   print(f"Mean difference: {np.abs(scipy_mean - torch_mean).max()}")

**When to Use Each Backend:**

.. code-block:: python

   # SciPy backend advantages:
   # - Mature statistical functions (CDF, SF, PPF)
   # - Robust numerical implementations
   # - Extensive statistical analysis tools
   # - CPU-optimized performance

   # PyTorch backend advantages:
   # - Automatic differentiation
   # - GPU acceleration
   # - Batch processing
   # - Integration with neural networks

Performance Characteristics
---------------------------

**CPU Performance:**

.. code-block:: python

   # SciPy optimized for CPU statistical computing
   large_alpha = np.ones(1000)
   exp_dir = ssp.expdirichlet(alpha=large_alpha)

   # Efficient sampling
   samples = exp_dir.rvs(size=10000)  # Fast CPU sampling

   # Statistical analysis
   percentiles = np.percentile(samples, [25, 50, 75], axis=0)

**Memory Efficiency:**

.. code-block:: python

   # Memory-efficient processing with SciPy
   def batch_evaluate(distribution, data, batch_size=1000):
       """Evaluate in batches to manage memory."""
       results = []
       for i in range(0, len(data), batch_size):
           batch = data[i:i+batch_size]
           batch_results = distribution.logpdf(batch)
           results.append(batch_results)
       return np.concatenate(results)

**Statistical Analysis Integration:**

.. code-block:: python

   # Leverage SciPy ecosystem for analysis
   from scipy import stats

   # Distribution fitting
   sample_data = lomax_scipy.rvs(size=1000)

   # Goodness of fit tests
   ks_stat, p_value = stats.kstest(
       sample_data,
       lambda x: lomax_scipy.cdf(x)
   )

   # Confidence intervals
   alpha_level = 0.05
   lower = lomax_scipy.ppf(alpha_level / 2)
   upper = lomax_scipy.ppf(1 - alpha_level / 2)

Integration with SciStanPy Models
---------------------------------

**Automatic Backend Selection:**

.. code-block:: python

   # SciStanPy automatically selects appropriate backend
   alpha = [1.0, 1.0, 1.0]

   # This uses PyTorch backend for model building
   exp_dir_param = ssp.parameters.expdirichlet(alpha=alpha)

   # But can use SciPy backend for analysis
   samples, _ = exp_dir_param.draw(n=1000)  # Uses SciPy for sampling

**Custom Distribution Development:**

.. code-block:: python

   # Template for new SciPy distribution
   from scipy.stats import rv_continuous

   class CustomSciPyDistribution(rv_continuous):
       """Template for custom SciPy distribution."""

       def __init__(self, custom_param, **kwargs):
           super().__init__(**kwargs)
           self.custom_param = custom_param

       def _pdf(self, x, custom_param):
           # Implement probability density function
           return density_function(x, custom_param)

       def _cdf(self, x, custom_param):
           # Implement cumulative distribution function
           return cdf_function(x, custom_param)

       def _rvs(self, custom_param, size=None, random_state=None):
           # Implement random number generation
           return generate_samples(custom_param, size, random_state)

**Validation and Testing:**

.. code-block:: python

   # Validate custom distributions
   def validate_distribution(scipy_dist, torch_dist, alpha):
       """Compare SciPy and PyTorch implementations."""

       # Sample comparison
       scipy_samples = scipy_dist.rvs(size=1000)
       torch_samples = torch_dist.sample((1000,)).detach().numpy()

       # Statistical moments
       scipy_mean = scipy_samples.mean(axis=0)
       torch_mean = torch_samples.mean(axis=0)

       # Check consistency
       mean_diff = np.abs(scipy_mean - torch_mean).max()
       assert mean_diff < 0.1, f"Implementations differ: {mean_diff}"

       print("✓ Implementations are consistent")

Numerical Considerations
------------------------

**Precision and Stability:**

.. code-block:: python

   # SciPy provides robust numerical implementations

   # Handle extreme parameter values
   extreme_alpha = np.array([1e-6, 1e6, 1e-3])
   stable_dist = ssp.expdirichlet(alpha=extreme_alpha)

   # Numerical stability in log-space
   very_small_probs = stable_dist.logpdf(extreme_samples)

   # Automatic overflow/underflow handling
   assert np.all(np.isfinite(very_small_probs))

**Error Handling:**

.. code-block:: python

   # Comprehensive parameter validation
   try:
       invalid_dist = ssp.Lomax(lambda_=-1.0, alpha=1.5)
   except ValueError as e:
       print(f"Parameter validation: {e}")

   # Runtime checks
   try:
       invalid_sample = lomax_dist.logpdf(np.array([1, 2, -1]))
   except ValueError as e:
       print(f"Sample validation: {e}")

Best Practices
--------------

1. **Use SciPy backend** for statistical analysis and exploratory work
2. **Leverage PyTorch backend** for gradient-based optimization
3. **Validate implementations** when developing custom distributions
4. **Handle extreme values** with appropriate numerical techniques
5. **Test statistical properties** against known analytical results
6. **Use appropriate precision** for your computational requirements
7. **Profile performance** for computational bottlenecks

The SciPy custom distributions provide robust, well-tested implementations that integrate seamlessly with the broader scientific Python ecosystem while maintaining full compatibility with SciStanPy's modeling framework.