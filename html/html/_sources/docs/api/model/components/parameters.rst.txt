Parameters API Reference
========================

This reference covers the parameter components and probability distributions available in SciStanPy.

Parameters Module
-----------------

.. automodule:: scistanpy.model.components.parameters
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Core Parameter Infrastructure
-----------------------------

ParameterMeta
~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.parameters.ParameterMeta
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Automatic CDF Generation:**

   The ParameterMeta metaclass automatically creates CDF-related transform classes for each parameter:

   .. code-block:: python

      # CDF transforms are automatically available
      normal_param = ssp.parameters.Normal(mu=0, sigma=1)

      # Access CDF transforms
      cdf_transform = normal_param.cdf(x=data_points)
      survival_transform = normal_param.ccdf(x=data_points)
      log_cdf_transform = normal_param.log_cdf(x=data_points)
      log_survival_transform = normal_param.log_ccdf(x=data_points)

ClassOrInstanceMethod
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.parameters.ClassOrInstanceMethod
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Flexible CDF Access:**

   .. code-block:: python

      # As instance method (uses parameter's values)
      normal_param = ssp.parameters.Normal(mu=0, sigma=1)
      cdf = normal_param.cdf(x=np.array([0, 1, 2]))

      # As class method (explicit parameters)
      cdf = ssp.parameters.Normal.cdf(mu=0, sigma=1, x=np.array([0, 1, 2]))

Base Parameter Classes
----------------------

Parameter
~~~~~~~~~

.. autoclass:: scistanpy.model.components.parameters.Parameter
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Core Parameter Functionality:**

   The Parameter class provides the foundation for all probabilistic components:

   .. code-block:: python

      # Basic parameter creation
      mu = ssp.parameters.Normal(mu=0, sigma=1)

      # Multi-dimensional parameters
      group_effects = ssp.parameters.Normal(
          mu=0, sigma=1, shape=(10,)  # 10 group effects
      )

      # Observable parameters
      y = ssp.parameters.Normal(mu=mu, sigma=0.5).as_observable()

   **PyTorch Integration:**

   .. code-block:: python

      # Initialize for PyTorch optimization
      param = ssp.parameters.Normal(mu=0, sigma=1)
      param.init_pytorch(init_val=0.5)

      # Access PyTorch tensor with constraints
      tensor = param.torch_parametrization

      # Compute log probabilities for gradient descent
      log_prob = param.get_torch_logprob()

ContinuousDistribution
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.parameters.ContinuousDistribution
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Mathematical Transformation Support:**

   .. code-block:: python

      # Continuous parameters support transformations
      x = ssp.parameters.Normal(mu=0, sigma=1)
      y = ssp.parameters.Normal(mu=0, sigma=1)

      # Mathematical operations create transformations
      sum_param = x + y
      product_param = x * y
      log_param = ssp.operations.log(x)

DiscreteDistribution
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.parameters.DiscreteDistribution
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Integer-Valued Distributions:**

   .. code-block:: python

      # Discrete parameters for count data
      counts = ssp.parameters.Poisson(lambda_=5)
      successes = ssp.parameters.Binomial(N=10, theta=0.3)

Continuous Distributions
------------------------

Normal Distribution
~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.parameters.Normal
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Standard and Non-Centered Parameterization:**

   .. code-block:: python

      # Standard normal parameter
      mu = ssp.parameters.Normal(mu=0, sigma=1)

      # Hierarchical with automatic non-centering
      sigma_global = ssp.parameters.LogNormal(mu=0, sigma=1)
      group_effects = ssp.parameters.Normal(
          mu=0,
          sigma=sigma_global,
          shape=(10,),
          noncentered=True  # Default for hierarchical parameters
      )

      # Force centered parameterization
      centered_param = ssp.parameters.Normal(
          mu=mu, sigma=1, noncentered=False
      )

   **Non-Centered Benefits:**

   .. code-block:: python

      # Non-centered parameterization improves MCMC efficiency
      # Stan code: z_raw ~ std_normal(); z = mu + sigma * z_raw;

      # Automatic detection of when to use non-centering:
      # - Not a hyperparameter (has parameter parents)
      # - Not observable
      # - noncentered=True (default)

HalfNormal Distribution
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.parameters.HalfNormal
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Positive-Constrained Normal:**

   .. code-block:: python

      # Scale parameters with half-normal priors
      tau = ssp.parameters.HalfNormal(sigma=1.0)

      # Hierarchical scale modeling
      global_scale = ssp.parameters.HalfNormal(sigma=2.0)
      local_scales = ssp.parameters.HalfNormal(sigma=global_scale)

UnitNormal Distribution
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.parameters.UnitNormal
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Standard Normal Convenience:**

   .. code-block:: python

      # Fixed N(0,1) distribution
      z = ssp.parameters.UnitNormal()

      # Useful for non-centered parameterizations
      z_raw = ssp.parameters.UnitNormal()
      x = mu + sigma * z_raw  # Conceptual transformation

LogNormal Distribution
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.parameters.LogNormal
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Positive-Valued with Log-Normal Structure:**

   .. code-block:: python

      # Scale parameters
      sigma = ssp.parameters.LogNormal(mu=0, sigma=1)

      # Multiplicative effects
      growth_factor = ssp.parameters.LogNormal(mu=np.log(1.1), sigma=0.1)

Beta Distribution
~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.parameters.Beta
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Probability and Proportion Modeling:**

   .. code-block:: python

      # Uniform prior on [0,1]
      p = ssp.parameters.Beta(alpha=1, beta=1)

      # Informative prior favoring small probabilities
      p_rare = ssp.parameters.Beta(alpha=1, beta=9)  # Mean = 0.1

      # Informative prior around 0.7
      p_informed = ssp.parameters.Beta(alpha=7, beta=3)

Gamma Distribution
~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.parameters.Gamma
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Shape-Rate Parameterization:**

   .. code-block:: python

      # Precision parameter (inverse variance)
      tau = ssp.parameters.Gamma(alpha=2, beta=1)  # Rate parameterization

      # Positive continuous variables
      waiting_time = ssp.parameters.Gamma(alpha=shape, beta=rate)

InverseGamma Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.parameters.InverseGamma
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Conjugate Prior for Variance:**

   .. code-block:: python

      # Variance parameter
      sigma_squared = ssp.parameters.InverseGamma(alpha=2, beta=1)

      # Hierarchical variance modeling
      tau_squared = ssp.parameters.InverseGamma(alpha=a_prior, beta=b_prior)

Exponential Distribution
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.parameters.Exponential
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Waiting Times and Survival:**

   .. code-block:: python

      # Waiting time with rate parameterization
      wait_time = ssp.parameters.Exponential(beta=1.5)

      # Survival analysis
      event_time = ssp.parameters.Exponential(beta=hazard_rate)

Custom Continuous Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.parameters.ExpExponential
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.parameters.Lomax
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.parameters.ExpLomax
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Heavy-Tailed and Log-Transformed Distributions:**

   .. code-block:: python

      # Heavy-tailed distributions for robust modeling
      wealth = ssp.parameters.Lomax(lambda_=1.0, alpha=2.0)

      # Log-scale heavy-tailed
      log_income = ssp.parameters.ExpLomax(lambda_=1.0, alpha=2.0)

      # Log-exponential (Gumbel-type)
      log_waiting_time = ssp.parameters.ExpExponential(beta=1.0)

Simplex Distributions
---------------------

Dirichlet Distribution
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.parameters.Dirichlet
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Probability Simplex Modeling:**

   .. code-block:: python

      # Symmetric Dirichlet (uniform on simplex)
      p = ssp.parameters.Dirichlet(alpha=1.0, shape=(4,))

      # Non-uniform concentrations
      p = ssp.parameters.Dirichlet(alpha=np.array([0.5, 1.0, 2.0, 0.5]))

      # Hierarchical concentration
      alpha_vec = ssp.parameters.Gamma(alpha=1, beta=1, shape=(4,))
      p = ssp.parameters.Dirichlet(alpha=alpha_vec)

ExpDirichlet Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.parameters.ExpDirichlet
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Log-Simplex for Numerical Stability:**

   .. code-block:: python

      # Log-probability vectors for high-dimensional problems
      log_p = ssp.parameters.ExpDirichlet(alpha=1.0, shape=(1000,))

      # More stable for extreme concentrations
      log_weights = ssp.parameters.ExpDirichlet(alpha=alpha_vec)

      # Constraint: exp(log_p).sum() == 1

Discrete Distributions
----------------------

Binomial Distribution
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.parameters.Binomial
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Success Counts in Fixed Trials:**

   .. code-block:: python

      # Number of successes in N trials
      successes = ssp.parameters.Binomial(N=20, theta=success_rate)

      # Proportion data
      observed_successes = ssp.parameters.Binomial(
          N=trial_counts, theta=success_probability
      ).as_observable()

Poisson Distribution
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.parameters.Poisson
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Count Data Modeling:**

   .. code-block:: python

      # Event counts
      counts = ssp.parameters.Poisson(lambda_=event_rate)

      # Observed count data
      observed_counts = ssp.parameters.Poisson(lambda_=fitted_rate).as_observable()

Multinomial Distributions
-------------------------

Standard Multinomial
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.parameters.Multinomial
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Categorical Count Data:**

   .. code-block:: python

      # Category counts with probability vector
      category_counts = ssp.parameters.Multinomial(
          theta=category_probs,  # Must sum to 1
          N=total_trials
      )

MultinomialLogit
~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.parameters.MultinomialLogit
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Logit-Parameterized Multinomial:**

   .. code-block:: python

      # Unconstrained parameterization
      logits = ssp.parameters.Normal(mu=0, sigma=1, shape=(5,))
      counts = ssp.parameters.MultinomialLogit(
          gamma=logits,  # Unconstrained
          N=total_trials
      )

MultinomialLogTheta
~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.parameters.MultinomialLogTheta
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Log-Probability Parameterized with Optimization:**

   .. code-block:: python

      # Log-simplex parameterization
      log_probs = ssp.parameters.ExpDirichlet(alpha=[1, 1, 1, 1])
      counts = ssp.parameters.MultinomialLogTheta(
          log_theta=log_probs,  # Log-simplex constraint
          N=total_trials
      )

      # Automatic coefficient optimization for observables
      observed_counts = ssp.parameters.MultinomialLogTheta(
          log_theta=log_probs, N=100
      ).as_observable()
      # Multinomial coefficient pre-computed for efficiency

Advanced Parameter Features
---------------------------

**Non-Centered Parameterization:**

.. code-block:: python

   # Automatic non-centering for hierarchical models
   def hierarchical_model():
       # Global parameters
       mu_global = ssp.parameters.Normal(mu=0, sigma=5)
       sigma_global = ssp.parameters.LogNormal(mu=0, sigma=1)

       # Group-level parameters (automatically non-centered)
       group_effects = ssp.parameters.Normal(
           mu=mu_global,
           sigma=sigma_global,
           shape=(10,)  # Automatically uses non-centered parameterization
       )

       return group_effects

**Observable Parameter Management:**

.. code-block:: python

   # Automatic observable detection
   def create_model_with_observables():
       # Parameters
       mu = ssp.parameters.Normal(mu=0, sigma=1)
       sigma = ssp.parameters.LogNormal(mu=0, sigma=0.5)

       # Observable (no children = automatically observable)
       y = ssp.parameters.Normal(mu=mu, sigma=sigma)

       # Explicitly mark as observable
       y_explicit = ssp.parameters.Normal(mu=mu, sigma=sigma).as_observable()

       return y, y_explicit

**CDF and Probability Functions:**

.. code-block:: python

   # Comprehensive probability function access
   normal_param = ssp.parameters.Normal(mu=0, sigma=1)
   data_points = np.array([-2, -1, 0, 1, 2])

   # CDF functions
   cdf_vals = normal_param.cdf(x=data_points)
   survival_vals = normal_param.ccdf(x=data_points)

   # Log-space for numerical stability
   log_cdf_vals = normal_param.log_cdf(x=data_points)
   log_survival_vals = normal_param.log_ccdf(x=data_points)

**Multi-Backend Integration:**

.. code-block:: python

   # SciPy backend for CPU sampling
   samples, _ = normal_param.draw(n=1000)

   # PyTorch backend for GPU optimization
   normal_param.init_pytorch()
   torch_tensor = normal_param.torch_parametrization
   log_prob = normal_param.get_torch_logprob()

   # Stan backend for MCMC
   model = ssp.Model(normal_param)
   stan_results = model.mcmc(chains=4, iter_sampling=1000)

Parameter Constraints and Bounds
--------------------------------

**Automatic Constraint Handling:**

.. code-block:: python

   # Positive parameters
   positive_param = ssp.parameters.Gamma(alpha=2, beta=1)  # > 0

   # Bounded parameters
   bounded_param = ssp.parameters.Beta(alpha=2, beta=3)   # ∈ (0, 1)

   # Simplex parameters
   simplex_param = ssp.parameters.Dirichlet(alpha=[1, 1, 1])  # Sums to 1

   # Custom bounds via constants
   custom_bounded = ssp.parameters.Normal(
       mu=ssp.constants.Constant(5.0, lower_bound=0, upper_bound=10),
       sigma=1
   )

**Stan Code Generation with Constraints:**

.. code-block:: python

   # Automatic Stan constraint generation
   param = ssp.parameters.Gamma(alpha=2, beta=1)
   stan_type = param.get_stan_dtype()  # "real<lower=0.0>"

   # Multi-dimensional with constraints
   param_vector = ssp.parameters.Gamma(alpha=2, beta=1, shape=(5,))
   stan_type = param_vector.get_stan_dtype()  # "vector<lower=0.0>[5]"

Model Building Patterns
-----------------------

**Hierarchical Model Construction:**

.. code-block:: python

   def build_hierarchical_model(group_sizes):
       """Build a hierarchical model with automatic parameterization."""

       # Population-level parameters
       mu_pop = ssp.parameters.Normal(mu=0, sigma=5)
       sigma_pop = ssp.parameters.LogNormal(mu=0, sigma=1)

       # Group-level parameters (automatic non-centering)
       mu_groups = ssp.parameters.Normal(
           mu=mu_pop,
           sigma=sigma_pop,
           shape=(len(group_sizes),)
       )

       # Individual-level parameters
       sigma_indiv = ssp.parameters.LogNormal(mu=-1, sigma=0.5)

       # Observations
       observations = []
       for i, size in enumerate(group_sizes):
           obs = ssp.parameters.Normal(
               mu=mu_groups[i],
               sigma=sigma_indiv,
               shape=(size,)
           ).as_observable()
           observations.append(obs)

       return {
           'population': {'mu': mu_pop, 'sigma': sigma_pop},
           'groups': mu_groups,
           'individual_sigma': sigma_indiv,
           'observations': observations
       }

**Mixture Model Construction:**

.. code-block:: python

   def build_mixture_model(n_components, n_obs):
       """Build a finite mixture model."""

       # Mixture weights
       weights = ssp.parameters.Dirichlet(alpha=1.0, shape=(n_components,))

       # Component parameters
       component_means = ssp.parameters.Normal(
           mu=0, sigma=5, shape=(n_components,)
       )
       component_sds = ssp.parameters.LogNormal(
           mu=0, sigma=1, shape=(n_components,)
       )

       # Mixture assignments (latent)
       assignments = ssp.parameters.MultinomialLogit(
           gamma=ssp.operations.log(weights),  # Convert weights to logits
           N=1,
           shape=(n_obs,)
       )

       # Observations (mixture likelihood would need custom implementation)
       # This is a conceptual example

       return {
           'weights': weights,
           'means': component_means,
           'sds': component_sds,
           'assignments': assignments
       }

Performance Optimization Tips
-----------------------------

**Efficient Parameter Specification:**

.. code-block:: python

   # Efficient: Single multi-dimensional parameter
   efficient = ssp.parameters.Normal(mu=0, sigma=1, shape=(100,))

   # Less efficient: Many individual parameters
   # inefficient = [ssp.parameters.Normal(mu=0, sigma=1) for _ in range(100)]

**Memory-Conscious Model Building:**

.. code-block:: python

   # Share parent parameters when possible
   shared_sigma = ssp.parameters.LogNormal(mu=0, sigma=1)

   observations = []
   for i in range(10):
       obs = ssp.parameters.Normal(
           mu=0,
           sigma=shared_sigma,  # Shared, not duplicated
           shape=(100,)
       )
       observations.append(obs)

**Stan Code Optimization:**

.. code-block:: python

   # SciStanPy automatically optimizes Stan code:
   # - Non-centered parameterization for hierarchical models
   # - Vectorized operations where possible
   # - Efficient constraint handling
   # - Optimal loop structure for multi-dimensional parameters

Best Practices
--------------

1. **Use appropriate parameterizations** for your modeling context
2. **Leverage automatic non-centering** for hierarchical models
3. **Choose efficient distributions** for your data type (continuous vs discrete)
4. **Use shape parameters** instead of creating many individual parameters
5. **Mark observables explicitly** when the automatic detection isn't sufficient
6. **Validate parameter bounds** match your domain knowledge
7. **Use CDF functions** for truncated distributions and model validation

The parameter system provides a comprehensive foundation for probabilistic modeling while maintaining mathematical rigor and computational efficiency across multiple backends.
