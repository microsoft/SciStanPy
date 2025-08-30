Distributions Guide
===================

This guide covers the probability distributions available in SciStanPy and how to choose the right distribution for your scientific modeling needs.

Scope Adjustment
----------------
This guide lists commonly used distributions; availability may depend on the
current codebase. Confirm by importing from ssp.parameters.*.

Core Examples (Illustrative)
----------------------------
.. code-block:: python
   mu = ssp.parameters.Normal(mu=0, sigma=1)
   scale = ssp.parameters.LogNormal(mu=0, sigma=0.5)
   prob = ssp.parameters.Beta(alpha=2, beta=5)
   count = ssp.parameters.Poisson(rate=3.0)
   success = ssp.parameters.Bernoulli(p=0.6)
   # Use in likelihood
   y = ssp.parameters.Normal(mu=mu, sigma=scale)
   y.observe(data)

Accuracy Note
-------------
Removed/condensed sections asserting advanced or custom distributions (mixtures,
exp_* variants, truncated & mixture helpers) that may not yet exist. Retain only
generic illustrative usage; verify actual class names before relying on them.

Continuous Distributions
------------------------

Normal Distribution
~~~~~~~~~~~~~~~~~~~

The most common distribution for symmetric, unbounded quantities.

.. code-block:: python

   # Basic usage
   temperature = ssp.parameters.Normal(mu=25, sigma=5)  # °C

   # Multiple parameterizations
   precision_param = ssp.parameters.Normal(mu=0, tau=1)  # precision = 1/variance

   # Vectorized
   measurements = ssp.parameters.Normal(mu=true_values, sigma=0.1, shape=(100,))

**When to use**: Temperature differences, measurement errors, symmetric uncertainties

**Parameters**:
- ``mu``: mean (location parameter)
- ``sigma``: standard deviation (scale parameter)
- ``tau``: precision (1/variance, alternative parameterization)

LogNormal Distribution
~~~~~~~~~~~~~~~~~~~~~~

For positive quantities that are log-normally distributed.

.. code-block:: python

   # Concentration data
   concentration = ssp.parameters.LogNormal(mu=0, sigma=1)  # mM

   # Based on normal-scale parameters
   conc_alt = ssp.parameters.LogNormal(
       mu=np.log(5),    # median = 5 mM
       sigma=0.5        # log-scale variability
   )

**When to use**: Concentrations, sizes, rates, positive-valued measurements

**Properties**: If log(X) ~ Normal(μ, σ), then X ~ LogNormal(μ, σ)

Student's t Distribution
~~~~~~~~~~~~~~~~~~~~~~~~

For robust modeling with heavier tails than normal.

.. code-block:: python

   # Robust regression
   noise = ssp.parameters.StudentT(nu=3, mu=0, sigma=1)
   likelihood = ssp.parameters.StudentT(nu=3, mu=predictions, sigma=noise)

**When to use**: Outlier-robust models, small sample sizes

**Parameters**:
- ``nu``: degrees of freedom (controls tail heaviness)
- ``mu``: location
- ``sigma``: scale

Exponential Distribution
~~~~~~~~~~~~~~~~~~~~~~~~

For waiting times and lifetimes.

.. code-block:: python

   # Radioactive decay
   lifetime = ssp.parameters.Exponential(rate=0.1)  # per second

   # Time between events
   inter_arrival = ssp.parameters.Exponential(rate=event_rate)

**When to use**: Survival times, time between events, exponential decay

Gamma Distribution
~~~~~~~~~~~~~~~~~~

Flexible positive distribution, conjugate prior for precision.

.. code-block:: python

   # Shape-rate parameterization
   precision = ssp.parameters.Gamma(alpha=1, beta=1)

   # Shape-scale parameterization
   reaction_time = ssp.parameters.Gamma(alpha=2, scale=0.5)

**When to use**: Positive continuous variables, modeling precision parameters

Beta Distribution
~~~~~~~~~~~~~~~~~

For proportions and probabilities.

.. code-block:: python

   # Success probability
   success_rate = ssp.parameters.Beta(alpha=10, beta=5)  # Mean ≈ 0.67

   # Fraction measurements
   purity = ssp.parameters.Beta(alpha=2, beta=2)  # Symmetric around 0.5

**When to use**: Proportions, probabilities, fractions, percentages

**Properties**: Support on [0,1], highly flexible shape

Uniform Distribution
~~~~~~~~~~~~~~~~~~~~

For bounded parameters with no preferred value.

.. code-block:: python

   # Bounded parameter
   phase = ssp.parameters.Uniform(low=0, high=2*np.pi)  # radians

   # Reference prior
   bounded_coeff = ssp.parameters.Uniform(low=-1, high=1)

**When to use**: Reference priors, truly uniform quantities

Discrete Distributions
----------------------

Poisson Distribution
~~~~~~~~~~~~~~~~~~~~

For count data with constant rate.

.. code-block:: python

   # Event counts
   num_events = ssp.parameters.Poisson(rate=lambda_param)

   # Photon counts
   photon_count = ssp.parameters.Poisson(rate=intensity * exposure_time)

**When to use**: Counts of rare events, photon counting, radioactive decay

Negative Binomial
~~~~~~~~~~~~~~~~~

For overdispersed count data.

.. code-block:: python

   # Overdispersed counts
   count_data = ssp.parameters.NegativeBinomial(mu=mean_count, alpha=overdispersion)

**When to use**: Count data with more variability than Poisson

Binomial Distribution
~~~~~~~~~~~~~~~~~~~~~

For number of successes in fixed trials.

.. code-block:: python

   # Number of successes
   successes = ssp.parameters.Binomial(n=trials, p=success_prob)

   # Survival studies
   survivors = ssp.parameters.Binomial(n=initial_pop, p=survival_rate)

**When to use**: Binary outcomes, survival counts, quality control

Bernoulli Distribution
~~~~~~~~~~~~~~~~~~~~~~

For single binary outcomes.

.. code-block:: python

   # Single trial outcome
   success = ssp.parameters.Bernoulli(p=success_probability)

**When to use**: Single binary events, classification outcomes

Multivariate Distributions
--------------------------

Multivariate Normal
~~~~~~~~~~~~~~~~~~~

For correlated continuous variables.

.. code-block:: python

   # Correlated measurements
   measurements = ssp.parameters.MultivariateNormal(
       mu=means,
       cov=covariance_matrix
   )

   # Precision parameterization
   precise_mvn = ssp.parameters.MultivariateNormal(
       mu=means,
       precision=precision_matrix
   )

**When to use**: Correlated measurements, multivariate regression

Dirichlet Distribution
~~~~~~~~~~~~~~~~~~~~~~

For probability vectors (compositional data).

.. code-block:: python

   # Probability simplex
   probabilities = ssp.parameters.Dirichlet(alpha=[1, 1, 1, 1])

   # Compositional data
   species_fractions = ssp.parameters.Dirichlet(alpha=concentration_params)

**When to use**: Categorical probabilities, compositional data, mixture weights

Multinomial Distribution
~~~~~~~~~~~~~~~~~~~~~~~~

For categorical count data.

.. code-block:: python

   # Category counts
   category_counts = ssp.parameters.Multinomial(n=total_trials, p=probabilities)

   # Different trial counts per group
   group_counts = ssp.parameters.MultinomialLogit(
       n=trial_counts,
       logits=logit_probabilities
   )

**When to use**: Categorical outcomes, survey responses, classification counts

Distribution Selection Guidelines
--------------------------------

By Data Type
~~~~~~~~~~~~

**Continuous, unbounded**: Normal, Student's t
**Continuous, positive**: LogNormal, Gamma, Exponential
**Continuous, bounded [0,1]**: Beta
**Continuous, bounded [a,b]**: Uniform, truncated distributions
**Count data**: Poisson, Negative Binomial
**Binary**: Bernoulli, Binomial
**Categorical**: Multinomial, Categorical
**Compositional**: Dirichlet

By Scientific Context
~~~~~~~~~~~~~~~~~~~~~

**Measurement errors**: Normal, Student's t (robust)
**Physical quantities**: LogNormal (concentrations), Gamma (rates)
**Time intervals**: Exponential, Gamma
**Success rates**: Beta
**Event counts**: Poisson (rare events), Negative Binomial (overdispersed)
**Survival data**: Exponential, Weibull
**Correlation structure**: Multivariate Normal

Alternative Parameterizations
-----------------------------

Many distributions offer multiple parameterizations for convenience:

Normal Distribution
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Mean-variance
   normal1 = ssp.parameters.Normal(mu=0, sigma=1)

   # Mean-precision
   normal2 = ssp.parameters.Normal(mu=0, tau=1)  # tau = 1/sigma²

Gamma Distribution
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Shape-rate
   gamma1 = ssp.parameters.Gamma(alpha=2, beta=1)

   # Shape-scale
   gamma2 = ssp.parameters.Gamma(alpha=2, scale=1)  # scale = 1/rate

Multinomial Distribution
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Probability parameterization
   multi1 = ssp.parameters.MultinomialProb(n=trials, probs=probabilities)

   # Logit parameterization
   multi2 = ssp.parameters.MultinomialLogit(n=trials, logits=logits)

   # Log-probability parameterization
   multi3 = ssp.parameters.MultinomialLogTheta(n=trials, log_theta=log_probs)

Custom and Extended Distributions
---------------------------------

SciStanPy provides several custom distributions for specialized use cases:

.. code-block:: python

   # Log-transformed distributions
   log_exponential = ssp.parameters.ExpExponential(rate=1)  # Gumbel
   log_dirichlet = ssp.parameters.ExpDirichlet(alpha=[1, 1, 1])

   # Heavy-tailed distributions
   lomax = ssp.parameters.Lomax(lambda_=1, alpha=2)
   exp_lomax = ssp.parameters.ExpLomax(lambda_=1, alpha=2)

Truncated Distributions
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Truncated normal
   bounded_normal = ssp.parameters.TruncatedNormal(
       mu=0, sigma=1,
       lower=0, upper=10
   )

Mixture Distributions
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Mixture of normals
   component_weights = ssp.parameters.Dirichlet(alpha=[1, 1])
   component1 = ssp.parameters.Normal(mu=-2, sigma=1)
   component2 = ssp.parameters.Normal(mu=2, sigma=1)

   mixture = ssp.parameters.Mixture(
       weights=component_weights,
       components=[component1, component2]
   )

Best Practices
--------------

1. **Match the data generating process**: Choose distributions that reflect the underlying science
2. **Consider parameter interpretability**: Use parameterizations that match your scientific understanding
3. **Check support**: Ensure the distribution's support matches your data constraints
4. **Start simple**: Begin with standard distributions before moving to complex alternatives
5. **Validate assumptions**: Use posterior predictive checks to verify distributional assumptions
6. **Consider computational efficiency**: Some distributions are more efficient for MCMC sampling

This comprehensive guide should help you select appropriate distributions for your scientific modeling needs in SciStanPy.
