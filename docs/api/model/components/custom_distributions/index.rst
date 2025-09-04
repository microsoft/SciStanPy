Custom Distributions API Reference
==================================

This reference covers the custom probability distributions available in SciStanPy for specialized modeling needs.

Custom Distributions Submodule Overview
---------------------------------------

The custom distributions submodule provides specialized probability distributions that extend the standard library for scientific modeling applications:

.. toctree::
   :maxdepth: 1

   custom_torch_dists
   custom_scipy_dists

Custom Distributions Framework
------------------------------

.. automodule:: scistanpy.model.components.custom_distributions
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Distribution Categories
-----------------------

**Log-Transformed Distributions**
   Distributions defined on the logarithm of standard distributions for improved numerical stability and parameterization flexibility

**Heavy-Tailed Distributions**
   Distributions with power-law or polynomial tails for robust modeling of extreme events

**Constrained Distributions**
   Distributions with complex constraint structures like log-simplexes and modified standard distributions

**Scientific Distributions**
   Domain-specific distributions commonly used in scientific applications

Key Features
------------

**Multi-Backend Support**
   All custom distributions provide implementations for both PyTorch and SciPy backends, enabling seamless integration across different computational contexts

**Stan Integration**
   Custom Stan function libraries provide efficient MCMC sampling for all custom distributions

**Automatic CDF Generation**
   All distributions automatically receive CDF, CCDF, log-CDF, and log-CCDF methods

**Numerical Stability**
   Implementations emphasize numerical stability through log-space arithmetic and careful handling of extreme values

Available Custom Distributions
------------------------------

**Log-Space Distributions:**

.. code-block:: python

   import scistanpy as ssp
   import numpy as np

   # Exp-Exponential: If exp(Y) ~ Exponential, then Y ~ ExpExponential
   log_lifetime = ssp.parameters.ExpExponential(beta=1.5)

   # Exp-Dirichlet: Log-simplex distribution
   log_probabilities = ssp.parameters.ExpDirichlet(alpha=np.ones(4))

   # Exp-Lomax: Heavy-tailed log-scale distribution
   log_income = ssp.parameters.ExpLomax(lambda_=1.0, alpha=2.0)

**Heavy-Tailed Distributions:**

.. code-block:: python

   # Lomax distribution for wealth, insurance claims
   claim_sizes = ssp.parameters.Lomax(lambda_=1000, alpha=1.5)

   # Power-law tails for extreme events
   earthquake_magnitudes = ssp.parameters.Lomax(lambda_=5.0, alpha=2.0)

**Multinomial Variants:**

.. code-block:: python

   # Multinomial with logit parameterization
   category_counts = ssp.parameters.MultinomialLogit(
       gamma=logit_probabilities,  # Unconstrained
       N=total_trials
   )

   # Multinomial with log-theta parameterization (optimized)
   optimized_counts = ssp.parameters.MultinomialLogTheta(
       log_theta=log_probabilities,
       N=total_trials,
       observable=True  # Enables coefficient optimization
   )

Scientific Applications
-----------------------

**Survival Analysis:**

.. code-block:: python

   # Log-transformed survival times for numerical stability
   log_survival_time = ssp.parameters.ExpExponential(beta=hazard_rate)

   # Heavy-tailed survival for heterogeneous populations
   survival_time = ssp.parameters.Lomax(lambda_=scale, alpha=shape)

**Compositional Data Analysis:**

.. code-block:: python

   # Log-simplex for high-dimensional compositional data
   log_composition = ssp.parameters.ExpDirichlet(
       alpha=concentration_params,
       shape=(n_components,)
   )

   # Robust to numerical issues with very small proportions
   composition = ssp.operations.exp(log_composition)

**Extreme Value Modeling:**

.. code-block:: python

   # Heavy-tailed distributions for extreme events
   extreme_events = ssp.parameters.Lomax(
       lambda_=threshold,
       alpha=tail_index
   )

   # Log-scale modeling for wide dynamic range
   log_extreme_events = ssp.parameters.ExpLomax(
       lambda_=log_threshold,
       alpha=tail_index
   )

**Count Data with Complex Structure:**

.. code-block:: python

   # Multinomial with unconstrained parameterization
   vote_counts = ssp.parameters.MultinomialLogit(
       gamma=candidate_logits,  # Can be any real values
       N=total_voters
   )

   # Efficient for large counts with automatic optimization
   large_counts = ssp.parameters.MultinomialLogTheta(
       log_theta=log_probabilities,
       N=large_sample_size,
       observable=True  # Triggers coefficient precomputation
   )

Implementation Details
----------------------

**PyTorch Integration:**

.. code-block:: python

   # Custom distributions work with PyTorch autodiff
   log_rates = ssp.parameters.Normal(mu=0, sigma=1)

   # Gradients flow through custom distributions
   exp_dist = ssp.parameters.ExpExponential(beta=ssp.operations.exp(log_rates))
   loss = -exp_dist.log_prob(observed_data)
   loss.backward()  # Gradients computed automatically

**Stan Function Integration:**

.. code-block:: python

   # Stan functions automatically included
   model = ssp.Model(exp_exponential_param)
   stan_model = model.to_stan()

   # Generated Stan includes:
   # functions {
   #     #include <expexponential.stanfunctions>
   # }

**Numerical Stability:**

.. code-block:: python

   # Log-space arithmetic prevents overflow/underflow
   very_small_alpha = ssp.constants.Constant(1e-10)
   stable_dirichlet = ssp.parameters.ExpDirichlet(alpha=very_small_alpha)

   # Handles extreme parameter values gracefully
   extreme_lambda = ssp.constants.Constant(1e6)
   stable_lomax = ssp.parameters.ExpLomax(lambda_=extreme_lambda, alpha=0.5)

Performance Considerations
--------------------------

**Efficient Parameterizations:**

.. code-block:: python

   # MultinomialLogTheta with observable=True triggers optimization
   efficient_multinomial = ssp.parameters.MultinomialLogTheta(
       log_theta=log_probs,
       N=large_n,
       observable=True  # Multinomial coefficient precomputed
   )

**Memory-Efficient Sampling:**

.. code-block:: python

   # Batch sampling for memory efficiency
   large_samples = exp_dist.draw(n=100000, batch_size=1000)

**Stan Code Optimization:**

.. code-block:: python

   # Vectorized operations in generated Stan code
   vector_param = ssp.parameters.ExpExponential(beta=rates, shape=(1000,))
   # Generates efficient vectorized Stan sampling

Integration Patterns
--------------------

**With Standard Distributions:**

.. code-block:: python

   # Combine custom and standard distributions
   location = ssp.parameters.Normal(mu=0, sigma=1)
   scale = ssp.parameters.ExpExponential(beta=1.0)  # Log-scale parameter

   # Use in hierarchical models
   observations = ssp.parameters.Normal(
       mu=location,
       sigma=ssp.operations.exp(scale)
   )

**In Complex Models:**

.. code-block:: python

   # Multi-level model with custom distributions
   class RobustHierarchicalModel(ssp.Model):
       def __init__(self, group_data):
           super().__init__()

           # Heavy-tailed group effects
           self.group_effects = ssp.parameters.Lomax(
               lambda_=1.0, alpha=1.5, shape=(len(group_data),)
           )

           # Log-simplex mixing proportions
           self.mixing_props = ssp.parameters.ExpDirichlet(
               alpha=np.ones(3)
           )

**Custom Distribution Development:**

.. code-block:: python

   # Framework for creating new custom distributions
   from scistanpy.model.components.custom_distributions import CustomDistribution

   class MyCustomDistribution(CustomDistribution):
       """Template for custom distribution development."""

       def __init__(self, param1, param2, **kwargs):
           # Parameter validation and setup
           super().__init__(param1=param1, param2=param2, **kwargs)

       def _create_torch_distribution(self, param1, param2):
           # PyTorch distribution implementation
           return CustomTorchDistribution(param1, param2)

       def get_supporting_functions(self):
           # Stan function includes
           return ["#include <mycustom.stanfunctions>"]

Best Practices
--------------

1. **Choose appropriate parameterizations** for your modeling context
2. **Use log-space distributions** for improved numerical stability
3. **Leverage automatic optimizations** like MultinomialLogTheta
4. **Consider heavy-tailed distributions** for robust modeling
5. **Test custom distributions** thoroughly before use in models
6. **Monitor numerical stability** with extreme parameter values
7. **Use vectorized operations** for computational efficiency
8. **Document custom distributions** with clear mathematical definitions

The custom distributions framework extends SciStanPy's modeling capabilities while maintaining consistency with the core parameter system and multi-backend computational support.
