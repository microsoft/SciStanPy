Custom PyTorch Distributions API Reference
==========================================

This reference covers the custom PyTorch-based probability distributions in SciStanPy.

Custom Torch Distributions Module
---------------------------------

.. automodule:: scistanpy.model.components.custom_distributions.custom_torch_dists

   :members:
   :undoc-members:
   :show-inheritance:

Distribution Architecture
-------------------------

Base Classes
~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.custom_distributions.custom_torch_dists.CustomDistribution

   :members:
   :undoc-members:
   :show-inheritance:

   **Foundation for Custom Distributions:**

   The CustomDistribution class provides the PyTorch backend for all custom probability distributions:

   .. code-block:: python

      import torch
      import torch.distributions as dist

      # Custom distributions inherit from PyTorch Distribution
      custom_dist = MyCustomDistribution(param1=1.0, param2=2.0)

      # Full PyTorch integration
      samples = custom_dist.sample((1000,))
      log_probs = custom_dist.log_prob(samples)

      # Automatic differentiation support
      params = torch.tensor([1.0, 2.0], requires_grad=True)
      loss = -custom_dist.log_prob(data).sum()
      loss.backward()  # Gradients computed automatically

Log-Transformed Distributions
-----------------------------

ExpExponential Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.custom_distributions.custom_torch_dists.ExpExponential

   :members:
   :undoc-members:
   :show-inheritance:

   **Log-Exponential Distribution:**

   If X ~ Exponential(β), then Y = log(X) ~ ExpExponential(β):

   .. code-block:: python

      # Standard exponential vs log-exponential
      beta = torch.tensor(2.0)

      # Standard exponential (always positive)
      exp_dist = dist.Exponential(beta)
      exp_samples = exp_dist.sample((1000,))  # All positive

      # Log-exponential (can be negative)
      log_exp_dist = ssp.ExpExponential(beta)
      log_samples = log_exp_dist.sample((1000,))  # Can be negative

      # Relationship: exp(log_samples) should follow Exponential(beta)
      recovered = torch.exp(log_samples)

   **Mathematical Properties:**

   - **Support**: Real line (-∞, ∞)
   - **PDF**: f(y|β) = β exp(y - β exp(y))
   - **Mean**: -γ - log(β) where γ is Euler's constant
   - **Applications**: Log-transformed waiting times, multiplicative processes

ExpDirichlet Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.custom_distributions.custom_torch_dists.ExpDirichlet

   :members:
   :undoc-members:
   :show-inheritance:

   **Log-Simplex Distribution:**

   If X ~ Dirichlet(α), then Y = log(X) ~ ExpDirichlet(α):

   .. code-block:: python

      # High-dimensional log-simplex
      alpha = torch.ones(1000)  # 1000-dimensional simplex

      # Standard Dirichlet (numerical issues with small values)
      dirichlet = dist.Dirichlet(alpha)
      simplex_samples = dirichlet.sample((100,))

      # Exp-Dirichlet (numerically stable)
      exp_dirichlet = ssp.ExpDirichlet(alpha)
      log_simplex_samples = exp_dirichlet.sample((100,))

      # Verify simplex constraint: exp(log_simplex).sum(dim=-1) ≈ 1
      recovered_simplex = torch.exp(log_simplex_samples)
      assert torch.allclose(recovered_simplex.sum(dim=-1), torch.tensor(1.0))

   **Numerical Advantages:**

   - **Stability**: Avoids underflow with very small probabilities
   - **High-dimensional**: Handles thousands of categories efficiently
   - **Constraint**: log(exp(Y).sum()) = 0 (log-simplex constraint)

ExpLomax Distribution
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.custom_distributions.custom_torch_dists.ExpLomax

   :members:
   :undoc-members:
   :show-inheritance:

   **Log-Heavy-Tailed Distribution:**

   If X ~ Lomax(λ, α), then Y = log(X) ~ ExpLomax(λ, α):

   .. code-block:: python

      # Heavy-tailed phenomena on log scale
      lambda_param = torch.tensor(1.0)
      alpha = torch.tensor(1.5)  # Shape parameter

      # Standard Lomax (heavy right tail)
      lomax = ssp.Lomax(lambda_param, alpha)
      lomax_samples = lomax.sample((1000,))

      # Log-Lomax (heavy tails on log scale)
      exp_lomax = ssp.ExpLomax(lambda_param, alpha)
      log_samples = exp_lomax.sample((1000,))

      # Applications: Log-income, log-wealth, log-city-sizes
      log_wealth = exp_lomax.sample((10000,))

   **Applications:**

   - **Economics**: Log-wealth, log-income distributions
   - **Urban studies**: Log-city sizes (Zipf's law)
   - **Internet**: Log-file sizes, log-connection times

Heavy-Tailed Distributions
--------------------------

Lomax Distribution
~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.custom_distributions.custom_torch_dists.Lomax

   :members:
   :undoc-members:
   :show-inheritance:

   **Power-Law Tail Distribution:**

   The Lomax distribution (Pareto Type II) has polynomial tails:

   .. code-block:: python

      # Insurance claims with heavy tails
      lambda_param = torch.tensor(1000.0)  # Scale parameter
      alpha = torch.tensor(1.2)            # Shape (tail index)

      lomax_dist = ssp.Lomax(lambda_param, alpha)

      # Generate insurance claims
      claims = lomax_dist.sample((10000,))

      # Heavy tail: P(X > x) ∼ x^(-α) for large x
      # Lower α → heavier tails
      very_heavy = ssp.Lomax(lambda_param, alpha=0.8)  # Very heavy tails
      moderate = ssp.Lomax(lambda_param, alpha=2.0)    # Moderate tails

   **Tail Behavior:**

   .. code-block:: python

      # Tail probability comparison
      x_large = torch.tensor(10000.0)

      # Lomax: P(X > x) ∼ (λ/(λ+x))^α
      lomax_tail = lomax_dist.ccdf(x_large)

      # Compare to exponential tail: P(X > x) ∼ exp(-x/β)
      exp_tail = dist.Exponential(1/1000.0).ccdf(x_large)

      print(f"Lomax tail probability: {lomax_tail}")
      print(f"Exponential tail probability: {exp_tail}")
      # Lomax will be much larger for extreme values

**Mathematical Properties:**

- **Support**: [0, ∞)
- **PDF**: f(x|λ,α) = α λ^α / (λ + x)^(α+1)
- **Mean**: λ/(α-1) for α > 1
- **Variance**: λ² α / ((α-1)² (α-2)) for α > 2
- **Tail index**: α (lower values = heavier tails)

Multinomial Variants
--------------------

MultinomialLogit Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.custom_distributions.custom_torch_dists.MultinomialLogit

   :members:
   :undoc-members:
   :show-inheritance:

   **Unconstrained Multinomial Parameterization:**

   .. code-block:: python

      # Unconstrained parameterization for optimization
      n_categories = 5
      total_trials = 100

      # Logits can be any real values
      logits = torch.randn(n_categories)  # Unconstrained

      # Multinomial with logit parameterization
      multinomial_logit = ssp.MultinomialLogit(gamma=logits, N=total_trials)

      # Sample category counts
      counts = multinomial_logit.sample()
      assert counts.sum() == total_trials

      # Convert to probabilities
      probabilities = torch.softmax(logits, dim=-1)

   **Optimization Advantages:**

   .. code-block:: python

      # Optimization-friendly parameterization
      class MulticlassModel(torch.nn.Module):
          def __init__(self, n_features, n_classes):
              super().__init__()
              self.linear = torch.nn.Linear(n_features, n_classes)

          def forward(self, x, n_trials):
              logits = self.linear(x)
              return ssp.MultinomialLogit(gamma=logits, N=n_trials)

      # No constraints on logits during optimization
      model = MulticlassModel(10, 5)
      optimizer = torch.optim.Adam(model.parameters())

MultinomialLogTheta Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.custom_distributions.custom_torch_dists.MultinomialLogTheta

   :members:
   :undoc-members:
   :show-inheritance:

   **Log-Probability Parameterized with Optimization:**

   .. code-block:: python

      # Log-simplex parameterization
      log_probabilities = torch.log(torch.tensor([0.2, 0.3, 0.1, 0.4]))
      total_trials = 1000

      # Efficient for large N with automatic coefficient handling
      multinomial_log = ssp.MultinomialLogTheta(
          log_theta=log_probabilities,
          N=total_trials
      )

      # For observed data, coefficient is precomputed
      observed_counts = torch.tensor([180, 320, 95, 405])
      log_prob = multinomial_log.log_prob(observed_counts)

   **Stan Integration Benefits:**

   .. code-block:: python

      # When used as observable in SciStanPy models
      counts = ssp.parameters.MultinomialLogTheta(
          log_theta=log_probs,
          N=total_n,
          observable=True  # Triggers Stan optimization
      )

      # Generated Stan code precomputes multinomial coefficient
      # in transformed data block for efficiency

PyTorch Integration Features
----------------------------

**Automatic Differentiation:**

.. code-block:: python

   # Gradients flow through all custom distributions
   params = torch.tensor([1.0, 2.0], requires_grad=True)

   custom_dist = ssp.ExpExponential(beta=params[0])
   data = torch.randn(100)

   # Likelihood with gradients
   log_likelihood = custom_dist.log_prob(data).sum()
   log_likelihood.backward()

   print(f"Gradient w.r.t. beta: {params.grad[0]}")

**Vectorization Support:**

.. code-block:: python

   # Batch processing with vectorized operations
   batch_size = 1000
   n_params = 50

   # Vectorized parameters
   betas = torch.rand(batch_size, n_params)

   # Vectorized distribution
   vectorized_dist = ssp.ExpExponential(beta=betas)

   # Vectorized sampling and evaluation
   samples = vectorized_dist.sample()  # Shape: (batch_size, n_params)
   log_probs = vectorized_dist.log_prob(samples)

**GPU Acceleration:**

.. code-block:: python

   # GPU computation for large-scale problems
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # Move parameters to GPU
   beta_gpu = torch.tensor(2.0, device=device)

   # GPU-accelerated distribution
   gpu_dist = ssp.ExpExponential(beta=beta_gpu)

   # GPU sampling and computation
   gpu_samples = gpu_dist.sample((100000,))  # Fast GPU sampling
   gpu_log_probs = gpu_dist.log_prob(gpu_samples)

Numerical Stability Features
----------------------------

**Log-Space Arithmetic:**

.. code-block:: python

   # Numerically stable implementations

   # ExpDirichlet uses log-sum-exp for stability
   large_alpha = torch.tensor([1e6, 1e6, 1e6])
   stable_exp_dir = ssp.ExpDirichlet(alpha=large_alpha)

   # ExpLomax handles extreme scale parameters
   huge_lambda = torch.tensor(1e10)
   stable_exp_lomax = ssp.ExpLomax(lambda_=huge_lambda, alpha=1.5)

**Gradient Stability:**

.. code-block:: python

   # Stable gradients even with extreme parameters
   extreme_beta = torch.tensor(1e-8, requires_grad=True)

   exp_exp_dist = ssp.ExpExponential(beta=extreme_beta)
   data = torch.tensor([0.0])

   log_prob = exp_exp_dist.log_prob(data)
   log_prob.backward()  # Stable gradients

Performance Optimization
------------------------

**Efficient Sampling:**

.. code-block:: python

   # Optimized sampling algorithms
   dist = ssp.ExpExponential(beta=1.0)

   # Uses inverse transform sampling for efficiency
   samples = dist.sample((10000,))

**Memory-Efficient Operations:**

.. code-block:: python

   # In-place operations where possible
   large_dist = ssp.Lomax(lambda_=1.0, alpha=1.5)

   # Memory-efficient evaluation
   large_data = torch.randn(1000000)
   log_probs = large_dist.log_prob(large_data)  # Efficient computation

Best Practices
--------------

1. **Use appropriate distributions** for your data characteristics
2. **Leverage log-space parameterizations** for numerical stability
3. **Take advantage of vectorization** for batch processing
4. **Monitor gradient flow** in optimization problems
5. **Use GPU acceleration** for large-scale computations
6. **Test numerical stability** with extreme parameter values
7. **Profile performance** for computational bottlenecks

The PyTorch custom distributions provide the computational foundation for advanced probabilistic modeling while maintaining full integration with PyTorch's automatic differentiation and GPU acceleration capabilities.
