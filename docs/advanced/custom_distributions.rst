Creating Custom Distributions
=============================

This guide covers how to create custom probability distributions for specialized scientific applications not covered by the standard distribution library.

When to Create Custom Distributions
-----------------------------------

Consider creating custom distributions when:

**Domain-Specific Requirements**
- Your field uses specialized distributions not in standard libraries
- You need distributions with domain-specific parameterizations
- Existing distributions don't capture your scientific understanding

**Performance Optimization**
- You need optimized implementations for specific use cases
- Standard implementations lack numerical stability for your parameter ranges
- You require custom sampling algorithms

**Research Innovation**
- You're developing new statistical methods
- Your research involves novel probability models
- You need to validate theoretical distributions

Types of Custom Distributions
-----------------------------

SciStanPy supports several approaches to creating custom distributions:

**Transformed Distributions**
Create new distributions by transforming existing ones:

.. code-block:: python

   from scistanpy.model.components.transformations.transformed_parameters import UnaryTransformedParameter

   class LogitNormalDistribution(UnaryTransformedParameter):
       """Normal distribution transformed by logit function."""

       def __init__(self, mu, sigma):
           # Base normal distribution
           base_normal = ssp.parameters.Normal(mu=mu, sigma=sigma)

           # Apply logit transformation
           super().__init__(dist1=base_normal)

       def run_np_torch_op(self, dist1):
           # Apply logit: log(x / (1 - x))
           return torch.log(dist1 / (1 - dist1))

       def write_stan_operation(self, dist1: str) -> str:
           return f"logit({dist1})"

**Custom PyTorch Distributions**
Implement new distributions in PyTorch framework:

.. code-block:: python

   import torch
   import torch.distributions as torch_dist
   from scistanpy.model.components.custom_distributions.custom_torch_dists import CustomDistribution

   class WeibullDistribution(torch_dist.Distribution, CustomDistribution):
       """Custom Weibull distribution for reliability analysis."""

       def __init__(self, scale, concentration, validate_args=None):
           self.scale = torch.as_tensor(scale)
           self.concentration = torch.as_tensor(concentration)

           batch_shape = torch.broadcast_shapes(self.scale.shape, self.concentration.shape)
           super().__init__(batch_shape, validate_args=validate_args)

       def log_prob(self, value):
           """Compute log probability density."""
           k = self.concentration
           lambda_param = self.scale

           return (torch.log(k) - torch.log(lambda_param) +
                   (k - 1) * (torch.log(value) - torch.log(lambda_param)) -
                   torch.pow(value / lambda_param, k))

       def sample(self, sample_shape=torch.Size()):
           """Generate samples using inverse transform sampling."""
           # Generate uniform samples
           uniform_samples = torch.rand(sample_shape + self._batch_shape)

           # Apply inverse CDF: F^(-1)(u) = λ * (-ln(1-u))^(1/k)
           return self.scale * torch.pow(-torch.log(1 - uniform_samples), 1 / self.concentration)

**Custom SciPy Distributions**
Extend SciPy distributions for additional functionality:

.. code-block:: python

   import numpy as np
   from scipy import stats
   from scistanpy.model.components.custom_distributions.custom_scipy_dists import TransformedScipyDist

   class TruncatedNormal(TransformedScipyDist):
       """Truncated normal distribution."""

       def __init__(self, mu, sigma, lower=-np.inf, upper=np.inf):
           # Use SciPy's truncated normal
           base_dist = stats.truncnorm(
               (lower - mu) / sigma,
               (upper - mu) / sigma,
               loc=mu, scale=sigma
           )
           super().__init__(base_dist)

           self.mu = mu
           self.sigma = sigma
           self.lower = lower
           self.upper = upper

       def transform(self, x):
           """Identity transformation (already truncated)."""
           return x

       def inverse_transform(self, x):
           """Identity transformation."""
           return x

       def log_jacobian_correction(self, x):
           """No transformation, so Jacobian is 1."""
           return np.zeros_like(x)

**Parameter Class Integration**
Integrate custom distributions with SciStanPy parameter system:

.. code-block:: python

   from scistanpy.model.components.parameters import ContinuousDistribution

   class WeibullParameter(ContinuousDistribution):
       """Weibull distribution parameter for SciStanPy models."""

       # Distribution mapping
       SCIPY_DIST = stats.weibull_min
       TORCH_DIST = WeibullDistribution  # Our custom implementation
       STAN_DIST = "weibull"

       # Parameter name mappings
       STAN_TO_SCIPY_NAMES = {"alpha": "c", "sigma": "scale"}
       STAN_TO_TORCH_NAMES = {"alpha": "concentration", "sigma": "scale"}

       def __init__(self, alpha, sigma, **kwargs):
           """Initialize Weibull distribution.

           :param alpha: Shape parameter (concentration)
           :param sigma: Scale parameter
           """
           super().__init__(alpha=alpha, sigma=sigma, **kwargs)

Scientific Domain Examples
--------------------------

Astronomy: Schechter Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Schechter function is commonly used in astronomy for galaxy luminosity functions:

.. code-block:: python

   class SchechterFunction(ContinuousDistribution):
       """Schechter function for galaxy luminosity modeling."""

       def __init__(self, phi_star, M_star, alpha, **kwargs):
           """Initialize Schechter function.

           :param phi_star: Normalization parameter
           :param M_star: Characteristic magnitude
           :param alpha: Faint-end slope
           """
           super().__init__(phi_star=phi_star, M_star=M_star, alpha=alpha, **kwargs)

       def log_prob(self, M):
           """Log probability density function."""
           phi_star = self.phi_star
           M_star = self.M_star
           alpha = self.alpha

           # Schechter function: φ(M) = φ* * ln(10)/2.5 * 10^(0.4*(α+1)*(M*-M)) * exp(-10^(0.4*(M*-M)))
           L_ratio = torch.pow(10, 0.4 * (M_star - M))

           return (torch.log(phi_star) + torch.log(torch.log(torch.tensor(10.0)) / 2.5) +
                   (alpha + 1) * torch.log(L_ratio) - L_ratio)

       def sample(self, sample_shape=torch.Size()):
           """Sample from Schechter function using rejection sampling."""
           # Implement rejection sampling algorithm
           # This is a simplified version - full implementation would be more robust
           pass

Chemistry: Dose-Response Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hill equation for dose-response relationships:

.. code-block:: python

   class HillEquationParameter(ContinuousDistribution):
       """Hill equation for dose-response modeling in pharmacology."""

       def __init__(self, EC50, hill_coefficient, baseline, max_response, **kwargs):
           """Initialize Hill equation parameters.

           :param EC50: Concentration producing 50% of maximum response
           :param hill_coefficient: Hill coefficient (cooperativity)
           :param baseline: Baseline response level
           :param max_response: Maximum response level
           """
           super().__init__(
               EC50=EC50,
               hill_coefficient=hill_coefficient,
               baseline=baseline,
               max_response=max_response,
               **kwargs
           )

       def mean_response(self, concentration):
           """Calculate mean response at given concentration."""
           EC50 = self.EC50
           n = self.hill_coefficient
           baseline = self.baseline
           max_resp = self.max_response

           # Hill equation: Response = baseline + (max - baseline) * [C]^n / (EC50^n + [C]^n)
           conc_n = torch.pow(concentration, n)
           EC50_n = torch.pow(EC50, n)

           return baseline + (max_resp - baseline) * conc_n / (EC50_n + conc_n)

Biology: Michaelis-Menten Kinetics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Custom distribution for enzyme kinetics:

.. code-block:: python

   class MichaelisMentenResponse(ContinuousDistribution):
       """Michaelis-Menten kinetics with measurement noise."""

       def __init__(self, V_max, K_m, substrate_conc, sigma, **kwargs):
           """Initialize Michaelis-Menten model.

           :param V_max: Maximum velocity
           :param K_m: Michaelis constant
           :param substrate_conc: Substrate concentrations (constant)
           :param sigma: Measurement noise standard deviation
           """
           # Store substrate concentrations as constant
           self.substrate_conc = substrate_conc

           super().__init__(V_max=V_max, K_m=K_m, sigma=sigma, **kwargs)

       def mean_velocity(self):
           """Calculate mean velocity using Michaelis-Menten equation."""
           V_max = self.V_max
           K_m = self.K_m
           S = self.substrate_conc

           return (V_max * S) / (K_m + S)

       def log_prob(self, observed_velocity):
           """Log probability of observed velocities."""
           predicted_velocity = self.mean_velocity()
           sigma = self.sigma

           # Normal likelihood around predicted velocity
           return torch.distributions.Normal(predicted_velocity, sigma).log_prob(observed_velocity)

Physics: Custom Detector Response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detector response function for experimental physics:

.. code-block:: python

   class DetectorResponse(ContinuousDistribution):
       """Custom detector response function with resolution effects."""

       def __init__(self, true_energy, resolution_sigma, efficiency, **kwargs):
           """Initialize detector response model.

           :param true_energy: True particle energy
           :param resolution_sigma: Energy resolution (Gaussian broadening)
           :param efficiency: Detection efficiency
           """
           super().__init__(
               true_energy=true_energy,
               resolution_sigma=resolution_sigma,
               efficiency=efficiency,
               **kwargs
           )

       def log_prob(self, measured_energy):
           """Log probability of measured energy given true energy."""
           # Gaussian broadening due to detector resolution
           broadened = torch.distributions.Normal(
               self.true_energy,
               self.resolution_sigma
           )

           # Account for detection efficiency
           detection_prob = torch.distributions.Bernoulli(self.efficiency)

           # Combined probability
           return broadened.log_prob(measured_energy) + detection_prob.log_prob(torch.ones_like(measured_energy))

Implementation Best Practices
----------------------------

Numerical Stability
~~~~~~~~~~~~~~~~~~

**Use Log-Space Computations:**

.. code-block:: python

   def log_prob(self, value):
       """Always implement log_prob for numerical stability."""
       # Good: Compute in log space
       log_normalizer = torch.logsumexp(log_unnormalized_probs, dim=-1)
       return log_unnormalized_probs - log_normalizer

       # Avoid: Computing probabilities then taking log
       # probs = torch.softmax(logits, dim=-1)
       # return torch.log(probs)  # Can lead to log(0)

**Handle Edge Cases:**

.. code-block:: python

   def log_prob(self, value):
       """Handle boundary conditions and invalid inputs."""
       # Check support constraints
       if torch.any(value < 0):
           return torch.full_like(value, -torch.inf)

       # Handle zero values for log-scale distributions
       value = torch.clamp(value, min=1e-8)  # Avoid log(0)

       return self._compute_log_prob(value)

**Validate Parameters:**

.. code-block:: python

   def __init__(self, scale, concentration, validate_args=None):
       """Validate parameters during initialization."""
       if validate_args is None:
           validate_args = torch.is_grad_enabled()

       if validate_args:
           if torch.any(scale <= 0):
               raise ValueError("Scale parameter must be positive")
           if torch.any(concentration <= 0):
               raise ValueError("Concentration parameter must be positive")

       self.scale = scale
       self.concentration = concentration

Stan Code Generation
~~~~~~~~~~~~~~~~~~~

For integration with Stan backend, implement Stan code generation:

.. code-block:: python

   class CustomWeibull(ContinuousDistribution):
       STAN_DIST = "weibull"  # Stan function name

       def get_stan_declaration(self) -> str:
           """Generate Stan parameter declaration."""
           return f"real<lower=0> {self.name};"

       def get_stan_prior_code(self) -> str:
           """Generate Stan prior specification."""
           alpha_code = self._get_stan_expression(self.alpha)
           sigma_code = self._get_stan_expression(self.sigma)
           return f"{self.name} ~ weibull({alpha_code}, {sigma_code});"

Testing Custom Distributions
----------------------------

Comprehensive Testing Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Unit Tests for Basic Functionality:**

.. code-block:: python

   import pytest
   import torch

   class TestWeibullDistribution:
       def test_initialization(self):
           """Test parameter validation and initialization."""
           dist = WeibullDistribution(scale=1.0, concentration=2.0)
           assert dist.scale == 1.0
           assert dist.concentration == 2.0

       def test_parameter_validation(self):
           """Test error handling for invalid parameters."""
           with pytest.raises(ValueError):
               WeibullDistribution(scale=-1.0, concentration=2.0)  # Negative scale

       def test_sampling(self):
           """Test that sampling produces reasonable results."""
           dist = WeibullDistribution(scale=1.0, concentration=2.0)
           samples = dist.sample((1000,))

           # Basic sanity checks
           assert samples.shape == (1000,)
           assert torch.all(samples >= 0)  # Weibull is positive
           assert torch.isfinite(samples).all()

**Statistical Validation:**

.. code-block:: python

   def test_against_scipy(self):
       """Validate against SciPy implementation."""
       from scipy.stats import weibull_min

       # Parameters
       scale, concentration = 2.0, 1.5

       # Our implementation
       custom_dist = WeibullDistribution(scale=scale, concentration=concentration)

       # SciPy equivalent
       scipy_dist = weibull_min(c=concentration, scale=scale)

       # Test points
       test_values = torch.tensor([0.5, 1.0, 2.0, 5.0])

       # Compare log probabilities
       custom_logprob = custom_dist.log_prob(test_values)
       scipy_logprob = torch.tensor(scipy_dist.logpdf(test_values.numpy()))

       torch.testing.assert_close(custom_logprob, scipy_logprob, rtol=1e-6)

**Monte Carlo Validation:**

.. code-block:: python

   def test_moments_via_sampling(self):
       """Test theoretical moments against sample moments."""
       scale, concentration = 2.0, 1.5
       dist = WeibullDistribution(scale=scale, concentration=concentration)

       # Generate large sample
       samples = dist.sample((100000,))

       # Theoretical mean for Weibull: λ * Γ(1 + 1/k)
       theoretical_mean = scale * torch.exp(torch.lgamma(1 + 1/concentration))
       sample_mean = samples.mean()

       # Should be close (within 3 standard errors)
       std_error = samples.std() / torch.sqrt(torch.tensor(100000.0))
       assert torch.abs(sample_mean - theoretical_mean) < 3 * std_error

Performance Optimization
------------------------

Vectorization
~~~~~~~~~~~~

Design distributions to handle batch operations efficiently:

.. code-block:: python

   def log_prob(self, value):
       """Vectorized log probability computation."""
       # Ensure all operations are vectorized
       scale = self.scale.unsqueeze(-1)  # Broadcasting
       concentration = self.concentration.unsqueeze(-1)

       # Compute log probability for all values simultaneously
       log_prob_values = (torch.log(concentration) - torch.log(scale) +
                         (concentration - 1) * (torch.log(value) - torch.log(scale)) -
                         torch.pow(value / scale, concentration))

       return log_prob_values

GPU Compatibility
~~~~~~~~~~~~~~~~

Ensure distributions work on GPU:

.. code-block:: python

   def sample(self, sample_shape=torch.Size()):
       """GPU-compatible sampling."""
       # Ensure device consistency
       device = self.scale.device
       dtype = self.scale.dtype

       # Generate samples on correct device
       uniform_samples = torch.rand(
           sample_shape + self._batch_shape,
           device=device,
           dtype=dtype
       )

       return self._inverse_cdf(uniform_samples)

Documentation and Examples
--------------------------

Always provide comprehensive documentation:

.. code-block:: python

   class WeibullDistribution(torch.distributions.Distribution):
       """Weibull distribution for reliability and survival analysis.

       The Weibull distribution is widely used in reliability engineering
       and survival analysis. It can model various failure rates depending
       on the shape parameter.

       :param scale: Scale parameter λ > 0
       :type scale: torch.Tensor
       :param concentration: Shape parameter k > 0
       :type concentration: torch.Tensor

       **Mathematical Definition:**

       The probability density function is:

       .. math::

          f(x; λ, k) = \\frac{k}{λ} \\left(\\frac{x}{λ}\\right)^{k-1}
          \\exp\\left(-\\left(\\frac{x}{λ}\\right)^k\\right)

       **Properties:**

       - **Mean**: λΓ(1 + 1/k)
       - **Variance**: λ²[Γ(1 + 2/k) - Γ²(1 + 1/k)]
       - **Support**: [0, ∞)

       **Use Cases:**

       - Equipment failure analysis
       - Material fatigue modeling
       - Survival time analysis
       - Wind speed modeling

       **Example:**

       .. code-block:: python

          # Model component lifetimes
          lifetime = WeibullDistribution(scale=1000, concentration=2)

          # Survival analysis
          survival_time = WeibullDistribution(scale=treatment_effect, concentration=shape)
       """

Advanced Custom Distributions
=============================

This guide covers advanced techniques for creating and using custom probability distributions in SciStanPy.

Creating Custom Distributions
-----------------------------

**Extending PyTorch Distributions:**

.. code-block:: python

   from scistanpy.model.components.custom_distributions.custom_torch_dists import CustomDistribution
   import torch

   class WeibullDistribution(CustomDistribution):
       """Weibull distribution for reliability analysis."""

       def __init__(self, shape, scale):
           self.shape = shape
           self.scale = scale
           super().__init__()

       def log_prob(self, value):
           # Weibull log-probability implementation
           return (torch.log(self.shape) - torch.log(self.scale) +
                   (self.shape - 1) * torch.log(value / self.scale) -
                   (value / self.scale) ** self.shape)

       def sample(self, sample_shape=torch.Size()):
           # Inverse transform sampling
           u = torch.rand(sample_shape)
           return self.scale * (-torch.log(1 - u)) ** (1 / self.shape)

**Domain-Specific Distributions:**

.. code-block:: python

   class PhylogeneticDistribution(CustomDistribution):
       """Distribution for phylogenetic tree branch lengths."""

       def __init__(self, tree_structure, rate_matrix):
           self.tree_structure = tree_structure
           self.rate_matrix = rate_matrix
           super().__init__()

       def log_prob(self, branch_lengths):
           # Phylogenetic likelihood computation
           return self.compute_phylogenetic_likelihood(
               branch_lengths, self.tree_structure, self.rate_matrix
           )

Advanced Distribution Features
-----------------------------

**Mixture Distributions:**

.. code-block:: python

   class GaussianMixture(CustomDistribution):
       """Gaussian mixture distribution."""

       def __init__(self, weights, means, stds):
           self.weights = weights
           self.means = means
           self.stds = stds
           super().__init__()

       def log_prob(self, value):
           # Mixture log-probability
           component_log_probs = torch.stack([
               torch.distributions.Normal(mean, std).log_prob(value)
               for mean, std in zip(self.means, self.stds)
           ])

           weighted_log_probs = component_log_probs + torch.log(self.weights)
           return torch.logsumexp(weighted_log_probs, dim=0)

**Truncated Distributions:**

.. code-block:: python

   class TruncatedNormal(CustomDistribution):
       """Normal distribution truncated to specified bounds."""

       def __init__(self, mu, sigma, lower=None, upper=None):
           self.mu = mu
           self.sigma = sigma
           self.lower = lower
           self.upper = upper
           super().__init__()

       def log_prob(self, value):
           # Truncated normal log-probability
           base_log_prob = torch.distributions.Normal(self.mu, self.sigma).log_prob(value)

           # Normalization constant for truncation
           normalizer = self.compute_truncation_normalizer()

           return base_log_prob - normalizer

Integration Patterns
-------------------

**Using Custom Distributions in Models:**

.. code-block:: python

   # Custom distribution in SciStanPy model
   reliability_param = WeibullDistribution(shape=2.0, scale=1000.0)

   # Use in likelihood
   likelihood = ssp.parameters.Normal(
       mu=reliability_param,
       sigma=measurement_error
   )
   likelihood.observe(failure_times)

**Stan Code Generation:**

.. code-block:: python

   class CustomDistributionWithStan(CustomDistribution):
       """Custom distribution with Stan code generation."""

       def write_stan_code(self):
           """Generate Stan code for this distribution."""
           return f"""
           real custom_lpdf(real x, real param1, real param2) {{
               // Stan implementation of log-probability
               return log_prob_expression;
           }}
           """
