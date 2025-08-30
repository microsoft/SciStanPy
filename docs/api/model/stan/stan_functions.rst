Stan Functions Library Reference
=================================

This reference covers the Stan function libraries that provide specialized probability distributions and mathematical operations for SciStanPy models.

Stan Functions Overview
-----------------------

SciStanPy includes a comprehensive library of Stan functions that extend the standard Stan library with specialized distributions and operations commonly needed in scientific modeling. These functions are automatically included in generated Stan programs based on the model components used.

Function Library Organization
-----------------------------

The Stan functions are organized into several specialized libraries:

**Multinomial Distributions**
   Enhanced multinomial distributions with multiple parameterizations

**Exp-Transformed Distributions**
   Log-transformed distributions for improved numerical stability

**Growth Models**
   Specialized functions for temporal and population dynamics

**Sequence Operations**
   Convolution and pattern matching for sequence analysis

Multinomial Function Library
----------------------------

**File**: ``multinomial.stanfunctions``

**Functions Provided:**

.. code-block:: stan

   // Log multinomial coefficient
   real log_multinomial_coeff(array[] int n)

   // Multinomial with log-theta parameterization
   real multinomial_logtheta_unnorm_lpmf(array[] int n, vector log_theta)
   real multinomial_logtheta_norm_lpmf(array[] int n, vector log_theta)
   real multinomial_logtheta_manual_norm_lpmf(array[] int n, vector log_theta, real coeff)
   array[] int multinomial_logtheta_rng(vector log_theta, int N)

**Usage in SciStanPy:**

.. code-block:: python

   # Automatically included when using MultinomialLogit
   counts = ssp.parameters.MultinomialLogit(
       n=100,
       logits=log_probabilities,
       observable=True
   )

**Mathematical Background:**

The multinomial distribution with log-theta parameterization uses:

.. math::

   p(n | \log\theta) = \binom{N}{n_1, \ldots, n_k} \prod_{i=1}^k \theta_i^{n_i}

where :math:`\theta_i = \exp(\log\theta_i)` and :math:`N = \sum_i n_i`.

Exp-Exponential Distribution
---------------------------

**File**: ``expexponential.stanfunctions``

**Functions Provided:**

.. code-block:: stan

   // Exp-exponential log probability density
   real expexponential_lpdf(real y, real beta)
   real expexponential_lpdf(vector y, vector beta)
   real expexponential_lpdf(vector y, real beta)

   // Random number generation
   real expexponential_rng(real beta)
   array[] real expexponential_rng(vector beta)

**Usage in SciStanPy:**

.. code-block:: python

   # For log-transformed exponential variables
   log_lifetime = ssp.parameters.ExpExponential(rate=0.1, observable=True)

**Mathematical Background:**

If :math:`X \sim \text{Exponential}(\beta)`, then :math:`Y = \log(X)` follows the exp-exponential distribution:

.. math::

   p(y | \beta) = \beta \exp(y - \beta e^y)

Exp-Dirichlet Distribution
-------------------------

**File**: ``expdirichlet.stanfunctions``

**Functions Provided:**

.. code-block:: stan

   // Beta function utilities
   real vectorized_lbeta(vector alpha)
   vector inv_ilr_log_simplex_constrain_jacobian(vector y)

   // Exp-Dirichlet log probability density
   real expdirichlet_unnorm_lpdf(vector y, vector alpha)
   real expdirichlet_norm_lpdf(vector y, vector alpha)

   // Random number generation
   vector expdirichlet_rng(vector alpha)

**Usage in SciStanPy:**

.. code-block:: python

   # For log-simplex variables
   log_proportions = ssp.parameters.ExpDirichlet(
       alpha=[2.0, 3.0, 1.0],
       observable=True
   )

**Mathematical Background:**

If :math:`X \sim \text{Dirichlet}(\alpha)`, then :math:`Y = \log(X)` follows the exp-Dirichlet distribution:

.. math::

   p(y | \alpha) = \frac{\Gamma(\sum_i \alpha_i)}{\prod_i \Gamma(\alpha_i)} \exp\left(\sum_i (\alpha_i - 1) y_i\right)

Exp-Lomax Distribution
---------------------

**File**: ``explomax.stanfunctions``

**Functions Provided:**

.. code-block:: stan

   // Exp-Lomax log probability density (multiple overloads)
   real explomax_lpdf(real y, real lambda, real alpha)
   real explomax_lpdf(vector y, vector lambda, vector alpha)
   real explomax_lpdf(vector y, real lambda, real alpha)
   real explomax_lpdf(vector y, real lambda, vector alpha)
   real explomax_lpdf(vector y, vector lambda, real alpha)

   // Random number generation (multiple overloads)
   real explomax_rng(real lambda, real alpha)
   array[] real explomax_rng(vector lambda, real alpha)
   array[] real explomax_rng(real lambda, vector alpha)
   array[] real explomax_rng(vector lambda, vector alpha)

**Usage in SciStanPy:**

.. code-block:: python

   # For log-transformed Lomax variables (heavy-tailed distributions)
   log_size = ssp.parameters.ExpLomax(
       lambda_=1.0,
       alpha=2.0,
       observable=True
   )

**Mathematical Background:**

If :math:`X \sim \text{Lomax}(\lambda, \alpha)`, then :math:`Y = \log(X)` follows the exp-Lomax distribution:

.. math::

   p(y | \lambda, \alpha) = \frac{\alpha \lambda^\alpha \exp(y)}{(\lambda + \exp(y))^{\alpha + 1}}

Sequence Convolution Functions
-----------------------------

**File**: ``pssm.stanfunctions``

**Functions Provided:**

.. code-block:: stan

   // Basic convolution
   real convolve(array[] vector weights, array[] int seq)

   // Sequence convolution
   vector convolve_sequence(array[] vector weights, array[] int seq)

**Usage in SciStanPy:**

.. code-block:: python

   # For sequence analysis (e.g., protein sequences)
   sequence = [0, 1, 2, 1, 0]  # Encoded sequence
   weights = ssp.parameters.Normal(mu=0, sigma=1, shape=(3, 3))

   convolved = ssp.operations.convolve_sequence(
       weights=weights,
       ordinals=sequence
   )

**Mathematical Background:**

1D convolution for sequences:

.. math::

   (f * g)[i] = \sum_{j=0}^{K-1} f[j] \cdot g[i+j]

where :math:`K` is the kernel size and the sequence is ordinally encoded.

Function Integration Process
---------------------------

**Automatic Inclusion:**

Functions are automatically included based on model components:

.. code-block:: python

   # This model will automatically include multinomial.stanfunctions
   class ModelWithMultinomial(ssp.Model):
       def __init__(self):
           super().__init__()
           self.counts = ssp.parameters.MultinomialLogit(
               n=100, logits=[0, 0, 0], observable=True
           )

**Include Path Configuration:**

.. code-block:: python

   # Functions are included via Stan's include mechanism
   # SciStanPy automatically configures include paths

   # Generated Stan code includes:
   # functions {
   #     #include <multinomial.stanfunctions>
   #     // Other required functions...
   # }

**Function Optimization:**

- **Duplicate Elimination**: Multiple uses of the same distribution type result in single function inclusion
- **Dependency Resolution**: Related functions are included together (e.g., MultinomialLogit includes Multinomial functions)
- **Vectorization**: Functions support both scalar and vector operations for efficiency

Numerical Stability Features
---------------------------

**Log-Space Computations:**

All exp-transformed distributions use log-space arithmetic to prevent:

- **Overflow**: For very large values
- **Underflow**: For very small probabilities
- **Precision Loss**: In extreme parameter ranges

**Robust Implementations:**

.. code-block:: stan

   // Example: Numerically stable log-sum-exp operations
   real stable_log_prob = log_sum_exp(log_alpha + log_lambda + y,
                                      log_normalization_constant)

**Gradient Stability:**

Functions are implemented to provide stable gradients for:

- **HMC Sampling**: Smooth energy landscapes
- **Variational Inference**: Stable gradient estimates
- **Optimization**: Reliable convergence properties

Custom Function Development
--------------------------

**Adding New Functions:**

To add new Stan functions to SciStanPy:

1. **Create .stanfunctions file** with function implementations
2. **Add to include paths** in Stan integration module
3. **Register with model components** that use the functions
4. **Test numerical stability** and gradient computation

**Function Template:**

.. code-block:: stan

   // Template for new distribution function
   real my_distribution_lpdf(real y, real param1, real param2) {
       // Implement log probability density
       // Use log-space operations for stability
       // Include proper normalization constants
       return log_prob_value;
   }

   real my_distribution_rng(real param1, real param2) {
       // Implement random number generation
       // Use Stan's built-in RNG functions
       return random_sample;
   }

**Best Practices for Function Development:**

1. **Use log-space arithmetic** for probability computations
2. **Implement multiple overloads** for scalar and vector operations
3. **Include proper normalization** constants
4. **Test against reference implementations** for correctness
5. **Validate numerical stability** across parameter ranges
6. **Document mathematical background** and usage patterns

Performance Considerations
-------------------------

**Vectorization Benefits:**

.. code-block:: stan

   // Vectorized operations are more efficient
   vector[N] log_probs = my_distribution_lpdf(y_vector | param_vector);

   // Than element-wise operations
   for (i in 1:N) {
       log_probs[i] = my_distribution_lpdf(y[i] | param[i]);
   }

**Memory Efficiency:**

- **In-place operations**: Minimize temporary array creation
- **Efficient indexing**: Use appropriate array vs matrix structures
- **Loop optimization**: Minimize nested loop complexity

The Stan functions library provides the mathematical foundation for SciStanPy's advanced probability distributions and specialized operations, enabling sophisticated scientific modeling while maintaining numerical stability and computational efficiency.
