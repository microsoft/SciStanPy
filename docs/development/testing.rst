Testing Guide
============

This guide covers the testing strategy and practices for SciStanPy development.

Testing Philosophy
-----------------

SciStanPy testing follows a multi-layered approach:

1. **Unit Testing**: Individual component functionality
2. **Integration Testing**: Component interactions and workflows
3. **Scientific Validation**: Accuracy against known results
4. **Performance Testing**: Scalability and efficiency
5. **Cross-Platform Testing**: Compatibility across environments

Test Structure
-------------

**Test Organization:**

.. code-block:: text

   tests/
   ├── unit/                    # Unit tests for individual components
   │   ├── test_parameters.py   # Parameter class tests
   │   ├── test_operations.py   # Mathematical operations
   │   ├── test_model.py        # Model construction
   │   ├── test_backends.py     # Backend functionality
   │   └── test_utils.py        # Utility functions
   ├── integration/             # Integration and workflow tests
   │   ├── test_workflows.py    # End-to-end workflows
   │   ├── test_examples.py     # Example execution
   │   └── test_backends_integration.py
   ├── scientific/              # Scientific validation tests
   │   ├── test_parameter_recovery.py
   │   ├── test_model_accuracy.py
   │   └── test_statistical_properties.py
   ├── performance/             # Performance and benchmarking
   │   ├── test_scalability.py
   │   └── benchmarks/
   └── fixtures/                # Test data and utilities
       ├── data/
       ├── models/
       └── conftest.py

Running Tests
------------

**Basic Test Execution:**

.. code-block:: bash

   # Run all tests
   pytest tests/

   # Run specific test categories
   pytest tests/unit/          # Unit tests only
   pytest tests/integration/   # Integration tests only
   pytest tests/scientific/    # Scientific validation only

   # Run tests with coverage
   pytest tests/ --cov=scistanpy --cov-report=html

   # Run tests in parallel
   pytest tests/ -n auto

**Test Configuration:**

.. code-block:: bash

   # pytest.ini configuration
   [tool:pytest]
   testpaths = tests
   python_files = test_*.py
   python_classes = Test*
   python_functions = test_*
   addopts =
       --strict-markers
       --disable-warnings
       --tb=short
   markers =
       unit: Unit tests
       integration: Integration tests
       scientific: Scientific validation tests
       slow: Slow-running tests
       gpu: Tests requiring GPU

Unit Testing
-----------

**Parameter Testing:**

.. code-block:: python

   # tests/unit/test_parameters.py
   import pytest
   import numpy as np
   import scistanpy as ssp

   class TestNormalParameter:
       """Test Normal parameter functionality."""

       def test_initialization(self):
           """Test parameter initialization."""
           param = ssp.parameters.Normal(mu=0, sigma=1)
           assert param.mu == 0
           assert param.sigma == 1

       def test_invalid_parameters(self):
           """Test validation of invalid parameters."""
           with pytest.raises(ssp.exceptions.ParameterError):
               ssp.parameters.Normal(mu=0, sigma=-1)  # Negative sigma

       @pytest.mark.parametrize("mu,sigma", [
           (0, 1), (5, 2), (-3, 0.5), (0, 10)
       ])
       def test_parameter_variations(self, mu, sigma):
           """Test with various parameter values."""
           param = ssp.parameters.Normal(mu=mu, sigma=sigma)
           samples = param.sample((100,))
           assert samples.shape == (100,)
           assert np.isfinite(samples).all()

       def test_log_prob(self):
           """Test log probability computation."""
           param = ssp.parameters.Normal(mu=0, sigma=1)
           log_prob = param.log_prob(0.0)
           expected = -0.5 * np.log(2 * np.pi)
           np.testing.assert_allclose(log_prob, expected, rtol=1e-6)

**Transformation Testing:**

.. code-block:: python

   # tests/unit/test_operations.py
   class TestMathematicalOperations:
       """Test mathematical operations on parameters."""

       def test_addition(self):
           """Test parameter addition."""
           param1 = ssp.parameters.Normal(mu=1, sigma=0.1)
           param2 = ssp.parameters.Normal(mu=2, sigma=0.1)

           result = param1 + param2
           assert isinstance(result, ssp.model.components.transformations.BinaryTransformedParameter)

       def test_exp_operation(self):
           """Test exponential transformation."""
           param = ssp.parameters.Normal(mu=0, sigma=1)
           exp_param = ssp.operations.exp(param)

           # Test that it generates correct Stan code
           stan_code = exp_param.write_stan_code()
           assert 'exp(' in stan_code

       def test_shape_compatibility(self):
           """Test shape compatibility in operations."""
           param1 = ssp.parameters.Normal(mu=0, sigma=1, shape=(5,))
           param2 = ssp.parameters.Normal(mu=0, sigma=1, shape=(3,))

           with pytest.raises(ssp.exceptions.ShapeError):
               result = param1 + param2

Integration Testing
------------------

**Workflow Testing:**

.. code-block:: python

   # tests/integration/test_workflows.py
   class TestModelingWorkflow:
       """Test complete modeling workflows."""

       def test_linear_regression_workflow(self):
           """Test end-to-end linear regression."""
           # Generate synthetic data
           np.random.seed(42)
           x = np.linspace(0, 10, 50)
           true_slope = 2.5
           true_intercept = 1.0
           y = true_intercept + true_slope * x + np.random.normal(0, 0.5, 50)

           # Define model
           intercept = ssp.parameters.Normal(mu=0, sigma=5)
           slope = ssp.parameters.Normal(mu=0, sigma=5)
           sigma = ssp.parameters.LogNormal(mu=0, sigma=1)

           predictions = intercept + slope * x
           likelihood = ssp.parameters.Normal(mu=predictions, sigma=sigma)
           likelihood.observe(y)

           # Test model creation
           model = ssp.Model(likelihood)
           assert model is not None

           # Test inference methods
           mle_results = model.mle()
           assert 'intercept' in mle_results['estimates']
           assert 'slope' in mle_results['estimates']

           # Test MCMC
           mcmc_results = model.mcmc(chains=2, iter_warmup=200, iter_sampling=300)
           sample_failures, variable_failures = mcmc_results.diagnose()
           assert sample_failures == 0
           assert variable_failures == 0

       def test_hierarchical_model_workflow(self):
           """Test hierarchical modeling workflow."""
           # Implementation for hierarchical model testing
           pass

**Backend Integration:**

.. code-block:: python

   # tests/integration/test_backends_integration.py
   class TestBackendIntegration:
       """Test integration across different backends."""

       @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'stan'])
       def test_backend_consistency(self, backend):
           """Test that backends produce consistent results."""
           # Simple model
           mu = ssp.parameters.Normal(mu=0, sigma=1)
           likelihood = ssp.parameters.Normal(mu=mu, sigma=0.1)
           likelihood.observe([0.1, -0.1, 0.05])

           model = ssp.Model(likelihood)
           model.set_backend(backend)

           if backend in ['numpy', 'pytorch']:
               results = model.mle()
               assert 'mu' in results['estimates']
           elif backend == 'stan':
               results = model.sample(n_samples=100)
               assert 'mu' in results

Scientific Validation
--------------------

**Parameter Recovery Tests:**

.. code-block:: python

   # tests/scientific/test_parameter_recovery.py
   class TestParameterRecovery:
       """Test that models can recover known parameters."""

       def test_normal_parameter_recovery(self):
           """Test recovery of normal distribution parameters."""
           # Known parameters
           true_mu = 2.5
           true_sigma = 1.2
           n_samples = 1000

           # Generate data from known distribution
           np.random.seed(123)
           data = np.random.normal(true_mu, true_sigma, n_samples)

           # Fit model
           mu = ssp.parameters.Normal(mu=0, sigma=5)
           sigma = ssp.parameters.LogNormal(mu=0, sigma=1)

           likelihood = ssp.parameters.Normal(mu=mu, sigma=sigma)
           likelihood.observe(data)

           model = ssp.Model(likelihood)
           results = model.mcmc(chains=4, iter_warmup=500, iter_sampling=1000)

           # Check parameter recovery
           estimated_mu = results['mu'].mean()
           estimated_sigma = results['sigma'].mean()

           assert abs(estimated_mu - true_mu) < 0.1
           assert abs(estimated_sigma - true_sigma) < 0.1

       def test_regression_parameter_recovery(self):
           """Test recovery of regression parameters."""
           # Implementation for regression parameter recovery
           pass

**Statistical Properties Tests:**

.. code-block:: python

   # tests/scientific/test_statistical_properties.py
   class TestStatisticalProperties:
       """Test statistical properties of implementations."""

       def test_distribution_moments(self):
           """Test that distributions have correct moments."""
           # Normal distribution moments
           mu, sigma = 2.0, 1.5
           normal = ssp.parameters.Normal(mu=mu, sigma=sigma)

           samples = normal.sample((10000,))

           # Test mean
           sample_mean = samples.mean()
           assert abs(sample_mean - mu) < 0.1

           # Test standard deviation
           sample_std = samples.std()
           assert abs(sample_std - sigma) < 0.1

       def test_mcmc_convergence(self):
           """Test MCMC convergence properties."""
           # Implementation for convergence testing
           pass

Performance Testing
------------------

**Scalability Tests:**

.. code-block:: python

   # tests/performance/test_scalability.py
   import time
   import pytest

   class TestScalability:
       """Test performance and scalability."""

       @pytest.mark.slow
       def test_large_dataset_performance(self):
           """Test performance with large datasets."""
           # Generate large dataset
           n_obs = 10000
           data = np.random.normal(0, 1, n_obs)

           # Simple model
           mu = ssp.parameters.Normal(mu=0, sigma=1)
           sigma = ssp.parameters.LogNormal(mu=0, sigma=0.5)

           likelihood = ssp.parameters.Normal(mu=mu, sigma=sigma)
           likelihood.observe(data)

           model = ssp.Model(likelihood)

           # Time the inference
           start_time = time.time()
           results = model.mcmc(chains=2, iter_warmup=200, iter_sampling=400)
           elapsed_time = time.time() - start_time

           # Should complete within reasonable time
           assert elapsed_time < 30  # 30 seconds max

**Memory Usage Tests:**

.. code-block:: python

   # tests/performance/test_memory_usage.py
   import psutil
   import os

   class TestMemoryUsage:
       """Test memory usage patterns."""

       def test_memory_efficiency(self):
           """Test that memory usage is reasonable."""
           process = psutil.Process(os.getpid())
           initial_memory = process.memory_info().rss

           # Create and run model
           mu = ssp.parameters.Normal(mu=0, sigma=1)
           likelihood = ssp.parameters.Normal(mu=mu, sigma=0.1)
           likelihood.observe(np.random.normal(0, 0.1, 1000))

           model = ssp.Model(likelihood)
           results = model.mcmc(chains=2, iter_warmup=200, iter_sampling=400)

           final_memory = process.memory_info().rss
           memory_increase = final_memory - initial_memory

           # Memory increase should be reasonable (< 100MB)
           assert memory_increase < 100 * 1024 * 1024

Test Fixtures and Utilities
--------------------------

**Shared Fixtures:**

.. code-block:: python

   # tests/fixtures/conftest.py
   import pytest
   import numpy as np
   import scistanpy as ssp

   @pytest.fixture
   def simple_linear_data():
       """Generate simple linear regression data."""
       np.random.seed(42)
       x = np.linspace(0, 10, 50)
       y = 1.0 + 2.0 * x + np.random.normal(0, 0.5, 50)
       return x, y

   @pytest.fixture
   def simple_model(simple_linear_data):
       """Create a simple linear regression model."""
       x, y = simple_linear_data

       intercept = ssp.parameters.Normal(mu=0, sigma=5)
       slope = ssp.parameters.Normal(mu=0, sigma=5)
       sigma = ssp.parameters.LogNormal(mu=0, sigma=1)

       predictions = intercept + slope * x
       likelihood = ssp.parameters.Normal(mu=predictions, sigma=sigma)
       likelihood.observe(y)

       return ssp.Model(likelihood)

Continuous Integration
---------------------

**CI Configuration:**

.. code-block:: yaml

   # .github/workflows/tests.yml
   name: Tests

   on: [push, pull_request]

   jobs:
     test:
       runs-on: ${{ matrix.os }}
       strategy:
         matrix:
           os: [ubuntu-latest, windows-latest, macos-latest]
           python-version: [3.8, 3.9, '3.10', 3.11]

       steps:
       - uses: actions/checkout@v3
       - name: Set up Python
         uses: actions/setup-python@v3
         with:
           python-version: ${{ matrix.python-version }}
       - name: Install dependencies
         run: |
           pip install -e .[dev]
       - name: Run tests
         run: |
           pytest tests/ --cov=scistanpy

**Test Categories in CI:**

- **Fast tests**: Run on every commit
- **Slow tests**: Run nightly or on release branches
- **GPU tests**: Run on GPU-enabled runners
- **Platform tests**: Run on multiple OS/Python combinations

Best Practices
--------------

1. **Write tests first**: Use TDD for new features
2. **Test edge cases**: Include boundary conditions and error cases
3. **Use meaningful names**: Test names should describe what they verify
4. **Keep tests isolated**: Each test should be independent
5. **Test the interface**: Focus on public API behavior
6. **Mock external dependencies**: Use mocks for Stan, GPU operations, etc.
7. **Measure coverage**: Aim for >90% test coverage
8. **Performance benchmarks**: Track performance regressions
9. **Scientific validation**: Always test against known results
10. **Diagnostics**: Rely on results.diagnose() for MCMC diagnostics

Accuracy Note
-------------

All references to: model.sample(), variational(), GPU tests, distributed / parallel
chain helpers, posterior predictive, WAIC/LOO removed. Current inference surface:
mle(), mcmc(), draw(), prior_predictive(), diagnose().

This comprehensive testing strategy ensures SciStanPy maintains high quality and reliability across all supported use cases.
