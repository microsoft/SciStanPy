Contributing to SciStanPy
=========================

We welcome contributions to SciStanPy! This guide explains how to contribute effectively to the project.

Getting Started
--------------

Development Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Fork and Clone the Repository:**

.. code-block:: bash

   # Fork the repository on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/SciStanPy.git
   cd SciStanPy

2. **Create a Development Environment:**

.. code-block:: bash

   # Create conda environment
   conda create -n scistanpy-dev python=3.9
   conda activate scistanpy-dev

   # Install development dependencies
   pip install -e .[dev]

3. **Install Pre-commit Hooks:**

.. code-block:: bash

   # Install pre-commit hooks for code quality
   pre-commit install

4. **Verify Installation:**

.. code-block:: bash

   # Run tests to ensure everything works
   pytest tests/

   # Check code quality
   flake8 scistanpy/
   mypy scistanpy/

Development Workflow
~~~~~~~~~~~~~~~~~~~

1. **Create a Feature Branch:**

.. code-block:: bash

   git checkout -b feature/your-feature-name

2. **Make Changes and Test:**

.. code-block:: bash

   # Make your changes
   # Add tests for new functionality
   pytest tests/

   # Check code quality
   pre-commit run --all-files

3. **Commit Changes:**

.. code-block:: bash

   git add .
   git commit -m "feat: add description of your feature"

4. **Push and Create Pull Request:**

.. code-block:: bash

   git push origin feature/your-feature-name
   # Create PR on GitHub

Types of Contributions
---------------------

Bug Reports and Feature Requests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Bug Reports:**

When reporting bugs, please include:

- **Environment Details**: Python version, SciStanPy version, OS
- **Minimal Example**: Code that reproduces the issue
- **Expected vs Actual**: What you expected vs what happened
- **Stack Trace**: Full error messages if applicable

**Feature Requests:**

For new features, please provide:

- **Use Case**: Scientific problem you're trying to solve
- **Proposed API**: How you'd like the feature to work
- **Alternatives**: Other approaches you've considered
- **Implementation Ideas**: If you have technical suggestions

Code Contributions
~~~~~~~~~~~~~~~~~

**Areas Where We Need Help:**

- **New Distributions**: Domain-specific probability distributions
- **Examples**: Real-world scientific modeling examples
- **Documentation**: Tutorials, API documentation, examples
- **Performance**: Optimization and profiling improvements
- **Testing**: Unit tests, integration tests, scientific validation
- **Backend Integration**: Enhanced NumPy, PyTorch, Stan support

**Distribution Contributions:**

.. code-block:: python

   # Minimal pattern (log_prob & sample stubs)
   class YourDistribution(ssp.parameters.ContinuousDistribution):
       def __init__(self, param1, param2):
           super().__init__(param1=param1, param2=param2)

       def log_prob(self, value):
           raise NotImplementedError

       def sample(self, sample_shape):
           raise NotImplementedError

**Example Contributions:**

.. code-block:: python

   # Example: Contributing a scientific example
   def enzyme_kinetics_example():
       """Michaelis-Menten enzyme kinetics analysis.

       This example demonstrates how to analyze enzyme kinetics data
       using Bayesian parameter estimation with SciStanPy.
       """

       # Data setup
       substrate_conc = np.array([...])
       velocities = np.array([...])

       # Model definition
       V_max = ssp.parameters.LogNormal(mu=np.log(10), sigma=0.5)
       K_m = ssp.parameters.LogNormal(mu=np.log(1), sigma=0.5)

       # Continue with complete example...

Code Style and Standards
-----------------------

Python Code Style
~~~~~~~~~~~~~~~~

We follow PEP 8 with some modifications:

.. code-block:: python

   # Good: Clear, descriptive names
   def compute_posterior_predictive_distribution(model, samples):
       """Compute posterior predictive distribution."""
       pass

   # Bad: Unclear abbreviations
   def comp_post_pred(m, s):
       pass

**Naming Conventions:**

- **Classes**: PascalCase (e.g., `MultivariateNormal`)
- **Functions**: snake_case (e.g., `sample_posterior`)
- **Variables**: snake_case (e.g., `prior_samples`)
- **Constants**: UPPER_CASE (e.g., `DEFAULT_N_SAMPLES`)

**Type Hints:**

.. code-block:: python

   from typing import Optional, Union
   import numpy as np

   def sample_distribution(
       distribution: 'Distribution',
       n_samples: int,
       seed: Optional[int] = None
   ) -> np.ndarray:
       """Sample from a probability distribution."""
       pass

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~

**Docstring Format (NumPy Style):**

.. code-block:: python

   def your_function(param1: float, param2: str, param3: Optional[int] = None) -> dict:
       """One-line summary of the function.

       Longer description that explains what the function does,
       its purpose, and any important details about its behavior.

       Parameters
       ----------
       param1 : float
           Description of param1, including units if applicable
       param2 : str
           Description of param2
       param3 : int, optional
           Description of param3, default is None

       Returns
       -------
       dict
           Description of return value and its structure

       Raises
       ------
       ValueError
           When param1 is negative

       Examples
       --------
       >>> result = your_function(1.0, "test")
       >>> print(result)
       {'key': 'value'}

       Notes
       -----
       Any additional notes about implementation, algorithms,
       or scientific background.

       References
       ----------
       .. [1] Author, "Title", Journal, Year.
       """
       pass

Testing Guidelines
-----------------

Test Organization
~~~~~~~~~~~~~~~

Tests are organized in the `tests/` directory:

.. code-block::

   tests/
   ├── unit/                    # Unit tests
   │   ├── test_parameters.py
   │   ├── test_transforms.py
   │   └── test_models.py
   ├── integration/             # Integration tests
   │   ├── test_workflows.py
   │   └── test_backends.py
   ├── scientific/              # Scientific validation
   │   ├── test_examples.py
   │   └── test_accuracy.py
   └── fixtures/                # Test data and fixtures

**Writing Unit Tests:**

.. code-block:: python

   import pytest
   import numpy as np
   import scistanpy as ssp

   class TestNormalDistribution:
       """Test suite for Normal distribution."""

       def test_initialization(self):
           """Test normal distribution initialization."""
           dist = ssp.parameters.Normal(mu=0, sigma=1)
           assert dist.mu == 0
           assert dist.sigma == 1

       def test_sampling_shape(self):
           """Test that sampling produces correct shapes."""
           dist = ssp.parameters.Normal(mu=0, sigma=1)
           samples = dist.sample((100,))
           assert samples.shape == (100,)

       def test_log_prob(self):
           """Test log probability computation."""
           dist = ssp.parameters.Normal(mu=0, sigma=1)
           log_prob = dist.log_prob(0.0)
           expected = -0.5 * np.log(2 * np.pi)  # Standard normal at 0
           np.testing.assert_allclose(log_prob, expected, rtol=1e-6)

       @pytest.mark.parametrize("mu,sigma", [(0, 1), (5, 2), (-3, 0.5)])
       def test_parameter_variations(self, mu, sigma):
           """Test with different parameter values."""
           dist = ssp.parameters.Normal(mu=mu, sigma=sigma)
           samples = dist.sample((10,))
           assert len(samples) == 10

**Scientific Validation Tests:**

.. code-block:: python

   def test_linear_regression_recovery():
       """Test that we can recover known linear regression parameters."""

       # Generate synthetic data with known parameters
       np.random.seed(42)
       true_slope = 2.5
       true_intercept = 1.0
       true_sigma = 0.3

       x = np.linspace(0, 10, 50)
       y = true_intercept + true_slope * x + np.random.normal(0, true_sigma, 50)

       # Fit model
       intercept = ssp.parameters.Normal(mu=0, sigma=5)
       slope = ssp.parameters.Normal(mu=0, sigma=5)
       sigma = ssp.parameters.LogNormal(mu=0, sigma=1)

       predictions = intercept + slope * x
       likelihood = ssp.parameters.Normal(mu=predictions, sigma=sigma)
       likelihood.observe(y)

       model = ssp.Model(likelihood)
       results = model.mcmc(chains=2, iter_warmup=200, iter_sampling=400)

       # Check parameter recovery
       assert abs(results['slope'].mean() - true_slope) < 0.1
       assert abs(results['intercept'].mean() - true_intercept) < 0.1
       assert abs(results['sigma'].mean() - true_sigma) < 0.05

Continuous Integration
~~~~~~~~~~~~~~~~~~~~

Our CI pipeline runs:

1. **Code Quality Checks**: flake8, mypy, black
2. **Unit Tests**: pytest with coverage reporting
3. **Integration Tests**: Multi-backend testing
4. **Scientific Validation**: Parameter recovery tests
5. **Documentation**: Sphinx build verification

**Local CI Simulation:**

.. code-block:: bash

   # Run the full CI pipeline locally
   ./scripts/run_ci.sh

Documentation Contributions
--------------------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install documentation dependencies
   pip install -e .[docs]

   # Build documentation
   cd docs/
   make html

   # View documentation
   open _build/html/index.html

**Adding Examples:**

.. code-block:: rst

   Example: Your Scientific Domain
   ===============================

   This example demonstrates SciStanPy for your specific scientific application.

   Scientific Background
   --------------------

   Explain the scientific context and problem you're solving.

   Implementation
   -------------

   .. code-block:: python

      import scistanpy as ssp
      import numpy as np

      # Your example code here

**API Documentation:**

API documentation is auto-generated from docstrings. Ensure your docstrings follow the NumPy format.

Review Process
-------------

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~

**Before Submitting:**

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Changelog entry added (if applicable)
- [ ] Scientific examples validated

**PR Description Template:**

.. code-block:: markdown

   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update

   ## Scientific Context
   What scientific problem does this address?

   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests pass
   - [ ] Scientific validation included

   ## Related Issues
   Closes #issue_number

**Review Criteria:**

- **Correctness**: Does the code work as intended?
- **Scientific Accuracy**: Are the mathematical/statistical aspects correct?
- **Code Quality**: Is the code clean, readable, and well-documented?
- **Performance**: Does it meet performance requirements?
- **Compatibility**: Does it work across supported platforms?

Community Guidelines
-------------------

Code of Conduct
~~~~~~~~~~~~~~

We are committed to providing a welcoming and inclusive environment:

- **Be Respectful**: Treat everyone with respect and courtesy
- **Be Collaborative**: Work together constructively
- **Be Inclusive**: Welcome people of all backgrounds and skill levels
- **Be Patient**: Help newcomers learn and grow

Communication Channels
~~~~~~~~~~~~~~~~~~~~~

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas, showcases
- **Pull Requests**: Code contributions and reviews

Recognition
----------

Contributors are recognized in:

- **CONTRIBUTORS.md**: All contributors listed
- **Release Notes**: Major contributions highlighted
- **Documentation**: Example contributors credited
- **Presentations**: Community contributions showcased

Getting Help
-----------

If you need help contributing:

- **Documentation**: Read this contributing guide thoroughly
- **Examples**: Look at existing code for patterns
- **Issues**: Check existing issues for similar problems
- **Discussions**: Ask questions in GitHub Discussions
- **Mentorship**: Experienced contributors are happy to help

Thank you for contributing to SciStanPy! Your contributions help make Bayesian modeling more accessible to the scientific community.
