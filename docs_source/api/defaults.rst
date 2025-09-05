Defaults API Reference
======================

This reference covers the default configuration values and settings used throughout SciStanPy.

Defaults Module
---------------

.. automodule:: scistanpy.defaults
   :undoc-members:
   :show-inheritance:

Default Configuration Values
----------------------------

The defaults module provides centralized configuration for SciStanPy behavior. This module should contain default values for various aspects of the framework:

**Expected Default Categories:**

- **Sampling Parameters**: Default values for MCMC sampling
- **Optimization Settings**: Default values for variational inference and MLE
- **Numerical Tolerances**: Default precision and convergence criteria
- **Backend Settings**: Default computational backend preferences
- **Validation Options**: Default validation and error checking behavior

**Example Usage Pattern:**

.. code-block:: python

   import scistanpy as ssp

   # Access default values (exact attributes depend on defaults.py content)
   # These are example attribute names that might exist:
   # default_samples = ssp.defaults.DEFAULT_N_SAMPLES
   # default_chains = ssp.defaults.DEFAULT_N_CHAINS
   # default_backend = ssp.defaults.DEFAULT_BACKEND

   # Use in model configuration
   model = ssp.Model(likelihood)
   # results = model.sample(
   #     n_samples=default_samples,
   #     n_chains=default_chains
   # )

Common Default Values
---------------------

Based on typical Bayesian modeling practices, the defaults module likely includes:

**Sampling Defaults:**

.. code-block:: python

   # Expected default values (to be confirmed with actual defaults.py):
   # DEFAULT_N_SAMPLES = 1000        # Number of posterior samples
   # DEFAULT_N_CHAINS = 4            # Number of MCMC chains
   # DEFAULT_N_WARMUP = 1000         # Warmup samples per chain
   # DEFAULT_THIN = 1                # Thinning factor
   # DEFAULT_ADAPT_DELTA = 0.8       # HMC adaptation parameter

**Backend Defaults:**

.. code-block:: python

   # Expected backend configuration:
   # DEFAULT_BACKEND = 'stan'        # Primary computational backend
   # DEFAULT_DEVICE = 'cpu'          # Default compute device
   # DEFAULT_COMPILE_MODEL = True    # Whether to compile models

**Numerical Defaults:**

.. code-block:: python

   # Expected numerical tolerances:
   # DEFAULT_RTOL = 1e-6            # Relative tolerance
   # DEFAULT_ATOL = 1e-8            # Absolute tolerance
   # DEFAULT_MAX_ITER = 10000       # Maximum iterations

**Validation Defaults:**

.. code-block:: python

   # Expected validation settings:
   # DEFAULT_VALIDATE_SHAPES = True  # Shape validation
   # DEFAULT_VALIDATE_PRIORS = True  # Prior validation
   # DEFAULT_CHECK_FINITE = True     # Finite value checking

Customizing Defaults
--------------------

**Modifying Defaults:**

.. code-block:: python

   # Modify defaults for your session (example pattern)
   # ssp.defaults.DEFAULT_N_SAMPLES = 2000
   # ssp.defaults.DEFAULT_BACKEND = 'pytorch'

**Configuration Context:**

.. code-block:: python

   # Use context managers for temporary changes (if implemented)
   # with ssp.defaults.temporary_config(n_samples=5000, backend='pytorch'):
   #     results = model.sample()  # Uses temporary settings
   # # Reverts to original defaults after context

Integration with Components
---------------------------

**Automatic Default Usage:**

.. code-block:: python

   # Components automatically use default values when not specified
   model = ssp.Model(likelihood)

   # This would use default sampling parameters:
   # results = model.sample()  # Uses DEFAULT_N_SAMPLES, DEFAULT_N_CHAINS, etc.

   # Explicit parameters override defaults:
   # results = model.sample(n_samples=2000)  # Overrides DEFAULT_N_SAMPLES

**Backend Selection:**

.. code-block:: python

   # Default backend selection logic (example)
   # model.set_backend()  # Uses DEFAULT_BACKEND
   # model.set_backend('pytorch')  # Explicit override

Configuration Management
------------------------

**Loading Configuration:**

.. code-block:: python

   # If configuration files are supported:
   # config = ssp.defaults.load_config('my_config.yaml')
   # ssp.defaults.update_defaults(config)

**Saving Configuration:**

.. code-block:: python

   # Save current defaults to file:
   # ssp.defaults.save_config('current_config.yaml')

Environment Variables
---------------------

**Environment-Based Configuration:**

The defaults module may support configuration via environment variables:

.. code-block:: bash

   # Example environment variable usage:
   export SCISTANPY_DEFAULT_BACKEND=pytorch
   export SCISTANPY_DEFAULT_N_SAMPLES=2000
   export SCISTANPY_DEVICE=cuda

**Priority Order:**

Configuration precedence (typical pattern):

1. Explicit function arguments (highest priority)
2. Runtime configuration changes
3. Environment variables
4. Configuration files
5. Built-in defaults (lowest priority)

See Also
--------

- :doc:`exceptions` - Error handling for invalid configuration
- :doc:`utils` - Utility functions that may use defaults
