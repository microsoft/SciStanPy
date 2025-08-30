.. SciStanPy documentation master file, created by
   sphinx-quickstart on Thu Aug 28 18:08:16 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SciStanPy: Intuitive Bayesian Modeling for Scientists
======================================================

Welcome to SciStanPy, a Python framework that makes Bayesian statistical modeling
accessible to scientists and researchers. Express your scientific models naturally
in Python while SciStanPy automatically handles the computational complexity.

.. note::
   SciStanPy is designed for scientists with basic Python knowledge who want to
   incorporate uncertainty quantification and Bayesian analysis into their research
   without needing deep statistical programming expertise.

Quick Start
-----------

Install SciStanPy and start modeling in minutes:

.. code-block:: bash

   pip install scistanpy

.. code-block:: python

   import scistanpy as ssp
   import numpy as np

   # Your experimental data
   measurements = np.array([10.2, 9.8, 10.1, 9.9, 10.3])

   # Define your model
   true_value = ssp.parameters.Normal(mu=10, sigma=1)
   likelihood = ssp.parameters.Normal(mu=true_value, sigma=0.2)
   likelihood.observe(measurements)

   # Build and run inference
   model = ssp.Model(likelihood)
   results = model.sample()

Key Features
------------

🔬 **Scientific Focus**
   Designed specifically for scientific applications and research workflows

🐍 **Intuitive Python Interface**
   Express models using familiar Python syntax and scientific thinking

⚡ **Multi-Backend Performance**
   Automatic compilation to high-performance Stan code with NumPy and PyTorch support

🛡️ **Built-in Validation**
   Comprehensive error checking catches modeling mistakes early

🎯 **Uncertainty Quantification**
   Natural handling of measurement uncertainty and parameter estimation

📊 **Rich Diagnostics**
   Comprehensive model checking and convergence diagnostics

.. toctree::
   :maxdepth: 3
   :caption: API Reference
   :hidden:

   docs/api/index
   docs/advanced/index
   docs/community/index
   docs/development/index
   docs/examples/index
   docs/installation
   docs/key_concepts
   docs/quickstart
   docs/user_guide/index


Common Use Cases
----------------

**Parameter Estimation**
   Determine values and uncertainties for physical constants, reaction rates,
   or model parameters from experimental data.

**Model Comparison**
   Compare different theoretical models and hypotheses using Bayesian model
   selection and information criteria.

**Uncertainty Propagation**
   Track how measurement uncertainties affect derived quantities and model
   predictions.

**Experimental Design**
   Optimize future experiments by predicting outcomes and quantifying
   expected information gain.

**Data Fusion**
   Combine multiple data sources with different uncertainties and measurement
   characteristics.

**Time Series Analysis**
   Model temporal dynamics, detect trends, and make forecasts with proper
   uncertainty quantification.

Scientific Domains
------------------

SciStanPy has been successfully applied across diverse scientific fields:

- **Physics & Astronomy**: Parameter estimation, cosmological modeling, detector calibration
- **Chemistry**: Reaction kinetics, spectral analysis, mixture modeling
- **Biology & Medicine**: Population dynamics, dose-response modeling, clinical trials
- **Environmental Science**: Climate modeling, pollution monitoring, ecosystem dynamics
- **Engineering**: Reliability analysis, quality control, system identification
- **Psychology & Social Sciences**: Cognitive modeling, survey analysis, behavioral studies

Getting Help
------------

- **Documentation**: Comprehensive guides and API reference
- **Examples**: Real-world scientific modeling examples
- **Community**: GitHub discussions and issue tracking
- **Support**: Professional consulting and training available

.. note::
   This documentation assumes basic familiarity with Python programming.
   No prior knowledge of Bayesian statistics or Stan is required.

Advanced Usage
--------------


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

API Documentation
=================

Core Modules
------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   scistanpy.model
   scistanpy.operations
   scistanpy.custom_types
   scistanpy.utils
   scistanpy.defaults
   scistanpy.exceptions
   scistanpy.plotting

Model Components
----------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   scistanpy.model
   scistanpy.model.model
   scistanpy.model.nn_module
   scistanpy.model.mle
   scistanpy.model.stan
   scistanpy.model.stan.stan_model
   scistanpy.model.results
   scistanpy.model.results.mle
   scistanpy.model.results.hmc
   scistanpy.model.components
   scistanpy.model.components.abstract_model_component
   scistanpy.model.components.parameters
   scistanpy.model.components.constants
   scistanpy.model.components.transformations
   scistanpy.model.components.transformations.transformed_parameters
   scistanpy.model.components.transformations.transformed_data
   scistanpy.model.components.transformations.cdfs
   scistanpy.model.components.custom_distributions
   scistanpy.model.components.custom_distributions.custom_torch_dists
   scistanpy.model.components.custom_distributions.custom_scipy_dists
   scistanpy.plotting
   scistanpy.plotting.plotting

Advanced
========


Copyright and License
=====================

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
- **Community**: GitHub discussions and issue tracking
- **Support**: Professional consulting and training available

.. note::
   This documentation assumes basic familiarity with Python programming.
   No prior knowledge of Bayesian statistics or Stan is required.