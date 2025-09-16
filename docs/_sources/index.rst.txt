SciStanPy: Intuitive Bayesian Modeling for Scientists
======================================================

Welcome to SciStanPy, a Python framework that makes Bayesian statistical modeling accessible to scientists and researchers. Whether you're analyzing experimental data, building predictive models, or exploring uncertainty in your research, SciStanPy provides an intuitive interface that bridges the gap between scientific thinking and advanced statistical computation. Express your scientific models naturally in Python while SciStanPy automatically handles the computational complexity of model fitting and sampling via `PyTorch <https://pytorch.org/>`_ and `Stan <https://mc-stan.org/>`_.

Why SciStanPy?
--------------
Bayesian modeling often requires deep statistical programming knowledge. SciStanPy changes this by letting you express your scientific models naturally in Python, while automatically handling the complex computational details behind the scenes.

🔬 **Scientific Focus**
   Designed specifically for scientific applications and research workflows

🐍 **Intuitive Python Interface**
   Express hypotheses using familiar Python syntax and scientific thinking

🎯 **Uncertainty Quantification**
   Natural handling of measurement uncertainty and parameter estimation

📊 **Rich Diagnostics**
   Comprehensive model checking and convergence diagnostics

⚡ **Multi-Backend Performance**
   Automatic compilation to Stan code and PyTorch modules

🛡️ **Built-in Validation**
   Comprehensive error checking catches modeling mistakes early

Key Concepts
------------

**Parameters vs Observations**: In SciStanPy, parameters represent unknown quantities you want to learn about ("infer"), while observations are your measured data.

**Distributions**: These encode your knowledge and uncertainty. Use them to express:

- Prior knowledge about parameters
- Expected relationships between variables
- Measurement uncertainty and noise

In SciStanPy, parameters and observations are defined using distributions.

**Model Building**: Parameters can depend on one another and/or be transformed using standard mathematical operations. Combine and transform parameters to build a SciStanPy model.

**Inference**: Once your model is built, SciStanPy handles compilation to Stan code or PyTorch modules, allowing for inference of parameter values.

.. note::
   SciStanPy is designed for scientists with basic Python knowledge who want to incorporate uncertainty quantification and Bayesian analysis into their research
   without needing deep statistical programming expertise. For an introduction to Bayesian modeling, the `BE/Bi 103b class <https://bebi103b.github.io/>`_ developed
   and taught by Justin Bois at Caltech is an excellent resource.


Quick Start
-----------
This section assumes that you have followed the installation instructions found on GitHub. If not, return to the `GitHub repository <https://github.com/microsoft/scistanpy>`_ for detailed setup instructions.

Comprehensive documentation for the SciStanPy API can be found :doc:`here <api/index>`. For starting quickly, however, the most important modules and objects are :doc:`model <api/model/model>`, :doc:`parameters <api/model/components/parameters>`, and :doc:`operations <api/operations>`. Less frequently used, but also important, is the :doc:`constants <api/model/components/constants>` module. Follow the links for detailed API references.

The :py:class:`~scistanpy.model.model.Model` object forms the backbone of all SciStanPy models, distributions defined in :doc:`parameters <api/model/components/parameters>` define those models' variables, :doc:`operations <api/operations>` define transformations of those variables, and :doc:`constants <api/model/components/constants>` provide fixed values used throughout the model. If you work with PyTorch, defining models in SciStanPy will feel quite familiar:

.. code-block:: python

   import scistanpy as ssp
   import numpy as np

   # Define/import experimental data
   temperatures = np.array([20, 25, 30, 35, 40])  # °C
   reaction_rates = np.array([0.1, 0.3, 0.8, 2.1, 5.2])  # units/sec

   # Define the model. Here we're modeling the effect of temperature on reaction
   # rates
   class MyModel(ssp.Model):
      def __init__(self, temperatures, reaction_rates):

         # Record default data
         super().__init__(default_data = {"reaction_rates": reaction_rates})

         # Define priors
         self.baseline_rate = ssp.parameters.LogNormal(mu=0.0, sigma=1.0)  # Rates are always positive
         self.temperature_effect = ssp.parameters.Normal(mu=0.0, sigma=0.5)  # Effect of temperature

         # Model the relationship (Arrhenius-like behavior with noise)
         self.reaction_rates = ssp.parameters.Normal(
               mu = self.baseline_rate * ssp.operations.exp(self.temperature_effect * temperatures),
               sigma = 0.1
         )

   # Build the model
   model = MyModel(temperatures, reaction_rates)

Once a model is defined, there are multiple ways to use it. First, you may wish to explore the effects of different hyperparameter values. This can be done through an interactive interface as below:

.. code-block:: python

   # If you are in a Jupyter notebook, you can create an interactive dashboard to
   # explore prior predictive distributions. Just run:
   model.prior_predictive()

Additional information on the dashboard can be found :doc:`here <api/plotting/prior_predictive>`.

You can also directly draw from the prior distribution with the below:

.. code-block:: python

   # Draw 100 samples from the prior distribution
   prior_samples = model.draw(100)

The returned dictionary will contain draws from all model parameters, each conditioned on one another as appropriate. See :py:meth:`~scistanpy.model.model.Model.draw` for more details.

SciStanPy models are also fully compatible with PyTorch, and can be compiled to a `PyTorch Module <https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html>`_ using the :py:meth:`~scistanpy.model.model.Model.to_pytorch` method as below:

.. code-block:: python

   # Convert to PyTorch module
   torch_model = model.to_pytorch()

The returned PyTorch Module will have a forward method defined that takes observed data as input (using keyword arguments named according to the observables defined in the SciStanPy model -- ``reaction_rates`` in the example above) and returns the log-likelihood of the data given the model. Parameter values for the returned model will initially be random, but can be optimized using standard PyTorch techniques. Compiling to a PyTorch module like this also allows SciStanPy models to be dropped into other frameworks that rely on or extend PyTorch.

As a convenience method, SciStanPy models can be converted to PyTorch Modules and optimized in a single step using the :py:meth:`~scistanpy.model.model.Model.mle` method:

.. code-block:: python

   # Perform maximum likelihood estimation (MLE) using PyTorch backend
   mle_result = model.mle(device='cuda')

Note that if default data was provided during model initialization (as above), then no data need be provided to this method call--the registered defaults will be used automatically. Also note that, because this is a PyTorch-based method, GPU accelerattion can be used, which is a particularly useful feature for larger models.

The :py:class:`~scistanpy.model.results.mle.MLE` object returned by the :py:meth:`~scistanpy.model.model.Model.mle` method contains the optimized parameter values (accessible as instance variables or via the ``model_to_varname`` dictionary) and utility methods for evaluating the fit. Extensive details can be found in the :py:class:`MLE documentation <scistanpy.model.results.mle.MLE>`, but one to highlight is the :py:meth:`~scistanpy.model.results.mle.MLE.get_inference_obj` method, which bootstraps samples from the optimized model, providing a cheap (relative to full MCMC sampling) alternative for uncertainty quantification. It can also be particularly helpful during the early stages of model development for assessing model validity before committing to MCMC sampling, especially for large models. Indeed, evaluating model fit with the returned :py:class:`~scistanpy.model.results.mle.MLEInferenceRes` instance is as straightforward as below:

.. code-block:: python

   # Get inference object from MLE result
   mle_inference = mle_result.get_inference_obj(n_samples=1000)

   # Evaluate model fit with posterior predictive checks
   mle_inference.run_ppc()

Additional methods exposed by the :py:class:`~scistanpy.model.results.mle.MLEInferenceRes` class can be found in its :doc:`associated documentation <api/model/results/mle>`. Note that the :py:class:`~scistanpy.model.results.mle.MLEInferenceRes` class also exposes the ``inference_obj`` instance variable, which is an `arviz\.InferenceData <https://python.arviz.org/en/latest/api/generated/arviz.InferenceData.html>`_ object containing the samples drawn from the model. This can be used to interface directly with the `ArviZ ecosystem <https://python.arviz.org/en/latest/index.html>`_ for analysis of Bayesian models.

The most notable feature of SciStanPy--and, indeed, the inspiration for its name--is its ability to automatically write and execute `Stan <https://mc-stan.org/>`_ programs. As with SciStanPy's PyTorch integration, Stan functionality can be accessed at both the two levels of granularity. At the lower level, to access an object representing a Stan model, run the following:

.. code-block:: python

   # Convert to a StanModel
   stan_model = model.to_stan()

The returned :py:class:`~scistanpy.model.stan.stan_model.StanModel` object is an extension of `cmdstanpy's <https://mc-stan.org/cmdstanpy/>`_ `CmdStanModel <https://mc-stan.org/cmdstanpy/api.html#cmdstanmodel>`_ class. When instantiated, it will automatically write and compile a Stan program to sample from the model defined in SciStanPy. The resulting StanModel instance will use this Stan program for subsequent operations. Currently, the :py:meth:`~scistanpy.model.stan.stan_model.StanModel.sample` method has full support, while other `CmdStanModel <https://mc-stan.org/cmdstanpy/api.html#cmdstanmodel>`_ methods have experimental support. See the :py:class:`~scistanpy.model.stan.stan_model.StanModel` documentation for more details.

To write, compile, and sample from a SciStanPy model directly, run the below:

.. code-block:: python

   # Run Hamiltonian Monte Carlo sampling
   mcmc_result = model.mcmc()

As with the PyTorch model, if the ``default_data`` argument was provided to the parent class on initialization, that data will be used for sampling. Otherwise, it should be provided as the ``data`` kwarg to :py:meth:`~scistanpy.model.model.Model.mcmc`.

The :py:class:`~scistanpy.model.results.hmc.SampleResults` instance that is returned by the :py:meth:`~scistanpy.model.model.Model.mcmc` method contains the samples drawn during sampling. The :py:class:`~scistanpy.model.results.hmc.SampleResults` class is an extension of the :py:class:`~scistanpy.model.results.mle.MLEInferenceRes` class introduced above. It shares the same methods and properties, plus some others. Most notable among the additional methods are :py:meth:`~scistanpy.model.results.hmc.SampleResults.diagnose`, which runs diagnostic checks on MCMC samples (see `Stan's documentation <https://mc-stan.org/learn-stan/diagnostics-warnings.html>`_ for a review of the different diagnostics), :py:meth:`~scistanpy.model.results.hmc.SampleResults.plot_sample_failure_quantile_traces`, which provides an interactive dashboard for visualizing *samples* that failed diagnostic checks, and :py:meth:`~scistanpy.model.results.hmc.SampleResults.plot_variable_failure_quantile_traces`, which provides an interactive dashboard for visualizing *variables* (i.e., parameters) that failed diagnostic checks.

Additional examples demonstrating usage of the above features can be found in the :doc:`examples/index` section of the documentation.

API Documentation
=================

Essential Components
--------------------
The below table links to the documentaiton for the most commonly used components of the SciStanPy API:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Component
     - Description
   * - :py:class:`~scistanpy.model.model.Model`
     - Parent class for all SciStanPy models, handling data management, model building, and inference.
   * - :py:class:`~scistanpy.model.components.constants.Constant`
     - Describes constant values used in SciStanPy models.
   * - :py:mod:`~scistanpy.model.components.parameters`
     - Contains objects that describe latent and observed parameters in SciStanPy models as probability distributions.
   * - :py:mod:`~scistanpy.operations`
     - Contains operations for transforming parameters and building relationships between them.


All Submodules
--------------
.. toctree::

   api/index

Copyright and License
=====================

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Contributing
============

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the `Microsoft Open Source Code of Conduct <https://opensource.microsoft.com/codeofconduct/>`_. For more information see the `Code of Conduct FAQ <https://opensource.microsoft.com/codeofconduct/faq/>`_ or contact `opencode@microsoft.com <mailto:opencode@microsoft.com>`_ with any additional questions or comments.

Trademarks
============
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow `Microsoft’s Trademark & Brand Guidelines <https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks>`_. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
