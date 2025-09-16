# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Model results analysis and visualization for SciStanPy.

This submodule provides comprehensive tools for analyzing and visualizing results
from fitted SciStanPy models, supporting both maximum likelihood estimation (MLE)
and Markov Chain Monte Carlo (MCMC) inference methods. It offers specialized
classes and utilities for processing, diagnosing, and presenting model outputs
in formats suitable for scientific analysis.

The submodule is organized around two main result types, each with related analysis
capabilities:

   1. :py:class:`scistanpy.model.results.mle.MLE`, which holds results from
      calls to :py:meth:`scistanpy.model.model.Model.mle` and provides tools for MLE-based
      parameter estimation analysis.
   2. :py:class:`scistanpy.model.results.mle.MLEInferenceRes`, which holds bootstrapped
      samples of observed data and parameters for uncertainty quantification and inference
      using MLE results.
   3. :py:class:`scistanpy.model.results.hmc.SampleResults`, which holds results from
      calls to :py:meth:`scistanpy.model.model.Model.mcmc` and provides tools for MCMC-based
      sampling analysis.

Some notable analysis features available to both results classes include:
    - **Interactive Analysis**: Widget-based interfaces for exploratory analysis
    - **Scalable Processing**: Efficient handling of large datasets and complex models
    - **Comprehensive Diagnostics**: Automated detection and reporting of model issues
    - **Integration with Scientific Python**: Interoperability with ArviZ, xarray, and Dask

Results classes are designed to integrate with SciStanPy modeling workflows and
external analysis tools. For example, the ``inference_obj`` attribute shared by
both results classes provides an ArviZ InferenceData object representing the results,
allowing for further analysis using ArviZ's rich set of diagnostics and plotting
functions.

Users will not typically instantiate result classes directly. Instead, they are
returned by model fitting methods such as :py:meth:`scistanpy.model.model.Model.mle`
and :py:meth:`scistanpy.model.model.Model.mcmc`, such as below:

    >>> import scistanpy as ssp
    >>>
    >>> # MLE analysis workflow
    >>> mle_result = model.mle(data=observed_data)
    >>> mle_analysis = mle_result.get_inference_obj() # Bootstrap observed data
    >>> dashboard = mle_analysis.run_ppc()  # Interactive analysis
    >>>
    >>> # MCMC analysis workflow
    >>> mcmc_results = model.mcmc(data=observed_data, chains=4) # Sample from posterior
    >>> sample_failures, var_failures = mcmc_results.diagnose()
    >>> analyzer = mcmc_results.plot_variable_failure_quantile_traces()
"""

from scistanpy.model.results.mle import MLEInferenceRes
from scistanpy.model.results.hmc import SampleResults
