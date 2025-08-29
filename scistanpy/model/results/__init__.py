"""Model results analysis and visualization for SciStanPy.

This submodule provides comprehensive tools for analyzing and visualizing results
from fitted SciStanPy models, supporting both maximum likelihood estimation (MLE)
and Markov Chain Monte Carlo (MCMC) inference methods. It offers specialized
classes and utilities for processing, diagnosing, and presenting model outputs
in formats suitable for scientific analysis.

The submodule is organized around two main result types, each with related analysis
capabilities:

Maximum Likelihood Estimation Results (:class:`MLEInferenceRes`):
    Specialized analysis tools for MLE-based parameter estimation including:

    - **Posterior Predictive Checking**: Model validation workflows
    - **Calibration Analysis**: Quantitative assessment of model calibration quality
    - **Uncertainty Quantification**: Sampling from fitted parameter distributions
    - **Interactive Visualization**: Dashboard-style analysis interfaces
    - **ArviZ Integration**: Compatibility with Bayesian analysis workflows

MCMC Sampling Results (:class:`SampleResults`):
    Advanced diagnostic and analysis tools for HMC sampling that can perform all
    tasks of `MLEInferenceRes` as well as tasks specific to MCMC, including:

    - **Convergence Diagnostics**: R-hat, ESS, and other MCMC quality metrics
    - **Sample Quality Assessment**: Detection of divergences, energy issues, and saturation
    - **Interactive Failure Analysis**: Detailed examination of problematic parameters
    - **Memory-Efficient Processing**: Support for large datasets with Dask integration
    - **Comprehensive Reporting**: Automated diagnostic summaries and recommendations

Key Features:
    - **Unified Interface**: Consistent API across different inference methods
    - **Interactive Analysis**: Widget-based interfaces for exploratory analysis
    - **Scalable Processing**: Efficient handling of large datasets and complex models
    - **Comprehensive Diagnostics**: Automated detection and reporting of model issues
    - **Integration with Scientific Python**: Interoperability with ArviZ, xarray, and Dask

Data Formats and Persistence:
    The submodule supports multiple data formats for efficient storage and sharing:

    - **NetCDF**: Self-describing, efficient storage for large datasets
    - **ArviZ InferenceData**: Standardized format for Bayesian analysis workflows
    - **CSV Integration**: Backward compatibility with Stan's native output format
    - **Memory Mapping**: Efficient access to large datasets without full loading

Workflow Integration:
    Results classes are designed to integrate with SciStanPy modeling workflows
    and external analysis tools:

    - Automatic creation from model fitting methods
    - Direct loading from saved results files
    - Export capabilities for downstream analysis
    - Integration with visualization and reporting pipelines

Performance Considerations:
    - Lazy loading and chunked processing for memory efficiency
    - Dask integration for parallel and out-of-core computation
    - Optimized data structures for fast diagnostic computation
    - Caching strategies to avoid redundant calculations

Example Usage:
    >>> import scistanpy as ssp
    >>>
    >>> # MLE analysis workflow
    >>> mle_result = model.mle(data=observed_data)
    >>> mle_analysis = mle_result.get_inference_obj()
    >>> dashboard = mle_analysis.run_ppc()  # Interactive analysis
    >>>
    >>> # MCMC analysis workflow
    >>> mcmc_results = model.mcmc(data=observed_data, chains=4)
    >>> sample_failures, var_failures = mcmc_results.diagnose()
    >>> analyzer = mcmc_results.plot_variable_failure_quantile_traces()

See Also:
    - :mod:`scistanpy.plotting`: Visualization utilities used by results classes
    - :mod:`scistanpy.model`: Model construction and fitting methods
    - :mod:`arviz`: External library for Bayesian analysis workflows
"""

from scistanpy.model.results.mle import MLEInferenceRes
from scistanpy.model.results.hmc import SampleResults
