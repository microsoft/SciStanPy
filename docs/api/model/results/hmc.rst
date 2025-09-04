Hamiltonian Monte Carlo Results API Reference
==============================================

This reference covers the analysis and diagnostic tools for Hamiltonian Monte Carlo sampling results in SciStanPy.

HMC Results Module
------------------

.. automodule:: scistanpy.model.results.hmc
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Core HMC Analysis Classes
-------------------------

Sample Results Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.results.hmc.SampleResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Comprehensive MCMC Diagnostics:**

   The SampleResults class provides complete diagnostic capabilities for HMC sampling:

   .. code-block:: python

      import scistanpy as ssp

      # Get MCMC results
      mcmc_results = model.mcmc(data=observed_data, chains=4, iter_sampling=2000)

      # Complete diagnostic pipeline
      sample_failures, var_failures = mcmc_results.diagnose()

      # Interactive analysis of problematic variables
      analyzer = mcmc_results.plot_variable_failure_quantile_traces()

      # Detailed calibration assessment
      calibration_analysis = mcmc_results.check_calibration()

Variable Failure Analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.results.hmc.VariableAnalyzer
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Interactive Diagnostic Analysis:**

   .. code-block:: python

      # Create interactive analyzer for failed variables
      analyzer = mcmc_results.plot_variable_failure_quantile_traces(
          width=1000,
          height=500,
          plot_quantiles=True  # Show quantiles vs raw values
      )

      # Analyzer provides:
      # - Variable selection widgets
      # - Metric selection (r_hat, ess_bulk, ess_tail)
      # - Index selection for multi-dimensional parameters
      # - Real-time trace plot updates

Efficient Data Processing
------------------------

CSV to NetCDF Conversion
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.results.hmc.CmdStanMCMCToNetCDFConverter
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **High-Performance Data Conversion:**

   .. code-block:: python

      # Direct conversion from CSV to NetCDF
      netcdf_path = cmdstan_csv_to_netcdf(
          path='model_output_*.csv',
          model=model,
          precision='single',    # Memory optimization
          mib_per_chunk=128     # Control chunk sizes
      )

      # Load converted results
      results = SampleResults.from_disk(netcdf_path, use_dask=True)

.. autofunction:: scistanpy.model.results.hmc.cmdstan_csv_to_netcdf
   :noindex:

   **Batch Conversion Workflow:**

   .. code-block:: python

      # Convert large datasets efficiently
      netcdf_file = cmdstan_csv_to_netcdf(
          path=cmdstan_fit,
          model=scistanpy_model,
          data=observed_data,
          output_filename='large_model_results.nc',
          precision='single',
          mib_per_chunk=64
      )

Utility Functions
~~~~~~~~~~~~~~~~

.. autofunction:: scistanpy.model.results.hmc.fit_from_csv_noload
   :noindex:

.. autofunction:: scistanpy.model.results.hmc.dask_enabled_summary_stats
   :noindex:

.. autofunction:: scistanpy.model.results.hmc.dask_enabled_diagnostics
   :noindex:

.. autofunction:: scistanpy.model.results.hmc._symmetrize_quantiles
   :noindex:

Comprehensive Diagnostic Framework
----------------------------------

Complete Diagnostic Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Full diagnostic workflow
   results = model.mcmc(data=data, chains=4, iter_sampling=2000)

   # Step 1: Calculate all diagnostic metrics
   diagnostics = results.calculate_diagnostics()

   # Step 2: Evaluate sample-level issues
   sample_tests = results.evaluate_sample_stats(
       ebfmi_thresh=0.2,           # Energy threshold
       max_tree_depth=None         # Uses model default
   )

   # Step 3: Evaluate variable-level issues
   var_tests = results.evaluate_variable_diagnostic_stats(
       r_hat_thresh=1.01,          # Convergence threshold
       ess_thresh=100              # ESS per chain threshold
   )

   # Step 4: Comprehensive failure analysis
   sample_failures, var_failures = results.identify_failed_diagnostics()

Diagnostic Test Categories
~~~~~~~~~~~~~~~~~~~~~~~~~

**Sample-Level Diagnostics:**

.. code-block:: python

   # Sample diagnostic tests
   sample_tests = results.evaluate_sample_stats()

   # Tests performed:
   # - low_ebfmi: Energy-based fraction of missing information
   # - max_tree_depth_reached: Tree depth saturation
   # - diverged: Hamiltonian divergences

   # Access specific test results
   diverged_samples = sample_tests.diverged.sum().item()
   energy_problems = sample_tests.low_ebfmi.sum().item()
   depth_issues = sample_tests.max_tree_depth_reached.sum().item()

**Variable-Level Diagnostics:**

.. code-block:: python

   # Variable diagnostic tests
   var_tests = results.evaluate_variable_diagnostic_stats()

   # Tests performed:
   # - r_hat: Split R-hat convergence diagnostic
   # - ess_bulk: Bulk effective sample size
   # - ess_tail: Tail effective sample size

   # Identify problematic variables
   failed_convergence = var_tests.sel(metric='r_hat')
   poor_mixing = var_tests.sel(metric='ess_bulk')

Advanced Visualization Tools
---------------------------

Failure Pattern Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Visualize sample failure patterns
   sample_failure_plots = results.plot_sample_failure_quantile_traces(
       display=False,
       width=800,
       height=600
   )

   # Available plots:
   # - Divergence quantile traces
   # - Energy failure patterns
   # - Tree depth saturation analysis

   # Each plot shows:
   # - Parameter fraction on x-axis (sorted by typical failure quantile)
   # - Quantile of failed samples relative to passing samples
   # - Individual failure traces and typical patterns

Interactive Variable Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Interactive analysis of problematic variables
   variable_analyzer = results.plot_variable_failure_quantile_traces(
       plot_quantiles=True,     # Show quantiles vs raw values
       width=1000,
       height=600
   )

   # Interactive features:
   # - Variable dropdown: Select from variables that failed tests
   # - Metric dropdown: Choose diagnostic metric (r_hat, ess_bulk, ess_tail)
   # - Index dropdown: For multi-dimensional parameters
   # - Real-time plot updates with chain-specific coloring

Memory-Efficient Processing
--------------------------

Dask Integration
~~~~~~~~~~~~~~

.. code-block:: python

   # Enable Dask for large datasets
   large_results = SampleResults.from_disk(
       'large_model.nc',
       use_dask=True  # Enable out-of-core computation
   )

   # Compute statistics efficiently
   summary_stats = large_results.calculate_summaries(kind='stats')
   diagnostics = large_results.calculate_diagnostics()

   # Dask automatically handles:
   # - Chunked computation
   # - Memory management
   # - Parallel processing

Chunking Strategies
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Convert with optimal chunking
   netcdf_path = cmdstan_csv_to_netcdf(
       path=csv_files,
       model=model,
       precision='single',        # Reduce memory footprint
       mib_per_chunk=64          # Control chunk size
   )

   # Chunking optimizes:
   # - Memory usage during conversion
   # - Subsequent analysis performance
   # - Parallel processing efficiency

Scientific Workflow Integration
------------------------------

Model Validation Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def validate_mcmc_model(model, data, n_chains=4, n_samples=2000):
       """Complete MCMC model validation workflow."""

       # 1. Sample from model
       results = model.mcmc(
           data=data,
           chains=n_chains,
           iter_sampling=n_samples
       )

       # 2. Run comprehensive diagnostics
       sample_failures, var_failures = results.diagnose()

       # 3. Check for critical issues
       critical_issues = []

       # Check for excessive divergences
       n_diverged = sum(len(fails[0]) for fails in sample_failures.values())
       if n_diverged > 0.01 * n_chains * n_samples:  # >1% divergences
           critical_issues.append(f"High divergence rate: {n_diverged}")

       # Check for convergence failures
       rhat_failures = len(var_failures.get('r_hat', {}))
       if rhat_failures > 0:
           critical_issues.append(f"Convergence issues: {rhat_failures} variables")

       # 4. Generate diagnostic report
       if critical_issues:
           print("⚠️  Critical Issues Detected:")
           for issue in critical_issues:
               print(f"   - {issue}")

           # Interactive analysis for problem diagnosis
           analyzer = results.plot_variable_failure_quantile_traces()
           return results, analyzer
       else:
           print("✅ Model passed all diagnostic tests")
           return results, None

Simulation-Based Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Parameter recovery study
   def parameter_recovery_study(model, true_params, n_sims=50):
       """Test parameter recovery with known true values."""

       recovery_results = []

       for sim in range(n_sims):
           # Generate synthetic data
           sim_data = model.simulate_data(true_params)

           # Fit model
           mcmc_results = model.mcmc(data=sim_data, chains=4, iter_sampling=1000)

           # Check diagnostics
           sample_fails, var_fails = mcmc_results.diagnose(silent=True)

           # Compute recovery metrics
           posterior_summary = mcmc_results.calculate_summaries()

           recovery_metrics = {}
           for param, true_val in true_params.items():
               post_mean = posterior_summary.sel(metric='mean')[param].item()
               post_std = posterior_summary.sel(metric='sd')[param].item()

               recovery_metrics[param] = {
                   'bias': post_mean - true_val,
                   'relative_bias': (post_mean - true_val) / true_val,
                   'coverage': abs(post_mean - true_val) < 2 * post_std
               }

           recovery_results.append({
               'simulation': sim,
               'diagnostics_passed': len(var_fails) == 0,
               'recovery_metrics': recovery_metrics
           })

       return recovery_results

Performance Optimization
-----------------------

Loading Strategies
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Flexible loading for different use cases

   # Fast loading without CSV metadata (limited functionality)
   results_fast = SampleResults.from_disk(
       'model_results.nc',
       skip_fit=True,      # Skip CSV loading
       use_dask=False      # In-memory processing
   )

   # Full loading with CSV metadata (complete functionality)
   results_full = SampleResults.from_disk(
       'model_results.nc',
       csv_files='model_output_*.csv',  # Explicit CSV files
       use_dask=True                    # Out-of-core processing
   )

   # Auto-detection of CSV files
   results_auto = SampleResults.from_disk(
       'model_results.nc'  # Automatically finds matching CSV files
   )

Precision Management
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Optimize storage precision based on needs

   # High precision for critical analysis
   netcdf_path = cmdstan_csv_to_netcdf(
       path=csv_files,
       model=model,
       precision='double'    # Full double precision
   )

   # Memory-optimized for large datasets
   netcdf_path = cmdstan_csv_to_netcdf(
       path=csv_files,
       model=model,
       precision='single',   # Reduced memory usage
       mib_per_chunk=32     # Small chunks for memory efficiency
   )

Integration with External Tools
------------------------------

ArviZ Ecosystem
~~~~~~~~~~~~~~

.. code-block:: python

   # Seamless ArviZ integration (inherits from MLEInferenceRes)
   import arviz as az

   # Use ArviZ plotting functions
   az.plot_trace(results.inference_obj)
   az.plot_rank(results.inference_obj)
   az.plot_energy(results.inference_obj)

   # ArviZ diagnostics
   az.rhat(results.inference_obj)
   az.ess(results.inference_obj)
   az.mcse(results.inference_obj)

   # Model comparison
   az.compare({
       'model1': results1.inference_obj,
       'model2': results2.inference_obj
   })

Stan Ecosystem Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Work with raw Stan outputs
   from cmdstanpy import CmdStanModel

   # Convert Stan results to SciStanPy format
   stan_fit = cmdstan_model.sample(data=stan_data)

   # Convert to SciStanPy results
   netcdf_path = cmdstan_csv_to_netcdf(
       path=stan_fit,
       model=scistanpy_model
   )
   results = SampleResults.from_disk(netcdf_path)

Best Practices
-------------

1. **Always run complete diagnostics** before interpreting MCMC results
2. **Use interactive analyzers** to understand the nature of sampling problems
3. **Convert to NetCDF format** for efficient storage and analysis
4. **Enable Dask for large datasets** to prevent memory issues
5. **Monitor convergence metrics** (R-hat < 1.01, ESS > 100 per chain)
6. **Investigate divergences immediately** as they indicate model problems
7. **Use simulation studies** to validate model implementation
8. **Save diagnostic results** for reproducible analysis workflows

The HMC results framework provides comprehensive tools for diagnosing and analyzing MCMC sampling results, enabling robust Bayesian inference with clear identification of potential issues and their solutions.
