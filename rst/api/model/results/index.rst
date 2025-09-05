Model Results API Reference
===========================

This reference covers the comprehensive analysis and visualization tools for SciStanPy model results.

The results submodule provides specialized analysis capabilities for both Maximum Likelihood Estimation (MLE) and Hamiltonian Monte Carlo (MCMC) inference methods, with a focus on model validation, diagnostic assessment, and scientific interpretation.

Results Submodule Overview
--------------------------

The results submodule consists of two main analysis frameworks:

.. toctree::
   :maxdepth: 2

   mle
   hmc

Results Module
--------------

.. automodule:: scistanpy.model.results
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Core Analysis Frameworks
------------------------

**Maximum Likelihood Estimation Analysis**
   Comprehensive tools for analyzing MLE results including posterior predictive checking, calibration assessment, and uncertainty quantification

**Hamiltonian Monte Carlo Analysis**
   Advanced diagnostic capabilities for MCMC results including convergence assessment, sample quality evaluation, and interactive failure analysis

Unified Analysis Interface
--------------------------

**Common Analysis Capabilities:**

Both MLE and MCMC results share a common foundation providing:

.. code-block:: python

   import scistanpy as ssp

   # MLE workflow
   mle_result = model.mle(data=observed_data)
   mle_analysis = mle_result.get_inference_obj()

   # MCMC workflow
   mcmc_result = model.mcmc(data=observed_data)
   # mcmc_result is already an analysis object

   # Common analysis methods available for both:
   # - Posterior predictive checking
   # - Model calibration assessment
   # - Summary statistics computation
   # - Interactive visualization dashboards

**Specialized Capabilities:**

Each framework provides method-specific tools:

.. code-block:: python

   # MLE-specific: Optimization diagnostics
   loss_plot = mle_result.plot_loss_curve()
   fitted_distributions = mle_result.model_varname_to_mle

   # MCMC-specific: Convergence diagnostics
   sample_failures, var_failures = mcmc_result.diagnose()
   convergence_analyzer = mcmc_result.plot_variable_failure_quantile_traces()

Key Features
------------

**Comprehensive Model Validation**
   Systematic approaches to assess model adequacy and identify potential issues

**Interactive Analysis Tools**
   Widget-based interfaces for exploratory analysis and problem diagnosis

**Memory-Efficient Processing**
   Support for large datasets through chunked processing and Dask integration

**Scientific Workflow Integration**
   Designed for integration with scientific analysis pipelines and reporting

**Standards Compatibility**
   Full integration with ArviZ ecosystem and scientific Python tools

Analysis Workflow Patterns
--------------------------

Standard MLE Analysis Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Complete MLE analysis workflow
   def analyze_mle_results(model, data):
       """Standard MLE analysis pipeline."""

       # 1. Fit model
       mle_result = model.mle(data=data, epochs=10000)

       # 2. Check optimization convergence
       loss_plot = mle_result.plot_loss_curve()

       # 3. Create inference object for detailed analysis
       analysis = mle_result.get_inference_obj(n=2000)

       # 4. Run comprehensive posterior predictive checking
       ppc_dashboard = analysis.run_ppc()

       # 5. Assess model calibration quantitatively
       cal_plots, deviances = analysis.check_calibration(return_deviance=True)

       # 6. Generate summary report
       summary_stats = analysis.calculate_summaries()

       return {
           'mle_result': mle_result,
           'analysis': analysis,
           'dashboard': ppc_dashboard,
           'calibration_deviances': deviances,
           'summary': summary_stats
       }

Standard MCMC Analysis Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Complete MCMC analysis workflow
   def analyze_mcmc_results(model, data):
       """Standard MCMC analysis pipeline."""

       # 1. Sample from model
       results = model.mcmc(data=data, chains=4, iter_sampling=2000)

       # 2. Run comprehensive diagnostics
       sample_failures, var_failures = results.diagnose()

       # 3. Investigate any failures
       if any(var_failures.values()) or any(sample_failures.values()):
           print("⚠️  Diagnostic issues detected")
           failure_analyzer = results.plot_variable_failure_quantile_traces()
           sample_plots = results.plot_sample_failure_quantile_traces()
       else:
           print("✅ All diagnostics passed")

       # 4. Posterior predictive checking
       ppc_dashboard = results.run_ppc()

       # 5. Model calibration assessment
       calibration_analysis = results.check_calibration()

       return {
           'results': results,
           'sample_failures': sample_failures,
           'variable_failures': var_failures,
           'ppc_dashboard': ppc_dashboard
       }

Comparative Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compare MLE vs MCMC results
   def compare_inference_methods(model, data):
       """Compare MLE and MCMC inference for the same model."""

       # Fit with both methods
       mle_result = model.mle(data=data)
       mcmc_result = model.mcmc(data=data, chains=4, iter_sampling=1000)

       # Create comparable analysis objects
       mle_analysis = mle_result.get_inference_obj(n=1000)

       # Compare summary statistics
       mle_summary = mle_analysis.calculate_summaries()
       mcmc_summary = mcmc_result.calculate_summaries()

       # Compare posterior predictive performance
       mle_cal_plots, mle_deviances = mle_analysis.check_calibration(
           return_deviance=True, display=False
       )
       mcmc_cal_plots, mcmc_deviances = mcmc_result.check_calibration(
           return_deviance=True, display=False
       )

       return {
           'mle': {'analysis': mle_analysis, 'deviances': mle_deviances},
           'mcmc': {'analysis': mcmc_result, 'deviances': mcmc_deviances},
           'comparison': {
               'calibration_comparison': {
                   var: {'mle': mle_deviances[var], 'mcmc': mcmc_deviances[var]}
                   for var in mle_deviances.keys()
               }
           }
       }

Data Management and Persistence
-------------------------------

Efficient Storage Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Efficient storage and loading patterns

   # Save MLE analysis for later use
   mle_analysis = mle_result.get_inference_obj()
   mle_analysis.save_netcdf('mle_analysis_results.nc')

   # Convert MCMC results to efficient format
   netcdf_path = cmdstan_csv_to_netcdf(
       path=mcmc_csv_files,
       model=model,
       precision='single',  # Optimize for storage
       mib_per_chunk=64
   )

   # Load results efficiently
   mle_reloaded = MLEInferenceRes.from_disk('mle_analysis_results.nc')
   mcmc_reloaded = SampleResults.from_disk(netcdf_path, use_dask=True)

Cross-Platform Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Export results for external analysis

   # Export to ArviZ format
   arviz_data = results.inference_obj
   arviz_data.to_netcdf('arviz_compatible.nc')

   # Export summary statistics to standard formats
   summary = results.calculate_summaries()
   summary.to_pandas().to_csv('model_summary.csv')

   # Export for visualization in other tools
   posterior_samples = results.inference_obj.posterior
   posterior_samples.to_pandas().to_parquet('posterior_samples.parquet')

Advanced Analysis Patterns
--------------------------

Model Selection and Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Systematic model comparison
   def model_selection_study(candidate_models, data):
       """Compare multiple candidate models systematically."""

       results = {}

       for model_name, model in candidate_models.items():
           print(f"Analyzing {model_name}...")

           # Fit with both methods for robustness
           mle_result = model.mle(data=data)
           mcmc_result = model.mcmc(data=data, chains=4, iter_sampling=1000)

           # Diagnostic checks
           sample_fails, var_fails = mcmc_result.diagnose(silent=True)
           diagnostics_passed = (len(var_fails) == 0 and
                                not any(len(fails[0]) > 0 for fails in sample_fails.values()))

           # Model performance metrics
           mle_analysis = mle_result.get_inference_obj()
           _, mle_deviances = mle_analysis.check_calibration(
               return_deviance=True, display=False
           )
           _, mcmc_deviances = mcmc_result.check_calibration(
               return_deviance=True, display=False
           )

           results[model_name] = {
               'mle_final_loss': mle_result.losses['-log pdf/pmf'].iloc[-1],
               'diagnostics_passed': diagnostics_passed,
               'mean_calibration_deviance_mle': np.mean(list(mle_deviances.values())),
               'mean_calibration_deviance_mcmc': np.mean(list(mcmc_deviances.values())),
               'mcmc_result': mcmc_result,
               'mle_result': mle_result
           }

       return results

Uncertainty Quantification Studies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Comprehensive uncertainty analysis
   def uncertainty_analysis(model, data, n_bootstrap=100):
       """Analyze parameter uncertainty through multiple approaches."""

       # 1. MCMC-based uncertainty (gold standard)
       mcmc_result = model.mcmc(data=data, chains=4, iter_sampling=2000)
       mcmc_summary = mcmc_result.calculate_summaries()

       # 2. Bootstrap uncertainty assessment
       mle_result = model.mle(data=data)

       bootstrap_estimates = []
       for i in range(n_bootstrap):
           # Parametric bootstrap
           bootstrap_data = {}
           for param_name, param_obj in mle_result.model_varname_to_mle.items():
               if model.all_model_components_dict[param_name].observable:
                   bootstrap_data[param_name] = param_obj.distribution.sample().detach().numpy()

           try:
               boot_mle = model.mle(data=bootstrap_data, epochs=5000)
               bootstrap_estimates.append(boot_mle.model_varname_to_mle)
           except:
               continue

       # 3. Compare uncertainty estimates
       uncertainty_comparison = {}
       for param in mcmc_summary.coords['variable']:
           param_name = param.item()

           # MCMC uncertainty
           mcmc_std = mcmc_summary.sel(variable=param_name, metric='sd').item()

           # Bootstrap uncertainty
           boot_values = [est[param_name].mle.item() if est[param_name].mle is not None
                         else np.nan for est in bootstrap_estimates]
           boot_std = np.nanstd(boot_values)

           uncertainty_comparison[param_name] = {
               'mcmc_std': mcmc_std,
               'bootstrap_std': boot_std,
               'ratio': boot_std / mcmc_std if mcmc_std > 0 else np.nan
           }

       return {
           'mcmc_result': mcmc_result,
           'mle_result': mle_result,
           'bootstrap_estimates': bootstrap_estimates,
           'uncertainty_comparison': uncertainty_comparison
       }

Integration with Scientific Workflows
-------------------------------------

Automated Reporting
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Generate automated analysis reports
   def generate_model_report(model, data, output_dir='./model_report/'):
       """Generate comprehensive automated model analysis report."""

       import os
       os.makedirs(output_dir, exist_ok=True)

       # Run both inference methods
       mle_result = model.mle(data=data)
       mcmc_result = model.mcmc(data=data, chains=4, iter_sampling=1000)

       # Generate analyses
       mle_analysis = mle_result.get_inference_obj()

       # Save key plots
       loss_plot = mle_result.plot_loss_curve()
       # loss_plot.save(f'{output_dir}/optimization_trajectory.html')

       # Diagnostic dashboard
       ppc_dashboard = mle_analysis.run_ppc()
       # ppc_dashboard.save(f'{output_dir}/posterior_predictive_checks.html')

       # MCMC diagnostics
       sample_fails, var_fails = mcmc_result.diagnose()

       # Summary statistics
       summary = mcmc_result.calculate_summaries()
       summary.to_pandas().to_csv(f'{output_dir}/parameter_summary.csv')

       # Generate text report
       with open(f'{output_dir}/analysis_summary.txt', 'w') as f:
           f.write("Model Analysis Summary\n")
           f.write("=====================\n\n")

           f.write(f"MLE Final Loss: {mle_result.losses['-log pdf/pmf'].iloc[-1]:.3f}\n")
           f.write(f"MCMC Chains: {mcmc_result.inference_obj.posterior.sizes['chain']}\n")
           f.write(f"MCMC Samples: {mcmc_result.inference_obj.posterior.sizes['draw']}\n\n")

           if any(var_fails.values()):
               f.write("⚠️  DIAGNOSTIC WARNINGS:\n")
               for metric, failures in var_fails.items():
                   if failures:
                       f.write(f"  - {len(failures)} variables failed {metric} test\n")
           else:
               f.write("✅ All MCMC diagnostics passed\n")

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Batch analysis for multiple datasets
   def batch_model_analysis(model, datasets, output_base_dir='./batch_analysis/'):
       """Analyze the same model across multiple datasets."""

       results = {}

       for dataset_name, data in datasets.items():
           print(f"Processing dataset: {dataset_name}")

           output_dir = f"{output_base_dir}/{dataset_name}/"
           os.makedirs(output_dir, exist_ok=True)

           try:
               # Run analysis
               mle_result = model.mle(data=data)
               mcmc_result = model.mcmc(data=data, chains=4, iter_sampling=1000)

               # Save results
               mle_analysis = mle_result.get_inference_obj()
               mle_analysis.save_netcdf(f'{output_dir}/mle_analysis.nc')

               # Convert MCMC to NetCDF
               netcdf_path = cmdstan_csv_to_netcdf(
                   path=mcmc_result.fit,
                   model=model,
                   output_filename=f'{output_dir}/mcmc_results.nc'
               )

               results[dataset_name] = {
                   'success': True,
                   'mle_path': f'{output_dir}/mle_analysis.nc',
                   'mcmc_path': netcdf_path
               }

           except Exception as e:
               print(f"Failed to process {dataset_name}: {e}")
               results[dataset_name] = {'success': False, 'error': str(e)}

       return results

Best Practices
--------------

1. **Always validate models** using posterior predictive checks before scientific interpretation
2. **Use appropriate inference methods** - MLE for speed, MCMC for uncertainty quantification
3. **Monitor diagnostics systematically** - convergence, calibration, and sample quality
4. **Leverage interactive tools** for exploratory analysis and problem identification
5. **Save analysis objects** using NetCDF format for efficient storage and sharing
6. **Document analysis workflows** for reproducible scientific research
7. **Compare inference methods** when computational resources allow
8. **Use batch processing** for systematic studies across multiple conditions

The results submodule provides a comprehensive framework for analyzing and validating Bayesian models, enabling robust scientific inference with clear diagnostic feedback and interactive exploration capabilities.
