Maximum Likelihood Estimation Results API Reference
===================================================

This reference covers the analysis and visualization tools for Maximum Likelihood Estimation results in SciStanPy.

MLE Results Module
------------------

.. automodule:: scistanpy.model.results.mle
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Core MLE Analysis Classes
-------------------------

MLE Inference Results
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.results.mle.MLEInferenceRes
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Comprehensive MLE Analysis:**

   The MLEInferenceRes class provides a complete analysis framework for MLE results:

   .. code-block:: python

      import scistanpy as ssp
      import numpy as np

      # Get MLE results
      mle_result = model.mle(data=observed_data)

      # Create inference analysis object
      mle_analysis = mle_result.get_inference_obj()

      # Run comprehensive posterior predictive checking
      dashboard = mle_analysis.run_ppc()

      # Save results for later analysis
      mle_analysis.save_netcdf('mle_analysis.nc')

   **Key Analysis Methods:**

   .. code-block:: python

      # Calculate summary statistics
      stats = mle_analysis.calculate_summaries(kind='stats')

      # Check model calibration
      cal_plots, deviances = mle_analysis.check_calibration(
          return_deviance=True, display=False
      )

      # Visualize posterior predictive samples
      pp_plots = mle_analysis.plot_posterior_predictive_samples(
          quantiles=(0.025, 0.25, 0.5, 0.75, 0.975)
      )

Individual Parameter Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.results.mle.MLEParam
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Parameter-Level Analysis:**

   .. code-block:: python

      # Access individual parameter results
      mu_param = mle_result.mu  # Direct attribute access

      # Sample from fitted distribution
      samples = mu_param.draw(1000, seed=42)

      # Access fitted distribution
      fitted_dist = mu_param.distribution
      log_prob = fitted_dist.log_prob(torch.tensor([0.5]))

Complete MLE Results Container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.results.mle.MLE
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   **Complete MLE Results Management:**

   .. code-block:: python

      # Access optimization diagnostics
      loss_plot = mle_result.plot_loss_curve(logy=True)

      # Sample from all fitted distributions
      parameter_samples = mle_result.draw(n=1000, as_xarray=True)

      # Create inference object for detailed analysis
      inference_obj = mle_result.get_inference_obj(n=2000)

Diagnostic Capabilities
-----------------------

Model Calibration Assessment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Quantitative Calibration Analysis:**

.. code-block:: python

   # Comprehensive calibration assessment
   calibration_plots, deviance_metrics = mle_analysis.check_calibration(
       return_deviance=True, display=False
   )

   # Examine calibration quality
   for var_name, deviance in deviance_metrics.items():
       print(f"{var_name} calibration deviance: {deviance:.3f}")

   # Ideal calibration has deviance = 0
   # Higher values indicate calibration problems

**Calibration Interpretation:**

- **Deviance = 0**: Perfect calibration
- **Deviance < 0.1**: Good calibration
- **Deviance > 0.2**: Poor calibration requiring model revision

Posterior Predictive Checking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Multi-Modal Analysis:**

.. code-block:: python

   # Complete posterior predictive checking workflow
   ppc_dashboard = mle_analysis.run_ppc(
       quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
       use_ranks=True,
       logy_ppc_samples=False
   )

   # Individual analysis components
   pp_plots = mle_analysis.plot_posterior_predictive_samples(
       quantiles=(0.025, 0.975),
       use_ranks=False,
       logy=True  # For positive-valued data
   )

   # Quantile analysis for systematic bias detection
   quantile_plots = mle_analysis.plot_observed_quantiles(
       use_ranks=True,
       windowsize=50  # Rolling window for trend analysis
   )

**Interpretation Guidelines:**

- **Posterior Predictive Plots**: Check if observed data falls within uncertainty intervals
- **Quantile Plots**: Look for systematic deviations from horizontal trend at 0.5
- **Calibration Plots**: Assess overall model reliability

Utility Functions
-----------------

Log Transformation Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: scistanpy.model.results.mle._log10_shift
   :noindex:

   **Numerical Stability for Visualization:**

   .. code-block:: python

      # Handle negative or zero values in log plots
      data1 = np.array([-5, 0, 5, 10])
      data2 = np.array([-2, 3, 8, 15])

      log_data1, log_data2 = _log10_shift(data1, data2)
      # All values are now positive and log10-transformable

Advanced Analysis Workflows
---------------------------

Model Comparison Using MLE
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compare multiple models using MLE
   models = {
       'linear': LinearModel(x_data, y_data),
       'quadratic': QuadraticModel(x_data, y_data),
       'cubic': CubicModel(x_data, y_data)
   }

   mle_results = {}
   for name, model in models.items():
       mle_results[name] = model.mle(epochs=10000)

   # Compare final losses (lower is better)
   for name, result in mle_results.items():
       final_loss = result.losses['-log pdf/pmf'].iloc[-1]
       print(f"{name}: final loss = {final_loss:.3f}")

Uncertainty Quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Propagate parameter uncertainty through model
   parameter_samples = mle_result.draw(n=1000)

   # Use samples for uncertainty propagation
   def model_prediction(params):
       return params['intercept'] + params['slope'] * x_new

   # Monte Carlo uncertainty propagation
   predictions = []
   for i in range(1000):
       sample_params = {k: v[i] for k, v in parameter_samples.items()}
       pred = model_prediction(sample_params)
       predictions.append(pred)

   predictions = np.array(predictions)
   pred_mean = predictions.mean(axis=0)
   pred_std = predictions.std(axis=0)

Bootstrap Analysis Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Parametric bootstrap for uncertainty assessment
   def parametric_bootstrap(mle_result, n_bootstrap=1000):
       bootstrap_estimates = []

       for i in range(n_bootstrap):
           # Generate bootstrap data from fitted distributions
           bootstrap_data = {}
           for obs_name in model.observable_dict:
               fitted_dist = mle_result.model_varname_to_mle[obs_name].distribution
               bootstrap_data[obs_name] = fitted_dist.sample().detach().numpy()

           # Fit model to bootstrap data
           try:
               bootstrap_mle = model.mle(data=bootstrap_data, epochs=5000)
               bootstrap_estimates.append(bootstrap_mle.model_varname_to_mle)
           except:
               continue  # Skip failed fits

       return bootstrap_estimates

Integration with ArviZ Ecosystem
--------------------------------

**Seamless ArviZ Integration:**

.. code-block:: python

   # Create ArviZ-compatible inference object
   inference_obj = mle_result.get_inference_obj(n=2000)

   # Use ArviZ plotting functions
   import arviz as az
   az.plot_trace(inference_obj.inference_obj)
   az.plot_posterior(inference_obj.inference_obj)

   # Compute ArviZ diagnostics
   summary = az.summary(inference_obj.inference_obj)

   # Model comparison with ArviZ
   az.compare({'model1': inference_obj1.inference_obj,
               'model2': inference_obj2.inference_obj})

Interactive Dashboard Features
------------------------------

**Complete Analysis Dashboard:**

.. code-block:: python

   # Create comprehensive interactive dashboard
   dashboard = mle_analysis.run_ppc(
       use_ranks=True,           # Use rank transformation for skewed data
       square_ecdf=True,         # Square ECDF plots for better comparison
       windowsize=100,           # Rolling window for trend analysis
       quantiles=(0.05, 0.5, 0.95),  # Custom confidence intervals
       logy_ppc_samples=False,   # Linear scale for posterior predictive
       subplot_width=800,        # Custom plot dimensions
       subplot_height=600
   )

   # Dashboard components:
   # 1. Posterior predictive samples with uncertainty intervals
   # 2. Observed quantiles with systematic pattern detection
   # 3. Calibration assessment with deviance metrics

**Widget-Based Exploration:**

The dashboard provides interactive widgets for:

- **Variable Selection**: Choose which observables to analyze
- **Plot Type Selection**: Switch between different visualization modes
- **Parameter Adjustment**: Modify plot parameters in real-time
- **Export Options**: Save plots and analysis results

Memory Management
-----------------

**Efficient Large Dataset Handling:**

.. code-block:: python

   # Memory-efficient sampling for large models
   large_samples = mle_result.draw(
       n=100000,
       batch_size=1000,    # Process in batches to avoid memory overflow
       as_xarray=True      # Structured output format
   )

   # Chunked analysis for very large datasets
   inference_obj = mle_result.get_inference_obj(
       n=50000,
       batch_size=5000     # Control memory usage during generation
   )

**Storage and Persistence:**

.. code-block:: python

   # Save analysis results for later use
   inference_obj.save_netcdf('detailed_mle_analysis.nc')

   # Load saved results
   reloaded_analysis = MLEInferenceRes.from_disk('detailed_mle_analysis.nc')

   # Continue analysis from saved state
   additional_diagnostics = reloaded_analysis.check_calibration()

Best Practices
--------------

1. **Always run posterior predictive checks** before trusting MLE results
2. **Use calibration analysis** to assess model reliability quantitatively
3. **Compare multiple models** using loss trajectories and information criteria
4. **Leverage interactive dashboards** for comprehensive exploratory analysis
5. **Save analysis objects** for reproducible research workflows
6. **Use appropriate transformations** (ranks, log-scale) for visualization
7. **Monitor convergence** through loss trajectory analysis
8. **Validate with simulation studies** using known parameter values

The MLE results framework provides comprehensive tools for analyzing maximum likelihood estimates with a focus on model validation, uncertainty quantification, and scientific interpretability.
