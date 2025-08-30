Plotting API Reference
=======================

This reference covers the comprehensive visualization and plotting functionality in SciStanPy.

The plotting submodule provides specialized visualization tools designed for Bayesian analysis, model diagnostics, and scientific data exploration. Built on HoloViews for interactive capabilities, these functions offer both immediate insights and publication-quality outputs.

Plotting Submodule Overview
----------------------------

The plotting submodule consists of core plotting functions and specialized analysis tools:

.. toctree::
   :maxdepth: 2

   plotting
   prior_predictive

Plotting Framework
-----------------

.. automodule:: scistanpy.plotting
   :members:
   :undoc-members:
   :show-inheritance:

Core Visualization Categories
----------------------------

**Distribution Visualization**
   Functions for exploring and comparing probability distributions from model results

**Model Diagnostics**
   Specialized plots for assessing model quality, calibration, and convergence

**Relationship Analysis**
   Tools for visualizing dependencies and relationships between variables with uncertainty

**Large Dataset Support**
   Memory-efficient visualization techniques for handling large sample sets

**Interactive Features**
   Widget-based exploration and dynamic plot updates for model investigation

Key Features
-----------

**Scientific Focus:**

All plotting functions are designed specifically for scientific analysis workflows:

.. code-block:: python

   import scistanpy as ssp
   import numpy as np

   # Visualize posterior distributions
   posterior_samples = model.sample()
   distribution_plot = ssp.plotting.plot_distribution(
       posterior_samples['parameter'],
       paramname='θ'
   )

   # Model calibration assessment
   calibration_plot, deviances = ssp.plotting.plot_calibration(
       reference_samples, observed_data
   )

**Interactive Exploration:**

.. code-block:: python

   # Interactive plotting with widgets
   from scistanpy.plotting import plot_distribution
   import hvplot.interactive

   # Create interactive dataset
   interactive_data = hvplot.interactive(posterior_df)

   # Dynamic plotting with parameter selection
   dynamic_plot = plot_distribution(
       interactive_data,
       paramname=param_selector_widget
   )

**Publication Quality:**

.. code-block:: python

   # Customizable styling for publication
   plot = ssp.plotting.quantile_plot(
       x_values, reference_data, [0.025, 0.25, 0.5, 0.75, 0.975],
       observed=observed_data,
       area_kwargs={'alpha': 0.3, 'color': 'steelblue'},
       median_kwargs={'line_width': 2, 'color': 'darkblue'},
       observed_kwargs={'color': 'red', 'size': 8}
   )

Available Plotting Functions
---------------------------

**Distribution Analysis:**

.. code-block:: python

   # Empirical CDF with kernel density estimates
   ecdf_kde_plots = ssp.plotting.plot_ecdf_kde(data_df, 'parameter')

   # Combined ECDF and violin plots for group comparisons
   group_plots = ssp.plotting.plot_ecdf_violin(grouped_df, 'parameter')

   # Comprehensive distribution visualization
   dist_plot = ssp.plotting.plot_distribution(
       samples, overlay=true_values, paramname='μ'
   )

**Model Validation:**

.. code-block:: python

   # Calibration assessment
   cal_plot, deviances = ssp.plotting.plot_calibration(
       posterior_predictive, observed_data
   )

   # Quantile-quantile plots
   qq_plot = ssp.plotting.quantile_plot(
       x, reference_samples, [0.1, 0.5, 0.9],
       observed=observations, return_quantiles=True
   )

**Relationship Visualization:**

.. code-block:: python

   # Parameter relationships over conditions
   relationship_plot = ssp.plotting.plot_relationship(
       time_series_df, 'parameter', datashade=True
   )

   # Large dataset visualization with trends
   hex_plot = ssp.plotting.hexgrid_with_mean(
       x_data, y_data, mean_windowsize=100
   )

**Utility Functions:**

.. code-block:: python

   # Calculate relative quantiles for validation
   quantiles = ssp.plotting.calculate_relative_quantiles(
       reference_samples, observed_values
   )

Integration with SciStanPy Models
--------------------------------

**Direct Model Integration:**

.. code-block:: python

   # Seamless integration with model results
   class ExperimentModel(ssp.Model):
       def __init__(self, data):
           # Model definition
           self.measurement_error = ssp.parameters.LogNormal(mu=0, sigma=1)
           self.true_value = ssp.parameters.Normal(mu=0, sigma=5)
           self.likelihood = ssp.parameters.Normal(
               mu=self.true_value,
               sigma=self.measurement_error
           )
           self.likelihood.observe(data)
           super().__init__(self.likelihood)

   # Fit model and visualize
   model = ExperimentModel(experimental_data)
   results = model.sample()

   # Automatic parameter name detection
   for param_name, samples in results.items():
       plot = ssp.plotting.plot_distribution(samples, paramname=param_name)

**Results Object Integration:**

.. code-block:: python

   # MLE results integration
   mle_result = model.mle()
   mle_analysis = mle_result.get_inference_obj()

   # Built-in plotting methods use this API
   posterior_predictive_plots = mle_analysis.plot_posterior_predictive_samples()
   calibration_analysis = mle_analysis.check_calibration()

   # MCMC results integration
   mcmc_results = model.mcmc()
   diagnostic_plots = mcmc_results.plot_variable_failure_quantile_traces()

Visualization Workflows
-----------------------

**Complete Model Analysis Pipeline:**

.. code-block:: python

   def analyze_model_results(model, data):
       """Complete visualization workflow for model analysis."""

       # 1. Prior predictive checking
       prior_samples = model.prior_predictive(n=1000)
       prior_plot = ssp.plotting.plot_distribution(
           prior_samples, paramname='Prior Predictions'
       )

       # 2. Fit model
       results = model.sample()

       # 3. Posterior distribution analysis
       posterior_plots = {}
       for param, samples in results.items():
           posterior_plots[param] = ssp.plotting.plot_distribution(
               samples, paramname=param
           )

       # 4. Model validation
       posterior_predictive = model.posterior_predictive(results, n=1000)
       calibration_plot, deviances = ssp.plotting.plot_calibration(
           posterior_predictive, data
       )

       return {
           'prior': prior_plot,
           'posterior': posterior_plots,
           'calibration': calibration_plot,
           'deviances': deviances
       }

**Comparative Analysis:**

.. code-block:: python

   def compare_models(models, data):
       """Compare multiple models using visualization."""

       comparison_plots = {}

       for model_name, model in models.items():
           results = model.sample()

           # Compare parameter estimates
           for param in results.keys():
               if param not in comparison_plots:
                   comparison_plots[param] = []

               plot = ssp.plotting.plot_distribution(
                   results[param], paramname=f'{model_name}_{param}'
               )
               comparison_plots[param].append(plot)

       # Combine plots for comparison
       combined_plots = {}
       for param, plots in comparison_plots.items():
           combined_plots[param] = plots[0]
           for plot in plots[1:]:
               combined_plots[param] *= plot

       return combined_plots

Performance Considerations
-------------------------

**Large Dataset Handling:**

.. code-block:: python

   # Automatic datashading for large datasets
   large_samples = np.random.randn(1000000)

   # Efficient visualization with datashading
   efficient_plot = ssp.plotting.plot_relationship(
       large_df, 'parameter', datashade=True
   )

   # Hexagonal binning for scatter-like data
   hex_plot = ssp.plotting.hexgrid_with_mean(
       x_large, y_large, mean_windowsize=1000
   )

**Memory Optimization:**

.. code-block:: python

   # Process data in chunks for memory efficiency
   def plot_large_posterior(samples, chunk_size=10000):
       """Plot very large posterior samples efficiently."""

       n_samples = len(samples)
       plots = []

       for i in range(0, n_samples, chunk_size):
           chunk = samples[i:i+chunk_size]
           chunk_plot = ssp.plotting.plot_distribution(chunk)
           plots.append(chunk_plot)

       # Combine plots
       return plots[0] if len(plots) == 1 else plots[0] * plots[1:]

Interactive Features
-------------------

**Widget Integration:**

.. code-block:: python

   import panel as pn
   import param

   class InteractiveModelExplorer(param.Parameterized):
       parameter_name = param.ObjectSelector(
           default='mu', objects=['mu', 'sigma', 'tau']
       )
       quantile_level = param.Number(default=0.95, bounds=(0.5, 0.99))

       def view(self):
           return ssp.plotting.plot_distribution(
               model_results[self.parameter_name],
               paramname=self.parameter_name
           )

   explorer = InteractiveModelExplorer()
   pn.Row(explorer.param, explorer.view)

**Dynamic Plot Updates:**

.. code-block:: python

   # Dynamic plotting with parameter selection
   @pn.depends(parameter_selector.param.value)
   def update_plot(param_name):
       return ssp.plotting.plot_distribution(
           results[param_name], paramname=param_name
       )

Customization and Styling
-------------------------

**Plot Customization:**

.. code-block:: python

   # Extensive customization options
   custom_plot = ssp.plotting.quantile_plot(
       x_data, reference_data, [0.025, 0.975],
       observed=observed_data,
       area_kwargs={
           'alpha': 0.4,
           'color': 'steelblue',
           'line_width': 2
       },
       median_kwargs={
           'line_width': 3,
           'color': 'navy',
           'line_dash': 'dashed'
       },
       observed_kwargs={
           'color': 'red',
           'size': 10,
           'marker': 'diamond'
       }
   )

**Publication-Ready Outputs:**

.. code-block:: python

   # Configure for publication quality
   publication_plot = ssp.plotting.plot_calibration(
       reference_data, observed_data
   ).opts(
       width=800, height=600,
       fontsize={'title': 16, 'labels': 14, 'xticks': 12, 'yticks': 12},
       bgcolor='white',
       show_grid=True
   )

Best Practices
-------------

1. **Use appropriate plot types** based on data structure and analysis goals
2. **Leverage interactive features** for exploratory analysis
3. **Apply datashading** for large datasets to maintain performance
4. **Customize styling** for publication-quality outputs
5. **Validate models** using calibration and quantile plots
6. **Combine multiple visualizations** for comprehensive analysis
7. **Consider memory usage** when plotting very large sample sets

The plotting API provides a comprehensive visualization framework specifically designed for Bayesian modeling and scientific data analysis, enabling both rapid exploration and rigorous model validation.
