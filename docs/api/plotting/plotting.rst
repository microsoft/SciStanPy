Plotting API Reference
=======================

This reference covers the visualization and plotting functionality in SciStanPy.

Plotting Module
---------------

.. automodule:: scistanpy.plotting.plotting
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Data Preparation Functions
---------------------------

DataFrame Construction
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: scistanpy.plotting.plotting.build_plotting_df
   :noindex:

   **Example Usage:**

   .. code-block:: python

      import numpy as np
      import scistanpy as ssp

      # Transform sample arrays into plotting-ready DataFrames
      samples = np.random.randn(100, 50, 10)  # 100 traces, 50 time points, 10 params
      df = ssp.plotting.plotting.build_plotting_df(
          samples, 'measurement', independent_dim=1
      )

Data Aggregation
~~~~~~~~~~~~~~~

.. autofunction:: scistanpy.plotting.plotting.aggregate_data
   :noindex:

   **Example Usage:**

   .. code-block:: python

      # Flatten multi-dimensional data for plotting
      data = np.random.randn(10, 5, 3)
      flat = ssp.plotting.plotting.aggregate_data(data)  # Shape: (150,)

      # Preserve specific dimension
      agg = ssp.plotting.plotting.aggregate_data(data, independent_dim=2)  # Shape: (50, 3)

Distribution Visualization
-------------------------

Main Distribution Plotting
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: scistanpy.plotting.plotting.plot_distribution
   :noindex:

   **Example Usage:**

   .. code-block:: python

      # Simple distribution plot
      plot = ssp.plotting.plotting.plot_distribution(
          posterior_samples, paramname='mu'
      )

      # With ground truth overlay
      plot = ssp.plotting.plotting.plot_distribution(
          samples, overlay=true_values, paramname='sigma'
      )

ECDF and KDE Plots
~~~~~~~~~~~~~~~~~

.. autofunction:: scistanpy.plotting.plotting.plot_ecdf_kde
   :noindex:

   **Example Usage:**

   .. code-block:: python

      import pandas as pd

      df = pd.DataFrame({'param': np.random.normal(0, 1, 1000)})
      plots = ssp.plotting.plotting.plot_ecdf_kde(df, 'param')
      # plots[0] is KDE, plots[1] is ECDF

Multi-Group Visualization
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: scistanpy.plotting.plotting.plot_ecdf_violin
   :noindex:

   **Example Usage:**

   .. code-block:: python

      # DataFrame with 'param' values and 'Independent Label' grouping
      plots = ssp.plotting.plotting.plot_ecdf_violin(grouped_df, 'param')

Relationship Plots
~~~~~~~~~~~~~~~~~

.. autofunction:: scistanpy.plotting.plotting.plot_relationship
   :noindex:

   **Example Usage:**

   .. code-block:: python

      # Plot parameter evolution over time/conditions
      plot = ssp.plotting.plotting.plot_relationship(
          time_series_df, 'param', datashade=True
      )

Model Validation Plots
----------------------

Calibration Analysis
~~~~~~~~~~~~~~~~~~~

.. autofunction:: scistanpy.plotting.plotting.plot_calibration
   :noindex:

   **Example Usage:**

   .. code-block:: python

      # Assess model calibration
      ref_data = posterior_predictive_samples  # Shape: (1000, 100)
      obs_data = actual_observations          # Shape: (10, 100)
      plot, deviances = ssp.plotting.plotting.plot_calibration(ref_data, obs_data)
      print(f"Mean deviance: {deviances.mean():.3f}")

.. autofunction:: scistanpy.plotting.plotting.calculate_relative_quantiles
   :noindex:

   **Example Usage:**

   .. code-block:: python

      # Calculate quantiles for calibration
      ref = np.random.normal(0, 1, (1000, 10))  # 1000 samples, 10 features
      obs = np.random.normal(0.5, 1, (5, 10))   # 5 observations, 10 features
      quantiles = ssp.plotting.plotting.calculate_relative_quantiles(ref, obs)

Quantile Plots
~~~~~~~~~~~~~

.. autofunction:: scistanpy.plotting.plotting.quantile_plot
   :noindex:

   **Example Usage:**

   .. code-block:: python

      # Create uncertainty bands around predictions
      x = np.linspace(0, 10, 100)
      ref = np.random.normal(np.sin(x), 0.1, (1000, 100))
      obs = np.sin(x) + 0.05 * np.random.randn(100)
      plot = ssp.plotting.plotting.quantile_plot(
          x, ref, [0.025, 0.25], observed=obs
      )

Large Dataset Visualization
--------------------------

Hexagonal Binning
~~~~~~~~~~~~~~~~

.. autofunction:: scistanpy.plotting.plotting.hexgrid_with_mean
   :noindex:

   **Example Usage:**

   .. code-block:: python

      # Large dataset with trend visualization
      x = np.random.randn(10000)
      y = 2*x + 0.5*np.random.randn(10000)
      plot = ssp.plotting.plotting.hexgrid_with_mean(x, y, mean_windowsize=200)

Utility Functions
----------------

Interactive Support
~~~~~~~~~~~~~~~~~~~

.. autofunction:: scistanpy.plotting.plotting.allow_interactive
   :noindex:

   **Usage as Decorator:**

   .. code-block:: python

      @ssp.plotting.plotting.allow_interactive
      def my_plot(df, param):
          return df.hvplot.line(y=param)

Plot Function Selection
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: scistanpy.plotting.plotting.choose_plotting_function
   :noindex:

   **Example Usage:**

   .. code-block:: python

      # Select appropriate plotting function based on data structure
      plotter = ssp.plotting.plotting.choose_plotting_function(None, None)  # ECDF/KDE
      plotter = ssp.plotting.plotting.choose_plotting_function(1, time_labels)  # Relationship

Integration with SciStanPy
--------------------------

The plotting module integrates seamlessly with SciStanPy model components:

**Backend Compatibility:**

.. code-block:: python

   # Works with both NumPy arrays and PyTorch tensors
   pytorch_samples = torch.randn(1000, 50)
   plot = ssp.plotting.plotting.plot_distribution(pytorch_samples)

**Model Integration:**

.. code-block:: python

   # Direct integration with model results
   model = ssp.Model(likelihood)
   results = model.sample()

   # Visualize posterior distributions
   for param_name in results.keys():
       plot = ssp.plotting.plotting.plot_distribution(
           results[param_name], paramname=param_name
       )

**Interactive Features:**

The plotting system supports both static and interactive visualizations:

- **Static plots**: Return HoloViews objects for immediate display
- **Interactive plots**: Support widgets and dynamic exploration
- **Jupyter integration**: Optimized for notebook environments

Best Practices
--------------

1. **Use appropriate plot types** based on data dimensionality
2. **Leverage interactive features** for exploratory analysis
3. **Customize styling** using kwargs for publication-quality plots
4. **Consider data size** when choosing between datashading and standard plots
5. **Validate models** using calibration and quantile plots
6. **Combine multiple visualizations** for comprehensive analysis

This plotting API provides comprehensive visualization tools specifically designed for Bayesian modeling and scientific data analysis workflows.
