Plotting API Reference
======================

.. automodule:: scistanpy.plotting.plotting
   :undoc-members:
   :show-inheritance:

DataFrame Construction
----------------------
The below functions are used to construct DataFrames used by plotting functions in this module.

.. autofunction:: scistanpy.plotting.plotting.aggregate_data

.. autofunction:: scistanpy.plotting.plotting.build_plotting_df

Distribution Visualization
--------------------------
The below functions are used to visualize distributions, particularly for samples from parameters in Bayesian models.

.. autofunction:: scistanpy.plotting.plotting.plot_ecdf_kde

.. autofunction:: scistanpy.plotting.plotting.plot_ecdf_violin

.. autofunction:: scistanpy.plotting.plotting.plot_relationship

.. autofunction:: scistanpy.plotting.plotting.choose_plotting_function

.. autofunction:: scistanpy.plotting.plotting.plot_distribution


Model Fit Analysis
------------------
The below functions are used to analyze model fit and calibration. They are used under the hood by SciStanPy's :doc:`results <../model/results/index>` objects.


.. autofunction:: scistanpy.plotting.plotting.calculate_relative_quantiles

.. autofunction:: scistanpy.plotting.plotting.plot_calibration

.. autofunction:: scistanpy.plotting.plotting.quantile_plot

Utility Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: scistanpy.plotting.plotting.hexgrid_with_mean

.. autofunction:: scistanpy.plotting.plotting.allow_interactive