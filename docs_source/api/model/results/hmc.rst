Hamiltonian Monte Carlo Results API Reference
=============================================

.. automodule:: scistanpy.model.results.hmc
   :undoc-members:
   :show-inheritance:

The main class that users will interact with is
:py:class:`~scistanpy.model.results.hmc.SampleResults`. Other classes in this module
provide supporting functionality for diagnostics, visualization, and data conversion.

Sample Results Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.results.hmc.SampleResults
   :members:
   :undoc-members:
   :show-inheritance:

Variable Failure Analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~

Users will not typically instantiate this class directly. It is the return type of :py:meth:`~scistanpy.model.results.hmc.SampleResults.plot_variable_failure_quantile_traces` and provides the interactive analysis interface.

.. autoclass:: scistanpy.model.results.hmc.VariableAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

CSV to NetCDF Conversion
~~~~~~~~~~~~~~~~~~~~~~~~

Stan results are output in CSV format, which is quite inefficient for large datasets. The following utilities are responsible for converting these CSV files into the more efficient NetCDF file format. Once in NetCDF format, it is easy to manipulate samples using packages such as xarray, dask, and arviz.

.. autofunction:: scistanpy.model.results.hmc.cmdstan_csv_to_netcdf

.. autoclass:: scistanpy.model.results.hmc.CmdStanMCMCToNetCDFConverter
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
~~~~~~~~~~~~~~~~~
The following utility functions are used internally by the other classes and functions in this module and will not typically be called directly by users.

.. autofunction:: scistanpy.model.results.hmc.dask_enabled_summary_stats

.. autofunction:: scistanpy.model.results.hmc.dask_enabled_diagnostics

.. autofunction:: scistanpy.model.results.hmc.fit_from_csv_noload