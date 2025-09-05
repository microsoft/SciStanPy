Plotting API Reference
======================

This page documents the public plotting utilities re-exported by the
`scistanpy.plotting` package. The package provides a small collection of
helper functions for visualizing model results and diagnostic checks. The
actual function implementations and additional plotting helpers live in
internal modules; the package `__init__` exposes the primary convenience
functions listed below.

Plotting Submodule Overview
---------------------------

.. toctree::
   :maxdepth: 4

   plotting
   prior_predictive

Public API
----------

The top-level package re-exports a curated set of plotting utilities. The
following functions are available at `scistanpy.plotting`::

- calculate_relative_quantiles
- hexgrid_with_mean
- plot_calibration
- plot_distribution
- plot_ecdf_kde
- plot_ecdf_violin
- plot_relationship
- quantile_plot

The package is implemented on top of HoloViews/Datashader for interactive
and large-data workflows; those libraries are optional at import time but
are required for some functions to produce interactive or datashaded
outputs.

Module reference
----------------

.. automodule:: scistanpy.plotting
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Minimal usage examples
----------------------

Simple distribution plot using numpy samples:

.. code-block:: python

   import numpy as np
   import scistanpy.plotting as ssp_plot

   samples = np.random.normal(size=1000)
   fig = ssp_plot.plot_distribution(samples, paramname='mu')

Quantile/coverage-style plot:

.. code-block:: python

   x = np.linspace(0, 1, 100)
   qplot = ssp_plot.quantile_plot(x, reference_samples=np.random.randn(100, 1000))

Notes and guidance
------------------

- For interactive or datashaded plots, ensure HoloViews and Datashader are
  installed in the environment.
- These are convenience wrappers intended to produce publication-quality
  defaults; for advanced customization, inspect the implementations in the
  plotting submodule and compose visual elements directly.

See also
--------

- :doc:`plotting` (detailed plotting routines)
- :doc:`prior_predictive` (prior predictive plotting helpers)
