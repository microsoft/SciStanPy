Stan Submodule API Reference
============================
.. automodule:: scistanpy.model.stan
   :undoc-members:
   :show-inheritance:


This submodule is organized into two main components:

1. A series of :doc:`stanfunctions <stan_functions>` files that define custom Stan functions for use in SciStanPy models.
2. The :py:mod:`~scistanpy.model.stan.stan_model` module that handles the conversion of SciStanPy models into Stan code, manages compilation, and provides interfaces for running inference.

There is also a single global configuration variable defined at the submodule level:

.. autodata:: scistanpy.model.stan.STAN_INCLUDE_PATHS
   :no-value:


.. toctree::
   :maxdepth: 1
   :hidden:

   stan_functions
   stan_model