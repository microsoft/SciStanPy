SciStanPy API Reference
=======================

This section provides the low-level, import-level documentation for SciStanPy.
It complements the user and advanced guides by exposing public classes, functions,
and modules intended for direct programmatic use.

Scope
-----
Only implemented, importable objects are listed. Experimental or planned interfaces
are intentionally omitted.

Top-Level Package
-----------------

The root package exposes core entry points:

.. automodule:: scistanpy
   :members:
   :undoc-members:
   :show-inheritance:

Module Overview
---------------

- scistanpy.utils: General utilities (lazy import, numerical stability, chunking, Dask context, correlation helper)
- scistanpy.defaults: Centralized default configuration values
- scistanpy.model: Model class and Stan code generation pipeline
- scistanpy.model.components: Model component building blocks
- scistanpy.model.results: Result and diagnostic classes (MLEInferenceRes, SampleResults)
- scistanpy.model.stan: Stan integration, program generation helpers (indirect via model.to_stan())

Quick Navigation
----------------

.. toctree::
   :maxdepth: 1

   custom_types
   defaults
   exceptions
   model/index
   plotting/index
   operations
   utils

Usage Notes
-----------
1. Prefer high-level methods (Model.draw, Model.mle, Model.mcmc, Model.prior_predictive) before diving into internal Stan code generation utilities.
2. Utilities in scistanpy.utils are backend-agnostic; choose_module helps dispatch between NumPy and PyTorch.
3. Defaults in scistanpy.defaults can be imported and overridden in application code where appropriate.

Stability
---------
Public objects documented here follow semantic versioning. Internal helpers (prefixed with underscores or not listed) may change without notice.

Missing Something?
------------------
If an object you expect to see is absent, verify it exists in the codebase and is publicly importable, then open a documentation issue or pull request.
