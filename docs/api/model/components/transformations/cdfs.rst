CDFs API Reference
==================

This page documents the CDF-style transformation classes implemented in `scistanpy.model.components.transformations.cdfs`.

.. automodule:: scistanpy.model.components.transformations.cdfs
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The module implements a family of transformation classes that compute cumulative-distribution-related operations for SciStanPy parameters. These classes are backend-aware: they dispatch to NumPy/SciPy or PyTorch implementations when running locally, and they generate Stan expressions when the model is translated to Stan.

Primary classes
---------------

- CDFLike
  - Abstract base class providing shared validation, backend dispatch, and Stan code generation support. Not intended to be instantiated directly.

- CDF
  - Standard cumulative distribution function (P(X ≤ x)).

- SurvivalFunction
  - Complementary CDF (P(X > x) = 1 − F(x)).

- LogCDF
  - Logarithmic CDF (log P(X ≤ x)) for numerical stability.

- LogSurvivalFunction
  - Logarithmic survival function (log P(X > x)).

Usage notes
-----------

- These transformations are normally accessed via parameter instances (for example `my_param.cdf(x=...)`) rather than instantiated directly.
- The classes provide methods used both at runtime (run_np_torch_op) and at compile-time (write_stan_operation) so they integrate cleanly into SciStanPy's multi-backend workflow.
- The module validates provided distribution parameters and raises informative errors if required parameters are missing or extra parameters are supplied.

Minimal examples
----------------

Accessing a CDF via a parameter instance (typical):

.. code-block:: python

   import scistanpy as ssp

   # Define a parameter in a model
   # (this example is illustrative — typical use is inside a Model subclass)
   normal_param = ssp.parameters.Normal(mu=0, sigma=1)

   # Compute CDF values (NumPy/SciPy backend)
   vals = normal_param.cdf(x=[-1.0, 0.0, 1.0])

Stan code generation
--------------------

When generating Stan code, the transformation classes produce the appropriate Stan function call (for example `normal_cdf`/`normal_lcdf`) with parameters ordered according to the distribution's Stan signature.

See also
--------

- :doc:`../parameters` for how Parameter types expose these transformation helpers
- :doc:`transformed_parameters` for the transformation base classes and integration details
