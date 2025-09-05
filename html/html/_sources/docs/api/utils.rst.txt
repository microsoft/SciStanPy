Utils API Reference
===================

This reference documents the currently implemented utility functions and helper components in SciStanPy.

Utils Module
------------

.. automodule:: scistanpy.utils
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Lazy Import System
------------------

Deferred importing to improve startup performance.

.. autofunction:: scistanpy.utils.lazy_import
   :noindex:

.. autoclass:: scistanpy.utils.LazyObjectProxy
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: scistanpy.utils.lazy_import_from
   :noindex:

Example Usage:

.. code-block:: python

   # Lazy import of a module
   np_mod = ssp.utils.lazy_import("numpy")
   arr = np_mod.array([1, 2, 3])

   # Lazy import of an object
   array = ssp.utils.lazy_import_from("numpy", "array")
   arr2 = array([4, 5, 6])

Backend Selection
-----------------

Automatic dispatch between NumPy and PyTorch.

.. autofunction:: scistanpy.utils.choose_module
   :noindex:

Example Usage:

.. code-block:: python

   import torch
   t = torch.tensor([1.0, 2.0])
   mod = ssp.utils.choose_module(t)   # torch
   out = mod.exp(t)

Numerical Stability
-------------------

Stable sigmoid avoiding overflow.

.. autofunction:: scistanpy.utils.stable_sigmoid
   :noindex:

Example Usage:

.. code-block:: python

   import numpy as np
   x = np.array([-1000, 0, 1000])
   y = ssp.utils.stable_sigmoid(x)

Array Chunking
--------------

Memory-aware Dask-friendly chunk shape calculation.

.. autofunction:: scistanpy.utils.get_chunk_shape
   :noindex:

Example Usage:

.. code-block:: python

   shape = ssp.utils.get_chunk_shape(
       (1000, 2000, 100),
       "double",
       mib_per_chunk=64,
       frozen_dims=(2,)
   )

Dask Integration
----------------

Enable / disable ArviZ Dask execution.

.. autoclass:: scistanpy.utils.az_dask
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage:

.. code-block:: python

   with ssp.utils.az_dask("parallelized", [float]):
       # ArviZ functions now use Dask (where supported)
       pass

Statistical Computing
---------------------

Pairwise Spearman rank correlation (row vs row) with NaN-aware optimization.

.. autofunction:: scistanpy.utils.faster_autocorrelation
   :noindex:

Example Usage:

.. code-block:: python

   import numpy as np
   x = np.random.randn(100, 50)
   x[x < -1] = np.nan
   rhos = ssp.utils.faster_autocorrelation(x)  # shape (100, 100)

Available Utilities Summary
---------------------------

- lazy_import / lazy_import_from / LazyObjectProxy: deferred imports
- choose_module: backend selection (NumPy vs PyTorch)
- stable_sigmoid: numerically stable logistic function
- get_chunk_shape: memory-constrained Dask chunk shape helper
- az_dask: context-managed ArviZ+Dask integration
- faster_autocorrelation: pairwise Spearman correlation matrix with NaN handling