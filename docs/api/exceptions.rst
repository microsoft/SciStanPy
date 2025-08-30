Exceptions API Reference
========================

This reference covers the custom exception classes used throughout SciStanPy for error handling.

Exceptions Module
----------------

.. automodule:: scistanpy.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Exception Hierarchy
-------------------

SciStanPy defines a hierarchy of custom exception classes to provide clear error reporting:

Base Exception
~~~~~~~~~~~~~

.. autoclass:: scistanpy.exceptions.SciStanPyError
   :members:
   :undoc-members:
   :show-inheritance:

   **Usage as Base Exception:**

   .. code-block:: python

      import scistanpy as ssp

      try:
          # Any SciStanPy operation
          model = ssp.Model(likelihood)
          results = model.sample()
      except ssp.exceptions.SciStanPyError as e:
          print(f"SciStanPy error occurred: {e}")

Specific Exception Types
-----------------------

Sample Validation Errors
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.exceptions.NumpySampleError
   :members:
   :undoc-members:
   :show-inheritance:

   **Common Use Cases:**

   .. code-block:: python

      import numpy as np
      import scistanpy as ssp

      try:
          # Operations with invalid NumPy sample data
          invalid_sample = np.array([1, 2, np.nan, 4])
          # Some operation that validates the sample
      except ssp.exceptions.NumpySampleError as e:
          print(f"Invalid sample data: {e}")

Error Handling Best Practices
----------------------------

**Catching Specific Exceptions:**

.. code-block:: python

   try:
       # SciStanPy operations
       model = ssp.Model(likelihood)
       results = model.sample()
   except ssp.exceptions.NumpySampleError:
       print("Check your sample data for invalid values")
   except ssp.exceptions.SciStanPyError as e:
       print(f"General SciStanPy error: {e}")

**Exception Hierarchy Usage:**

.. code-block:: python

   # Catch all SciStanPy exceptions
   try:
       # Model operations
       pass
   except ssp.exceptions.SciStanPyError:
       # Handle any SciStanPy-related error
       pass
   except Exception:
       # Handle other exceptions
       pass

See Also
--------

- :doc:`model` - Model construction that may raise exceptions
- :doc:`parameters` - Parameter validation and error conditions
- :doc:`utils` - Utility functions that use exception handling
   except ssp.exceptions.ShapeError as e:
       print(f"Shape mismatch: {e}")
       print(f"Parameter 1 shape: {e.shape1}")
       print(f"Parameter 2 shape: {e.shape2}")
       print(f"Suggested fix: {e.suggestion}")
