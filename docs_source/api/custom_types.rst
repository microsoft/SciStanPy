Custom Types API Reference
==========================

.. automodule:: scistanpy.custom_types

Scalar Types
~~~~~~~~~~~~
These types combine both NumPy and PyTorch scalar types for convenience.

.. autodata:: scistanpy.custom_types.Integer

.. autodata:: scistanpy.custom_types.Float

Parameter Types
~~~~~~~~~~~~~~~
These types represent different groupings of objects that can be used within SciStanPy models.

.. autodata:: scistanpy.custom_types.SampleType

.. autodata:: scistanpy.custom_types.BaseParameterType

.. autodata:: scistanpy.custom_types.ContinuousParameterType

.. autodata:: scistanpy.custom_types.DiscreteParameterType

.. autodata:: scistanpy.custom_types.CombinableParameterType

Distribution Types
~~~~~~~~~~~~~~~~~~

.. autodata:: scistanpy.custom_types.SciStanPyDistribution

Diagnostic Types
~~~~~~~~~~~~~~~~

.. autodata:: scistanpy.custom_types.ProcessedTestRes

.. autodata:: scistanpy.custom_types.StrippedTestRes

Utility Types
~~~~~~~~~~~~~

.. autodata:: scistanpy.custom_types.IndexType

Usage Examples
--------------

**Type Checking in Functions:**

.. code-block:: python

   from scistanpy.custom_types import ContinuousParameterType

   def process_parameter(param: ContinuousParameterType) -> float:
       """Process a continuous parameter."""
       # Function accepts any continuous parameter type
       return float(param)