Custom Types API Reference
==========================

This reference covers the type definitions and type aliases used throughout SciStanPy.

Custom Types Module
-------------------

.. automodule:: scistanpy.custom_types

   :members:
   :undoc-members:
   :show-inheritance:

Scalar Types
~~~~~~~~~~~~

.. autodata:: scistanpy.custom_types.Integer
   :annotation:

   Type alias for integer values, accepting both Python int and NumPy integer types.

.. autodata:: scistanpy.custom_types.Float
   :annotation:

   Type alias for floating-point values, accepting both Python float and NumPy floating types.

Parameter Types
~~~~~~~~~~~~~~~

.. autodata:: scistanpy.custom_types.SampleType
   :annotation:

   Type alias for values that can be sampled from distributions.

.. autodata:: scistanpy.custom_types.BaseParameterType
   :annotation:

   Base parameter types including transformed parameters and constants.

.. autodata:: scistanpy.custom_types.ContinuousParameterType
   :annotation:

   Type alias for continuous-valued parameters.

.. autodata:: scistanpy.custom_types.DiscreteParameterType
   :annotation:

   Type alias for discrete-valued parameters.

.. autodata:: scistanpy.custom_types.CombinableParameterType
   :annotation:

   Type alias for parameters that can be combined in operations.

Distribution Types
~~~~~~~~~~~~~~~~~~

.. autodata:: scistanpy.custom_types.SciStanPyDistribution
   :annotation:

   Type alias for PyTorch-compatible distributions used in SciStanPy.

Diagnostic Types
~~~~~~~~~~~~~~~~

.. autodata:: scistanpy.custom_types.ProcessedTestRes
   :annotation:

   Type alias for processed diagnostic test results.

.. autodata:: scistanpy.custom_types.StrippedTestRes
   :annotation:

   Type alias for simplified diagnostic test results.

Utility Types
~~~~~~~~~~~~~

.. autodata:: scistanpy.custom_types.IndexType
   :annotation:

   Type alias for array indexing operations.

Usage Examples
--------------

**Type Checking in Functions:**

.. code-block:: python

   from scistanpy.custom_types import ContinuousParameterType

   def process_parameter(param: ContinuousParameterType) -> float:
       """Process a continuous parameter."""
       # Function accepts any continuous parameter type
       return float(param)

**Working with Distribution Types:**

.. code-block:: python

   from scistanpy.custom_types import SciStanPyDistribution

   def sample_from_distribution(dist: SciStanPyDistribution, n: int) -> np.ndarray:
       """Sample from any SciStanPy-compatible distribution."""
       return dist.sample((n,))

**Parameter Combinations:**

.. code-block:: python

   from scistanpy.custom_types import CombinableParameterType

   def combine_parameters(
       param1: CombinableParameterType,
       param2: CombinableParameterType
   ) -> CombinableParameterType:
       """Combine two parameters mathematically."""
       return param1 + param2
