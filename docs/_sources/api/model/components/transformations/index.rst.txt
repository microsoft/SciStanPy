Transformations API Reference
=============================

This reference covers the mathematical transformations and operator overloading framework in SciStanPy.

The transformations submodule provides the foundation for mathematical operations on model components, enabling complex model construction through operator overloading and automatic dependency tracking.

Transformations Submodule Overview
----------------------------------

The transformations submodule consists of several key components:

.. toctree::
   :maxdepth: 1

   cdfs
   transformed_data
   transformed_parameters


Mathematical Transformations
----------------------------

The transformations system allows you to create complex mathematical relationships:

**Example Usage:**

.. code-block:: python

   import scistanpy as ssp
   import numpy as np

   # Basic parameter
   log_rate = ssp.parameters.Normal(mu=0, sigma=1)

   # Transform to ensure positive values
   rate = ssp.operations.exp(log_rate)

   # Use in further calculations
   concentration = initial_conc * ssp.operations.exp(-rate * time)

Transformation Types
--------------------

SciStanPy provides several types of transformations:

- **Unary transformations**: Operations on single parameters (exp, log, abs, etc.)
- **Binary transformations**: Operations between parameters (+, -, \*, /, etc.)
- **Specialized transformations**: Domain-specific mathematical functions

All transformations:

- **Preserve gradients**: Support automatic differentiation
- **Generate Stan code**: Automatically convert to Stan syntax
- **Handle broadcasting**: Work with arrays and scalars consistently
- **Validate inputs**: Check mathematical constraints

Stan Code Generation
--------------------

Transformations automatically generate appropriate Stan code:

.. code-block:: python

   # Python expression
   result = ssp.operations.exp(log_param) + constant_value

   # Automatically generates Stan code like:
   # result = exp(log_param) + constant_value;
