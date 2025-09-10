Operations API Reference
========================

.. automodule:: scistanpy.operations
   :undoc-members:
   :show-inheritance:

Operation Framework
-------------------

The operations module provides a framework for creating mathematical operations from :py:class:`~scistanpy.model.components.transformations.transformed_parameters.TransformedParameter` classes that work with both SciStanPy model components and raw numerical data (NumPy arrays/PyTorch tensors).

**Core Classes:**

.. autoclass:: scistanpy.operations.MetaOperation
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.operations.Operation
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

**Operation Builder:**

The :py:func:`~scistanpy.operations.build_operation` function allows users to create custom operations easily by wrapping a :py:class:`~scistanpy.model.components.transformations.transformed_parameters.TransformedParameter` subclass. This is the main interface for creating new operations from custom :py:class:`~scistanpy.model.components.transformations.transformed_parameters.TransformedParameter` classes and is what is used under the hood to create all built-in operations.

.. autofunction:: scistanpy.operations.build_operation

Operation Usage Patterns
------------------------
Operations can be used in two main contexts: with SciStanPy model components (:py:class:`~scistanpy.model.components.parameters.Parameter`, :py:class:`~scistanpy.model.components.transformations.transformed_parameters.TransformedParameter`, and :py:class:`~scistanpy.model.components.constants.Constant`) and with raw numerical data (NumPy arrays or PyTorch tensors). The operations automatically dispatch to the appropriate behavior based on the input type.

**With Model Components:**

.. code-block:: python

   # Operations with parameters create transformed parameters
   base_param = ssp.parameters.Normal(mu=0.0, sigma=1.0)
   transformed = ssp.operations.exp(base_param)

   # Use in model definitions
   observed = ssp.parameters.Normal(mu=transformed, sigma=0.1)

**With Numerical Data:**

.. code-block:: python

   # Operations with numerical data compute immediately
   import numpy as np

   data = np.array([1, 2, 3])
   result = ssp.operations.exp(data)  # Returns exp([1, 2, 3])

**Intelligent Dispatching:**

.. code-block:: python

   # Operations automatically detect input types
   def my_function(x):
       return ssp.operations.log(ssp.operations.exp(x) + 1)

   # Works with both parameters and data
   param_result = my_function(ssp.parameters.Normal(mu=0, sigma=1))
   data_result = my_function(np.array([1, 2, 3]))

Available Operations
--------------------
The following operations are available in the SciStanPy operations module. Each operation is documented with its usage patterns and examples.

If there's a specific mathematical operation that you need which is not listed here (and, given that this is an evolving library, there are certainly many missing), please consider (A) raising an issue on the SciStanPy GitHub repository requesting its addition, or (B) creating a custom operation using the :py:func:`~scistanpy.operations.build_operation` function. If you take the latter approach, also consider contributing your custom operation back to the SciStanPy project for inclusion in future releases!

Mathematical Functions
~~~~~~~~~~~~~~~~~~~~~~
Below are the basic mathematical operations provided by SciStanPy.

.. autofunction:: scistanpy.operations.abs_

.. autofunction:: scistanpy.operations.exp

.. autofunction:: scistanpy.operations.log

.. autofunction:: scistanpy.operations.log1p_exp

.. autofunction:: scistanpy.operations.sigmoid

.. autofunction:: scistanpy.operations.log_sigmoid

Normalization Operations
~~~~~~~~~~~~~~~~~~~~~~~~
Normalization operations that can be used to build constrained transformed parameters from unconstrained parameters. Take care with these when sampling! It is easy to create non-identifiable models if the normalization is not used correctly.

.. note::

   The normalization operations always operate over the last dimension of the input parameter when applied to SciStanPy parameters.

.. autofunction:: scistanpy.operations.normalize

.. autofunction:: scistanpy.operations.normalize_log

Reduction Operations
~~~~~~~~~~~~~~~~~~~~
Reduction operations allow for aggregating values across dimensions of parameters or data. As with the normalization operations, these always operate over the last dimension of the input parameter when applied to SciStanPy parameters.

.. autofunction:: scistanpy.operations.sum_

.. autofunction:: scistanpy.operations.logsumexp

Growth Model Operations
~~~~~~~~~~~~~~~~~~~~~~~
Growth model operations provide a set of mathematical transformations for modeling population growth and similar temporal dynamics. These operations can be used to define growth models in a probabilistic framework (e.g., for deep mutational scanning).

.. autofunction:: scistanpy.operations.exponential_growth

.. autofunction:: scistanpy.operations.binary_exponential_growth

.. autofunction:: scistanpy.operations.log_exponential_growth

.. autofunction:: scistanpy.operations.binary_log_exponential_growth

.. autofunction:: scistanpy.operations.sigmoid_growth

.. autofunction:: scistanpy.operations.log_sigmoid_growth

.. autofunction:: scistanpy.operations.sigmoid_growth_init_param

.. autofunction:: scistanpy.operations.log_sigmoid_growth_init_param

Specialized Operations
~~~~~~~~~~~~~~~~~~~~~~
Specialized operations provide additional mathematical transformations that are useful in specific modeling contexts. There is plenty of room for expansion in this category, and users are encouraged to contribute new operations as needed.

.. autofunction:: scistanpy.operations.convolve_sequence
