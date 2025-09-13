Transformed Parameters API Reference
====================================

.. automodule:: scistanpy.model.components.transformations.transformed_parameters
   :undoc-members:
   :show-inheritance:


The transformation system supports the following classes of operations:

**Basic Arithmetic via Python Operators**
    - Addition (``+``) via :py:class:`~scistanpy.model.components.transformations.transformed_parameters.AddParameter`
    - Subtraction (``-``) via :py:class:`~scistanpy.model.components.transformations.transformed_parameters.SubtractParameter`
    - Multiplication (``*``) via :py:class:`~scistanpy.model.components.transformations.transformed_parameters.MultiplyParameter`
    - Division (``/``) via :py:class:`~scistanpy.model.components.transformations.transformed_parameters.DivideParameter`
    - Power (``**``) via :py:class:`~scistanpy.model.components.transformations.transformed_parameters.PowerParameter`
    - Negation (Unary ``-``) via :py:class:`~scistanpy.model.components.transformations.transformed_parameters.NegateParameter`

**Standard Mathematical Functions**
    - Absolute Value via :py:class:`~scistanpy.model.components.transformations.transformed_parameters.AbsParameter`
    - Natural Logarithm via :py:class:`~scistanpy.model.components.transformations.transformed_parameters.LogParameter`
    - Exponential via :py:class:`~scistanpy.model.components.transformations.transformed_parameters.ExpParameter`
    - Sigmoid via :py:class:`~scistanpy.model.components.transformations.transformed_parameters.SigmoidParameter`
    - Log Sigmoid via :py:class:`~scistanpy.model.components.transformations.transformed_parameters.LogSigmoidParameter`
    - :math:`\log{1 + \exp{x}}` via :py:class:`~scistanpy.model.components.transformations.transformed_parameters.Log1pExpParameter`

**Normalization Operations**
    - Normalization (sum to 1) via :py:class:`~scistanpy.model.components.transformations.transformed_parameters.NormalizeParameter`
    - Log-Space Normalization (sum of exponents to 1) via :py:class:`~scistanpy.model.components.transformations.transformed_parameters.NormalizeLogParameter`

**Reduction Operations**
    - Sum Reduction via :py:class:`~scistanpy.model.components.transformations.transformed_parameters.SumParameter`
    - Log-Sum-Exp via :py:class:`~scistanpy.model.components.transformations.transformed_parameters.LogSumExpParameter`

**Growth Models**
    - Exponential Growth via :py:class:`~scistanpy.model.components. transformations.transformed_parameters.ExponentialGrowth` or :py:class:`~scistanpy.model.components.transformations.transformed_parameters.BinaryExponentialGrowth`
    - Log Exponential Growth via :py:class:`~scistanpy.model.components.transformations.transformed_parameters.LogExponentialGrowth` or :py:class:`~scistanpy.model.components.transformations.transformed_parameters.BinaryLogExponentialGrowth`
    - Sigmoid Growth via :py:class:`~scistanpy.model.components.transformations.
      transformed_parameters.SigmoidGrowth` or :py:class:`~scistanpy.model.components.transformations.transformed_parameters.SigmoidGrowthInitParametrization`
    - Log Sigmoid Growth via :py:class:`~scistanpy.model.components.transformations.transformed_parameters.LogSigmoidGrowth` or :py:class:`~scistanpy.model.components.transformations.transformed_parameters.LogSigmoidGrowthInitParametrization`

**Special Functions**
    - Sequence Convolution via :py:class:`~scistanpy.model.components.transformations.transformed_parameters.ConvolveSequence`

**Indexing and Array Operations**
    - Indexing and slicing via :py:class:`~scistanpy.model.components.transformations.transformed_parameters.IndexParameter`

Note that none of these classes will typically be instantiated directly. Instead, end users will access them either via Python's inbuilt operators between other model components or else use SciStanPy's :py:mod:`~scistanpy.operations` submodule.

Base Classes
------------
Transformed parameters are composed using a hierarchy of base classes that define their behavior and interfaces. The main base classes are:

   - :py:class:`~scistanpy.model.components.transformations.transformed_parameters.TransformableParameter`, which activates operator overloading for mathematical expressions. Note that, in addition to :py:class:`~scistanpy.model.components.transformations.transformed_parameters.TransformedParameter` subclasses, all :py:class:`~scistanpy.model.components.parameters.ContinuousDistribution` subclasses also inherit from this class. Classes that inherit from this base are, as the name suggests, transformable via any Python operators or functions defined in the :py:mod:`~scistanpy.operations` module.
   - :py:class:`~scistanpy.model.components.transformations.transformed_parameters.Transformation`, which is the abstract base class for all transformations (i.e., both :py:class:`~scistanpy.model.components.transformations.transformed_parameters.TransformedParameter` and :py:class:`~scistanpy.model.components.transformations.transformed_data.TransformedData`). This class defines the core interface and behavior for all transformed parameters, including methods for drawing samples, generating Stan code, and managing parent components.
   - :py:class:`~scistanpy.model.components.transformations.transformed_parameters.TransformedParameter`, which is the base class for all SciStanPy transformed parameters.
   - :py:class:`~scistanpy.model.components.transformations.transformed_parameters.UnaryTransformedParameter`, which is a convenience base class for transformations that take a single input parameter.
   - :py:class:`~scistanpy.model.components.transformations.transformed_parameters.BinaryTransformedParameter`, which is a convenience base class for transformations that take two input parameters.

Documentation for each of these base classes is provided below:

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.TransformableParameter
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __add__, __radd__, __sub__, __rsub__, __mul__, __rmul__, __truediv__, __rtruediv__, __pow__, __rpow__, __neg__

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.Transformation
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.TransformedParameter
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.UnaryTransformedParameter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.BinaryTransformedParameter
   :members:
   :undoc-members:
   :show-inheritance:

Basic Arithmetic Operations
---------------------------
.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.AddParameter
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.SubtractParameter
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.MultiplyParameter
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.DivideParameter
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.PowerParameter
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.NegateParameter
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

Standard Mathematical Functions
-------------------------------

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.AbsParameter
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.ExpParameter
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.LogParameter
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.SigmoidParameter
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.LogSigmoidParameter
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.Log1pExpParameter
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

Normalization Transformations
-----------------------------
.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.NormalizeParameter
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.NormalizeLogParameter
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

Reduction Operations
--------------------
Reductions are built from an additional intermediate base class, :py:class:`~scistanpy.model.components.transformations.transformed_parameters.Reduction`, which provides shared functionality for all reduction operations. It is documented below followed by the specific reduction transformations.

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.Reduction
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.SumParameter
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: NP_FUNC, TORCH_FUNC, write_stan_operation

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.LogSumExpParameter
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: NP_FUNC, TORCH_FUNC, write_stan_operation

Growth Model Transformations
----------------------------
SciStanPy has been thoroughly applied to modeling biological growth processes, particularly in the context of deep mutational scanning experiments. The following transformations provide a suite of growth model operations that can be used to define complex growth dynamics in a probabilistic framework. Note that all of these transformations could also be implemented using the basic arithmetic and mathematical functions provided above, but these classes provide a more convenient interface and better integration with SciStanPy's model component system. If there are other compound functions that would be useful to include in SciStanPy (whether growth-related or not), please consider raising an issue on the SciStanPy GitHub repository or contributing a custom transformation.

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.ExponentialGrowth
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.LogExponentialGrowth
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.BinaryExponentialGrowth
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.BinaryLogExponentialGrowth
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.SigmoidGrowth
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.LogSigmoidGrowth
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.SigmoidGrowthInitParametrization
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.LogSigmoidGrowthInitParametrization
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation

Special Functions
-----------------
.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.ConvolveSequence
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation, get_index_offset, get_right_side, get_supporting_functions

Indexing and Array Operations
-----------------------------
.. autoclass:: scistanpy.model.components.transformations.transformed_parameters.IndexParameter
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: run_np_torch_op, write_stan_operation, get_assign_depth, get_right_side, get_transformation_assignment