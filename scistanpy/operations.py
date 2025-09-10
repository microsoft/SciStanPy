# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Custom operations for use in SciStanPy Stan models.

This module provides a framework for adding mathematical operations to the model
graph. Operations are built from
:py:class:`~scistanpy.model.components.transformations.transformed_parameters.TransformedParameter`
classes and can handle both immediate computation on NumPy/PyTorch data and deferred
computation within model graphs. This module should be the access point to all operations
available in SciStanPy--users should not need to directly interact with the underlying
transformation classes.
"""

from __future__ import annotations

from scistanpy.model.components import abstract_model_component
from scistanpy.model.components.transformations import transformed_parameters

# pylint: disable=line-too-long


class MetaOperation(type):
    """Metaclass for dynamically creating operation classes.

    This metaclass is responsible for creating operation classes from
    :py:class:`~scistanpy.model.components.transformations.transformed_parameters.TransformedParameter`
    classes. It validates that a ``DISTCLASS`` attribute is provided and appropriate,
    then automatically inherits documentation from the underlying transformation
    class. In general, users will not need to interact with this metaclass directly.

    :param name: Name of the class being created
    :type name: str
    :param bases: Base classes for the new class
    :type bases: tuple
    :param attrs: Class attributes dictionary
    :type attrs: dict

    :raises ValueError: If ``DISTCLASS`` is not provided in class attributes
    :raises TypeError: If ``DISTCLASS`` is not a subclass of
        :py:class:`~scistanpy.model.components.transformations.transformed_parameters.TransformedParameter`

    The metaclass performs the following validations and setup:

    - Ensures ``DISTCLASS`` attribute exists
    - Validates ``DISTCLASS`` inheritance from
      :py:class:`~scistanpy.model.components.transformations.transformed_parameters.TransformedParameter`
    - Inherits documentation from the ``DISTCLASS`` to the ``__call__`` method
    """

    def __new__(mcs, name, bases, attrs):

        # There must be a DISTCLASS in the class_attrs
        if "DISTCLASS" not in attrs:
            raise ValueError("DISTCLASS must be provided in class_attrs")

        # The DISTCLASS must be a subclass of TransformedParameter
        if not issubclass(
            attrs["DISTCLASS"], transformed_parameters.TransformedParameter
        ):
            raise TypeError("DISTCLASS must be a subclass of TransformedParameter")

        # Create the class
        cls = super().__new__(mcs, name, bases, attrs)

        return cls

    def __init__(cls, name, bases, attrs):

        # Run base init
        super().__init__(name, bases, attrs)

        # Create a new call method that runs the inherited call method but that
        # uses DISTCLASS's __call__ method's docstring
        def __call__(self, *args, **kwargs):
            return super(cls, self).__call__(*args, **kwargs)

        # Update the docstring of the call method based on the function
        cls.__call__ = __call__
        cls.__call__.__doc__ = cls.DISTCLASS.__doc__


class Operation:
    """Base class for SciStanPy mathematical operations.

    This class provides the foundation for all mathematical operations that can
    be used in SciStanPy models. Operations can handle both immediate computation
    on numerical data and deferred computation when used with model components.

    The class should never be instantiated directly. Instead, use the
    :py:func:`~scistanpy.operations.build_operation` function to create specific
    operation instances from
    :py:class:`~scistanpy.model.components.transformations.transformed_parameters.TransformedParameter`
    classes.

    :cvar DISTCLASS: The
        :py:class:`~scistanpy.model.components.transformations.transformed_parameters.TransformedParameter`
        class this operation wraps.
    :type DISTCLASS: type[transformed_parameters.TransformedParameter]

    .. note::
        Operations automatically detect whether they are being called with
        SciStanPy model components or raw numerical data and handle each
        case appropriately.
    """

    DISTCLASS: type[transformed_parameters.TransformedParameter]

    def __call__(self, *args, **kwargs):
        """Apply the operation to the provided inputs.

        This method provides intelligent dispatching based on the input types.
        When called with SciStanPy model components, it returns a new
        :py:class:`~scistanpy.model.components.transformations.transformed_parameters.TransformedParameter`
        instance for deferred computation. When called with raw numerical data
        (numpy), it performs immediate computation.

        :param args: Positional arguments for the operation. Can be model components
            or numerical data.
        :type args: tuple
        :param kwargs: Keyword arguments for the operation. Can contain model components
            or numerical data.
        :type kwargs: dict

        :returns: Either a
            :py:class:`~scistanpy.model.components.transformations.transformed_parameters.TransformedParameter`
            instance (for deferred computation) or the immediate result of the
            operation on numerical data.
        :rtype: Union[transformed_parameters.TransformedParameter, Any]

        The method behavior depends on input types:

        - If any argument is a SciStanPy model component: Returns a
          :py:class:`~scistanpy.model.components.transformations.transformed_parameters.TransformedParameter`
          instance for later evaluation
        - If all arguments are numerical data: Performs immediate computation using
          the underlying
          :py:meth:`TransformedParameter.run_np_torch_op() <scistanpy.model.components.transformations.transformed_parameters.TransformedParameter.run_np_torch_op>`
          method to compute and return the result directly.
        """
        # If any of the args or kwargs are scistanpy parameters, return an instance
        # of the appropriate TransformedParameter class. This effectively delays
        # the call to the `run_np_torch_op` method until the operation is actually
        # applied to the parameters.
        if any(
            isinstance(arg, abstract_model_component.AbstractModelComponent)
            for arg in args
        ) or any(
            isinstance(value, abstract_model_component.AbstractModelComponent)
            for value in kwargs.values()
        ):
            return self.__class__.DISTCLASS(*args, **kwargs)

        # Otherwise, call the `run_np_torch_op` method as a static method
        return self.__class__.DISTCLASS.run_np_torch_op(self=None, *args, **kwargs)


def build_operation(
    distclass: type[transformed_parameters.TransformedParameter],
) -> Operation:
    """Build an operation instance from a TransformedParameter class.

    This function creates a new operation class using the
    :py:class:`~scistanpy.operations.MetaOperation` metaclass in combination with
    the :py:class:`~scistanpy.operations.Operation` base class, then returns an
    instance of that class. The resulting operation inherits documentation and
    behavior from the provided
    :py:class:`~scistanpy.model.components.transformations.transformed_parameters.TransformedParameter`
    class.

    :param distclass: The :py:class:`~scistanpy.model.components.transformations.transformed_parameters.TransformedParameter`
        class to build the operation from.
    :type distclass: type[transformed_parameters.TransformedParameter]

    :returns: A new operation instance that wraps the provided
        :py:class:`~scistanpy.model.components.transformations.transformed_parameters.TransformedParameter`
        class.
    :rtype: Operation

    :raises ValueError: If distclass is not provided
    :raises TypeError: If distclass is not a subclass of TransformedParameter

    Example:

    .. code-block:: python

       from scistanpy.operations import build_operation
       from scistanpy.model.components.transformations.transformed_parameters import UnaryTransformedParameter

       class MyTransformation(UnaryTransformedParameter):
           # Custom mathematical transformation.

           def run_np_torch_op(self, x):
               # Implementation for numerical data
               return x**2 + 1

           def write_stan_operation(self, x: str) -> str:
               # Stan code generation
               return f"square({x}) + 1"

       # Create operation
       my_operation = build_operation(MyTransformation)

       # Use the operation
       param = ssp.parameters.Normal(mu=0.0, sigma=1.0)
       transformed = my_operation(param)

    """
    # Build the class via the metaclass
    return MetaOperation(
        distclass.__name__.lower(),
        (Operation,),
        {"DISTCLASS": distclass, "__doc__": distclass.__doc__},
    )()


# Define our operations
abs_ = build_operation(transformed_parameters.AbsParameter)
"""Absolute value operation.

Computes the absolute value of the input parameter or numerical data. See also,
:py:class:`~scistanpy.model.components.transformations.transformed_parameters.AbsParameter`.

    **Usage:**

    .. code-block:: python

      # With model components
      param = ssp.parameters.Normal(mu=0.0, sigma=1.0)
      abs_param = ssp.operations.abs_(param)

      # With numerical data
      result = ssp.operations.abs_([-1.0, -2.0, 3.0])  # Returns [1.0, 2.0, 3.0]
"""

binary_exponential_growth = build_operation(
    transformed_parameters.BinaryExponentialGrowth
)
"""Binary exponential growth operation.

Models exponential growth over two timepoints, taking in the population size at
the starting time (assumed to be at t = 0) and outputting the population size at
the ending time (assumed to be at = 1). See also,
:py:class:`~scistanpy.model.components.transformations.transformed_parameters.BinaryExponentialGrowth`.

   **Usage:**

   .. code-block:: python

      # Two-timepoint exponential growth
      initial_size = ssp.parameters.LogNormal(mu=np.log(50), sigma=0.2)
      growth_rate = ssp.parameters.Normal(mu=0.2, sigma=0.1)

      final_size = ssp.operations.binary_exponential_growth(
          A=initial_size, r=growth_rate
      )
"""

binary_log_exponential_growth = build_operation(
    transformed_parameters.BinaryLogExponentialGrowth
)
"""Binary log-exponential growth operation.

Identical to :py:func:`~scistanpy.operations.binary_exponential_growth`, only models
the *log* of the population size. See also,
:py:class:`~scistanpy.model.components.transformations.transformed_parameters.BinaryLogExponentialGrowth`.
"""

exp = build_operation(transformed_parameters.ExpParameter)
"""Exponential operation.

Computes the exponential (e^x) of the input parameter or numerical data. See also,
:py:class:`~scistanpy.model.components.transformations.transformed_parameters.ExpParameter`.

**Usage:**

.. code-block:: python

      # Exponential transformation
      log_rate = ssp.parameters.Normal(mu=0.0, sigma=1.0)
      rate = ssp.operations.exp(log_rate)  # Ensures positive values
"""

log = build_operation(transformed_parameters.LogParameter)
"""Natural logarithm operation.

Computes the natural logarithm of the input parameter or numerical data. See also,
:py:class:`~scistanpy.model.components.transformations.transformed_parameters.LogParameter`.

**Usage:**

.. code-block:: python

   # Natural logarithm
   positive_param = ssp.parameters.LogNormal(mu=0.0, sigma=1.0)
   log_param = ssp.operations.log(positive_param)
"""

log1p_exp = build_operation(transformed_parameters.Log1pExpParameter)
"""``log(1 + exp(x))`` operation.

Computes ``log(1 + exp(x))`` in a numerically stable way. See also,
:py:class:`~scistanpy.model.components.transformations.transformed_parameters.Log1pExpParameter`.
"""

log_exponential_growth = build_operation(transformed_parameters.LogExponentialGrowth)
"""Log-exponential growth operation.

Identical to `~scistanpy.operations.exponential_growth`, only models the *log* of
the population size. See also,
:py:class:`~scistanpy.model.components.transformations.transformed_parameters.LogExponentialGrowth`.
"""

log_sigmoid = build_operation(transformed_parameters.LogSigmoidParameter)
"""Log-sigmoid operation.

Computes the logarithm of the sigmoid function in a numerically stable way. See also,
:py:class:`~scistanpy.model.components.transformations.transformed_parameters.LogSigmoidParameter`.
"""

log_sigmoid_growth = build_operation(transformed_parameters.LogSigmoidGrowth)
"""Log-sigmoid growth operation.

Identical to :py:func:`~scistanpy.operations.log_sigmoid_growth`, only models the
*log* of the population size. See also,
:py:class:`~scistanpy.model.components.transformations.transformed_parameters.LogSigmoidGrowth`.
"""

log_sigmoid_growth_init_param = build_operation(
    transformed_parameters.LogSigmoidGrowthInitParametrization
)
"""An alternative parametrization of :py:func:`~scistanpy.operations.log_sigmoid_growth`
parametrized using initial population abundance rather than carrying capacity. See also,
:py:class:`~scistanpy.model.components.transformations.transformed_parameters.LogSigmoidGrowthInitParametrization`.
"""

logsumexp = build_operation(transformed_parameters.LogSumExpParameter)
"""Log-sum-exp operation.

Note that for SciStanPy operations, the sum is always performed over the last
dimension.

Computes log(sum(exp(x))) in a numerically stable way. See also,
:py:class:`~scistanpy.model.components.transformations.transformed_parameters.LogSumExpParameter`.
"""

normalize = build_operation(transformed_parameters.NormalizeParameter)
"""Normalization operation.

Normalizes input data to unit sum. Note that when applied to SciStanPy parameters,
this operation is always performed over the last dimension. See also,
:py:class:`~scistanpy.model.components.transformations.transformed_parameters.NormalizeParameter`.

**Usage:**

.. code-block:: python

    # Normalize to unit sum (over last dimension)
    weights = ssp.parameters.LogNormal(mu=0.0, sigma=1.0, shape=(3, 9))
    probabilities = ssp.operations.normalize(weights) # Each row sums to 1
"""

normalize_log = build_operation(transformed_parameters.NormalizeLogParameter)
"""Log-space normalization operation.

    This function ensures that the sum of exponentials is equal to 1 over the last dimension
    of the input. See also,
    :py:class:`~scistanpy.model.components.transformations.transformed_parameters.NormalizeLogParameter`.

    **Usage:**

    .. code-block:: python

        # Normalize to unit sum (over last dimension)
        weights = ssp.parameters.Normal(mu=0.0, sigma=1.0, shape=(3, 9))
        probabilities = ssp.operations.exp(
            ssp.operations.normalize(weights)
        ) # Each row sums to 1
"""

sigmoid = build_operation(transformed_parameters.SigmoidParameter)
"""Sigmoid operation.

Computes the sigmoid function (1 / (1 + exp(-x))) of the input. See also,
:py:class:`~scistanpy.model.components.transformations.transformed_parameters.SigmoidParameter`.

**Usage:**

.. code-block:: python

   # Convert logits to probabilities
   logits = ssp.parameters.Normal(mu=0.0, sigma=1.0)
   probabilities = ssp.operations.sigmoid(logits)
"""

sigmoid_growth = build_operation(transformed_parameters.SigmoidGrowth)
"""Sigmoid growth operation.

Models sigmoid growth patterns in time series data. See also,
:py:class:`~scistanpy.model.components.transformations.transformed_parameters.SigmoidGrowth`.

**Usage:**

.. code-block:: python

   # Logistic growth with carrying capacity
   time_points = np.array([0, 5, 10, 15, 20])
   carrying_capacity = ssp.parameters.LogNormal(mu=np.log(1000), sigma=0.1)
   growth_rate = ssp.parameters.Normal(mu=0.3, sigma=0.1)
   inflection_time = ssp.parameters.Normal(mu=10.0, sigma=2.0)

   population = ssp.operations.sigmoid_growth(
       t=time_points, A=carrying_capacity, r=growth_rate, c=inflection_time
   )
"""

sigmoid_growth_init_param = build_operation(
    transformed_parameters.SigmoidGrowthInitParametrization
)
"""An alternate parametrization of :py:func:`~scistanpy.operations.sigmoid_growth`
parametrized using initial population abundance rather than carrying capacity. See also,
:py:class:`~scistanpy.model.components.transformations.transformed_parameters.SigmoidGrowthInitParametrization`.

**Usage:**

.. code-block:: python

   # Alternative sigmoid parametrization using initial population
   time_points = np.array([0, 5, 10])
   carrying_capacity = ssp.parameters.LogNormal(mu=np.log(1000), sigma=0.1)
   growth_rate = ssp.parameters.Normal(mu=0.3, sigma=0.1)
   init_pop = ssp.parameters.LogNormal(mu=np.log(10), sigma=0.2)

   population = ssp.operations.sigmoid_growth_init_param(
       t=time_points, A=carrying_capacity, r=growth_rate, A0=init_pop
   )
"""

sum_ = build_operation(transformed_parameters.SumParameter)
"""Summation operation.

Computes the sum over an input parameter dimension or over numerical data. Note
that for SciStanPy operations, the sum is always performed over the last dimension.
See also,
:py:class:`~scistanpy.model.components.transformations.transformed_parameters.SumParameter`.

**Usage:**

.. code-block:: python

   # Sum over last dimension
   values = ssp.parameters.Normal(mu=0.0, sigma=1.0, shape=(5, 3))
   total = ssp.operations.sum_(values) # Shape = (5,)

   # Sum but keep dimensions
   total_keepdims = ssp.operations.sum_(values, keepdims=True) # Shape = (5, 1)
"""

exponential_growth = build_operation(transformed_parameters.ExponentialGrowth)
"""Exponential growth operation.

Models exponential growth patterns in time series data. See also,
:py:class:`~scistanpy.model.components.transformations.transformed_parameters.ExponentialGrowth`.

**Usage:**

.. code-block:: python

   # Model exponential population growth
   time_points = np.array([0, 1, 2, 3, 4])
   initial_pop = ssp.parameters.LogNormal(mu=np.log(100), sigma=0.1)
   growth_rate = ssp.parameters.Normal(mu=0.1, sigma=0.05)

   population = ssp.operations.exponential_growth(
       t=time_points, A=initial_pop, r=growth_rate
   )
"""

binary_exponential_growth = build_operation(
    transformed_parameters.BinaryExponentialGrowth
)
"""Binary exponential growth operation.

Models exponential growth over two timepoints, taking in the population size at
the starting time (assumed to be at t = 0) and outputting the population size at
the ending time (assumed to be at t = 1). See also,
:py:class:`~scistanpy.model.components.transformations.transformed_parameters.BinaryExponentialGrowth`.

**Usage:**

.. code-block:: python

   # Two-timepoint exponential growth
   initial_size = ssp.parameters.LogNormal(mu=np.log(50), sigma=0.2)
   growth_rate = ssp.parameters.Normal(mu=0.2, sigma=0.1)

   final_size = ssp.operations.binary_exponential_growth(
       A=initial_size, r=growth_rate
   )
"""

convolve_sequence = build_operation(transformed_parameters.ConvolveSequence)
"""Sequence convolution operation.

Performs convolution operations on sequence data. See also,
:py:class:`~scistanpy.model.components.transformations.transformed_parameters.ConvolveSequence`.

**Usage:**

.. code-block:: python

    # Sequence convolution for pattern matching
    weights = Normal(mu=0, sigma=1, shape=(motif_length, 4))  # 4 nucleotides
    dna_sequence = Constant(encoded_dna)  # 0,1,2,3 for A,C,G,T

    # Perform convolution of the sequence with the weights
    convolved = ssp.operations.convolve_sequence(
        weights=weights, ordinals=dna_sequence
    )
"""
