"""Custom operations for use in SciStanPy Stan models.

This module provides a framework for adding mathematical operations to the model
graph. Operations are built from TransformedParameter classes and can handle both
immediate computation on NumPy/PyTorch data and deferred computation within model
graphs.

The module uses a metaclass pattern to dynamically create operation classes that
maintain proper documentation inheritance and provide flexible parameter handling.
Operations can work with both raw numerical data and SciStanPy model components.
"""

from __future__ import annotations

from scistanpy.model.components import abstract_model_component
from scistanpy.model.components.transformations import transformed_parameters


class MetaOperation(type):
    """Metaclass for dynamically creating operation classes.

    This metaclass is responsible for creating operation classes from
    TransformedParameter classes. It validates that the provided distclass
    is appropriate and automatically inherits documentation from the
    underlying transformation class.

    :param name: Name of the class being created
    :type name: str
    :param bases: Base classes for the new class
    :type bases: tuple
    :param attrs: Class attributes dictionary
    :type attrs: dict

    :raises ValueError: If DISTCLASS is not provided in class attributes
    :raises TypeError: If DISTCLASS is not a subclass of TransformedParameter

    The metaclass performs the following validations and setup:
    - Ensures DISTCLASS attribute exists
    - Validates DISTCLASS inheritance from TransformedParameter
    - Inherits documentation from the DISTCLASS to the __call__ method
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

        # Update the docstring of the call method based on the function
        cls.__call__.__doc__ = cls.DISTCLASS.__doc__

        return cls


class Operation:
    """Base class for SciStanPy mathematical operations.

    This class provides the foundation for all mathematical operations that can
    be used in SciStanPy models. Operations can handle both immediate computation
    on numerical data and deferred computation when used with model components.

    The class should never be instantiated directly. Instead, use the
    `build_operation` function to create specific operation instances from
    TransformedParameter classes.

    :cvar DISTCLASS: The TransformedParameter class this operation wraps
    :type DISTCLASS: type[transformed_parameters.TransformedParameter]

    Note:
        Operations automatically detect whether they are being called with
        SciStanPy model components or raw numerical data and handle each
        case appropriately.
    """

    DISTCLASS: type[transformed_parameters.TransformedParameter]

    def __call__(self, *args, **kwargs):
        """Apply the operation to the provided inputs.

        This method provides intelligent dispatching based on the input types.
        When called with SciStanPy model components, it returns a new
        TransformedParameter instance for deferred computation. When called
        with raw numerical data, it performs immediate computation.

        :param args: Positional arguments for the operation. Can be model
                    components or numerical data.
        :type args: tuple
        :param kwargs: Keyword arguments for the operation. Can contain model
                      components or numerical data.
        :type kwargs: dict

        :returns: Either a TransformedParameter instance (for deferred computation)
                 or the immediate result of the operation on numerical data
        :rtype: Union[transformed_parameters.TransformedParameter, Any]

        The method behavior depends on input types:
        - If any argument is a SciStanPy model component: Returns a
          TransformedParameter instance for later evaluation
        - If all arguments are numerical data: Performs immediate computation
          using the underlying transformation's run_np_torch_op method
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

    This function creates a new operation class using the MetaOperation metaclass
    and returns an instance of that class. The resulting operation inherits
    documentation and behavior from the provided TransformedParameter class.

    :param distclass: The TransformedParameter class to build the operation from.
                     Must be a subclass of TransformedParameter.
    :type distclass: type[transformed_parameters.TransformedParameter]

    :returns: A new operation instance that wraps the provided TransformedParameter
    :rtype: Operation

    :raises ValueError: If distclass is not provided
    :raises TypeError: If distclass is not a subclass of TransformedParameter

    Example:
        >>> # Create a custom operation from a TransformedParameter
        >>> my_operation = build_operation(MyTransformedParameter)
        >>> result = my_operation(input_data)
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

Computes the absolute value of the input parameter or numerical data.
"""

binary_exponential_growth = build_operation(
    transformed_parameters.BinaryExponentialGrowth
)
"""Binary exponential growth operation.

Models exponential growth over two timepoints, taking in the population size at
the starting time (assumed to be at t = 0) and outputting the population size at
the ending time (assumed to be at = 1).
"""

binary_log_exponential_growth = build_operation(
    transformed_parameters.BinaryLogExponentialGrowth
)
"""Binary log-exponential growth operation.

Identical to `binary_exponential_growth`, only models the log of the population
size.
"""

convolve_sequence = build_operation(transformed_parameters.ConvolveSequence)
"""Sequence convolution operation.

Performs convolution operations on sequence data.
"""

exp = build_operation(transformed_parameters.ExpParameter)
"""Exponential operation.

Computes the exponential (e^x) of the input parameter or numerical data.
"""

exponential_growth = build_operation(transformed_parameters.ExponentialGrowth)
"""Exponential growth operation.

Models exponential growth patterns in time series data.
"""

log = build_operation(transformed_parameters.LogParameter)
"""Natural logarithm operation.

Computes the natural logarithm of the input parameter or numerical data.
"""

log1p_exp = build_operation(transformed_parameters.Log1pExpParameter)
"""Log(1 + exp(x)) operation.

Computes log(1 + exp(x)) in a numerically stable way.
"""

log_exponential_growth = build_operation(transformed_parameters.LogExponentialGrowth)
"""Log-exponential growth operation.

Identical to `exponential_growth`, only models the log of the population size.
"""

log_sigmoid = build_operation(transformed_parameters.LogSigmoidParameter)
"""Log-sigmoid operation.

Computes the logarithm of the sigmoid function in a numerically stable way.
"""

log_sigmoid_growth = build_operation(transformed_parameters.LogSigmoidGrowth)
"""Log-sigmoid growth operation.

Models the logarithm of population size growing according to log-sigmoid growth
patterns in time series data.
"""

log_sigmoid_growth_init_param = build_operation(
    transformed_parameters.LogSigmoidGrowthInitParametrization
)
"""An alternative parametrization of `log_sigmoid_growth`.

Parametrizes a model of log sigmoid growth using initial population abundance.
"""

logsumexp = build_operation(transformed_parameters.LogSumExpParameter)
"""Log-sum-exp operation.

Computes log(sum(exp(x))) in a numerically stable way.
"""

normalize = build_operation(transformed_parameters.NormalizeParameter)
"""Normalization operation.

Normalizes input data to unit sum or other normalization schemes. Note that when
applied to SciStanPy parameters, this operation is always performed over the last
dimension.
"""

normalize_log = build_operation(transformed_parameters.NormalizeLogParameter)
"""Log-space normalization operation.

Performs normalization such that the log-sum of exponentials of terms is equal to
0 (and, consequently, such that the the sum of exponentials is equal to 1).
"""

sigmoid = build_operation(transformed_parameters.SigmoidParameter)
"""Sigmoid operation.

Computes the sigmoid function (1 / (1 + exp(-x))) of the input.
"""

sigmoid_growth = build_operation(transformed_parameters.SigmoidGrowth)
"""Sigmoid growth operation.

Models sigmoid growth patterns in time series data.
"""

sigmoid_growth_init_param = build_operation(
    transformed_parameters.SigmoidGrowthInitParametrization
)
"""An alternate parametrization of `sigmoid_growth`.

Parametrizes a model of sigmoid growth using initial population abundance.
"""

sum_ = build_operation(transformed_parameters.SumParameter)
"""Summation operation.

Computes the sum of input parameters or numerical data. Note that for SciStanPy
operations, the sum is always performed over the last dimensions.
"""
