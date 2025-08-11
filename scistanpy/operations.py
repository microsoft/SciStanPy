"""Defines custom operations that we can use in Stan models."""

from __future__ import annotations

from scistanpy.model.components import abstract_model_component
from scistanpy.model.components.transformations import transformed_parameters


class MetaOperation(type):
    """
    Metaclass for operations. This sets the __doc__ attribute of the __call__ method
    of the class based on the function that the operation will be performing.
    """

    def __new__(mcs, name, bases, attrs):

        # There must be a distclass in the class_attrs
        if "distclass" not in attrs:
            raise ValueError("distclass must be provided in class_attrs")

        # The distclass must be a subclass of TransformedParameter
        if not issubclass(
            attrs["distclass"], transformed_parameters.TransformedParameter
        ):
            raise TypeError("distclass must be a subclass of TransformedParameter")

        # Create the class
        cls = super().__new__(mcs, name, bases, attrs)

        # Update the docstring of the call method based on the function
        cls.__call__.__doc__ = cls.distclass.__doc__

        return cls


class Operation:
    """
    Base class for scistanpy operations. This should never be instantiated directly,
    but instead is used within the `build_operation` function to create operations
    from `transformed_parameters.TransformedParameter` classes.
    """

    distclass: type[transformed_parameters.TransformedParameter]

    def __call__(self, *args, **kwargs):
        """
        Apply the operation to the input.

        Args:
            *args: The input to the operation. If a scistanpy parameter, the operation
                is performed on the parameter. Otherwise, the operation is performed
                using PLACEHOLDER with *args passed to that function.

            **kwargs: Additional arguments to pass to PLACEHOLDER.

        Returns:
            The result of applying the operation to the input.
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
            return self.distclass(*args, **kwargs)

        # Otherwise, call the `run_np_torch_op` method as a static method
        return self.distclass.run_np_torch_op(self=None, *args, **kwargs)


def build_operation(
    distclass: type[transformed_parameters.TransformedParameter],
) -> Operation:
    """
    Build an operation from a TransformedParameter class.

    Args:
        distclass: The TransformedParameter class to build the operation from.

    Returns:
        Operation: An instance of the Operation class with the distclass set.
    """
    # Build the class via the metaclass
    return MetaOperation(
        distclass.__name__.lower(),
        (Operation,),
        {"distclass": distclass, "__doc__": distclass.__doc__},
    )()


# Define our operations
abs_ = build_operation(transformed_parameters.AbsParameter)
exp = build_operation(transformed_parameters.ExpParameter)
log = build_operation(transformed_parameters.LogParameter)
logsumexp = build_operation(transformed_parameters.LogSumExpParameter)
normalize = build_operation(transformed_parameters.NormalizeParameter)
normalize_log = build_operation(transformed_parameters.NormalizeLogParameter)
sigmoid = build_operation(transformed_parameters.SigmoidParameter)
log_sigmoid = build_operation(transformed_parameters.LogSigmoidParameter)
exponential_growth = build_operation(transformed_parameters.ExponentialGrowth)
binary_exponential_growth = build_operation(
    transformed_parameters.BinaryExponentialGrowth
)
log_exponential_growth = build_operation(transformed_parameters.LogExponentialGrowth)
binary_log_exponential_growth = build_operation(
    transformed_parameters.BinaryLogExponentialGrowth
)
sigmoid_growth = build_operation(transformed_parameters.SigmoidGrowth)
log_sigmoid_growth = build_operation(transformed_parameters.LogSigmoidGrowth)
sigmoid_growth_init_param = build_operation(
    transformed_parameters.SigmoidGrowthInitParametrization
)
log_sigmoid_growth_init_param = build_operation(
    transformed_parameters.LogSigmoidGrowthInitParametrization
)
