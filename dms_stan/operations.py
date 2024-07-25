"""Defines custom operations that we can use in Stan models."""

from typing import Any, Callable

import numpy as np

from dms_stan.param import (
    AbsParameter,
    AbstractParameter,
    ExpParameter,
    LogParameter,
    UnaryTransformedParameter,
)


class Operation:
    """Base class for custom dms_stan operations."""


class UnaryOperation(Operation):
    """Base class for dms_stan unary operations."""

    def __init__(self, distclass: type[UnaryTransformedParameter], func: Callable):
        """
        Sets up an operation, which is a unary transformation of a parameter.

        Args:
            distclass: The parameter class to use when applied to dms_stan parameters.
            func: The function to use when applied to anything else.
        """
        self.distclass = distclass
        self.func = func

    def __call__(self, x: Any, **kwargs) -> Any:
        """
        If x is a dms_stan parameter, apply the parameter transformation.
        Otherwise, apply the other function passed at init.

        Args:
            x: The input to the operation.
            **kwargs: Additional arguments to pass to the function, if any.

        Returns:
            The result of applying the operation to x.
        """
        # If a dms_stan parameter, apply the transformation
        if isinstance(x, AbstractParameter):
            return self.distclass(x)

        # Otherwise, apply the numpy function
        return self.func(x, **kwargs)


# Define our operations
abs_ = UnaryOperation(AbsParameter, np.abs)
exp = UnaryOperation(ExpParameter, np.exp)
log = UnaryOperation(LogParameter, np.log)
