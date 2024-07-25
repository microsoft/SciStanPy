"""Defines custom operations that we can use in Stan models."""

from typing import Any, Callable

import numpy as np

from dms_stan.modeling import (
    AbsDistribution,
    AbstractDistribution,
    ExpDistribution,
    LogDistribution,
    Parameter,
    UnaryTransformedDistribution,
)


class Operation:
    """Base class for custom dms_stan operations."""


class UnaryOperation(Operation):
    """Base class for dms_stan unary operations."""

    def __init__(self, distclass: type[UnaryTransformedDistribution], func: Callable):
        """
        Sets up an operation, which is a unary transformation of a distribution.

        Args:
            distclass: The distribution class to use when applied to dms_stan distributions.
            func: The function to use when applied to anything else.
        """
        self.distclass = distclass
        self.func = func

    def __call__(self, x: Any, **kwargs) -> Any:
        """
        If x is a dms_stan distribution, apply the distribution transformation.
        Otherwise, apply the other function passed at init.

        Args:
            x: The input to the operation.
            **kwargs: Additional arguments to pass to the function, if any.

        Returns:
            The result of applying the operation to x.
        """
        # If a dms_stan parameter, transform the underlying distribution object
        # and return a new parameter
        if isinstance(x, Parameter):
            return Parameter(self.distclass(x.distribution), **kwargs)

        # If a dms_stan distribution, apply the distribution transformation
        if isinstance(x, AbstractDistribution):
            return self.distclass(x)

        # Otherwise, apply the numpy function
        return self.func(x, **kwargs)


# Define our operations
abs_ = UnaryOperation(AbsDistribution, np.abs)
exp = UnaryOperation(ExpDistribution, np.exp)
log = UnaryOperation(LogDistribution, np.log)
