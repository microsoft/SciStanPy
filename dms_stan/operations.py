"""Defines custom operations that we can use in Stan models."""

from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, overload

import numpy as np
import numpy.typing as npt
import scipy.special as sp

from dms_stan.model.components import (
    AbsParameter,
    ExpParameter,
    LogParameter,
    NormalizeLogParameter,
    NormalizeParameter,
    UnaryTransformedParameter,
)
from dms_stan.model.components.abstract_model_component import AbstractModelComponent


def _normalize_parameter(
    x: npt.NDArray[np.floating], **kwargs
) -> npt.NDArray[np.floating]:
    """
    Normalize such that x sums to 1.

    Args:
        x: The parameter.
        **kwargs: Additional arguments to pass to np.sum.

    Returns:
        The normalized parameter.
    """
    return x / np.sum(x, **kwargs)


def _normalize_log_parameter(
    log_x: npt.NDArray[np.floating], **kwargs
) -> npt.NDArray[np.floating]:
    """
    Normalize such that exp(x) sums to 1. By extension, this assumes that the input
    is log-transformed.

    Args:
        log_x: The log-transformed parameter.
        **kwargs: Additional arguments to pass to sp.logsumexp.

    Returns:
        The normalized parameter.
    """
    return log_x - sp.logsumexp(log_x, **kwargs)


class MetaOperation(ABCMeta):
    """
    Metaclass for operations. This sets the __doc__ attribute of the __call__ method
    of the class based on the function that the operation will be performing.
    """

    def __new__(
        mcs,
        classname,
        class_parents,
        class_attrs,
        distclass: Optional[type[UnaryTransformedParameter]] = None,
        func: Optional[Callable[..., npt.NDArray]] = None,
        funcname: Optional[str] = None,
    ):

        # Update the class attributes with the distclass and func
        class_attrs["distclass"] = distclass
        class_attrs["func"] = func

        # Create the class
        cls = super().__new__(mcs, classname, class_parents, class_attrs)

        # Update the docstring of the call method based on the function
        if func is not None:

            # Get the name. We might need to walk up the MRO to find the docstring.
            for parent in cls.mro():
                if (template_doc := parent.__call__.__doc__) is not None:
                    break
            else:
                raise ValueError("Could not find a docstring in the MRO.")

            # Update the docstring
            cls.__call__.__doc__ = template_doc.replace(
                "PLACEHOLDER", funcname if funcname else func.__name__
            )

        return cls


class Operation(metaclass=MetaOperation):
    """
    Base class for custom dms_stan operations. The distclass attribute is the class
    to use when applying the operation to dms_stan parameters, while the func attribute
    is the function to use when applying the operation to numpy arrays.
    """

    distclass: type[UnaryTransformedParameter]
    func: Callable[..., npt.NDArray]

    def __init__(self):
        """Makes sure that `distclass` and `func` are both set."""
        if not hasattr(self, "distclass") or self.distclass is None:
            raise ValueError("`distclass` must be set.")
        if not hasattr(self, "func") or self.func is None:
            raise ValueError("`func` must be set.")

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Apply the operation to the input.

        Args:
            *args: The input to the operation. If a dms_stan parameter, the operation
                is performed on the parameter. Otherwise, the operation is performed
                using PLACEHOLDER with *args passed to that function.

            **kwargs: Additional arguments to pass to PLACEHOLDER.

        Returns:
            The result of applying the operation to the input.
        """


class UnaryOperation(Operation):
    """Base class for dms_stan unary operations."""

    @overload
    def __call__(self, x: AbstractModelComponent) -> AbstractModelComponent: ...

    @overload
    def __call__(
        self, x: npt.NDArray[np.floating], **kwargs
    ) -> npt.NDArray[np.floating]: ...

    def __call__(self, x, **kwargs):
        """
        If x is a dms_stan parameter, apply the parameter transformation.
        Otherwise, apply PLACEHOLDER.

        Args:
            x: The input to the operation. If a dms_stan parameter, the operation
                is performed on the parameter. Otherwise, the operation is performed
                using the PLACEHOLDER function with `x` as the first argument.

            **kwargs: Additional arguments to pass to PLACEHOLDER. Note that these
                are only used if x is not a dms_stan parameter.

        Returns:
            The result of applying the operation to x.
        """
        # If a dms_stan parameter, apply the transformation
        if isinstance(x, AbstractModelComponent):
            return self.distclass(x)

        # Otherwise, apply the numpy function
        return self.func(x, **kwargs)


def _unary_transformation_factory(
    distclass: type[UnaryTransformedParameter],
    func: Callable[..., npt.NDArray],
    funcname: Optional[str] = None,
) -> UnaryOperation:
    """
    Constructor for our unary operations. See the `MetaOperation` metaclass for
    more details on the arguments to this function
    """

    # Define a new class
    class DmsStanUnaryOperation(  # pylint: disable=missing-class-docstring
        UnaryOperation, distclass=distclass, func=func, funcname=funcname
    ): ...

    # Create an instance of the class and return it
    return DmsStanUnaryOperation()


# Define our operations
abs_ = _unary_transformation_factory(AbsParameter, np.abs, "np.abs")
exp = _unary_transformation_factory(ExpParameter, np.exp, "np.exp")
log = _unary_transformation_factory(LogParameter, np.log, "np.log")
normalize = _unary_transformation_factory(NormalizeParameter, _normalize_parameter)
normalize_log = _unary_transformation_factory(
    NormalizeLogParameter, _normalize_log_parameter
)
