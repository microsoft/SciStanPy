"""Holds parameter transformations for DMS Stan models."""

from abc import abstractmethod
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import scipy.special as sp

import dms_stan.model.components as dms_components

from .abstract_classes import AbstractParameter
from .constants import Constant
from .pytorch import (
    AbsTransformedContainer,
    AddTransformedContainer,
    BinaryTransformedContainer,
    DivideTransformedContainer,
    ExpTransformedContainer,
    LogExponentialGrowthContainer,
    LogSigmoidGrowthContainer,
    LogTransformedContainer,
    MultiplyTransformedContainer,
    NegateTransformedContainer,
    NormalizeLogTransformedContainer,
    NormalizeTransformedContainer,
    PowerTransformedContainer,
    SubtractTransformedContainer,
    TransformedContainer,
    UnaryTransformedContainer,
)


class TransformedParameter(AbstractParameter):
    """
    Base class representing a parameter that is the result of combining other
    parameters using mathematical operations.
    """

    _torch_container_class: TransformedContainer

    def draw(self, n: int) -> npt.NDArray:
        """Sample from this parameter's distribution `n` times."""

        # Perform the operation on the draws
        return self.operation(**super().draw(n))

    @abstractmethod
    def operation(
        self, **draws: "dms_components.custom_types.SampleType"
    ) -> npt.NDArray:
        """Perform the operation on the draws"""

    def get_transformation_assignment(self, index_opts: tuple[str, ...]) -> str:
        """Return the assignment for the transformation."""
        return (
            f"{self.get_indexed_varname(index_opts)} = "
            + self.get_stan_transformation(index_opts)[0]
        )

    # TODO: We need to handle the set hyperparameter values. How do we populate
    # them in the distribution?
    def get_stan_transformation(self, index_opts: tuple[str, ...]) -> tuple[str, bool]:
        """
        Return the Stan transformation for this parameter. This recursively calls
        the equivalent method on parent transformed parameters until we hit a
        non-transformed parameter.
        """
        # Recursively gather the transformations until we hit a non-transformed
        # parameter or a recorded variable
        to_format: dict[str, tuple[str, bool]] = {}
        for name, param in self.parameters.items():

            # If the parameter is non-transformed, record
            if isinstance(param, (Constant, dms_components.parameters.Parameter)):
                to_format[name] = param.get_indexed_varname(index_opts)

            # Otherwise, get the transformation of the parent unless the parent
            # transformation is named, in which case we just use the name
            elif isinstance(param, TransformedParameter):
                if param.is_named:
                    to_format[name] = param.get_indexed_varname(index_opts)
                else:
                    transformation, is_vector = param.get_stan_transformation(
                        index_opts
                    )
                    to_format[name] = (f"( {transformation} )", is_vector)

            # Otherwise, raise an error
            else:
                raise TypeError(f"Unknown model component type {type(param)}")

        # We are a vector if any formatted variables are vectors
        is_vector = any(is_vector for _, is_vector in to_format.values())

        # Format the transformation
        return self.format_stan_transformation(**to_format), is_vector

    @abstractmethod
    def format_stan_transformation(self, **param_vals: tuple[str, bool]) -> str:
        """Return the base Stan transformation for this parameter."""

    # Calling this class should return the result of the operation.
    def __call__(self, *args, **kwargs):
        return self.operation(*args, **kwargs)


class BinaryTransformedParameter(TransformedParameter):
    """
    Identical to the TransformedParameter class, but only for operations involving
    two parameters. In other words, two parameters must be passed to the class.
    """

    _torch_container_class: BinaryTransformedContainer

    def __init__(
        self,
        dist1: "dms_components.custom_types.CombinableParameterType",
        dist2: "dms_components.custom_types.CombinableParameterType",
        shape: Optional[tuple[int, ...]] = None,
    ):
        super().__init__(dist1=dist1, dist2=dist2, shape=shape)

    @abstractmethod
    def operation(  # pylint: disable=arguments-differ
        self,
        dist1: "dms_components.custom_types.SampleType",
        dist2: "dms_components.custom_types.SampleType",
    ): ...

    @abstractmethod
    def format_stan_transformation(  # pylint: disable=arguments-differ
        self, dist1: tuple[str, bool], dist2: tuple[str, bool]
    ) -> tuple[str, str, bool]:

        # Unpack the variable names and scalar flags
        dist1_name, dist1_is_vector = dist1
        dist2_name, dist2_is_vector = dist2

        # Return the names and whether or not this will be an elementwise operation
        return dist1_name, dist2_name, dist1_is_vector and dist2_is_vector


class UnaryTransformedParameter(TransformedParameter):
    """Transformed parameter that only requires one parameter."""

    _torch_container_class: UnaryTransformedContainer

    def __init__(
        self,
        dist1: "dms_components.custom_types.CombinableParameterType",
        shape: Optional[tuple[int, ...]] = None,
        **kwargs: Any,
    ):
        super().__init__(dist1=dist1, shape=shape)

        # Store the kwargs for the operation
        self.operation_kwargs = kwargs

    @abstractmethod
    def operation(  # pylint: disable=arguments-differ
        self, dist1: "dms_components.custom_types.SampleType"
    ) -> npt.NDArray: ...

    @abstractmethod
    def format_stan_transformation(  # pylint: disable=arguments-differ
        self, dist1: tuple[str, bool]
    ) -> tuple[str, bool]: ...


class AddParameter(BinaryTransformedParameter):
    """Defines a parameter that is the sum of two other parameters."""

    _torch_container_class = AddTransformedContainer

    def operation(
        self,
        dist1: "dms_components.custom_types.SampleType",
        dist2: "dms_components.custom_types.SampleType",
    ) -> npt.NDArray:
        return dist1 + dist2

    def format_stan_transformation(
        self,
        dist1: tuple[str, bool],
        dist2: tuple[str, bool],
    ) -> str:
        # Unpack the variable names
        dist1_name, dist2_name, _ = super().format_stan_transformation(dist1, dist2)

        return f"{dist1_name} + {dist2_name}"


class SubtractParameter(BinaryTransformedParameter):
    """Defines a parameter that is the difference of two other parameters."""

    _torch_container_class = SubtractTransformedContainer

    def operation(
        self,
        dist1: "dms_components.custom_types.SampleType",
        dist2: "dms_components.custom_types.SampleType",
    ) -> npt.NDArray:
        return dist1 - dist2

    def format_stan_transformation(
        self,
        dist1: tuple[str, bool],
        dist2: tuple[str, bool],
    ) -> str:
        # Unpack the variable names
        dist1_name, dist2_name, _ = super().format_stan_transformation(dist1, dist2)

        return f"{dist1_name} - {dist2_name}"


class MultiplyParameter(BinaryTransformedParameter):
    """Defines a parameter that is the product of two other parameters."""

    _torch_container_class = MultiplyTransformedContainer

    def operation(
        self,
        dist1: "dms_components.custom_types.SampleType",
        dist2: "dms_components.custom_types.SampleType",
    ) -> npt.NDArray:
        return dist1 * dist2

    def format_stan_transformation(
        self,
        dist1: tuple[str, bool],
        dist2: tuple[str, bool],
    ) -> str:
        # Unpack the variable names and determine if this is an elementwise operation
        dist1_name, dist2_name, elementwise = super().format_stan_transformation(
            dist1, dist2
        )

        # Get the operator
        operator = ".*" if elementwise else "*"

        return f"{dist1_name} {operator} {dist2_name}"


class DivideParameter(BinaryTransformedParameter):
    """Defines a parameter that is the quotient of two other parameters."""

    _torch_container_class = DivideTransformedContainer

    def operation(
        self,
        dist1: "dms_components.custom_types.SampleType",
        dist2: "dms_components.custom_types.SampleType",
    ) -> npt.NDArray:
        return dist1 / dist2

    def format_stan_transformation(
        self,
        dist1: tuple[str, bool],
        dist2: tuple[str, bool],
    ) -> str:
        # Unpack the variable names and determine if this is an elementwise operation
        dist1_name, dist2_name, elementwise = super().format_stan_transformation(
            dist1, dist2
        )

        # Get the operator
        operator = "./" if elementwise else "/"

        return f"{dist1_name} {operator} {dist2_name}"


class PowerParameter(BinaryTransformedParameter):
    """Defines a parameter raised to the power of another parameter."""

    _torch_container_class = PowerTransformedContainer

    def operation(
        self,
        dist1: "dms_components.custom_types.SampleType",
        dist2: "dms_components.custom_types.SampleType",
    ) -> npt.NDArray:
        return dist1**dist2

    def format_stan_transformation(
        self,
        dist1: tuple[str, bool],
        dist2: tuple[str, bool],
    ) -> str:
        # Unpack the variable names and determine if this is an elementwise operation
        dist1_name, dist2_name, elementwise = super().format_stan_transformation(
            dist1, dist2
        )

        # Get the operator
        operator = ".^" if elementwise else "^"

        return f"{dist1_name} {operator} {dist2_name}"


class NegateParameter(UnaryTransformedParameter):
    """Defines a parameter that is the negative of another parameter."""

    _torch_container_class = NegateTransformedContainer

    def operation(self, dist1: "dms_components.custom_types.SampleType") -> npt.NDArray:
        return -dist1

    def format_stan_transformation(self, dist1: tuple[str, bool]) -> str:
        return f"-{dist1[0]}"


class AbsParameter(UnaryTransformedParameter):
    """Defines a parameter that is the absolute value of another."""

    _torch_container_class = AbsTransformedContainer
    stan_lower_bound: float = 0.0

    def operation(self, dist1: "dms_components.custom_types.SampleType") -> npt.NDArray:
        return np.abs(dist1, **self.operation_kwargs)

    def format_stan_transformation(self, dist1: tuple[str, bool]) -> str:
        return f"abs({dist1[0]})"


class LogParameter(UnaryTransformedParameter):
    """Defines a parameter that is the natural logarithm of another."""

    # The distribution must be positive
    POSITIVE_PARAMS = {"dist1"}

    _torch_container_class = LogTransformedContainer
    stan_lower_bound: float = 0.0

    def operation(self, dist1: "dms_components.custom_types.SampleType") -> npt.NDArray:
        return np.log(dist1, **self.operation_kwargs)

    def format_stan_transformation(self, dist1: tuple[str, bool]) -> str:
        return f"log({dist1[0]})"


class ExpParameter(UnaryTransformedParameter):
    """Defines a parameter that is the exponential of another."""

    _torch_container_class = ExpTransformedContainer
    stan_lower_bound: float = 0.0

    def operation(self, dist1: "dms_components.custom_types.SampleType") -> npt.NDArray:
        return np.exp(dist1, **self.operation_kwargs)

    def format_stan_transformation(self, dist1: tuple[str, bool]) -> str:
        return f"exp({dist1[0]})"


class NormalizeParameter(UnaryTransformedParameter):
    """Defines a parameter that is normalized to sum to 1."""

    _torch_container_class = NormalizeTransformedContainer
    stan_lower_bound: float = 0.0
    stan_upper_bound: float = 1.0

    def operation(self, dist1: "dms_components.custom_types.SampleType") -> npt.NDArray:
        return dist1 / np.sum(dist1, keepdims=True, **self.operation_kwargs)

    def format_stan_transformation(self, dist1: tuple[str, bool]) -> str:
        return f"{dist1[0]} / sum({dist1[0]})"


class NormalizeLogParameter(UnaryTransformedParameter):
    """
    Defines a parameter that is normalized such that exp(x) sums to 1. By extension,
    this assumes that the input is log-transformed.
    """

    _torch_container_class = NormalizeLogTransformedContainer
    stan_upper_bound: float = 0.0

    def operation(self, dist1: "dms_components.custom_types.SampleType") -> npt.NDArray:
        return dist1 - sp.logsumexp(dist1, keepdims=True, **self.operation_kwargs)

    def format_stan_transformation(self, dist1: tuple[str, bool]) -> str:
        return f"{dist1[0]} - log_sum_exp({dist1[0]})"


class Growth(TransformedParameter):
    """Base class for growth models."""

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        t: "dms_components.custom_types.CombinableParameterType",
        shape: Optional[tuple[int, ...]] = None,
        **params: "dms_components.custom_types.CombinableParameterType",
    ):
        # Store all parameters as a list by calling the super class
        super().__init__(t=t, shape=shape, **params)


class LogExponentialGrowth(Growth):
    """
    A distribution that models the natural log of the `ExponentialGrowth` distribution.
    Specifically, parameters `t`, `log_A`, and `r` are used to calculate the log
    of the exponential growth model as follows:

    $$
    log(x) = log_A + rt
    $$

    Note that, with this parametrization, we guarantee that $x > 0$. It is also
    only defined for $A > 0$ and $r > 0$, assuming that the time parameter $t$ is
    always positive.

    This parametrization is particularly useful for modeling the proportions of
    different populations as is done in DMS Stan, as proportions are always positive.
    """

    _torch_container_class = LogExponentialGrowthContainer

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        t: "dms_components.custom_types.CombinableParameterType",
        log_A: "dms_components.custom_types.CombinableParameterType",
        r: "dms_components.custom_types.CombinableParameterType",
        shape: Optional[tuple[int, ...]] = None,
    ):
        """Initializes the LogExponentialGrowth distribution.

        Args:
            t ("dms_components.custom_types.SampleType"): The time parameter.

            log_A ("dms_components.custom_types.SampleType"): The log of the amplitude parameter.

            r ("dms_components.custom_types.SampleType"): The rate parameter.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super().__init__(t=t, log_A=log_A, r=r, shape=shape)

    def operation(  # pylint: disable=arguments-differ
        self,
        *,
        t: "dms_components.custom_types.SampleType",
        log_A: "dms_components.custom_types.SampleType",
        r: "dms_components.custom_types.SampleType",
    ) -> npt.NDArray:
        return log_A + r * t

    def format_stan_transformation(  # pylint: disable=arguments-differ
        self, t: tuple[str, bool], log_A: [str, bool], r: [str, bool]
    ) -> str:
        # Decide on operator between r and t
        operator = ".*" if t[1] and r[1] else "*"

        # Build the transformation
        return f"{log_A[0]} + {r[0]} {operator} {t[0]}"


class LogSigmoidGrowth(Growth):
    r"""
    A distribution that models the natural log of the `SigmoidGrowth` distribution.
    Specifically, parameters `t`, `log_A`, `r`, and `c` are used to calculate the
    log of the sigmoid growth model as follows:

    $$
    log(x) = log_A - log(1 + \textrm{e}^{-r(t - c)})
    $$

    As with the `LogExponentialGrowth` distribution, this parametrization guarantees
    that $x > 0$.
    """

    _torch_container_class = LogSigmoidGrowthContainer

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        t: "dms_components.custom_types.CombinableParameterType",
        log_A: "dms_components.custom_types.CombinableParameterType",
        r: "dms_components.custom_types.CombinableParameterType",
        c: "dms_components.custom_types.CombinableParameterType",
        shape: Optional[tuple[int, ...]] = None,
    ):
        """Initializes the LogSigmoidGrowth distribution.

        Args:
            t ("dms_components.custom_types.SampleType"): The time parameter.

            log_A ("dms_components.custom_types.SampleType"): The log of the amplitude parameter.

            r ("dms_components.custom_types.SampleType"): The rate parameter.

            c ("dms_components.custom_types.SampleType"): The offset parameter.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super().__init__(t=t, log_A=log_A, r=r, c=c, shape=shape)

    def operation(  # pylint: disable=arguments-differ
        self,
        *,
        t: "dms_components.custom_types.SampleType",
        log_A: "dms_components.custom_types.SampleType",
        r: "dms_components.custom_types.SampleType",
        c: "dms_components.custom_types.SampleType",
    ) -> npt.NDArray:
        return log_A - np.log(1 + np.exp(-r * (t - c)))

    def format_stan_transformation(  # pylint: disable=arguments-differ
        self, t: str, log_A: str, r: str, c: str
    ) -> str:
        # Determine the operator between r and t
        operator = ".*" if r[1] and t[1] else "*"

        return f"{log_A} - log(1 + exp(-{r} {operator} ({t} - {c})))"
