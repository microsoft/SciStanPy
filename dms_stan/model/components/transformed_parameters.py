"""Holds parameter transformations for DMS Stan models."""

from abc import abstractmethod
from typing import Optional, overload

import numpy as np
import numpy.typing as npt
import scipy.special as sp
import torch
import torch.nn.functional as F

import dms_stan as dms

from .abstract_model_component import AbstractModelComponent


@overload
def _choose_module(dist: torch.Tensor) -> torch: ...


@overload
def _choose_module(dist: dms.custom_types.SampleType) -> np: ...


def _choose_module(dist):
    """
    Choose the module to use for the operation based on the type of the distribution.
    """
    return torch if isinstance(dist, torch.Tensor) else np


class TransformableParameter:
    """
    Mixin class for parameters that can be transformed using mathematical operations.
    """

    def __add__(self, other: "dms.custom_types.CombinableParameterType"):
        return AddParameter(self, other)

    def __radd__(self, other: "dms.custom_types.CombinableParameterType"):
        return AddParameter(other, self)

    def __sub__(self, other: "dms.custom_types.CombinableParameterType"):
        return SubtractParameter(self, other)

    def __rsub__(self, other: "dms.custom_types.CombinableParameterType"):
        return SubtractParameter(other, self)

    def __mul__(self, other: "dms.custom_types.CombinableParameterType"):
        return MultiplyParameter(self, other)

    def __rmul__(self, other: "dms.custom_types.CombinableParameterType"):
        return MultiplyParameter(other, self)

    def __truediv__(self, other: "dms.custom_types.CombinableParameterType"):
        return DivideParameter(self, other)

    def __rtruediv__(self, other: "dms.custom_types.CombinableParameterType"):
        return DivideParameter(other, self)

    def __pow__(self, other: "dms.custom_types.CombinableParameterType"):
        return PowerParameter(self, other)

    def __rpow__(self, other: "dms.custom_types.CombinableParameterType"):
        return PowerParameter(other, self)

    def __neg__(self):
        return NegateParameter(self)


class Transformation(AbstractModelComponent):
    """
    Base class for transformations, including `transformed parameters` and
    `transformed data`.
    """

    def _transformation(self, index_opts: tuple[str, ...]) -> str:
        """Return the transformation for the parameter."""
        return f"{self.get_indexed_varname(index_opts)} = " + self.get_right_side(
            index_opts
        )

    @abstractmethod
    def _write_operation(self, **to_format: str) -> str:
        """Write the operation in Stan code."""

    def get_right_side(self, index_opts: tuple[str, ...] | None) -> str:
        """Gets the right-hand-side of the assignment operation for this parameter."""
        # Call the inherited method to get a dictionary mapping parent names to
        # either their indexed variable names (if they are named) or the thread
        # of operations that define them (if they are not named).
        components = super().get_right_side(index_opts)

        # Wrap the declaration for any unnamed parents in parentheses. This is to
        # ensure that the order of operations is correct in Stan.
        components = {
            name: (value if self._parents[name] else f"( {value} )")
            for name, value in components.items()
        }

        # Format the right-hand side of the operation. Exactly how formatting is
        # done depends on the child class.
        return self._write_operation(**components)

    def __str__(self) -> str:
        right_side = (
            self.get_right_side(None).replace("[start:end]", "").replace("__", ".")
        )
        return f"{self.model_varname} = {right_side}"


class TransformedParameter(Transformation, TransformableParameter):
    """
    Base class representing a parameter that is the result of combining other
    parameters using mathematical operations.
    """

    STAN_OPERATOR: str = ""  # Operator for the operation in Stan

    # The transformation is renamed to be more specific in the child classes
    get_transformation_assignment = Transformation._transformation

    def _draw(
        self,
        n: int,
        level_draws: dict[str, npt.NDArray],
        seed: Optional[int],  # pylint: disable=unused-argument
    ) -> npt.NDArray:
        """Sample from this parameter's distribution `n` times."""
        # Perform the operation on the draws
        return self.operation(**level_draws)

    @overload
    def operation(self, **draws: torch.Tensor) -> torch.Tensor: ...

    @overload
    def operation(self, **draws: "dms.custom_types.SampleType") -> npt.NDArray: ...

    @abstractmethod
    def operation(self, **draws):
        """Perform the operation on the draws or torch parameters."""

    @abstractmethod
    def _write_operation(self, **to_format: str) -> str:
        """Write the operation in Stan code."""
        # The Stan operator must be defined in the child class
        if self.STAN_OPERATOR == "":
            raise NotImplementedError("The STAN_OPERATOR must be defined.")

        return ""

    # Calling this class should return the result of the operation.
    def __call__(self, *args, **kwargs):
        return self.operation(*args, **kwargs)

    @property
    def torch_parametrization(self) -> torch.Tensor:
        # This is just the operation performed on the torch parameters of the parents
        return self.operation(
            **{
                name: param.torch_parametrization
                for name, param in self._parents.items()
            }
        )


class BinaryTransformedParameter(TransformedParameter):
    """
    Identical to the TransformedParameter class, but only for operations involving
    two parameters. In other words, two parameters must be passed to the class.
    """

    def __init__(
        self,
        dist1: "dms.custom_types.CombinableParameterType",
        dist2: "dms.custom_types.CombinableParameterType",
        **kwargs,
    ):
        super().__init__(dist1=dist1, dist2=dist2, **kwargs)

    # pylint: disable=arguments-differ
    @overload
    def operation(self, dist1: torch.Tensor, dist2: torch.Tensor) -> torch.Tensor: ...

    @overload
    def operation(
        self, dist1: "dms.custom_types.SampleType", dist2: "dms.custom_types.SampleType"
    ) -> npt.NDArray: ...

    @abstractmethod
    def operation(self, dist1, dist2): ...

    def _write_operation(self, dist1: str, dist2: str) -> str:
        super()._write_operation()
        return f"{dist1} {self.STAN_OPERATOR} {dist2}"

    # pylint: enable=arguments-differ


class UnaryTransformedParameter(TransformedParameter):
    """Transformed parameter that only requires one parameter."""

    def __init__(
        self,
        dist1: "dms.custom_types.CombinableParameterType",
        **kwargs,
    ):
        super().__init__(dist1=dist1, **kwargs)

    # pylint: disable=arguments-differ
    @overload
    def operation(self, dist1: torch.Tensor) -> torch.Tensor: ...

    @overload
    def operation(self, dist1: "dms.custom_types.SampleType") -> npt.NDArray: ...

    @abstractmethod
    def operation(self, dist1): ...

    def _write_operation(self, dist1: str) -> str:
        super()._write_operation()
        return f"{self.STAN_OPERATOR}{dist1}"

    # pylint: enable=arguments-differ


class AddParameter(BinaryTransformedParameter):
    """Defines a parameter that is the sum of two other parameters."""

    STAN_OPERATOR: str = "+"

    def operation(self, dist1, dist2):
        return dist1 + dist2


class SubtractParameter(BinaryTransformedParameter):
    """Defines a parameter that is the difference of two other parameters."""

    STAN_OPERATOR: str = "-"

    def operation(self, dist1, dist2):
        return dist1 - dist2


class MultiplyParameter(BinaryTransformedParameter):
    """Defines a parameter that is the product of two other parameters."""

    STAN_OPERATOR: str = ".*"

    def operation(self, dist1, dist2):
        return dist1 * dist2


class DivideParameter(BinaryTransformedParameter):
    """Defines a parameter that is the quotient of two other parameters."""

    STAN_OPERATOR: str = "./"

    def operation(self, dist1, dist2):
        return dist1 / dist2


class PowerParameter(BinaryTransformedParameter):
    """Defines a parameter raised to the power of another parameter."""

    STAN_OPERATOR: str = ".^"

    def operation(self, dist1, dist2):
        return dist1**dist2


class NegateParameter(UnaryTransformedParameter):
    """Defines a parameter that is the negative of another parameter."""

    STAN_OPERATOR: str = "-"

    def operation(self, dist1):
        return -dist1


class AbsParameter(UnaryTransformedParameter):
    """Defines a parameter that is the absolute value of another."""

    LOWER_BOUND: float = 0.0

    def operation(self, dist1):
        return _choose_module(dist1).abs(dist1)

    def _write_operation(self, dist1: str) -> str:
        return f"abs({dist1})"


class LogParameter(UnaryTransformedParameter):
    """Defines a parameter that is the natural logarithm of another."""

    # The distribution must be positive
    POSITIVE_PARAMS = {"dist1"}

    def operation(self, dist1):
        return _choose_module(dist1).log(dist1)

    def _write_operation(self, dist1: str) -> str:
        return f"log({dist1})"


class ExpParameter(UnaryTransformedParameter):
    """Defines a parameter that is the exponential of another."""

    LOWER_BOUND: float = 0.0

    def operation(self, dist1):

        return _choose_module(dist1).exp(dist1)

    def _write_operation(self, dist1: str) -> str:
        return f"exp({dist1})"


class NormalizeParameter(UnaryTransformedParameter):
    """Defines a parameter that is normalized to sum to 1."""

    LOWER_BOUND: float = 0.0
    UPPER_BOUND: float = 1.0

    def operation(self, dist1):
        if isinstance(dist1, torch.Tensor):
            return dist1 / dist1.sum(dim=-1, keepdim=True)
        else:
            return dist1 / np.sum(dist1, keepdims=True, axis=-1)

    def _write_operation(self, dist1: str) -> str:
        return f"{dist1} / sum({dist1})"


class NormalizeLogParameter(UnaryTransformedParameter):
    """
    Defines a parameter that is normalized such that exp(x) sums to 1. By extension,
    this assumes that the input is log-transformed.
    """

    UPPER_BOUND: float = 0.0

    def operation(self, dist1):
        if isinstance(dist1, torch.Tensor):
            return dist1 - torch.logsumexp(dist1, keepdims=True, dim=-1)
        else:
            return dist1 - sp.logsumexp(dist1, keepdims=True, axis=-1)

    def _write_operation(self, dist1: str) -> str:
        return f"{dist1} - log_sum_exp({dist1})"


class LogSumExpParameter(UnaryTransformedParameter):
    """
    Defines a parameter that computes the log of the sum of exponentials of another.
    This can only be applied over the last dimension and occurs with or without
    `keepdims` set to True.
    """

    def __init__(
        self,
        dist1: "dms.custom_types.CombinableParameterType",
        keepdims: bool = False,
        **kwargs,
    ):
        """
        Initializes the LogSumExpParameter. This applies the log-sum-exp operation
        over the last dimension of the input parameter, either with or without
        keeping the last dimension.
        """
        # Record whether to keep the last dimension
        self.keepdims = keepdims

        # The shape is the leading dimensions of the input parameter plus a singleton
        # dimension if keepdims is True.
        shape = dist1.shape[:-1]
        if keepdims:
            shape += (1,)

        # Init as normal
        super().__init__(dist1=dist1, shape=shape, **kwargs)

    # No shape checking for this class
    def _set_shape(self, *args, **kwargs):
        pass

    def operation(self, dist1):
        if isinstance(dist1, torch.Tensor):
            return torch.logsumexp(dist1, keepdim=self.keepdims, dim=-1)
        else:
            return sp.logsumexp(dist1, keepdims=self.keepdims, axis=-1)

    def _write_operation(self, dist1: str) -> str:
        return f"log_sum_exp({dist1})"


class SigmoidParameter(UnaryTransformedParameter):
    """Defines a parameter that is the sigmoid of another."""

    UPPER_BOUND: float = 1.0
    LOWER_BOUND: float = 0.0

    def operation(self, dist1):
        """
        Calculates the inverse logit (sigmoid) function in a numerically stable
        way.
        """
        # If using torch, use the sigmoid function directly.
        if isinstance(dist1, torch.Tensor):
            return torch.sigmoid(dist1)

        # If using numpy, we manually calculate the sigmoid function using a more
        # numerically stable approach.
        elif isinstance(dist1, np.ndarray):
            return dms.utils.stable_sigmoid(dist1)

        # If using a different type, raise an error.
        else:
            raise TypeError(
                "Unsupported type for dist1. Expected torch.Tensor or np.ndarray."
            )

    def _write_operation(self, dist1: str) -> str:
        return f"inv_logit({dist1})"


class LogSigmoidParameter(UnaryTransformedParameter):
    """Defines a parameter that is the log of the sigmoid of another."""

    UPPER_BOUND: float = 0.0

    def operation(self, dist1):
        if isinstance(dist1, torch.Tensor):
            return F.logsigmoid(dist1)  # pylint: disable=not-callable
        elif isinstance(dist1, np.ndarray):
            return np.log(dms.utils.stable_sigmoid(dist1))
        else:
            raise TypeError(
                "Unsupported type for dist1. Expected torch.Tensor or np.ndarray."
            )

    def _write_operation(self, dist1: str) -> str:
        return f"log_inv_logit({dist1})"


class ExponentialGrowth(ExpParameter):
    r"""
    A transformed parameter that models exponential growth. Specifically, parameters
    `t`, `A`, and `r` are used to calculate the exponential growth model as follows:

    $$
    x = A\textrm{e}^{rt)
    $$
    """

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        t: "dms.custom_types.CombinableParameterType",
        A: "dms.custom_types.CombinableParameterType",
        r: "dms.custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initializes the LogExponentialGrowth distribution.

        Args:
            t ("dms.custom_types.SampleType"): The time parameter.

            A ("dms.custom_types.SampleType"): The amplitude parameter.

            r ("dms.custom_types.SampleType"): The rate parameter.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super(UnaryTransformedParameter, self).__init__(t=t, A=A, r=r, **kwargs)

    # pylint: disable=arguments-differ
    @overload
    def operation(
        self, t: torch.Tensor, A: torch.Tensor, r: torch.Tensor
    ) -> torch.Tensor: ...

    @overload
    def operation(
        self,
        t: "dms.custom_types.SampleType",
        A: "dms.custom_types.SampleType",
        r: "dms.custom_types.SampleType",
    ) -> npt.NDArray: ...

    def operation(self, *, t, A, r):
        return A * super().operation(r * t)

    # pylint: enable=arguments-differ

    def _write_operation(  # pylint: disable=arguments-differ
        self, t: str, A: str, r: str
    ) -> str:
        par_string = super()._write_operation(f"{r} .* {t}")
        return f"{A} .* {par_string}"


class BinaryExponentialGrowth(ExpParameter):
    """
    Special case of `ExponentialGrowth` used for modeling when only two timepoints
    are available. In this case, we assume that t0 = 0 and t1 = 1, reducing the
    operation to:

    $$
    x = A\textrm{e}^{r}
    $$

    """

    def __init__(
        self,
        A: "dms.custom_types.CombinableParameterType",
        r: "dms.custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initializes the BinaryExponentialGrowth distribution.

        Args:
            A ("dms.custom_types.SampleType"): The amplitude parameter.

            r ("dms.custom_types.SampleType"): The rate parameter.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super(UnaryTransformedParameter, self).__init__(A=A, r=r, **kwargs)

    # pylint: disable=arguments-differ
    @overload
    def operation(self, A: torch.Tensor, r: torch.Tensor) -> torch.Tensor: ...
    @overload
    def operation(
        self,
        A: "dms.custom_types.SampleType",
        r: "dms.custom_types.SampleType",
    ) -> npt.NDArray: ...
    def operation(self, *, A, r):
        return A * super().operation(r)

    def _write_operation(self, A: str, r: str) -> str:
        return f"{A} .* {super()._write_operation(r)}"

    # pylint: enable=arguments-differ


class LogExponentialGrowth(TransformedParameter):
    """
    A distribution that models the natural log of the `ExponentialGrowth` distribution.
    Specifically, parameters `t`, `log_A`, and `r` are used to calculate the log
    of the exponential growth model as follows:

    $$
    log(x) = log_A + rt
    $$

    Note that, with this parametrization, we guarantee that $x > 0$.

    This parametrization is particularly useful for modeling the proportions of
    different populations as is done in DMS Stan, as proportions are always positive.
    """

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        t: "dms.custom_types.CombinableParameterType",
        log_A: "dms.custom_types.CombinableParameterType",
        r: "dms.custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initializes the LogExponentialGrowth distribution.

        Args:
            t ("dms.custom_types.SampleType"): The time parameter.

            log_A ("dms.custom_types.SampleType"): The log of the amplitude parameter.

            r ("dms.custom_types.SampleType"): The rate parameter.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super().__init__(t=t, log_A=log_A, r=r, **kwargs)

    # pylint: disable=arguments-differ
    @overload
    def operation(
        self, t: torch.Tensor, log_A: torch.Tensor, r: torch.Tensor
    ) -> torch.Tensor: ...

    @overload
    def operation(
        self,
        t: "dms.custom_types.SampleType",
        log_A: "dms.custom_types.SampleType",
        r: "dms.custom_types.SampleType",
    ) -> npt.NDArray: ...

    def operation(self, *, t, log_A, r):
        return log_A + r * t

    # pylint: enable=arguments-differ

    def _write_operation(  # pylint: disable=arguments-differ
        self, t: str, log_A: str, r: str
    ) -> str:
        return f"{log_A} + {r} .* {t}"


class BinaryLogExponentialGrowth(TransformedParameter):
    """
    Special case of `LogExponentialGrowth` used for modeling when only two timepoints
    are available. In this case, we assume that t0 = 0 and t1 = 1, reducing the
    operation to:

    $$
    log(x) = log_A + r
    $$

    """

    def __init__(
        self,
        log_A: "dms.custom_types.CombinableParameterType",
        r: "dms.custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initializes the BinaryLogExponentialGrowth distribution.

        Args:
            log_A ("dms.custom_types.SampleType"): The log of the amplitude parameter.

            r ("dms.custom_types.SampleType"): The rate parameter.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super().__init__(log_A=log_A, r=r, **kwargs)

    # pylint: disable=arguments-differ
    @overload
    def operation(self, log_A: torch.Tensor, r: torch.Tensor) -> torch.Tensor: ...
    @overload
    def operation(
        self,
        log_A: "dms.custom_types.SampleType",
        r: "dms.custom_types.SampleType",
    ) -> npt.NDArray: ...
    def operation(self, *, log_A, r):
        return log_A + r

    def _write_operation(self, log_A: str, r: str) -> str:
        return f"{log_A} + {r}"

    # pylint: enable=arguments-differ


class SigmoidGrowth(SigmoidParameter):
    r"""
    A transformed parameter that models sigmoid growth. Specifically, parameters
    `t`, `A`, `r`, and `c` are used to calculate the sigmoid growth model as follows:

    $$
    x = \frac{A}{1 + \textrm{e}^{-r(t - c)}}
    $$
    """

    LOWER_BOUND: float = 0.0
    UPPER_BOUND: None = None

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        t: "dms.custom_types.CombinableParameterType",
        A: "dms.custom_types.CombinableParameterType",
        r: "dms.custom_types.CombinableParameterType",
        c: "dms.custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initializes the LogSigmoidGrowth distribution.

        Args:
            t ("dms.custom_types.SampleType"): The time parameter.

            A ("dms.custom_types.SampleType"): The amplitude parameter. Specifically,
                this is the maximum value of the sigmoid function.

            r ("dms.custom_types.SampleType"): The rate parameter.

            c ("dms.custom_types.SampleType"): The offset parameter. This is the
                inflection point of the sigmoid function.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super(UnaryTransformedParameter, self).__init__(t=t, A=A, r=r, c=c, **kwargs)

    # pylint: disable=arguments-differ
    @overload
    def operation(
        self, t: torch.Tensor, A: torch.Tensor, r: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor: ...

    @overload
    def operation(
        self,
        t: "dms.custom_types.SampleType",
        A: "dms.custom_types.SampleType",
        r: "dms.custom_types.SampleType",
        c: "dms.custom_types.SampleType",
    ) -> npt.NDArray: ...

    def operation(self, *, t, A, r, c):
        return A * super().operation(r * (t - c))

    # pylint: enable=arguments-differ

    def _write_operation(  # pylint: disable=arguments-differ
        self, t: str, A: str, r: str, c: str
    ) -> str:
        par_string = super()._write_operation(f"{r} .* ({t} - {c})")
        return f"{A} .* {par_string}"


class LogSigmoidGrowth(LogSigmoidParameter):
    r"""
    A distribution that models the natural log of the `SigmoidGrowth` distribution.
    Specifically, parameters `t`, `log_A`, `r`, and `c` are used to calculate
    the log of the sigmoid growth model as follows:

    $$
    log(x) = log_A - log(1 + \textrm{e}^{-r(t - c)})
    $$

    As with the `LogExponentialGrowth` distribution, this parametrization guarantees
    that $x > 0$.
    """

    LOWER_BOUND: None = None
    UPPER_BOUND: None = None

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        t: "dms.custom_types.CombinableParameterType",
        log_A: "dms.custom_types.CombinableParameterType",
        r: "dms.custom_types.CombinableParameterType",
        c: "dms.custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initializes the LogSigmoidGrowth distribution.

        Args:
            exp_t ("dms.custom_types.SampleType"): The exponentiated time parameter.

            log_A ("dms.custom_types.SampleType"): The log of the amplitude parameter.

            r ("dms.custom_types.SampleType"): The rate parameter.

            c ("dms.custom_types.SampleType"): The offset parameter.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super(UnaryTransformedParameter, self).__init__(
            t=t, log_A=log_A, r=r, c=c, **kwargs
        )

    # pylint: disable=arguments-differ
    @overload
    def operation(
        self, t: torch.Tensor, log_A: torch.Tensor, r: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor: ...

    @overload
    def operation(
        self,
        t: "dms.custom_types.SampleType",
        log_A: "dms.custom_types.SampleType",
        r: "dms.custom_types.SampleType",
        c: "dms.custom_types.SampleType",
    ) -> npt.NDArray: ...

    def operation(self, *, t, log_A, r, c):
        return log_A + super().operation(r * (t - c))

    # pylint: enable=arguments-differ

    def _write_operation(  # pylint: disable=arguments-differ
        self, t: str, log_A: str, r: str, c: str
    ) -> str:
        par_string = super()._write_operation(f"{r} .* ({t} - {c})")
        return f"{log_A} + {par_string}"


class SigmoidGrowthInitParametrization(TransformedParameter):
    r"""
    An alternative parametrization of the sigmoid growth function that parametrizes
    in terms of starting abundances rather than the maximum abundances.
    """

    LOWER_BOUND: float = 0.0
    UPPER_BOUND: None = None

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        t: "dms.custom_types.CombinableParameterType",
        x0: "dms.custom_types.CombinableParameterType",  # pylint: disable=invalid-name
        r: "dms.custom_types.CombinableParameterType",
        c: "dms.custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initializes the SigmoidGrowthInitParametrization distribution.

        Args:
            t (dms.custom_types.CombinableParameterType): The time parameter.
            x0 (dms.custom_types.CombinableParameterType): Initial abundances.
            r (dms.custom_types.CombinableParameterType): Growth rate.
            c (dms.custom_types.CombinableParameterType): Offset parameter.
            shape (tuple[int, ...], optional): The shape of the distribution. Defaults
            to ().
        """
        # Initialize using the base transformed parameter class
        super().__init__(t=t, x0=x0, r=r, c=c, **kwargs)

    # pylint: disable=arguments-renamed, arguments-differ
    @overload
    def operation(
        self, t: torch.Tensor, x0: torch.Tensor, r: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor: ...

    @overload
    def operation(
        self,
        t: "dms.custom_types.SampleType",
        x0: "dms.custom_types.SampleType",  # pylint: disable=invalid-name
        r: "dms.custom_types.SampleType",
        c: "dms.custom_types.SampleType",
    ) -> npt.NDArray: ...

    def operation(self, t, x0, r, c):
        """We use a log-add-exp trick to calculate in a numerically stable way."""
        # Get the module
        mod = _choose_module(x0)

        # Get the fold-change. We use the log-add-exp function to calculate this
        # in a more numerically stable way
        zero = 0.0 if mod is np else torch.tensor(0.0, device=x0.device)
        foldchange = mod.exp(
            mod.logaddexp(zero, r * c) - mod.logaddexp(zero, r * (c - t))
        )

        # Calculate the abundance
        return x0 * foldchange

    def _write_operation(
        self, t: str, x0: str, r: str, c: str  # pylint: disable=invalid-name
    ) -> str:
        """Calculate using Stan's log1p_exp function."""
        return f"{x0} .* exp(log1p_exp({r} .* {c}) - log1p_exp({r} .* ({c} - {t})))"

    # pylint: enable=arguments-renamed, arguments-differ


class LogSigmoidGrowthInitParametrization(TransformedParameter):
    r"""
    An alternative parametrization of the log sigmoid growth function that
    parametrizes in terms of starting abundances rather than the maximum abundances.
    """

    LOWER_BOUND: None = None
    UPPER_BOUND: None = None

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        t: "dms.custom_types.CombinableParameterType",
        log_x0: "dms.custom_types.CombinableParameterType",  # pylint: disable=invalid-name
        r: "dms.custom_types.CombinableParameterType",
        c: "dms.custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initializes the LogSigmoidGrowthInitParametrization distribution.

        Args:
            t (dms.custom_types.CombinableParameterType): The time parameter.
            log_x0 (dms.custom_types.CombinableParameterType): Logarithm of initial
                abundances.
            r (dms.custom_types.CombinableParameterType): Growth rate.
            c (dms.custom_types.CombinableParameterType): Offset parameter.
            shape (tuple[int, ...], optional): The shape of the distribution. Defaults
            to ().
        """
        # Initialize using the base transformed parameter class
        super().__init__(t=t, log_x0=log_x0, r=r, c=c, **kwargs)

    # pylint: disable=arguments-renamed, arguments-differ
    @overload
    def operation(
        self,
        t: torch.Tensor,
        log_x0: torch.Tensor,  # pylint: disable=invalid-name
        r: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor: ...

    @overload
    def operation(
        self,
        t: "dms.custom_types.SampleType",
        log_x0: "dms.custom_types.SampleType",  # pylint: disable=invalid-name
        r: "dms.custom_types.SampleType",
        c: "dms.custom_types.SampleType",
    ) -> npt.NDArray: ...

    def operation(self, t, log_x0, r, c):
        """We use a log-add-exp trick to calculate in a numerically stable way."""
        # Get the module
        mod = _choose_module(log_x0)

        # Define zero
        zero = 0.0 if mod is np else torch.tensor(0.0, device=log_x0.device)

        # Calculate
        return log_x0 + mod.logaddexp(zero, r * c) - mod.logaddexp(zero, r * (c - t))

    def _write_operation(
        self, t: str, log_x0: str, r: str, c: str  # pylint: disable=invalid-name
    ) -> str:
        """Calculate using Stan's log1p_exp function."""
        return f"{log_x0} + log1p_exp({r} .* {c}) - log1p_exp({r} .* ({c} - {t}))"
