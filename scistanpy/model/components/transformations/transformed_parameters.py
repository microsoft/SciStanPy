"""Holds parameter transformations for SciStanPy models."""

from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Optional, overload, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import scipy.special as sp
import torch
import torch.nn.functional as F

from scistanpy import utils
from scistanpy.model.components import abstract_model_component, constants

if TYPE_CHECKING:
    from scistanpy import custom_types

# pylint: disable=too-many-lines


class TransformableParameter:
    """
    Mixin class for parameters that can be transformed using mathematical operations.
    """

    def __add__(self, other: "custom_types.CombinableParameterType"):
        return AddParameter(self, other)

    def __radd__(self, other: "custom_types.CombinableParameterType"):
        return AddParameter(other, self)

    def __sub__(self, other: "custom_types.CombinableParameterType"):
        return SubtractParameter(self, other)

    def __rsub__(self, other: "custom_types.CombinableParameterType"):
        return SubtractParameter(other, self)

    def __mul__(self, other: "custom_types.CombinableParameterType"):
        return MultiplyParameter(self, other)

    def __rmul__(self, other: "custom_types.CombinableParameterType"):
        return MultiplyParameter(other, self)

    def __truediv__(self, other: "custom_types.CombinableParameterType"):
        return DivideParameter(self, other)

    def __rtruediv__(self, other: "custom_types.CombinableParameterType"):
        return DivideParameter(other, self)

    def __pow__(self, other: "custom_types.CombinableParameterType"):
        return PowerParameter(self, other)

    def __rpow__(self, other: "custom_types.CombinableParameterType"):
        return PowerParameter(other, self)

    def __neg__(self):
        return NegateParameter(self)


class Transformation(abstract_model_component.AbstractModelComponent):
    """
    Base class for transformations, including `transformed parameters` and
    `transformed data`.
    """

    SHAPE_CHECK: bool = True

    def _transformation(self, index_opts: tuple[str, ...]) -> str:
        """Return the transformation for the parameter."""
        return f"{self.get_indexed_varname(index_opts)} = " + self.get_right_side(
            index_opts
        )

    @abstractmethod
    def write_stan_operation(self, **to_format: str) -> str:
        """Write the operation in Stan code."""

    def get_right_side(
        self,
        index_opts: tuple[str, ...] | None,
        start_dims: dict[str, "custom_types.Integer"] | None = None,
        end_dims: dict[str, "custom_types.Integer"] | None = None,
    ) -> str:
        """Gets the right-hand-side of the assignment operation for this parameter."""
        # Call the inherited method to get a dictionary mapping parent names to
        # either their indexed variable names (if they are named) or the thread
        # of operations that define them (if they are not named).
        components = super().get_right_side(
            index_opts, start_dims=start_dims, end_dims=end_dims
        )

        # Wrap the declaration for any unnamed parents in parentheses. This is to
        # ensure that the order of operations is correct in Stan.
        components = {
            name: (value if self._parents[name].is_named else f"( {value} )")
            for name, value in components.items()
        }

        # Format the right-hand side of the operation. Exactly how formatting is
        # done depends on the child class.
        return self.write_stan_operation(**components)

    def _set_shape(self) -> None:
        """
        Some transformations are reductions. When they are reductions, we skip
        checking the shape
        """
        if self.SHAPE_CHECK:
            super()._set_shape()

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
        n: "custom_types.Integer",
        level_draws: dict[str, npt.NDArray],
        seed: Optional["custom_types.Integer"],  # pylint: disable=unused-argument
    ) -> npt.NDArray:
        """Sample from this parameter's distribution `n` times."""
        # Perform the operation on the draws
        return self.run_np_torch_op(**level_draws)

    @overload
    def run_np_torch_op(self, **draws: torch.Tensor) -> torch.Tensor: ...

    @overload
    def run_np_torch_op(self, **draws: "custom_types.SampleType") -> npt.NDArray: ...

    @abstractmethod
    def run_np_torch_op(self, **draws):
        """Perform the operation on the draws or torch parameters."""

    @abstractmethod
    def write_stan_operation(self, **to_format: str) -> str:
        """Write the operation in Stan code."""
        # The Stan operator must be defined in the child class
        if self.STAN_OPERATOR == "":
            raise NotImplementedError("The STAN_OPERATOR must be defined.")

        return ""

    # Calling this class should return the result of the operation.
    def __call__(self, *args, **kwargs):
        return self.run_np_torch_op(*args, **kwargs)

    @property
    def torch_parametrization(self) -> torch.Tensor:
        # This is just the operation performed on the torch parameters of the parents
        return self.run_np_torch_op(
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
        dist1: "custom_types.CombinableParameterType",
        dist2: "custom_types.CombinableParameterType",
        **kwargs,
    ):
        super().__init__(dist1=dist1, dist2=dist2, **kwargs)

    # pylint: disable=arguments-differ
    @overload
    def run_np_torch_op(
        self, dist1: torch.Tensor, dist2: torch.Tensor
    ) -> torch.Tensor: ...

    @overload
    def run_np_torch_op(
        self, dist1: "custom_types.SampleType", dist2: "custom_types.SampleType"
    ) -> npt.NDArray: ...

    @abstractmethod
    def run_np_torch_op(self, dist1, dist2): ...

    def write_stan_operation(self, dist1: str, dist2: str) -> str:
        super().write_stan_operation()
        return f"{dist1} {self.STAN_OPERATOR} {dist2}"

    # pylint: enable=arguments-differ


class UnaryTransformedParameter(TransformedParameter):
    """Transformed parameter that only requires one parameter."""

    def __init__(
        self,
        dist1: "custom_types.CombinableParameterType",
        **kwargs,
    ):
        super().__init__(dist1=dist1, **kwargs)

    # pylint: disable=arguments-differ
    @overload
    def run_np_torch_op(self, dist1: torch.Tensor) -> torch.Tensor: ...

    @overload
    def run_np_torch_op(self, dist1: "custom_types.SampleType") -> npt.NDArray: ...

    @abstractmethod
    def run_np_torch_op(self, dist1): ...

    def write_stan_operation(self, dist1: str) -> str:
        super().write_stan_operation()
        return f"{self.STAN_OPERATOR}{dist1}"

    # pylint: enable=arguments-differ


class AddParameter(BinaryTransformedParameter):
    """Defines a parameter that is the sum of two other parameters."""

    STAN_OPERATOR: str = "+"

    def run_np_torch_op(self, dist1, dist2):
        return dist1 + dist2


class SubtractParameter(BinaryTransformedParameter):
    """Defines a parameter that is the difference of two other parameters."""

    STAN_OPERATOR: str = "-"

    def run_np_torch_op(self, dist1, dist2):
        return dist1 - dist2


class MultiplyParameter(BinaryTransformedParameter):
    """Defines a parameter that is the product of two other parameters."""

    STAN_OPERATOR: str = ".*"

    def run_np_torch_op(self, dist1, dist2):
        return dist1 * dist2


class DivideParameter(BinaryTransformedParameter):
    """Defines a parameter that is the quotient of two other parameters."""

    STAN_OPERATOR: str = "./"

    def run_np_torch_op(self, dist1, dist2):
        return dist1 / dist2


class PowerParameter(BinaryTransformedParameter):
    """Defines a parameter raised to the power of another parameter."""

    STAN_OPERATOR: str = ".^"

    def run_np_torch_op(self, dist1, dist2):
        return dist1**dist2


class NegateParameter(UnaryTransformedParameter):
    """Defines a parameter that is the negative of another parameter."""

    STAN_OPERATOR: str = "-"

    def run_np_torch_op(self, dist1):
        return -dist1


class AbsParameter(UnaryTransformedParameter):
    """Defines a parameter that is the absolute value of another."""

    LOWER_BOUND: "custom_types.Float" = 0.0

    def run_np_torch_op(self, dist1):
        return utils.choose_module(dist1).abs(dist1)

    def write_stan_operation(self, dist1: str) -> str:
        return f"abs({dist1})"


class LogParameter(UnaryTransformedParameter):
    """Defines a parameter that is the natural logarithm of another."""

    # The distribution must be positive
    POSITIVE_PARAMS = {"dist1"}

    def run_np_torch_op(self, dist1):
        return utils.choose_module(dist1).log(dist1)

    def write_stan_operation(self, dist1: str) -> str:
        return f"log({dist1})"


class ExpParameter(UnaryTransformedParameter):
    """Defines a parameter that is the exponential of another."""

    LOWER_BOUND: "custom_types.Float" = 0.0

    def run_np_torch_op(self, dist1):

        return utils.choose_module(dist1).exp(dist1)

    def write_stan_operation(self, dist1: str) -> str:
        return f"exp({dist1})"


class NormalizeParameter(UnaryTransformedParameter):
    """Defines a parameter that is normalized to sum to 1."""

    LOWER_BOUND: "custom_types.Float" = 0.0
    UPPER_BOUND: "custom_types.Float" = 1.0

    def run_np_torch_op(self, dist1):
        if isinstance(dist1, torch.Tensor):
            return dist1 / dist1.sum(dim=-1, keepdim=True)
        else:
            return dist1 / np.sum(dist1, keepdims=True, axis=-1)

    def write_stan_operation(self, dist1: str) -> str:
        return f"{dist1} / sum({dist1})"


class NormalizeLogParameter(UnaryTransformedParameter):
    """
    Defines a parameter that is normalized such that exp(x) sums to 1. By extension,
    this assumes that the input is log-transformed.
    """

    UPPER_BOUND: "custom_types.Float" = 0.0

    def run_np_torch_op(self, dist1):
        if isinstance(dist1, torch.Tensor):
            return dist1 - torch.logsumexp(dist1, keepdims=True, dim=-1)
        else:
            return dist1 - sp.logsumexp(dist1, keepdims=True, axis=-1)

    def write_stan_operation(self, dist1: str) -> str:
        return f"{dist1} - log_sum_exp({dist1})"


class Reduction(UnaryTransformedParameter):
    """
    Base class for any operations that reduce dimensionality
    """

    SHAPE_CHECK = False
    TORCH_FUNC: Callable[[npt.NDArray], npt.NDArray]
    NP_FUNC: Callable[[npt.NDArray], npt.NDArray]

    def __init__(
        self,
        dist1: "custom_types.CombinableParameterType",
        keepdims: bool = False,
        **kwargs,
    ):
        """
        Initializes the reduction. This applies the reduction over the last dimension
        of the input parameter, either with or without keeping the last dimension.
        """
        # Record whether to keep the last dimension
        self.keepdims = keepdims

        # The shape is the leading dimensions of the input parameter plus a singleton
        # dimension if keepdims is True.
        if "shape" not in kwargs:
            shape = dist1.shape[:-1]
            if keepdims:
                shape += (1,)
            kwargs["shape"] = shape

        # Init as normal
        super().__init__(dist1=dist1, **kwargs)

    def run_np_torch_op(self, dist1, keepdim: bool | None = None):

        # Keepdim can only be provided if called as a static method
        if self is None:
            keepdim = bool(keepdim)
        elif keepdim is not None:
            raise ValueError(
                "The `keepdim` argument can only be provided when calling this method "
                "as a static method."
            )
        else:
            keepdim = self.keepdims

        if isinstance(dist1, torch.Tensor):
            return self.__class__.TORCH_FUNC(dist1, keepdim=keepdim, dim=-1)
        else:
            return self.__class__.NP_FUNC(dist1, keepdims=keepdim, axis=-1)


class LogSumExpParameter(Reduction):
    """
    Defines a parameter that computes the log of the sum of exponentials of another.
    This can only be applied over the last dimension and occurs with or without
    `keepdims` set to True.
    """

    TORCH_FUNC = torch.logsumexp
    NP_FUNC = sp.logsumexp

    def write_stan_operation(self, dist1: str) -> str:
        return f"log_sum_exp({dist1})"


class SumParameter(Reduction):
    """
    Defines a parameter that computes the sum over the final axis of another
    parameter.
    """

    TORCH_FUNC = torch.sum
    NP_FUNC = np.sum

    def write_stan_operation(self, dist1: str) -> str:
        return f"sum({dist1})"


class Log1pExpParameter(UnaryTransformedParameter):
    """
    Defines a parameter that takes the logarithm of one plus the natural exponentiation
    of another parameter. This uses numerically stable alternatives to writing the
    option explicitly.
    """

    def run_np_torch_op(self, dist1):
        if isinstance(dist1, torch.Tensor):
            return torch.logaddexp(torch.tensor(0.0, device=dist1.device), dist1)
        else:
            return np.logaddexp(0.0, dist1)

    def write_stan_operation(self, dist1: str) -> str:
        return f"log1p_exp({dist1})"


class SigmoidParameter(UnaryTransformedParameter):
    """Defines a parameter that is the sigmoid of another."""

    UPPER_BOUND: "custom_types.Float" = 1.0
    LOWER_BOUND: "custom_types.Float" = 0.0

    def run_np_torch_op(self, dist1):
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
            return utils.stable_sigmoid(dist1)

        # If using a different type, raise an error.
        else:
            raise TypeError(
                "Unsupported type for dist1. Expected torch.Tensor or np.ndarray."
            )

    def write_stan_operation(self, dist1: str) -> str:
        return f"inv_logit({dist1})"


class LogSigmoidParameter(UnaryTransformedParameter):
    """Defines a parameter that is the log of the sigmoid of another."""

    UPPER_BOUND: "custom_types.Float" = 0.0

    def run_np_torch_op(self, dist1):
        if isinstance(dist1, torch.Tensor):
            return F.logsigmoid(dist1)  # pylint: disable=not-callable
        elif isinstance(dist1, np.ndarray):
            return np.log(utils.stable_sigmoid(dist1))
        else:
            raise TypeError(
                "Unsupported type for dist1. Expected torch.Tensor or np.ndarray."
            )

    def write_stan_operation(self, dist1: str) -> str:
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
        t: "custom_types.CombinableParameterType",
        A: "custom_types.CombinableParameterType",
        r: "custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initializes the LogExponentialGrowth distribution.

        Args:
            t ("custom_types.SampleType"): The time parameter.

            A ("custom_types.SampleType"): The amplitude parameter.

            r ("custom_types.SampleType"): The rate parameter.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super(UnaryTransformedParameter, self).__init__(t=t, A=A, r=r, **kwargs)

    # pylint: disable=arguments-differ
    @overload
    def run_np_torch_op(
        self, t: torch.Tensor, A: torch.Tensor, r: torch.Tensor
    ) -> torch.Tensor: ...

    @overload
    def run_np_torch_op(
        self,
        t: "custom_types.SampleType",
        A: "custom_types.SampleType",
        r: "custom_types.SampleType",
    ) -> npt.NDArray: ...

    def run_np_torch_op(self, *, t, A, r):
        return A * ExpParameter.run_np_torch_op(self, r * t)

    # pylint: enable=arguments-differ

    def write_stan_operation(  # pylint: disable=arguments-differ
        self, t: str, A: str, r: str
    ) -> str:
        par_string = super().write_stan_operation(f"{r} .* {t}")
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
        A: "custom_types.CombinableParameterType",
        r: "custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initializes the BinaryExponentialGrowth distribution.

        Args:
            A ("custom_types.SampleType"): The amplitude parameter.

            r ("custom_types.SampleType"): The rate parameter.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super(UnaryTransformedParameter, self).__init__(A=A, r=r, **kwargs)

    # pylint: disable=arguments-differ
    @overload
    def run_np_torch_op(self, A: torch.Tensor, r: torch.Tensor) -> torch.Tensor: ...
    @overload
    def run_np_torch_op(
        self,
        A: "custom_types.SampleType",
        r: "custom_types.SampleType",
    ) -> npt.NDArray: ...
    def run_np_torch_op(self, *, A, r):
        return A * ExpParameter.run_np_torch_op(self, r)

    def write_stan_operation(self, A: str, r: str) -> str:
        return f"{A} .* {super().write_stan_operation(r)}"

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
    different populations as is done in SciStanPy, as proportions are always positive.
    """

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        t: "custom_types.CombinableParameterType",
        log_A: "custom_types.CombinableParameterType",
        r: "custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initializes the LogExponentialGrowth distribution.

        Args:
            t ("custom_types.SampleType"): The time parameter.

            log_A ("custom_types.SampleType"): The log of the amplitude parameter.

            r ("custom_types.SampleType"): The rate parameter.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super().__init__(t=t, log_A=log_A, r=r, **kwargs)

    # pylint: disable=arguments-differ
    @overload
    def run_np_torch_op(
        self, t: torch.Tensor, log_A: torch.Tensor, r: torch.Tensor
    ) -> torch.Tensor: ...

    @overload
    def run_np_torch_op(
        self,
        t: "custom_types.SampleType",
        log_A: "custom_types.SampleType",
        r: "custom_types.SampleType",
    ) -> npt.NDArray: ...

    def run_np_torch_op(self, *, t, log_A, r):
        return log_A + r * t

    # pylint: enable=arguments-differ

    def write_stan_operation(  # pylint: disable=arguments-differ
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
        log_A: "custom_types.CombinableParameterType",
        r: "custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initializes the BinaryLogExponentialGrowth distribution.

        Args:
            log_A ("custom_types.SampleType"): The log of the amplitude parameter.

            r ("custom_types.SampleType"): The rate parameter.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super().__init__(log_A=log_A, r=r, **kwargs)

    # pylint: disable=arguments-differ
    @overload
    def run_np_torch_op(self, log_A: torch.Tensor, r: torch.Tensor) -> torch.Tensor: ...
    @overload
    def run_np_torch_op(
        self,
        log_A: "custom_types.SampleType",
        r: "custom_types.SampleType",
    ) -> npt.NDArray: ...
    def run_np_torch_op(self, *, log_A, r):
        return log_A + r

    def write_stan_operation(self, log_A: str, r: str) -> str:
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

    LOWER_BOUND: "custom_types.Float" = 0.0
    UPPER_BOUND: None = None

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        t: "custom_types.CombinableParameterType",
        A: "custom_types.CombinableParameterType",
        r: "custom_types.CombinableParameterType",
        c: "custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initializes the LogSigmoidGrowth distribution.

        Args:
            t ("custom_types.SampleType"): The time parameter.

            A ("custom_types.SampleType"): The amplitude parameter. Specifically,
                this is the maximum value of the sigmoid function.

            r ("custom_types.SampleType"): The rate parameter.

            c ("custom_types.SampleType"): The offset parameter. This is the
                inflection point of the sigmoid function.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super(UnaryTransformedParameter, self).__init__(t=t, A=A, r=r, c=c, **kwargs)

    # pylint: disable=arguments-differ
    @overload
    def run_np_torch_op(
        self, t: torch.Tensor, A: torch.Tensor, r: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor: ...

    @overload
    def run_np_torch_op(
        self,
        t: "custom_types.SampleType",
        A: "custom_types.SampleType",
        r: "custom_types.SampleType",
        c: "custom_types.SampleType",
    ) -> npt.NDArray: ...

    def run_np_torch_op(self, *, t, A, r, c):
        return A * SigmoidParameter.run_np_torch_op(self, r * (t - c))

    # pylint: enable=arguments-differ

    def write_stan_operation(  # pylint: disable=arguments-differ
        self, t: str, A: str, r: str, c: str
    ) -> str:
        par_string = super().write_stan_operation(f"{r} .* ({t} - {c})")
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
        t: "custom_types.CombinableParameterType",
        log_A: "custom_types.CombinableParameterType",
        r: "custom_types.CombinableParameterType",
        c: "custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initializes the LogSigmoidGrowth distribution.

        Args:
            exp_t ("custom_types.SampleType"): The exponentiated time parameter.

            log_A ("custom_types.SampleType"): The log of the amplitude parameter.

            r ("custom_types.SampleType"): The rate parameter.

            c ("custom_types.SampleType"): The offset parameter.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super(UnaryTransformedParameter, self).__init__(
            t=t, log_A=log_A, r=r, c=c, **kwargs
        )

    # pylint: disable=arguments-differ
    @overload
    def run_np_torch_op(
        self, t: torch.Tensor, log_A: torch.Tensor, r: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor: ...

    @overload
    def run_np_torch_op(
        self,
        t: "custom_types.SampleType",
        log_A: "custom_types.SampleType",
        r: "custom_types.SampleType",
        c: "custom_types.SampleType",
    ) -> npt.NDArray: ...

    def run_np_torch_op(self, *, t, log_A, r, c):
        return log_A + LogSigmoidParameter.run_np_torch_op(self, r * (t - c))

    # pylint: enable=arguments-differ

    def write_stan_operation(  # pylint: disable=arguments-differ
        self, t: str, log_A: str, r: str, c: str
    ) -> str:
        par_string = super().write_stan_operation(f"{r} .* ({t} - {c})")
        return f"{log_A} + {par_string}"


class SigmoidGrowthInitParametrization(TransformedParameter):
    r"""
    An alternative parametrization of the sigmoid growth function that parametrizes
    in terms of starting abundances rather than the maximum abundances.
    """

    LOWER_BOUND: "custom_types.Float" = 0.0
    UPPER_BOUND: None = None

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        t: "custom_types.CombinableParameterType",
        x0: "custom_types.CombinableParameterType",  # pylint: disable=invalid-name
        r: "custom_types.CombinableParameterType",
        c: "custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initializes the SigmoidGrowthInitParametrization distribution.

        Args:
            t (custom_types.CombinableParameterType): The time parameter.
            x0 (custom_types.CombinableParameterType): Initial abundances.
            r (custom_types.CombinableParameterType): Growth rate.
            c (custom_types.CombinableParameterType): Offset parameter.
            shape (tuple[int, ...], optional): The shape of the distribution. Defaults
            to ().
        """
        # Initialize using the base transformed parameter class
        super().__init__(t=t, x0=x0, r=r, c=c, **kwargs)

    # pylint: disable=arguments-renamed, arguments-differ
    @overload
    def run_np_torch_op(
        self, t: torch.Tensor, x0: torch.Tensor, r: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor: ...

    @overload
    def run_np_torch_op(
        self,
        t: "custom_types.SampleType",
        x0: "custom_types.SampleType",  # pylint: disable=invalid-name
        r: "custom_types.SampleType",
        c: "custom_types.SampleType",
    ) -> npt.NDArray: ...

    def run_np_torch_op(self, t, x0, r, c):
        """We use a log-add-exp trick to calculate in a numerically stable way."""
        # Get the module
        mod = utils.choose_module(x0)

        # Get the fold-change. We use the log-add-exp function to calculate this
        # in a more numerically stable way
        zero = 0.0 if mod is np else torch.tensor(0.0, device=x0.device)
        foldchange = mod.exp(
            mod.logaddexp(zero, r * c) - mod.logaddexp(zero, r * (c - t))
        )

        # Calculate the abundance
        return x0 * foldchange

    def write_stan_operation(
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
        t: "custom_types.CombinableParameterType",
        log_x0: "custom_types.CombinableParameterType",  # pylint: disable=invalid-name
        r: "custom_types.CombinableParameterType",
        c: "custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initializes the LogSigmoidGrowthInitParametrization distribution.

        Args:
            t (custom_types.CombinableParameterType): The time parameter.
            log_x0 (custom_types.CombinableParameterType): Logarithm of initial
                abundances.
            r (custom_types.CombinableParameterType): Growth rate.
            c (custom_types.CombinableParameterType): Offset parameter.
            shape (tuple[int, ...], optional): The shape of the distribution. Defaults
            to ().
        """
        # Initialize using the base transformed parameter class
        super().__init__(t=t, log_x0=log_x0, r=r, c=c, **kwargs)

    # pylint: disable=arguments-renamed, arguments-differ
    @overload
    def run_np_torch_op(
        self,
        t: torch.Tensor,
        log_x0: torch.Tensor,  # pylint: disable=invalid-name
        r: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor: ...

    @overload
    def run_np_torch_op(
        self,
        t: "custom_types.SampleType",
        log_x0: "custom_types.SampleType",  # pylint: disable=invalid-name
        r: "custom_types.SampleType",
        c: "custom_types.SampleType",
    ) -> npt.NDArray: ...

    def run_np_torch_op(self, t, log_x0, r, c):
        """We use a log-add-exp trick to calculate in a numerically stable way."""
        # Get the module
        mod = utils.choose_module(log_x0)

        # Define zero
        zero = 0.0 if mod is np else torch.tensor(0.0, device=log_x0.device)

        # Calculate
        return log_x0 + mod.logaddexp(zero, r * c) - mod.logaddexp(zero, r * (c - t))

    def write_stan_operation(
        self, t: str, log_x0: str, r: str, c: str  # pylint: disable=invalid-name
    ) -> str:
        """Calculate using Stan's log1p_exp function."""
        return f"{log_x0} + log1p_exp({r} .* {c}) - log1p_exp({r} .* ({c} - {t}))"


class ConvolveSequence(TransformedParameter):
    """
    Using a matrix of provided weights, performs a convolution operation on an
    ordinally-encoded array of sequences. For broadcasting purposes, the last two
    dimensions of the weights matrix and the last dimension of the seqence array
    are ignored.
    """

    # Do not check the shape
    SHAPE_CHECK = False

    # We must reset the loops to define this parameter
    FORCE_LOOP_RESET = True

    # Parents must be named
    FORCE_PARENT_NAME = True

    def __init__(
        self,
        *,
        weights: "custom_types.CombinableParameterType",
        ordinals: "custom_types.CombinableParameterType",
        **kwargs,
    ):
        # Weights must be at least 2D.
        if weights.ndim < 2:
            raise ValueError("Weights must be at least a 2D parameter.")

        # Sequence must be at least 1D
        if ordinals.ndim < 1:
            raise ValueError("Sequence must be at least a 1D parameter.")

        # Note features of the weights. This is the last two dimensions.
        self.kernel_size, self.alphabet_size = weights.shape[-2:]

        # The first N - 2 dimensions of the weights must align with the first
        # N - 1 dimensions of the ordinals
        try:
            batch_dims = np.broadcast_shapes(weights.shape[:-2], ordinals.shape[:-1])
        except ValueError as err:
            raise ValueError(
                "Incompatible shapes between weights and ordinals. The shapes must "
                "be broadcastable in the batch dimensions (all but last two for "
                "the weights and all but the last for the ordinals). Got "
                f"weights: {weights.shape}, ordinals: {ordinals.shape}"
            ) from err

        # The final dimension has the size of the sequence length adjusted by the
        # convolution
        shape = batch_dims + (ordinals.shape[-1] - self.kernel_size + 1,)

        # Init using inherited method.
        super().__init__(weights=weights, ordinals=ordinals, shape=shape, **kwargs)

    def run_np_torch_op(self, weights, ordinals):  # pylint: disable=arguments-differ
        """Performs the convolution"""

        @overload
        def process_single(
            single_weights: npt.NDArray[np.float64],
            single_ordinals: [npt.NDArray[np.int64]],
        ) -> npt.NDArray[np.float64]: ...

        @overload
        def process_single(
            single_weights: torch.Tensor, single_ordinals: torch.Tensor
        ) -> torch.Tensor: ...

        def process_single(single_weights, single_ordinals):
            """
            Processes a single sample. For torch, this is used directly. For numpy,
            we loop over the sampling dimension and apply repeatedly.
            """
            # Set output array
            output_arr = module.full(self.shape, np.nan)

            # Create a set of indices for the filters. If torch, send arrays to
            # appropriate device
            filter_indices = module.arange(self.kernel_size)
            if module is torch:
                filter_indices = filter_indices.to(single_weights.device)
                output_arr = output_arr.to(single_weights.device)

            # Loop over the different weights
            for weights_inds in np.ndindex(single_weights.shape[:-2]):

                # Prepend `None` to the weight indices if needed
                weights_inds = (None,) * weights_n_prepends + weights_inds

                # Determine the ordinal and output indices. If weights or ordinals
                # are a singleton, slice all for the ordinal indices.
                ordinal_inds = []
                output_inds = []
                for dim, (weight_dim_size, ord_dim_size) in enumerate(
                    zip(padded_weights_shape, padded_ordinals_shape)
                ):

                    # We can never have both weight and ord dim sizes be `None`
                    assert not (weight_dim_size is None and ord_dim_size is None)

                    # If the ordinal dimension is `None`, then the output dimension is whatever
                    # the weight dimension is. We do not record an ordinal index.
                    weight_ind = weights_inds[dim]
                    if ord_dim_size is None:
                        output_inds.append(weight_ind)
                        continue

                    # If the weight dimension is a singleton we slice all for the ordinal and
                    # the output
                    if weight_dim_size == 1 or weight_dim_size is None:
                        ordinal_inds.append(slice(None))
                        output_inds.append(slice(None))

                    # If the ordinal dimension is a singleton, add "0" to the indices for the
                    # ordinals and the weights ind for the output
                    elif ord_dim_size == 1:
                        ordinal_inds.append(0)
                        output_inds.append(weight_ind)

                    # Otherwise, identical index to the weights for both
                    else:
                        ordinal_inds.append(weight_ind)
                        output_inds.append(weight_ind)

                # Convert indices to tuples
                ordinal_inds = tuple(ordinal_inds)
                output_inds = tuple(output_inds)
                assert len(output_inds) == len(self.shape) - 1

                # Get the matrix and set of sequences to which it will be applied
                weights_matrix = single_weights[weights_inds]
                ordinal_matrix = single_ordinals[ordinal_inds]
                assert weights_matrix.ndim == 2

                # Run convolution for this batch by sliding over the sequence length
                for convind, upper_slice in enumerate(
                    range(self.kernel_size, ordinal_matrix.shape[-1] + 1)
                ):

                    # Get the lower bound
                    lower = upper_slice - self.kernel_size

                    # Slice the sequence and pull the appropriate weights. Sum the weights.
                    output_arr[output_inds + (convind,)] = weights_matrix[
                        filter_indices, ordinal_matrix[..., lower:upper_slice]
                    ].sum(**{"dim" if module is torch else "axis": -1})

            # No Nan's in output
            assert not module.any(module.isnan(output_arr))

            return output_arr

        # Decide on the module for the operation
        module = utils.choose_module(weights)

        # Determine the number of dimensions to prepend to each array
        weights_n_prepends = len(self.shape[:-1]) - len(self.weights.shape[:-2])
        ordinal_n_prepends = len(self.shape[:-1]) - len(self.ordinals.shape[:-1])

        # Get the padded shapes. This is just aligning the shapes for broadcasting.
        padded_weights_shape = (None,) * weights_n_prepends + self.weights.shape[:-2]
        padded_ordinals_shape = (None,) * ordinal_n_prepends + self.ordinals.shape[:-1]
        assert len(padded_weights_shape) == len(padded_ordinals_shape)

        # If torch, apply the inner function directly
        if module is torch:
            return process_single(single_weights=weights, single_ordinals=ordinals)

        # A quirk of having a matrix input with the weights is that a dimension
        # will be added to the ordinals that should not be there (see the `draw`
        # function of the abstract model component.) If we add more functions like
        # this one, longer term, we will want to add a class variable or similar
        # for handling things like this. As it's the only one right now, we add
        # the below patch:
        assert module is np
        assert ordinals.shape[1] == 1
        ordinals = ordinals[:, 0]

        # If numpy, loop over the leading dimension
        assert weights.shape[1:] == self.weights.shape
        assert ordinals.shape[1:] == self.ordinals.shape
        assert weights.shape[0] == ordinals.shape[0]
        return np.stack(
            [
                process_single(
                    single_weights=weights_sample, single_ordinals=ordinals_sample
                )
                for weights_sample, ordinals_sample in zip(weights, ordinals)
            ]
        )

    def get_right_side(
        self,
        index_opts: tuple[str, ...] | None,
        start_dims: dict[str, "custom_types.Integer"] | None = None,
        end_dims: dict[str, "custom_types.Integer"] | None = None,
    ) -> str:

        # Different default for end dims here
        end_dims = end_dims or {"weights": 1}

        # Run the AbstractModelParameter version of the method to get each model
        # component formatted appropriately. Note that we ignore the last dimension
        # of the weights. This is because we need both dimensions in the Stan code.
        return super().get_right_side(index_opts, end_dims=end_dims)

    def write_stan_operation(  # pylint: disable=arguments-differ
        self, weights: str, ordinals: str
    ) -> str:
        """
        We need weights with no indexing. Indexing for sequence should be the standard
        (as we automatically assume that the last dimension is vectorized)
        """
        # This runs a custom function
        return f"convolve_sequence({weights}, {ordinals})"

    def get_supporting_functions(self) -> list[str]:
        return super().get_supporting_functions() + ["#include pssm.stanfunctions"]


class IndexParameter(TransformedParameter):
    """
    Used for indexing one parameter to create another with a different shape. Currently
    supports slicing, indexing with scalars, and indexing with single-dimension
    arrays. Multi-dimensional indexing is not supported. Boolean indexing is not
    supported.

    IMPORTANT: Indexing follows the same rules as NUMPY, not Stan. This means that
    the following:

    ```python
    test = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    ind1 = [0, 2, 1]
    ind2 = [1, 0, 2]
    ```

    will yield

    ```python
    test[ind1, ind2]

    >>   array([2, 7, 6])
    ```

    which is different from Stan's indexing rules. Using Stan's rules (and adjusting
    for Stan's use of 1-indexing rather than 0-indexing), this would yield
    ```stan
    test[ind1, ind2]
    >>   array([
        [2, 1, 3],
        [8, 7, 9],
        [5, 4, 6]
    ])
    ```
    """

    # Do not check shapes
    SHAPE_CHECK = False

    # Parents of this parameter must be named
    FORCE_PARENT_NAME = True

    def __init__(
        self,
        dist: "custom_types.CombinableParameterType",
        *indices: "custom_types.IndexType",
    ):
        """
        Initializes the IndexParameter transformation.

        Args:
            dist ("custom_types.SampleType"): The parameter to index.
            *indices ("custom_types.IndexType"): The indices to use for indexing the
                parameter. These can be a mix of `slice`, `np.ndarray`, or `int`.
                The indices must be compatible with the shape of the distribution.
        """
        # We need the shape of what we're indexing to prep for parent init
        self._dist_shape = dist.shape

        # We need the input indices for torch and numpy operations
        self._python_indices = indices

        # Process and unify the different index types
        shape, self._stan_indices, parents = self._process_indices(indices)

        # Init using parent method. Provide the shape with `None` values removed --
        # these are the dimensions that are removed by indexing
        super().__init__(dist=dist, shape=shape, **parents)

    @overload
    def neg_to_pos(
        self, neg_ind: "custom_types.Integer", dim: "custom_types.Integer"
    ) -> "custom_types.Integer": ...

    @overload
    def neg_to_pos(
        self, neg_ind: npt.NDArray[np.int64], dim: "custom_types.Integer"
    ) -> npt.NDArray[np.int64]: ...

    def neg_to_pos(self, neg_ind, dim):
        """Converts negative indices to positive indices in Python format"""
        # If a numpy array, we update negative positions only
        if isinstance(neg_ind, np.ndarray):
            out = neg_ind.copy()
            out[out < 0] += self._dist_shape[dim]

            # There should be no negatives
            if np.any(out < 0):
                raise ValueError(
                    f"Negative indices {neg_ind} cannot be converted to positive "
                    f"indices for dimension {dim} with shape {self._dist_shape[dim]}."
                )

            # The max should be less than the dimension size
            if np.any(out >= self._dist_shape[dim]):
                raise ValueError(
                    f"Indices {neg_ind} exceed the size of dimension {dim} "
                    f"with shape {self._dist_shape[dim]}."
                )

            return out

        # If a single integer, we convert it directly.
        elif isinstance(neg_ind, int):
            out = neg_ind + self._dist_shape[dim] if neg_ind < 0 else neg_ind

            # Check that the index is within bounds
            if out < 0:
                raise ValueError(
                    f"Negative index {neg_ind} cannot be converted to positive "
                    f"index for dimension {dim} with shape {self._dist_shape[dim]}."
                )
            if out >= self._dist_shape[dim]:
                raise ValueError(
                    f"Index {neg_ind} exceeds the size of dimension {dim} "
                    f"with shape {self._dist_shape[dim]}."
                )

            return out

        # Error if the type is not supported
        raise TypeError(
            f"Unsupported index type {type(neg_ind)}. Expected int or numpy array."
        )

    def _process_indices(
        self,
        indices: tuple["custom_types.IndexType", ...],
    ) -> tuple[
        tuple[int, ...],
        tuple["custom_types.IndexType", ...],
        dict[str, constants.Constant],
    ]:
        """
        Processes the indices provided to the IndexParameter transformation to unify
        their format and determine the output shape.
        """

        def process_ellipsis() -> int:
            """Helper function to process Ellipses"""
            # We can only have one ellipsis
            if sum(1 for ind in indices if ind is Ellipsis) > 1:
                raise ValueError("Only one ellipsis is allowed in indexing.")

            # Add slices to the processed dimensions
            n_real_dims = sum(
                1 for ind in indices if ind is not Ellipsis and ind is not None
            )
            n_to_add = len(self._dist_shape) - n_real_dims
            processed_inds.extend([slice(None) for _ in range(n_to_add)])

            # The shape is extended by the number added
            shape.extend(self._dist_shape[shape_ind : shape_ind + n_to_add])

            # Return the number of added dimensions
            return n_to_add

        def process_slice() -> None:
            """Helper function to process slices."""
            # Step cannot be set
            if ind.step is not None and ind.step != 1:
                raise ValueError(
                    f"Step size {ind.step} is not supported in IndexParameter transformation."
                )

            # Get the size of the output shape (stop - start after converting negatives
            # to positives)
            start = 0 if ind.start is None else self.neg_to_pos(ind.start, shape_ind)
            stop = (
                self._dist_shape[shape_ind]
                if ind.stop is None
                else self.neg_to_pos(ind.stop, shape_ind)
            )

            # Update outputs. Note that processed outputs are a new slice and that
            # we do not add 1 to stop because Stan slices are inclusive while Python
            # are exclusive
            shape.append(stop - start)
            processed_inds.append(slice(start + 1, stop))

        def process_array() -> int:
            """Helper function to process numpy arrays and constants."""

            # Must be a 1D array
            if ind.ndim > 1:
                raise IndexError(
                    "Cannot index with numpy array with more than 1 dimension"
                )
            elif ind.ndim == 0:
                raise AssertionError("Should not get here")

            # Ensure the array contains integers
            if ind.dtype != np.int64:
                raise TypeError(
                    f"Indexing with non-integer arrays is not supported. Got dtype "
                    f"{ind.dtype}."
                )

            # Must be the same as previous 1-d arrays
            arrlen = len(ind)
            if int_arr_len > 0 and int_arr_len != arrlen:
                raise ValueError(
                    f"All 1-dimensional integer arrays must have the same length. "
                    f"Got lengths {int_arr_len} and {arrlen}."
                )

            # Build a constant for the index. This involves adjusting the indices
            # to be Stan-compatible (1-indexed, no negative indices).
            constant_arr = constants.Constant(
                self.neg_to_pos(ind, shape_ind) + 1, togglable=False
            )

            # Record
            shape.append(arrlen)
            parents[f"idx_{len(parents)}"] = constant_arr
            processed_inds.append(constant_arr)

            return arrlen

        # Set up for recording
        shape = []  # This parameter's shape
        processed_inds = []  # Indices processed for use in Stan
        shape_ind = 0  # Current dimension in the indexed parameter
        parents: dict[str, constants.Constant] = {}  # Constants for arrays
        int_arr_len = 0  # Length of integer arrays

        # Process indices
        for ind in indices:

            # Process ellipses
            if ind is Ellipsis:
                shape_ind += process_ellipsis()

            # Process slices
            elif isinstance(ind, slice):
                process_slice()
                shape_ind += 1

            # Numpy arrays are also processed by their own function
            elif isinstance(ind, np.ndarray):
                int_arr_len = max(int_arr_len, process_array())
                shape_ind += 1

            # Integers must be made positive
            elif isinstance(ind, int):
                processed_inds.append(self.neg_to_pos(ind, shape_ind) + 1)
                shape_ind += 1

            # `None` values add a new dimension to the output.
            elif ind is None:
                shape.append(1)

            # Nothing else is legal
            else:
                raise TypeError(
                    "Indexing supported by slicing, numpy arrays, and integers only."
                    f"Got {type(ind)}"
                )

        # Remove None values from the shape
        return tuple(shape), tuple(processed_inds), parents

    # Note that parents are ignored here as their indices have been adjusted to
    # reflect Stan's 1-indexing and no negative indices. We use the Python indices
    # stored earlier as a result. The parents kwargs is included for compatibility
    def run_np_torch_op(  # pylint: disable=arguments-differ, unused-argument
        self, dist, **parents
    ):
        # If torch, numpy arrays must go to torch. If numpy, we will have an
        # additional leading index for sampling
        module = utils.choose_module(dist)
        if module is torch:
            inds = tuple(
                (
                    torch.from_numpy(ind).to(dist.device)
                    if isinstance(ind, np.ndarray)
                    else ind
                )
                for ind in self._python_indices
            )
        else:
            inds = (slice(None), *self._python_indices)

        # Index and check shape
        indexed = dist[inds]
        if module is torch:
            assert indexed.shape == self.shape
        else:
            assert indexed.shape[1:] == self.shape

        return indexed

    def get_right_side(
        self,
        index_opts: tuple[str, ...] | None,
        start_dims: dict[str, "custom_types.Integer"] | None = None,
        end_dims: dict[str, "custom_types.Integer"] | None = None,
    ) -> str:
        """
        Gets the name of the variable that is being indexed, then passes it to
        the `write_stan_operation` method to get the full Stan code for the transformation
        """
        return self.write_stan_operation(dist=self.dist.model_varname)

    def write_stan_operation(  # pylint: disable=arguments-differ
        self, dist: str
    ) -> str:

        # Compile all indices. Every time we encounter an array index, we start
        # a new indexing operation. This allows us to mimic numpy behavior in Stan.
        components = []
        current_component = []
        index_pos = 0
        array_counter = 0
        for ind in self._stan_indices:

            # Handle slices
            if isinstance(ind, slice):
                start = "" if ind.start is None else str(ind.start)
                end = "" if ind.stop is None else str(ind.stop)
                index_pos += 1  # We keep a dimension with this operation
                current_component.append(f"{start}:{end}")

            # Handle integers
            elif isinstance(ind, int):
                current_component.append(str(ind))

            # If an array, we need to use the constant that we defined
            elif isinstance(ind, constants.Constant):

                # Must be a 1D array
                assert isinstance(ind.value, np.ndarray)
                assert ind.value.ndim == 1

                # If we have already encountered an array, start a new component,
                # padding out the current component with colons.
                if array_counter > 0:
                    components.append(current_component)
                    current_component = [":"] * (index_pos + 1)

                # Record the array as a component
                current_component.append(
                    self._parents[f"idx_{array_counter}"].get_indexed_varname(None)
                )

                # Update counters
                index_pos += 1  # We keep a dimension with this operation
                array_counter += 1  # Note finding another array

            # Error with anything else
            else:
                raise ValueError(f"Unsupported index type: {type(ind)}")

        # Record the last component
        components.append(current_component)

        # Join all components
        return dist + "[" + "][".join(",".join(c) for c in components) + "]"

    # The definition depth is always 0 for this transformation
    @property
    def assign_depth(self) -> int:  # pylint: disable=C0116
        return 0
