"""Holds parameter transformations for DMS Stan models."""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional, overload, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import scipy.special as sp
import torch
import torch.nn.functional as F

from dms_stan import utils
from dms_stan.model.components import abstract_model_component, constants

if TYPE_CHECKING:
    from dms_stan import custom_types

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

    def __getitem__(self, key: "custom_types.IndexType"):
        # If key is not a tuple, make it one
        if not isinstance(key, tuple):
            key = (key,)

        return IndexParameter(self, *key)


class Transformation(abstract_model_component.AbstractModelComponent):
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
    def write_stan_operation(self, **to_format: str) -> str:
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
            name: (value if self._parents[name].is_named else f"( {value} )")
            for name, value in components.items()
        }

        # Format the right-hand side of the operation. Exactly how formatting is
        # done depends on the child class.
        return self.write_stan_operation(**components)

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

    LOWER_BOUND: float = 0.0

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

    LOWER_BOUND: float = 0.0

    def run_np_torch_op(self, dist1):

        return utils.choose_module(dist1).exp(dist1)

    def write_stan_operation(self, dist1: str) -> str:
        return f"exp({dist1})"


class NormalizeParameter(UnaryTransformedParameter):
    """Defines a parameter that is normalized to sum to 1."""

    LOWER_BOUND: float = 0.0
    UPPER_BOUND: float = 1.0

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

    UPPER_BOUND: float = 0.0

    def run_np_torch_op(self, dist1):
        if isinstance(dist1, torch.Tensor):
            return dist1 - torch.logsumexp(dist1, keepdims=True, dim=-1)
        else:
            return dist1 - sp.logsumexp(dist1, keepdims=True, axis=-1)

    def write_stan_operation(self, dist1: str) -> str:
        return f"{dist1} - log_sum_exp({dist1})"


class LogSumExpParameter(UnaryTransformedParameter):
    """
    Defines a parameter that computes the log of the sum of exponentials of another.
    This can only be applied over the last dimension and occurs with or without
    `keepdims` set to True.
    """

    def __init__(
        self,
        dist1: "custom_types.CombinableParameterType",
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
            return torch.logsumexp(dist1, keepdim=keepdim, dim=-1)
        else:
            return sp.logsumexp(dist1, keepdims=keepdim, axis=-1)

    def write_stan_operation(self, dist1: str) -> str:
        return f"log_sum_exp({dist1})"


class SigmoidParameter(UnaryTransformedParameter):
    """Defines a parameter that is the sigmoid of another."""

    UPPER_BOUND: float = 1.0
    LOWER_BOUND: float = 0.0

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

    UPPER_BOUND: float = 0.0

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
    different populations as is done in DMS Stan, as proportions are always positive.
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

    LOWER_BOUND: float = 0.0
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

    LOWER_BOUND: float = 0.0
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
        # Process and unify the different index types
        shape, self.indices, parents = self._process_indices(dist, indices)

        # Init using parent method. Provide the shape with `None` values removed --
        # these are the dimensions that are removed by indexing
        super().__init__(dist=dist, shape=shape, **parents)

    @overload
    def neg_to_pos(self, neg_ind: int, dim: int) -> int: ...

    @overload
    def neg_to_pos(
        self, neg_ind: npt.NDArray[np.int64], dim: int
    ) -> npt.NDArray[np.int64]: ...

    def neg_to_pos(self, neg_ind, dim):
        """Converts negative indices to positive indices in Python format"""
        # If a numpy array, we update negative positions only
        if isinstance(neg_ind, np.ndarray):
            out = neg_ind.copy()
            out[out < 0] += self.dist.shape[dim]

            # There should be no negatives
            if np.any(out < 0):
                raise ValueError(
                    f"Negative indices {neg_ind} cannot be converted to positive "
                    f"indices for dimension {dim} with shape {self.dist.shape[dim]}."
                )

            # The max should be less than the dimension size
            if np.any(out >= self.dist.shape[dim]):
                raise ValueError(
                    f"Indices {neg_ind} exceed the size of dimension {dim} "
                    f"with shape {self.dist.shape[dim]}."
                )

            return out

        # If a single integer, we convert it directly.
        elif isinstance(neg_ind, int):
            out = neg_ind + self.dist.shape[dim] if neg_ind < 0 else neg_ind

            # Check that the index is within bounds
            if out < 0:
                raise ValueError(
                    f"Negative index {neg_ind} cannot be converted to positive "
                    f"index for dimension {dim} with shape {self.dist.shape[dim]}."
                )
            if out >= self.dist.shape[dim]:
                raise ValueError(
                    f"Index {neg_ind} exceeds the size of dimension {dim} "
                    f"with shape {self.dist.shape[dim]}."
                )

            return out

        # Error if the type is not supported
        raise TypeError(
            f"Unsupported index type {type(neg_ind)}. Expected int or numpy array."
        )

    def _process_indices(
        self,
        dist: "custom_types.CombinableParameterType",
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

        def process_slice() -> None:
            """Helper function to process slices."""
            # Step cannot be set
            if ind.step is not None and ind.step != 1:
                raise ValueError(
                    f"Step size {ind.step} is not supported in IndexParameter transformation."
                )

            # Get the size of the output shape (stop - start after converting negatives
            # to positives)
            start = 0 if ind.start is None else self.neg_to_pos(ind.start, indpos)
            stop = (
                dist.shape[indpos]
                if ind.stop is None
                else self.neg_to_pos(ind.stop, indpos)
            )
            shape[indpos] = stop - start

        def process_array() -> int:
            """Helper function to process numpy arrays and constants."""

            # Ensure the array contains integers
            if ind.dtype is not np.int64:
                raise TypeError(
                    f"Indexing with non-integer arrays is not supported. Got dtype "
                    f"{ind.dtype}."
                )

            # Different approaches for 1 vs 0 d
            if ind.ndim == 1:

                # Must be the same as previous 1-d arrays
                arrlen = len(ind)
                if int_arr_len > 0 and int_arr_len != arrlen:
                    raise ValueError(
                        f"All 1-dimensional integer arrays must have the same length. "
                        f"Got lengths {int_arr_len} and {arrlen}."
                    )

                # Store the index length
                shape[indpos] = arrlen

                # Build a constant for the index. This involves adjusting the indices
                # to be Stan-compatible (1-indexed, no negative indices).
                parents[f"idx_{indpos}"] = constants.Constant(
                    self.neg_to_pos(ind, indpos) + 1
                )

                return arrlen

            # If not 1-d or 0-d, raise an error
            elif ind.ndim > 1:
                raise ValueError(
                    f"Indexing with multi-dimensional arrays is not supported. "
                    f"Got {ind.ndim} dimensions."
                )

            return 0

        # We cannot have more indices than dimensions in the distribution.
        if (n_inds := len(indices)) > dist.ndim:
            raise ValueError(
                f"Too many indices provided. Expected at most {dist.ndim}, got {n_inds}."
            )

        # Each index must be a Constant, numpy array, slice, or int and must be
        # compatible with the distribution's shape.
        shape = [None] * n_inds
        parents: dict[str, constants.Constant] = {}
        int_arr_len = 0
        indices = shape.copy()
        for indpos, ind in enumerate(indices):

            # Process slices
            if isinstance(ind, slice):
                process_slice()

            # Process numpy arrays
            elif isinstance(ind, np.ndarray):
                int_arr_len = max(int_arr_len, process_array())

            # If not an integer at this point, error
            elif not isinstance(ind, int):
                raise TypeError(
                    f"Indexing with {type(ind)} is not supported. Expected a Constant, "
                    "numpy array, slice, or int."
                )

            # Record index
            indices[indpos] = ind

        # Index list should be full now and no longer changing
        assert None not in indices, "Not all indices were processed."

        # Remove None values from the shape
        return tuple(s for s in shape if s is not None), tuple(indices), parents

    # Note that parents are ignored here as their indices have been adjusted to
    # reflect Stan's 1-indexing and no negative indices.
    def run_np_torch_op(  # pylint: disable=arguments-differ, unused-argument
        self, dist, **parents
    ):
        # We just index the input and return
        return dist[self.indices]

    def get_right_side(self, index_opts: tuple[str, ...] | None) -> str:
        """
        Gets the name of the variable that is being indexed, then passes it to
        the `write_stan_operation` method to get the full Stan code for the transformation
        """
        return self.write_stan_operation(dist=self.dist.model_varname)

    def write_stan_operation(  # pylint: disable=arguments-differ
        self, dist: str
    ) -> str:
        def python_to_stan_inds() -> str:
            """
            There are a few major differences between Python and Stan indexing:

            1. Python uses 0-indexing, while Stan uses 1-indexing.
            2. Python allows for negative indexing, while Stan does not.
            3. Stan upper bounds are inclusive, while Python's are exclusive.

            This function converts Python-style indices to Stan-style indices. Note
            that the Constants that were created as part of initialization already
            obey this format. Pre-processing the indices in these is necessary to
            avoid writing the entire array as neat text into the Stan code (i.e.,
            it allows us to define it as 'data')
            """
            # If a slice, convert to positive as needed, then convert to Stan format.
            # This is done by converting both to positive indices as needed, then
            # adding 1 to the start and 0 to the stop index. We do not add 1 to
            # the stop because Stan's stop index is inclusive while Python's is
            # exclusive (e.g., :2 in Python takes indices 0 and 1 while in Stan
            # takes indices 1 and 2 -- the first two in both cases. 1: takes everything
            # after the first index in Python, while in Stan it takes everything.
            if isinstance(ind, slice):
                start = (
                    ""
                    if ind.start is None
                    else str(self.neg_to_pos(ind.start, dim) + 1)
                )
                end = "" if ind.stop is None else str(self.neg_to_pos(ind.stop, dim))
                return f"{start}:{end}"

            # If an integer, convert to positive and add 1 for Stan's 1-indexing.
            elif isinstance(ind, int):
                return str(self.neg_to_pos(ind, dim) + 1)

            # If an array, we need to use the constant that we defined if 1D, otherwise
            # we extract and update the single value
            elif isinstance(ind, np.ndarray):
                if ind.ndim == 1:
                    return self._parents[f"idx_{dim}"].get_indexed_varname(None)
                elif ind.ndim == 0:
                    return str(self.neg_to_pos(ind.item(), dim) + 1)
                raise AssertionError("This should have been caught before")

        # Compile all indices. Every time we encounter an array index, we start
        # a new indexing operation. This allows us to mimic numpy behavior in Stan.
        components = []
        current_component = []
        arr_encountered = False
        for dim, ind in enumerate(self.indices):

            # If an array and we have already encountered an array, we need to
            # start a new component. If we have not already encountered an array,
            # note now that we have.
            if isinstance(ind, np.ndarray):
                if arr_encountered:
                    components.append(current_component)
                    current_component = []
                    arr_encountered = False
                else:
                    arr_encountered = True

            # Add the processed index to the current component
            current_component.append(python_to_stan_inds())

        # Record the last component
        components.append(current_component)

        # Join all components
        return dist + "[" + "][".join(",".join(c) for c in components) + "]"

    # Stan code level is always 0 for this transformation
    @property
    def stan_code_level(self) -> int:
        return 0
