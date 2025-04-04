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


class TransformedParameter(AbstractModelComponent, TransformableParameter):
    """
    Base class representing a parameter that is the result of combining other
    parameters using mathematical operations.
    """

    STAN_OPERATOR: str = ""  # Operator for the operation in Stan

    def _draw(
        self, n: int, level_draws: dict[str, npt.NDArray], seed: Optional[int]
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

    def get_transformation_assignment(self, index_opts: tuple[str, ...]) -> str:
        """Return the assignment for the transformation."""
        return f"{self.get_indexed_varname(index_opts)} = " + self.get_stan_code(
            index_opts
        )

    def get_target_incrementation(self, index_opts: tuple[str, ...]) -> str:
        """Null operation for transformed parameters by default."""
        return ""

    def _handle_transformation_code(
        self, param: AbstractModelComponent, index_opts: tuple[str, ...]
    ) -> str:
        """Works with the `get_stan_code` method from the parent class"""
        if param.is_named:
            return param.get_indexed_varname(index_opts)
        else:
            return f"( {param.get_stan_code(index_opts)} )"

    @abstractmethod
    def _write_operation(self, **to_format: str) -> str:
        """Write the operation in Stan code."""
        # The Stan operator must be defined in the child class
        if self.STAN_OPERATOR == "":
            raise NotImplementedError("The STAN_OPERATOR must be defined.")

        return ""

    def get_stan_code(self, index_opts: tuple[str, ...]) -> str:
        return self._write_operation(**super().get_stan_code(index_opts))

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
        shape: tuple[int, ...] = (),
    ):
        super().__init__(dist1=dist1, dist2=dist2, shape=shape)

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
        shape: tuple[int, ...] = (),
    ):
        super().__init__(dist1=dist1, shape=shape)

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
        shape: tuple[int, ...] = (),
    ):
        """Initializes the LogExponentialGrowth distribution.

        Args:
            t ("dms.custom_types.SampleType"): The time parameter.

            A ("dms.custom_types.SampleType"): The amplitude parameter.

            r ("dms.custom_types.SampleType"): The rate parameter.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super(UnaryTransformedParameter, self).__init__(t=t, A=A, r=r, shape=shape)

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
        shape: tuple[int, ...] = (),
    ):
        """Initializes the LogExponentialGrowth distribution.

        Args:
            t ("dms.custom_types.SampleType"): The time parameter.

            log_A ("dms.custom_types.SampleType"): The log of the amplitude parameter.

            r ("dms.custom_types.SampleType"): The rate parameter.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super().__init__(t=t, log_A=log_A, r=r, shape=shape)

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
        shape: tuple[int, ...] = (),
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
        super(UnaryTransformedParameter, self).__init__(t=t, A=A, r=r, c=c, shape=shape)

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
        shape: tuple[int, ...] = (),
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
            t=t, log_A=log_A, r=r, c=c, shape=shape
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
    in terms of starting abundances and rate. This parametrization parallels the
    parametrization used by `ExponentialGrowth`. Specifically, we calculate the
    abundance `x` at time `t` for a population with initial abundance `x0` growing
    with rate `r` as:

    \begin{align}
    x(t) = \frac{x0\exp{rt}}{1 + x0(\exp(rt) - 1)}
    \end{align}.

    Note that we define this function such that the carrying capacity of the system
    is "1". This parametrization enforces that x0 obeys the following equality,
    which is simply the standard sigmoid growth curve evaluated at `t = 0`:

    \begin{align}
    x0 = (1 + \exp(rc))^{-1}
    \end{align},

    where `c` is the offset parameter described in `SigmoidGrowth`. This means that
    `x0` is implicity capturing the shift parameter, allowing it to be ignored.
    Indeed, by rearranging the above equality for `x0` to solve for `c`, then plugging
    c into the the standard logistic growth equation, we arrive at the initial-abundance
    form of the equation:

    \begin{align}
    x0 &= (1 + \exp(rc))^{-1}
    x0^{-1} - 1 &= \exp(rc)
    \frac{1}{r}\ln{(x0^{-1} - 1)} &= c

    x(t) &= \frac{1}{1 + \exp{-r(t - c)}}
    &= \frac{1}{1 + \exp{-rt + rc}}
    &= \frac{1}{1 + \exp{-rt}\exp{r(\frac{1}{r}\ln{(x0^{-1} - 1)})}}
    &= \frac{1}{1 + \exp{-rt}\exp{\ln{(x0^{-1} - 1)}}}
    &= \frac{1}{1 + (x0^{-1} - 1)\exp{-rt}}
    &= \frac{1}{x0 + (1 - x0)\exp{-rt}}
    &= \frac{\exp{rt}}{x0\exp{rt} + (1 - x0)}
    \end{align}
    """

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        t: "dms.custom_types.CombinableParameterType",
        x0: "dms.custom_types.CombinableParameterType",
        r: "dms.custom_types.CombinableParameterType",
        shape: tuple[int, ...] = (),
    ):
        """Initializes the SigmoidGrowthInitParametrization distribution.

        Args:
            t (dms.custom_types.CombinableParameterType): The time parameter.
            x0 (dms.custom_types.CombinableParameterType): Initial abundances.
            r (dms.custom_types.CombinableParameterType): Growth rate.
            shape (tuple[int, ...], optional): The shape of the distribution. Defaults
            to ().
        """
        super().__init__(t=t, x0=x0, r=r, shape=shape)

    # pylint: disable=arguments-differ
    @overload
    def operation(
        self, t: torch.Tensor, x0: torch.Tensor, r: torch.Tensor
    ) -> torch.Tensor: ...

    @overload
    def operation(
        self,
        t: "dms.custom_types.SampleType",
        x0: "dms.custom_types.SampleType",
        r: "dms.custom_types.SampleType",
    ) -> npt.NDArray: ...

    def operation(self, t, x0, r):
        return dms.utils.stable_x0_sigmoid_growth(t=t, x0=x0, r=r)

    def _write_operation(self, t: str, x0: str, r: str):
        """We need a custom stan function for this"""
        return f"sigmoid_growth_init_param({t}, {x0}, {r})"

    # pylint: enable=arguments-differ
