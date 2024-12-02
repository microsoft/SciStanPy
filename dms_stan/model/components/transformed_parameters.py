"""Holds parameter transformations for DMS Stan models."""

from abc import abstractmethod
from typing import Optional

import numpy as np
import numpy.typing as npt
import scipy.special as sp
import torch
import torch.nn as nn

import dms_stan as dms

from .abstract_model_component import AbstractModelComponent


def _is_elementwise_operation(*params: AbstractModelComponent) -> bool:
    """
    If all parameters have more than 0 dimensions and the last dimension is larger
    than 1, return True. This indicates that the operation is elementwise.
    """
    return all(param.ndim > 0 and param.shape[-1] > 1 for param in params)


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

    missing_param_order: tuple[str, ...]

    def __init__(
        self,
        *,
        shape: tuple[int, ...] = (),
        **parameters: "dms.custom_types.CombinableParameterType",
    ):
        # Run the parent class method
        super().__init__(shape=shape, **parameters)

        # Make sure missing param order is set
        if set(self._parents) != set(self.missing_param_order):
            raise ValueError("Missing parameter order must be set.")

        # We will need a shallow copy of the torch parameters at some point
        self._copied_torch_parameters: dict[str, torch.Tensor]

        # We will also need to set a missing parameter
        self._missing_param: str

        # The shape is the same as the draw shape
        self._shape = self.draw_shape

    def _set_missing_param(self):
        """
        Selects the missing parameter from the list of potential missing parameters.
        This is the degree of freedom that we solve for in order to calculate the
        inverse-transformed parameters.
        """
        # Set the missing parameter
        for paramname in self.missing_param_order:

            # The parameter cannot be shared
            if paramname in self._shared_parameters:
                continue

            # Take the first learnable parameter
            if isinstance(self._torch_parameters[paramname], nn.Parameter):
                self._missing_param = paramname
                break
        else:
            raise ValueError(
                "All parameters are constants and/or shared already. Cannot backpropagate."
            )

        # Remove the missing parameter from the dictionary
        self._torch_parameters = self._copied_torch_parameters.copy()
        del self._torch_parameters[self._missing_param]

    def init_pytorch(
        self, draws: dict[AbstractModelComponent, npt.NDArray] | None = None
    ) -> None:

        # Run the parent class method
        super().init_pytorch(draws)

        # The parameters must overlap with the missing parameter order
        assert set(self._torch_parameters.keys()) == set(self.missing_param_order)

        # Copy the torch parameters
        self._copied_torch_parameters = self._torch_parameters.copy()

        # Choose a missing parameter
        self._set_missing_param()

    def _link_to_sibling(
        self, parent: AbstractModelComponent, sibling: AbstractModelComponent
    ) -> None:

        # Run the parent class method
        super()._link_to_sibling(parent, sibling)

        # If the missing parameter is now shared, we need to reset it
        if self._missing_param in self._shared_parameters:
            self._set_missing_param()

    @abstractmethod
    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:
        """
        Calculates the missing parameter from the observable. This is necessary
        for calculating the inverse-transformed parameters.

        Args:
            observable (torch.Tensor): The observable value.

        Returns:
            torch.Tensor: The missing parameter.
        """

    def _get_transformed_torch_parameters(self) -> dict[str, torch.Tensor]:

        # Run the parent class method
        transformed_params = super()._get_transformed_torch_parameters()

        # Get the observable. This is what we will use to calculate the missing
        # parameter.
        observable = self.get_torch_observables()

        # Calculate the missing parameter
        assert self._missing_param not in transformed_params
        transformed_params[self._missing_param] = self.calculate_missing_param(
            observable
        )

        # Only return named parameters
        transformed_params = {
            k: v for k, v in transformed_params.items() if k in self._parents
        }
        assert set(transformed_params) == set(self._parents)

        return transformed_params

    def _draw(self, n: int, level_draws: dict[str, npt.NDArray]) -> npt.NDArray:
        """Sample from this parameter's distribution `n` times."""
        # Perform the operation on the draws
        return self.operation(**level_draws)

    @abstractmethod
    def operation(self, **draws: "dms.custom_types.SampleType") -> npt.NDArray:
        """Perform the operation on the draws"""

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

    # Calling this class should return the result of the operation.
    def __call__(self, *args, **kwargs):
        return self.operation(*args, **kwargs)


class BinaryTransformedParameter(TransformedParameter):
    """
    Identical to the TransformedParameter class, but only for operations involving
    two parameters. In other words, two parameters must be passed to the class.
    """

    missing_param_order = ("dist1", "dist2")

    def __init__(
        self,
        dist1: "dms.custom_types.CombinableParameterType",
        dist2: "dms.custom_types.CombinableParameterType",
        shape: tuple[int, ...] = (),
    ):
        super().__init__(dist1=dist1, dist2=dist2, shape=shape)

    def init_pytorch(
        self, draws: dict[AbstractModelComponent, npt.NDArray] | None = None
    ) -> None:
        # Run the parent class method
        super().init_pytorch(draws)

        # There should be one torch parameter
        assert len(self._torch_parameters) == 1

    @abstractmethod
    def operation(  # pylint: disable=arguments-differ
        self,
        dist1: "dms.custom_types.SampleType",
        dist2: "dms.custom_types.SampleType",
    ): ...

    @abstractmethod
    def calculate_dist1(self, observable: torch.Tensor) -> torch.Tensor:
        """
        Calculates the first distribution from the observable. This is necessary
        for calculating the missing parameter.

        Args:
            observable (torch.Tensor): The observable value.

        Returns:
            torch.Tensor: The first distribution.
        """

    @abstractmethod
    def calculate_dist2(self, observable: torch.Tensor) -> torch.Tensor:
        """
        Calculates the second distribution from the observable. This is necessary
        for calculating the missing parameter.

        Args:
            observable (torch.Tensor): The observable value.

        Returns:
            torch.Tensor: The second distribution.
        """

    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:
        if self._missing_param == "dist1":
            return self.calculate_dist1(observable)
        elif self._missing_param == "dist2":
            return self.calculate_dist2(observable)
        else:
            raise ValueError("Missing parameter not recognized.")

    @abstractmethod
    def format_stan_code(  # pylint: disable=arguments-differ
        self, dist1: str, dist2: str
    ) -> str: ...

    @property
    def is_elementwise_operation(self) -> bool:
        """Return whether the operation is elementwise or not."""
        return _is_elementwise_operation(*self.parents)


class UnaryTransformedParameter(TransformedParameter):
    """Transformed parameter that only requires one parameter."""

    missing_param_order = ("dist1",)  # The one and only parameter

    def __init__(
        self,
        dist1: "dms.custom_types.CombinableParameterType",
        shape: tuple[int, ...] = (),
    ):
        super().__init__(dist1=dist1, shape=shape)

    def init_pytorch(
        self, draws: dict[AbstractModelComponent, npt.NDArray] | None = None
    ) -> None:
        # Run the parent class method
        super().init_pytorch(draws)

        # There should be no torch parameters
        assert len(self._torch_parameters) == 0

    @abstractmethod
    def operation(  # pylint: disable=arguments-differ
        self, dist1: "dms.custom_types.SampleType"
    ) -> npt.NDArray: ...

    @abstractmethod
    def format_stan_code(  # pylint: disable=arguments-differ
        self, dist1: str
    ) -> str: ...


class AddParameter(BinaryTransformedParameter):
    """Defines a parameter that is the sum of two other parameters."""

    def operation(
        self,
        dist1: "dms.custom_types.SampleType",
        dist2: "dms.custom_types.SampleType",
    ) -> npt.NDArray:
        return dist1 + dist2

    def format_stan_code(self, dist1: str, dist2: str) -> str:
        return f"{dist1} + {dist2}"

    def calculate_dist1(self, observable: torch.Tensor) -> torch.Tensor:
        return observable - self._get_torch_parameter("dist2")

    def calculate_dist2(self, observable: torch.Tensor) -> torch.Tensor:
        return observable - self._get_torch_parameter("dist1")


class SubtractParameter(BinaryTransformedParameter):
    """Defines a parameter that is the difference of two other parameters."""

    def operation(
        self,
        dist1: "dms.custom_types.SampleType",
        dist2: "dms.custom_types.SampleType",
    ) -> npt.NDArray:
        return dist1 - dist2

    def format_stan_code(self, dist1: str, dist2: str) -> str:
        return f"{dist1} - {dist2}"

    def calculate_dist1(self, observable: torch.Tensor) -> torch.Tensor:
        return observable + self._get_torch_parameter("dist2")

    def calculate_dist2(self, observable: torch.Tensor) -> torch.Tensor:
        return self._get_torch_parameter("dist1") - observable


class MultiplyParameter(BinaryTransformedParameter):
    """Defines a parameter that is the product of two other parameters."""

    def operation(
        self,
        dist1: "dms.custom_types.SampleType",
        dist2: "dms.custom_types.SampleType",
    ) -> npt.NDArray:
        return dist1 * dist2

    def format_stan_code(self, dist1: str, dist2: str) -> str:

        # Get the operator
        operator = ".*" if self.is_elementwise_operation else "*"

        return f"{dist1} {operator} {dist2}"

    def calculate_dist1(self, observable: torch.Tensor) -> torch.Tensor:
        return observable / self._get_torch_parameter("dist2")

    def calculate_dist2(self, observable: torch.Tensor) -> torch.Tensor:
        return observable / self._get_torch_parameter("dist1")


class DivideParameter(BinaryTransformedParameter):
    """Defines a parameter that is the quotient of two other parameters."""

    def operation(
        self,
        dist1: "dms.custom_types.SampleType",
        dist2: "dms.custom_types.SampleType",
    ) -> npt.NDArray:
        return dist1 / dist2

    def format_stan_code(self, dist1: str, dist2: str) -> str:
        # Get the operator
        operator = "./" if self.is_elementwise_operation else "/"

        return f"{dist1} {operator} {dist2}"

    def calculate_dist1(self, observable: torch.Tensor) -> torch.Tensor:
        return observable * self._get_torch_parameter("dist2")

    def calculate_dist2(self, observable: torch.Tensor) -> torch.Tensor:
        return self._get_torch_parameter("dist1") / observable


class PowerParameter(BinaryTransformedParameter):
    """Defines a parameter raised to the power of another parameter."""

    def operation(
        self,
        dist1: "dms.custom_types.SampleType",
        dist2: "dms.custom_types.SampleType",
    ) -> npt.NDArray:
        return dist1**dist2

    def format_stan_code(self, dist1: str, dist2: str) -> str:

        # Get the operator
        operator = ".^" if self.is_elementwise_operation else "^"

        return f"{dist1} {operator} {dist2}"

    def calculate_dist1(self, observable: torch.Tensor) -> torch.Tensor:
        return observable ** (1 / self._get_torch_parameter("dist2"))

    def calculate_dist2(self, observable: torch.Tensor) -> torch.Tensor:
        return torch.log(observable) / torch.log(self._get_torch_parameter("dist1"))


class NegateParameter(UnaryTransformedParameter):
    """Defines a parameter that is the negative of another parameter."""

    def operation(self, dist1: "dms.custom_types.SampleType") -> npt.NDArray:
        return -dist1

    def format_stan_code(self, dist1: str) -> str:
        return f"-{dist1}"

    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:
        return -observable


class AbsParameter(UnaryTransformedParameter):
    """Defines a parameter that is the absolute value of another."""

    STAN_LOWER_BOUND: float = 0.0

    def init_pytorch(
        self, draws: Optional[dict[AbstractModelComponent, npt.NDArray]] = None
    ) -> None:
        # Run inherited
        super().init_pytorch(draws)

        # We have an additional parameter that we need to learn (the sign)
        self._torch_parameters["sign"] = nn.Parameter(torch.rand(self.dist1.shape))

    def operation(self, dist1: "dms.custom_types.SampleType") -> npt.NDArray:
        return np.abs(dist1)

    def format_stan_code(self, dist1: str) -> str:
        return f"abs({dist1})"

    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:

        # Get the sign of the observable
        sign = torch.sigmoid(self._torch_parameters["sign"]) * 2 - 1

        # Add the sign to the observable
        return observable * sign


class LogParameter(UnaryTransformedParameter):
    """Defines a parameter that is the natural logarithm of another."""

    # The distribution must be positive
    POSITIVE_PARAMS = {"dist1"}

    STAN_LOWER_BOUND: float = 0.0

    def operation(self, dist1: "dms.custom_types.SampleType") -> npt.NDArray:
        return np.log(dist1)

    def format_stan_code(self, dist1: str) -> str:
        return f"log({dist1})"

    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:
        return torch.exp(observable)


class ExpParameter(UnaryTransformedParameter):
    """Defines a parameter that is the exponential of another."""

    STAN_LOWER_BOUND: float = 0.0

    def operation(self, dist1: "dms.custom_types.SampleType") -> npt.NDArray:
        return np.exp(dist1)

    def format_stan_code(self, dist1: str) -> str:
        return f"exp({dist1})"

    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:
        return torch.log(observable)


class NormalizeParameter(UnaryTransformedParameter):
    """Defines a parameter that is normalized to sum to 1."""

    STAN_LOWER_BOUND: float = 0.0
    STAN_UPPER_BOUND: float = 1.0

    def operation(self, dist1: "dms.custom_types.SampleType") -> npt.NDArray:
        return dist1 / np.sum(dist1, keepdims=True, axis=-1)

    def format_stan_code(self, dist1: str) -> str:
        return f"{dist1} / sum({dist1})"

    def init_pytorch(
        self, draws: Optional[dict[AbstractModelComponent, npt.NDArray]] = None
    ) -> None:

        # Run the parent class method
        super().init_pytorch(draws)

        # We have an additional parameter to learn (the scale)
        self._torch_parameters["scale"] = nn.Parameter(
            torch.ones(self.dist1.shape[:-1] + (1,))
        )

    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:
        return observable * self._torch_parameters["scale"]


class NormalizeLogParameter(UnaryTransformedParameter):
    """
    Defines a parameter that is normalized such that exp(x) sums to 1. By extension,
    this assumes that the input is log-transformed.
    """

    STAN_UPPER_BOUND: float = 0.0

    def operation(self, dist1: "dms.custom_types.SampleType") -> npt.NDArray:
        return dist1 - sp.logsumexp(dist1, keepdims=True, axis=-1)

    def format_stan_code(self, dist1: str) -> str:
        return f"{dist1} - log_sum_exp({dist1})"

    def init_pytorch(
        self, draws: Optional[dict[AbstractModelComponent, npt.NDArray]] = None
    ) -> None:

        # Run the parent class method
        super().init_pytorch(draws)

        # We have an additional parameter to learn (the scale)
        self._torch_parameters["scale"] = nn.Parameter(
            torch.zeros(self.dist1.shape[:-1] + (1,))
        )

    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:
        return observable + self._torch_parameters["scale"]


class Growth(TransformedParameter):
    """Base class for growth models."""

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        t: "dms.custom_types.CombinableParameterType",
        shape: tuple[int, ...] = (),
        **params: "dms.custom_types.CombinableParameterType",
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

    missing_param_order = ("log_A", "r", "t")

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

    def operation(  # pylint: disable=arguments-differ
        self,
        *,
        t: "dms.custom_types.SampleType",
        log_A: "dms.custom_types.SampleType",
        r: "dms.custom_types.SampleType",
    ) -> npt.NDArray:
        return log_A + r * t

    def format_stan_code(  # pylint: disable=arguments-differ
        self, t: str, log_A: str, r: str
    ) -> str:
        # Decide on operator between r and t
        operator = ".*" if _is_elementwise_operation(self.t, self.r) else "*"

        # Build the transformation
        return f"{log_A} + {r} {operator} {t}"

    def calculate_logA(  # pylint: disable=invalid-name
        self, observable: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the log of the amplitude from the observable."""
        return observable - self._get_torch_parameter("r") * self._get_torch_parameter(
            "t"
        )

    def calculate_r(self, observable: torch.Tensor) -> torch.Tensor:
        """Calculates the growth rate from the observable."""
        # Calling `self.parameters` multiple times is inefficient. We call the
        # function wrapped by that property once and store the result.
        return (
            observable - self._get_torch_parameter("log_A")
        ) / self._get_torch_parameter("t")

    def calculate_t(self, observable: torch.Tensor) -> torch.Tensor:
        """Calculates the time from the observable."""
        # Calling `self.parameters` multiple times is inefficient. We call the
        # function wrapped by that property once and store the result.
        return (
            observable - self._get_torch_parameter("log_A")
        ) / self._get_torch_parameter("r")

    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:
        if self._missing_param == "log_A":
            return self.calculate_logA(observable)
        elif self._missing_param == "r":
            return self.calculate_r(observable)
        elif self._missing_param == "t":
            return self.calculate_t(observable)
        else:
            raise ValueError("Missing parameter not recognized.")


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

    missing_param_order = ("log_A", "r", "c", "t")

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
            t ("dms.custom_types.SampleType"): The time parameter.

            log_A ("dms.custom_types.SampleType"): The log of the amplitude parameter.

            r ("dms.custom_types.SampleType"): The rate parameter.

            c ("dms.custom_types.SampleType"): The offset parameter.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super().__init__(t=t, log_A=log_A, r=r, c=c, shape=shape)

    def operation(  # pylint: disable=arguments-differ
        self,
        *,
        t: "dms.custom_types.SampleType",
        log_A: "dms.custom_types.SampleType",
        r: "dms.custom_types.SampleType",
        c: "dms.custom_types.SampleType",
    ) -> npt.NDArray:
        return log_A - np.log(1 + np.exp(-r * (t - c)))

    def format_stan_code(  # pylint: disable=arguments-differ
        self, t: str, log_A: str, r: str, c: str
    ) -> str:
        # Determine the operator between r and t
        operator = ".*" if _is_elementwise_operation(self.t, self.r) else "*"

        return f"{log_A} - log(1 + exp(-{r} {operator} ({t} - {c})))"

    def calculate_logA(  # pylint: disable=invalid-name
        self, observable: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the log of the amplitude from the observable."""

        return observable + torch.log(
            1
            + torch.exp(
                self._get_torch_parameter("r")
                * (self._get_torch_parameter("c") - self._get_torch_parameter("t"))
            )
        )

    def calculate_r(self, observable: torch.Tensor) -> torch.Tensor:
        """Calculates the growth rate from the observable."""

        return torch.log(
            torch.exp(self._get_torch_parameter("log_A") - observable) - 1
        ) / (self._get_torch_parameter("c") - self._get_torch_parameter("t"))

    def calculate_c(self, observable: torch.Tensor) -> torch.Tensor:
        """Calculates the inflection point from the observable."""

        return (
            torch.log(torch.exp(self._get_torch_parameter("log_A") - observable) - 1)
            / self._get_torch_parameter("r")
        ) + self._get_torch_parameter("t")

    def calculate_t(self, observable: torch.Tensor) -> torch.Tensor:
        """Calculates the time from the observable."""

        return self._get_torch_parameter("c") - (
            torch.log(torch.exp(self._get_torch_parameter("log_A") - observable) - 1)
            / self._get_torch_parameter("r")
        )

    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:
        if self._missing_param == "log_A":
            return self.calculate_logA(observable)
        elif self._missing_param == "r":
            return self.calculate_r(observable)
        elif self._missing_param == "t":
            return self.calculate_t(observable)
        elif self._missing_param == "c":
            return self.calculate_c(observable)
        else:
            raise ValueError("Missing parameter not recognized.")
