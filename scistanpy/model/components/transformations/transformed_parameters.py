"""Parameter transformation components for SciStanPy models.

This module provides a comprehensive library of mathematical transformations that
can be applied to model parameters. These transformations enable complex model
construction through composition of simple mathematical operations while maintaining
automatic differentiation capabilities and Stan code generation.

The transformation system supports:

Transformation Categories:
    - **Arithmetic Operations**: Addition, subtraction, multiplication, division, powers
    - **Mathematical Functions**: Logarithms, exponentials, absolute values, sigmoids
    - **Statistical Operations**: Normalization, log-sum-exp, reductions
    - **Growth Models**: Exponential and sigmoid growth parameterizations
    - **Array Operations**: Indexing, slicing, convolution

Key Features:
    - **Multi-Backend Support**: Works with NumPy, PyTorch, and Stan
    - **Automatic Differentiation**: Maintains gradient flow for PyTorch optimization
    - **Stan Code Generation**: Automatic translation to Stan programming language
    - **Operator Overloading**: Natural mathematical syntax through Python operators
    - **Shape Broadcasting**: Automatic handling of multi-dimensional operations
    - **Type Safety**: Comprehensive validation and error checking

Transformation Architecture:
    The module provides several base classes that establish the transformation framework:
    - **TransformableParameter**: Mixin enabling operator overloading
    - **Transformation**: Base class for all transformation types
    - **TransformedParameter**: Foundation for parameter transformations
    - **BinaryTransformedParameter**: Two-operand mathematical operations
    - **UnaryTransformedParameter**: Single-operand mathematical operations

Advanced Transformations:
    Beyond basic arithmetic, the module includes specialized transformations for:
    - Growth modeling (exponential, sigmoid) with multiple parameterizations
    - Sequence processing (convolution operations)
    - Array manipulation (indexing with NumPy-compatible syntax)
    - Numerical stability (log-space operations, stable sigmoid)
    - Statistical operations (normalization, reductions)

Note that these will not typically be instantiated directly. Instead, use Python's
inbuilt operators between other model components or else use SciStanPy's `operations`
submodule.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Optional, overload, TYPE_CHECKING, Union

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
    """Mixin class enabling mathematical operator overloading for parameters.

    This mixin class provides Python operator overloading capabilities that allow
    parameters to be combined using natural mathematical syntax. Each operator
    creates an appropriate TransformedParameter instance that represents the
    mathematical operation.

    The mixin supports all standard arithmetic operators:
    - Addition (+), subtraction (-)
    - Multiplication (*), division (/)
    - Exponentiation (**)
    - Unary negation (-)

    Each operation supports both left and right operand positioning, enabling
    flexible mathematical expressions with mixed parameter and constant types.

    Example:
        >>> param1 = Normal(mu=0, sigma=1)
        >>> param2 = Normal(mu=1, sigma=0.5)
        >>> # All of these create TransformedParameter instances
        >>> sum_param = param1 + param2
        >>> scaled_param = 2 * param1
        >>> ratio_param = param1 / param2
        >>> power_param = param1 ** 2
        >>> negated_param = -param1
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
    """Base class for all parameter transformations in SciStanPy.

    This abstract base class provides the foundational infrastructure for
    creating transformations that can be used in both the transformed parameters
    and transformed data blocks of Stan models. It handles the common aspects
    of transformation operations including shape validation and Stan code generation.

    :cvar SHAPE_CHECK: Whether to perform automatic shape checking. Defaults to True.

    Key Responsibilities:
    - Coordinate transformation assignment code generation
    - Manage shape validation for transformation outputs
    - Provide abstract interface for Stan operation writing
    - Handle index options and assignment formatting

    The class provides methods for generating Stan code assignments and manages
    the interaction between transformation inputs and outputs. Subclasses must
    implement the write_stan_operation method to define their specific
    mathematical operations.

    Shape handling can be disabled for transformations that perform reductions
    or other operations that fundamentally change dimensionality.
    """

    SHAPE_CHECK: bool = True

    def _transformation(
        self,
        index_opts: tuple[str, ...] | None,
        assignment_kwargs: dict | None = None,
        right_side_kwargs: dict | None = None,
    ) -> str:
        """Generate complete transformation assignment code.

        :param index_opts: Index variable names for multi-dimensional operations
        :type index_opts: Optional[tuple[str, ...]]
        :param assignment_kwargs: Keyword arguments for assignment formatting. Defaults to None.
        :type assignment_kwargs: Optional[dict]
        :param right_side_kwargs: Keyword arguments for right-side formatting. Defaults to None.
        :type right_side_kwargs: Optional[dict]

        :returns: Complete Stan assignment statement
        :rtype: str

        This method combines the left-hand side variable name with the right-hand
        side operation to create a complete Stan assignment statement suitable
        for use in transformed parameters or transformed data blocks.
        """
        # Set defaults
        assignment_kwargs = assignment_kwargs or {}
        right_side_kwargs = right_side_kwargs or {}

        # Build assignment
        return (
            f"{self.get_indexed_varname(index_opts, **assignment_kwargs)} = "
            + self.get_right_side(index_opts, **right_side_kwargs)
        )

    @abstractmethod
    def write_stan_operation(self, **to_format: str) -> str:
        """Generate Stan code for the specific transformation operation.

        :param to_format: Formatted parameter strings for Stan code
        :type to_format: str

        :returns: Stan code representing the mathematical operation
        :rtype: str

        This abstract method must be implemented by all concrete transformation
        classes to define how their specific mathematical operation is represented
        in Stan code.
        """

    def get_right_side(
        self,
        index_opts: tuple[str, ...] | None,
        start_dims: dict[str, "custom_types.Integer"] | None = None,
        end_dims: dict[str, "custom_types.Integer"] | None = None,
        offset_adjustment: int = 0,
    ) -> str:
        """Generate right-hand side of transformation assignment.

        :param index_opts: Index options for multi-dimensional operations
        :type index_opts: Optional[tuple[str, ...]]
        :param start_dims: First indexable dimension of parent parameters. Defaults to None.
        :type start_dims: Optional[dict[str, custom_types.Integer]]
        :param end_dims: Last indexable dimension of parent parameters. Defaults to None.
        :type end_dims: Optional[dict[str, custom_types.Integer]]
        :param offset_adjustment: Index offset adjustment. Defaults to 0.
        :type offset_adjustment: int

        :returns: Stan code for the right-hand side of the assignment
        :rtype: str

        This method coordinates the formatting of parent parameters and applies
        the transformation operation. It automatically adds parentheses around
        operations when the transformation is not named.
        """
        # Call the inherited method to get a dictionary mapping parent names to
        # either their indexed variable names (if they are named) or the thread
        # of operations that define them (if they are not named).
        components = super().get_right_side(
            index_opts,
            start_dims=start_dims,
            end_dims=end_dims,
            offset_adjustment=offset_adjustment,
        )

        # Format the right-hand side of the operation. The declaration for any operation
        # that is not named should be wrapped in parentheses. Otherwise, exactly
        # how formatting is done depends on the child class.
        stan_op = self.write_stan_operation(**components)
        if not self.is_named:
            stan_op = f"({stan_op})"
        return stan_op

    def _set_shape(self) -> None:
        """Set component shape with optional shape checking bypass.

        Some transformations perform reductions or other operations that change
        dimensionality. When SHAPE_CHECK is False, automatic shape validation
        is bypassed to allow these operations.
        """
        if self.SHAPE_CHECK:
            super()._set_shape()

    def __str__(self) -> str:
        """Return human-readable string representation of the transformation.

        :returns: String showing transformation assignment
        :rtype: str

        Creates a readable representation of the transformation for debugging
        and model inspection, showing the assignment operation with cleaned
        formatting.
        """
        right_side = (
            self.get_right_side(None).replace("[start:end]", "").replace("__", ".")
        )
        return f"{self.model_varname} = {right_side}"


class TransformedParameter(Transformation, TransformableParameter):
    """Base class for transformed parameters that can be used in parameter blocks.

    This class provides the foundation for parameters that result from mathematical
    operations on other parameters. It handles both the computational aspects
    (sampling, PyTorch operations) and code generation aspects (Stan assignments).

    :cvar STAN_OPERATOR: Stan operator string for simple operations

    Transformed parameters support:
    - Sampling through parent parameter sampling and operation application
    - PyTorch operations with automatic differentiation
    - Stan code generation for transformed parameters block
    - Further transformation through operator overloading

    The class provides the infrastructure for creating complex mathematical
    expressions while maintaining compatibility with all SciStanPy backends.

    Subclasses must implement:
    - run_np_torch_op: The core mathematical operation
    - write_stan_operation: Stan code generation for the operation

    Example Usage:
        TransformedParameter subclasses are typically created through operator
        overloading or the `operations` submodule rather than direct instantiation,
        but can be used directly for custom operations.
    """

    STAN_OPERATOR: str = ""

    # The transformation is renamed to be more specific in the child classes
    get_transformation_assignment = Transformation._transformation

    def _draw(
        self,
        level_draws: dict[str, Union[npt.NDArray, "custom_types.Float"]],
        seed: Optional["custom_types.Integer"],  # pylint: disable=unused-argument
    ) -> Union[npt.NDArray, "custom_types.Float"]:
        """Draw samples by applying transformation to parent samples.

        :param level_draws: Samples from parent parameters
        :type level_draws: dict[str, Union[npt.NDArray, custom_types.Float]]
        :param seed: Random seed (unused for deterministic transformations)
        :type seed: Optional[custom_types.Integer]

        :returns: Transformed samples
        :rtype: Union[npt.NDArray, custom_types.Float]

        This method applies the mathematical transformation to samples drawn
        from parent parameters, enabling sampling from transformed distributions.
        """
        # Perform the operation on the draws
        return self.run_np_torch_op(**level_draws)

    @overload
    def run_np_torch_op(self, **draws: torch.Tensor) -> torch.Tensor: ...

    @overload
    def run_np_torch_op(self, **draws: "custom_types.SampleType") -> npt.NDArray: ...

    @abstractmethod
    def run_np_torch_op(self, **draws):
        """Execute the mathematical operation using NumPy or PyTorch.

        :param draws: Input values for the operation
        :type draws: Union[torch.Tensor, custom_types.SampleType]

        :returns: Result of the mathematical operation
        :rtype: Union[torch.Tensor, npt.NDArray]

        This abstract method defines the core computational logic for the
        transformation. It must handle both NumPy and PyTorch inputs
        appropriately to maintain backend compatibility.
        """

    @abstractmethod
    def write_stan_operation(self, **to_format: str) -> str:
        """Generate Stan code for the transformation operation.

        :param to_format: Formatted parameter strings
        :type to_format: str

        :returns: Stan code for the operation
        :rtype: str

        :raises NotImplementedError: If STAN_OPERATOR is not defined for simple operations

        This method must be implemented to provide appropriate Stan code
        for the mathematical operation represented by this transformation.
        """
        # The Stan operator must be defined in the child class
        if self.STAN_OPERATOR == "":
            raise NotImplementedError("The STAN_OPERATOR must be defined.")

        return ""

    def __call__(self, *args, **kwargs):
        """Enable calling transformed parameters as functions.

        :param args: Positional arguments passed to run_np_torch_op
        :param kwargs: Keyword arguments passed to run_np_torch_op

        :returns: Result of the operation

        This allows transformed parameters to be used as callable functions,
        providing a convenient interface for applying operations directly.
        """
        return self.run_np_torch_op(*args, **kwargs)

    @property
    def torch_parametrization(self) -> torch.Tensor:
        """Get PyTorch representation with transformations applied.

        :returns: PyTorch tensor with operation applied to parent tensors
        :rtype: torch.Tensor

        This property applies the transformation operation to the PyTorch
        parameterizations of all parent parameters, maintaining gradient
        flow for optimization.
        """
        # This is just the operation performed on the torch parameters of the parents
        return self.run_np_torch_op(
            **{
                name: param.torch_parametrization
                for name, param in self._parents.items()
            }
        )


class BinaryTransformedParameter(TransformedParameter):
    """Base class for transformations involving exactly two parameters.

    This class provides a specialized interface for binary mathematical operations
    such as addition, subtraction, multiplication, and division. It enforces the
    two-parameter constraint and provides appropriate method signatures.

    :param dist1: First parameter for the operation
    :type dist1: custom_types.CombinableParameterType
    :param dist2: Second parameter for the operation
    :type dist2: custom_types.CombinableParameterType
    :param kwargs: Additional arguments passed to parent class

    Binary operations are the foundation for arithmetic expressions and provide
    the building blocks for more complex mathematical relationships between
    parameters.

    Subclasses must implement run_np_torch_op with the (dist1, dist2) signature
    and can optionally override write_stan_operation for custom Stan formatting.
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
    def run_np_torch_op(self, dist1, dist2):
        """Execute binary operation on two inputs.

        :param dist1: First operand
        :param dist2: Second operand

        :returns: Result of binary operation

        This method implements the core binary mathematical operation
        for both NumPy and PyTorch backends.
        """

    def write_stan_operation(self, dist1: str, dist2: str) -> str:
        """Generate Stan code for binary operations using STAN_OPERATOR.

        :param dist1: Formatted string for first parameter
        :type dist1: str
        :param dist2: Formatted string for second parameter
        :type dist2: str

        :returns: Stan code with infix operator
        :rtype: str

        Generates Stan code in the format "dist1 OPERATOR dist2" for
        standard binary operations.
        """
        super().write_stan_operation()
        return f"{dist1} {self.STAN_OPERATOR} {dist2}"

    # pylint: enable=arguments-differ


class UnaryTransformedParameter(TransformedParameter):
    """Base class for transformations involving exactly one parameter.

    This class provides a specialized interface for unary mathematical operations
    such as negation, absolute value, logarithms, and exponentials. It enforces
    the single-parameter constraint and provides appropriate method signatures.

    :param dist1: Parameter for the operation
    :type dist1: custom_types.CombinableParameterType
    :param kwargs: Additional arguments passed to parent class

    Unary operations are essential for mathematical transformations and provide
    common functions needed in statistical modeling and parameter reparameterization.

    Subclasses must implement run_np_torch_op with the single-parameter signature
    and can optionally override write_stan_operation for custom Stan formatting.
    """

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
    def run_np_torch_op(self, dist1):
        """Execute unary operation on single input.

        :param dist1: Input operand

        :returns: Result of unary operation

        This method implements the core unary mathematical operation
        for both NumPy and PyTorch backends.
        """

    def write_stan_operation(self, dist1: str) -> str:
        """Generate Stan code for unary operations using STAN_OPERATOR.

        :param dist1: Formatted string for the parameter
        :type dist1: str

        :returns: Stan code with prefix operator
        :rtype: str

        Generates Stan code in the format "OPERATOR dist1" for
        standard unary operations.
        """
        super().write_stan_operation()
        return f"{self.STAN_OPERATOR}{dist1}"

    # pylint: enable=arguments-differ


# Basic arithmetic operations
class AddParameter(BinaryTransformedParameter):
    """Addition transformation for two parameters.

    Implements element-wise addition of two parameters: result = dist1 + dist2

    Mathematical Properties:
    - Commutative: a + b = b + a
    - Associative: (a + b) + c = a + (b + c)
    - Identity element: 0

    This transformation preserves the shape through broadcasting and is commonly
    used for combining effects, offsets, and linear combinations of parameters.
    """

    STAN_OPERATOR: str = "+"

    def run_np_torch_op(self, dist1, dist2):
        """Perform element-wise addition.

        :param dist1: First addend
        :param dist2: Second addend

        :returns: Sum of the two inputs
        """
        return dist1 + dist2


class SubtractParameter(BinaryTransformedParameter):
    """Subtraction transformation for two parameters.

    Implements element-wise subtraction of two parameters: result = dist1 - dist2

    Mathematical Properties:
    - Non-commutative: a - b ≠ b - a (generally)
    - Non-associative: (a - b) - c ≠ a - (b - c) (generally)
    - Identity element: 0 (for right operand)

    This transformation is commonly used for computing differences, residuals,
    and relative measures between parameters.
    """

    STAN_OPERATOR: str = "-"

    def run_np_torch_op(self, dist1, dist2):
        """Perform element-wise subtraction.

        :param dist1: Minuend
        :param dist2: Subtrahend

        :returns: Difference of the two inputs
        """
        return dist1 - dist2


class MultiplyParameter(BinaryTransformedParameter):
    """Element-wise multiplication transformation for two parameters.

    Implements element-wise multiplication of two parameters: result = dist1 * dist2

    Mathematical Properties:
    - Commutative: a * b = b * a
    - Associative: (a * b) * c = a * (b * c)
    - Identity element: 1

    This transformation is fundamental for scaling, interaction effects, and
    product relationships between parameters.
    """

    STAN_OPERATOR: str = ".*"

    def run_np_torch_op(self, dist1, dist2):
        """Perform element-wise multiplication.

        :param dist1: First factor
        :param dist2: Second factor

        :returns: Product of the two inputs
        """
        return dist1 * dist2


class DivideParameter(BinaryTransformedParameter):
    """Element-wise division transformation for two parameters.

    Implements element-wise division of two parameters: result = dist1 / dist2

    Mathematical Properties:
    - Non-commutative: a / b ≠ b / a (generally)
    - Non-associative: (a / b) / c ≠ a / (b / c) (generally)
    - Identity element: 1 (for right operand)

    This transformation is used for ratios, rates, normalized quantities,
    and relative measures. Care must be taken to avoid division by zero.
    """

    STAN_OPERATOR: str = "./"

    def run_np_torch_op(self, dist1, dist2):
        """Perform element-wise division.

        :param dist1: Dividend
        :param dist2: Divisor

        :returns: Quotient of the two inputs
        """
        return dist1 / dist2


class PowerParameter(BinaryTransformedParameter):
    """Element-wise exponentiation transformation for two parameters.

    Implements element-wise exponentiation: result = dist1 ^ dist2

    Mathematical Properties:
    - Non-commutative: a^b ≠ b^a (generally)
    - Non-associative: (a^b)^c ≠ a^(b^c) (generally)
    - Identity element: 1 (for exponent), any base^1 = base

    This transformation is used for power relationships, polynomial terms,
    and exponential scaling effects.
    """

    STAN_OPERATOR: str = ".^"

    def run_np_torch_op(self, dist1, dist2):
        """Perform element-wise exponentiation.

        :param dist1: Base
        :param dist2: Exponent

        :returns: dist1 raised to the power of dist2
        """
        return dist1**dist2


class NegateParameter(UnaryTransformedParameter):
    """Unary negation transformation for parameters.

    Implements element-wise negation of a parameter: result = -dist1

    Mathematical Properties:
    - Involutory: -(-a) = a
    - Distributive over addition: -(a + b) = -a + -b
    - Anti-distributive over multiplication: -(a * b) = (-a) * b = a * (-b)

    This transformation is used for sign changes, directional effects,
    and creating opposite relationships.
    """

    STAN_OPERATOR: str = "-"

    def run_np_torch_op(self, dist1):
        """Perform element-wise negation.

        :param dist1: Input parameter

        :returns: Negated input
        """
        return -dist1


class AbsParameter(UnaryTransformedParameter):
    r"""Absolute value transformation for parameters.

    Implements element-wise absolute value: result = \|dist1\|

    :cvar LOWER_BOUND: Absolute values are always non-negative (0.0)

    Mathematical Properties:
    - Non-negative output: \|a\| ≥ 0
    - Idempotent for non-negative inputs: \|\|a\|\| = \|a\|
    - Triangle inequality: \|a + b\| ≤ \|a\| + \|b\|

    This transformation ensures non-negative values and is commonly used
    for magnitudes, distances, and ensuring positive parameters.
    """

    LOWER_BOUND: "custom_types.Float" = 0.0

    def run_np_torch_op(self, dist1):
        """Compute element-wise absolute value.

        :param dist1: Input parameter

        :returns: Absolute value of input
        """
        return utils.choose_module(dist1).abs(dist1)

    def write_stan_operation(self, dist1: str) -> str:
        """Generate Stan absolute value function call.

        :param dist1: Formatted parameter string
        :type dist1: str

        :returns: Stan abs() function call
        :rtype: str
        """
        return f"abs({dist1})"


class LogParameter(UnaryTransformedParameter):
    """Natural logarithm transformation for parameters.

    Implements element-wise natural logarithm: result = ln(dist1)

    :cvar POSITIVE_PARAMS: Input must be positive ({"dist1"})

    Mathematical Properties:
    - Domain: (0, ∞)
    - Range: (-∞, ∞)
    - Logarithm laws: ln(ab) = ln(a) + ln(b), ln(a^b) = b*ln(a)
    - Inverse: exp(ln(a)) = a for a > 0

    This transformation is fundamental for log-scale modeling, multiplicative
    effects on additive scales, and ensuring positive-valued parameters.
    """

    POSITIVE_PARAMS = {"dist1"}

    def run_np_torch_op(self, dist1):
        """Compute element-wise natural logarithm.

        :param dist1: Input parameter (must be positive)

        :returns: Natural logarithm of input
        """
        return utils.choose_module(dist1).log(dist1)

    def write_stan_operation(self, dist1: str) -> str:
        """Generate Stan logarithm function call.

        :param dist1: Formatted parameter string
        :type dist1: str

        :returns: Stan log() function call
        :rtype: str
        """
        return f"log({dist1})"


class ExpParameter(UnaryTransformedParameter):
    """Exponential transformation for parameters.

    Implements element-wise exponential function: result = exp(dist1)

    :cvar LOWER_BOUND: Exponential values are always positive (0.0)

    Mathematical Properties:
    - Domain: (-∞, ∞)
    - Range: (0, ∞)
    - Exponential laws: exp(a+b) = exp(a)*exp(b), exp(a*b) ≠ exp(a)*exp(b)
    - Inverse: ln(exp(a)) = a

    This transformation is used for ensuring positive values, exponential
    growth models, and converting from log-scale to natural scale.
    """

    LOWER_BOUND: "custom_types.Float" = 0.0

    def run_np_torch_op(self, dist1):

        return utils.choose_module(dist1).exp(dist1)

    def write_stan_operation(self, dist1: str) -> str:
        """Generate Stan exponential function call.

        :param dist1: Formatted parameter string
        :type dist1: str

        :returns: Stan exp() function call
        :rtype: str
        """
        return f"exp({dist1})"


class NormalizeParameter(UnaryTransformedParameter):
    """Normalization transformation that scales values to sum to 1 in the last dimension
    of the input.

    Implements element-wise normalization where each vector is divided by its sum,
    creating probability vectors or normalized weights that sum to unity.

    :cvar LOWER_BOUND: Normalized values are non-negative (0.0)
    :cvar UPPER_BOUND: Normalized values are bounded above by 1.0

    Mathematical Properties:
    - Domain: [0, ∞)^n (non-negative input required)
    - Range: Probability simplex {x : Σxᵢ = 1, xᵢ ≥ 0}
    - Operation: xᵢ' = xᵢ / Σⱼ xⱼ
    - Preserves ratios between elements

    This transformation is essential for:
    - Converting counts to proportions
    - Creating probability vectors from non-negative weights
    - Normalizing attention weights or mixture components
    - Ensuring categorical probabilities sum to one

    The normalization is applied along the last dimension only, making it suitable
    for batch processing of multiple probability vectors.
    """

    LOWER_BOUND: "custom_types.Float" = 0.0
    UPPER_BOUND: "custom_types.Float" = 1.0

    def run_np_torch_op(self, dist1):
        """Compute normalization by dividing by sum along last dimension.

        :param dist1: Input parameter (must be non-negative)

        :returns: Normalized values that sum to 1 along last dimension
        """
        if isinstance(dist1, np.ndarray):
            return dist1 / np.sum(dist1, keepdims=True, axis=-1)
        elif isinstance(dist1, torch.Tensor):
            return dist1 / dist1.sum(dim=-1, keepdim=True)
        # Error if the type is not supported
        else:
            raise TypeError(
                "Unsupported type for dist1. Expected torch.Tensor or np.ndarray."
            )

    def write_stan_operation(self, dist1: str) -> str:
        """Generate Stan normalization code.

        :param dist1: Formatted parameter string
        :type dist1: str

        :returns: Stan code dividing by sum
        :rtype: str
        """
        return f"{dist1} / sum({dist1})"


class NormalizeLogParameter(UnaryTransformedParameter):
    """Log-space normalization transformation for log-probability vectors over the
    last dimension of the input.

    Implements normalization in log-space where log-probabilities are adjusted
    so that their exponentiated values sum to 1. This is equivalent to
    subtracting the log-sum-exp from each element.

    :cvar UPPER_BOUND: Log-probabilities are bounded above by 0.0

    Mathematical Properties:
    - Domain: (-∞, ∞)^n
    - Range: Log-simplex {x : Σexp(xᵢ) = 1}
    - Operation: xᵢ' = xᵢ - log(Σⱼ exp(xⱼ))
    - Numerically stable for extreme log-probabilities

    This transformation is crucial for:
    - Normalizing log-probabilities without exponentiation
    - Stable computation with very small probabilities
    - Log-space categorical distributions
    - Attention mechanisms in log-space

    The log-sum-exp operation provides numerical stability by avoiding
    overflow/underflow issues common with direct exponentiation. As with the `NormalizeParameter`,
    this is performed over the last dimension only.
    """

    UPPER_BOUND: "custom_types.Float" = 0.0

    def run_np_torch_op(self, dist1):
        """Compute log-space normalization using log-sum-exp.

        :param dist1: Input log-probabilities

        :returns: Normalized log-probabilities
        """
        if isinstance(dist1, torch.Tensor):
            return dist1 - torch.logsumexp(dist1, keepdims=True, dim=-1)
        elif isinstance(dist1, np.ndarray):
            return dist1 - sp.logsumexp(dist1, keepdims=True, axis=-1)
        else:
            raise TypeError(
                "Unsupported type for dist1. Expected torch.Tensor or np.ndarray."
            )

    def write_stan_operation(self, dist1: str) -> str:
        """Generate Stan log-space normalization code.

        :param dist1: Formatted parameter string
        :type dist1: str

        :returns: Stan code using log_sum_exp function
        :rtype: str
        """
        return f"{dist1} - log_sum_exp({dist1})"


class Reduction(UnaryTransformedParameter):
    """Base class for operations that reduce dimensionality.

    This abstract base class provides infrastructure for transformations that
    reduce the size of the last dimension through operations like sum, mean,
    or log-sum-exp. It handles shape management and provides specialized
    indexing behavior for reductions.

    :cvar SHAPE_CHECK: Disabled for reductions (False)
    :cvar TORCH_FUNC: PyTorch function for the reduction operation
    :cvar NP_FUNC: NumPy function for the reduction operation

    :param dist1: Parameter to reduce
    :type dist1: custom_types.CombinableParameterType
    :param keepdims: Whether to keep the reduced dimension as size 1. Defaults to False.
    :type keepdims: bool
    :param kwargs: Additional arguments passed to parent class

    Key Features:
    - Automatic shape calculation for reduced dimensions
    - Specialized index offset handling (always returns 0)
    - Increased assignment depth for proper Stan loop structure
    - Support for keepdims option to preserve dimensionality

    Subclasses must define TORCH_FUNC and NP_FUNC class variables that
    point to the appropriate reduction functions in each library.

    Example:
        Subclasses like SumParameter and LogSumExpParameter inherit from
        this base class and define their specific reduction operations.
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
        """Initialize reduction with automatic shape calculation. Reduction is always
        over the last dimension.

        :param dist1: Parameter to reduce
        :type dist1: custom_types.CombinableParameterType
        :param keepdims: Whether to keep reduced dimension. Defaults to False.
        :type keepdims: bool
        :param kwargs: Additional arguments for parent initialization

        The initialization automatically calculates the output shape by
        removing the last dimension (or keeping it as size 1 if keepdims=True).
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
        """Execute reduction operation with backend-appropriate function.

        :param dist1: Input parameter to reduce
        :param keepdim: Whether to keep dimensions (static method only)
        :type keepdim: Optional[bool]

        :returns: Reduced parameter values

        :raises ValueError: If keepdim is provided for instance method calls

        The method automatically selects between PyTorch and NumPy reduction
        functions and applies them along the last dimension.
        """
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

    def get_index_offset(
        self,
        query: Union[str, "abstract_model_component.AbstractModelComponent"],
        offset_adjustment: int = 0,
    ) -> int:
        """Return zero offset for all reduction operations.

        :param query: Component or parameter name (ignored)
        :param offset_adjustment: Offset adjustment (ignored)

        :returns: Always returns 0
        :rtype: int

        Reductions always return zero offset because they operate on the
        last dimension and don't require complex indexing adjustments.
        """
        return 0

    def get_assign_depth(self) -> int:
        """Return assignment depth one level higher than normal.

        :returns: Assignment depth + 1
        :rtype: int

        Reductions require one additional level of loop nesting to properly
        iterate over the dimension being reduced.
        """
        return super().get_assign_depth() + 1


class LogSumExpParameter(Reduction):
    """Log-sum-exp reduction transformation.

    Computes the logarithm of the sum of exponentials along the last dimension,
    providing a numerically stable way to compute log(Σᵢ exp(xᵢ)).

    :cvar TORCH_FUNC: torch.logsumexp
    :cvar NP_FUNC: scipy.special.logsumexp

    Mathematical Properties:
    - Operation: log(Σᵢ exp(xᵢ))
    - Numerically stable for extreme values
    - Maintains precision in log-space
    - Essential for probabilistic computations

    This transformation is fundamental for:
    - Normalizing log-probabilities
    - Computing partition functions
    - Stable softmax computations
    - Log-space mixture models

    The log-sum-exp function is the smooth maximum approximation and
    appears frequently in machine learning and statistics.

    Example:
        >>> log_weights = LogParameter(weights)
        >>> log_partition = LogSumExpParameter(log_weights)
        >>> normalized_log_weights = log_weights - log_partition
    """

    TORCH_FUNC = torch.logsumexp
    NP_FUNC = sp.logsumexp

    def write_stan_operation(self, dist1: str) -> str:
        """Generate Stan log_sum_exp function call.

        :param dist1: Formatted parameter string
        :type dist1: str

        :returns: Stan log_sum_exp function call
        :rtype: str
        """
        return f"log_sum_exp({dist1})"


class SumParameter(Reduction):
    """Sum reduction transformation.

    Computes the sum of values along the last dimension: Σᵢ xᵢ

    :cvar TORCH_FUNC: torch.sum
    :cvar NP_FUNC: numpy.sum

    Mathematical Properties:
    - Operation: Σᵢ xᵢ
    - Linear operation
    - Preserves units and scale
    - Fundamental aggregation operation

    This transformation is used for:
    - Computing totals and aggregates
    - Reducing dimensionality through summation
    - Calculating marginal quantities
    - Creating scalar summaries from vectors

    Example:
        >>> category_counts = Poisson(lambda_=rates)
        >>> total_count = SumParameter(category_counts)
    """

    TORCH_FUNC = torch.sum
    NP_FUNC = np.sum

    def write_stan_operation(self, dist1: str) -> str:
        """Generate Stan sum function call.

        :param dist1: Formatted parameter string
        :type dist1: str

        :returns: Stan sum function call
        :rtype: str
        """
        return f"sum({dist1})"


class Log1pExpParameter(UnaryTransformedParameter):
    """Log(1 + exp(x)) transformation with numerical stability.

    Computes log(1 + exp(x)) using numerically stable implementations that
    avoid overflow for large positive values and underflow for large negative values.

    Mathematical Properties:
    - Domain: (-∞, ∞)
    - Range: [0, ∞)
    - Smooth approximation to max(0, x)
    - Also known as "softplus" function

    This transformation is essential for:
    - Numerical stability in probabilistic models
    - Log-space computations avoiding direct exponentiation

    Example:
        >>> log_odds = Normal(mu=0, sigma=1)
        >>> log_prob = Log1pExpParameter(log_odds)
    """

    def run_np_torch_op(self, dist1):
        """Compute log(1 + exp(x)) with numerical stability.

        :param dist1: Input parameter

        :returns: log(1 + exp(dist1))

        Uses logaddexp(0, x) for numerical stability, which handles
        both overflow (large positive x) and underflow (large negative x).
        """
        # If using torch, use the logaddexp function directly.
        if isinstance(dist1, torch.Tensor):
            return torch.logaddexp(torch.tensor(0.0, device=dist1.device), dist1)
        # If using numpy, use the logaddexp function from scipy.
        elif isinstance(dist1, np.ndarray):
            return np.logaddexp(0.0, dist1)
        # Error if the type is not supported
        else:
            raise TypeError(
                "Unsupported type for dist1. Expected torch.Tensor or np.ndarray."
            )

    def write_stan_operation(self, dist1: str) -> str:
        """Generate Stan log1p_exp function call.

        :param dist1: Formatted parameter string
        :type dist1: str

        :returns: Stan log1p_exp function call
        :rtype: str
        """
        return f"log1p_exp({dist1})"


class SigmoidParameter(UnaryTransformedParameter):
    """Sigmoid (logistic) transformation for parameters.

    Implements the sigmoid function: result = 1 / (1 + exp(-dist1))

    :cvar LOWER_BOUND: Sigmoid values are bounded below by 0.0
    :cvar UPPER_BOUND: Sigmoid values are bounded above by 1.0

    Mathematical Properties:
    - Domain: (-∞, ∞)
    - Range: (0, 1)
    - S-shaped curve with inflection point at x=0, y=0.5
    - Inverse: logit function ln(p/(1-p))

    The sigmoid function is essential for:
    - Converting unbounded values to probabilities
    - Logistic regression and classification
    - Smooth transitions between bounds
    - Activation functions in neural networks

    This implementation uses numerically stable computation methods to
    avoid overflow/underflow issues.
    """

    UPPER_BOUND: "custom_types.Float" = 1.0
    LOWER_BOUND: "custom_types.Float" = 0.0

    def run_np_torch_op(self, dist1):
        """Compute sigmoid function with numerical stability.

        :param dist1: Input parameter (logits)

        :returns: Sigmoid-transformed values in (0, 1)
        :rtype: Union[torch.Tensor, np.ndarray]

        :raises TypeError: If input type is not supported

        Uses numerically stable implementations:
        - PyTorch: Built-in torch.sigmoid function
        - NumPy: Custom stable implementation from utils
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
        """Generate Stan inverse logit function call.

        :param dist1: Formatted parameter string
        :type dist1: str

        :returns: Stan inv_logit() function call
        :rtype: str
        """
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

    def __init__(
        self,
        *,
        t: "custom_types.CombinableParameterType",
        A: "custom_types.CombinableParameterType",
        r: "custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initialize exponential growth model.

        :param t: Time parameter
        :param A: Amplitude parameter
        :param r: Rate parameter
        :param kwargs: Additional arguments
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
        """Compute exponential growth: A * exp(r * t).

        :param t: Time values
        :param A: Amplitude values
        :param r: Rate values

        :returns: Exponential growth values
        """
        return A * ExpParameter.run_np_torch_op(self, r * t)

    # pylint: enable=arguments-differ

    def write_stan_operation(  # pylint: disable=arguments-differ
        self, t: str, A: str, r: str
    ) -> str:
        par_string = super().write_stan_operation(f"{r} .* {t}")
        return f"{A} .* {par_string}"


class BinaryExponentialGrowth(ExpParameter):
    """Binary exponential growth for two time points.

    Special case of exponential growth for modeling with only two time points,
    assuming t₀ = 0 and t₁ = 1. This reduces to: x = A * exp(r)

    :param A: Amplitude parameter (value at t=0)
    :type A: custom_types.CombinableParameterType
    :param r: Growth rate parameter
    :type r: custom_types.CombinableParameterType
    :param kwargs: Additional arguments passed to parent class

    Mathematical Properties:
    - Simplified exponential model for binary time points
    - A represents initial value
    - exp(r) represents fold-change from t=0 to t=1
    - r is log-fold-change

    This transformation is useful for:
    - Before/after comparisons
    - Treatment effect modeling
    - Two-timepoint studies
    - Fold-change analysis

    Example:
        >>> baseline = Normal(mu=100, sigma=10)
        >>> log_fold_change = Normal(mu=0, sigma=0.5)
        >>> final_value = BinaryExponentialGrowth(A=baseline, r=log_fold_change)
    """

    def __init__(
        self,
        A: "custom_types.CombinableParameterType",
        r: "custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initialize binary exponential growth.

        :param A: Amplitude parameter
        :param r: Rate parameter
        :param kwargs: Additional arguments
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
        """Compute binary exponential growth: A * exp(r).

        :param A: Amplitude values
        :param r: Rate values

        :returns: A * exp(r)
        """
        return A * ExpParameter.run_np_torch_op(self, r)

    def write_stan_operation(self, A: str, r: str) -> str:
        """Generate Stan code for binary exponential growth.

        :param A: Formatted amplitude parameter
        :type A: str
        :param r: Formatted rate parameter
        :type r: str

        :returns: Stan code for A .* exp(r)
        :rtype: str
        """
        return f"{A} .* {super().write_stan_operation(r)}"


class LogExponentialGrowth(TransformedParameter):
    """Log-scale exponential growth model transformation.

    Implements the logarithm of exponential growth: log(x) = log_A + r * t

    :param t: Time parameter
    :type t: custom_types.CombinableParameterType
    :param log_A: Log-amplitude parameter (log of initial value)
    :type log_A: custom_types.CombinableParameterType
    :param r: Growth rate parameter
    :type r: custom_types.CombinableParameterType
    :param kwargs: Additional arguments passed to parent class

    Mathematical Properties:
    - Linear in log-space: log(x) = log_A + r * t
    - Guaranteed positive output: x = exp(log_A + r * t) > 0
    - Growth rate r directly additive in log-space
    - Natural for multiplicative processes

    This transformation is particularly useful for:
    - Population modeling where values must be positive
    - Multiplicative growth processes
    - Log-scale regression models
    - Ensuring positive-valued outcomes

    The log-scale parameterization avoids issues with negative values
    and provides numerical stability for extreme growth rates.

    Example:
        >>> time_points = Constant([0, 1, 2, 3])
        >>> log_initial_pop = Normal(mu=4.6, sigma=0.1)  # log(100) ≈ 4.6
        >>> growth_rate = Normal(mu=0.1, sigma=0.02)
        >>> log_population = LogExponentialGrowth(t=time_points, log_A=log_initial_pop,
        >>>                                         r=growth_rate)
    """

    def __init__(
        self,
        *,
        t: "custom_types.CombinableParameterType",
        log_A: "custom_types.CombinableParameterType",
        r: "custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initialize log-exponential growth model.

        :param t: Time parameter
        :param log_A: Log-amplitude parameter
        :param r: Rate parameter
        :param kwargs: Additional arguments
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
        """Compute log-exponential growth: log_A + r * t.

        :param t: Time values
        :param log_A: Log-amplitude values
        :param r: Rate values

        :returns: Log-exponential growth values
        """
        return log_A + r * t

    # pylint: enable=arguments-differ

    def write_stan_operation(  # pylint: disable=arguments-differ
        self, t: str, log_A: str, r: str
    ) -> str:
        return f"{log_A} + {r} .* {t}"


class BinaryLogExponentialGrowth(TransformedParameter):
    """Binary log-exponential growth for two time points.

    Special case of log-exponential growth for two time points,
    reducing to: log(x) = log_A + r

    :param log_A: Log-amplitude parameter (log of initial value)
    :type log_A: custom_types.CombinableParameterType
    :param r: Growth rate parameter
    :type r: custom_types.CombinableParameterType
    :param kwargs: Additional arguments passed to parent class

    Mathematical Properties:
    - Simple additive model in log-space
    - r represents log-fold-change
    - Guaranteed positive output when exponentiated
    - Linear relationship in log-scale

    This transformation is ideal for:
    - Binary treatment effects in log-space
    - Fold-change modeling
    - Two-timepoint growth analysis
    - Log-scale before/after comparisons

    Example:
        >>> log_baseline = Normal(mu=4.6, sigma=0.1)  # log(100)
        >>> log_fold_change = Normal(mu=0, sigma=0.5)
        >>> log_final = BinaryLogExponentialGrowth(log_A=log_baseline, r=log_fold_change)
    """

    def __init__(
        self,
        log_A: "custom_types.CombinableParameterType",
        r: "custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initialize binary log-exponential growth.

        :param log_A: Log-amplitude parameter
        :param r: Rate parameter
        :param kwargs: Additional arguments
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
        """Compute binary log-exponential growth: log_A + r.

        :param log_A: Log-amplitude values
        :param r: Rate values

        :returns: log_A + r
        """
        return log_A + r

    def write_stan_operation(self, log_A: str, r: str) -> str:
        """Generate Stan code for binary log-exponential growth.

        :param log_A: Formatted log-amplitude parameter
        :type log_A: str
        :param r: Formatted rate parameter
        :type r: str

        :returns: Stan code for log_A + r
        :rtype: str
        """
        return f"{log_A} + {r}"


class SigmoidGrowth(SigmoidParameter):
    """Sigmoid growth model transformation.

    Implements sigmoid growth: x = A / (1 + exp(-r*(t - c)))
    This models S-shaped growth curves with carrying capacity A.

    :param t: Time parameter
    :type t: custom_types.CombinableParameterType
    :param A: Amplitude parameter (carrying capacity)
    :type A: custom_types.CombinableParameterType
    :param r: Growth rate parameter
    :type r: custom_types.CombinableParameterType
    :param c: Inflection point parameter (time of fastest growth)
    :type c: custom_types.CombinableParameterType
    :param kwargs: Additional arguments passed to parent class

    :cvar LOWER_BOUND: Values are bounded below by 0.0
    :cvar UPPER_BOUND: No upper bound (None) as A can be any positive value

    Mathematical Properties:
    - S-shaped growth curve
    - A represents carrying capacity (maximum value)
    - c represents inflection point (time of fastest growth)
    - r controls steepness of growth
    - Approaches 0 as t → -∞ and A as t → +∞

    This transformation is essential for:
    - Population growth with carrying capacity
    - Any growth process with saturation

    Example:
        >>> time_points = Constant(np.linspace(0, 10, 100))
        >>> carrying_capacity = Normal(mu=1000, sigma=50)
        >>> growth_rate = Normal(mu=1.0, sigma=0.1)
        >>> inflection_time = Normal(mu=5.0, sigma=0.5)
        >>> population = SigmoidGrowth(t=time_points, A=carrying_capacity,
        >>>                             r=growth_rate, c=inflection_time)
    """

    LOWER_BOUND: "custom_types.Float" = 0.0
    UPPER_BOUND: None = None

    def __init__(
        self,
        *,
        t: "custom_types.CombinableParameterType",
        A: "custom_types.CombinableParameterType",
        r: "custom_types.CombinableParameterType",
        c: "custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initialize sigmoid growth model.

        :param t: Time parameter
        :param A: Amplitude/carrying capacity parameter
        :param r: Rate parameter
        :param c: Inflection point parameter
        :param kwargs: Additional arguments
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
        """Compute sigmoid growth: A * sigmoid(r * (t - c)).

        :param t: Time values
        :param A: Amplitude values
        :param r: Rate values
        :param c: Inflection point values

        :returns: Sigmoid growth values
        """
        return A * SigmoidParameter.run_np_torch_op(self, r * (t - c))

    # pylint: enable=arguments-differ

    def write_stan_operation(  # pylint: disable=arguments-differ
        self, t: str, A: str, r: str, c: str
    ) -> str:
        par_string = super().write_stan_operation(f"{r} .* ({t} - {c})")
        return f"{A} .* {par_string}"


class LogSigmoidGrowth(LogSigmoidParameter):
    """Log-scale sigmoid growth model transformation.

    Implements the logarithm of sigmoid growth: log(x) = log_A + log_sigmoid(r*(t - c))
    This provides numerical stability and ensures positive values.

    :param t: Time parameter
    :type t: custom_types.CombinableParameterType
    :param log_A: Log-amplitude parameter (log of carrying capacity)
    :type log_A: custom_types.CombinableParameterType
    :param r: Growth rate parameter
    :type r: custom_types.CombinableParameterType
    :param c: Inflection point parameter
    :type c: custom_types.CombinableParameterType
    :param kwargs: Additional arguments passed to parent class

    :cvar LOWER_BOUND: No lower bound (None)
    :cvar UPPER_BOUND: No upper bound (None)

    This parameterization is ideal for:
    - Extreme parameter regimes
    - Log-scale statistical modeling
    - When initial conditions are naturally in log-space
    - Maximum numerical precision requirements

    Example:
        >>> time_points = Constant(np.linspace(0, 10, 100))
        >>> log_carrying_capacity = Normal(mu=6.9, sigma=0.1)  # log(1000)
        >>> growth_rate = Normal(mu=1.0, sigma=0.1)
        >>> inflection_time = Normal(mu=5.0, sigma=0.5)
        >>> log_population = LogSigmoidGrowth(t=time_points, log_A=log_carrying_capacity,
        >>>                                     r=growth_rate, c=inflection_time)
    """

    LOWER_BOUND: None = None
    UPPER_BOUND: None = None

    def __init__(
        self,
        *,
        t: "custom_types.CombinableParameterType",
        log_A: "custom_types.CombinableParameterType",
        r: "custom_types.CombinableParameterType",
        c: "custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initialize log-sigmoid growth model.

        :param t: Time parameter
        :param log_A: Log-amplitude parameter
        :param r: Rate parameter
        :param c: Inflection point parameter
        :param kwargs: Additional arguments
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
        """Compute log-sigmoid growth: log_A + log_sigmoid(r * (t - c)).

        :param t: Time values
        :param log_A: Log-amplitude values
        :param r: Rate values
        :param c: Inflection point values

        :returns: Log-sigmoid growth values
        """
        return log_A + LogSigmoidParameter.run_np_torch_op(self, r * (t - c))

    # pylint: enable=arguments-differ

    def write_stan_operation(  # pylint: disable=arguments-differ
        self, t: str, log_A: str, r: str, c: str
    ) -> str:
        par_string = super().write_stan_operation(f"{r} .* ({t} - {c})")
        return f"{log_A} + {par_string}"


class SigmoidGrowthInitParametrization(TransformedParameter):
    """Sigmoid growth with initial value parameterization.

    Alternative parameterization of sigmoid growth in terms of initial abundances
    rather than carrying capacity. Uses numerically stable log-add-exp computation.

    :param t: Time parameter
    :type t: custom_types.CombinableParameterType
    :param x0: Initial abundance parameter
    :type x0: custom_types.CombinableParameterType
    :param r: Growth rate parameter
    :type r: custom_types.CombinableParameterType
    :param c: Offset parameter (related to carrying capacity)
    :type c: custom_types.CombinableParameterType
    :param kwargs: Additional arguments passed to parent class

    :cvar LOWER_BOUND: Values are bounded below by 0.0
    :cvar UPPER_BOUND: No upper bound (None)

    Mathematical Properties:
    - Parameterizes sigmoid growth by initial value x0
    - Uses log-add-exp trick for numerical stability
    - Avoids direct computation of large exponentials
    - Maintains sigmoid growth dynamics

    This parameterization is useful when:
    - Initial conditions are better known than carrying capacity
    - Numerical stability is crucial
    - Working with extreme parameter values
    - Modeling relative growth from baseline

    The transformation uses the identity:
    x(t) = x0 * exp(log(1+exp(r*c)) - log(1+exp(r*(c-t))))
    """

    LOWER_BOUND: "custom_types.Float" = 0.0
    UPPER_BOUND: None = None

    def __init__(
        self,
        *,
        t: "custom_types.CombinableParameterType",
        x0: "custom_types.CombinableParameterType",
        r: "custom_types.CombinableParameterType",
        c: "custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initialize sigmoid growth with initial value parameterization.

        :param t: Time parameter
        :param x0: Initial abundance parameter
        :param r: Growth rate parameter
        :param c: Offset parameter
        :param kwargs: Additional arguments
        """
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
        x0: "custom_types.SampleType",
        r: "custom_types.SampleType",
        c: "custom_types.SampleType",
    ) -> npt.NDArray: ...

    def run_np_torch_op(self, t, x0, r, c):
        """Compute sigmoid growth with initial parameterization using log-add-exp.

        :param t: Time values
        :param x0: Initial abundance values
        :param r: Rate values
        :param c: Offset values

        :returns: Sigmoid growth values

        Uses log-add-exp for numerical stability in computing:
        x0 * exp(log(1+exp(r*c)) - log(1+exp(r*(c-t))))
        """
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
    """Log-scale sigmoid growth with initial value parameterization.

    Log-space version of sigmoid growth parameterized by initial values,
    providing numerical stability and guaranteed positive outputs.

    :param t: Time parameter
    :type t: custom_types.CombinableParameterType
    :param log_x0: Log of initial abundance parameter
    :type log_x0: custom_types.CombinableParameterType
    :param r: Growth rate parameter
    :type r: custom_types.CombinableParameterType
    :param c: Offset parameter
    :type c: custom_types.CombinableParameterType
    :param kwargs: Additional arguments passed to parent class

    :cvar LOWER_BOUND: No lower bound (None)
    :cvar UPPER_BOUND: No upper bound (None)

    Mathematical Properties:
    - Fully operates in log-space for numerical stability
    - Parameterized by log of initial conditions
    - Uses log-add-exp computations throughout
    - Maintains sigmoid growth dynamics

    This parameterization is ideal for:
    - Extreme parameter regimes
    - Log-scale statistical modeling
    - When initial conditions are naturally in log-space
    - Maximum numerical precision requirements

    Example:
        >>> time_points = Constant(np.linspace(0, 10, 100))
        >>> log_initial = Normal(mu=4.6, sigma=0.1)  # log(100)
        >>> growth_rate = Normal(mu=1.0, sigma=0.1)
        >>> offset = Normal(mu=5.0, sigma=0.5)
        >>> log_abundance = LogSigmoidGrowthInitParametrization(t=time_points,
        >>>                 log_x0=log_initial, r=growth_rate, c=offset)
    """

    LOWER_BOUND: None = None
    UPPER_BOUND: None = None

    def __init__(
        self,
        *,
        t: "custom_types.CombinableParameterType",
        log_x0: "custom_types.CombinableParameterType",
        r: "custom_types.CombinableParameterType",
        c: "custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initialize log-sigmoid growth with initial value parameterization.

        :param t: Time parameter
        :param log_x0: Log of initial abundance parameter
        :param r: Growth rate parameter
        :param c: Offset parameter
        :param kwargs: Additional arguments
        """
        super().__init__(t=t, log_x0=log_x0, r=r, c=c, **kwargs)

    # pylint: disable=arguments-renamed, arguments-differ
    @overload
    def run_np_torch_op(
        self,
        t: torch.Tensor,
        log_x0: torch.Tensor,
        r: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor: ...

    @overload
    def run_np_torch_op(
        self,
        t: "custom_types.SampleType",
        log_x0: "custom_types.SampleType",
        r: "custom_types.SampleType",
        c: "custom_types.SampleType",
    ) -> npt.NDArray: ...

    def run_np_torch_op(self, t, log_x0, r, c):
        """Compute log-sigmoid growth with initial parameterization.

        :param t: Time values
        :param log_x0: Log initial abundance values
        :param r: Rate values
        :param c: Offset values

        :returns: Log-sigmoid growth values

        Computes: log_x0 + log(1+exp(r*c)) - log(1+exp(r*(c-t)))
        using numerically stable log-add-exp operations.
        """
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
    """Sequence convolution transformation using weight matrices.

    Performs convolution operation on ordinally-encoded sequences using provided
    weight matrices. This is commonly used for sequence modeling and pattern
    recognition in biological sequences or text data.

    :param weights: Weight matrix for convolution (at least 2D)
    :type weights: custom_types.CombinableParameterType
    :param ordinals: Ordinally-encoded sequence array (at least 1D)
    :type ordinals: custom_types.CombinableParameterType
    :param kwargs: Additional arguments passed to parent class

    :cvar SHAPE_CHECK: Disabled for complex shape operations (False)
    :cvar FORCE_LOOP_RESET: Forces loop reset in Stan code (True)
    :cvar FORCE_PARENT_NAME: Forces parent naming in Stan code (True)

    Shape Requirements:
    - Weights: (..., kernel_size, alphabet_size)
    - Ordinals: (..., sequence_length)
    - Output: (..., sequence_length - kernel_size + 1)

    The transformation applies convolution by:
    1. Sliding a kernel of size kernel_size over the sequence
    2. Using ordinal values to index into the weight matrix
    3. Summing weighted values for each position

    This is commonly used for:
    - DNA/RNA sequence analysis
    - Protein sequence modeling
    - Text processing with character-level models
    - Pattern recognition in discrete sequences

    Example:
        >>> # DNA sequence convolution
        >>> weights = Normal(mu=0, sigma=1, shape=(motif_length, 4))  # 4 nucleotides
        >>> dna_sequence = Constant(encoded_dna)  # 0,1,2,3 for A,C,G,T
        >>> motif_scores = ConvolveSequence(weights=weights, ordinals=dna_sequence)
    """

    SHAPE_CHECK = False
    FORCE_LOOP_RESET = True
    FORCE_PARENT_NAME = True

    def __init__(
        self,
        *,
        weights: "custom_types.CombinableParameterType",
        ordinals: "custom_types.CombinableParameterType",
        **kwargs,
    ):
        """Initialize sequence convolution with shape validation.

        :param weights: Weight matrix (kernel_size × alphabet_size in last 2 dims)
        :param ordinals: Ordinal sequence array
        :param kwargs: Additional arguments

        :raises ValueError: If weights is not at least 2D
        :raises ValueError: If ordinals is not at least 1D
        :raises ValueError: If shapes are incompatible for broadcasting
        """
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

        # If numpy, loop over the leading dimension
        assert weights.shape == self.weights.shape
        assert ordinals.shape == self.ordinals.shape

        # Decide on the module for the operation
        module = utils.choose_module(weights)

        # Determine the number of dimensions to prepend to each array
        weights_n_prepends = len(self.shape[:-1]) - len(self.weights.shape[:-2])
        ordinal_n_prepends = len(self.shape[:-1]) - len(self.ordinals.shape[:-1])

        # Get the padded shapes. This is just aligning the shapes for broadcasting.
        padded_weights_shape = (None,) * weights_n_prepends + self.weights.shape[:-2]
        padded_ordinals_shape = (None,) * ordinal_n_prepends + self.ordinals.shape[:-1]
        assert len(padded_weights_shape) == len(padded_ordinals_shape)

        # Set output array and build a set of filter indices
        output_arr = module.full(self.shape, np.nan)
        filter_indices = module.arange(self.kernel_size)

        # If torch, send arrays to appropriate device
        if module is torch:
            filter_indices = filter_indices.to(weights.device)
            output_arr = output_arr.to(weights.device)

        # Loop over the different weights
        for weights_inds in np.ndindex(weights.shape[:-2]):

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
            weights_matrix = weights[weights_inds]
            ordinal_matrix = ordinals[ordinal_inds]
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

    def get_index_offset(
        self,
        query: Union[str, "abstract_model_component.AbstractModelComponent"],
        offset_adjustment: int = 0,
    ) -> int:
        """Calculate index offset with special handling for weights.

        :param query: Component or parameter name to query
        :param offset_adjustment: Base offset adjustment

        :returns: Index offset (adjusted +1 for weights parameter)
        :rtype: int

        The weights parameter requires special offset handling because
        its last two dimensions are used directly in the convolution
        rather than being broadcast.
        """
        # Run the inherited method
        offset = super().get_index_offset(query, offset_adjustment)

        # Adjust if needed
        if query == "weights" or query is self.weights:
            offset += 1

        return offset

    def get_right_side(
        self,
        index_opts: tuple[str, ...] | None,
        start_dims: dict[str, "custom_types.Integer"] | None = None,
        end_dims: dict[str, "custom_types.Integer"] | None = None,
        offset_adjustment: int = 0,
    ) -> str:
        """Generate right-side code with proper dimension handling.

        Sets default end_dims to exclude the last weight dimension (-2),
        ensuring both kernel_size and alphabet_size dimensions are preserved.
        """
        end_dims = end_dims or {"weights": -2}

        # Run the AbstractModelParameter version of the method to get each model
        # component formatted appropriately. Note that we ignore the last dimension
        # of the weights. This is because we need both dimensions in the Stan code.
        return super().get_right_side(
            index_opts, end_dims=end_dims, offset_adjustment=offset_adjustment
        )

    def write_stan_operation(  # pylint: disable=arguments-differ
        self, weights: str, ordinals: str
    ) -> str:
        """Generate Stan convolution function call.

        :param weights: Formatted weights parameter name
        :type weights: str
        :param ordinals: Formatted ordinals parameter name
        :type ordinals: str

        :returns: Stan function call for sequence convolution
        :rtype: str
        """
        # This runs a custom function
        return f"convolve_sequence({weights}, {ordinals})"

    def get_supporting_functions(self) -> list[str]:
        """Return required Stan function includes.

        :returns: List including pssm.stanfunctions for convolution support
        :rtype: list[str]
        """
        return super().get_supporting_functions() + ["#include pssm.stanfunctions"]


class IndexParameter(TransformedParameter):
    """Array indexing transformation with NumPy-compatible semantics.

    Creates indexed subsets of parameters using slicing, scalar indexing,
    and array indexing. Follows NumPy indexing conventions rather than
    Stan conventions for consistency with Python data manipulation.

    :param dist: Parameter to index
    :type dist: custom_types.CombinableParameterType
    :param indices: Indexing specifications (slices, integers, arrays)
    :type indices: custom_types.IndexType

    :cvar SHAPE_CHECK: Disabled for complex indexing operations (False)
    :cvar FORCE_PARENT_NAME: Forces parent naming in Stan code (True)

    Supported Index Types:
    - **slice**: Standard Python slicing with start:stop (step=1 only)
    - **int**: Single element selection
    - **np.ndarray**: Advanced indexing with integer arrays. 1D only. Follows numpy convention.
    - **Ellipsis**: Automatic dimension filling
    - **None**: New axis insertion

    Important Differences from Stan:
    - Uses 0-based indexing (Python convention)
    - Advanced indexing follows NumPy broadcasting rules
    - Negative indices are supported and converted appropriately

    Example:
        >>> data = Normal(mu=0, sigma=1, shape=(10, 5))
        >>> # Slice first 5 rows
        >>> subset = IndexParameter(data, slice(0, 5))
        >>> # Select specific elements
        >>> selected = IndexParameter(data, [0, 2, 4], [1, 3, 0])  # NumPy-style
    """

    SHAPE_CHECK = False
    FORCE_PARENT_NAME = True

    def __init__(
        self,
        dist: "custom_types.CombinableParameterType",
        *indices: "custom_types.IndexType",
    ):
        """Initialize indexing transformation.

        :param dist: Parameter to index
        :param indices: Indexing specifications

        The initialization processes all index types, converts negative
        indices to positive, validates array dimensions, and creates
        appropriate constant parameters for array indices.
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
        """Convert negative indices to positive indices.

        :param neg_ind: Negative index or array of indices
        :param dim: Dimension size for conversion

        :returns: Positive indices

        :raises ValueError: If indices are out of bounds

        Handles both scalar and array indices, performing bounds checking
        and conversion from Python's negative indexing convention.
        """
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
        """Process and validate all indexing specifications.

        :param indices: Raw indexing specifications

        :returns: Tuple of (output_shape, processed_indices, constant_parents)

        This method handles the complex logic of:
        - Processing different index types (slices, integers, arrays, ellipsis)
        - Calculating output shapes
        - Converting to Stan-compatible 1-based indexing
        - Creating constant parameters for array indices
        - Validating consistency across multiple array indices
        """

        def process_ellipsis() -> "custom_types.Integer":
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

        def process_array() -> "custom_types.Integer":
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
        # If torch, numpy arrays must go to torch
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
            inds = self._python_indices

        # Index and check shape
        indexed = dist[inds]
        assert indexed.shape == self.shape

        return indexed

    def get_right_side(
        self,
        index_opts: tuple[str, ...] | None,
        start_dims: dict[str, "custom_types.Integer"] | None = None,
        end_dims: dict[str, "custom_types.Integer"] | None = None,
        offset_adjustment: int = 0,
    ) -> str:
        """Generate Stan indexing code.

        :returns: Stan indexing expression
        :rtype: str

        Gets the name of the variable that is being indexed, then passes it to
        the `write_stan_operation` method to get the full Stan code for the transformation
        """
        return self.write_stan_operation(dist=self.dist.model_varname)

    def write_stan_operation(  # pylint: disable=arguments-differ
        self, dist: str
    ) -> str:
        """Generate complete Stan indexing expression.

        :param dist: Variable name to index
        :type dist: str

        :returns: Stan indexing expression with bracket notation
        :rtype: str

        Handles complex indexing patterns including:
        - Multiple array indices (creates separate bracket groups)
        - Mixed slicing and array indexing
        - Proper 1-based index conversion for Stan
        - Colon notation for full dimension slicing
        """
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

    def get_transformation_assignment(  # pylint: disable=unused-argument
        self, index_opts: tuple[str, ...] | None
    ) -> str:
        """Generate transformation assignment without index options.

        :param index_opts: Index options (ignored for direct assignment)

        :returns: Stan assignment statement
        :rtype: str

        Indexing parameters are assigned directly without loop indexing
        since they represent specific element selection rather than
        computed transformations.
        """
        # pylint: disable=no-value-for-parameter
        return super().get_transformation_assignment(None)  # Assigned directly

    # The definition depth is always 0 for this transformation
    def get_assign_depth(self) -> "custom_types.Integer":  # pylint: disable=C0116
        return 0

    @property
    def force_name(self) -> bool:
        """Force explicit naming for indexed parameters.

        :returns: Always True
        :rtype: bool

        Indexed parameters must be explicitly named in Stan code to
        enable proper variable reference and assignment.
        """
        return True
