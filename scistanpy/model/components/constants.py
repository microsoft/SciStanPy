# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Constant value components for SciStanPy models.

This module provides the Constant class for representing fixed values in SciStanPy
models. Constants serve as the foundational building blocks for model hierarchies,
providing fixed hyperparameters, unmodeled data values, and other numerical
components that don't change during inference.

The Constant class integrates with SciStanPy's model component hierarchy while
providing specialized functionality for fixed-value components. It is particularly
important for interactive model manipulation during prior predictive checks.

**Basic Usage:**

.. code-block:: python

    import scistanpy as ssp
    import numpy as np

    # Scalar constants
    learning_rate = ssp.constants.Constant(0.01)
    n_samples = ssp.constants.Constant(100)

    # Array constants
    design_matrix = ssp.constants.Constant(X_data)
    time_points = ssp.constants.Constant(np.linspace(0, 10, 11))

    # Constants with bounds
    probability = ssp.constants.Constant(0.5, lower_bound=0.0, upper_bound=1.0)

    # Force constant to NOT be modifiable in interactive contexts
    temperature = ssp.constants.Constant(
        300.0,
        lower_bound=250.0,
        upper_bound=350.0,
        togglable=False
    )
"""

from __future__ import annotations

from typing import Literal, Optional, TYPE_CHECKING, Union

import numpy as np
import numpy.typing as npt
import torch

from scistanpy.model.components import abstract_model_component

if TYPE_CHECKING:
    from scistanpy import custom_types


class Constant(abstract_model_component.AbstractModelComponent):
    """Represents a constant value component in SciStanPy models.

    This class wraps fixed numerical values to integrate them into the SciStanPy
    model component hierarchy. Constants provide the foundation for model
    construction by supplying fixed hyperparameters, unmodeled data, and other
    numerical values that remain unchanged during inference.

    :param value: The constant value to wrap
    :type value: Union[custom_types.Integer, custom_types.Float, npt.NDArray, np.integer,
        np.floating]
    :param lower_bound: Optional lower bound for value validation. Defaults to None.
    :type lower_bound: Optional[custom_types.Float]
    :param upper_bound: Optional upper bound for value validation. Defaults to None.
    :type upper_bound: Optional[custom_types.Float]
    :param togglable: Whether value can be toggled in interactive interfaces. Auto-detected
        if None. Defaults to None.
    :type togglable: Optional[bool]
    :param enforce_uniformity: Whether to require all array elements to be identical.
        Defaults to False.
    :type enforce_uniformity: bool
    :param kwargs: Additional keyword arguments passed to parent class

    :ivar value: The stored constant value as a NumPy array
    :ivar BASE_STAN_DTYPE: Stan data type ("real" or "int") inferred from value
    :ivar LOWER_BOUND: Lower bound constraint (if specified)
    :ivar UPPER_BOUND: Upper bound constraint (if specified)
    :ivar is_togglable: Whether the constant can be modified in interactive contexts

    :raises ValueError: If value violates specified bounds
    :raises ValueError: If enforce_uniformity=True but array has non-uniform values

    Key Features:
        - **Automatic Type Inference**: Determines appropriate Stan data types
        - **Bound Checking**: Validates values against optional constraints
        - **Interactive Support**: Configures sliders for model exploration

    The class automatically handles:
        - Conversion of Python scalars to NumPy arrays
        - Shape inference from array inputs
        - Data type detection for Stan code generation
        - Bound validation at initialization

    .. hint::
        Under the hood, the :py:meth:`Model.prior_predictive()
        <scistanpy.model.model.Model.prior_predictive>` method parses the model
        components to identify constants. Any identified constant with ``is_togglable=True``
        is represented as an interactive slider in the generated interface. Updating
        slider values updates the underlying constant value used in the model.

    .. important::
        By default, constants with floating-point values are considered ``togglable=True``,
        allowing interactive modification. Integer-valued constants are ``togglable=False``
        by default to prevent invalid states. As a result, it is important to specify
        true data types when defining constants to ensure desired behavior. For example,

        >>> n = ssp.Constant(100)  # n is not togglable
        >>> n = ssp.Constant(100.0)  # n is togglable

        This is also important when defining parameters, as raw Python integers and
        floats are converted to constants interally. For example,

        >>> mean = ssp.Normal(mu = 0, sigma = 1)  # Neither mu nor sigma are togglable
        >>> mean = ssp.Normal(mu = 0.0, sigma = 1.0)  # Both mu and sigma are togglable
    """

    def __init__(
        self,
        value: Union[
            "custom_types.Integer",
            "custom_types.Float",
            npt.NDArray,
            np.integer,
            np.floating,
        ],
        *,
        lower_bound: Optional["custom_types.Float"] = None,
        upper_bound: Optional["custom_types.Float"] = None,
        togglable: Optional[bool] = None,
        enforce_uniformity: bool = False,
        **kwargs,
    ):
        """Initialize a constant value component with validation and type inference.

        The initialization process:
            1. Converts input values to NumPy arrays
            2. Infers appropriate Stan data types
            3. Validates bounds if specified
            4. Sets up PyTorch tensor representations
            5. Configures interactive properties
        """
        # If the value is a numpy array, get the shape
        if isinstance(value, np.ndarray):
            self.BASE_STAN_DTYPE = (  # pylint: disable=invalid-name
                "real" if np.issubdtype(value.dtype, np.floating) else "int"
            )
            kwargs["shape"] = value.shape

        # Otherwise, convert to a numpy array
        else:
            self.BASE_STAN_DTYPE = "real" if isinstance(value, float) else "int"
            value = np.array(value, dtype=type(value))

        # Check bounds if provided
        if lower_bound is not None and (minimum := value.min()) < lower_bound:
            raise ValueError(
                f"Value {minimum.item()} is less than lower bound {lower_bound}."
            )
        if upper_bound is not None and (maximum := value.max()) > upper_bound:
            raise ValueError(
                f"Value {maximum.item()} is greater than upper bound {upper_bound}."
            )

        # If enforcing uniformity, there can only be one value. Every time this
        # attribute is checked, it will be checked that the value is a single value.
        self._enforce_uniformity = enforce_uniformity
        self.enforce_uniformity  # Runs check, pylint: disable=pointless-statement

        # Set upper and lower bounds
        self.LOWER_BOUND = lower_bound  # pylint: disable=invalid-name
        self.UPPER_BOUND = upper_bound  # pylint: disable=invalid-name

        # Set the value
        self.value = value
        self._torch_parametrization = torch.from_numpy(value)

        # Set whether the value is togglable. By default, it is for floats and
        # is not for integers.
        self.is_togglable = (
            (self.BASE_STAN_DTYPE == "real") if togglable is None else togglable
        )

        # Initialize the parent class
        super().__init__(**kwargs)

    def _draw(
        self,
        level_draws: dict[str, Union[npt.NDArray, "custom_types.Float"]],
        seed: Optional["custom_types.Integer"],
    ) -> Union[npt.NDArray, "custom_types.Float", "custom_types.Integer"]:
        """Draw values for this constant component (returns the fixed value).

        :param level_draws: Parent component draws (should be empty for constants)
        :type level_draws: dict[str, Union[npt.NDArray, custom_types.Float]]
        :param seed: Random seed (unused for constants)
        :type seed: Optional[custom_types.Integer]

        :returns: The constant value
        :rtype: Union[npt.NDArray, custom_types.Float, custom_types.Integer]

        :raises AssertionError: If level_draws is not empty (constants have no parents)

        Since constants represent fixed values, this method simply returns
        the stored value without any random sampling.
        """
        # Level draws should be empty
        assert not level_draws

        # Just return the value
        return self.value

    def get_right_side(
        self,
        index_opts: tuple[str, ...] | None,
        start_dims: dict[str, "custom_types.Integer"] | None = None,
        end_dims: dict[str, "custom_types.Integer"] | None = None,
        offset_adjustment: int = 0,
    ) -> str:
        """Return Stan code for right-hand side (empty for constants).

        :param index_opts: Indexing options (unused for constants)
        :type index_opts: Optional[tuple[str, ...]]
        :param start_dims: Starting dimensions (unused for constants)
        :type start_dims: Optional[dict[str, custom_types.Integer]]
        :param end_dims: Ending dimensions (unused for constants)
        :type end_dims: Optional[dict[str, custom_types.Integer]]
        :param offset_adjustment: Index offset (unused for constants)
        :type offset_adjustment: int

        :returns: Empty string (constants don't have right-hand side expressions)
        :rtype: str

        There is no right-hand side expression for constants, as they represent
        fixed values rather than derived quantities. Thus, this method returns
        an empty string.
        """
        return ""

    def _get_lim(
        self, lim_type: Literal["low", "high"]
    ) -> tuple["custom_types.Float", "custom_types.Float"]:
        """Calculate appropriate limits for interactive sliders.

        :param lim_type: Type of limit to calculate ("low" or "high")
        :type lim_type: Literal["low", "high"]

        :returns: Tuple of (limit_value, order_of_magnitude)
        :rtype: tuple[custom_types.Float, custom_types.Float]

        :raises ValueError: If lim_type is not "low" or "high"

        This method calculates reasonable bounds for interactive sliders
        based on the order of magnitude of the constant value. It provides
        a range that allows meaningful exploration around the current value.
        """
        # Get the largest absolute value
        maxval = np.abs(self.value).max()

        # If the value is zero, order of magnitude is 0
        if maxval == 0:
            order = 0

        # Otherwise, calculate the order of magnitude
        else:
            order = np.log10(maxval)
            order = np.floor(order) if order < 0 else np.ceil(order)

        # Minimum order is 0
        order = max(0, order)

        # Add or subtract one order of magnitude to get the limit
        if lim_type == "high":
            return maxval + 10**order, order
        elif lim_type == "low":
            return maxval - 10**order, order
        else:
            raise ValueError("Invalid limit type.")

    def __str__(self) -> str:
        """Return human-readable string representation.

        :returns: String showing constant assignment
        :rtype: str

        Creates a readable representation showing the constant name and value,
        useful for model inspection and debugging.
        """
        return f"{self.model_varname} = {self.value}"

    @property
    def slider_start(self) -> "custom_types.Float":
        """Get starting value for interactive sliders when constant is used for
        interactive prior predictive checks.

        :returns: Appropriate starting value for sliders
        :rtype: custom_types.Float

        Returns the lower bound if specified, otherwise calculates a reasonable
        lower limit based on the value's order of magnitude.
        """
        # Lower bound if we have one
        if self.LOWER_BOUND is not None:
            return self.LOWER_BOUND

        # Otherwise, retrieve lower limit
        return self._get_lim("low")[0]

    @property
    def slider_end(self) -> "custom_types.Float":
        """Get ending value for interactive sliders when constant is used for
        interactive prior predictive checks.

        :returns: Appropriate ending value for sliders
        :rtype: custom_types.Float

        Returns the upper bound if specified, otherwise calculates a reasonable
        upper limit based on the value's order of magnitude.
        """
        # Upper bound if we have one
        if self.UPPER_BOUND is not None:
            return self.UPPER_BOUND

        # Otherwise, retrieve upper limit
        return self._get_lim("high")[0]

    @property
    def slider_step_size(self) -> "custom_types.Float":
        """Get step size for interactive sliders when constant is used for
        interactive prior predictive checks.

        :returns: Appropriate step size for sliders
        :rtype: custom_types.Float

        Calculates a step size that provides approximately 100 steps between
        the slider start and end values, enabling fine-grained control.
        """
        # We allow 100 steps between the start and end values
        return (self.slider_end - self.slider_start) / 100

    @property
    def torch_parametrization(self) -> torch.Tensor:
        """Get PyTorch tensor representation of the constant.

        :returns: PyTorch tensor containing the constant value
        :rtype: torch.Tensor

        Provides a tensor representation for PyTorch-based computations,
        enabling integration with gradient-based operations.
        """
        return self._torch_parametrization

    @property
    def enforce_uniformity(self) -> bool:
        """Check if uniformity enforcement is enabled.

        :returns: True if uniformity is enforced
        :rtype: bool

        :raises ValueError: If uniformity is enforced but value is not uniform

        When enabled, this property ensures that all elements of array-valued
        constants have the same value, which is useful for symmetric priors
        and other modeling contexts requiring uniform parameters.
        """
        if self._enforce_uniformity and np.unique(self.value).size > 1:
            raise ValueError(
                "If enforcing uniformity, the value must be a single value."
            )
        return self._enforce_uniformity

    @enforce_uniformity.setter
    def enforce_uniformity(self, value: bool) -> None:
        """Set uniformity enforcement with validation.

        :param value: Whether to enforce uniformity
        :type value: bool

        :raises ValueError: If enabling uniformity but current value is not uniform

        When enabling uniformity enforcement, validates that the current value
        satisfies the uniformity constraint before updating the setting.
        """
        if value and np.unique(self.value).size > 1:
            raise ValueError(
                "If enforcing uniformity, the value must be a single value."
            )
        self._enforce_uniformity = value
