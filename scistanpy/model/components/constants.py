"""Holds code for working with constant values in a SciStanPy model."""

from __future__ import annotations

from typing import Literal, Optional, TYPE_CHECKING, Union

import numpy as np
import numpy.typing as npt
import torch

from scistanpy.model.components import abstract_model_component

if TYPE_CHECKING:
    from scistanpy import custom_types


class Constant(abstract_model_component.AbstractModelComponent):
    """
    Abstract class for components that pass through the values of their children.
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
        """
        Wraps the value in a Constant instance. Any numerical type is legal.
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

    # We need a draw method
    def _draw(
        self,
        level_draws: dict[str, Union[npt.NDArray, "custom_types.Float"]],
        seed: Optional["custom_types.Integer"],
    ) -> Union[npt.NDArray, "custom_types.Float", "custom_types.Integer"]:
        """Draw values for this component."""
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
        """Return the Stan code for this component (there is none)."""
        return ""

    def _get_lim(
        self, lim_type: Literal["low", "high"]
    ) -> tuple["custom_types.Float", "custom_types.Float"]:
        """Order of magnitude of the value."""
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
        return f"{self.model_varname} = {self.value}"

    @property
    def slider_start(self) -> "custom_types.Float":
        """Starting values for sliders in prior predictive checks."""
        # Lower bound if we have one
        if self.LOWER_BOUND is not None:
            return self.LOWER_BOUND

        # Otherwise, retrieve lower limit
        return self._get_lim("low")[0]

    @property
    def slider_end(self) -> "custom_types.Float":
        """Ending values for sliders in prior predictive checks."""
        # Upper bound if we have one
        if self.UPPER_BOUND is not None:
            return self.UPPER_BOUND

        # Otherwise, retrieve upper limit
        return self._get_lim("high")[0]

    @property
    def slider_step_size(self) -> "custom_types.Float":
        """We allow 100 steps between the start and end values"""
        # Get the order of magnitude of the value. We want to round to 2 orders
        return (self.slider_end - self.slider_start) / 100

    @property
    def torch_parametrization(self) -> torch.Tensor:
        return self._torch_parametrization

    @property
    def enforce_uniformity(self) -> bool:
        """
        Whether to enforce uniformity in the value. If True, makes sure that value
        is indeed a single value.
        """
        if self._enforce_uniformity and np.unique(self.value).size > 1:
            raise ValueError(
                "If enforcing uniformity, the value must be a single value."
            )
        return self._enforce_uniformity

    @enforce_uniformity.setter
    def enforce_uniformity(self, value: bool) -> None:
        """
        Checks to be sure that the values are uniform, then updates the private
        attribute `_enforce_uniformity` if they are.
        """
        if value and np.unique(self.value).size > 1:
            raise ValueError(
                "If enforcing uniformity, the value must be a single value."
            )
        self._enforce_uniformity = value
