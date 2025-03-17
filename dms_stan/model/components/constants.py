"""Holds code for working with constant values in a DMS Stan model."""

from typing import Literal, Optional, Union

import numpy as np
import numpy.typing as npt
import torch

from .abstract_model_component import AbstractModelComponent


class Constant(AbstractModelComponent):
    """
    Abstract class for components that pass through the values of their children.
    """

    def __init__(
        self,
        value: Union[int, float, npt.NDArray, np.integer, np.floating],
        *,
        shape: tuple[int, ...] = (),
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        togglable: Optional[bool] = None,
        enforce_uniformity: bool = False,
    ):
        """
        Wraps the value in a Constant instance. Any numerical type is legal.
        """
        # If the value is a numpy array, get the shape
        if isinstance(value, np.ndarray):
            self.BASE_STAN_DTYPE = (  # pylint: disable=invalid-name
                "real" if np.issubdtype(value.dtype, np.floating) else "int"
            )
            shape = value.shape

        # Otherwise, convert to a numpy array
        else:
            self.BASE_STAN_DTYPE = "real" if isinstance(value, float) else "int"
            value = np.array(value, dtype=type(value))

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
        super().__init__(shape=shape)

    # We need a draw method
    def _draw(
        self, n: int, level_draws: dict[str, npt.NDArray], seed: Optional[int]
    ) -> npt.NDArray:
        """Draw values for this component."""
        # Level draws should be empty
        assert not level_draws

        # Repeat the value n times
        return np.broadcast_to(self.value[None], (n,) + self.value.shape)

    def _handle_transformation_code(
        self, param: AbstractModelComponent, index_opts: tuple[str, ...]
    ):
        """
        Handle the transformation code for this component. This function should
        never be called.
        """
        raise AssertionError("This function should never be called.")

    def get_stan_code(self, index_opts: tuple[str, ...]) -> str:
        """Return the Stan code for this component (there is none)."""
        return ""

    def get_target_incrementation(self, index_opts: tuple[str, ...]) -> str:
        """Constants cannot increment the target variable."""
        return ""

    def get_transformation_assignment(self, index_opts: tuple[str, ...]) -> str:
        """Constants are never transformed."""
        return ""

    def _get_lim(self, lim_type: Literal["low", "high"]) -> tuple[float, float]:
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

    @property
    def slider_start(self) -> float:
        """Starting values for sliders in prior predictive checks."""
        # Lower bound if we have one
        if self.LOWER_BOUND is not None:
            return self.LOWER_BOUND

        # Otherwise, retrieve lower limit
        return self._get_lim("low")[0]

    @property
    def slider_end(self) -> float:
        """Ending values for sliders in prior predictive checks."""
        # Upper bound if we have one
        if self.UPPER_BOUND is not None:
            return self.UPPER_BOUND

        # Otherwise, retrieve upper limit
        return self._get_lim("high")[0]

    @property
    def slider_step_size(self) -> float:
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
