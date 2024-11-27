"""Holds code for working with constant values in a DMS Stan model."""

from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from .abstract_model_component import AbstractModelComponent


class Constant(AbstractModelComponent):
    """
    Abstract class for components that pass through the values of their children.
    """

    def __init__(
        self,
        *,
        shape: tuple[int, ...] = (),
        value: Union[int, float, npt.NDArray],
        stan_lower_bound: Optional[float] = None,
        stan_upper_bound: Optional[float] = None
    ):
        """
        Wraps the value in a Constant instance. Any numerical type is legal.
        """
        # If the value is a numpy array, get the shape
        if isinstance(value, np.ndarray):
            self.BASE_STAN_DTYPE = (  # pylint: disable=invalid-name
                "real" if isinstance(value.dtype, np.floating) else "int"
            )
            shape = value.shape

        # Otherwise, convert to a numpy array
        else:
            self.BASE_STAN_DTYPE = "real" if isinstance(value, float) else "int"
            value = np.array(value, dtype=type(value))

        # Set upper and lower bounds
        self.STAN_LOWER_BOUND = stan_lower_bound  # pylint: disable=invalid-name
        self.STAN_UPPER_BOUND = stan_upper_bound  # pylint: disable=invalid-name

        # Set the value
        self.value = value

        # Set whether the value is togglable. By default, it is for floats and
        # is not for integers.
        self.is_togglable = self.BASE_STAN_DTYPE == "real"

        # Initialize the parent class
        super().__init__(shape=shape)

    # We need a draw method
    def _draw(self, n: int, level_draws: dict[str, npt.NDArray]) -> npt.NDArray:
        """Draw values for this component."""
        # Level draws should be empty
        assert not level_draws

        # Repeat the value n times
        return np.repeat(self.value[None], n, axis=0)

    def _handle_transformation_code(
        self, param: AbstractModelComponent, index_opts: tuple[str, ...]
    ):
        """
        Handle the transformation code for this component. This function should
        never be called.
        """
        raise AssertionError("This function should never be called.")

    def format_stan_code(self, **to_format: str) -> str:
        """There is no transformation code to format."""
        assert len(to_format) == 0
        return ""

    def get_target_incrementation(self, index_opts: tuple[str, ...]) -> str:
        """Constants cannot increment the target variable."""
        return ""

    def get_transformation_assignment(self, index_opts: tuple[str, ...]) -> str:
        """Constants are never transformed."""
        return ""
