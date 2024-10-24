"""Holds code for working with constant values in a DMS Stan model."""

from typing import Union

import numpy as np
import numpy.typing as npt

from .abstract_classes import AbstractModelComponent


class Constant(AbstractModelComponent):
    """
    Abstract class for components that pass through the values of their children.
    """

    def __init__(
        self, *, shape: tuple[int, ...] = (), value: Union[int, float, npt.NDArray]
    ):
        """
        Wraps the value in a Constant instance. Any numerical type is legal.
        """
        # If the value is a numpy array, get the shape
        if isinstance(value, np.ndarray):
            self.base_stan_dtype = (
                "real" if isinstance(value.dtype, np.floating) else "int"
            )
            shape = value.shape

        # Otherwise, convert to a numpy array
        else:
            self.base_stan_dtype = "real" if isinstance(value, float) else "int"
            value = np.array(value, shape=shape, dtype=type(value))

        # Initialize the parent class
        super().__init__(shape=shape, value=value)

    # Override the _set_parents method to just set an empty dictionary
    def _set_parents(self):
        """Set the parents of this component."""
        self._parents = {}

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
