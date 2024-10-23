"""Holds code for working with constant values in a DMS Stan model."""

from typing import Union

import numpy as np
import numpy.typing as npt

import dms_stan.model.components as dms_components


class Constant(dms_components.abstract_classes.AbstractModelComponent):
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
            shape = value.shape

        # Otherwise, convert to a numpy array
        else:
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
