"""
Defines the `Constant` class for wrapping values that are intended to stay constant
in the Stan model.
"""

from typing import Union

import numpy as np
import numpy.typing as npt


class Constant:
    """
    This class is used to wrap values that are intended to stay constant in the
    Stan model. This is effectively a wrapper around the value that forwards all
    mathematical operations and attribute access to the value. Note that because
    it is a wrapper, in place operations made to the value will be reflected in
    the instance of this class.
    """

    def __init__(self, value: Union[int, float, npt.NDArray]):
        """
        Wraps the value in a Constant instance. Any numerical type is legal.
        """
        # Assign the value
        self.value: npt.NDArray = np.array(value)

    def __getattr__(self, name):
        return getattr(self.value, name)

    def __repr__(self):
        return f"Constant({self.value.__repr__()})"

    def __getitem__(self, key):
        return self.value[key]

    # Mathematical operations are forwarded to the value
    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return other + self.value

    def __sub__(self, other):
        return self.value - other

    def __rsub__(self, other):
        return other - self.value

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    def __truediv__(self, other):
        return self.value / other

    def __rtruediv__(self, other):
        return other / self.value

    def __floordiv__(self, other):
        return self.value // other

    def __rflooriv__(self, other):
        return other // self.value

    def __mod__(self, other):
        return self.value % other

    def __rmod__(self, other):
        return other % self.value

    def __pow__(self, other):
        return self.value**other

    def __rpow__(self, other):
        return other**self.value

    def __matmul__(self, other):
        return self.value @ other

    def __rmatmul__(self, other):
        return other @ self.value
