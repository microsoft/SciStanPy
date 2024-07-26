"""
Defines the `Constant` class for wrapping values that are intended to stay constant
in the Stan model.
"""

import numpy.typing as npt


class Constant:
    """
    This class is used to wrap values that are intended to stay constant in the
    Stan model. This is effectively a wrapper around the value that forwards all
    mathematical operations and attribute access to the value. Note that because
    it is a wrapper, in place operations made to the value will be reflected in
    the instance of this class.
    """

    def __init__(self, value: [int, float, npt.NDArray]):
        """
        Wraps the value in a Constant instance. Any numerical type is legal.
        """
        # Assign the value
        self.value = value

        # All mathematical operations are forwarded to the value
        for op in ("add", "sub", "mul", "truediv", "floordiv", "mod", "pow", "matmul"):
            for prefix in ("", "r"):

                # If the attribute does not exist for the value, continue
                if not hasattr(self.value, operation := f"__{prefix}{op}__"):
                    continue

                # Set the method bound to the value as the method of this instance
                setattr(self, operation, getattr(self.value, operation))

    def __getattr__(self, name):
        return getattr(self.value, name)

    def __repr__(self):
        return f"Constant({self.value.__repr__()})"
