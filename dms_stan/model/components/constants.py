"""Holds code for working with constant values in a DMS Stan model."""

from typing import Optional, Union

import numpy.typing as npt

import dms_stan.model.components as dms_components


class Constant(dms_components.abstract_classes.AbstractPassthrough):
    """
    This class is used to wrap values that are intended to stay constant in the
    Stan model. This is effectively a wrapper around the value that forwards all
    mathematical operations and attribute access to the value. Note that because
    it is a wrapper, in place operations made to the value will be reflected in
    the instance of this class.
    """


class Hyperparameter(dms_components.abstract_classes.AbstractPassthrough):
    """
    Identical to the Constant class, but is used to wrap values that are intended
    as hyperparameters in the Stan model.
    """

    def __init__(self, value: Union[int, float, npt.NDArray]) -> None:
        super().__init__(value)
        self._child: Optional[dms_components.parameters.Parameter] = None
        self._name: Optional[str] = None

    @property
    def child(self) -> "dms_components.parameters.Parameter":
        """
        Returns the child parameter with which this hyperparameter is associated.
        """
        return self._child

    @child.setter
    def child(self, child: "dms_components.parameters.Parameter") -> None:
        """
        Sets the child parameter with which this hyperparameter is associated.
        """
        # We cannot have a child if we already have one
        if self._child is not None:
            raise ValueError("Child already set for this hyperparameter.")

        # Record the child
        self._child = child

    @property
    def name(self) -> str:
        """Returns the name of the hyperparameter."""
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """Sets the name of the hyperparameter."""
        self._name = name

    @property
    def model_varname(self) -> str:
        """Returns the variable name of the child."""
        # Name and child must be set
        if self._name is None or self._child is None:
            raise ValueError("Name and child must be set before calling model_varname.")

        return "".join([self._child.model_varname, "_autoname_", self._name])
