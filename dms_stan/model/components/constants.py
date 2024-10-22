"""Holds code for working with constant values in a DMS Stan model."""

from typing import Union

import numpy as np
import numpy.typing as npt

from dms_stan.model.components.abstract_classes import AbstractPassthrough


class Constant(AbstractPassthrough):
    """
    This class is used to wrap values that are intended to stay constant in the
    Stan model. This is effectively a wrapper around the value that forwards all
    mathematical operations and attribute access to the value. Note that because
    it is a wrapper, in place operations made to the value will be reflected in
    the instance of this class.
    """


class Hyperparameter(AbstractPassthrough):
    """
    Identical to the Constant class, but is used to wrap values that are intended
    as hyperparameters in the Stan model.
    """
