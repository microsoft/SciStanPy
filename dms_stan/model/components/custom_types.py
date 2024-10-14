"""Holds custom component types for DMS Stan models."""

from typing import Union

import numpy as np
import numpy.typing as npt

from dms_stan.model.components.constants import Constant
from dms_stan.model.components.parameters import (
    ContinuousDistribution,
    DiscreteDistribution,
)
from dms_stan.model.components.transformed_parameters import TransformedParameter


# Define custom types for this module
SampleType = Union[int, float, npt.NDArray]
ContinuousParameterType = Union[
    ContinuousDistribution,
    TransformedParameter,
    Constant,
    float,
    npt.NDArray[np.floating],
]
DiscreteParameterType = Union[
    DiscreteDistribution,
    TransformedParameter,
    Constant,
    int,
    npt.NDArray[np.integer],
]
CombinableParameterType = Union[ContinuousParameterType, DiscreteParameterType]
