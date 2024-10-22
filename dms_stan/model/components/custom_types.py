"""Holds custom component types for DMS Stan models."""

from typing import Union

import numpy as np
import numpy.typing as npt

from dms_stan.model.components.constants import Constant, Hyperparameter
from dms_stan.model.components.parameters import (
    ContinuousDistribution,
    DiscreteDistribution,
)
from dms_stan.model.components.transformed_parameters import TransformedParameter


# Define custom types for this module
SampleType = Union[int, float, npt.NDArray]
BaseParameterType = Union[TransformedParameter, Constant, Hyperparameter]
ContinuousParameterType = Union[
    BaseParameterType,
    ContinuousDistribution,
    float,
    npt.NDArray[np.floating],
]
DiscreteParameterType = Union[
    BaseParameterType,
    DiscreteDistribution,
    int,
    npt.NDArray[np.integer],
]
CombinableParameterType = Union[ContinuousParameterType, DiscreteParameterType]
