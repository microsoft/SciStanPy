"""Holds custom component types for DMS Stan models."""

from typing import Union

import numpy as np
import numpy.typing as npt
import torch.distributions as dist

from .model.components import (
    Constant,
    ContinuousDistribution,
    DiscreteDistribution,
    TransformedParameter,
)
from .model.components.abstract_model_component import AbstractModelComponent
from .model.components.custom_torch_dists import Multinomial


# Parameter types
SampleType = Union[int, float, npt.NDArray]
BaseParameterType = Union[TransformedParameter, Constant]
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

# Distribution types
DMSStanDistribution = Union[dist.Distribution, Multinomial]

# For building Stan models
StanTreeType = list[Union[AbstractModelComponent, "StanTreeType"]]
