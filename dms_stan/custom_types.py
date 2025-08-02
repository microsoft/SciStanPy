"""Holds custom component types for DMS Stan models."""

from typing import Union

import numpy as np
import numpy.typing as npt
import torch.distributions as dist

from dms_stan.model import components

# Parameter types
SampleType = Union[int, float, npt.NDArray]
BaseParameterType = Union[
    components.transformations.TransformedParameter, components.constants.Constant
]
ContinuousParameterType = Union[
    BaseParameterType,
    components.parameters.ContinuousDistribution,
    float,
    npt.NDArray[np.floating],
]
DiscreteParameterType = Union[
    BaseParameterType,
    components.parameters.DiscreteDistribution,
    int,
    npt.NDArray[np.integer],
]
CombinableParameterType = Union[ContinuousParameterType, DiscreteParameterType]

# Distribution types
DMSStanDistribution = Union[
    dist.Distribution,
    components.custom_distributions.custom_torch_dists.CustomDistribution,
]

# Diagnostic output types
ProcessedTestRes = dict[str, tuple[tuple[npt.NDArray, ...], int]]
StrippedTestRes = dict[str, tuple[npt.NDArray, ...]]
