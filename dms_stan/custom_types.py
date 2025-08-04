"""Holds custom component types for DMS Stan models."""

from typing import TYPE_CHECKING, Union

# Everything in this file is only imported if TYPE_CHECKING is True.
if TYPE_CHECKING:

    import numpy as np
    import numpy.typing as npt
    import torch.distributions as dist

    from dms_stan.model.components import constants, parameters
    from dms_stan.model.components.custom_distributions import custom_torch_dists
    from dms_stan.model.components.transformations import transformed_parameters


# Parameter types
SampleType = Union[int, float, "npt.NDArray"]
BaseParameterType = Union[
    "transformed_parameters.TransformedParameter", "constants.Constant"
]
ContinuousParameterType = Union[
    BaseParameterType,
    "parameters.ContinuousDistribution",
    float,
    "npt.NDArray[np.floating]",
]
DiscreteParameterType = Union[
    BaseParameterType,
    "parameters.DiscreteDistribution",
    int,
    "npt.NDArray[np.integer]",
]
CombinableParameterType = Union[ContinuousParameterType, DiscreteParameterType]

# Distribution types
DMSStanDistribution = Union[
    "dist.Distribution",
    "custom_torch_dists.CustomDistribution",
]

# Diagnostic output types
ProcessedTestRes = dict[str, tuple[tuple["npt.NDArray", ...], int]]
StrippedTestRes = dict[str, tuple["npt.NDArray", ...]]
