"""Holds custom component types for SciStanPy models."""

from typing import TYPE_CHECKING, Union

# Everything in this file is only imported if TYPE_CHECKING is True.
if TYPE_CHECKING:

    import numpy as np
    import numpy.typing as npt
    import torch.distributions as dist

    from scistanpy.model.components import constants, parameters
    from scistanpy.model.components.custom_distributions import custom_torch_dists
    from scistanpy.model.components.transformations import transformed_parameters


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
SciStanPyDistribution = Union[
    "dist.Distribution",
    "custom_torch_dists.CustomDistribution",
]

# Diagnostic output types
ProcessedTestRes = dict[str, tuple[tuple["npt.NDArray", ...], int]]
StrippedTestRes = dict[str, tuple["npt.NDArray", ...]]

# Type for indexing
IndexType = Union["npt.NDArray[np.integer]", slice, int]
