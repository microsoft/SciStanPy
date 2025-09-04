# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Custom type definitions for SciStanPy models.

This module provides type aliases and unions for various components used throughout
the SciStanPy package, including parameter types, distribution types, and utility
types for type checking and documentation purposes.

All imports are conditional on TYPE_CHECKING to avoid circular imports while
maintaining proper type hints for development and documentation tools.
"""

from typing import TYPE_CHECKING, Union

# Everything in this file is only imported if TYPE_CHECKING is True.
if TYPE_CHECKING:

    import numpy as np
    import numpy.typing as npt
    import torch.distributions as dist

    from scistanpy.model.components import constants, parameters
    from scistanpy.model.components.custom_distributions import custom_torch_dists
    from scistanpy.model.components.transformations import transformed_parameters

# Scalar types
Integer = Union[int, "np.integer"]
"""Type alias for integer values.

Accepts both Python's built-in int and NumPy integer types.

:type: Union[int, np.integer]
"""

Float = Union[float, "np.floating"]
"""Type alias for floating-point values.

Accepts both Python's built-in float and NumPy floating-point types.

:type: Union[float, np.floating]
"""

# Parameter types
SampleType = Union[int, float, "npt.NDArray"]
"""Type alias for sample values that can be returned from distributions.

Used to represent values that can be sampled from probability distributions,
including scalars and arrays.

:type: Union[int, float, npt.NDArray]
"""

BaseParameterType = Union[
    "transformed_parameters.TransformedParameter", "constants.Constant"
]
"""Composite type of transformed parameters and constants, whihc are the types
typically used to define hyperparameters.

:type: Union[transformed_parameters.TransformedParameter, constants.Constant]
"""

ContinuousParameterType = Union[
    BaseParameterType,
    "parameters.ContinuousDistribution",
    float,
    "npt.NDArray[np.floating]",
]
"""Type alias for continuous-valued parameters.

Encompasses all parameter types that can take continuous values, including
base parameters, continuous distributions, and numeric values.

:type: Union[BaseParameterType, parameters.ContinuousDistribution, float, npt.NDArray[np.floating]]
"""

DiscreteParameterType = Union[
    BaseParameterType,
    "parameters.DiscreteDistribution",
    int,
    "npt.NDArray[np.integer]",
]
"""Type alias for discrete-valued parameters.

Encompasses all parameter types that can take discrete values, including
base parameters, discrete distributions, and integer values.

:type: Union[BaseParameterType, parameters.DiscreteDistribution, int, npt.NDArray[np.integer]]
"""

CombinableParameterType = Union[ContinuousParameterType, DiscreteParameterType]
"""Type alias for parameters that can be combined in model operations.

Represents any parameter type that can be used in mathematical operations
and model construction, covering both continuous and discrete parameters.

:type: Union[ContinuousParameterType, DiscreteParameterType]
"""

# Distribution types
SciStanPyDistribution = Union[
    "dist.Distribution",
    "custom_torch_dists.CustomDistribution",
]
"""Type alias for PyTorch-compatible probability distributions used in SciStanPy.

Encompasses both PyTorch distributions and custom SciStanPy distribution
implementations.

:type: Union[dist.Distribution, custom_torch_dists.CustomDistribution]
"""

# Diagnostic output types
ProcessedTestRes = dict[str, tuple[tuple["npt.NDArray", ...], int]]
"""Type alias for processed diagnostic test results.

Used for storing test results with associated metadata.

:type: dict[str, tuple[tuple[npt.NDArray, ...], int]]
"""

StrippedTestRes = dict[str, tuple["npt.NDArray", ...]]
"""Type alias for stripped diagnostic test results.

Simplified version of test results containing only the essential array data.

:type: dict[str, tuple[npt.NDArray, ...]]
"""

# Type for indexing
IndexType = Union["npt.NDArray[np.integer]", slice, int, Ellipsis, None]
"""Type alias for array indexing operations.

Covers all valid indexing types for NumPy arrays and similar data structures,
including integer arrays, slices, single integers, ellipsis, and None.

:type: Union[npt.NDArray[np.integer], slice, int, Ellipsis, None]
"""
