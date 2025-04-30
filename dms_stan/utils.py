"""Utility functions for the DMS Stan package."""

from typing import overload

import numpy as np
import numpy.typing as npt
import torch


@overload
def _get_module(exponent: npt.NDArray[np.floating]) -> np: ...


@overload
def _get_module(exponent: torch.Tensor) -> torch: ...


def _get_module(exponent):
    """
    Get the module (numpy or torch) based on the type of the input.
    """
    return np if isinstance(exponent, np.ndarray) else torch


@overload
def stable_sigmoid(exponent: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]: ...


@overload
def stable_sigmoid(exponent: torch.Tensor) -> torch.Tensor: ...


def stable_sigmoid(exponent):
    """
    Stable sigmoid function to avoid overflow.
    """
    # Are we working with torch or numpy?
    module = _get_module(exponent)

    # Empty array to store the results
    sigma_exponent = module.full_like(exponent, module.nan)

    # Different approach for positive and negative values
    mask = exponent >= 0

    # Calculate the sigmoid function for the positives
    pos_calc = module.exp(exponent[~mask])
    sigma_exponent[~mask] = pos_calc / (1 + pos_calc)

    # Calculate the sigmoid function for the negatives
    sigma_exponent[mask] = 1 / (1 + module.exp(-exponent[mask]))

    # We should have no NaN values in the result
    assert not module.any(module.isnan(sigma_exponent))
    return sigma_exponent
