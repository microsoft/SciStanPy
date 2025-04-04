"""Utility functions for the DMS Stan package."""

from typing import overload

import numpy as np
import numpy.typing as npt
import torch


def stable_sigmoid(dist1: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Stable sigmoid function to avoid overflow.
    """
    # Empty array to store the results
    sigma_dist1 = np.full_like(dist1, np.nan)

    # Different approach for positive and negative values
    mask = dist1 >= 0

    # Calculate the sigmoid function for the positives
    pos_calc = np.exp(dist1[~mask])
    sigma_dist1[~mask] = pos_calc / (1 + pos_calc)

    # Calculate the sigmoid function for the negatives
    sigma_dist1[mask] = 1 / (1 + np.exp(-dist1[mask]))

    # We should have no NaN values in the result
    assert not np.any(np.isnan(sigma_dist1))
    return sigma_dist1


@overload
def stable_x0_sigmoid_growth(
    t: npt.NDArray[np.floating],
    x0: npt.NDArray[np.floating],
    r: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]: ...


@overload
def stable_x0_sigmoid_growth(
    t: torch.Tensor, x0: torch.Tensor, r: torch.Tensor
) -> torch.Tensor: ...


def stable_x0_sigmoid_growth(t, x0, r):
    """
    Calculates the sigmoid growth function parametrized from starting amount
    `x0`. Note that all members of x0 must be positive.
    """

    def positive_path():

        # Select relevant components
        x0_m = x0[mask]

        # Get the exponentiation multiplied by x0
        exponentiation = x0_m * getattr(module, "exp")(exponent[mask])

        # Finish calculation
        return exponentiation / (exponentiation + 1 - x0_m)

    def negative_path():

        # Select relevant components
        x0_m = x0[inv_mask]

        # Calculate
        return x0_m / ((1 - x0_m) * getattr(module, "exp")(-exponent[inv_mask]) + x0_m)

    # Get the module
    module = torch if isinstance(x0, torch.Tensor) else np

    # We are assuming that all x0 are between 0 and 1. The lower bound is because
    # we cannot have negative abundance. The upper bound is because we have set
    # the carrying capacity to be 1 to eliminate a parameter.
    min_x0, max_x0 = getattr(module, "min")(x0), getattr(module, "max")(x0)
    if min_x0 == 0:
        raise ValueError("Minimum x0 is '0'. Growth is not possible from nothing.")
    elif min_x0 < 0:
        raise ValueError(
            f"Minimum x0 is {min_x0}, but x0 must be positive. Negative abundance "
            "is nonsensical."
        )
    if max_x0 == 1:
        raise ValueError("Maximum x0 is '1'. No more growth possible.")
    elif max_x0 > 1:
        raise ValueError(
            f"Maximum x0 is {max_x0}, but x0 must be <1. This i because we assume "
            "a carrying capacity of '1'."
        )

    # The output array is the shape of the three elements broadcast together
    outshape = np.broadcast_shapes(
        *[tuple(getattr(arr, "shape", ())) for arr in (t, x0, r)]
    )

    # Broadcast the inputs to the output shape
    broadcaster = getattr(module, "broadcast_to")
    t = broadcaster(t, outshape)
    x0 = broadcaster(x0, outshape)
    r = broadcaster(r, outshape)

    # Get the product of t and r. Depending on whether this is positive or negative,
    # we use different forms of the equation.
    exponent = r * t
    mask = exponent >= 0
    inv_mask = ~mask

    # Compute both paths
    positives, negatives = positive_path(), negative_path()

    # Build the output array
    output = getattr(module, "empty_like")(x0, dtype=positives.dtype)
    output[mask] = positives
    output[inv_mask] = negatives

    return output
