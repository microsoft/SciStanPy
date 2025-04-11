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
    A: npt.NDArray[np.floating],  # pylint: disable=invalid-name
    r: npt.NDArray[np.floating],
    x0: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]: ...


@overload
def stable_x0_sigmoid_growth(
    t: torch.Tensor,
    A: torch.Tensor,  # pylint: disable=invalid-name
    r: torch.Tensor,
    x0: torch.Tensor,
) -> torch.Tensor: ...


def stable_x0_sigmoid_growth(t, A, r, x0):  # pylint: disable=invalid-name
    """
    Calculates the sigmoid growth function parametrized from starting amount
    `x0`. Note that all members of x0 must be positive.
    """

    def positive_path():

        # Get the exponentiation multiplied by x0
        exponentiation = getattr(module, "exp")(exponent[mask])

        # Finish calculation
        return exponentiation / (exponentiation + rel_change[mask])

    def negative_path():

        # Calculate
        return 1 / (
            1 + (rel_change[inv_mask] * getattr(module, "exp")(-exponent[inv_mask]))
        )

    # Get the module
    module = torch if isinstance(x0, torch.Tensor) else np

    # The output array is the shape of the three elements broadcast together
    outshape = np.broadcast_shapes(
        *[tuple(getattr(arr, "shape", ())) for arr in (t, x0, r)]
    )

    # Broadcast the inputs to the output shape
    broadcaster = getattr(module, "broadcast_to")
    t = broadcaster(t, outshape)
    x0 = broadcaster(x0, outshape)
    r = broadcaster(r, outshape)
    A = broadcaster(A, outshape)

    # Get the product of t and r. Depending on whether this is positive or negative,
    # we use different forms of the equation.
    exponent = r * t
    mask = exponent >= 0
    inv_mask = ~mask

    # What's the relative change of the abundance?
    rel_change = (A - x0) / x0

    # Compute both paths
    positives, negatives = positive_path(), negative_path()

    # Build the output array
    output = getattr(module, "empty_like")(x0, dtype=positives.dtype)
    output[mask] = positives
    output[inv_mask] = negatives

    # Scale the output by A
    output = output * A

    return output
