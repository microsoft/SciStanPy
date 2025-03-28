"""Utility functions for the DMS Stan package."""

import numpy as np
import numpy.typing as npt


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
