"""Utility functions for the DMS Stan package."""

from typing import Collection, Literal, overload

import dask.config
import numpy as np
import numpy.typing as npt
import torch

from arviz.utils import Dask


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


def get_chunk_shape(
    array_shape: tuple[int, ...],
    array_precision: Literal["double", "single", "half"],
    mib_per_chunk: int | None = None,
    frozen_dims: Collection[int] = (),
) -> tuple[int, ...]:
    """
    Get the shape of chunks for a dask array based on the shape of the array and
    the desired chunk size in MB.

    Parameters
    ----------
    array_shape (tuple[int, ...]): Shape of the array that will be chunked.
    array_precision (Literal["double", "single", "half"]): Precision of the array
        that will be chunked.
    mib_per_chunk (int): Desired chunk size in MiB. Default is to use the default
        chunk size of Dask.
    frozen_dims (tuple[int, ...]): Dimensions that should not be chunked. Note that
        the target chunk size may be exceeded if the frozen dimensions are large.

    Returns
    -------
    tuple[int, ...]: Shape of the chunks.
    """
    mib_per_chunk = mib_per_chunk or int(
        dask.config.get("array.chunk-size").removesuffix("MiB")
    )
    if mib_per_chunk < 0:
        raise ValueError("`mib_per_chunk` must be a positive integer or `None`.")

    # Set up frozen dimensions: Negatives to positive equivalent, the whole thing
    # to a set
    frozen_dims = {
        len(array_shape) + dimind if dimind < 0 else dimind for dimind in frozen_dims
    }
    if len(frozen_dims) != 0 and (
        min(frozen_dims) < 0 or max(frozen_dims) >= len(array_shape)
    ):
        raise IndexError("Dimensions out of range for array shape.")

    # Get the number of bytes per entry
    mib_per_entry = {
        "double": 8,
        "single": 4,
        "half": 2,
    }[array_precision] / 1024**2

    # The base chunk shape is a list of ones, except for the frozen dimensions,
    # which are set to the size of the dimension.
    chunk_shape = [
        dimsize if dimind in frozen_dims else 1
        for dimind, dimsize in enumerate(array_shape)
    ]

    # We have a base volume that depends on the frozen dimensions. If this volume
    # is larger than the MiB limit, we are already done.
    volume = np.prod(chunk_shape) * mib_per_entry
    if volume >= mib_per_chunk:
        return tuple(chunk_shape)

    # Otherwise, we loop over the dimensions from last to first and determine the
    # chunk size. We start with the last dimension and go backwards until we hit
    # the MiB limit.
    for dimind in range(len(array_shape) - 1, -1, -1):

        # Skip frozen dimensions. We have already accounted for these.
        if dimind in frozen_dims:
            continue

        # Record the size of this dimension
        dimsize = array_shape[dimind]

        # How many elements on this dimension do we need?
        num_elements = mib_per_chunk // volume
        assert num_elements > 0, "Chunk size is too small."

        # If the number of elements is larger than the size of the dimension, we
        # set the chunk size to the size of the dimension, update the volume, and
        # continue.
        if num_elements > dimsize:
            chunk_shape[dimind] = dimsize
            volume *= dimsize
            continue

        # Otherwise, we set the chunk size to the number of elements. We are done.
        chunk_shape[dimind] = int(num_elements.item())
        return tuple(chunk_shape)

    # Should we reach the end of the loop, we have not exceeded the MiB limit. We
    # can set the chunk size to the size of the array.
    return array_shape


class az_dask:  # pylint: disable=invalid-name
    """Context manager to enable Dask with ArviZ."""

    def __init__(
        self, dask_type: str = "parallelized", output_dtypes: list[object] | None = None
    ):

        # Record the dask type and output dtypes
        self.dask_type = dask_type
        self.output_dtypes = output_dtypes or [float]

    def __enter__(self):
        """Enable Dask with ArviZ."""
        Dask.enable_dask(
            dask_kwargs={"dask": self.dask_type, "output_dtypes": self.output_dtypes}
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        Dask.disable_dask()
