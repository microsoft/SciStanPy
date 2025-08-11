"""Utility functions for the SciStanPy package."""

from __future__ import annotations

import importlib.util
import sys

from typing import Collection, Literal, overload, TYPE_CHECKING

import dask.config
import numpy as np
import numpy.typing as npt
import torch

from arviz.utils import Dask
from scipy import stats
from tqdm import tqdm

if TYPE_CHECKING:
    from scistanpy import custom_types


def lazy_import(name: str):
    """
    Used for importing a module only when it is needed. This is for speeding up
    the import time of the package.

    Args:
        name: The module name to import (e.g., 'scistanpy.model.components.constants')

    Returns:
        The imported module
    """
    # Check if the module is already imported
    if name in sys.modules:
        return sys.modules[name]

    # If not, import it lazily (modified from here:
    # https://docs.python.org/3/library/importlib.html#implementing-lazy-imports)
    # Get the spec
    spec = importlib.util.find_spec(name)

    # If the spec is None, raise an ImportError
    if spec is None:
        raise ImportError(f"Module '{name}' not found.")

    # Create the module with a lazy loader
    spec.loader = importlib.util.LazyLoader(spec.loader)
    module = importlib.util.module_from_spec(spec)

    # Store with the alias if provided, otherwise use original name
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class LazyObjectProxy:
    """A proxy that delays importing a module and accessing an object until first use."""

    def __init__(self, module_name: str, obj_name: str):
        self._module_name = module_name
        self._obj_name = obj_name
        self._cached_obj = None

    def _get_object(self):
        """Import the module and get the object if not already cached."""
        if self._cached_obj is None:
            module = lazy_import(self._module_name)
            try:
                self._cached_obj = getattr(module, self._obj_name)
            except AttributeError as e:
                raise ImportError(
                    f"cannot import name '{self._obj_name}' from '{self._module_name}'"
                ) from e
        return self._cached_obj

    def __call__(self, *args, **kwargs):
        """Forward calls to the actual object."""
        return self._get_object()(*args, **kwargs)

    def __getattr__(self, name):
        """Forward attribute access to the actual object."""
        return getattr(self._get_object(), name)

    def __repr__(self):
        """Return a representation of the proxy."""
        if self._cached_obj is not None:
            return repr(self._cached_obj)
        return f"<LazyObjectProxy for {self._module_name}.{self._obj_name}>"


def lazy_import_from(module_name: str, obj_name: str):
    """
    Return a proxy that will lazily import a specific object from a module only when first used.
    Equivalent to 'from module_name import obj_name' but delayed until actual use.

    Args:
        module_name: The module name to import from (e.g., 'scistanpy.model.components.constants')
        obj_name: The object name to import (e.g., 'Constant')

    Returns:
        LazyObjectProxy: A proxy that will import and return the object when first accessed
    """
    return LazyObjectProxy(module_name, obj_name)


@overload
def choose_module(dist: torch.Tensor) -> torch: ...


@overload
def choose_module(dist: "custom_types.SampleType") -> np: ...


def choose_module(dist):
    """
    Choose the module to use for the operation based on the type of the distribution.
    """
    if isinstance(dist, torch.Tensor):
        return torch
    elif isinstance(dist, np.ndarray):
        return np
    else:
        raise TypeError(f"Unsupported type for determining module: {type(dist)}.")


@overload
def stable_sigmoid(exponent: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]: ...


@overload
def stable_sigmoid(exponent: torch.Tensor) -> torch.Tensor: ...


def stable_sigmoid(exponent):
    """
    Stable sigmoid function to avoid overflow.
    """
    # Are we working with torch or numpy?
    module = choose_module(exponent)

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


def faster_autocorrelation(x):
    """
    Faster version of scipy.stats.spearmanr when x and y might contain NaNs. `x`
    is assumed to be a 2D array with shape (n, m), where `n` is the number of
    samples and `m` is the number of features (i.e., like scipy's `spearmanr` but
    with `axis=1`.)
    """
    # Build a mask for the non-NaN values in each row
    nonnan_mask = ~np.isnan(x)

    # Get an output array for the correlations
    rhos = np.full((x.shape[0], x.shape[0]), np.nan)

    # Calculate the rhos
    for i, row_1 in tqdm(
        enumerate(x), total=x.shape[0], smoothing=1.0, desc="Calculating rhos"
    ):
        for j, row_2 in enumerate(x):

            # If i == j, then we can just set the value to 1
            if i == j:
                rhos[i, j] = 1.0
                continue

            # If i > j, then we can pull the value from the other side of the matrix
            if i > j:
                rhos[i, j] = rhos[j, i]
                continue

            # Calculate the correlation for the non-NaN values
            mask = nonnan_mask[i] & nonnan_mask[j]
            rhos[i, j] = stats.spearmanr(row_1[mask], row_2[mask]).statistic

    # There should be no NaNs
    assert not np.any(np.isnan(rhos))
    return rhos
