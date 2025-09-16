# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Utility functions and classes for the SciStanPy package.

This module provides various utility functions and classes that support
the core functionality of SciStanPy, including:

    - Lazy importing mechanisms for performance optimization
    - Mathematical utility functions for numerical stability
    - Array chunking utilities for efficient memory management
    - Context managers for external library integration
    - Optimized statistical computation functions

Users will not typically need to interact with this module directly--it is designed
to be used internally by SciStanPy.
"""

from __future__ import annotations

import importlib.util
import sys

from types import ModuleType
from typing import Collection, Literal, overload, TYPE_CHECKING, Union

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
    """Import a module only when it is first needed.

    This function implements lazy module importing to improve package import
    performance by deferring module loading until actual use.

    :param name: The fully qualified module name to import
    :type name: str

    :returns: The imported module
    :rtype: module

    :raises ImportError: If the specified module cannot be found

    Example:
        >>> # Module is not loaded until first use
        >>> numpy_module = lazy_import('numpy')
        >>> # Now numpy is actually imported
        >>> array = numpy_module.array([1, 2, 3])

    .. note::
        If the module is already imported, returns the cached version
        from sys.modules for efficiency.
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
    """A proxy that delays importing a module and accessing an object until first use.

    This class provides a lazy loading mechanism for specific objects within
    modules, allowing fine-grained control over when imports occur. The proxy
    forwards all method calls and attribute access to the actual object once
    it's loaded.

    :param module_name: The fully qualified name of the module containing the object
    :type module_name: str
    :param obj_name: The name of the object to import from the module
    :type obj_name: str

    :ivar _module_name: Stored module name for lazy loading
    :ivar _obj_name: Stored object name for lazy loading
    :ivar _cached_obj: Cached reference to the imported object (None until first use)

    Example:
        >>> # Create a proxy for numpy.array
        >>> array_proxy = LazyObjectProxy('numpy', 'array')
        >>> # numpy is not imported yet
        >>> my_array = array_proxy([1, 2, 3])  # Now numpy is imported
    """

    def __init__(self, module_name: str, obj_name: str):
        self._module_name = module_name
        self._obj_name = obj_name
        self._cached_obj = None

    def _get_object(self):
        """Import the module and get the object if not already cached.

        :returns: The imported object
        :raises ImportError: If the module or object cannot be imported
        """
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
        """Forward calls to the actual object.

        :param args: Positional arguments to forward
        :param kwargs: Keyword arguments to forward
        :returns: Result of calling the proxied object
        """
        return self._get_object()(*args, **kwargs)

    def __getattr__(self, name):
        """Forward attribute access to the actual object.

        :param name: Name of the attribute to access
        :returns: The requested attribute from the proxied object
        """
        return getattr(self._get_object(), name)

    def __repr__(self):
        """Return a representation of the proxy.

        :returns: String representation of the proxy or proxied object
        :rtype: str
        """
        if self._cached_obj is not None:
            return repr(self._cached_obj)
        return f"<LazyObjectProxy for {self._module_name}.{self._obj_name}>"


def lazy_import_from(module_name: str, obj_name: str):
    """Create a lazy import proxy for a specific object from a module.

    This function provides a convenient way to create lazy import proxies,
    equivalent to ``from module_name import obj_name`` but with deferred loading.

    :param module_name: The fully qualified module name to import from
    :type module_name: str
    :param obj_name: The name of the object to import from the module
    :type obj_name: str

    :returns: A proxy that will import and return the object when first accessed
    :rtype: LazyObjectProxy

    Example:
        >>> # Equivalent to 'from numpy import array' but lazy
        >>> array = lazy_import_from('numpy', 'array')
        >>> my_array = array([1, 2, 3])  # numpy imported here
    """
    return LazyObjectProxy(module_name, obj_name)


def choose_module(dist: Union[torch.Tensor, "custom_types.SampleType"]) -> ModuleType:
    """Choose the appropriate computational module based on input type.

    This function provides automatic backend selection between NumPy and
    PyTorch based on the type of the input data.

    :param dist: Input data whose type determines the module choice
    :type dist: Union[torch.Tensor, np.ndarray, custom_types.SampleType]

    :returns: The appropriate module (torch for tensors, numpy for arrays)
    :rtype: Union[torch, np]

    :raises TypeError: If the input type is not supported

    Example:
        >>> import torch
        >>> tensor = torch.tensor([1.0, 2.0])
        >>> module = choose_module(tensor)  # Returns torch module
        >>> result = module.exp(tensor)
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
    r"""Compute sigmoid function in a numerically stable way.

    This function implements a numerically stable version of the sigmoid
    function that avoids overflow issues by using different computational
    approaches for positive and negative inputs.

    :param exponent: Input values for sigmoid computation
    :type exponent: Union[torch.Tensor, npt.NDArray[np.floating]]

    :returns: Sigmoid values with the same type and shape as input
    :rtype: Union[torch.Tensor, npt.NDArray[np.floating]]

    The function uses the identity:

    .. math::

        \sigma(x) =
        \begin{cases}
            \frac{1}{1 + e^{-x}} & \text{if } x \geq 0 \\
            \frac{e^{x}}{1 + e^{x}} & \text{if } x < 0
        \end{cases}
        """
    # Are we working with torch or numpy?
    module = choose_module(exponent)

    # Empty array to store the results
    sigma_exponent = module.full_like(exponent, module.nan)

    # Different approach for positive and negative values
    mask = exponent >= 0

    # Calculate the sigmoid function for the positives
    sigma_exponent[mask] = 1 / (1 + module.exp(-exponent[mask]))

    # Calculate the sigmoid function for the negatives
    neg_calc = module.exp(exponent[~mask])
    sigma_exponent[~mask] = neg_calc / (1 + neg_calc)

    # We should have no NaN values in the result
    assert not module.any(module.isnan(sigma_exponent))
    return sigma_exponent


def get_chunk_shape(
    array_shape: tuple[custom_types.Integer, ...],
    array_precision: Literal["double", "single", "half"],
    mib_per_chunk: custom_types.Integer | None = None,
    frozen_dims: Collection[custom_types.Integer] = (),
) -> tuple[custom_types.Integer, ...]:
    """Calculate optimal chunk shape for Dask arrays based on memory constraints.

    This function determines the optimal chunking strategy for large arrays
    processed with Dask, balancing memory usage with computational efficiency.
    It respects frozen dimensions that should not be chunked.

    :param array_shape: Shape of the array to be chunked
    :type array_shape: tuple[custom_types.Integer, ...]
    :param array_precision: Numerical precision assumed in calculating memory usage.
    :type array_precision: Literal["double", "single", "half"]
    :param mib_per_chunk: Target chunk size in MiB. If None, uses Dask default
    :type mib_per_chunk: Union[custom_types.Integer, None]
    :param frozen_dims: Dimensions that should not be chunked
    :type frozen_dims: Collection[custom_types.Integer]

    :returns: Optimal chunk shape for the array
    :rtype: tuple[custom_types.Integer, ...]

    :raises ValueError: If mib_per_chunk is negative
    :raises IndexError: If frozen_dims contains invalid dimension indices

    The algorithm:
        1. Calculates memory usage per array element based on precision
        2. Sets frozen dimensions to their full size
        3. Iteratively determines chunk sizes for remaining dimensions
        4. Ensures total chunk memory stays within the specified limit (or as close to
           it as possible if frozen dimensions result in a smallest possible size above
           the limit)

    Example:
        >>> # Chunk a (1000, 2000, 100) array, keeping last dim intact
        >>> shape = get_chunk_shape(
        ...     (1000, 2000, 100), "double",
        ...     mib_per_chunk=64, frozen_dims=(2,)
        ... )
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
    """Context manager for enabling Dask integration with ArviZ.

    This context manager provides a convenient way to enable Dask-based
    parallel computation within ArviZ operations, automatically handling
    the setup and teardown of Dask configuration.

    :param dask_type: Type of Dask computation to enable
    :type dask_type: str
    :param output_dtypes: Expected output data types for Dask operations
    :type output_dtypes: Union[list[object], None]

    :ivar dask_type: Stored Dask computation type
    :ivar output_dtypes: Stored output data types configuration

    Example:
        >>> with az_dask() as dask_ctx:
        ...     # ArviZ operations here will use Dask parallelization
        ...     result = az.summary(trace_data)

    .. note::
        The context manager automatically disables Dask when exiting,
        ensuring clean state management.
    """

    def __init__(
        self, dask_type: str = "parallelized", output_dtypes: list[object] | None = None
    ):

        # Record the dask type and output dtypes
        self.dask_type = dask_type
        self.output_dtypes = output_dtypes or [float]

    def __enter__(self):
        """Enable Dask with ArviZ.

        :returns: Self for use in with statement
        :rtype: az_dask
        """
        Dask.enable_dask(
            dask_kwargs={"dask": self.dask_type, "output_dtypes": self.output_dtypes}
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Disable Dask when exiting the context.

        :param exc_type: Exception type if an exception occurred
        :param exc_value: Exception value if an exception occurred
        :param traceback: Exception traceback if an exception occurred
        """
        Dask.disable_dask()


def faster_autocorrelation(x):
    """Compute Spearman rank correlation matrix with optimized NaN handling.

    This function provides a faster implementation of Spearman rank correlation
    computation for 2D arrays that may contain NaN values. It's optimized for
    cases where missing data patterns vary across samples.

    :param x: Input array with shape (n, m) where n is samples and m is features
    :type x: npt.NDArray[np.floating]

    :returns: Spearman rank correlation matrix of shape (n, n)
    :rtype: npt.NDArray[np.floating]

    The function:
        1. Builds masks for non-NaN values in each row
        2. Computes pairwise correlations using only overlapping valid data
        3. Uses matrix symmetry to avoid redundant calculations
        4. Provides progress tracking for long computations

    Example:
        >>> import numpy as np
        >>> # Data with some NaN values
        >>> data = np.random.randn(100, 50)
        >>> data[data < -1] = np.nan  # Introduce some NaNs
        >>> corr_matrix = faster_autocorrelation(data)

    Note:
        This function assumes the input follows scipy.stats.spearmanr
        conventions but with axis=1 behavior and NaN-aware computation.
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
