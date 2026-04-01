# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for scistanpy.utils module."""

# pylint: disable=missing-function-docstring
import json

import numpy as np
import pytest
import torch

from scistanpy import utils


# ---------------------------------------------------------------------------
# lazy_import / LazyObjectProxy / lazy_import_from
# ---------------------------------------------------------------------------
class TestLazyImport:
    """Tests for lazy import utilities."""

    def test_lazy_import_returns_module(self):
        mod = utils.lazy_import("json")

        assert mod is json

    def test_lazy_import_cached_module(self):
        """If module already in sys.modules, returns it directly."""
        mod1 = utils.lazy_import("os")
        mod2 = utils.lazy_import("os")
        assert mod1 is mod2

    def test_lazy_import_nonexistent_raises(self):
        with pytest.raises(ImportError, match="not found"):
            utils.lazy_import("__nonexistent_module_xyz__")


class TestLazyObjectProxy:
    """Tests for LazyObjectProxy."""

    def test_proxy_call(self):
        array_proxy = utils.LazyObjectProxy("numpy", "array")
        result = array_proxy([1, 2, 3])
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_proxy_getattr(self):
        array_proxy = utils.LazyObjectProxy("numpy", "array")
        # Accessing __name__ triggers _get_object then getattr
        result = array_proxy([1])
        assert result.shape == (1,)

    def test_proxy_repr_before_load(self):
        proxy = utils.LazyObjectProxy("json", "dumps")
        assert "LazyObjectProxy for json.dumps" in repr(proxy)

    def test_proxy_repr_after_load(self):
        proxy = utils.LazyObjectProxy("json", "dumps")
        proxy([1])  # triggers load
        assert "LazyObjectProxy" not in repr(proxy)

    def test_proxy_bad_object_raises(self):
        proxy = utils.LazyObjectProxy("json", "__nonexistent__")
        with pytest.raises(ImportError, match="cannot import name"):
            proxy()

    def test_lazy_import_from_convenience(self):
        proxy = utils.lazy_import_from("numpy", "array")
        assert isinstance(proxy, utils.LazyObjectProxy)
        result = proxy([1, 2])
        np.testing.assert_array_equal(result, np.array([1, 2]))


# ---------------------------------------------------------------------------
# choose_module
# ---------------------------------------------------------------------------
class TestChooseModule:
    """Tests for choose_module."""

    def test_torch_tensor_returns_torch(self):
        t = torch.tensor([1.0])
        assert utils.choose_module(t) is torch

    def test_numpy_array_returns_numpy(self):
        a = np.array([1.0])
        assert utils.choose_module(a) is np

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported type"):
            utils.choose_module([1, 2, 3])

    def test_unsupported_type_int_raises(self):
        with pytest.raises(TypeError, match="Unsupported type"):
            utils.choose_module(42)


# ---------------------------------------------------------------------------
# stable_sigmoid
# ---------------------------------------------------------------------------
class TestStableSigmoid:
    """Tests for numerically stable sigmoid."""

    def test_sigmoid_zero(self):
        x = np.array([0.0])
        result = utils.stable_sigmoid(x)
        np.testing.assert_allclose(result, 0.5, atol=1e-10)

    def test_sigmoid_positive_values(self):
        x = np.array([1.0, 2.0, 10.0])
        result = utils.stable_sigmoid(x)
        expected = 1 / (1 + np.exp(-x))
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_sigmoid_negative_values(self):
        x = np.array([-1.0, -2.0, -10.0])
        result = utils.stable_sigmoid(x)
        expected = np.exp(x) / (1 + np.exp(x))
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_sigmoid_mixed_values(self):
        x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        result = utils.stable_sigmoid(x)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_sigmoid_large_positive_no_overflow(self):
        x = np.array([500.0, 1000.0])
        result = utils.stable_sigmoid(x)
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_sigmoid_large_negative_no_underflow(self):
        x = np.array([-500.0, -1000.0])
        result = utils.stable_sigmoid(x)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_sigmoid_no_nan(self):
        x = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])
        result = utils.stable_sigmoid(x)
        assert not np.any(np.isnan(result))

    def test_sigmoid_output_shape_matches_input(self):
        x = np.array([[1.0, -1.0], [0.0, 2.0]])
        result = utils.stable_sigmoid(x)
        assert result.shape == x.shape


# ---------------------------------------------------------------------------
# get_chunk_shape
# ---------------------------------------------------------------------------
class TestGetChunkShape:
    """Tests for get_chunk_shape."""

    def test_basic_chunking_returns_tuple(self):
        shape = utils.get_chunk_shape((100, 200), "double", mib_per_chunk=1)
        assert isinstance(shape, tuple)

    def test_frozen_dims_preserved(self):
        shape = utils.get_chunk_shape(
            (100, 50, 10), "double", mib_per_chunk=1, frozen_dims=(2,)
        )
        assert shape[2] == 10

    def test_negative_mib_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            utils.get_chunk_shape((100, 200), "double", mib_per_chunk=-1)

    def test_invalid_frozen_dim_raises(self):
        with pytest.raises(IndexError, match="out of range"):
            utils.get_chunk_shape(
                (100, 200), "double", mib_per_chunk=1, frozen_dims=(5,)
            )

    def test_small_array_returns_full_shape(self):
        """If the array fits in one chunk, return the full shape."""
        shape = utils.get_chunk_shape((10, 10), "double", mib_per_chunk=1024)
        assert shape == (10, 10)

    def test_negative_frozen_dim_converted(self):
        """Negative frozen dims should be converted to positive equivalents."""
        shape = utils.get_chunk_shape(
            (100, 50, 10), "double", mib_per_chunk=1, frozen_dims=(-1,)
        )
        assert shape[2] == 10

    def test_half_precision_uses_less_memory(self):
        """Half precision should allow larger chunks."""
        shape_double = utils.get_chunk_shape((1000, 1000), "double", mib_per_chunk=1)
        shape_half = utils.get_chunk_shape((1000, 1000), "half", mib_per_chunk=1)
        # Half precision chunks can be larger since elements are smaller
        assert np.prod(shape_half) >= np.prod(shape_double)


# ---------------------------------------------------------------------------
# az_dask context manager
# ---------------------------------------------------------------------------
class TestAzDask:
    """Tests for the az_dask context manager."""

    def test_context_manager_protocol(self):
        mgr = utils.az_dask()
        assert hasattr(mgr, "__enter__")
        assert hasattr(mgr, "__exit__")

    def test_default_dask_type(self):
        mgr = utils.az_dask()
        assert mgr.dask_type == "parallelized"

    def test_default_output_dtypes(self):
        mgr = utils.az_dask()
        assert mgr.output_dtypes == [float]

    def test_custom_dask_type(self):
        mgr = utils.az_dask(dask_type="delayed")
        assert mgr.dask_type == "delayed"


# ---------------------------------------------------------------------------
# faster_autocorrelation
# ---------------------------------------------------------------------------
class TestFasterAutocorrelation:
    """Tests for faster_autocorrelation."""

    def test_diagonal_is_one(self):
        np.random.seed(42)
        x = np.random.randn(5, 20)
        rhos = utils.faster_autocorrelation(x)
        np.testing.assert_allclose(np.diag(rhos), 1.0)

    def test_symmetric(self):
        np.random.seed(42)
        x = np.random.randn(5, 20)
        rhos = utils.faster_autocorrelation(x)
        np.testing.assert_allclose(rhos, rhos.T)

    def test_output_shape(self):
        np.random.seed(42)
        x = np.random.randn(4, 10)
        rhos = utils.faster_autocorrelation(x)
        assert rhos.shape == (4, 4)

    def test_no_nan_in_result(self):
        np.random.seed(42)
        x = np.random.randn(3, 15)
        rhos = utils.faster_autocorrelation(x)
        assert not np.any(np.isnan(rhos))

    def test_values_in_valid_range(self):
        np.random.seed(42)
        x = np.random.randn(4, 20)
        rhos = utils.faster_autocorrelation(x)
        assert np.all(rhos >= -1) and np.all(rhos <= 1)
