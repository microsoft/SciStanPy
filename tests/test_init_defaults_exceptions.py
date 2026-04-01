# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for scistanpy top-level package, defaults, exceptions, and custom_types."""

# pylint: disable=missing-function-docstring

import numpy as np
import pytest
import torch

import scistanpy as ssp
from scistanpy import defaults, exceptions


# ---------------------------------------------------------------------------
# __init__.py
# ---------------------------------------------------------------------------
class TestInit:
    """Tests for scistanpy.__init__ module."""

    def test_version_is_string(self):
        assert isinstance(ssp.__version__, str)

    def test_manual_seed_sets_rng(self):
        ssp.manual_seed(123)
        val1 = ssp.RNG.random()
        ssp.manual_seed(123)
        val2 = ssp.RNG.random()
        assert val1 == val2

    def test_manual_seed_none_creates_rng(self):
        ssp.manual_seed(None)
        assert isinstance(ssp.RNG, np.random.Generator)

    def test_manual_seed_sets_torch_seed(self):
        ssp.manual_seed(99)
        t1 = torch.randn(5)
        ssp.manual_seed(99)
        t2 = torch.randn(5)
        assert torch.allclose(t1, t2)

    def test_manual_seed_none_does_not_set_torch(self):
        """When seed is None, torch.manual_seed should NOT be called."""
        ssp.manual_seed(0)
        t1 = torch.randn(3)
        ssp.manual_seed(None)
        t2 = torch.randn(3)
        # With None, torch seed is not set, so t2 should differ (very unlikely to match)
        assert not torch.allclose(t1, t2)

    def test_rng_exists_after_import(self):
        assert hasattr(ssp, "RNG")
        assert isinstance(ssp.RNG, np.random.Generator)

    def test_public_exports(self):
        """Key classes and modules should be accessible from the top-level."""
        assert hasattr(ssp, "Model")
        assert hasattr(ssp, "Constant")
        assert hasattr(ssp, "parameters")
        assert hasattr(ssp, "results")


# ---------------------------------------------------------------------------
# defaults.py
# ---------------------------------------------------------------------------
class TestDefaults:
    """Tests for default configuration values."""

    def test_default_n_epochs_is_positive_int(self):
        assert isinstance(defaults.DEFAULT_N_EPOCHS, int)
        assert defaults.DEFAULT_N_EPOCHS > 0

    def test_default_early_stop_is_positive_int(self):
        assert isinstance(defaults.DEFAULT_EARLY_STOP, int)
        assert defaults.DEFAULT_EARLY_STOP > 0

    def test_default_lr_is_positive_float(self):
        assert isinstance(defaults.DEFAULT_LR, float)
        assert defaults.DEFAULT_LR > 0

    def test_default_index_order_is_tuple_of_strings(self):
        assert isinstance(defaults.DEFAULT_INDEX_ORDER, tuple)
        assert all(isinstance(s, str) for s in defaults.DEFAULT_INDEX_ORDER)

    def test_default_dim_names_excludes_n(self):
        assert "n" not in defaults.DEFAULT_DIM_NAMES
        assert "a" in defaults.DEFAULT_DIM_NAMES
        assert "z" in defaults.DEFAULT_DIM_NAMES

    def test_default_stanc_options(self):
        assert isinstance(defaults.DEFAULT_STANC_OPTIONS, dict)

    def test_default_cpp_options(self):
        assert isinstance(defaults.DEFAULT_CPP_OPTIONS, dict)

    def test_default_diagnostic_thresholds(self):
        assert 0 < defaults.DEFAULT_EBFMI_THRESH < 1
        assert defaults.DEFAULT_ESS_THRESH > 0
        assert defaults.DEFAULT_RHAT_THRESH >= 1.0


# ---------------------------------------------------------------------------
# exceptions.py
# ---------------------------------------------------------------------------
class TestExceptions:
    """Tests for custom exception hierarchy."""

    def test_scistanpy_error_is_exception(self):
        assert issubclass(exceptions.SciStanPyError, Exception)

    def test_numpy_sample_error_inherits_from_base(self):
        assert issubclass(exceptions.NumpySampleError, exceptions.SciStanPyError)

    def test_scistanpy_error_can_be_raised(self):
        with pytest.raises(exceptions.SciStanPyError):
            raise exceptions.SciStanPyError("test message")

    def test_numpy_sample_error_caught_as_base(self):
        with pytest.raises(exceptions.SciStanPyError):
            raise exceptions.NumpySampleError("bad sample")

    def test_error_message_preserved(self):
        msg = "detailed error info"
        err = exceptions.NumpySampleError(msg)
        assert str(err) == msg
