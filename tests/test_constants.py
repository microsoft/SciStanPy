# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for scistanpy.model.components.constants module."""
# pylint: disable=missing-function-docstring, protected-access

import numpy as np
import pytest
import torch

from scistanpy.model.components.constants import Constant


class TestConstantInit:
    """Tests for Constant initialization."""

    def test_scalar_float(self):
        c = Constant(3.14)
        assert c.value == pytest.approx(3.14)
        assert c.shape == ()
        assert c.BASE_STAN_DTYPE == "real"

    def test_scalar_int(self):
        c = Constant(42)
        assert c.value == 42
        assert c.shape == ()
        assert c.BASE_STAN_DTYPE == "int"

    def test_numpy_array_float(self):
        arr = np.array([1.0, 2.0, 3.0])
        c = Constant(arr)
        np.testing.assert_array_equal(c.value, arr)
        assert c.shape == (3,)
        assert c.BASE_STAN_DTYPE == "real"

    def test_numpy_array_int(self):
        arr = np.array([1, 2, 3])
        c = Constant(arr)
        np.testing.assert_array_equal(c.value, arr)
        assert c.shape == (3,)
        assert c.BASE_STAN_DTYPE == "int"

    def test_2d_array(self):
        arr = np.ones((3, 4))
        c = Constant(arr)
        assert c.shape == (3, 4)

    def test_lower_bound_violated_raises(self):
        with pytest.raises(ValueError, match="less than lower bound"):
            Constant(-1.0, lower_bound=0.0)

    def test_upper_bound_violated_raises(self):
        with pytest.raises(ValueError, match="greater than upper bound"):
            Constant(2.0, upper_bound=1.0)

    def test_bounds_stored(self):
        c = Constant(0.5, lower_bound=0.0, upper_bound=1.0)
        assert c.LOWER_BOUND == 0.0
        assert c.UPPER_BOUND == 1.0

    def test_bounds_valid_values(self):
        c = Constant(0.5, lower_bound=0.0, upper_bound=1.0)
        assert c.value == pytest.approx(0.5)


class TestConstantTogglability:
    """Tests for togglable behavior."""

    def test_float_is_togglable_by_default(self):
        c = Constant(1.0)
        assert c.is_togglable is True

    def test_int_is_not_togglable_by_default(self):
        c = Constant(1)
        assert c.is_togglable is False

    def test_togglable_override_true(self):
        c = Constant(42, togglable=True)
        assert c.is_togglable is True

    def test_togglable_override_false(self):
        c = Constant(1.0, togglable=False)
        assert c.is_togglable is False


class TestConstantDraw:
    """Tests for Constant._draw and draw."""

    def test_draw_returns_value(self):
        c = Constant(5.0)
        c.model_varname = "test_const"
        result = c._draw({}, None)
        assert result == pytest.approx(5.0)

    def test_draw_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        c = Constant(arr)
        c.model_varname = "test_const"
        result = c._draw({}, None)
        np.testing.assert_array_equal(result, arr)

    def test_draw_with_nonempty_parents_asserts(self):
        c = Constant(1.0)
        c.model_varname = "test_const"
        with pytest.raises(AssertionError):
            c._draw({"x": 1.0}, None)


class TestConstantRightSide:
    """Tests for Constant.get_right_side."""

    def test_right_side_is_empty(self):
        c = Constant(1.0)
        assert c.get_right_side(None) == ""


class TestConstantTorchParam:
    """Tests for Constant.torch_parametrization."""

    def test_torch_param_is_tensor(self):
        c = Constant(3.14)
        assert isinstance(c.torch_parametrization, torch.Tensor)

    def test_torch_param_matches_value(self):
        arr = np.array([1.0, 2.0])
        c = Constant(arr)
        np.testing.assert_allclose(c.torch_parametrization.numpy(), arr, atol=1e-10)


class TestConstantEnforceUniformity:
    """Tests for enforce_uniformity."""

    def test_uniform_array_passes(self):
        """Setting enforce_uniformity on a uniform array succeeds."""
        c = Constant(np.full(5, 3.0))
        c.enforce_uniformity = True
        assert c._enforce_uniformity is True

    def test_non_uniform_array_raises(self):
        """Setting enforce_uniformity on a non-uniform array raises."""
        c = Constant(np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="single value"):
            c.enforce_uniformity = True

    def test_setter_enables_uniformity(self):
        c = Constant(np.full(3, 2.0))
        c.enforce_uniformity = True
        assert c._enforce_uniformity is True

    def test_setter_rejects_non_uniform(self):
        c = Constant(np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="single value"):
            c.enforce_uniformity = True


class TestConstantSlider:
    """Tests for slider properties."""

    def test_slider_start_with_lower_bound(self):
        c = Constant(0.5, lower_bound=0.0)
        assert c.slider_start == 0.0

    def test_slider_end_with_upper_bound(self):
        c = Constant(0.5, upper_bound=1.0)
        assert c.slider_end == 1.0

    def test_slider_start_without_bound(self):
        c = Constant(5.0)
        start = c.slider_start
        assert start < 5.0

    def test_slider_end_without_bound(self):
        c = Constant(5.0)
        end = c.slider_end
        assert end > 5.0

    def test_slider_step_positive(self):
        c = Constant(5.0)
        assert c.slider_step_size > 0

    def test_slider_zero_value(self):
        c = Constant(0.0)
        start = c.slider_start
        end = c.slider_end
        assert start < end


class TestConstantStr:
    """Tests for Constant.__str__."""

    def test_str_contains_value(self):
        c = Constant(42.0)
        c.model_varname = "my_const"
        assert "my_const" in str(c)
        assert "42" in str(c)
