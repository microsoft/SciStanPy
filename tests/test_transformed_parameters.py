# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for scistanpy.model.components.transformations.transformed_parameters module."""
# pylint: disable=missing-function-docstring, invalid-name

import numpy as np
import torch
from scipy.special import log_expit, logsumexp  # pylint: disable=no-name-in-module

from scistanpy.model.components.constants import Constant
from scistanpy.model.components.parameters import Normal
from scistanpy.model.components.transformations.transformed_parameters import (
    AbsParameter,
    AddParameter,
    BinaryExponentialGrowth,
    DivideParameter,
    ExponentialGrowth,
    ExpParameter,
    Log1pExpParameter,
    LogParameter,
    LogSigmoidGrowth,
    LogSigmoidParameter,
    LogSumExpParameter,
    MultiplyParameter,
    NegateParameter,
    NormalizeLogParameter,
    NormalizeParameter,
    PowerParameter,
    SigmoidGrowth,
    SigmoidParameter,
    SubtractParameter,
    SumParameter,
)
from scistanpy.utils import stable_sigmoid


# ---------------------------------------------------------------------------
# Binary operations (run_np_torch_op)
# ---------------------------------------------------------------------------
class TestBinaryOperations:
    """Test binary transformed parameter operations."""

    def test_add_numpy(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        op = AddParameter(Constant(np.zeros(2)), Constant(np.zeros(2)))
        result = op.run_np_torch_op(dist1=a, dist2=b)
        np.testing.assert_array_equal(result, a + b)

    def test_add_torch(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        op = AddParameter(Constant(np.zeros(2)), Constant(np.zeros(2)))
        result = op.run_np_torch_op(dist1=a, dist2=b)
        assert torch.allclose(result, a + b)

    def test_subtract_numpy(self):
        a = np.array([5.0, 6.0])
        b = np.array([1.0, 2.0])
        op = SubtractParameter(Constant(np.zeros(2)), Constant(np.zeros(2)))
        result = op.run_np_torch_op(dist1=a, dist2=b)
        np.testing.assert_array_equal(result, a - b)

    def test_multiply_numpy(self):
        a = np.array([2.0, 3.0])
        b = np.array([4.0, 5.0])
        op = MultiplyParameter(Constant(np.zeros(2)), Constant(np.zeros(2)))
        result = op.run_np_torch_op(dist1=a, dist2=b)
        np.testing.assert_array_equal(result, a * b)

    def test_divide_numpy(self):
        a = np.array([10.0, 20.0])
        b = np.array([2.0, 4.0])
        op = DivideParameter(Constant(np.zeros(2)), Constant(np.zeros(2)))
        result = op.run_np_torch_op(dist1=a, dist2=b)
        np.testing.assert_array_equal(result, a / b)

    def test_power_numpy(self):
        a = np.array([2.0, 3.0])
        b = np.array([3.0, 2.0])
        op = PowerParameter(Constant(np.zeros(2)), Constant(np.zeros(2)))
        result = op.run_np_torch_op(dist1=a, dist2=b)
        np.testing.assert_array_equal(result, a**b)


# ---------------------------------------------------------------------------
# Unary operations
# ---------------------------------------------------------------------------
class TestUnaryOperations:
    """Test unary transformed parameter operations."""

    def test_negate_numpy(self):
        a = np.array([1.0, -2.0])
        op = NegateParameter(Constant(np.zeros(2)))
        result = op.run_np_torch_op(dist1=a)
        np.testing.assert_array_equal(result, -a)

    def test_abs_numpy(self):
        a = np.array([-3.0, 2.0, -1.0])
        op = AbsParameter(Constant(np.zeros(3)))
        result = op.run_np_torch_op(dist1=a)
        np.testing.assert_array_equal(result, np.abs(a))

    def test_abs_lower_bound(self):
        op = AbsParameter(Constant(np.zeros(2)))
        assert op.LOWER_BOUND == 0.0

    def test_log_numpy(self):
        a = np.array([1.0, 2.0, np.e])
        op = LogParameter(Constant(np.ones(3)))
        result = op.run_np_torch_op(dist1=a)
        np.testing.assert_allclose(result, np.log(a))

    def test_exp_numpy(self):
        a = np.array([0.0, 1.0, 2.0])
        op = ExpParameter(Constant(np.zeros(3)))
        result = op.run_np_torch_op(dist1=a)
        np.testing.assert_allclose(result, np.exp(a))

    def test_exp_lower_bound(self):
        op = ExpParameter(Constant(np.zeros(2)))
        assert op.LOWER_BOUND == 0.0

    def test_log1p_exp_numpy(self):
        a = np.array([0.0, 1.0, -1.0])
        op = Log1pExpParameter(Constant(np.zeros(3)))
        result = op.run_np_torch_op(dist1=a)
        expected = np.logaddexp(0, a)
        np.testing.assert_allclose(result, expected)

    def test_sigmoid_numpy(self):
        a = np.array([0.0, 5.0, -5.0])
        op = SigmoidParameter(Constant(np.zeros(3)))
        result = op.run_np_torch_op(dist1=a)
        assert np.all(result >= 0) and np.all(result <= 1)
        np.testing.assert_allclose(result[0], 0.5, atol=1e-10)

    def test_sigmoid_bounds(self):
        op = SigmoidParameter(Constant(np.zeros(2)))
        assert op.LOWER_BOUND == 0.0
        assert op.UPPER_BOUND == 1.0

    def test_log_sigmoid_numpy(self):
        a = np.array([0.0, 5.0, -5.0])
        op = LogSigmoidParameter(Constant(np.zeros(3)))
        result = op.run_np_torch_op(dist1=a)
        assert np.all(result <= 0)  # log of [0,1] is <= 0

    def test_log_sigmoid_upper_bound(self):
        op = LogSigmoidParameter(Constant(np.zeros(2)))
        assert op.UPPER_BOUND == 0.0

    def test_log1p_exp_torch(self):
        a = torch.tensor([0.0, 1.0, -1.0])
        op = Log1pExpParameter(Constant(np.zeros(3)))
        result = op.run_np_torch_op(dist1=a)
        expected = torch.nn.functional.softplus(a)  # pylint: disable=not-callable
        assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# Normalize operations
# ---------------------------------------------------------------------------
class TestNormalize:
    """Test normalize and normalize-log operations."""

    def test_normalize_sums_to_one(self):
        a = np.array([2.0, 3.0, 5.0])
        op = NormalizeParameter(Constant(np.ones(3)))
        result = op.run_np_torch_op(dist1=a)
        np.testing.assert_allclose(result.sum(), 1.0)
        np.testing.assert_allclose(result, a / a.sum())

    def test_normalize_bounds(self):
        op = NormalizeParameter(Constant(np.ones(3)))
        assert op.LOWER_BOUND == 0.0
        assert op.UPPER_BOUND == 1.0

    def test_normalize_log(self):
        """NormalizeLog: x - logsumexp(x), output in log-simplex."""
        a = np.array([1.0, 2.0, 3.0])
        op = NormalizeLogParameter(Constant(np.ones(3)))
        result = op.run_np_torch_op(dist1=a)
        # exp of result should sum to 1
        np.testing.assert_allclose(np.exp(result).sum(), 1.0, atol=1e-10)
        assert np.all(result <= 0)

    def test_normalize_log_upper_bound(self):
        op = NormalizeLogParameter(Constant(np.ones(3)))
        assert op.UPPER_BOUND == 0.0


# ---------------------------------------------------------------------------
# Reduction operations
# ---------------------------------------------------------------------------
class TestReductions:
    """Test reduction operations (Sum, LogSumExp)."""

    def test_sum_reduces_last_dim(self):
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        op = SumParameter(Constant(np.ones((2, 3))))
        result = op.run_np_torch_op(dist1=a)
        np.testing.assert_array_equal(result, np.array([6.0, 15.0]))

    def test_sum_shape_reduction(self):
        parent = Constant(np.ones((5, 3)))
        op = SumParameter(parent)
        assert op.shape == (5,)

    def test_logsumexp_reduces_last_dim(self):

        a = np.array([[1.0, 2.0, 3.0]])
        op = LogSumExpParameter(Constant(np.ones((1, 3))))
        result = op.run_np_torch_op(dist1=a)
        expected = logsumexp(a, axis=-1)
        np.testing.assert_allclose(result, expected)

    def test_logsumexp_shape(self):
        parent = Constant(np.ones((4, 5)))
        op = LogSumExpParameter(parent)
        assert op.shape == (4,)

    def test_sum_torch(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        op = SumParameter(Constant(np.ones((2, 2))))
        result = op.run_np_torch_op(dist1=a)
        expected = torch.tensor([3.0, 7.0])
        assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# Growth models
# ---------------------------------------------------------------------------
class TestGrowthModels:
    """Test growth model transformations."""

    def test_exponential_growth_formula(self):
        """A * exp(r * t) formula."""
        A = Constant(2.0)
        r = Normal(mu=0.0, sigma=1.0)
        t = Constant(np.array([0.0, 1.0, 2.0]))
        eg = ExponentialGrowth(A=A, r=r, t=t)
        # Draw r manually
        r_val = np.array(0.5)
        A_val = np.array(2.0)
        t_val = np.array([0.0, 1.0, 2.0])
        result = eg.run_np_torch_op(A=A_val, r=r_val, t=t_val)
        expected = A_val * np.exp(r_val * t_val)
        np.testing.assert_allclose(result, expected)

    def test_binary_exponential_growth_formula(self):
        """A * exp(r) formula (no time)."""
        A = Constant(3.0)
        r = Normal(mu=0.0, sigma=1.0)
        beg = BinaryExponentialGrowth(A=A, r=r)
        result = beg.run_np_torch_op(A=np.array(3.0), r=np.array(1.0))
        expected = 3.0 * np.exp(1.0)
        np.testing.assert_allclose(result, expected)

    def test_sigmoid_growth_formula(self):
        """A * sigmoid(r * (t - c)) formula."""

        A = Constant(10.0)
        r = Normal(mu=0.0, sigma=1.0)
        c = Normal(mu=0.0, sigma=1.0)
        t = Constant(np.array([0.0, 1.0, 2.0]))
        sg = SigmoidGrowth(A=A, r=r, c=c, t=t)
        A_v, r_v, c_v = np.array(10.0), np.array(0.5), np.array(1.0)
        t_v = np.array([0.0, 1.0, 2.0])
        result = sg.run_np_torch_op(A=A_v, r=r_v, c=c_v, t=t_v)
        expected = A_v * stable_sigmoid(r_v * (t_v - c_v))
        np.testing.assert_allclose(result, expected)

    def test_log_sigmoid_growth_formula(self):
        """log(A) + log_sigmoid(r * (t - c)) formula."""

        log_A = Constant(np.log(10.0))
        r = Normal(mu=0.0, sigma=1.0)
        c = Normal(mu=0.0, sigma=1.0)
        t = Constant(np.array([0.0, 1.0, 2.0]))
        lsg = LogSigmoidGrowth(log_A=log_A, r=r, c=c, t=t)
        log_A_v = np.log(10.0)
        r_v, c_v = np.array(0.5), np.array(1.0)
        t_v = np.array([0.0, 1.0, 2.0])
        result = lsg.run_np_torch_op(log_A=log_A_v, r=r_v, c=c_v, t=t_v)
        expected = log_A_v + log_expit(r_v * (t_v - c_v))
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# Operator-created transformations via Parameter overloading
# ---------------------------------------------------------------------------
class TestOperatorCreatedTransformations:
    """Test that operator overloading creates correct transformation types."""

    def test_add_creates_add_parameter(self):
        a = Normal(mu=0.0, sigma=1.0)
        b = Normal(mu=0.0, sigma=1.0)
        result = a + b
        assert isinstance(result, AddParameter)

    def test_add_constant_numeric(self):
        a = Normal(mu=0.0, sigma=1.0)
        result = a + 1.0
        assert isinstance(result, AddParameter)

    def test_sub_creates_subtract(self):
        a = Normal(mu=0.0, sigma=1.0)
        result = a - 1.0
        assert isinstance(result, SubtractParameter)

    def test_mul_creates_multiply(self):
        a = Normal(mu=0.0, sigma=1.0)
        result = a * 2.0
        assert isinstance(result, MultiplyParameter)

    def test_div_creates_divide(self):
        a = Normal(mu=0.0, sigma=1.0)
        result = a / 2.0
        assert isinstance(result, DivideParameter)

    def test_pow_creates_power(self):
        a = Normal(mu=0.0, sigma=1.0)
        result = a**2
        assert isinstance(result, PowerParameter)

    def test_neg_creates_negate(self):
        a = Normal(mu=0.0, sigma=1.0)
        result = -a
        assert isinstance(result, NegateParameter)


# ---------------------------------------------------------------------------
# Stan operation strings
# ---------------------------------------------------------------------------
class TestStanOperations:
    """Test Stan code generation for transformations."""

    def test_add_stan_op(self):
        op = AddParameter(Constant(np.zeros(2)), Constant(np.zeros(2)))
        result = op.write_stan_operation(dist1="a", dist2="b")
        assert "+" in result

    def test_subtract_stan_op(self):
        op = SubtractParameter(Constant(np.zeros(2)), Constant(np.zeros(2)))
        result = op.write_stan_operation(dist1="a", dist2="b")
        assert "-" in result

    def test_multiply_stan_op(self):
        op = MultiplyParameter(Constant(np.zeros(2)), Constant(np.zeros(2)))
        result = op.write_stan_operation(dist1="a", dist2="b")
        assert ".*" in result

    def test_divide_stan_op(self):
        op = DivideParameter(Constant(np.zeros(2)), Constant(np.zeros(2)))
        result = op.write_stan_operation(dist1="a", dist2="b")
        assert "./" in result

    def test_power_stan_op(self):
        op = PowerParameter(Constant(np.zeros(2)), Constant(np.zeros(2)))
        result = op.write_stan_operation(dist1="a", dist2="b")
        assert ".^" in result or "^" in result

    def test_exp_stan_op(self):
        op = ExpParameter(Constant(np.zeros(2)))
        result = op.write_stan_operation(dist1="x")
        assert "exp" in result

    def test_log_stan_op(self):
        op = LogParameter(Constant(np.ones(2)))
        result = op.write_stan_operation(dist1="x")
        assert "log" in result

    def test_sigmoid_stan_op(self):
        op = SigmoidParameter(Constant(np.zeros(2)))
        result = op.write_stan_operation(dist1="x")
        assert "inv_logit" in result

    def test_log_sigmoid_stan_op(self):
        op = LogSigmoidParameter(Constant(np.zeros(2)))
        result = op.write_stan_operation(dist1="x")
        assert "log_inv_logit" in result

    def test_normalize_stan_op(self):
        op = NormalizeParameter(Constant(np.ones(3)))
        result = op.write_stan_operation(dist1="x")
        assert "sum" in result

    def test_normalize_log_stan_op(self):
        op = NormalizeLogParameter(Constant(np.ones(3)))
        result = op.write_stan_operation(dist1="x")
        assert "log_sum_exp" in result


# ---------------------------------------------------------------------------
# Torch parametrization pass-through
# ---------------------------------------------------------------------------
class TestTorchParametrization:
    """Test torch_parametrization property on transformed params."""

    def test_add_torch_parametrization(self):
        a = Normal(mu=0.0, sigma=1.0)
        b = Normal(mu=0.0, sigma=1.0)
        tp = a + b
        a.init_pytorch(seed=0)
        b.init_pytorch(seed=1)
        result = tp.torch_parametrization
        expected = a.torch_parametrization + b.torch_parametrization
        assert torch.allclose(result, expected)

    def test_neg_torch_parametrization(self):
        a = Normal(mu=0.0, sigma=1.0)
        tp = -a
        a.init_pytorch(seed=0)
        result = tp.torch_parametrization
        expected = -a.torch_parametrization
        assert torch.allclose(result, expected)

    def test_exp_torch_parametrization(self):
        a = Normal(mu=0.0, sigma=1.0)
        tp = ExpParameter(a)
        a.init_pytorch(seed=0)
        result = tp.torch_parametrization
        expected = torch.exp(a.torch_parametrization)
        assert torch.allclose(result, expected)
