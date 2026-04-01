# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for scistanpy.operations module."""

# pylint: disable=missing-function-docstring, missing-class-docstring

import numpy as np
import pytest

import scistanpy as ssp
from scistanpy import operations
from scistanpy.model.components.transformations import transformed_parameters


# ---------------------------------------------------------------------------
# MetaOperation / build_operation
# ---------------------------------------------------------------------------
class TestBuildOperation:
    """Tests for MetaOperation metaclass and build_operation factory."""

    def test_build_operation_returns_operation_instance(self):
        op = operations.build_operation(transformed_parameters.ExpParameter)
        assert isinstance(op, operations.Operation)

    def test_missing_distclass_raises(self):
        with pytest.raises(ValueError, match="DISTCLASS"):
            operations.MetaOperation("Bad", (operations.Operation,), {})

    def test_wrong_distclass_type_raises(self):
        class NotATransformedParam:
            pass

        with pytest.raises(TypeError, match="subclass of TransformedParameter"):
            operations.MetaOperation(
                "Bad",
                (operations.Operation,),
                {"DISTCLASS": NotATransformedParam},
            )


# ---------------------------------------------------------------------------
# Operation.__call__ dispatch
# ---------------------------------------------------------------------------
class TestOperationDispatch:
    """Tests for Operation dispatching between model components and raw data."""

    def test_call_with_numpy_returns_numpy(self):
        result = operations.exp(np.array([0.0, 1.0]))
        assert isinstance(result, np.ndarray)

    def test_call_with_model_component_returns_transformed_param(self):
        param = ssp.parameters.Normal(mu=0.0, sigma=1.0)
        result = operations.exp(param)
        assert isinstance(result, transformed_parameters.TransformedParameter)

    def test_call_with_mixed_kwargs_returns_transformed_param(self):
        param = ssp.parameters.Normal(mu=0.0, sigma=1.0)
        time = ssp.Constant(np.array([0.0, 1.0, 2.0]))
        result = operations.exponential_growth(t=time, A=param, r=0.1)
        assert isinstance(result, transformed_parameters.TransformedParameter)


# ---------------------------------------------------------------------------
# Unary operations with raw numerical data
# ---------------------------------------------------------------------------
class TestUnaryOperations:
    """Tests for unary operations on raw numerical data."""

    def test_abs(self):
        result = operations.abs_(np.array([-1.0, 2.0, -3.0]))
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_exp(self):
        result = operations.exp(np.array([0.0, 1.0]))
        np.testing.assert_allclose(result, np.array([1.0, np.e]))

    def test_log(self):
        result = operations.log(np.array([1.0, np.e]))
        np.testing.assert_allclose(result, np.array([0.0, 1.0]), atol=1e-10)

    def test_sigmoid(self):
        result = operations.sigmoid(np.array([0.0]))
        np.testing.assert_allclose(result, np.array([0.5]), atol=1e-10)

    def test_log_sigmoid(self):
        result = operations.log_sigmoid(np.array([0.0]))
        np.testing.assert_allclose(result, np.array([np.log(0.5)]), atol=1e-10)

    def test_log1p_exp(self):
        result = operations.log1p_exp(np.array([0.0]))
        np.testing.assert_allclose(result, np.array([np.log(2.0)]), atol=1e-10)

    def test_normalize(self):
        result = operations.normalize(np.array([2.0, 3.0, 5.0]))
        np.testing.assert_allclose(result, np.array([0.2, 0.3, 0.5]))

    def test_normalize_2d(self):
        x = np.array([[1.0, 3.0], [2.0, 2.0]])
        result = operations.normalize(x)
        np.testing.assert_allclose(result.sum(axis=-1), [1.0, 1.0])

    def test_normalize_log(self):
        x = np.array([0.0, 0.0, 0.0])
        result = operations.normalize_log(x)
        # After normalize_log, exp(result) should sum to 1
        np.testing.assert_allclose(np.exp(result).sum(), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Reduction operations
# ---------------------------------------------------------------------------
class TestReductionOperations:
    """Tests for reduction operations on raw data."""

    def test_sum(self):
        result = operations.sum_(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result, 6.0)

    def test_sum_2d(self):
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = operations.sum_(x)
        np.testing.assert_allclose(result, np.array([6.0, 15.0]))

    def test_logsumexp(self):
        x = np.array([0.0, 0.0])
        result = operations.logsumexp(x)
        np.testing.assert_allclose(result, np.log(2.0), atol=1e-10)

    def test_sum_keepdims(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = operations.sum_(x, keepdim=True)
        assert result.shape == (2, 1)
        np.testing.assert_allclose(result, np.array([[3.0], [7.0]]))

    def test_sum_no_keepdims(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = operations.sum_(x, keepdim=False)
        assert result.shape == (2,)
        np.testing.assert_allclose(result, np.array([3.0, 7.0]))


# ---------------------------------------------------------------------------
# Growth model operations
# ---------------------------------------------------------------------------
class TestGrowthOperations:
    """Tests for growth model operations on raw data."""

    def test_exponential_growth(self):
        result = operations.exponential_growth(
            t=np.array([0.0, 1.0]),
            A=np.array([1.0, 1.0]),
            r=np.array([0.0, 0.0]),
        )
        np.testing.assert_allclose(result, np.array([1.0, 1.0]))

    def test_exponential_growth_with_rate(self):
        result = operations.exponential_growth(
            t=np.array([1.0]),
            A=np.array([2.0]),
            r=np.array([np.log(2)]),
        )
        np.testing.assert_allclose(result, np.array([4.0]), rtol=1e-6)

    def test_binary_exponential_growth(self):
        result = operations.binary_exponential_growth(
            A=np.array([1.0]),
            r=np.array([0.0]),
        )
        np.testing.assert_allclose(result, np.array([1.0]))

    def test_binary_exponential_growth_positive_rate(self):
        result = operations.binary_exponential_growth(
            A=np.array([10.0]),
            r=np.array([np.log(2)]),
        )
        np.testing.assert_allclose(result, np.array([20.0]), rtol=1e-6)


# ---------------------------------------------------------------------------
# Operations with model components (deferred computation)
# ---------------------------------------------------------------------------
class TestOperationsWithComponents:
    """Tests for operations that create TransformedParameter instances."""

    def test_exp_with_parameter(self):
        param = ssp.parameters.Normal(mu=0.0, sigma=1.0)
        result = operations.exp(param)
        assert isinstance(result, transformed_parameters.ExpParameter)

    def test_log_with_parameter(self):
        param = ssp.parameters.LogNormal(mu=0.0, sigma=1.0)
        result = operations.log(param)
        assert isinstance(result, transformed_parameters.LogParameter)

    def test_sum_with_parameter(self):
        param = ssp.parameters.Normal(mu=0.0, sigma=1.0, shape=(5,))
        result = operations.sum_(param)
        assert isinstance(result, transformed_parameters.SumParameter)
        assert result.shape == ()

    def test_sum_keepdims_with_parameter(self):
        param = ssp.parameters.Normal(mu=0.0, sigma=1.0, shape=(5,))
        result = operations.sum_(param, keepdims=True)
        assert isinstance(result, transformed_parameters.SumParameter)
        assert result.shape == (1,)

    def test_normalize_with_parameter(self):
        param = ssp.parameters.Exponential(beta=1.0, shape=(5,))
        result = operations.normalize(param)
        assert isinstance(result, transformed_parameters.NormalizeParameter)
        assert result.shape == (5,)


# ---------------------------------------------------------------------------
# Module-level operation instances are proper types
# ---------------------------------------------------------------------------
class TestOperationInstances:
    """Verify module-level operation instances exist and are Operations."""

    @pytest.mark.parametrize(
        "name",
        [
            "abs_",
            "binary_exponential_growth",
            "binary_log_exponential_growth",
            "exp",
            "log",
            "log1p_exp",
            "log_sigmoid",
            "logsumexp",
            "normalize",
            "normalize_log",
            "sigmoid",
            "sum_",
            "exponential_growth",
        ],
    )
    def test_operation_exists(self, name):
        op = getattr(operations, name)
        assert isinstance(op, operations.Operation)
