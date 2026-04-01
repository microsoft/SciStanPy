# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for scistanpy.model.components.parameters module."""

# pylint: disable=missing-function-docstring, protected-access

import numpy as np
import pytest
import torch

from scistanpy.model.components import parameters
from scistanpy.model.components.constants import Constant
from scistanpy.model.components.transformations import cdfs, transformed_parameters


# ---------------------------------------------------------------------------
# Parameter base class
# ---------------------------------------------------------------------------
class TestParameterBase:
    """Tests for Parameter base class and metaclass behavior."""

    def test_parameter_has_cdf_classes(self):
        """ParameterMeta should auto-create CDF, SF, LOG_CDF, LOG_SF."""
        assert hasattr(parameters.Normal, "CDF")
        assert hasattr(parameters.Normal, "SF")
        assert hasattr(parameters.Normal, "LOG_CDF")
        assert hasattr(parameters.Normal, "LOG_SF")

    def test_cdf_class_inherits_correctly(self):
        assert issubclass(parameters.Normal.CDF, cdfs.CDF)
        assert issubclass(parameters.Normal.SF, cdfs.SurvivalFunction)

    def test_missing_distribution_params_raises(self):
        # pylint: disable=missing-kwoa
        with pytest.raises(TypeError):
            parameters.Normal(mu=0.0)  # Missing sigma

    def test_observable_property_leaf_parameter(self):
        """A parameter with no children should be observable."""
        p = parameters.Normal(mu=0.0, sigma=1.0)
        assert p.observable is True

    def test_non_observable_with_child(self):
        """A parameter with children should not be observable."""
        mu = parameters.Normal(mu=0.0, sigma=1.0)
        _y = parameters.Normal(mu=mu, sigma=1.0)
        assert mu.observable is False

    def test_as_observable_marks_parameter(self):
        mu = parameters.Normal(mu=0.0, sigma=1.0)
        _y = parameters.Normal(mu=mu, sigma=1.0)
        mu.as_observable()
        assert mu._observable is True


# ---------------------------------------------------------------------------
# Normal distribution
# ---------------------------------------------------------------------------
class TestNormal:
    """Tests for Normal distribution parameter."""

    def test_basic_creation(self):
        p = parameters.Normal(mu=0.0, sigma=1.0)
        assert p.shape == ()

    def test_shape_parameter(self):
        p = parameters.Normal(mu=0.0, sigma=1.0, shape=(5, 3))
        assert p.shape == (5, 3)

    def test_stan_dist_name(self):
        p = parameters.Normal(mu=0.0, sigma=1.0)
        assert p.STAN_DIST == "normal"

    def test_draw_returns_float(self):
        p = parameters.Normal(mu=0.0, sigma=1.0)
        p.model_varname = "test"
        draws, _ = p.draw(10)
        assert draws.shape == (10,)

    def test_draw_shaped(self):
        p = parameters.Normal(mu=0.0, sigma=1.0, shape=(3,))
        p.model_varname = "test"
        draws, _ = p.draw(5)
        assert draws.shape == (5, 3)

    def test_draw_reproducible_with_seed(self):
        p = parameters.Normal(mu=0.0, sigma=1.0)
        p.model_varname = "test"
        d1, _ = p.draw(10, seed=42)
        d2, _ = p.draw(10, seed=42)
        np.testing.assert_array_equal(d1, d2)

    def test_noncentered_default_true(self):
        p = parameters.Normal(mu=0.0, sigma=1.0)
        assert p._noncentered is True

    def test_noncentered_hyperparameter_returns_false(self):
        """Non-centered should be False for hyperparameters."""
        p = parameters.Normal(mu=0.0, sigma=1.0)
        assert p.is_noncentered is False  # Hyperparameter has only constants

    def test_noncentered_hierarchical(self):
        """is_noncentered requires: _noncentered=True AND not hyperparameter AND not observable."""
        mu = parameters.Normal(mu=0.0, sigma=1.0)
        child = parameters.Normal(mu=mu, sigma=1.0)
        # mu is a hyperparameter (only constant parents) -> is_noncentered False
        assert mu.is_noncentered is False
        # child is observable (leaf param, no children) -> is_noncentered False
        assert child.is_noncentered is False

    def test_centered_flag(self):
        p = parameters.Normal(mu=0.0, sigma=1.0, noncentered=False)
        assert p._noncentered is False


# ---------------------------------------------------------------------------
# HalfNormal distribution
# ---------------------------------------------------------------------------
class TestHalfNormal:
    """Tests for HalfNormal distribution."""

    def test_lower_bound_zero(self):
        p = parameters.HalfNormal(sigma=1.0)
        assert p.LOWER_BOUND == 0.0

    def test_draw_positive(self):
        p = parameters.HalfNormal(sigma=1.0, shape=(10,))
        p.model_varname = "test"
        draws, _ = p.draw(20)
        assert np.all(draws >= 0)

    def test_stan_dist_is_normal(self):
        """HalfNormal uses Stan's normal distribution with lower bound."""
        p = parameters.HalfNormal(sigma=1.0)
        assert p.STAN_DIST == "normal"


# ---------------------------------------------------------------------------
# UnitNormal distribution
# ---------------------------------------------------------------------------
class TestUnitNormal:
    """Tests for UnitNormal (standard normal)."""

    def test_creation_no_args(self):
        p = parameters.UnitNormal()
        assert p.shape == ()

    def test_stan_dist(self):
        p = parameters.UnitNormal()
        assert p.STAN_DIST == "std_normal"

    def test_draw_centered_around_zero(self):
        p = parameters.UnitNormal(shape=(100,))
        p.model_varname = "test"
        draws, _ = p.draw(1, seed=42)
        # Mean should be close to 0 for large enough shape
        assert abs(draws.mean()) < 1.0


# ---------------------------------------------------------------------------
# LogNormal distribution
# ---------------------------------------------------------------------------
class TestLogNormal:
    """Tests for LogNormal distribution."""

    def test_lower_bound_positive(self):
        p = parameters.LogNormal(mu=0.0, sigma=1.0)
        assert p.LOWER_BOUND == 0.0

    def test_draw_positive(self):
        p = parameters.LogNormal(mu=0.0, sigma=0.5, shape=(10,))
        p.model_varname = "test"
        draws, _ = p.draw(5)
        assert np.all(draws > 0)

    def test_stan_dist_name(self):
        p = parameters.LogNormal(mu=0.0, sigma=1.0)
        assert p.STAN_DIST == "lognormal"


# ---------------------------------------------------------------------------
# Beta distribution
# ---------------------------------------------------------------------------
class TestBeta:
    """Tests for Beta distribution."""

    def test_bounds(self):
        p = parameters.Beta(alpha=2.0, beta=2.0)
        assert p.LOWER_BOUND == 0.0
        assert p.UPPER_BOUND == 1.0

    def test_draw_in_range(self):
        p = parameters.Beta(alpha=2.0, beta=5.0, shape=(10,))
        p.model_varname = "test"
        draws, _ = p.draw(5)
        assert np.all(draws >= 0) and np.all(draws <= 1)


# ---------------------------------------------------------------------------
# Gamma distribution
# ---------------------------------------------------------------------------
class TestGamma:
    """Tests for Gamma distribution."""

    def test_lower_bound(self):
        p = parameters.Gamma(alpha=2.0, beta=1.0)
        assert p.LOWER_BOUND == 0.0

    def test_draw_positive(self):
        p = parameters.Gamma(alpha=2.0, beta=1.0, shape=(10,))
        p.model_varname = "test"
        draws, _ = p.draw(5)
        assert np.all(draws > 0)


# ---------------------------------------------------------------------------
# InverseGamma distribution
# ---------------------------------------------------------------------------
class TestInverseGamma:
    """Tests for InverseGamma distribution."""

    def test_lower_bound(self):
        p = parameters.InverseGamma(alpha=3.0, beta=1.0)
        assert p.LOWER_BOUND == 0.0

    def test_draw_positive(self):
        p = parameters.InverseGamma(alpha=3.0, beta=1.0, shape=(10,))
        p.model_varname = "test"
        draws, _ = p.draw(5)
        assert np.all(draws > 0)


# ---------------------------------------------------------------------------
# Exponential distribution
# ---------------------------------------------------------------------------
class TestExponential:
    """Tests for Exponential distribution."""

    def test_lower_bound(self):
        p = parameters.Exponential(beta=1.0)
        assert p.LOWER_BOUND == 0.0

    def test_draw_positive(self):
        p = parameters.Exponential(beta=2.0, shape=(10,))
        p.model_varname = "test"
        draws, _ = p.draw(5)
        assert np.all(draws >= 0)


# ---------------------------------------------------------------------------
# Dirichlet distribution
# ---------------------------------------------------------------------------
class TestDirichlet:
    """Tests for Dirichlet distribution."""

    def test_simplex_constraint(self):
        p = parameters.Dirichlet(alpha=1.0, shape=(4,))
        assert p.IS_SIMPLEX is True

    def test_draw_sums_to_one(self):
        p = parameters.Dirichlet(alpha=1.0, shape=(4,))
        p.model_varname = "test"
        draws, _ = p.draw(5)
        np.testing.assert_allclose(draws.sum(axis=-1), 1.0, atol=1e-10)

    def test_scalar_alpha_requires_shape(self):
        with pytest.raises(ValueError, match="shape must be provided"):
            parameters.Dirichlet(alpha=1.0)

    def test_array_alpha(self):
        alpha = np.array([1.0, 2.0, 3.0])
        p = parameters.Dirichlet(alpha=alpha)
        assert p.shape == (3,)


# ---------------------------------------------------------------------------
# ExpDirichlet distribution
# ---------------------------------------------------------------------------
class TestExpDirichlet:
    """Tests for ExpDirichlet distribution."""

    def test_log_simplex_constraint(self):
        p = parameters.ExpDirichlet(alpha=1.0, shape=(4,))
        assert p.IS_LOG_SIMPLEX is True

    def test_draw_upper_bound_zero(self):
        p = parameters.ExpDirichlet(alpha=1.0, shape=(4,))
        p.model_varname = "test"
        draws, _ = p.draw(5)
        assert np.all(draws <= 0)

    def test_draw_exp_sums_to_one(self):
        p = parameters.ExpDirichlet(alpha=1.0, shape=(4,))
        p.model_varname = "test"
        draws, _ = p.draw(5)
        np.testing.assert_allclose(np.exp(draws).sum(axis=-1), 1.0, atol=1e-6)

    def test_has_raw_varname(self):
        p = parameters.ExpDirichlet(alpha=1.0, shape=(4,))
        assert p.HAS_RAW_VARNAME is True


# ---------------------------------------------------------------------------
# Discrete distributions
# ---------------------------------------------------------------------------
class TestBinomial:
    """Tests for Binomial distribution."""

    def test_lower_bound(self):
        p = parameters.Binomial(N=10, theta=0.5)
        assert p.LOWER_BOUND == 0

    def test_draw_in_range(self):
        p = parameters.Binomial(N=10, theta=0.5, shape=(5,))
        p.model_varname = "test"
        draws, _ = p.draw(3)
        assert np.all(draws >= 0) and np.all(draws <= 10)

    def test_base_stan_dtype_int(self):
        p = parameters.Binomial(N=10, theta=0.5)
        assert p.BASE_STAN_DTYPE == "int"


class TestPoisson:
    """Tests for Poisson distribution."""

    def test_lower_bound(self):
        p = parameters.Poisson(lambda_=5.0)
        assert p.LOWER_BOUND == 0

    def test_draw_non_negative(self):
        p = parameters.Poisson(lambda_=3.0, shape=(10,))
        p.model_varname = "test"
        draws, _ = p.draw(5)
        assert np.all(draws >= 0)


class TestMultinomial:
    """Tests for Multinomial distribution."""

    def test_simplex_params(self):
        assert "theta" in parameters.Multinomial.SIMPLEX_PARAMS

    def test_stan_dist(self):
        p = parameters.Multinomial(theta=np.array([0.5, 0.5]), N=10)
        assert p.STAN_DIST == "multinomial"


class TestMultinomialLogit:
    """Tests for MultinomialLogit distribution."""

    def test_stan_dist(self):
        p = parameters.MultinomialLogit(gamma=np.array([0.0, 0.0]), N=10)
        assert p.STAN_DIST == "multinomial_logit"


class TestMultinomialLogTheta:
    """Tests for MultinomialLogTheta distribution."""

    def test_stan_dist(self):
        log_theta = np.log(np.array([0.5, 0.5]))
        p = parameters.MultinomialLogTheta(log_theta=log_theta, N=10)
        assert p.STAN_DIST == "multinomial_logtheta"

    def test_coefficient_created(self):
        log_theta = np.log(np.array([0.5, 0.5]))
        p = parameters.MultinomialLogTheta(log_theta=log_theta, N=10)
        assert p.coefficient is not None

    def test_supporting_functions_includes_multinomial(self):
        log_theta = np.log(np.array([0.5, 0.5]))
        p = parameters.MultinomialLogTheta(log_theta=log_theta, N=10)
        funcs = p.get_supporting_functions()
        assert any("multinomial" in f for f in funcs)


# ---------------------------------------------------------------------------
# ExpGamma, ExpExponential, Lomax, ExpLomax
# ---------------------------------------------------------------------------
class TestExpDistributions:
    """Tests for Exp- variant distributions."""

    def test_expgamma_no_lower_bound(self):
        p = parameters.ExpGamma(alpha=2.0, beta=1.0)
        assert p.LOWER_BOUND is None

    def test_expexponential_no_lower_bound(self):
        p = parameters.ExpExponential(beta=1.0)
        assert p.LOWER_BOUND is None

    def test_lomax_lower_bound(self):
        p = parameters.Lomax(lambda_=1.0, alpha=2.0)
        assert p.LOWER_BOUND == 0.0

    def test_explomax_no_lower_bound(self):
        p = parameters.ExpLomax(lambda_=1.0, alpha=2.0)
        assert p.LOWER_BOUND is None

    def test_expgamma_supporting_functions(self):
        p = parameters.ExpGamma(alpha=2.0, beta=1.0)
        funcs = p.get_supporting_functions()
        assert any("expgamma" in f for f in funcs)

    def test_expexponential_supporting_functions(self):
        p = parameters.ExpExponential(beta=1.0)
        funcs = p.get_supporting_functions()
        assert any("expexponential" in f for f in funcs)


# ---------------------------------------------------------------------------
# Parameter PyTorch integration
# ---------------------------------------------------------------------------
class TestParameterTorch:
    """Tests for Parameter PyTorch integration."""

    def test_init_pytorch(self):
        p = parameters.Normal(mu=0.0, sigma=1.0)
        _child = parameters.Normal(mu=p, sigma=1.0)  # Make p non-observable
        p.init_pytorch(seed=42)
        assert p._torch_parametrization is not None

    def test_init_pytorch_observable_raises(self):
        p = parameters.Normal(mu=0.0, sigma=1.0)
        # p is a leaf => observable
        with pytest.raises(ValueError, match="Observables"):
            p.init_pytorch()

    def test_init_pytorch_shape_match(self):
        p = parameters.Normal(mu=0.0, sigma=1.0, shape=(3, 4))
        _child = parameters.Normal(mu=p, sigma=1.0)
        p.init_pytorch(seed=42)
        assert p._torch_parametrization.shape == (3, 4)

    def test_init_pytorch_wrong_shape_raises(self):
        p = parameters.Normal(mu=0.0, sigma=1.0, shape=(3,))
        _child = parameters.Normal(mu=p, sigma=1.0)
        with pytest.raises(ValueError, match="shape"):
            p.init_pytorch(init_val=np.zeros((5,)))

    def test_torch_parametrization_unbounded(self):
        """Unbounded parameters return raw parametrization."""
        p = parameters.Normal(mu=0.0, sigma=1.0)
        _child = parameters.Normal(mu=p, sigma=1.0)
        p.init_pytorch(seed=0)
        result = p.torch_parametrization
        assert isinstance(result, torch.Tensor)

    def test_torch_parametrization_lower_bounded(self):
        """Lower-bounded parameters (e.g., HalfNormal) apply exp."""
        p = parameters.HalfNormal(sigma=1.0)
        _child = parameters.Normal(mu=p, sigma=1.0)
        p.init_pytorch(seed=0)
        result = p.torch_parametrization
        assert torch.all(result >= 0)

    def test_torch_parametrization_double_bounded(self):
        """Double-bounded parameters (e.g., Beta) use sigmoid."""
        p = parameters.Beta(alpha=2.0, beta=2.0)
        _child = parameters.Normal(mu=p, sigma=1.0)
        p.init_pytorch(seed=0)
        result = p.torch_parametrization
        assert torch.all(result >= 0) and torch.all(result <= 1)

    def test_torch_parametrization_simplex(self):
        """Simplex parameters use softmax."""
        p = parameters.Dirichlet(alpha=1.0, shape=(4,))
        _child = parameters.Multinomial(theta=p, N=10)
        p.init_pytorch(seed=0)
        result = p.torch_parametrization
        assert torch.allclose(
            result.sum(dim=-1),
            torch.tensor(1.0, dtype=result.dtype),
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Operator overloading / TransformableParameter
# ---------------------------------------------------------------------------
class TestOperatorOverloading:
    """Tests for operator overloading on ContinuousDistribution."""

    def test_add(self):
        a = parameters.Normal(mu=0.0, sigma=1.0)
        b = parameters.Normal(mu=0.0, sigma=1.0)
        result = a + b
        assert isinstance(result, transformed_parameters.AddParameter)

    def test_radd(self):
        a = parameters.Normal(mu=0.0, sigma=1.0)
        result = 1.0 + a
        assert isinstance(result, transformed_parameters.AddParameter)

    def test_sub(self):
        a = parameters.Normal(mu=0.0, sigma=1.0)
        b = parameters.Normal(mu=0.0, sigma=1.0)
        result = a - b
        assert isinstance(result, transformed_parameters.SubtractParameter)

    def test_rsub(self):
        a = parameters.Normal(mu=0.0, sigma=1.0)
        result = 1.0 - a
        assert isinstance(result, transformed_parameters.SubtractParameter)

    def test_mul(self):
        a = parameters.Normal(mu=0.0, sigma=1.0)
        result = a * 2.0
        assert isinstance(result, transformed_parameters.MultiplyParameter)

    def test_rmul(self):
        a = parameters.Normal(mu=0.0, sigma=1.0)
        result = 2.0 * a
        assert isinstance(result, transformed_parameters.MultiplyParameter)

    def test_truediv(self):
        a = parameters.Normal(mu=0.0, sigma=1.0)
        result = a / 2.0
        assert isinstance(result, transformed_parameters.DivideParameter)

    def test_rtruediv(self):
        a = parameters.Normal(mu=0.0, sigma=1.0)
        result = 1.0 / a
        assert isinstance(result, transformed_parameters.DivideParameter)

    def test_pow(self):
        a = parameters.Normal(mu=0.0, sigma=1.0)
        result = a**2
        assert isinstance(result, transformed_parameters.PowerParameter)

    def test_rpow(self):
        a = parameters.Normal(mu=0.0, sigma=1.0)
        result = 2.0**a
        assert isinstance(result, transformed_parameters.PowerParameter)

    def test_neg(self):
        a = parameters.Normal(mu=0.0, sigma=1.0)
        result = -a
        assert isinstance(result, transformed_parameters.NegateParameter)


# ---------------------------------------------------------------------------
# CDF-like methods
# ---------------------------------------------------------------------------
class TestCDFMethods:
    """Tests for CDF convenience methods."""

    def test_cdf_instance_method(self):
        p = parameters.Normal(mu=0.0, sigma=1.0)
        result = p.cdf(x=Constant(0.0))
        assert isinstance(result, cdfs.CDF)

    def test_ccdf_instance_method(self):
        p = parameters.Normal(mu=0.0, sigma=1.0)
        result = p.ccdf(x=Constant(0.0))
        assert isinstance(result, cdfs.SurvivalFunction)

    def test_log_cdf_instance_method(self):
        p = parameters.Normal(mu=0.0, sigma=1.0)
        result = p.log_cdf(x=Constant(0.0))
        assert isinstance(result, cdfs.LogCDF)

    def test_log_ccdf_instance_method(self):
        p = parameters.Normal(mu=0.0, sigma=1.0)
        result = p.log_ccdf(x=Constant(0.0))
        assert isinstance(result, cdfs.LogSurvivalFunction)

    def test_cdf_class_method(self):
        # pylint: disable=no-value-for-parameter
        result = parameters.Normal.cdf(mu=0.0, sigma=1.0, x=Constant(0.0))
        assert isinstance(result, cdfs.CDF)

    def test_class_method_missing_params_raises(self):
        # pylint: disable=no-value-for-parameter
        with pytest.raises(ValueError, match="must be provided"):
            parameters.Normal.cdf(mu=0.0, x=Constant(0.0))  # Missing sigma

    def test_unexpected_kwargs_raises(self):
        # pylint: disable=no-value-for-parameter
        with pytest.raises(ValueError, match="Unexpected"):
            parameters.Normal.cdf(mu=0.0, sigma=1.0, x=Constant(0.0), foo=42)

    def test_missing_x_raises(self):
        # pylint: disable=no-value-for-parameter
        with pytest.raises(ValueError, match="Expected `x`"):
            parameters.Normal.cdf(mu=0.0, sigma=1.0)  # Missing x


# ---------------------------------------------------------------------------
# Parameter string representation
# ---------------------------------------------------------------------------
class TestParameterStr:
    """Tests for Parameter.__str__."""

    def test_str_contains_distribution_name(self):
        p = parameters.Normal(mu=0.0, sigma=1.0)
        p.model_varname = "my_param"
        s = str(p)
        assert "my_param" in s
        assert "Normal" in s or "normal" in s.lower()
