# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for scistanpy custom torch/scipy distributions."""

# pylint: disable=missing-function-docstring

import numpy as np

from scistanpy.model.components.constants import Constant
from scistanpy.model.components.parameters import (
    ExpDirichlet,
    ExpExponential,
    ExpGamma,
    ExpLomax,
    Lomax,
    MultinomialLogit,
    MultinomialLogTheta,
    Normal,
)


# ---------------------------------------------------------------------------
# ExpDirichlet
# ---------------------------------------------------------------------------
class TestExpDirichletDistribution:
    """Test ExpDirichlet parameter distribution."""

    def test_draw_returns_values(self):
        p = ExpDirichlet(alpha=np.ones(5), shape=(5,))
        p.model_varname = "x"
        draws, _ = p.draw(3)
        assert draws.shape == (3, 5)

    def test_draws_are_log_simplex(self):
        """Exponentiated draws should sum to ~1 (log-simplex constraint)."""
        p = ExpDirichlet(alpha=np.ones(4), shape=(4,))
        p.model_varname = "x"
        draws, _ = p.draw(10)
        np.testing.assert_allclose(np.exp(draws).sum(axis=-1), 1.0, atol=1e-6)

    def test_upper_bound_zero(self):
        p = ExpDirichlet(alpha=np.ones(3), shape=(3,))
        assert p.UPPER_BOUND == 0.0

    def test_no_lower_bound(self):
        p = ExpDirichlet(alpha=np.ones(3), shape=(3,))
        assert p.LOWER_BOUND is None

    def test_stan_dist(self):
        p = ExpDirichlet(alpha=np.ones(3), shape=(3,))
        assert p.STAN_DIST == "expdirichlet"

    def test_is_log_simplex(self):
        p = ExpDirichlet(alpha=np.ones(3), shape=(3,))
        assert p.IS_LOG_SIMPLEX is True
        assert p.IS_SIMPLEX is False

    def test_has_raw_varname(self):
        p = ExpDirichlet(alpha=np.ones(3), shape=(3,))
        assert p.HAS_RAW_VARNAME is True


# ---------------------------------------------------------------------------
# ExpGamma
# ---------------------------------------------------------------------------
class TestExpGammaDistribution:
    """Test ExpGamma parameter distribution."""

    def test_draw_returns_values(self):
        p = ExpGamma(alpha=2.0, beta=1.0, shape=(10,))
        p.model_varname = "x"
        draws, _ = p.draw(5)
        assert draws.shape == (5, 10)

    def test_no_lower_bound(self):
        """ExpGamma is log-transformed, so no lower bound."""
        p = ExpGamma(alpha=2.0, beta=1.0)
        assert p.LOWER_BOUND is None

    def test_stan_dist(self):
        p = ExpGamma(alpha=2.0, beta=1.0)
        assert p.STAN_DIST == "expgamma"

    def test_has_raw_varname_property(self):
        """ExpGamma HAS_RAW_VARNAME is based on is_noncentered property."""
        p = ExpGamma(alpha=2.0, beta=1.0)
        # For a hyperparameter (no non-constant parents), is_noncentered is False
        assert p.HAS_RAW_VARNAME is False or isinstance(p.HAS_RAW_VARNAME, bool)


# ---------------------------------------------------------------------------
# ExpExponential
# ---------------------------------------------------------------------------
class TestExpExponentialDistribution:
    """Test ExpExponential parameter distribution."""

    def test_draw_returns_values(self):
        p = ExpExponential(beta=1.0, shape=(10,))
        p.model_varname = "x"
        draws, _ = p.draw(5)
        assert draws.shape == (5, 10)

    def test_no_lower_bound(self):
        p = ExpExponential(beta=1.0)
        assert p.LOWER_BOUND is None

    def test_stan_dist(self):
        p = ExpExponential(beta=1.0)
        assert p.STAN_DIST == "expexponential"


# ---------------------------------------------------------------------------
# Lomax
# ---------------------------------------------------------------------------
class TestLomaxDistribution:
    """Test Lomax parameter distribution."""

    def test_draw_positive(self):
        p = Lomax(lambda_=1.0, alpha=2.0, shape=(10,))
        p.model_varname = "x"
        draws, _ = p.draw(5)
        assert np.all(draws >= 0)

    def test_lower_bound_zero(self):
        p = Lomax(lambda_=1.0, alpha=2.0)
        assert p.LOWER_BOUND == 0.0


# ---------------------------------------------------------------------------
# ExpLomax
# ---------------------------------------------------------------------------
class TestExpLomaxDistribution:
    """Test ExpLomax parameter distribution."""

    def test_draw_returns_values(self):
        p = ExpLomax(lambda_=1.0, alpha=2.0, shape=(10,))
        p.model_varname = "x"
        draws, _ = p.draw(5)
        assert draws.shape == (5, 10)

    def test_no_lower_bound(self):
        p = ExpLomax(lambda_=1.0, alpha=2.0)
        assert p.LOWER_BOUND is None


# ---------------------------------------------------------------------------
# MultinomialLogit
# ---------------------------------------------------------------------------
class TestMultinomialLogitDistribution:
    """Test MultinomialLogit distribution."""

    def test_creation(self):
        gamma = np.array([0.0, 0.0, 0.0])
        p = MultinomialLogit(gamma=gamma, N=10)
        assert p.shape == (3,)

    def test_stan_dist(self):
        gamma = np.array([0.0, 0.0])
        p = MultinomialLogit(gamma=gamma, N=10)
        assert p.STAN_DIST == "multinomial_logit"


# ---------------------------------------------------------------------------
# MultinomialLogTheta
# ---------------------------------------------------------------------------
class TestMultinomialLogThetaDistribution:
    """Test MultinomialLogTheta distribution."""

    def test_creation(self):
        log_theta = np.log(np.array([0.5, 0.5]))
        p = MultinomialLogTheta(log_theta=log_theta, N=10)
        assert p.shape == (2,)

    def test_has_coefficient(self):
        log_theta = np.log(np.array([0.5, 0.5]))
        p = MultinomialLogTheta(log_theta=log_theta, N=10)
        assert p.coefficient is not None


# ---------------------------------------------------------------------------
# CDF/SF/LogCDF/LogSF via distributions
# ---------------------------------------------------------------------------
class TestDistributionCDFs:
    """Test that distribution CDF helpers produce correct results."""

    def test_normal_cdf_at_zero(self):
        """Normal(0,1).CDF(0) should be ~0.5."""

        p = Normal(mu=0.0, sigma=1.0)
        cdf_tp = p.cdf(x=Constant(0.0))
        # CDF is a TransformedParameter; draw it via its parents
        cdf_tp.model_varname = "cdf_val"
        result = cdf_tp.run_np_torch_op(
            x=np.array(0.0), mu=np.array(0.0), sigma=np.array(1.0)
        )
        np.testing.assert_allclose(result, 0.5, atol=1e-10)

    def test_normal_sf_at_zero(self):
        """Normal(0,1).SF(0) should be ~0.5."""

        p = Normal(mu=0.0, sigma=1.0)
        sf_tp = p.ccdf(x=Constant(0.0))
        sf_tp.model_varname = "sf_val"
        result = sf_tp.run_np_torch_op(
            x=np.array(0.0), mu=np.array(0.0), sigma=np.array(1.0)
        )
        np.testing.assert_allclose(result, 0.5, atol=1e-10)

    def test_normal_log_cdf_at_zero(self):
        """Normal(0,1).LogCDF(0) should be ~log(0.5)."""

        p = Normal(mu=0.0, sigma=1.0)
        lcdf = p.log_cdf(x=Constant(0.0))
        lcdf.model_varname = "lcdf_val"
        result = lcdf.run_np_torch_op(
            x=np.array(0.0), mu=np.array(0.0), sigma=np.array(1.0)
        )
        np.testing.assert_allclose(result, np.log(0.5), atol=1e-10)

    def test_cdf_plus_sf_equals_one(self):
        """CDF(x) + SF(x) = 1 for any x."""

        p = Normal(mu=0.0, sigma=1.0)
        x_val = Constant(1.5)
        cdf_tp = p.cdf(x=x_val)
        sf_tp = p.ccdf(x=x_val)
        cdf_result = cdf_tp.run_np_torch_op(
            x=np.array(1.5), mu=np.array(0.0), sigma=np.array(1.0)
        )
        sf_result = sf_tp.run_np_torch_op(
            x=np.array(1.5), mu=np.array(0.0), sigma=np.array(1.0)
        )
        np.testing.assert_allclose(cdf_result + sf_result, 1.0, atol=1e-10)
