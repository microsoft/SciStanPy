# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for scistanpy.model.results module (MLE results)."""

# pylint: disable=missing-function-docstring, missing-class-docstring

import numpy as np

from scistanpy.model.components.parameters import HalfNormal, Normal
from scistanpy.model.model import Model


# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------
class SimpleModel(Model):
    def __init__(self):
        super().__init__()
        self.mu = Normal(mu=0.0, sigma=1.0)
        self.y = Normal(mu=self.mu, sigma=1.0, shape=(5,))


class HierarchicalModel(Model):
    def __init__(self):
        super().__init__()
        self.mu = Normal(mu=0.0, sigma=1.0)
        self.sigma = HalfNormal(sigma=1.0)
        self.y = Normal(mu=self.mu, sigma=self.sigma, shape=(10,))


# ---------------------------------------------------------------------------
# MLE results
# ---------------------------------------------------------------------------
class TestMLEResults:
    """Test MLE result object."""

    def test_mle_has_loss_trajectory(self):
        m = SimpleModel()
        data = {"y": np.random.randn(5)}
        result = m.mle(data=data, epochs=50, seed=0)
        assert hasattr(result, "losses")
        assert len(result.losses) > 0

    def test_mle_has_parameter_attributes(self):
        m = SimpleModel()
        data = {"y": np.random.randn(5)}
        result = m.mle(data=data, epochs=50, seed=0)
        assert hasattr(result, "mu")

    def test_mle_param_has_mle_estimate(self):
        m = SimpleModel()
        data = {"y": np.random.randn(5)}
        result = m.mle(data=data, epochs=50, seed=0)
        param = result.mu  # pylint: disable=no-member
        assert param.mle is not None

    def test_mle_param_draw(self):
        m = SimpleModel()
        data = {"y": np.random.randn(5)}
        result = m.mle(data=data, epochs=50, seed=0)
        param = result.mu  # pylint: disable=no-member
        draws = param.draw(10)
        assert draws.shape[0] == 10

    def test_mle_draw_all(self):
        m = SimpleModel()
        data = {"y": np.random.randn(5)}
        result = m.mle(data=data, epochs=50, seed=0)
        all_draws = result.draw(10)
        assert isinstance(all_draws, dict)
        assert "mu" in all_draws


# ---------------------------------------------------------------------------
# MLE with simulate
# ---------------------------------------------------------------------------
class TestMLESimulate:
    """Test MLE via simulation."""

    def test_simulate_mle_converges_on_simple_model(self):
        """Simulated data from the model should allow recovery."""
        m = SimpleModel()
        sim_data, result = m.simulate_mle(epochs=500, seed=42)
        # sim_data contains only observables, not latent parameters
        assert "y" in sim_data
        estimated_mu = result.mu.mle.item()  # pylint: disable=no-member
        # Should be a reasonable value (finite)
        assert np.isfinite(estimated_mu)

    def test_simulate_mle_returns_sim_data_with_obs(self):
        m = SimpleModel()
        sim_data, _ = m.simulate_mle(epochs=50, seed=0)
        assert "y" in sim_data


# ---------------------------------------------------------------------------
# MLE inference object
# ---------------------------------------------------------------------------
class TestMLEInference:
    """Test MLE inference object construction."""

    def test_get_inference_obj(self):
        m = SimpleModel()
        data = {"y": np.random.randn(5)}
        result = m.mle(data=data, epochs=100, seed=0)
        inf_obj = result.get_inference_obj(n=50, seed=0)
        assert inf_obj is not None
        assert inf_obj.inference_obj is not None

    def test_inference_obj_has_posterior(self):
        m = SimpleModel()
        data = {"y": np.random.randn(5)}
        result = m.mle(data=data, epochs=100, seed=0)
        inf_obj = result.get_inference_obj(n=50, seed=0)
        assert hasattr(inf_obj.inference_obj, "posterior")
