# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for scistanpy Model class."""

# pylint: disable=missing-function-docstring, missing-class-docstring

import numpy as np
import pytest
import torch

from scistanpy.model.components.constants import Constant
from scistanpy.model.components.parameters import HalfNormal, Normal, Poisson
from scistanpy.model.model import Model


# ---------------------------------------------------------------------------
# Helper: minimal model subclasses
# ---------------------------------------------------------------------------
class SimpleModel(Model):
    def __init__(self):
        super().__init__()
        self.mu = Normal(mu=0.0, sigma=1.0)
        self.y = Normal(mu=self.mu, sigma=1.0, shape=(5,))


class TwoObservableModel(Model):
    def __init__(self):
        super().__init__()
        self.mu = Normal(mu=0.0, sigma=1.0)
        self.sigma = HalfNormal(sigma=1.0)
        self.y1 = Normal(mu=self.mu, sigma=self.sigma, shape=(3,))
        self.y2 = Normal(mu=0.0, sigma=self.sigma, shape=(4,))


class TransformedModel(Model):
    def __init__(self):
        super().__init__()
        self.a = Normal(mu=0.0, sigma=1.0)
        self.b = Normal(mu=0.0, sigma=1.0)
        self.ab = self.a + self.b
        self.y = Normal(mu=self.ab, sigma=1.0)


class ConstantModel(Model):
    def __init__(self):
        super().__init__()
        self.offset = Constant(5.0)
        self.mu = Normal(mu=0.0, sigma=1.0)
        self.y = Normal(mu=self.mu + self.offset, sigma=1.0)


class DiscreteModel(Model):
    def __init__(self):
        super().__init__()
        self.rate = HalfNormal(sigma=5.0)
        self.counts = Poisson(lambda_=self.rate, shape=(10,))


# ---------------------------------------------------------------------------
# Model construction / __init_subclass__
# ---------------------------------------------------------------------------
class TestModelConstruction:
    """Test model auto-discovery of components."""

    def test_simple_model_construction(self):
        m = SimpleModel()
        assert "mu" in m
        assert "y" in m

    def test_parameters_property(self):
        m = SimpleModel()
        params = m.parameters
        names = [p.model_varname for p in params]
        assert "mu" in names

    def test_observables_property(self):
        m = SimpleModel()
        obs = m.observables
        names = [p.model_varname for p in obs]
        assert "y" in names

    def test_constants_property(self):
        m = ConstantModel()
        consts = m.constants
        names = [c.model_varname for c in consts]
        assert "offset" in names

    def test_transformed_parameters_property(self):
        m = TransformedModel()
        tps = m.transformed_parameters
        names = [tp.model_varname for tp in tps]
        assert "ab" in names

    def test_hyperparameters_property(self):
        m = SimpleModel()
        hypers = m.hyperparameters
        # mu has only constant parents -> hyperparameter
        names = [p.model_varname for p in hypers]
        assert "mu" in names

    def test_two_observable_model(self):
        m = TwoObservableModel()
        obs = m.observables
        names = [p.model_varname for p in obs]
        assert "y1" in names and "y2" in names


# ---------------------------------------------------------------------------
# Forbidden names
# ---------------------------------------------------------------------------
class TestForbiddenNames:
    """Test that forbidden attribute names raise errors at model construction."""

    def test_double_underscore_forbidden(self):
        with pytest.raises(ValueError, match="__"):

            class BadModel(Model):
                def __init__(self):
                    super().__init__()
                    self.my__param = Normal(mu=0.0, sigma=1.0)

            BadModel()

    def test_leading_underscore_forbidden(self):
        with pytest.raises(ValueError, match="_"):

            class BadModel(Model):
                def __init__(self):
                    super().__init__()
                    self._hidden = Normal(mu=0.0, sigma=1.0)

            BadModel()


# ---------------------------------------------------------------------------
# __contains__ / __getitem__
# ---------------------------------------------------------------------------
class TestModelAccess:
    """Test component access via __contains__ and __getitem__."""

    def test_contains(self):
        m = SimpleModel()
        assert "mu" in m
        assert "y" in m
        assert "nonexistent" not in m

    def test_getitem(self):
        m = SimpleModel()
        mu = m["mu"]
        assert mu.model_varname == "mu"

    def test_getitem_missing_raises(self):
        m = SimpleModel()
        with pytest.raises(KeyError):
            m["nonexistent"]  # pylint: disable=pointless-statement


# ---------------------------------------------------------------------------
# __setattr__ protection
# ---------------------------------------------------------------------------
class TestModelImmutability:
    """Test that model components can't be reassigned after init."""

    def test_reassign_component_raises(self):
        m = SimpleModel()
        with pytest.raises(AttributeError):
            m.mu = Normal(mu=1.0, sigma=2.0)

    def test_new_component_after_init_allowed(self):
        """Adding a new model component after init is allowed (only reassignment is blocked)."""
        m = SimpleModel()
        # __setattr__ only blocks reassignment of existing components
        # pylint: disable=attribute-defined-outside-init
        m.new_param = Normal(mu=0.0, sigma=1.0)  # Should not raise

    def test_non_component_attribute_ok(self):
        """Non-component attributes can be set freely."""
        m = SimpleModel()
        # pylint: disable=attribute-defined-outside-init
        m.some_flag = True  # Should not raise
        assert m.some_flag is True


# ---------------------------------------------------------------------------
# draw()
# ---------------------------------------------------------------------------
class TestModelDraw:
    """Test model-level draw functionality."""

    def test_draw_returns_dict(self):
        m = SimpleModel()
        result = m.draw(5)
        assert isinstance(result, dict)

    def test_draw_named_only_default(self):
        m = SimpleModel()
        result = m.draw(5, named_only=True)
        # Should contain named components
        assert any(isinstance(k, str) for k in result.keys())

    def test_draw_shape_correct(self):
        m = SimpleModel()
        result = m.draw(10)
        # y has shape (5,), so draws should be (10, 5)
        assert result["y"].shape == (10, 5)

    def test_draw_scalar_param(self):
        m = SimpleModel()
        result = m.draw(10)
        assert result["mu"].shape == (10,)

    def test_draw_reproducible(self):
        m = SimpleModel()
        d1 = m.draw(5, seed=42)
        d2 = m.draw(5, seed=42)
        np.testing.assert_array_equal(d1["mu"], d2["mu"])
        np.testing.assert_array_equal(d1["y"], d2["y"])

    def test_draw_as_xarray(self):
        xr = pytest.importorskip("xarray")
        m = SimpleModel()
        result = m.draw(5, as_xarray=True)
        assert isinstance(result, xr.Dataset)


# ---------------------------------------------------------------------------
# default_data
# ---------------------------------------------------------------------------
class TestDefaultData:
    """Test default_data getter and setter."""

    def test_has_default_data_false_initially(self):
        m = SimpleModel()
        assert m.has_default_data is False

    def test_get_default_data_without_setting_raises(self):
        m = SimpleModel()
        with pytest.raises(ValueError):
            _ = m.default_data

    def test_set_default_data(self):
        m = SimpleModel()
        data = {"y": np.zeros(5)}
        m.default_data = data
        assert m.has_default_data is True
        np.testing.assert_array_equal(m.default_data["y"], np.zeros(5))

    def test_set_default_data_missing_key_raises(self):
        m = SimpleModel()
        with pytest.raises(ValueError, match="missing"):
            m.default_data = {}  # Missing 'y'

    def test_set_default_data_extra_key_raises(self):
        m = SimpleModel()
        with pytest.raises(ValueError, match="extra"):
            m.default_data = {"y": np.zeros(5), "extra": np.zeros(3)}


# ---------------------------------------------------------------------------
# to_pytorch()
# ---------------------------------------------------------------------------
class TestToPyTorch:
    """Test conversion to PyTorchModel."""

    def test_to_pytorch_returns_module(self):

        m = SimpleModel()
        pt = m.to_pytorch(seed=0)
        assert isinstance(pt, torch.nn.Module)

    def test_to_pytorch_has_params(self):
        m = SimpleModel()
        pt = m.to_pytorch(seed=0)
        params = list(pt.parameters())
        assert len(params) > 0


# ---------------------------------------------------------------------------
# mle()
# ---------------------------------------------------------------------------
class TestMLE:
    """Test MLE fitting."""

    def test_mle_runs(self):
        m = SimpleModel()
        data = {"y": np.random.randn(5)}
        result = m.mle(data=data, epochs=50, seed=0)
        assert result is not None

    def test_mle_with_default_data(self):
        m = SimpleModel()
        m.default_data = {"y": np.random.randn(5)}
        result = m.mle(epochs=50, seed=0)
        assert result is not None

    def test_mle_no_data_no_default_raises(self):
        m = SimpleModel()
        with pytest.raises(ValueError):
            m.mle(epochs=50)


# ---------------------------------------------------------------------------
# simulate_mle()
# ---------------------------------------------------------------------------
class TestSimulateMLE:
    """Test simulation-based MLE."""

    def test_simulate_mle_returns_data_and_result(self):
        m = SimpleModel()
        sim_data, result = m.simulate_mle(epochs=50, seed=0)
        assert isinstance(sim_data, dict)
        assert "y" in sim_data
        assert result is not None


# ---------------------------------------------------------------------------
# mcmc() validation
# ---------------------------------------------------------------------------
class TestMCMCValidation:
    """Test mcmc validation without actually running Stan."""

    def test_delay_run_without_output_dir_raises(self):
        m = SimpleModel()
        data = {"y": np.random.randn(5)}
        with pytest.raises(ValueError, match="output"):
            m.mcmc(data=data, delay_run=True)
