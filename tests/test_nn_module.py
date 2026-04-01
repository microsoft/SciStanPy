# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for scistanpy.model.nn_module module."""

# pylint: disable=missing-function-docstring, missing-class-docstring

import numpy as np
import pytest
import torch

from scistanpy.model.components.parameters import HalfNormal, Normal
from scistanpy.model.model import Model
from scistanpy.model.nn_module import PyTorchModel, check_observable_data


# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------
class SimpleModel(Model):
    def __init__(self):
        super().__init__()
        self.mu = Normal(mu=0.0, sigma=1.0)
        self.y = Normal(mu=self.mu, sigma=1.0, shape=(5,))


class TwoObsModel(Model):
    def __init__(self):
        super().__init__()
        self.mu = Normal(mu=0.0, sigma=1.0)
        self.sigma = HalfNormal(sigma=1.0)
        self.y1 = Normal(mu=self.mu, sigma=self.sigma, shape=(3,))
        self.y2 = Normal(mu=0.0, sigma=self.sigma, shape=(4,))


# ---------------------------------------------------------------------------
# check_observable_data
# ---------------------------------------------------------------------------
class TestCheckObservableData:
    """Test observable data validation."""

    def test_valid_data_passes(self):
        m = SimpleModel()
        data = {"y": torch.from_numpy(np.random.randn(5).astype(np.float32))}
        check_observable_data(m, data)  # Should not raise

    def test_missing_key_raises(self):
        m = SimpleModel()
        with pytest.raises(ValueError, match="missing"):
            check_observable_data(m, {})

    def test_extra_key_raises(self):
        m = SimpleModel()
        data = {
            "y": torch.randn(5),
            "extra": torch.zeros(3),
        }
        with pytest.raises(ValueError, match="extra|addition"):
            check_observable_data(m, data)

    def test_shape_mismatch_raises(self):
        m = SimpleModel()
        data = {"y": torch.randn(10)}  # Wrong shape: (10,) vs (5,)
        with pytest.raises(ValueError, match="shape"):
            check_observable_data(m, data)

    def test_two_obs_valid(self):
        m = TwoObsModel()
        data = {"y1": torch.randn(3), "y2": torch.randn(4)}
        check_observable_data(m, data)  # Should not raise

    def test_two_obs_missing_one_raises(self):
        m = TwoObsModel()
        with pytest.raises(ValueError, match="missing"):
            check_observable_data(m, {"y1": torch.randn(3)})


# ---------------------------------------------------------------------------
# PyTorchModel construction
# ---------------------------------------------------------------------------
class TestPyTorchModelConstruction:
    """Test PyTorchModel creation."""

    def test_construction(self):
        m = SimpleModel()
        pt = PyTorchModel(m, seed=0)
        assert isinstance(pt, torch.nn.Module)

    def test_has_learnable_params(self):
        m = SimpleModel()
        pt = PyTorchModel(m, seed=0)
        params = list(pt.parameters())
        assert len(params) > 0

    def test_learnable_params_require_grad(self):
        m = SimpleModel()
        pt = PyTorchModel(m, seed=0)
        for p in pt.parameters():
            assert p.requires_grad


# ---------------------------------------------------------------------------
# forward()
# ---------------------------------------------------------------------------
class TestForward:
    """Test forward pass (log probability computation)."""

    def test_forward_returns_scalar(self):
        m = SimpleModel()
        pt = PyTorchModel(m, seed=0)
        data = {"y": torch.randn(5)}
        log_prob = pt(**data)
        assert log_prob.dim() == 0

    def test_forward_is_finite(self):
        m = SimpleModel()
        pt = PyTorchModel(m, seed=0)
        data = {"y": torch.randn(5)}
        log_prob = pt(**data)
        assert torch.isfinite(log_prob)


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------
class TestFit:
    """Test model fitting."""

    def test_fit_returns_loss_tensor(self):
        m = SimpleModel()
        pt = PyTorchModel(m, seed=0)
        data = {"y": np.random.randn(5)}
        losses = pt.fit(data=data, epochs=50, early_stop=10, lr=0.01)
        assert isinstance(losses, torch.Tensor)

    def test_fit_loss_decreases(self):
        m = SimpleModel()
        pt = PyTorchModel(m, seed=0)
        data = {"y": np.random.randn(5)}
        losses = pt.fit(data=data, epochs=200, early_stop=50, lr=0.01)
        # Loss should generally decrease from start to end
        first_losses = losses[:10].mean().item()
        last_losses = losses[-10:].mean().item()
        assert last_losses < first_losses

    def test_fit_with_tensors(self):
        m = SimpleModel()
        pt = PyTorchModel(m, seed=0)
        data = {"y": torch.randn(5)}
        losses = pt.fit(data=data, epochs=30, early_stop=10, lr=0.01)
        assert len(losses) > 0  # May trigger early stopping


# ---------------------------------------------------------------------------
# export_params()
# ---------------------------------------------------------------------------
class TestExportParams:
    """Test parameter export."""

    def test_export_params_contains_non_observables(self):
        m = SimpleModel()
        pt = PyTorchModel(m, seed=0)
        params = pt.export_params()
        assert "mu" in params

    def test_export_params_excludes_observables(self):
        m = SimpleModel()
        pt = PyTorchModel(m, seed=0)
        params = pt.export_params()
        assert "y" not in params

    def test_export_params_are_tensors(self):
        m = SimpleModel()
        pt = PyTorchModel(m, seed=0)
        params = pt.export_params()
        for v in params.values():
            assert isinstance(v, torch.Tensor)


# ---------------------------------------------------------------------------
# export_distributions()
# ---------------------------------------------------------------------------
class TestExportDistributions:
    """Test distribution export."""

    def test_export_distributions_returns_dict(self):
        m = SimpleModel()
        pt = PyTorchModel(m, seed=0)
        dists = pt.export_distributions()
        assert isinstance(dists, dict)

    def test_export_distributions_includes_all(self):
        m = SimpleModel()
        pt = PyTorchModel(m, seed=0)
        dists = pt.export_distributions()
        # Should include both parameters and observables
        assert "mu" in dists
        assert "y" in dists

    def test_export_distributions_are_torch_dists(self):
        m = SimpleModel()
        pt = PyTorchModel(m, seed=0)
        dists = pt.export_distributions()
        for v in dists.values():
            assert isinstance(v, torch.distributions.Distribution)


# ---------------------------------------------------------------------------
# Device movement
# ---------------------------------------------------------------------------
class TestDeviceMovement:
    """Test device transfer."""

    def test_cpu_movement(self):
        m = SimpleModel()
        pt = PyTorchModel(m, seed=0)
        pt.cpu()  # Should not raise
        for p in pt.parameters():
            assert p.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_movement(self):
        m = SimpleModel()
        pt = PyTorchModel(m, seed=0)
        pt.cuda()
        for p in pt.parameters():
            assert p.device.type == "cuda"
