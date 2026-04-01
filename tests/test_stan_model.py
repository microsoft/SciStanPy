# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for scistanpy.model.stan.stan_model module.

These tests verify Stan code generation without requiring CmdStan.
"""
# pylint: disable=missing-function-docstring, missing-class-docstring

import numpy as np

from scistanpy.model.components.constants import Constant
from scistanpy.model.components.parameters import (
    Beta,
    Dirichlet,
    HalfNormal,
    Multinomial,
    Normal,
    Poisson,
)
from scistanpy.model.model import Model
from scistanpy.model.stan.stan_model import StanProgram


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


class TransformedModel(Model):
    def __init__(self):
        super().__init__()
        self.a = Normal(mu=0.0, sigma=1.0)
        self.b = Normal(mu=0.0, sigma=1.0)
        self.ab = self.a + self.b
        self.y = Normal(mu=self.ab, sigma=1.0)


class BoundedModel(Model):
    def __init__(self):
        super().__init__()
        self.theta = Beta(alpha=2.0, beta=2.0)
        self.y = Normal(mu=self.theta, sigma=1.0, shape=(5,))


class ConstantInModel(Model):
    def __init__(self):
        super().__init__()
        self.offset = Constant(np.array([1.0, 2.0, 3.0]))
        self.mu = Normal(mu=0.0, sigma=1.0)
        self.y = Normal(mu=self.mu + self.offset, sigma=1.0)


class DiscreteModel(Model):
    def __init__(self):
        super().__init__()
        self.rate = HalfNormal(sigma=5.0)
        self.counts = Poisson(lambda_=self.rate, shape=(10,))


class SimplexModel(Model):
    def __init__(self):
        super().__init__()
        self.theta = Dirichlet(alpha=1.0, shape=(4,))
        self.counts = Multinomial(theta=self.theta, N=10)


# ---------------------------------------------------------------------------
# StanProgram construction
# ---------------------------------------------------------------------------
class TestStanProgram:
    """Test Stan program generation."""

    def test_construction(self):
        m = SimpleModel()
        prog = StanProgram(m)
        assert prog is not None

    def test_code_is_string(self):
        m = SimpleModel()
        prog = StanProgram(m)
        code = prog.code
        assert isinstance(code, str)
        assert len(code) > 0

    def test_code_contains_data_block(self):
        m = SimpleModel()
        prog = StanProgram(m)
        code = prog.code
        assert "data {" in code or "data{" in code

    def test_code_contains_parameters_block(self):
        m = SimpleModel()
        prog = StanProgram(m)
        code = prog.code
        assert "parameters {" in code or "parameters{" in code

    def test_code_contains_model_block(self):
        m = SimpleModel()
        prog = StanProgram(m)
        code = prog.code
        assert "model {" in code or "model{" in code


# ---------------------------------------------------------------------------
# Data block
# ---------------------------------------------------------------------------
class TestDataBlock:
    """Test data block generation."""

    def test_observable_in_data(self):
        m = SimpleModel()
        prog = StanProgram(m)
        data_code = prog.data_block
        assert "y" in data_code

    def test_constant_in_data(self):
        m = ConstantInModel()
        prog = StanProgram(m)
        data_code = prog.data_block
        assert "offset" in data_code

    def test_int_observable_declared_as_int(self):
        m = DiscreteModel()
        prog = StanProgram(m)
        data_code = prog.data_block
        assert "int" in data_code


# ---------------------------------------------------------------------------
# Parameters block
# ---------------------------------------------------------------------------
class TestParametersBlock:
    """Test parameters block generation."""

    def test_latent_param_in_parameters(self):
        m = SimpleModel()
        prog = StanProgram(m)
        params_code = prog.parameters_block
        assert "mu" in params_code

    def test_bounded_param_declared_with_bounds(self):
        m = BoundedModel()
        prog = StanProgram(m)
        params_code = prog.parameters_block
        assert "lower" in params_code and "upper" in params_code

    def test_half_normal_has_lower_bound(self):
        m = HierarchicalModel()
        prog = StanProgram(m)
        params_code = prog.parameters_block
        assert "lower" in params_code

    def test_simplex_declared(self):
        m = SimplexModel()
        prog = StanProgram(m)
        params_code = prog.parameters_block
        assert "simplex" in params_code


# ---------------------------------------------------------------------------
# Model block
# ---------------------------------------------------------------------------
class TestModelBlock:
    """Test model block generation."""

    def test_contains_target_increment(self):
        m = SimpleModel()
        prog = StanProgram(m)
        model_code = prog.model_block
        assert "~" in model_code or "target" in model_code

    def test_normal_distribution_referenced(self):
        m = SimpleModel()
        prog = StanProgram(m)
        model_code = prog.model_block
        assert "normal" in model_code


# ---------------------------------------------------------------------------
# Transformed parameters block
# ---------------------------------------------------------------------------
class TestTransformedParametersBlock:
    """Test transformed parameters block."""

    def test_named_transformation_present(self):
        m = TransformedModel()
        prog = StanProgram(m)
        tp_code = prog.transformed_parameters_block
        assert "ab" in tp_code


# ---------------------------------------------------------------------------
# Generated quantities block
# ---------------------------------------------------------------------------
class TestGeneratedQuantities:
    """Test generated quantities block for posterior predictive."""

    def test_generated_quantities_present(self):
        m = SimpleModel()
        prog = StanProgram(m)
        code = prog.code
        assert "generated quantities" in code


# ---------------------------------------------------------------------------
# Full code validity
# ---------------------------------------------------------------------------
class TestFullCode:
    """Test properties of complete Stan code."""

    def test_hierarchical_model_code(self):
        m = HierarchicalModel()
        prog = StanProgram(m)
        code = prog.code
        # Should contain all blocks
        assert "data" in code
        assert "parameters" in code
        assert "model" in code

    def test_no_python_artifacts(self):
        """Stan code should not contain Python-specific syntax."""
        m = SimpleModel()
        prog = StanProgram(m)
        code = prog.code
        assert "import " not in code
        assert "def " not in code
        assert "self." not in code

    def test_balanced_braces(self):
        """Stan code should have balanced curly braces."""
        m = SimpleModel()
        prog = StanProgram(m)
        code = prog.code
        assert code.count("{") == code.count("}")
