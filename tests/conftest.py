# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Shared fixtures and configuration for SciStanPy test suite."""

# pylint: disable=missing-class-docstring

import pytest

import scistanpy as ssp


@pytest.fixture(autouse=True)
def set_seed():
    """Set deterministic seeds before every test."""
    ssp.manual_seed(42)
    yield


@pytest.fixture
def simple_model():
    """A minimal model with one parameter and one observable."""

    class SimpleModel(ssp.Model):
        def __init__(self):
            super().__init__()
            self.mu = ssp.parameters.Normal(mu=0.0, sigma=1.0)
            self.y = ssp.parameters.Normal(mu=self.mu, sigma=1.0, shape=(5,))

    return SimpleModel()


@pytest.fixture
def hierarchical_model():
    """A hierarchical model with multiple levels."""

    class HierarchicalModel(ssp.Model):
        def __init__(self):
            super().__init__()
            self.mu = ssp.parameters.Normal(mu=0.0, sigma=10.0)
            self.sigma = ssp.parameters.HalfNormal(sigma=1.0)
            self.y = ssp.parameters.Normal(mu=self.mu, sigma=self.sigma, shape=(10,))

    return HierarchicalModel()


@pytest.fixture
def multivariate_model():
    """A model with multivariate parameters."""

    class MultivariateModel(ssp.Model):
        def __init__(self):
            super().__init__()
            self.alpha = ssp.parameters.Dirichlet(alpha=1.0, shape=(4,))
            self.y = ssp.parameters.Multinomial(theta=self.alpha, N=100, shape=(4,))

    return MultivariateModel()
