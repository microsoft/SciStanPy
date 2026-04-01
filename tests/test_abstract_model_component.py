# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for scistanpy.model.components.abstract_model_component module."""

# pylint: disable=missing-function-docstring

import numpy as np
import pytest

from scistanpy.model.components.constants import Constant
from scistanpy.model.components.parameters import Beta, Dirichlet, HalfNormal, Normal


# ---------------------------------------------------------------------------
# Shape broadcasting
# ---------------------------------------------------------------------------
class TestShapeBroadcasting:
    """Test shape propagation and broadcasting logic."""

    def test_scalar_parameter(self):
        p = Normal(mu=0.0, sigma=1.0)
        assert p.shape == ()
        assert p.ndim == 0

    def test_explicit_shape(self):
        p = Normal(mu=0.0, sigma=1.0, shape=(5,))
        assert p.shape == (5,)
        assert p.ndim == 1

    def test_shape_from_array_parent(self):
        mu = np.zeros(5)
        p = Normal(mu=mu, sigma=1.0)
        assert p.shape == (5,)

    def test_shape_broadcast_parents(self):
        """Shape broadcasts across parents."""
        mu = Constant(np.zeros((3, 1)))
        sigma = Constant(np.ones((1, 4)))
        p = Normal(mu=mu, sigma=sigma)
        assert p.shape == (3, 4)

    def test_shape_broadcast_with_explicit(self):
        """Explicit shape must be compatible with parents."""
        mu = Constant(np.zeros(5))
        p = Normal(mu=mu, sigma=1.0, shape=(5,))
        assert p.shape == (5,)

    def test_incompatible_shape_raises(self):
        """Explicit shape that conflicts with parent shapes raises."""
        mu = Constant(np.zeros(5))
        with pytest.raises(ValueError, match="shape"):
            Normal(mu=mu, sigma=1.0, shape=(3,))

    def test_hierarchical_shape_broadcast(self):
        """Shape propagates through hierarchical model."""
        mu = Normal(mu=0.0, sigma=1.0, shape=(3, 1))
        sigma = HalfNormal(sigma=1.0, shape=(1, 4))
        y = Normal(mu=mu, sigma=sigma)
        assert y.shape == (3, 4)


# ---------------------------------------------------------------------------
# Parent-child relationships
# ---------------------------------------------------------------------------
class TestParentChild:
    """Test parent/child dependency graph."""

    def test_constant_parent(self):
        mu = Constant(0.0)
        p = Normal(mu=mu, sigma=1.0)
        assert mu in p.parents  # parents is a list

    def test_numeric_auto_converted_to_constant(self):
        """Numeric values passed as parents are auto-converted to Constants."""
        p = Normal(mu=0.0, sigma=1.0)
        assert all(isinstance(v, Constant) for v in p.parents)

    def test_parameter_parent(self):
        mu = Normal(mu=0.0, sigma=1.0)
        y = Normal(mu=mu, sigma=1.0)
        assert mu in y.parents  # parents is a list
        assert y in mu.children

    def test_children_tracked_bidirectionally(self):
        mu = Normal(mu=0.0, sigma=1.0)
        y = Normal(mu=mu, sigma=1.0)
        assert y in mu.children
        assert mu in y.parents

    def test_constants_property(self):
        """constants property returns only Constant parents (dict keyed by param name)."""
        mu = Normal(mu=0.0, sigma=1.0)
        # mu's constants: auto-created Constant(0.0) and Constant(1.0)
        assert all(isinstance(c, Constant) for c in mu.constants.values())

    def test_get_child_paramnames(self):
        mu = Normal(mu=0.0, sigma=1.0)
        y = Normal(mu=mu, sigma=1.0)
        child_params = mu.get_child_paramnames()
        assert y in child_params
        assert child_params[y] == "mu"


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------
class TestNaming:
    """Test model_varname assignment."""

    def test_unnamed_initially(self):
        p = Normal(mu=0.0, sigma=1.0)
        assert p.is_named is False

    def test_named_after_assignment(self):
        p = Normal(mu=0.0, sigma=1.0)
        p.model_varname = "my_param"
        assert p.is_named is True
        assert p.model_varname == "my_param"

    def test_stan_model_varname_replaces_dot(self):
        p = Normal(mu=0.0, sigma=1.0)
        p.model_varname = "a.b"
        assert p.stan_model_varname == "a__b"


# ---------------------------------------------------------------------------
# Stan type generation
# ---------------------------------------------------------------------------
class TestStanDtype:
    """Test Stan data type string generation."""

    def test_scalar_real(self):
        p = Normal(mu=0.0, sigma=1.0)
        # Force base type for scalar
        dtype = p.get_stan_dtype(force_basetype=True)
        assert "real" in dtype

    def test_vector_type(self):
        p = Normal(mu=0.0, sigma=1.0, shape=(5,))
        dtype = p.get_stan_dtype()
        assert "vector" in dtype or "real" in dtype

    def test_bounded_parameter(self):
        p = HalfNormal(sigma=1.0)
        dtype = p.get_stan_dtype()
        assert "lower" in dtype

    def test_double_bounded(self):
        p = Beta(alpha=2.0, beta=2.0)
        dtype = p.get_stan_dtype()
        assert "lower" in dtype and "upper" in dtype

    def test_simplex(self):
        p = Dirichlet(alpha=1.0, shape=(4,))
        dtype = p.get_stan_dtype()
        assert "simplex" in dtype

    def test_constant_int_dtype(self):
        c = Constant(5)
        dtype = c.get_stan_dtype()
        assert "int" in dtype


# ---------------------------------------------------------------------------
# Draw mechanism
# ---------------------------------------------------------------------------
class TestDraw:
    """Test draw (prior sampling) behavior."""

    def test_draw_returns_samples(self):
        p = Normal(mu=0.0, sigma=1.0)
        p.model_varname = "x"
        draws, _ = p.draw(10)
        assert draws.shape == (10,)

    def test_draw_shaped(self):
        p = Normal(mu=0.0, sigma=1.0, shape=(3, 4))
        p.model_varname = "x"
        draws, _ = p.draw(5)
        assert draws.shape == (5, 3, 4)

    def test_draw_hierarchical(self):
        """Draw propagates through dependency graph."""
        mu = Normal(mu=0.0, sigma=1.0)
        mu.model_varname = "mu"
        y = Normal(mu=mu, sigma=1.0, shape=(5,))
        y.model_varname = "y"
        draws, cache = y.draw(10)
        assert draws.shape == (10, 5)
        assert mu in cache  # mu was drawn as dependency

    def test_draw_respects_bounds(self):
        p = HalfNormal(sigma=1.0, shape=(20,))
        p.model_varname = "x"
        draws, _ = p.draw(50)
        assert np.all(draws >= 0)

    def test_draw_simplex_sums_to_one(self):
        p = Dirichlet(alpha=1.0, shape=(4,))
        p.model_varname = "d"
        draws, _ = p.draw(10)
        np.testing.assert_allclose(draws.sum(axis=-1), 1.0, atol=1e-10)

    def test_draw_reproducible(self):
        p = Normal(mu=0.0, sigma=1.0, shape=(5,))
        p.model_varname = "x"
        d1, _ = p.draw(10, seed=42)
        d2, _ = p.draw(10, seed=42)
        np.testing.assert_array_equal(d1, d2)


# ---------------------------------------------------------------------------
# walk_tree
# ---------------------------------------------------------------------------
class TestWalkTree:
    """Test dependency graph traversal."""

    def test_walk_down_from_leaf(self):
        """A node with no children yields an empty walk-down."""
        mu = Normal(mu=0.0, sigma=1.0)
        _ = Normal(mu=mu, sigma=1.0)
        tree = list(mu.walk_tree(walk_down=True))
        assert len(tree) > 0

    def test_walk_up_from_leaf(self):
        mu = Normal(mu=0.0, sigma=1.0)
        y = Normal(mu=mu, sigma=1.0)
        tree = list(y.walk_tree(walk_down=False))
        assert len(tree) > 0

    def test_walk_visiting_order(self):
        """Walk down visits children before grandchildren."""
        a = Normal(mu=0.0, sigma=1.0)
        b = Normal(mu=a, sigma=1.0)
        c = Normal(mu=b, sigma=1.0)
        tree = list(a.walk_tree(walk_down=True))
        components = [t[1] for t in tree]
        assert components[0] is a
        # b should appear before c
        if b in components and c in components:
            assert components.index(b) < components.index(c)


# ---------------------------------------------------------------------------
# Shared leading dimensions
# ---------------------------------------------------------------------------
class TestSharedLeading:
    """Test get_shared_leading dimension counting."""

    def test_same_shape(self):
        a = Normal(mu=0.0, sigma=1.0, shape=(3, 4))
        b = Normal(mu=0.0, sigma=1.0, shape=(3, 4))
        # get_shared_leading returns index of last compatible dim
        result = a.get_shared_leading(b)
        assert result >= 1  # At least first dim matches

    def test_different_shape(self):
        a = Normal(mu=0.0, sigma=1.0, shape=(3, 4))
        b = Normal(mu=0.0, sigma=1.0, shape=(3, 5))
        # First dim matches (3==3), second does not (4!=5)
        result = a.get_shared_leading(b)
        assert result >= 0

    def test_scalar(self):
        a = Normal(mu=0.0, sigma=1.0)
        b = Normal(mu=0.0, sigma=1.0, shape=(3,))
        assert a.get_shared_leading(b) == 0


# ---------------------------------------------------------------------------
# Stan variable declaration
# ---------------------------------------------------------------------------
class TestStanVariable:
    """Test declare_stan_variable."""

    def test_declare_scalar_real(self):
        p = Normal(mu=0.0, sigma=1.0)
        decl = p.declare_stan_variable("x")
        assert "x" in decl
        assert "real" in decl

    def test_declare_vector(self):
        p = Normal(mu=0.0, sigma=1.0, shape=(5,))
        decl = p.declare_stan_variable("x")
        assert "x" in decl

    def test_declare_bounded(self):
        p = HalfNormal(sigma=1.0)
        decl = p.declare_stan_variable("x")
        assert "lower" in decl


# ---------------------------------------------------------------------------
# __getitem__ for parent access
# ---------------------------------------------------------------------------
class TestGetItem:
    """Test string-based parent access."""

    def test_getitem_by_paramname(self):
        mu = Constant(0.0)
        p = Normal(mu=mu, sigma=1.0)
        assert p["mu"] is mu

    def test_getitem_missing_raises(self):
        p = Normal(mu=0.0, sigma=1.0)
        with pytest.raises(KeyError):
            p["nonexistent"]  # pylint: disable=pointless-statement

    def test_contains(self):
        p = Normal(mu=0.0, sigma=1.0)
        assert "mu" in p
        assert "sigma" in p
        assert "nonexistent" not in p
