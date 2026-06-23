"""
Extended tests for Phase 1 audit — covers gaps in original test suite.
Tests for: DirichletDist, math_utils functions, edge cases, FP16 safety.
"""

import math
import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings, strategies as st

from themis.core.distributions import GaussianDist, CategoricalDist, DirichletDist
from themis.core.math_utils import (
    logsumexp,
    softmax_with_temp,
    precision_weighted_error,
    precision_weighted_mse,
    kl_categorical,
    kl_gaussian,
    entropy_categorical,
    entropy_gaussian,
)


# =============================================================================
# 1. DirichletDist Tests (was completely untested)
# =============================================================================

class TestDirichletDist:

    def test_dirichlet_mean(self):
        """E[X] = alpha / alpha_0 should be a valid probability simplex."""
        conc = torch.tensor([[2.0, 3.0, 5.0], [1.0, 1.0, 1.0]])
        d = DirichletDist(conc)
        
        # Mean should sum to 1
        assert torch.allclose(d.mean.sum(dim=-1), torch.ones(2), atol=1e-6)
        # Specific values
        assert torch.allclose(d.mean[0], torch.tensor([0.2, 0.3, 0.5]), atol=1e-6)
        assert torch.allclose(d.mean[1], torch.tensor([1/3, 1/3, 1/3]), atol=1e-5)

    def test_dirichlet_sample_simplex(self):
        """Samples from Dirichlet should lie on the probability simplex."""
        conc = torch.tensor([[2.0, 3.0, 5.0]])
        d = DirichletDist(conc)
        sample = d.sample()
        
        # All values >= 0
        assert (sample >= 0).all()
        # Sum to 1
        assert torch.allclose(sample.sum(dim=-1), torch.ones(1), atol=1e-5)

    def test_dirichlet_kl_self_zero(self):
        """KL(D || D) should be 0."""
        conc = torch.tensor([[2.0, 3.0, 5.0], [1.5, 2.5, 4.0]])
        d = DirichletDist(conc)
        kl = d.kl_divergence(d)
        assert torch.allclose(kl, torch.zeros(2), atol=1e-5)

    def test_dirichlet_kl_nonnegative(self):
        """KL should always be non-negative."""
        conc_a = torch.tensor([[2.0, 3.0, 5.0], [1.0, 1.0, 1.0]])
        conc_b = torch.tensor([[1.0, 1.0, 1.0], [2.0, 3.0, 5.0]])
        d_a = DirichletDist(conc_a)
        d_b = DirichletDist(conc_b)
        kl = d_a.kl_divergence(d_b)
        assert (kl >= -1e-5).all()

    def test_dirichlet_update(self):
        """Bayesian update should add counts to concentration."""
        conc = torch.tensor([[1.0, 1.0, 1.0]])
        d = DirichletDist(conc)
        obs = torch.tensor([[0.0, 3.0, 0.0]])  # 3 counts for category 1
        d_post = d.update(obs)
        assert torch.allclose(d_post.concentration, torch.tensor([[1.0, 4.0, 1.0]]))

    def test_dirichlet_entropy_uniform_is_max(self):
        """Uniform Dirichlet (all alpha=1) should have higher entropy than peaked."""
        uniform = DirichletDist(torch.tensor([[1.0, 1.0, 1.0]]))
        peaked = DirichletDist(torch.tensor([[10.0, 10.0, 10.0]]))
        assert uniform.entropy().item() > peaked.entropy().item()


# =============================================================================
# 2. math_utils Tests (were completely untested)
# =============================================================================

class TestLogsumexp:

    def test_basic(self):
        x = torch.tensor([[1.0, 2.0, 3.0]])
        result = logsumexp(x, dim=-1)
        expected = torch.log(torch.exp(torch.tensor([1.0, 2.0, 3.0])).sum())
        assert torch.allclose(result.squeeze(), expected, atol=1e-5)

    def test_numerical_stability(self):
        """Should handle large values without overflow."""
        x = torch.tensor([[1000.0, 1001.0, 1002.0]])
        result = logsumexp(x, dim=-1)
        # Should not be inf
        assert torch.isfinite(result).all()


class TestSoftmaxWithTemp:

    def test_temperature_1(self):
        """At temp=1, should match standard softmax."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = softmax_with_temp(logits, temperature=1.0)
        expected = F.softmax(logits, dim=-1)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_high_temperature(self):
        """High temperature should approach uniform."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = softmax_with_temp(logits, temperature=100.0)
        uniform = torch.ones(3) / 3.0
        assert torch.allclose(result.squeeze(), uniform, atol=0.01)

    def test_zero_temperature_hard_argmax(self):
        """Near-zero temperature should give one-hot at argmax."""
        logits = torch.tensor([[1.0, 5.0, 3.0]])
        result = softmax_with_temp(logits, temperature=1e-6)
        expected = torch.tensor([[0.0, 1.0, 0.0]])
        assert torch.allclose(result, expected)

    def test_output_shape_preserved(self):
        """Output shape must match input shape."""
        logits = torch.randn(4, 8)
        result = softmax_with_temp(logits, temperature=0.5)
        assert result.shape == logits.shape

    def test_zero_temp_shape_preserved(self):
        """Hard argmax path must also preserve shape."""
        logits = torch.randn(4, 8)
        result = softmax_with_temp(logits, temperature=1e-8)
        assert result.shape == logits.shape
        # Each row should sum to 1
        assert torch.allclose(result.sum(dim=-1), torch.ones(4), atol=1e-6)


class TestPrecisionWeighted:

    def test_error_direction(self):
        """Error should be positive when target > prediction."""
        pred = torch.tensor([[0.0, 0.0]])
        target = torch.tensor([[1.0, 2.0]])
        prec = torch.tensor([[1.0, 1.0]])
        error = precision_weighted_error(pred, target, prec)
        assert (error > 0).all()

    def test_high_precision_amplifies(self):
        """Higher precision should amplify the error."""
        pred = torch.zeros(1, 3)
        target = torch.ones(1, 3)
        low_prec = torch.ones(1, 3)
        high_prec = torch.full((1, 3), 10.0)
        
        error_low = precision_weighted_error(pred, target, low_prec)
        error_high = precision_weighted_error(pred, target, high_prec)
        assert (error_high.abs() > error_low.abs()).all()

    def test_mse_nonnegative(self):
        """Precision-weighted MSE should always be >= 0."""
        pred = torch.randn(5, 4)
        target = torch.randn(5, 4)
        prec = torch.rand(5, 4).clamp(min=0.1)
        mse = precision_weighted_mse(pred, target, prec)
        assert (mse >= 0).all()


# =============================================================================
# 3. Edge Cases & Robustness
# =============================================================================

class TestEdgeCases:

    def test_gaussian_kl_asymmetric(self):
        """KL(q||p) != KL(p||q) in general."""
        q = GaussianDist(torch.tensor([[0.0]]), torch.tensor([[0.0]]))
        p = GaussianDist(torch.tensor([[2.0]]), torch.tensor([[1.0]]))
        kl_qp = q.kl_divergence(p)
        kl_pq = p.kl_divergence(q)
        assert not torch.allclose(kl_qp, kl_pq)

    def test_categorical_kl_nonneg(self):
        """Categorical KL should be >= 0 for arbitrary logits."""
        for _ in range(10):
            logits_q = torch.randn(3, 5)
            logits_p = torch.randn(3, 5)
            kl = kl_categorical(logits_q, logits_p)
            assert (kl >= -1e-5).all(), f"KL was negative: {kl}"

    def test_entropy_categorical_bounds(self):
        """Entropy should be between 0 and ln(K)."""
        K = 5
        for _ in range(10):
            logits = torch.randn(3, K)
            h = entropy_categorical(logits)
            assert (h >= -1e-5).all()
            assert (h <= math.log(K) + 1e-5).all()

    def test_gaussian_from_params(self):
        """GaussianDist.from_params should correctly split and clamp."""
        params = torch.tensor([[1.0, 2.0, -20.0, 5.0]])  # 2 mean dims, 2 logvar dims
        g = GaussianDist.from_params(params, min_log_var=-10.0, max_log_var=2.0)
        assert torch.allclose(g.mean, torch.tensor([[1.0, 2.0]]))
        assert g.log_var[0, 0].item() == -10.0  # clamped from -20
        assert g.log_var[0, 1].item() == 2.0    # clamped from 5

    def test_gaussian_detach(self):
        """Detached distribution should not require grad."""
        mean = torch.randn(2, 3, requires_grad=True)
        logvar = torch.randn(2, 3, requires_grad=True)
        g = GaussianDist(mean, logvar)
        g_det = g.detach()
        assert not g_det.mean.requires_grad
        assert not g_det.log_var.requires_grad

    def test_categorical_sample_is_onehot(self):
        """Gumbel-softmax hard samples should be one-hot."""
        logits = torch.randn(4, 6)
        c = CategoricalDist(logits)
        sample = c.sample()
        # Each row should have exactly one 1.0 and rest 0.0
        assert torch.allclose(sample.sum(dim=-1), torch.ones(4))
        assert ((sample == 0.0) | (sample == 1.0)).all()

    def test_categorical_kl_from_uniform(self):
        """KL from uniform for a uniform distribution should be 0."""
        logits = torch.zeros(2, 5)  # uniform
        c = CategoricalDist(logits)
        kl = c.kl_from_uniform()
        assert torch.allclose(kl, torch.zeros(2), atol=1e-5)


# =============================================================================
# 4. Property-Based Tests (extended)
# =============================================================================

@given(
    k=st.integers(min_value=2, max_value=10),
)
@settings(max_examples=20, deadline=None)
def test_categorical_entropy_bounds_property(k):
    """Entropy of any categorical with K categories is in [0, ln(K)]."""
    logits = torch.randn(1, k)
    h = entropy_categorical(logits).item()
    assert h >= -1e-5
    assert h <= math.log(k) + 1e-5


@given(
    dim=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=20, deadline=None)
def test_gaussian_kl_self_zero_property(dim):
    """KL(q || q) should always be 0 regardless of dim."""
    mean = torch.randn(1, dim)
    logvar = torch.randn(1, dim)
    kl = kl_gaussian(mean, logvar, mean, logvar).item()
    assert abs(kl) < 1e-4
