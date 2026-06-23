"""
Themis Core — Probability Distributions

Implements distributional representations that preserve uncertainty at every layer.
Every belief in Themis is a distribution, not a point estimate.

Key distributions:
  - GaussianDist: For continuous latent states (mean + log_var)
  - CategoricalDist: For discrete latent variables
  - DirichletDist: For concentration parameters (beliefs about beliefs)

All operations support batched computation and FP16.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Gaussian Distribution — The workhorse of continuous Active Inference
# =============================================================================

@dataclass
class GaussianDist:
    """
    Diagonal Gaussian distribution parameterized by mean and log-variance.

    Using log_var instead of std because:
    1. Numerically stable (no need to clamp std > 0)
    2. Network can output any real number for log_var
    3. KL divergence has a clean closed-form

    Shapes: mean, log_var are both [batch, ..., dim]
    """
    mean: Tensor
    log_var: Tensor

    @property
    def std(self) -> Tensor:
        """Standard deviation, clamped for numerical stability."""
        return torch.exp(0.5 * self.log_var).clamp(min=1e-6)

    @property
    def var(self) -> Tensor:
        """Variance."""
        return torch.exp(self.log_var).clamp(min=1e-8)

    @property
    def precision(self) -> Tensor:
        """Precision (inverse variance) — central to predictive coding."""
        return 1.0 / self.var

    @property
    def dim(self) -> int:
        """Dimensionality of the distribution."""
        return self.mean.shape[-1]

    def sample(self, n_samples: int = 1) -> Tensor:
        """
        Reparameterized sampling: z = μ + σ * ε, where ε ~ N(0, I).
        Gradient flows through μ and σ (not through ε).
        """
        if n_samples == 1:
            eps = torch.randn_like(self.mean)
            return self.mean + self.std * eps
        else:
            # [n_samples, batch, ..., dim]
            shape = (n_samples,) + self.mean.shape
            eps = torch.randn(shape, device=self.mean.device, dtype=self.mean.dtype)
            return self.mean.unsqueeze(0) + self.std.unsqueeze(0) * eps

    def log_prob(self, x: Tensor) -> Tensor:
        """
        Log probability of x under this Gaussian.
        Returns per-dimension log-prob, sum over last dim for total.

        ln N(x | μ, σ²) = -0.5 * [ln(2π) + ln(σ²) + (x-μ)²/σ²]
        """
        # Use clamped var for numerical stability (consistent with self.var property)
        var = self.var  # clamped min=1e-8
        log_var = torch.log(var)
        return -0.5 * (
            math.log(2 * math.pi)
            + log_var
            + (x - self.mean).pow(2) / var
        )

    def log_prob_total(self, x: Tensor) -> Tensor:
        """Total log probability (summed over dimensions)."""
        return self.log_prob(x).sum(dim=-1)

    def entropy(self) -> Tensor:
        """
        Differential entropy of Gaussian: H = 0.5 * d * (1 + ln(2π)) + 0.5 * Σ ln(σ²)
        Returns per-sample entropy (summed over dimensions).
        """
        return 0.5 * (self.dim * (1.0 + math.log(2 * math.pi)) + self.log_var.sum(dim=-1))

    def kl_divergence(self, other: GaussianDist) -> Tensor:
        """
        KL(self || other) for diagonal Gaussians.

        KL(q || p) = 0.5 * Σ [ln(σ_p²/σ_q²) + (σ_q² + (μ_q - μ_p)²)/σ_p² - 1]

        Returns per-sample KL (summed over dimensions).
        """
        return 0.5 * (
            other.log_var - self.log_var
            + self.var / other.var
            + (self.mean - other.mean).pow(2) / other.var
            - 1.0
        ).sum(dim=-1)

    def kl_from_standard_normal(self) -> Tensor:
        """
        KL(self || N(0, I)) — common regularizer.

        KL = -0.5 * Σ [1 + ln(σ²) - μ² - σ²]
        """
        return -0.5 * (1.0 + self.log_var - self.mean.pow(2) - self.var).sum(dim=-1)

    def precision_weighted_error(self, observation: Tensor) -> Tensor:
        """
        Precision-weighted prediction error — the core of predictive coding.
        ε = Π * (o - μ) where Π = 1/σ²

        Returns per-dimension weighted error.
        """
        return self.precision * (observation - self.mean)

    def precision_weighted_mse(self, observation: Tensor) -> Tensor:
        """
        Precision-weighted mean squared error (scalar per sample).
        This is the "accuracy" term in Free Energy.
        """
        return (self.precision * (observation - self.mean).pow(2)).sum(dim=-1)

    @staticmethod
    def from_params(params: Tensor, min_log_var: float = -10.0, max_log_var: float = 2.0) -> GaussianDist:
        """
        Create GaussianDist from a single parameter tensor.
        Splits last dim in half: [mean, log_var].
        Clamps log_var for stability.
        """
        d = params.shape[-1] // 2
        mean = params[..., :d]
        log_var = params[..., d:].clamp(min=min_log_var, max=max_log_var)
        return GaussianDist(mean=mean, log_var=log_var)

    @staticmethod
    def standard_normal(shape: tuple, device: torch.device, dtype: torch.dtype = torch.float32) -> GaussianDist:
        """Create a standard normal N(0, I)."""
        return GaussianDist(
            mean=torch.zeros(shape, device=device, dtype=dtype),
            log_var=torch.zeros(shape, device=device, dtype=dtype),
        )

    def detach(self) -> GaussianDist:
        """Detach from computation graph."""
        return GaussianDist(mean=self.mean.detach(), log_var=self.log_var.detach())

    def to(self, device: torch.device) -> GaussianDist:
        return GaussianDist(mean=self.mean.to(device), log_var=self.log_var.to(device))


# =============================================================================
# Categorical Distribution — For discrete latent states & actions
# =============================================================================

@dataclass
class CategoricalDist:
    """
    Categorical distribution parameterized by logits.

    Used for:
    - Discrete action selection (policy)
    - Categorical latent states (structured world model)
    - Token prediction (text generation)

    Shape: logits is [batch, ..., n_categories]
    """
    logits: Tensor

    @property
    def probs(self) -> Tensor:
        """Normalized probabilities via softmax."""
        return F.softmax(self.logits, dim=-1)

    @property
    def log_probs(self) -> Tensor:
        """Log probabilities via log-softmax (numerically stable)."""
        return F.log_softmax(self.logits, dim=-1)

    @property
    def n_categories(self) -> int:
        return self.logits.shape[-1]

    def sample(self, n_samples: int = 1) -> Tensor:
        """
        Sample from categorical using Gumbel-softmax for differentiability.
        Returns one-hot vectors during forward, hard samples for actual use.
        """
        if n_samples == 1:
            return F.gumbel_softmax(self.logits, tau=1.0, hard=True)
        else:
            samples = []
            for _ in range(n_samples):
                samples.append(F.gumbel_softmax(self.logits, tau=1.0, hard=True))
            return torch.stack(samples, dim=0)

    def sample_index(self) -> Tensor:
        """Sample category indices (non-differentiable)."""
        return torch.multinomial(self.probs, num_samples=1).squeeze(-1)

    def log_prob(self, x: Tensor) -> Tensor:
        """
        Log probability of one-hot x.
        x: one-hot tensor [batch, ..., n_categories]
        """
        return (self.log_probs * x).sum(dim=-1)

    def log_prob_index(self, indices: Tensor) -> Tensor:
        """Log probability of category indices."""
        return self.log_probs.gather(-1, indices.unsqueeze(-1)).squeeze(-1)

    def entropy(self) -> Tensor:
        """
        Entropy H = -Σ p * ln(p).
        Returns per-sample entropy.
        """
        p = self.probs
        log_p = self.log_probs
        return -(p * log_p).sum(dim=-1)

    def kl_divergence(self, other: CategoricalDist) -> Tensor:
        """
        KL(self || other) = Σ p_self * (ln p_self - ln p_other)
        """
        return (self.probs * (self.log_probs - other.log_probs)).sum(dim=-1)

    def kl_from_uniform(self) -> Tensor:
        """KL(self || Uniform) = ln(K) - H(self) where K = n_categories."""
        return math.log(self.n_categories) - self.entropy()

    @staticmethod
    def uniform(shape: tuple, n_categories: int, device: torch.device, dtype: torch.dtype = torch.float32) -> CategoricalDist:
        """Create a uniform categorical distribution."""
        logits = torch.zeros(*shape, n_categories, device=device, dtype=dtype)
        return CategoricalDist(logits=logits)

    def detach(self) -> CategoricalDist:
        return CategoricalDist(logits=self.logits.detach())


# =============================================================================
# Dirichlet Distribution — For beliefs about beliefs (meta-uncertainty)
# =============================================================================

@dataclass
class DirichletDist:
    """
    Dirichlet distribution for modeling concentration parameters.

    Used in structure learning (Layer 6) for maintaining beliefs
    about the parameters of categorical distributions.

    This is how Themis maintains "beliefs about beliefs" — essential
    for Bayesian model selection and structure learning.

    Shape: concentration is [batch, ..., n_categories]
    """
    concentration: Tensor  # α parameters, all > 0

    @property
    def sum_concentration(self) -> Tensor:
        """α₀ = Σ αᵢ"""
        return self.concentration.sum(dim=-1)

    @property
    def mean(self) -> Tensor:
        """Expected value E[X] = α / α₀"""
        return self.concentration / self.sum_concentration.unsqueeze(-1)

    @property
    def n_categories(self) -> int:
        return self.concentration.shape[-1]

    def entropy(self) -> Tensor:
        """
        Entropy of the Dirichlet distribution.
        H = ln B(α) + (α₀ - K)ψ(α₀) - Σ(αᵢ - 1)ψ(αᵢ)
        """
        a = self.concentration
        a0 = self.sum_concentration
        K = self.n_categories
        # Use lgamma for log-Beta function
        log_beta = torch.lgamma(a).sum(dim=-1) - torch.lgamma(a0)
        return (
            log_beta
            + (a0 - K) * torch.digamma(a0)
            - ((a - 1.0) * torch.digamma(a)).sum(dim=-1)
        )

    def kl_divergence(self, other: DirichletDist) -> Tensor:
        """KL(self || other) for Dirichlet distributions."""
        a = self.concentration
        b = other.concentration
        a0 = self.sum_concentration
        b0 = other.sum_concentration

        return (
            torch.lgamma(a0) - torch.lgamma(b0)
            - (torch.lgamma(a) - torch.lgamma(b)).sum(dim=-1)
            + ((a - b) * (torch.digamma(a) - torch.digamma(a0).unsqueeze(-1))).sum(dim=-1)
        )

    def update(self, observations: Tensor) -> DirichletDist:
        """
        Bayesian update: posterior = prior + counts.
        observations: one-hot or count tensor.
        """
        return DirichletDist(concentration=self.concentration + observations)

    def sample(self) -> Tensor:
        """Sample from Dirichlet (returns a probability simplex)."""
        # Use gamma distribution to sample Dirichlet
        gamma_samples = torch.distributions.Gamma(self.concentration, torch.ones_like(self.concentration)).sample()
        return gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)
