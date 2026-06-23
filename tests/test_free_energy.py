import math
import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings, strategies as st

from themis.core import (
    GaussianDist,
    CategoricalDist,
    kl_gaussian,
    kl_categorical,
    entropy_gaussian,
    entropy_categorical,
    variational_free_energy,
    free_energy_gaussian,
    free_energy_predictive_coding,
    epistemic_value_gaussian,
    extrinsic_value_gaussian,
    expected_free_energy_gaussian,
    expected_free_energy_categorical
)


# =============================================================================
# Helper function to generate clean tensor ranges for tests
# =============================================================================
def generate_tensors(batch: int, dim: int):
    # Generates deterministic yet diverse tensors for validation
    mean = torch.linspace(-2.0, 2.0, steps=batch*dim).view(batch, dim)
    logvar = torch.linspace(-3.0, 1.0, steps=batch*dim).view(batch, dim)
    return mean, logvar


# =============================================================================
# 1. Distribution Tests
# =============================================================================

def test_gaussian_dist_properties():
    # Simple check on GaussianDist constructor and properties
    mean = torch.tensor([[0.0, 1.0], [2.0, -1.0]])
    log_var = torch.tensor([[0.0, -2.0], [1.0, -0.5]])
    
    q = GaussianDist(mean, log_var)
    
    assert torch.allclose(q.std, torch.exp(0.5 * log_var))
    assert torch.allclose(q.var, torch.exp(log_var))
    assert torch.allclose(q.precision, 1.0 / q.var)
    assert q.dim == 2
    
    # Check sampling shape
    assert q.sample(1).shape == q.mean.shape
    assert q.sample(5).shape == (5, 2, 2)


def test_gaussian_kl_vs_functional():
    # Verify class method kl_divergence matches functional utility kl_gaussian
    mean_q, logvar_q = generate_tensors(4, 8)
    mean_p, logvar_p = generate_tensors(4, 8)
    
    q = GaussianDist(mean_q, logvar_q)
    p = GaussianDist(mean_p, logvar_p)
    
    kl_method = q.kl_divergence(p)
    kl_func = kl_gaussian(mean_q, logvar_q, mean_p, logvar_p, dim=-1)
    
    assert torch.allclose(kl_method, kl_func, atol=1e-5)
    # KL to standard normal helper
    std_normal = GaussianDist.standard_normal(q.mean.shape, q.mean.device)
    assert torch.allclose(q.kl_divergence(std_normal), q.kl_from_standard_normal(), atol=1e-5)


def test_gaussian_entropy():
    # Verify GaussianDist entropy matches functional entropy_gaussian
    mean, logvar = generate_tensors(2, 4)
    q = GaussianDist(mean, logvar)
    
    assert torch.allclose(q.entropy(), entropy_gaussian(logvar, dim=-1), atol=1e-5)


def test_categorical_dist_properties():
    logits = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, -5.0]])
    q = CategoricalDist(logits)
    
    assert q.n_categories == 3
    assert torch.allclose(q.probs.sum(dim=-1), torch.ones(2))
    assert torch.allclose(q.log_probs, torch.log_softmax(logits, dim=-1))
    
    # Entropy matches functional entropy_categorical
    assert torch.allclose(q.entropy(), entropy_categorical(logits, dim=-1))


def test_categorical_kl_divergence():
    logits_q = torch.tensor([[2.0, 1.0, 0.0], [0.5, 0.5, 0.5]])
    logits_p = torch.tensor([[1.0, 1.0, 1.0], [0.1, 0.2, 0.7]])
    
    q = CategoricalDist(logits_q)
    p = CategoricalDist(logits_p)
    
    kl_method = q.kl_divergence(p)
    kl_func = kl_categorical(logits_q, logits_p, dim=-1)
    
    assert torch.allclose(kl_method, kl_func, atol=1e-5)


# =============================================================================
# 2. Variational Free Energy (VFE) Tests
# =============================================================================

class MockDecoder(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        
    def forward(self, state: torch.Tensor) -> GaussianDist:
        # Simple projection from state to observation mean
        # Let's say state_dim maps to obs_dim
        batch_size = state.shape[0]
        # Just return state slice or pad
        if state.shape[-1] >= self.obs_dim:
            mean = state[..., :self.obs_dim]
        else:
            mean = torch.cat([state, torch.zeros(batch_size, self.obs_dim - state.shape[-1], device=state.device)], dim=-1)
        log_var = torch.zeros_like(mean)
        return GaussianDist(mean, log_var)


def test_variational_free_energy_gaussian():
    # Verify free_energy_gaussian implementation details
    batch, state_dim, obs_dim = 2, 4, 3
    
    # Using extremely low variance to make the state samples deterministic (practically equal to mean)
    q_state = GaussianDist(torch.zeros(batch, state_dim), torch.full((batch, state_dim), -30.0))
    p_state = GaussianDist(torch.zeros(batch, state_dim), torch.zeros(batch, state_dim))
    observation = torch.zeros(batch, obs_dim)
    
    decoder = MockDecoder(obs_dim)
    
    f, complexity, accuracy = free_energy_gaussian(
        q_state=q_state,
        p_state=p_state,
        observation=observation,
        observation_model_decoder=decoder,
        n_samples=5
    )
    
    # q ~ N(0, e^-30), p ~ N(0, 1)
    # KL = 0.5 * sum( 0 - (-30) + e^-30 - 1 ) = 0.5 * 4 * (30 - 1) = 58.0
    expected_complexity = 58.0
    assert torch.allclose(complexity, torch.tensor([expected_complexity, expected_complexity]), atol=1e-4)
    
    # Since the state samples are practically 0, the decoder predicts mean=0, logvar=0.
    # Logprob of 0 under standard normal N(0, I) per dimension is -0.5 * ln(2pi)
    # Total logprob = 3 * -0.5 * ln(2pi) = -1.5 * ln(2pi)
    expected_accuracy = -1.5 * math.log(2 * math.pi)
    assert torch.allclose(accuracy, torch.tensor([expected_accuracy, expected_accuracy]), atol=1e-2)
    
    # F = Complexity - Accuracy
    assert torch.allclose(f, complexity - accuracy, atol=1e-4)


def test_predictive_coding_vfe():
    batch, dim = 2, 4
    obs = torch.ones(batch, dim)
    pred_mean = torch.zeros(batch, dim)
    pred_logvar = torch.zeros(batch, dim)
    predicted_obs = GaussianDist(pred_mean, pred_logvar)
    
    q_state = GaussianDist(torch.ones(batch, dim), torch.zeros(batch, dim))
    p_state = GaussianDist(torch.zeros(batch, dim), torch.zeros(batch, dim))
    
    f, complexity, accuracy_error = free_energy_predictive_coding(
        observation=obs,
        predicted_obs=predicted_obs,
        state_q=q_state,
        state_p=p_state
    )
    
    # check relations
    assert torch.allclose(f, complexity + accuracy_error, atol=1e-5)


# =============================================================================
# 3. Expected Free Energy (EFE) Tests
# =============================================================================

def test_expected_free_energy_gaussian():
    batch, state_dim, obs_dim = 2, 4, 3
    q_state = GaussianDist(torch.zeros(batch, state_dim), torch.zeros(batch, state_dim))
    decoder = MockDecoder(obs_dim)
    preference = GaussianDist(torch.zeros(batch, obs_dim), torch.zeros(batch, obs_dim))
    
    g, epistemic, extrinsic = expected_free_energy_gaussian(
        q_state=q_state,
        observation_decoder=decoder,
        preference=preference,
        n_samples=10,
        extrinsic_weight=1.0,
        epistemic_weight=1.0
    )
    
    # Output shapes
    assert g.shape == (batch,)
    assert epistemic.shape == (batch,)
    assert extrinsic.shape == (batch,)
    
    # Check that Epistemic value is >= 0
    assert (epistemic >= -1e-5).all()
    # Check EFE decomposition: G = -Epistemic - Extrinsic
    assert torch.allclose(g, -epistemic - extrinsic, atol=1e-5)


def test_expected_free_energy_categorical():
    batch, state_dim, obs_dim, action_dim = 2, 3, 2, 2
    
    # Belied over state
    q_state_logits = torch.ones(batch, state_dim)
    
    # Transition: next_state, action, current_state
    transition_B = torch.softmax(torch.randn(batch, state_dim, action_dim, state_dim), dim=1)
    
    # Likelihood: obs, state
    likelihood_A = torch.softmax(torch.randn(batch, obs_dim, state_dim), dim=1)
    
    # Preference: logs over observations
    preference_C = torch.log(torch.tensor([[0.8, 0.2], [0.5, 0.5]]))
    
    g = expected_free_energy_categorical(
        q_state_logits=q_state_logits,
        transition_matrix_B=transition_B,
        likelihood_matrix_A=likelihood_A,
        preference_C=preference_C,
        action_index=0
    )
    
    assert g.shape == (batch,)


# =============================================================================
# 4. Property-Based Testing (Hypothesis)
# =============================================================================

@given(
    mean_q=st.floats(min_value=-5.0, max_value=5.0),
    logvar_q=st.floats(min_value=-3.0, max_value=1.5),
    mean_p=st.floats(min_value=-5.0, max_value=5.0),
    logvar_p=st.floats(min_value=-3.0, max_value=1.5)
)
@settings(max_examples=50, deadline=None)
def test_gaussian_kl_properties_property_based(mean_q, logvar_q, mean_p, logvar_p):
    q_m = torch.tensor([mean_q])
    q_l = torch.tensor([logvar_q])
    p_m = torch.tensor([mean_p])
    p_l = torch.tensor([logvar_p])
    
    kl = kl_gaussian(q_m, q_l, p_m, p_l, dim=-1)
    
    # KL is always non-negative: KL(q || p) >= 0
    assert kl.item() >= -1e-5
    
    # KL is zero if and only if q == p
    kl_self = kl_gaussian(q_m, q_l, q_m, q_l, dim=-1)
    assert abs(kl_self.item()) < 1e-4
