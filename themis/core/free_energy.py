"""
Themis Core — Variational Free Energy (VFE)

Implements the Variational Free Energy computation:
F = Complexity - Accuracy
  = D_KL[q(s) || p(s)] - E_q[ln p(o|s)]

Where q(s) is the agent's current belief (approximate posterior),
p(s) is the prior expectation, and p(o|s) is the likelihood of observations.
Minimizing F is mathematically equivalent to maximizing the model evidence (ln p(o)).
"""

from typing import Union
import torch
from torch import Tensor

from themis.core.distributions import GaussianDist, CategoricalDist, DirichletDist


def variational_free_energy(
    q: Union[GaussianDist, CategoricalDist, DirichletDist],
    p: Union[GaussianDist, CategoricalDist, DirichletDist],
    log_likelihood: Tensor
) -> Tensor:
    """
    Computes generic Variational Free Energy:
    F = KL(q || p) - log_likelihood
    
    Args:
        q: Current belief distribution (approximate posterior)
        p: Prior/predicted belief distribution
        log_likelihood: Log-likelihood of observation(s) under belief states q,
                        typically computed as p(o|s) for s sampled from q.
                        Shape: [batch, ...]
    
    Returns:
        F: Free energy value per sample. Shape: [batch]
    """
    # Complexity: D_KL[q || p]
    complexity = q.kl_divergence(p)  # Shape: [batch]
    
    # F = Complexity - Accuracy
    return complexity - log_likelihood


def free_energy_gaussian(
    q_state: GaussianDist,
    p_state: GaussianDist,
    observation: Tensor,
    observation_model_decoder: torch.nn.Module,
    n_samples: int = 1
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Computes VFE for a continuous Gaussian state-space model:
    F = Complexity - Accuracy
      = KL(q_state || p_state) - E_q[ln p(o | s)]
      
    Approximates the expectation E_q[ln p(o|s)] via Monte Carlo sampling from q_state.
    
    Args:
        q_state: Posterior over latent states.
        p_state: Prior/predicted latent states.
        observation: True sensory observation.
        observation_model_decoder: A model that takes states s and returns an observation distribution
                                   (e.g., GaussianDist representing p(o|s)).
        n_samples: Number of Monte Carlo samples to approximate expected log-likelihood.
        
    Returns:
        F: Variational Free Energy [batch]
        complexity: KL divergence [batch]
        accuracy: Expected log-likelihood [batch]
    """
    # 1. Complexity: D_KL[q(s) || p(s)]
    complexity = q_state.kl_divergence(p_state)  # [batch]
    
    # 2. Accuracy: E_q[ln p(o|s)] via Monte Carlo
    if n_samples == 1:
        # Sample state from posterior: s ~ q(s)
        s_sample = q_state.sample()  # [batch, state_dim]
        # Decode state to observation distribution: p(o|s)
        pred_obs_dist = observation_model_decoder(s_sample)  # GaussianDist or CategoricalDist
        # Compute log-likelihood of true observation: ln p(o|s)
        accuracy = pred_obs_dist.log_prob_total(observation)  # [batch]
    else:
        # Multiple samples for lower variance estimate
        s_samples = q_state.sample(n_samples)  # [n_samples, batch, state_dim]
        log_probs = []
        for i in range(n_samples):
            pred_obs_dist = observation_model_decoder(s_samples[i])
            log_probs.append(pred_obs_dist.log_prob_total(observation))  # [batch]
        accuracy = torch.stack(log_probs, dim=0).mean(dim=0)  # [batch]
        
    # F = Complexity - Accuracy
    f = complexity - accuracy
    return f, complexity, accuracy


def free_energy_predictive_coding(
    observation: Tensor,
    predicted_obs: GaussianDist,
    state_q: GaussianDist,
    state_p: GaussianDist
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Computes VFE specifically for Gaussian predictive coding (Layer 2).
    In predictive coding, the VFE represents the sum of prediction errors:
    F = Complexity (state error) + Accuracy Error (sensory error)
      = KL(q_state || p_state) - ln p(obs | q_state)
    
    Using the Laplace approximation (where states are evaluated at their mean):
    F = KL(q_state || p_state) + 0.5 * sum( precision_obs * (obs - mean_obs)^2 + log_var_obs + ln(2pi) )
    
    Args:
        observation: Actual sensory input.
        predicted_obs: Predicted observation distribution (e.g. from state mean).
        state_q: Current belief distribution (approximate posterior).
        state_p: Top-down prior expectation distribution.
        
    Returns:
        F: Free energy value per sample [batch]
        complexity: State prediction error (KL) [batch]
        accuracy_error: Sensory prediction error (negative log prob) [batch]
    """
    complexity = state_q.kl_divergence(state_p)  # [batch]
    accuracy_error = -predicted_obs.log_prob_total(observation)  # [batch]
    
    f = complexity + accuracy_error
    return f, complexity, accuracy_error
