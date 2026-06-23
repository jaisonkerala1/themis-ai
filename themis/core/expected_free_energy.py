"""
Themis Core — Expected Free Energy (EFE)

Implements Expected Free Energy (G) computation for policy selection and planning.
For a future time step tau under policy pi:
G(pi, tau) = - Epistemic Value (Information Gain) - Extrinsic Value (Pragmatic Value)

Where:
- Epistemic Value: Mutual Information between future states and future observations.
  Encourages exploration and uncertainty reduction.
- Extrinsic Value: Expected log probability of preferred outcomes (C).
  Encourages goal-directed behavior.
"""

from typing import Callable, Union
import torch
import torch.nn as nn
from torch import Tensor

from themis.core.distributions import GaussianDist, CategoricalDist


def epistemic_value_gaussian(
    q_states: GaussianDist,
    decoded_obs_dists: GaussianDist
) -> Tensor:
    """
    Computes Epistemic Value (Information Gain) for continuous Gaussian variables.
    Approximated as the Mutual Information: I(S; O) = H(O) - H(O | S)
    
    Where:
    - H(O) is the entropy of the marginal predicted observation distribution q(o|pi)
      (approximated using moment matching over sample dimensions).
    - H(O | S) is the expected entropy of observation likelihood p(o|s).
    
    Args:
        q_states: Batched posterior states. Shape: [n_samples, batch, state_dim]
        decoded_obs_dists: Observation distributions predicted from states.
                           Parameters have shape [n_samples, batch, obs_dim]
                           
    Returns:
        epistemic: Information gain per batch item. Shape: [batch]
    """
    # 1. H(O|S): Average entropy of predicted observations across samples
    # decoded_obs_dists.entropy() returns [n_samples, batch]
    expected_conditional_entropy = decoded_obs_dists.entropy().mean(dim=0)  # [batch]
    
    # 2. H(O): Entropy of marginal predicted observations q(o|pi)
    # We approximate the marginal as a single Gaussian via moment matching
    sample_means = decoded_obs_dists.mean  # [n_samples, batch, obs_dim]
    sample_vars = decoded_obs_dists.var    # [n_samples, batch, obs_dim]
    
    # Marginal Mean = E_s[ E[O|s] ]
    marginal_mean = sample_means.mean(dim=0)  # [batch, obs_dim]
    
    # Marginal Variance = E_s[ Var(O|s) ] + Var_s( E[O|s] )
    expected_conditional_var = sample_vars.mean(dim=0)  # [batch, obs_dim]
    var_of_expected_conditional = sample_means.var(dim=0, unbiased=False)  # [batch, obs_dim]
    marginal_var = expected_conditional_var + var_of_expected_conditional
    
    # Construct marginal Gaussian distribution
    marginal_log_var = torch.log(marginal_var.clamp(min=1e-8))
    marginal_dist = GaussianDist(mean=marginal_mean, log_var=marginal_log_var)
    marginal_entropy = marginal_dist.entropy()  # [batch]
    
    # Mutual Information = H(O) - H(O|S)
    # Clamped to >= 0 since MI is mathematically non-negative
    epistemic = (marginal_entropy - expected_conditional_entropy).clamp(min=0.0)
    return epistemic


def extrinsic_value_gaussian(
    decoded_obs_dists: GaussianDist,
    preference: GaussianDist
) -> Tensor:
    """
    Computes Extrinsic (Pragmatic) Value for Gaussian outcomes.
    Value = E_q(o|pi)[ ln p(o | C) ]
    
    Approximated by sampling from the decoded observations and evaluating
    their log probability under the preference distribution.
    
    Args:
        decoded_obs_dists: Predicted observation distributions. Shape: [n_samples, batch, obs_dim]
        preference: Target preference distribution C. Shape: [batch, obs_dim] (or broadcastable)
        
    Returns:
        extrinsic: Pragmatic value per batch item. Shape: [batch]
    """
    # Sample from the observation distributions
    # sample shape: [n_samples, batch, obs_dim]
    o_samples = decoded_obs_dists.sample()
    
    # Evaluate under preference distribution: ln p(o_sample | C)
    # preference.log_prob_total(o_samples) will compute log prob for each sample
    # and sum over last dimension. Shape: [n_samples, batch]
    log_probs = preference.log_prob_total(o_samples)
    
    # Expectation: average over samples
    return log_probs.mean(dim=0)  # [batch]


def expected_free_energy_gaussian(
    q_state: GaussianDist,
    observation_decoder: Callable[[Tensor], GaussianDist],
    preference: GaussianDist,
    n_samples: int = 4,
    extrinsic_weight: float = 1.0,
    epistemic_weight: float = 1.0
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Computes the Expected Free Energy (EFE) for a future step (Gaussian state/obs).
    G = - epistemic_weight * Epistemic - extrinsic_weight * Extrinsic
    
    Args:
        q_state: Predicted posterior over future states. Shape: [batch, state_dim]
        observation_decoder: A function/module that decodes state tensor to a GaussianDist.
        preference: Preference distribution over observations. Shape: [batch, obs_dim]
        n_samples: Number of state samples for expectation.
        extrinsic_weight: Scaling factor for goal-directed value.
        epistemic_weight: Scaling factor for information gain.
        
    Returns:
        G: Expected Free Energy [batch]
        epistemic: Information gain component [batch]
        extrinsic: Pragmatic component [batch]
    """
    # 1. Sample states from prior/predicted belief: s ~ q(s)
    # s_samples shape: [n_samples, batch, state_dim]
    s_samples = q_state.sample(n_samples)
    
    # Reshape to batch-first for decoder efficiency, then recover sample dimension
    n_s, batch_size, state_dim = s_samples.shape
    s_flat = s_samples.view(n_s * batch_size, state_dim)
    
    # 2. Decode states to predicted observations
    decoded_flat = observation_decoder(s_flat)
    
    # Recover original shape parameters
    mean_obs = decoded_flat.mean.view(n_s, batch_size, -1)
    logvar_obs = decoded_flat.log_var.view(n_s, batch_size, -1)
    decoded_obs_dists = GaussianDist(mean=mean_obs, log_var=logvar_obs)
    
    # 3. Compute Epistemic & Extrinsic Values
    epistemic = epistemic_value_gaussian(q_state, decoded_obs_dists)
    extrinsic = extrinsic_value_gaussian(decoded_obs_dists, preference)
    
    # 4. G = - epistemic - extrinsic
    g = - (epistemic_weight * epistemic) - (extrinsic_weight * extrinsic)
    
    return g, epistemic, extrinsic


def expected_free_energy_categorical(
    q_state_logits: Tensor,
    transition_matrix_B: Tensor,  # [batch, state_dim, action_dim, state_dim]
    likelihood_matrix_A: Tensor,  # [batch, obs_dim, state_dim]
    preference_C: Tensor,          # [batch, obs_dim] (log probabilities)
    action_index: int
) -> Tensor:
    """
    Analytical EFE for Discrete Categorical states & observations (standard pymdp style).
    
    G(u) = sum_s' ( q(s' | u) * [ ln q(s' | u) - ln q(s' | o, u) - ln p(o | C) ] )
    
    Args:
        q_state_logits: Logits of current state belief [batch, state_dim]
        transition_matrix_B: State transition matrices [batch, state_dim, action_dim, state_dim]
        likelihood_matrix_A: Observation likelihood [batch, obs_dim, state_dim]
        preference_C: Pref log probability [batch, obs_dim]
        action_index: Selected action
        
    Returns:
        g: EFE scalar value per batch item [batch]
    """
    # 1. State belief at t
    q_s = torch.softmax(q_state_logits, dim=-1)  # [batch, state_dim]
    
    # 2. Transition under selected action -> predicted future state q(s_tau | u)
    # transition B is [batch, next_state, action, current_state]
    B_a = transition_matrix_B[:, :, action_index, :]  # [batch, state_dim, state_dim]
    q_s_next = torch.bmm(B_a, q_s.unsqueeze(-1)).squeeze(-1)  # [batch, state_dim]
    
    # 3. Predicted future observation q(o_tau | u) = A * q(s_tau | u)
    # likelihood_matrix_A is [batch, obs_dim, state_dim]
    q_o_next = torch.bmm(likelihood_matrix_A, q_s_next.unsqueeze(-1)).squeeze(-1)  # [batch, obs_dim]
    # Clamping for logs
    q_o_next_log = torch.log(q_o_next.clamp(min=1e-8))
    
    # 4. Pragmatic (Extrinsic) Value: sum_o ( q(o|u) * C(o) )
    # preference_C is already in log space
    extrinsic = (q_o_next * preference_C).sum(dim=-1)  # [batch]
    
    # 5. Epistemic Value (Mutual Information)
    # sum_s ( q(s) * sum_o ( A_os * (ln A_os - ln q(o|u)) ) )
    # Compute: ln A - ln q(o|u) for each s and o
    A_log = torch.log(likelihood_matrix_A.clamp(min=1e-8))  # [batch, obs_dim, state_dim]
    # Broadcast subtraction: log_A - log_q_o [batch, obs_dim, state_dim]
    info_gain = A_log - q_o_next_log.unsqueeze(-1)
    # Expected information gain over likelihood: sum_o ( A_os * info_gain )
    expected_info_gain = (likelihood_matrix_A * info_gain).sum(dim=1)  # [batch, state_dim]
    # Expectation over predicted states: sum_s ( q(s_next) * expected_info_gain )
    epistemic = (q_s_next * expected_info_gain).sum(dim=-1)  # [batch]
    
    # 6. G = - Epistemic - Extrinsic
    return - epistemic - extrinsic
