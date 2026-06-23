"""
Themis Layers — Layer 4: Planning Engine

Implements Expected Free Energy (EFE) based planning as inference.
Generates candidate action sequences using an amortized policy,
simulates their trajectories in the world model, and selects policies
by minimizing Expected Free Energy.
"""

from typing import List, Dict, Tuple, Optional, Callable, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from themis.config import ThemisConfig
from themis.core.distributions import GaussianDist, CategoricalDist
from themis.core.expected_free_energy import epistemic_value_gaussian, extrinsic_value_gaussian


class AmortizedPolicy(nn.Module):
    """
    Amortized Policy Network

    A lightweight actor network that proposes candidate actions (token IDs)
    given the current belief states. Used to bootstrap policy search.
    """
    def __init__(self, config: ThemisConfig):
        super().__init__()
        self.config = config
        wm_config = config.world_model
        dims = config.dims
        
        self.z_dim = wm_config.state_dim_stochastic
        self.h_dim = wm_config.state_dim_deterministic
        self.hidden_dim = wm_config.hidden_dim
        
        # Policy maps Level 1 state (z + h) to vocabulary action logits
        self.net = nn.Sequential(
            nn.Linear(self.z_dim + self.h_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, dims.vocab_size)
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, z: Tensor, h: Tensor) -> CategoricalDist:
        """
        Predicts action distribution.
        """
        state = torch.cat([z, h], dim=-1)
        logits = self.net(state)
        return CategoricalDist(logits=logits)


class PlanningEngine(nn.Module):
    """
    Layer 4: Planning Engine

    Evaluates and selects actions by minimizing Expected Free Energy (EFE)
    over a planning horizon.
    """
    def __init__(self, config: ThemisConfig, world_model: nn.Module):
        super().__init__()
        self.config = config
        self.world_model = world_model
        self.plan_config = config.planning
        
        # Amortized policy for action proposals
        self.amortized_policy = AmortizedPolicy(config)
        
    def forward(
        self,
        current_states: List[Dict[str, Tensor]],
        preference: Optional[GaussianDist] = None
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Plans next action given current belief states.
        
        Args:
            current_states: List of state dictionaries [Level 1, Level 2, Level 3].
            preference: Optional target preference distribution.
            
        Returns:
            best_actions: Discrete token action IDs [batch_size]
            metrics: Dictionary of planning metrics.
        """
        batch_size = current_states[0]["h"].shape[0]
        device = self.config.resolve_device()
        dtype = self.config.resolve_dtype()
        
        K = self.plan_config.n_candidate_policies  # e.g., 32
        M = self.plan_config.n_policy_samples      # e.g., 4
        H = self.plan_config.planning_horizon      # e.g., 8
        
        # 1. Expand current states to size [batch_size, K, M] to run parallel rollouts
        # We flatten this to [batch_size * K * M] for parallel batch dynamics in the world model.
        flat_size = batch_size * K * M
        
        flat_states = []
        for state_dict in current_states:
            h_expanded = state_dict["h"].unsqueeze(1).unsqueeze(2).expand(-1, K, M, -1)
            h_flat = h_expanded.reshape(flat_size, -1)
            
            z_expanded = state_dict["z"].unsqueeze(1).unsqueeze(2).expand(-1, K, M, -1)
            z_flat = z_expanded.reshape(flat_size, -1)
            
            flat_states.append({"h": h_flat, "z": z_flat})
            
        # 2. Planning horizon simulation loop
        # We track EFE (G) accumulated over the horizon for each of the K candidate policies.
        # G has shape [batch_size, K]
        G_total = torch.zeros(batch_size, K, device=device, dtype=torch.float32)
        log_prob_total = torch.zeros(batch_size, K, device=device, dtype=torch.float32)
        
        # We store the first action sampled for each candidate policy.
        # shape [batch_size, K]
        first_actions = None
        
        state = flat_states
        
        for t in range(H):
            # A. Propose actions using the amortized policy based on Level 1 state
            # shape: [flat_size, vocab_size]
            policy_dist = self.amortized_policy(state[0]["z"], state[0]["h"])
            
            if t == 0:
                # Sample K actions per batch item: shape [batch_size, K]
                z_init = current_states[0]["z"]
                h_init = current_states[0]["h"]
                policy_dist_init = self.amortized_policy(z_init, h_init)
                first_actions = torch.multinomial(policy_dist_init.probs, num_samples=K, replacement=True)
                
                # Expand first_actions to [batch_size, K, M] and then flatten to [flat_size]
                action_ids = first_actions.unsqueeze(2).expand(-1, -1, M).reshape(flat_size)
            else:
                # Sample discrete action IDs [flat_size]
                action_ids = policy_dist.sample_index()
            
            # Get log probability of the sampled action under the amortized policy
            log_prob = policy_dist.log_prob_index(action_ids)
            log_prob_step = log_prob.view(batch_size, K, M).mean(dim=-1) # [batch_size, K]
            log_prob_total += log_prob_step
                
            # B. Simulate next step in the world model (transition B)
            next_state, priors = self.world_model.imagine_step(state, action_ids, use_mean=True)
            
            # C. Decode expected observations (likelihood A)
            # pred_obs shape parameters: [flat_size, obs_embed_dim]
            pred_obs_dist = self.world_model.likelihood_decoder(next_state[0]["z"], next_state[0]["h"])
            
            # D. Reshape variables to separate Monte Carlo sample dimension M
            # next_z_samples shape: [M, batch_size * K, z_dim]
            z_dim = next_state[0]["z"].shape[-1]
            next_z_samples = next_state[0]["z"].view(batch_size * K, M, z_dim).transpose(0, 1)
            
            # pred_obs_mean/var shape: [M, batch_size * K, obs_dim]
            obs_dim = pred_obs_dist.mean.shape[-1]
            obs_mean = pred_obs_dist.mean.view(batch_size * K, M, obs_dim).transpose(0, 1)
            obs_log_var = pred_obs_dist.log_var.view(batch_size * K, M, obs_dim).transpose(0, 1)
            
            # Build structures representing the sampled paths
            q_state_samples = GaussianDist(
                mean=next_z_samples,
                log_var=torch.zeros_like(next_z_samples) # dummy logvar for MI function
            )
            decoded_obs_samples = GaussianDist(mean=obs_mean, log_var=obs_log_var)
            
            # E. Compute EFE (Epistemic + Extrinsic)
            # Epistemic: Information gain (Mutual Information)
            # shape: [batch_size * K]
            epistemic = epistemic_value_gaussian(q_state_samples, decoded_obs_samples)
            
            # Extrinsic: Pragmatic value (Satisfaction of goal preference C)
            # Get preference distribution C
            if preference is not None:
                # preference has shape [batch_size, obs_embed_dim]
                # We want to expand it to [batch_size, K, obs_embed_dim] and then reshape to [batch_size * K, obs_embed_dim]
                pref_mean = preference.mean.unsqueeze(1).expand(-1, K, -1).reshape(batch_size * K, -1)
                pref_log_var = preference.log_var.unsqueeze(1).expand(-1, K, -1).reshape(batch_size * K, -1)
                pref = GaussianDist(mean=pref_mean, log_var=pref_log_var)
            else:
                pref = self.world_model.get_preference(batch_size * K)
            # shape: [batch_size * K]
            extrinsic = extrinsic_value_gaussian(decoded_obs_samples, pref)
            
            # EFE G = - epistemic - extrinsic
            # shape: [batch_size * K]
            g_step = - (self.plan_config.efe_epistemic_weight * epistemic) - (self.plan_config.efe_extrinsic_weight * extrinsic)
            
            # Accumulate
            G_total += g_step.view(batch_size, K)
            
            # Update state for next planning step
            state = next_state
            
        # 3. Policy selection via softmax over logits (log_prob_prior - G / temp)
        # policy_probs shape: [batch_size, K]
        policy_logits = log_prob_total - (G_total / self.plan_config.temperature)
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        # Sample policy index for each batch item
        # policy_idx shape: [batch_size]
        policy_idx = torch.multinomial(policy_probs, num_samples=1).squeeze(-1)
        
        # Extract the chosen action (first step action of the chosen policy)
        # best_actions shape: [batch_size]
        best_actions = first_actions.gather(1, policy_idx.unsqueeze(-1)).squeeze(-1)
        
        metrics = {
            "G_min": G_total.min().item(),
            "G_mean": G_total.mean().item(),
            "selected_policy_probs": policy_probs.gather(1, policy_idx.unsqueeze(-1)).mean().item()
        }
        
        return best_actions, metrics
