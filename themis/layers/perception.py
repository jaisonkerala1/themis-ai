"""
Themis Layers — Layer 2: Perception Engine

Implements hierarchical predictive coding/variational inference.
Updates beliefs about hidden states (z) by running an inner-loop optimizer
to minimize the total Variational Free Energy (VFE).
"""

from typing import List, Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from themis.config import ThemisConfig
from themis.core.distributions import GaussianDist
from themis.core.free_energy import variational_free_energy


class PerceptionEngine(nn.Module):
    """
    Layer 2: Perception Engine

    Runs variational inference to update posterior beliefs q(z) at all levels
    of the hierarchy given the current sensory observations.
    
    Instead of analytical message-passing, it uses PyTorch autograd and an inner-loop
    optimizer to directly minimize the Variational Free Energy.
    """
    def __init__(self, config: ThemisConfig):
        super().__init__()
        self.config = config
        self.pc_config = config.perception
        
    def update_beliefs(
        self,
        world_model: nn.Module,
        observation: GaussianDist,
        prev_states: List[Dict[str, Tensor]],
        action: Optional[Tensor] = None,
        n_iterations: Optional[int] = None
    ) -> Tuple[List[GaussianDist], Dict[str, Any]]:
        """
        Runs the perception optimization loop to update beliefs.
        
        Args:
            world_model: The Layer 3 World Model, which provides transition priors
                         and observation likelihoods.
            observation: GaussianDist representing current sensory inputs q(z_sensory).
            prev_states: List of state dictionaries from the previous step [Level 1, Level 2, Level 3].
                         Each dict contains: {"h": Tensor, "z": Tensor}
            action: Selected action from the previous step.
            n_iterations: Number of optimization iterations (defaults to config value).
            
        Returns:
            posteriors: List[GaussianDist] representing the inferred posterior q(z^i_t) for each level.
            metrics: Dictionary of training/inference metrics (VFE, complexity, accuracy).
        """
        device = self.config.resolve_device()
        dtype = self.config.resolve_dtype()
        
        if n_iterations is None:
            n_iterations = self.pc_config.n_iterations
            
        # 1. Compute deterministic hidden states (h^i_t) and prior transition distributions p(z^i_t | h^i_t)
        # using the world model's prior step. This is context that remains fixed during perception.
        # h_states: list of h tensors per level.
        # priors: list of GaussianDist representing transition priors p(z^i_t | h^i_t) per level.
        observation = observation.detach()
        with torch.no_grad():
            h_states, priors = world_model.compute_priors(prev_states, action)
            h_states = [h.detach() for h in h_states]
            priors = [p.detach() for p in priors]
        
        # 2. Instantiate learnable posterior parameters for each level's stochastic state q(z^i_t).
        # We initialize the mean and log_var of the posterior at the prior's mean/log_var (carrying expectation forward).
        # We make these leaf tensors with requires_grad=True.
        post_means = []
        post_log_vars = []
        
        for i, prior in enumerate(priors):
            # We clone and detach to avoid backpropagating through prior initialization
            mean_param = prior.mean.clone().detach().requires_grad_(True)
            log_var_param = prior.log_var.clone().detach().requires_grad_(True)
            
            post_means.append(mean_param)
            post_log_vars.append(log_var_param)
            
        # 3. Setup inner-loop optimizer over the posterior parameters
        # Local Adam optimizer with configured belief learning rate
        optimizer = optim.Adam(post_means + post_log_vars, lr=self.pc_config.learning_rate_mu)
        
        # 4. Perception loop (minimizing VFE)
        vfe_history = []
        
        for step in range(n_iterations):
            optimizer.zero_grad()
            
            # Construct posterior distributions from current parameter states
            posteriors = [
                GaussianDist(mean=m, log_var=lv)
                for m, lv in zip(post_means, post_log_vars)
            ]
            
            # Compute VFE:
            # VFE = Sum_i KL( q(z^i_t) || p(z^i_t | h^i_t) ) - Expected Log Likelihood p(o_t | z^1_t, h^1_t)
            
            # A. Complexity: KL divergences across levels
            complexity_terms = [
                q.kl_divergence(p) for q, p in zip(posteriors, priors)
            ]
            total_complexity = torch.stack(complexity_terms, dim=0).sum(dim=0) # [batch]
            
            # B. Accuracy: Expectation of observation log_prob under q(z^1_t)
            # Sample from Level 1 posterior: z^1_t ~ q(z^1_t)
            # Using reparameterization trick so gradients flow back to posterior params
            z1_sample = posteriors[0].sample() # [batch, z_dim]
            h1 = h_states[0] # [batch, h_dim]
            
            # Predict observation distribution from Level 1 state
            predicted_obs_dist = world_model.likelihood_decoder(z1_sample, h1) # GaussianDist
            
            # Compute log likelihood of the true observation (sensory input)
            # The observation is a GaussianDist, we compute log prob under predicted_obs_dist
            # We sample from observation to get concrete values and evaluate
            obs_sample = observation.mean # use mean representation as observation target
            accuracy = predicted_obs_dist.log_prob_total(obs_sample) # [batch]
            
            # C. Total Variational Free Energy: F = Complexity - Accuracy
            vfe = total_complexity - accuracy # [batch]
            
            # Mean VFE over batch for scalar loss
            loss = vfe.mean()
            
            # Backpropagate to compute gradients on posterior parameters
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Track progress
            vfe_history.append(loss.item())
            
        # 5. Build final posteriors and return metrics
        final_posteriors = [
            GaussianDist(mean=m.detach(), log_var=lv.detach())
            for m, lv in zip(post_means, post_log_vars)
        ]
        
        metrics = {
            "vfe_start": vfe_history[0],
            "vfe_end": vfe_history[-1],
            "vfe_history": vfe_history,
            "complexity": total_complexity.mean().item(),
            "accuracy": accuracy.mean().item()
        }
        
        return final_posteriors, metrics
