"""
Themis Models — Recurrent State-Space Level

Defines the single-level RSSM component, combining a deterministic RNN/GRU state (h)
with a stochastic latent state (z).
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch import Tensor

from themis.config import ThemisConfig
from themis.core.distributions import GaussianDist


class StateSpaceLevel(nn.Module):
    """
    A single level of the Hierarchical Recurrent State-Space Model.
    
    Deterministic state (h_t) is updated using a GRU cell.
    Stochastic state (z_t) prior is predicted from h_t.
    
    Variables:
    - h_t: deterministic state, shape [batch_size, deterministic_dim]
    - z_t: stochastic state (sampled), shape [batch_size, stochastic_dim]
    """
    def __init__(
        self,
        config: ThemisConfig,
        level_idx: int,
        context_dim: int = 0
    ):
        super().__init__()
        self.config = config
        self.level_idx = level_idx
        
        wm_config = config.world_model
        self.z_dim = wm_config.state_dim_stochastic
        self.h_dim = wm_config.state_dim_deterministic
        self.hidden_dim = wm_config.hidden_dim
        
        # Action input is only provided at the lowest level (Level 1) or can be passed to all
        # To make it uniform, Level 1 takes action_dim, others take 0 action input
        self.action_dim = wm_config.action_dim if level_idx == 0 else 0
        
        # Input to GRU cell: previous z + previous action (if level 0) + context from higher levels
        gru_input_size = self.z_dim + self.action_dim + context_dim
        
        self.gru = nn.GRUCell(
            input_size=gru_input_size,
            hidden_size=self.h_dim
        )
        
        # Transition Prior Net: h_t -> p(z_t | h_t) parameters (mean, log_var)
        self.prior_net = nn.Sequential(
            nn.Linear(self.h_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.z_dim * 2)
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize GRU and linear layers
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
                
        for layer in self.prior_net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.constant_(layer.bias, 0.0)
                
        # Clamping helpers: bias log_var to start around -2.0
        with torch.no_grad():
            self.prior_net[-1].bias[self.z_dim:].fill_(-2.0)
            
    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        """Returns zeroed initial states (h_0, z_0)."""
        h = torch.zeros(batch_size, self.h_dim, device=device, dtype=dtype)
        z = torch.zeros(batch_size, self.z_dim, device=device, dtype=dtype)
        return h, z

    def forward(
        self,
        prev_h: Tensor,
        prev_z: Tensor,
        action: Optional[Tensor] = None,
        context: Optional[Tensor] = None
    ) -> Tuple[Tensor, GaussianDist]:
        """
        Updates deterministic hidden state h_t and predicts the stochastic state prior p(z_t | h_t).
        
        Args:
            prev_h: Previous deterministic state [batch_size, h_dim]
            prev_z: Previous stochastic state [batch_size, z_dim]
            action: Selected action [batch_size, action_dim] (only used if level_idx == 0)
            context: Top-down context [batch_size, context_dim] (from level above)
            
        Returns:
            h: Updated deterministic state [batch_size, h_dim]
            prior: GaussianDist representing prior p(z_t | h_t) [batch_size, z_dim]
        """
        inputs = [prev_z]
        
        # Add action input for Level 1
        if self.level_idx == 0 and action is not None:
            inputs.append(action)
        elif self.level_idx == 0:
            # Zero action fallback
            inputs.append(torch.zeros(prev_h.shape[0], self.action_dim, device=prev_h.device, dtype=prev_h.dtype))
            
        # Add top-down context if provided
        if context is not None:
            inputs.append(context)
            
        gru_input = torch.cat(inputs, dim=-1)
        
        # Update deterministic hidden state
        h = self.gru(gru_input, prev_h)
        
        # Predict prior over stochastic state
        prior_params = self.prior_net(h)
        prior = GaussianDist.from_params(prior_params)
        
        return h, prior
