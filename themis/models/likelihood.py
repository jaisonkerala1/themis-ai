"""
Themis Models — Likelihood Decoder

Implements the Likelihood model (equivalent to the A matrix)
which projects latent states to predict observation distributions.
"""

import torch
import torch.nn as nn
from torch import Tensor

from themis.config import ThemisConfig
from themis.core.distributions import GaussianDist


class LikelihoodDecoder(nn.Module):
    """
    Likelihood Decoder

    Predicts the sensory observation distribution from the Level 1 latent state.
    Maps: (z^1_t, h^1_t) -> p(o_t | z^1_t, h^1_t)
    """
    def __init__(self, config: ThemisConfig):
        super().__init__()
        self.config = config
        wm_config = config.world_model
        dims = config.dims
        
        self.z_dim = wm_config.state_dim_stochastic
        self.h_dim = wm_config.state_dim_deterministic
        self.obs_dim = dims.obs_embed_dim
        self.hidden_dim = wm_config.hidden_dim
        
        # Mapping from level 1 state (z_dim + h_dim) to observation parameters (obs_dim * 2)
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim + self.h_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.obs_dim * 2)
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.constant_(layer.bias, 0.0)
                
        # Initialize observation log_var bias to start around -2.0 for stability
        with torch.no_grad():
            self.decoder[-1].bias[self.obs_dim:].fill_(-2.0)
            
    def forward(self, z: Tensor, h: Tensor) -> GaussianDist:
        """
        Args:
            z: Level 1 stochastic state [batch_size, z_dim]
            h: Level 1 deterministic state [batch_size, h_dim]
        Returns:
            pred_obs: GaussianDist representing the predicted observation embedding [batch_size, obs_dim]
        """
        state = torch.cat([z, h], dim=-1)
        params = self.decoder(state)
        return GaussianDist.from_params(params)
