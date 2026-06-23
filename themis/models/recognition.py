"""
Themis Models — Recognition Network (Bottom-Up Inference)

Implements amortized inference via recognition networks that map observations
directly to posterior parameters in a single forward pass.

This is the modern approach used in VAEs and Deep Active Inference, avoiding
the computational cost and numerical instability of iterative optimization.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from themis.config import ThemisConfig
from themis.core.distributions import GaussianDist


class RecognitionNetwork(nn.Module):
    """
    Amortized recognition network for single-step posterior inference.
    
    Maps: (observation, h_state, prior) -> posterior parameters
    
    This implements the "encoder" in VAE terminology, performing bottom-up
    inference from observations to latent beliefs in a single feedforward pass.
    """
    
    def __init__(self, config: ThemisConfig, level_idx: int):
        super().__init__()
        self.config = config
        self.level_idx = level_idx
        
        dims = config.dims
        wm_config = config.world_model
        
        self.z_dim = wm_config.state_dim_stochastic
        self.h_dim = wm_config.state_dim_deterministic
        self.hidden_dim = wm_config.hidden_dim
        self.obs_dim = dims.obs_embed_dim
        
        # Input: observation embedding (128) + h_state (256) + prior_mean (32) + prior_log_var (32)
        if level_idx == 0:
            # Level 1: Has access to observations
            input_dim = self.obs_dim + self.h_dim + self.z_dim * 2
        else:
            # Higher levels: No direct observations, just h and prior
            input_dim = self.h_dim + self.z_dim * 2
        
        # Network architecture: 2-layer MLP
        self.network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.z_dim * 2)  # Output: mean and log_var
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize network weights"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.5)
                nn.init.constant_(layer.bias, 0.0)
        
        # Initialize last layer with modest scale
        # Small enough to start near-prior, large enough to learn quickly
        with torch.no_grad():
            self.network[-1].weight.mul_(0.1)
            self.network[-1].bias[self.z_dim:].fill_(-2.0)  # log_var bias
    
    def forward(
        self,
        observation: Optional[Tensor],
        h_state: Tensor,
        prior: GaussianDist
    ) -> GaussianDist:
        """
        Compute posterior q(z | o, h, prior) via amortized inference.
        
        Args:
            observation: Observation embedding [batch, obs_dim] (None for higher levels)
            h_state: Deterministic hidden state [batch, h_dim]
            prior: Prior distribution p(z | h) from world model
            
        Returns:
            posterior: Inferred posterior distribution q(z | o, h)
        """
        # Concatenate inputs
        if self.level_idx == 0 and observation is not None:
            # Level 1: Use observation
            inputs = torch.cat([
                observation,
                h_state,
                prior.mean,
                prior.log_var
            ], dim=-1)
        else:
            # Higher levels: No observation
            inputs = torch.cat([
                h_state,
                prior.mean,
                prior.log_var
            ], dim=-1)
        
        # Forward pass through recognition network
        output = self.network(inputs)
        
        # Split into mean and log_var
        post_mean = output[..., :self.z_dim]
        post_log_var = output[..., self.z_dim:]
        
        # Clamp log_var for numerical stability
        post_log_var = torch.clamp(post_log_var, min=-10.0, max=2.0)
        
        # During training, allow recognition to fully express itself
        # During inference, can optionally blend with prior for stability
        # For now: NO BLENDING - let the network learn!
        # blend_factor = 0.8
        # post_mean = blend_factor * post_mean + (1 - blend_factor) * prior.mean
        # post_log_var = blend_factor * post_log_var + (1 - blend_factor) * prior.log_var
        
        return GaussianDist(mean=post_mean, log_var=post_log_var)


class HierarchicalRecognition(nn.Module):
    """
    Recognition networks for all hierarchical levels.
    
    Performs bottom-up inference across the 3-level hierarchy.
    """
    
    def __init__(self, config: ThemisConfig):
        super().__init__()
        self.config = config
        
        # Create recognition network for each level
        self.level1_recognition = RecognitionNetwork(config, level_idx=0)
        self.level2_recognition = RecognitionNetwork(config, level_idx=1)
        self.level3_recognition = RecognitionNetwork(config, level_idx=2)
    
    def forward(
        self,
        observation: Tensor,
        h_states: list,
        priors: list
    ) -> list:
        """
        Compute posteriors for all levels via recognition networks.
        
        Args:
            observation: Observation embedding [batch, obs_dim]
            h_states: List of h states for each level
            priors: List of prior distributions for each level
            
        Returns:
            posteriors: List of posterior distributions [Level 1, Level 2, Level 3]
        """
        # Level 1: Uses observation
        post1 = self.level1_recognition(observation, h_states[0], priors[0])
        
        # Level 2: No direct observation
        post2 = self.level2_recognition(None, h_states[1], priors[1])
        
        # Level 3: No direct observation
        post3 = self.level3_recognition(None, h_states[2], priors[2])
        
        return [post1, post2, post3]
