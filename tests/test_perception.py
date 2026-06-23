"""
Tests for Layer 2: Perception Engine
"""

import pytest
import torch

from themis.config import ThemisConfig
from themis.layers.world_model import WorldModel
from themis.layers.perception import PerceptionEngine
from themis.core.distributions import GaussianDist


def test_perception_engine_vfe_reduction():
    config = ThemisConfig()
    device = torch.device("cpu")
    dtype = torch.float32
    
    # Settle configs for clean testing
    config.perception.n_iterations = 12
    config.perception.learning_rate_mu = 0.2
    
    world_model = WorldModel(config).to(device=device, dtype=dtype)
    perception_engine = PerceptionEngine(config).to(device=device, dtype=dtype)
    
    batch_size = 2
    
    # Initialize states and action
    prev_states = world_model.get_initial_states(batch_size, device, dtype)
    prev_action = torch.randint(0, config.dims.vocab_size, (batch_size,), device=device)
    
    # Sensory input (GaussianDist) representing observation
    obs_mean = torch.randn(batch_size, config.dims.obs_embed_dim, device=device, dtype=dtype)
    obs_log_var = torch.zeros(batch_size, config.dims.obs_embed_dim, device=device, dtype=dtype)
    observation = GaussianDist(mean=obs_mean, log_var=obs_log_var)
    
    # Run the belief update loop
    posteriors, metrics = perception_engine.update_beliefs(
        world_model=world_model,
        observation=observation,
        prev_states=prev_states,
        action=prev_action
    )
    
    # Verify posteriors are correct shape
    assert len(posteriors) == 3
    for post in posteriors:
        assert post.mean.shape == (batch_size, config.world_model.state_dim_stochastic)
        assert post.log_var.shape == (batch_size, config.world_model.state_dim_stochastic)
        
    # Verify that VFE has decreased
    vfe_start = metrics["vfe_start"]
    vfe_end = metrics["vfe_end"]
    
    print(f"VFE Start: {vfe_start:.4f} -> VFE End: {vfe_end:.4f}")
    assert vfe_end < vfe_start, "Perception did not minimize Variational Free Energy!"
    
    # Verify history tracking
    assert len(metrics["vfe_history"]) == 12
    assert metrics["vfe_history"][-1] == vfe_end
