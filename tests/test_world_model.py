"""
Tests for Layer 3: Hierarchical World Model (Generative Model)
"""

import pytest
import torch

from themis.config import ThemisConfig
from themis.layers.world_model import WorldModel
from themis.core.distributions import GaussianDist


def test_world_model_initialization():
    config = ThemisConfig()
    device = torch.device("cpu")
    dtype = torch.float32
    
    world_model = WorldModel(config).to(device=device, dtype=dtype)
    
    # Verify submodules are present
    assert hasattr(world_model, "action_embedding")
    assert hasattr(world_model, "level1")
    assert hasattr(world_model, "level2")
    assert hasattr(world_model, "level3")
    assert hasattr(world_model, "likelihood_decoder")
    
    # Verify shapes of initial state
    batch_size = 4
    states = world_model.get_initial_states(batch_size, device, dtype)
    
    assert len(states) == 3
    for i, state in enumerate(states):
        assert "h" in state
        assert "z" in state
        assert state["h"].shape == (batch_size, config.world_model.state_dim_deterministic)
        assert state["z"].shape == (batch_size, config.world_model.state_dim_stochastic)


def test_world_model_priors_and_transitions():
    config = ThemisConfig()
    device = torch.device("cpu")
    dtype = torch.float32
    
    world_model = WorldModel(config).to(device=device, dtype=dtype)
    batch_size = 2
    
    states = world_model.get_initial_states(batch_size, device, dtype)
    action_ids = torch.randint(0, config.dims.vocab_size, (batch_size,), device=device)
    
    h_states, priors = world_model.compute_priors(states, action_ids)
    
    assert len(h_states) == 3
    assert len(priors) == 3
    for i, (h, prior) in enumerate(zip(h_states, priors)):
        assert h.shape == (batch_size, config.world_model.state_dim_deterministic)
        assert prior.mean.shape == (batch_size, config.world_model.state_dim_stochastic)
        assert prior.log_var.shape == (batch_size, config.world_model.state_dim_stochastic)


def test_world_model_imagination_rollout():
    config = ThemisConfig()
    device = torch.device("cpu")
    dtype = torch.float32
    
    world_model = WorldModel(config).to(device=device, dtype=dtype)
    batch_size = 3
    
    states = world_model.get_initial_states(batch_size, device, dtype)
    
    # Rollout over 8 steps
    horizon = 8
    curr_states = states
    
    for t in range(horizon):
        action_ids = torch.randint(0, config.dims.vocab_size, (batch_size,), device=device)
        next_states, priors = world_model.imagine_step(curr_states, action_ids)
        
        assert len(next_states) == 3
        assert len(priors) == 3
        
        # Verify transition continuity
        for i in range(3):
            assert next_states[i]["h"].shape == (batch_size, config.world_model.state_dim_deterministic)
            assert next_states[i]["z"].shape == (batch_size, config.world_model.state_dim_stochastic)
            # Ensure no NaNs or Infs
            assert not torch.isnan(next_states[i]["h"]).any()
            assert not torch.isnan(next_states[i]["z"]).any()
            
        curr_states = next_states
