"""
Tests for Layer 4: Planning Engine
"""

import pytest
import torch

from themis.config import ThemisConfig
from themis.layers.world_model import WorldModel
from themis.layers.planning import PlanningEngine


def test_planning_engine_selection():
    config = ThemisConfig()
    device = torch.device("cpu")
    dtype = torch.float32
    
    # Configure planning parameters for fast tests
    config.planning.n_candidate_policies = 8
    config.planning.n_policy_samples = 2
    config.planning.planning_horizon = 3
    
    world_model = WorldModel(config).to(device=device, dtype=dtype)
    planning_engine = PlanningEngine(config, world_model).to(device=device, dtype=dtype)
    
    batch_size = 2
    states = world_model.get_initial_states(batch_size, device, dtype)
    
    # Run planning forward pass
    best_actions, metrics = planning_engine(states)
    
    # Verify outputs
    assert best_actions.shape == (batch_size,)
    assert (best_actions >= 0).all()
    assert (best_actions < config.dims.vocab_size).all()
    
    # Verify metrics
    assert "G_min" in metrics
    assert "G_mean" in metrics
    assert "selected_policy_probs" in metrics
    assert metrics["selected_policy_probs"] >= 0.0
    assert metrics["selected_policy_probs"] <= 1.0
