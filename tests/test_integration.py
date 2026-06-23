"""
Integration tests for the full Themis coordinates loop (Layers 1-7).
"""

import pytest
import torch

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator


def test_full_agent_step_and_reset():
    config = ThemisConfig()
    device = torch.device("cpu")
    
    # Minimize config sizes for testing speed
    config.perception.n_iterations = 4
    config.planning.n_candidate_policies = 4
    config.planning.n_policy_samples = 2
    config.planning.planning_horizon = 2
    config.meta_learning.consolidation_interval = 2 # quick prune
    
    orchestrator = Orchestrator(config)
    orchestrator.reset(batch_size=1)
    
    # 1. Run step with single observation
    action_ids, action_tokens, metrics = orchestrator.step("hello world")
    
    # Verify outputs
    assert action_ids.shape == (1,)
    assert len(action_tokens) == 1
    assert isinstance(action_tokens[0], str)
    
    # Verify metrics structure
    assert "vfe" in metrics
    assert "surprise" in metrics
    assert "n_iterations" in metrics
    assert metrics["step"] == 1
    
    # 2. Run second step
    action_ids2, action_tokens2, metrics2 = orchestrator.step("how are you")
    assert metrics2["step"] == 2
    
    # Verify adaptive compute check (was boost triggered?)
    assert "compute_boost" in metrics2
    
    # Verify meta metrics logged on the consolidation interval step
    assert "pruning_stats" in metrics2["meta"]
    assert metrics2["meta"]["pruning_stats"]["total_dim"] == config.world_model.state_dim_stochastic


def test_batch_agent_step():
    config = ThemisConfig()
    
    config.perception.n_iterations = 3
    config.planning.n_candidate_policies = 3
    config.planning.n_policy_samples = 2
    config.planning.planning_horizon = 2
    
    orchestrator = Orchestrator(config)
    orchestrator.reset(batch_size=2)
    
    # Batch step
    observations = ["hello world", "artificial intelligence"]
    action_ids, action_tokens, metrics = orchestrator.step(observations)
    
    assert action_ids.shape == (2,)
    assert len(action_tokens) == 2
    assert metrics["step"] == 1
