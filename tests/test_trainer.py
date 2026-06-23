"""
Tests for Phase 3 training components (ReplayBuffer, Trainer, and Consolidation).
"""

import pytest
import torch

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator
from themis.training.replay_buffer import ReplayBuffer
from themis.training.trainer import ActiveInferenceTrainer
from themis.training.consolidation import run_consolidation_sleep


def test_replay_buffer():
    config = ThemisConfig()
    config.training.replay_buffer_size = 50 # small buffer
    
    buffer = ReplayBuffer(config)
    
    # 1. Add trajectories
    obs = ["The brain ", "is a machine", "for active inference"]
    actions = [12, 45, 99]
    dones = [False, False, True]
    
    buffer.add_trajectory(obs, actions, dones)
    
    assert len(buffer) == 3
    
    # Add a second trajectory to exceed capacity and trigger pruning
    obs2 = [f"token_{i}" for i in range(60)]
    actions2 = list(range(60))
    dones2 = [False] * 59 + [True]
    
    buffer.add_trajectory(obs2, actions2, dones2)
    
    # Verify max size constraint (should pop trajectory 1 to stay <= 50)
    assert len(buffer) <= 50
    assert len(buffer.buffer) == 1 # oldest popped
    
    # 2. Sample batch
    device = torch.device("cpu")
    obs_batch, action_batch, done_batch = buffer.sample_batch(batch_size=2, seq_len=4, device=device)
    
    assert len(obs_batch) == 2
    assert len(obs_batch[0]) == 4
    assert action_batch.shape == (2, 4)
    assert done_batch.shape == (2, 4)


def test_trainer_and_sleep_cycle():
    config = ThemisConfig()
    device = torch.device("cpu")
    
    # Set config settings for fast test executions
    config.perception.n_iterations = 2
    config.planning.n_candidate_policies = 2
    config.planning.n_policy_samples = 2
    config.planning.planning_horizon = 2
    config.training.batch_size = 2
    
    orchestrator = Orchestrator(config)
    trainer = ActiveInferenceTrainer(config, orchestrator)
    buffer = ReplayBuffer(config)
    
    # Add some mock trajectories to buffer
    obs1 = ["worda", "wordb", "wordc", "wordd", "worde"]
    actions1 = [10, 20, 30, 40, 50]
    dones1 = [False, False, False, False, True]
    
    buffer.add_trajectory(obs1, actions1, dones1)
    buffer.add_trajectory(obs1, actions1, dones1) # duplicate to have enough for batch size 2
    
    # Sample training batch
    obs_batch, action_batch, done_batch = buffer.sample_batch(
        batch_size=config.training.batch_size,
        seq_len=4,
        device=device
    )
    
    # Run a single training step
    metrics = trainer.train_step(obs_batch, action_batch, done_batch)
    
    # Verify loss output metrics
    assert "loss" in metrics
    assert "vfe" in metrics
    assert "complexity" in metrics
    assert "accuracy" in metrics
    assert "policy_loss" in metrics
    
    # Ensure values are scalars
    assert isinstance(metrics["loss"], float)
    assert not torch.isnan(torch.tensor(metrics["loss"]))
    
    # Run consolidation sleep cycle
    sleep_metrics = run_consolidation_sleep(
        config=config,
        orchestrator=orchestrator,
        replay_buffer=buffer,
        trainer=trainer,
        sleep_steps=2
    )
    
    assert sleep_metrics["status"] == "completed"
    assert sleep_metrics["steps_replayed"] == 2
    assert "structure" in sleep_metrics
    assert "pruning" in sleep_metrics
