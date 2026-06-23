"""
Themis Training — Consolidation ("Sleep" Phase)

Implements the offline consolidation phase ("sleep cycles").
Replays trajectories from the experience replay buffer to consolidate world dynamics,
and performs structure learning checks to reduce model redundancy.
"""

from typing import Dict, Any, List
import torch
import torch.nn as nn

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator
from themis.training.replay_buffer import ReplayBuffer
from themis.training.trainer import ActiveInferenceTrainer


def run_consolidation_sleep(
    config: ThemisConfig,
    orchestrator: Orchestrator,
    replay_buffer: ReplayBuffer,
    trainer: ActiveInferenceTrainer,
    sleep_steps: int = 5
) -> Dict[str, Any]:
    """
    Executes an offline consolidation "sleep cycle" for the agent.
    
    1. Replays stored experiences through the trainer to consolidate world model dynamics.
    2. Runs Bayesian model reduction to identify redundant dimensions.
    
    Args:
        config: ThemisConfig
        orchestrator: Orchestrator
        replay_buffer: ReplayBuffer
        trainer: ActiveInferenceTrainer
        sleep_steps: Number of replay training iterations.
        
    Returns:
        sleep_metrics: Dictionary of consolidation results.
    """
    device = config.resolve_device()
    
    if len(replay_buffer) == 0:
        return {"status": "skipped", "reason": "empty_buffer"}
        
    vfe_reductions = []
    policy_losses = []
    
    # 1. Offline Replay Loop
    orchestrator.train() # Set to train mode
    
    for step in range(sleep_steps):
        # Sample sequence slice from replay buffer
        obs, actions, dones = replay_buffer.sample_batch(
            batch_size=config.training.batch_size,
            seq_len=32, # standard training slice length
            device=device
        )
        
        # Optimize parameters on replay slice
        metrics = trainer.train_step(obs, actions, dones)
        
        vfe_reductions.append(metrics["vfe"])
        policy_losses.append(metrics["policy_loss"])
        
    orchestrator.eval() # Set back to eval mode
    
    # 2. Structure Learning Consolidation (Pruning checks)
    # Checks latent standard deviation and VFE average
    structure_logs = orchestrator.meta_learning.apply_structure_change(orchestrator.world_model)
    
    # Optional: Apply pruning mask to model states
    # We retrieve the pruning mask and log it
    keep_mask, pruning_stats = orchestrator.meta_learning.get_pruning_mask()
    
    sleep_metrics = {
        "status": "completed",
        "steps_replayed": sleep_steps,
        "vfe_start": vfe_reductions[0] if vfe_reductions else 0.0,
        "vfe_end": vfe_reductions[-1] if vfe_reductions else 0.0,
        "policy_loss_avg": sum(policy_losses) / len(policy_losses) if policy_losses else 0.0,
        "structure": structure_logs,
        "pruning": pruning_stats
    }
    
    return sleep_metrics
