"""
Themis Training Script — Model Fitting

Exposes a simple training pipeline.
1. Collects trajectories in the reasoning environment to populate the ReplayBuffer.
2. Trains the H-RSSM and policy networks over multiple epochs, showing VFE decrease.
3. Saves a model checkpoint to disk.
"""

import sys
import os
import torch

# Add CWD to pythonpath
sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator
from themis.training.trainer import ActiveInferenceTrainer
from themis.training.replay_buffer import ReplayBuffer
from environments.reasoning_env import ReasoningEnv


def run_training_loop():
    print("====================================================")
    print("           THEMIS ONLINE MODEL TRAINING             ")
    print("====================================================")
    
    # Initialize Config & Hardware
    config = ThemisConfig()
    # Configure hyper-params for fast training convergence
    config.perception.n_iterations = 4
    config.planning.n_candidate_policies = 4
    config.planning.n_policy_samples = 2
    config.planning.planning_horizon = 2
    config.training.learning_rate = 1e-3
    config.training.batch_size = 8
    
    device = config.resolve_device()
    print(f"Device: {device}")
    
    orchestrator = Orchestrator(config)
    trainer = ActiveInferenceTrainer(config, orchestrator)
    replay_buffer = ReplayBuffer(config)
    env = ReasoningEnv()
    
    print("\n--- 1. Collecting Replay Buffer Experiences ---")
    # Collect 30 trajectories by letting the agent step in the environments using expert actions
    for episode in range(30):
        obs = env.reset()
        obs_list = [obs]
        actions_list = []
        dones_list = []
        
        # Let orchestrator reset
        orchestrator.reset(batch_size=1)
        
        # Get target preference and target completion
        target_text = env.get_preference()
        target_completion = env.current_target
        
        # Tokenize target completion
        tokenizer = orchestrator.markov_blanket.tokenizer
        expert_action_ids = tokenizer.encode(target_completion, add_special_tokens=False)
        
        done = False
        step = 0
        
        for expert_id in expert_action_ids:
            if done:
                break
                
            # Get expert action string
            action_str = tokenizer.decode([expert_id])
            
            # Step orchestrator (we pass target_text so EFE planning can run perception)
            orchestrator.step(obs, target_text=target_text)
            
            # Override orchestrator's prev_action_ids to the expert action to keep states aligned
            orchestrator.prev_action_ids = torch.tensor([expert_id], dtype=torch.long, device=device)
            
            # Step the environment with expert action
            next_obs, _, done, info = env.step(action_str)
            
            # Track
            obs_list.append(next_obs)
            actions_list.append(expert_id)
            dones_list.append(done)
            
            obs = next_obs
            step += 1
            
        replay_buffer.add_trajectory(obs_list, actions_list, dones_list)
        
    print(f"Buffer populated with {len(replay_buffer)} transitions.")
    
    print("\n--- 2. Optimization Epochs (BPTT) ---")
    epochs = 1200
    
    for epoch in range(1, epochs + 1):
        # Sample training batch
        obs_b, action_b, done_b = replay_buffer.sample_batch(
            batch_size=config.training.batch_size,
            seq_len=4,
            device=device
        )
        
        # Run BPTT optimization step
        metrics = trainer.train_step(obs_b, action_b, done_b)
        
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs:03d} | VFE: {metrics['vfe']:.4f} | Policy Loss: {metrics['policy_loss']:.4f} | Total Loss: {metrics['loss']:.4f}", flush=True)
            
    # Save checkpoint
    checkpoint_path = "checkpoint.pt"
    orchestrator.save_checkpoint(checkpoint_path)
    print(f"\nTraining completed! Checkpoint saved to: '{checkpoint_path}' [PASS]")
    print("====================================================")


if __name__ == "__main__":
    run_training_loop()
