"""
Themis Improved Training Script — Model Fitting with Sufficient Data

Improvements:
1. Increased episodes from 30 to 500 (70+ examples per task)
2. Increased epochs from 1200 to 3000 for better convergence
3. Added progress tracking with loss curves
4. Better batch size for GPU training
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
    print("    THEMIS IMPROVED TRAINING - SUFFICIENT DATA      ")
    print("====================================================")
    
    # Initialize Config & Hardware
    config = ThemisConfig()
    # Configure hyper-params for GPU training
    config.perception.n_iterations = 4
    config.planning.n_candidate_policies = 4
    config.planning.n_policy_samples = 2
    config.planning.planning_horizon = 2
    config.training.learning_rate = 1e-3
    config.training.batch_size = 16  # Increased for GPU
    
    device = config.resolve_device()
    print(f"Device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    orchestrator = Orchestrator(config)
    trainer = ActiveInferenceTrainer(config, orchestrator)
    replay_buffer = ReplayBuffer(config)
    env = ReasoningEnv()
    
    print(f"\n--- 1. Collecting Replay Buffer Experiences ---")
    print(f"Dataset size: {len(env.dataset)} unique tasks")
    
    # Collect 500 trajectories (70+ examples per task)
    num_episodes = 500
    print(f"Collecting {num_episodes} episodes ({num_episodes / len(env.dataset):.1f} per task)...")
    
    for episode in range(num_episodes):
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
        
        # Progress indicator
        if (episode + 1) % 50 == 0:
            print(f"  Collected {episode + 1}/{num_episodes} episodes...")
        
    print(f"✓ Buffer populated with {len(replay_buffer)} transitions.")
    
    print("\n--- 2. Optimization Epochs (BPTT) ---")
    epochs = 3000
    print(f"Training for {epochs} epochs...")
    
    # Track metrics
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # Sample training batch
        obs_b, action_b, done_b = replay_buffer.sample_batch(
            batch_size=config.training.batch_size,
            seq_len=4,
            device=device
        )
        
        # Run BPTT optimization step
        metrics = trainer.train_step(obs_b, action_b, done_b)
        
        # Track best
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
        
        # Print progress
        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d}/{epochs:04d} | VFE: {metrics['vfe']:7.4f} | Policy Loss: {metrics['policy_loss']:7.4f} | Total Loss: {metrics['loss']:7.4f} | Best: {best_loss:7.4f}", flush=True)
            
    # Save checkpoint
    checkpoint_path = "checkpoint.pt"
    orchestrator.save_checkpoint(checkpoint_path)
    print(f"\n✓ Training completed!")
    print(f"✓ Final Loss: {metrics['loss']:.4f}")
    print(f"✓ Best Loss: {best_loss:.4f}")
    print(f"✓ Checkpoint saved to: '{checkpoint_path}'")
    print("====================================================")


if __name__ == "__main__":
    run_training_loop()
