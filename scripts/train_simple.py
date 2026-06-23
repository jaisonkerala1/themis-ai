"""
Simple Baseline Training - Train ONLY the policy network

Instead of training the full active inference stack (which has numerical issues),
we train just the amortized policy network with supervised learning.

This is essentially behavior cloning - the policy learns to imitate expert actions.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator
from environments.reasoning_env import ReasoningEnv


def train_policy_only():
    print("=" * 60)
    print("  SIMPLE POLICY TRAINING - BEHAVIOR CLONING")
    print("=" * 60)
    
    config = ThemisConfig()
    device = config.resolve_device()
    print(f"Device: {device}")
    
    # Initialize orchestrator
    orchestrator = Orchestrator(config)
    orchestrator.to(device)
    
    # We'll train ONLY the amortized policy network
    policy_net = orchestrator.planning_engine.amortized_policy
    
    # Optimizer for policy only
    optimizer = optim.AdamW(policy_net.parameters(), lr=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    env = ReasoningEnv()
    tokenizer = orchestrator.markov_blanket.tokenizer
    
    print(f"\n[1] Dataset: {len(env.dataset)} tasks")
    
    # Collect training data
    print("\n[2] Collecting expert demonstrations...")
    training_data = []
    
    for _ in range(100):  # 100 episodes (14 per task)
        env.reset()
        target = env.current_target
        prefix = env.current_prefix
        
        # Encode observation
        with torch.no_grad():
            obs_dist = orchestrator.markov_blanket.encode_batch([prefix])
            obs_embed = obs_dist.mean  # [1, 128]
        
        # Get expert action tokens
        expert_tokens = tokenizer.encode(target, add_special_tokens=False)
        
        # Store (observation, action) pairs
        for token_id in expert_tokens:
            training_data.append((obs_embed.clone(), token_id))
    
    print(f"✓ Collected {len(training_data)} training examples")
    
    # Train the policy network
    print("\n[3] Training policy network...")
    epochs = 2000
    batch_size = 32
    
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # Sample batch
        indices = torch.randint(0, len(training_data), (batch_size,))
        
        obs_batch = []
        action_batch = []
        for idx in indices:
            obs, action = training_data[idx]
            obs_batch.append(obs)
            action_batch.append(action)
        
        obs_batch = torch.cat(obs_batch, dim=0).to(device)  # [batch_size, 128]
        action_batch = torch.tensor(action_batch, dtype=torch.long, device=device)  # [batch_size]
        
        # Create dummy h and z states (policy network needs them as input)
        z_dim = config.world_model.state_dim_stochastic
        h_dim = config.world_model.state_dim_deterministic
        
        z_dummy = torch.zeros(batch_size, z_dim, device=device)
        h_dummy = obs_batch[:, :h_dim] if h_dim <= 128 else torch.zeros(batch_size, h_dim, device=device)
        
        # Forward pass
        policy_dist = policy_net(z_dummy, h_dummy)
        logits = policy_dist.logits
        
        # Compute loss
        loss = loss_fn(logits, action_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d}/{epochs} | Loss: {loss.item():.4f} | Best: {best_loss:.4f}")
    
    # Save checkpoint
    checkpoint_path = "checkpoint_simple.pt"
    torch.save({
        'model_state': orchestrator.state_dict(),
        'config': config,
        'step': epochs
    }, checkpoint_path)
    
    print(f"\n✓ Training completed!")
    print(f"✓ Best Loss: {best_loss:.4f}")
    print(f"✓ Checkpoint saved: {checkpoint_path}")
    print("=" * 60)


if __name__ == "__main__":
    train_policy_only()
