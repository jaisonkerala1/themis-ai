"""
Simple Policy Training - Large Dataset
Behavior cloning approach: train just the policy network to predict answers.
No Active Inference complexity - just supervised learning.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import random

sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator


def load_dataset(filename):
    """Load the training dataset"""
    with open(filename, 'r') as f:
        return json.load(f)


def train_simple_policy():
    print("=" * 80)
    print("SIMPLE POLICY TRAINING - LARGE DATASET")
    print("=" * 80)
    print()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()
    
    # Load large dataset
    print("[1] Loading dataset...")
    dataset = load_dataset("training_dataset_large.json")
    print(f"✓ Loaded {len(dataset)} tasks")
    print()
    
    # Count tasks by category
    categories = {}
    for task in dataset:
        cat = task['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print("Dataset breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"  • {cat}: {count} tasks")
    print()
    
    # Initialize model
    print("[2] Initializing policy network...")
    config = ThemisConfig()
    config.device = str(device)
    orchestrator = Orchestrator(config)
    orchestrator = orchestrator.to(device)
    
    # Extract just the policy network
    policy = orchestrator.planning_engine.amortized_policy
    tokenizer = orchestrator.markov_blanket.tokenizer
    
    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"✓ Policy network: {total_params:,} parameters")
    print()
    
    # Training setup
    optimizer = optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training parameters
    epochs = 3000
    batch_size = 32  # Increased for better learning
    
    print("[3] Collecting training data...")
    training_data = []
    
    for task in dataset:
        question = task['question']
        answer = task['answer']
        
        # Encode the question
        obs_dist = orchestrator.markov_blanket.encode(question)
        obs_embed = obs_dist.mean.detach()  # [1, 128] - detach to avoid graph issues
        
        # Get answer token IDs
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        
        # Store (observation, action) pairs
        for token_id in answer_tokens:
            training_data.append((obs_embed.clone(), token_id))
    
    print(f"✓ Collected {len(training_data)} training examples")
    print()
    
    print("[4] Training policy network...")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Training examples: {len(training_data)}")
    print()
    
    z_dim = config.world_model.state_dim_stochastic
    h_dim = config.world_model.state_dim_deterministic
    
    best_loss = float('inf')
    best_accuracy = 0.0
    
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
        
        # Create dummy h and z states
        z_dummy = torch.zeros(batch_size, z_dim, device=device)
        h_dummy = obs_batch[:, :h_dim] if h_dim <= 128 else torch.zeros(batch_size, h_dim, device=device)
        
        # Forward pass
        policy_dist = policy(z_dummy, h_dummy)
        logits = policy_dist.logits  # [batch_size, vocab_size]
        
        # Compute loss
        loss = loss_fn(logits, action_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        
        # Track metrics
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = (preds == action_batch).sum().item()
            accuracy = 100.0 * correct / batch_size
            
            if loss.item() < best_loss:
                best_loss = loss.item()
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
        
        # Print progress
        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss.item():.4f} | Acc: {accuracy:5.1f}% | Best Acc: {best_accuracy:5.1f}%")
        
        # Save checkpoint every 500 epochs
        if epoch % 500 == 0:
            checkpoint_path = f"checkpoint_simple_large_epoch{epoch}.pt"
            torch.save({
                'model_state': orchestrator.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'best_accuracy': best_accuracy,
                'config': config
            }, checkpoint_path)
            if os.path.exists(checkpoint_path):
                print(f"  💾 Saved: {checkpoint_path}")
    
    print()
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print()
    
    # Save final model
    final_path = "checkpoint_simple_large.pt"
    torch.save({
        'model_state': orchestrator.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epochs,
        'best_accuracy': best_accuracy,
        'config': config
    }, final_path)
    
    if os.path.exists(final_path):
        file_size = os.path.getsize(final_path)
        print(f"✓ Model saved: {final_path} ({file_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"❌ ERROR: Failed to save {final_path}")
    
    print(f"✓ Final accuracy: {best_accuracy:.1f}%")
    print(f"✓ Best loss: {best_loss:.4f}")
    print()
    print("Next step: Test with:")
    print("  .venv\\Scripts\\python.exe scripts\\test_simple_large.py")
    print("=" * 80)


if __name__ == "__main__":
    train_simple_policy()
