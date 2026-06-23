"""
Demo for the simple trained model
"""

import sys
import os
import torch

sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator
from environments.reasoning_env import ReasoningEnv


def test_simple_model():
    print("=" * 60)
    print("  TESTING SIMPLE TRAINED MODEL")
    print("=" * 60)
    
    config = ThemisConfig()
    device = config.resolve_device()
    
    # Load the simple checkpoint
    orchestrator = Orchestrator(config)
    checkpoint = torch.load("checkpoint_simple.pt", map_location=device, weights_only=False)
    orchestrator.load_state_dict(checkpoint['model_state'])
    orchestrator.eval()
    
    print(f"✓ Loaded checkpoint: checkpoint_simple.pt")
    print(f"Device: {device}\n")
    
    # Test on all tasks
    env = ReasoningEnv()
    
    print("Testing on all tasks:\n")
    print("-" * 60)
    
    correct = 0
    total = len(env.dataset)
    
    for i, (prefix, target) in enumerate(env.dataset):
        # Encode the prefix observation
        with torch.no_grad():
            obs_dist = orchestrator.markov_blanket.encode_batch([prefix])
            
            # Get policy prediction (using dummy states)
            z_dim = config.world_model.state_dim_stochastic
            h_dim = config.world_model.state_dim_deterministic
            
            z_dummy = torch.zeros(1, z_dim, device=device)
            h_dummy = obs_dist.mean[:, :h_dim] if h_dim <= 128 else torch.zeros(1, h_dim, device=device)
            
            # Get policy action
            policy_dist = orchestrator.planning_engine.amortized_policy(z_dummy, h_dummy)
            action_id = policy_dist.sample_index()
            
            # Decode action
            predicted = orchestrator.markov_blanket.tokenizer.decode([action_id.item()])
        
        # Check if correct (first character match)
        is_correct = predicted.strip() == target[0] if len(target) > 0 else False
        status = "✓" if is_correct else "✗"
        
        if is_correct:
            correct += 1
        
        print(f"[{i}] {status} '{prefix.strip()}'")
        print(f"    Target: '{target}' | Predicted: '{predicted.strip()}'")
    
    print("-" * 60)
    accuracy = (correct / total) * 100
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    test_simple_model()
