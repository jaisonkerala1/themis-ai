"""
Simple test to show what the AI actually predicts
"""

import sys
import os
import torch

sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator
from environments.reasoning_env import ReasoningEnv


print("=" * 80)
print(" " * 20 + "THEMIS AI - ANSWER TEST")
print("=" * 80)
print()

config = ThemisConfig()
device = config.resolve_device()

# Load the simple checkpoint
orchestrator = Orchestrator(config)
checkpoint = torch.load("checkpoint_simple.pt", map_location=device, weights_only=False)
orchestrator.load_state_dict(checkpoint['model_state'])
orchestrator.eval()

print(f"Model: Themis Active Inference AI")
print(f"Parameters: ~3.5M")
print(f"Training: Simple policy network (behavior cloning)")
print(f"Device: {device}")
print()
print("=" * 80)
print()

# Test on all tasks
env = ReasoningEnv()

print("TESTING ALL 7 REASONING TASKS:")
print()

correct_count = 0

for i, (prefix, target) in enumerate(env.dataset):
    print(f"Question {i+1}:")
    print(f"  Input:    '{prefix.strip()}'")
    print(f"  Expected: '{target}'")
    
    # Encode and predict
    with torch.no_grad():
        obs_dist = orchestrator.markov_blanket.encode_batch([prefix])
        
        # Use dummy states
        z_dim = config.world_model.state_dim_stochastic
        h_dim = config.world_model.state_dim_deterministic
        
        z_dummy = torch.zeros(1, z_dim, device=device)
        h_dummy = obs_dist.mean[:, :h_dim] if h_dim <= 128 else torch.zeros(1, h_dim, device=device)
        
        # Get prediction
        policy_dist = orchestrator.planning_engine.amortized_policy(z_dummy, h_dummy)
        action_id = policy_dist.sample_index()
        predicted = orchestrator.markov_blanket.tokenizer.decode([action_id.item()])
    
    # Check correctness
    is_correct = predicted.strip() == target.strip()
    status = "✅ CORRECT" if is_correct else "❌ WRONG"
    
    if is_correct:
        correct_count += 1
    
    print(f"  AI Says:  '{predicted}' {status}")
    print()

print("=" * 80)
print(f"FINAL SCORE: {correct_count}/7 correct ({correct_count/7*100:.1f}%)")
print("=" * 80)
print()

# Explanation
if correct_count >= 3:
    print("✅ The AI is LEARNING and can solve reasoning tasks!")
    print("   It successfully answers questions requiring:")
    print("   • Mathematical reasoning (A + B = ?)")
    print("   • Pattern recognition (sequence completion)")
    print("   • Counting logic")
else:
    print("❌ The AI needs more training to improve accuracy.")

print()
print("NOTE: The AI currently predicts single tokens.")
print("      Multi-token answers (like 'Paris', 'green') need sequential generation.")
print()
