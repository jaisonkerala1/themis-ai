"""
Quick check: Are posteriors now different after removing blending?
"""

import sys
import os
import torch

sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator


def check_posteriors():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load("checkpoint_real_ai.pt", map_location=device, weights_only=False)
    config = checkpoint['config']
    config.device = str(device)
    
    orchestrator = Orchestrator(config)
    orchestrator.load_state_dict(checkpoint['model_state'])
    orchestrator = orchestrator.to(device)
    orchestrator.eval()
    
    test_inputs = [
        "If A = 1 and B = 2, then A + B =",
        "If A = 5 and B = 5, then A + B =",
        "What is 2 + 2 =",
    ]
    
    print("=" * 80)
    print("CHECKING POSTERIOR DIVERSITY")
    print("=" * 80)
    print()
    
    all_posteriors = []
    
    for text in test_inputs:
        obs_dist = orchestrator.markov_blanket.encode(text)
        states = orchestrator.world_model.get_initial_states(1, device, torch.float32)
        h_states, priors = orchestrator.world_model.compute_priors(states, None)
        posteriors = orchestrator.world_model.recognition(obs_dist.mean, h_states, priors)
        
        all_posteriors.append(posteriors[0].mean)
        
        print(f"Text: '{text}'")
        print(f"  Posterior L1 mean: {posteriors[0].mean.mean().item():.6f}")
        print(f"  Posterior L1 std: {posteriors[0].mean.std().item():.6f}")
        print(f"  First 5 values: {posteriors[0].mean[0, :5].tolist()}")
        print()
    
    print("Differences between posteriors:")
    for i in range(len(all_posteriors)):
        for j in range(i+1, len(all_posteriors)):
            diff = (all_posteriors[i] - all_posteriors[j]).abs().mean().item()
            print(f"  Input {i} vs {j}: {diff:.6f}")
    
    print()
    
    avg_diff = sum([(all_posteriors[i] - all_posteriors[j]).abs().mean().item() 
                    for i in range(len(all_posteriors)) 
                    for j in range(i+1, len(all_posteriors))]) / 3
    
    if avg_diff < 0.001:
        print("❌ STILL BROKEN: Posteriors are nearly identical!")
    elif avg_diff < 0.01:
        print("⚠️  BETTER but still too similar")
    elif avg_diff < 0.1:
        print("✅ GOOD: Posteriors show variation")
    else:
        print("✅ EXCELLENT: Strong variation in posteriors")
    
    print(f"Average difference: {avg_diff:.6f}")
    print()
    
    # Also check policy outputs
    print("=" * 80)
    print("POLICY OUTPUTS")
    print("=" * 80)
    print()
    
    tokenizer = orchestrator.markov_blanket.tokenizer
    
    for text in test_inputs:
        obs_dist = orchestrator.markov_blanket.encode(text)
        states = orchestrator.world_model.get_initial_states(1, device, torch.float32)
        h_states, priors = orchestrator.world_model.compute_priors(states, None)
        posteriors = orchestrator.world_model.recognition(obs_dist.mean, h_states, priors)
        
        z1 = posteriors[0].sample()
        policy_dist = orchestrator.planning_engine.amortized_policy(z1, h_states[0])
        probs = torch.softmax(policy_dist.logits, dim=-1)
        
        top_k = 5
        top_probs, top_ids = torch.topk(probs[0], k=top_k)
        
        print(f"Text: '{text}'")
        for i in range(top_k):
            token = tokenizer.decode([top_ids[i].item()])
            print(f"  {i+1}. '{token}' ({top_probs[i].item()*100:.1f}%)")
        print()


if __name__ == "__main__":
    check_posteriors()
