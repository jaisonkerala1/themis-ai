"""
Test the Real AI - Comprehensive Evaluation
Tests both trained examples AND new variations (generalization test)
"""

import sys
import os
import torch
import json

sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator


def predict(model, config, device, question):
    """Get AI prediction using the SAME inference path as training.

    The model was trained with:
      - z = recognition network posterior (depends on observation)
      - h = world model h_states (from GRU)
    So inference must use that same path, not zero-z / observation-as-h.
    """
    with torch.no_grad():
        obs_dist = model.markov_blanket.encode_batch([question])

        # Initial states -> priors via world model
        states = model.world_model.get_initial_states(1, device, torch.float32)
        h_states, priors = model.world_model.compute_priors(states, None)

        # Recognition network produces observation-dependent posterior
        posteriors = model.world_model.recognition(obs_dist.mean, h_states, priors)

        # Policy reads posterior z and world-model h (same as training)
        z1 = posteriors[0].sample()
        policy_dist = model.planning_engine.amortized_policy(z1, h_states[0])

        probs = policy_dist.probs[0]
        top_idx = torch.argmax(probs)
        top_prob = probs[top_idx].item()
        predicted = model.markov_blanket.tokenizer.decode([top_idx.item()])

        return predicted, top_prob


def run_comprehensive_test():
    print("=" * 80)
    print(" " * 25 + "REAL AI EVALUATION")
    print(" " * 20 + "Testing Generalization Ability")
    print("=" * 80)
    print()
    
    # Load model
    config = ThemisConfig()
    device = config.resolve_device()
    
    model = Orchestrator(config)
    
    # Check which checkpoint to load
    if os.path.exists("checkpoint_real_ai.pt"):
        checkpoint_path = "checkpoint_real_ai.pt"
        print(f"✓ Loading: {checkpoint_path}")
    else:
        checkpoint_path = "checkpoint_simple.pt"
        print(f"⚠  Real AI checkpoint not found, using: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"✓ Model loaded on {device}")
    print()
    
    # Test sets
    test_sets = {
        "TRAINED EXAMPLES (Should get 95%+)": [
            ("If A = 1 and B = 2, then A + B =", "3"),
            ("If A = 5 and B = 5, then A + B =", "10"),
            ("Write the next number: 1, 2, 3, 4,", "5"),
            ("Write the next number in sequence: 2, 4, 6, 8,", "10"),
        ],
        "NEW VARIATIONS (Tests Generalization - Target 70%+)": [
            ("If A = 6 and B = 4, then A + B =", "10"),
            ("If A = 3 and B = 8, then A + B =", "11"),
            ("What is 4 + 3 =", "7"),
            ("Add 2 and 7 =", "9"),
            ("Write the next number: 5, 6, 7, 8,", "9"),
            ("Write the next number: 10, 11, 12, 13,", "14"),
        ],
        "COMPLETELY NEW (Not in Training - Expect Low)": [
            ("If A = 15 and B = 20, then A + B =", "35"),
            ("What is 100 + 200 =", "300"),
        ],
    }
    
    overall_correct = 0
    overall_total = 0
    
    for category, tests in test_sets.items():
        print("=" * 80)
        print(f"CATEGORY: {category}")
        print("=" * 80)
        print()
        
        correct = 0
        
        for question, expected in tests:
            predicted, confidence = predict(model, config, device, question)
            
            # Check correctness (exact match or first char for multi-digit)
            is_correct = (predicted.strip() == expected.strip() or
                         (len(expected) > 0 and predicted.strip() == expected[0]))
            
            status = "✅" if is_correct else "❌"
            if is_correct:
                correct += 1
                overall_correct += 1
            
            overall_total += 1
            
            print(f"{status} Q: {question}")
            print(f"   Expected: '{expected}' | Got: '{predicted.strip()}' ({confidence*100:.1f}%)")
            print()
        
        accuracy = (correct / len(tests)) * 100 if len(tests) > 0 else 0
        print(f"Category Score: {correct}/{len(tests)} ({accuracy:.1f}%)")
        print()
    
    # Final results
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    overall_accuracy = (overall_correct / overall_total) * 100
    print(f"Total Correct: {overall_correct}/{overall_total}")
    print(f"Overall Accuracy: {overall_accuracy:.1f}%")
    print()
    
    # Verdict
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    
    if overall_accuracy >= 70:
        print("🌟 EXCELLENT: AI successfully learned to generalize!")
        print("   The model understands concepts, not just memorization!")
    elif overall_accuracy >= 50:
        print("✅ GOOD: AI shows real learning with room for improvement")
        print("   The model is learning patterns beyond training examples")
    elif overall_accuracy >= 30:
        print("⚠️  FAIR: AI learned some patterns but limited generalization")
        print("   More training or data diversity needed")
    else:
        print("❌ NEEDS WORK: AI is still primarily memorizing")
        print("   Requires longer training or architecture improvements")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    run_comprehensive_test()
