"""
Comprehensive Testing Suite for Themis AI
Tests the model on many different examples to understand its capabilities
"""

import sys
import os
import torch

sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator


def predict(model, config, device, question):
    """Get AI prediction for a question"""
    with torch.no_grad():
        obs_dist = model.markov_blanket.encode_batch([question])
        
        z_dim = config.world_model.state_dim_stochastic
        h_dim = config.world_model.state_dim_deterministic
        
        z_dummy = torch.zeros(1, z_dim, device=device)
        h_dummy = obs_dist.mean[:, :h_dim] if h_dim <= 128 else torch.zeros(1, h_dim, device=device)
        
        policy_dist = model.planning_engine.amortized_policy(z_dummy, h_dummy)
        
        # Get top prediction
        probs = policy_dist.probs[0]
        top_idx = torch.argmax(probs)
        top_prob = probs[top_idx].item()
        predicted = model.markov_blanket.tokenizer.decode([top_idx.item()])
        
        return predicted, top_prob


def run_tests():
    print("=" * 80)
    print(" " * 25 + "COMPREHENSIVE AI TEST SUITE")
    print("=" * 80)
    print()
    
    # Load model
    config = ThemisConfig()
    device = config.resolve_device()
    
    model = Orchestrator(config)
    checkpoint = torch.load("checkpoint_simple.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    print()
    
    # Test categories
    test_sets = {
        "MATH - Addition (Trained)": [
            ("If A = 1 and B = 2, then A + B =", "3"),
        ],
        "MATH - Addition (New Numbers)": [
            ("If A = 2 and B = 3, then A + B =", "5"),
            ("If A = 3 and B = 4, then A + B =", "7"),
            ("If A = 5 and B = 7, then A + B =", "12"),
            ("If A = 10 and B = 5, then A + B =", "15"),
            ("If A = 0 and B = 1, then A + B =", "1"),
        ],
        "MATH - Simple Single Digit": [
            ("What is 2 + 2 =", "4"),
            ("What is 3 + 3 =", "6"),
            ("What is 5 + 5 =", "10"),
        ],
        "NUMBER SEQUENCES (Trained)": [
            ("Write the next number in sequence: 2, 4, 6, 8,", "10"),
        ],
        "NUMBER SEQUENCES (New Patterns)": [
            ("Write the next number: 1, 2, 3, 4,", "5"),
            ("Write the next number: 5, 10, 15, 20,", "25"),
            ("Write the next number: 10, 20, 30, 40,", "50"),
            ("Write the next number: 1, 3, 5, 7,", "9"),
        ],
        "LETTER SEQUENCES (Trained)": [
            ("Write the next letters in sequence: A, C, E, G,", "I"),
        ],
        "LETTER SEQUENCES (New Patterns)": [
            ("Write the next letter: A, B, C, D,", "E"),
            ("Write the next letter: M, N, O, P,", "Q"),
        ],
        "COUNTING (Trained)": [
            ("If count of X in XXYXX is", "4"),
        ],
        "COUNTING (New Examples)": [
            ("If count of A in AAABBA is", "4"),
            ("If count of O in OOXXOO is", "4"),
            ("If count of B in ABABAB is", "3"),
        ],
        "WORD COMPLETION (Trained)": [
            ("The antonym of hot is", "cold"),
            ("Complete the logic: sky is blue, grass is", "green"),
            ("Capital of France is", "Paris"),
        ],
        "WORD COMPLETION (New)": [
            ("The opposite of up is", "down"),
            ("The color of snow is", "white"),
            ("The opposite of big is", "small"),
        ],
    }
    
    overall_correct = 0
    overall_total = 0
    
    for category, tests in test_sets.items():
        print("=" * 80)
        print(f"CATEGORY: {category}")
        print("=" * 80)
        
        correct = 0
        
        for question, expected in tests:
            predicted, confidence = predict(model, config, device, question)
            
            # Check if correct (exact match or first character match for multi-char answers)
            is_correct = (predicted.strip() == expected.strip() or 
                         predicted.strip() == expected[0] if len(expected) > 0 else False)
            
            status = "✅" if is_correct else "❌"
            if is_correct:
                correct += 1
                overall_correct += 1
            
            overall_total += 1
            
            # Truncate long questions for display
            q_display = question if len(question) <= 50 else question[:47] + "..."
            
            print(f"{status} Q: {q_display}")
            print(f"   Expected: '{expected}' | Got: '{predicted.strip()}' ({confidence*100:.1f}% confidence)")
            print()
        
        accuracy = (correct / len(tests)) * 100 if len(tests) > 0 else 0
        print(f"Category Score: {correct}/{len(tests)} ({accuracy:.1f}%)")
        print()
    
    # Final summary
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    overall_accuracy = (overall_correct / overall_total) * 100
    print(f"Total Correct: {overall_correct}/{overall_total}")
    print(f"Overall Accuracy: {overall_accuracy:.1f}%")
    print()
    
    # Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    if overall_accuracy >= 70:
        print("🌟 EXCELLENT: AI generalizes well to new examples!")
    elif overall_accuracy >= 50:
        print("✅ GOOD: AI learned patterns but struggles with variations")
    elif overall_accuracy >= 30:
        print("⚠️  FAIR: AI memorized training data but doesn't generalize")
    else:
        print("❌ POOR: AI needs more training or better architecture")
    
    print()
    print("Key Findings:")
    print("• Tests exact training examples vs. variations")
    print("• Shows which patterns the AI truly learned")
    print("• Reveals generalization capability")
    print("• Identifies strengths and weaknesses")
    print()


if __name__ == "__main__":
    run_tests()
