"""
Comprehensive testing for the simple policy model
Tests a wide variety of questions to understand capabilities and limitations
"""

import sys
import os
import torch

sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator


def test_question(orchestrator, question, expected_answer):
    """Test a single question and return result"""
    device = orchestrator.device
    
    # Encode the question
    obs_dist = orchestrator.markov_blanket.encode(question)
    obs_embed = obs_dist.mean  # [1, 128]
    
    # Get config for dummy states
    z_dim = orchestrator.config.world_model.state_dim_stochastic
    h_dim = orchestrator.config.world_model.state_dim_deterministic
    
    # Create dummy states
    z_dummy = torch.zeros(1, z_dim, device=device)
    h_dummy = obs_embed[:, :h_dim] if h_dim <= 128 else torch.zeros(1, h_dim, device=device)
    
    # Predict via policy
    with torch.no_grad():
        policy_dist = orchestrator.planning_engine.amortized_policy(z_dummy, h_dummy)
        logits = policy_dist.logits  # [1, vocab_size]
        
        # Get top 3 predictions
        probs = torch.softmax(logits, dim=-1)
        top_k = 3
        top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
        
        # Decode predictions
        tokenizer = orchestrator.markov_blanket.tokenizer
        predictions = []
        for i in range(top_k):
            pred_id = top_ids[0, i].item()
            pred_prob = top_probs[0, i].item()
            token = tokenizer.decode([pred_id])
            if token not in ['[PAD]', '[UNK]', '[SEP]', '[CLS]']:
                predictions.append((token, pred_prob))
        
        # Use first non-special token
        generated = predictions[0][0] if predictions else ""
    
    # Check if correct
    is_correct = generated.strip() == expected_answer.strip()
    
    return generated, is_correct, predictions


def run_comprehensive_tests():
    print("=" * 80)
    print("COMPREHENSIVE SIMPLE POLICY EVALUATION")
    print("=" * 80)
    print()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint_path = "checkpoint_simple_large.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    config.device = str(device)
    
    orchestrator = Orchestrator(config)
    orchestrator.load_state_dict(checkpoint['model_state'])
    orchestrator = orchestrator.to(device)
    orchestrator.eval()
    
    print(f"✓ Model loaded on {device}")
    print()
    
    # Comprehensive test cases
    test_categories = [
        ("Basic Addition (Single Digit)", [
            ("If A = 1 and B = 1, then A + B =", "2"),
            ("If A = 1 and B = 2, then A + B =", "3"),
            ("If A = 2 and B = 2, then A + B =", "4"),
            ("If A = 2 and B = 3, then A + B =", "5"),
            ("If A = 3 and B = 3, then A + B =", "6"),
            ("If A = 3 and B = 4, then A + B =", "7"),
            ("If A = 4 and B = 4, then A + B =", "8"),
            ("If A = 4 and B = 5, then A + B =", "9"),
        ]),
        
        ("Addition (Different Formats)", [
            ("What is 1 + 1 =", "2"),
            ("What is 2 + 2 =", "4"),
            ("What is 3 + 3 =", "6"),
            ("Add 2 and 3 =", "5"),
            ("Add 1 and 4 =", "5"),
        ]),
        
        ("Addition (Two Digits - Expected to Fail)", [
            ("If A = 5 and B = 5, then A + B =", "10"),
            ("If A = 6 and B = 6, then A + B =", "12"),
            ("What is 10 + 5 =", "15"),
        ]),
        
        ("Sequences (+1)", [
            ("Write the next number: 0, 1, 2, 3,", "4"),
            ("Write the next number: 1, 2, 3, 4,", "5"),
            ("Write the next number: 2, 3, 4, 5,", "6"),
            ("Write the next number: 5, 6, 7, 8,", "9"),
        ]),
        
        ("Sequences (+5)", [
            ("Write the next number: 0, 5, 10, 15,", "20"),  # May fail
            ("Write the next number: 5, 10, 15, 20,", "25"),  # May fail
        ]),
        
        ("Sequences (+2)", [
            ("Write the next number: 0, 2, 4, 6,", "8"),
            ("Write the next number: 2, 4, 6, 8,", "10"),  # May fail (two digits)
        ]),
        
        ("Letter Sequences", [
            ("Write the next letters in sequence: A, B, C, D,", "E"),
            ("Write the next letters in sequence: C, D, E, F,", "G"),
        ]),
        
        ("Counting", [
            ("If count of X in X is", "1"),
            ("If count of X in XX is", "2"),
            ("If count of X in XXX is", "3"),
            ("If count of A in AAA is", "3"),
        ]),
        
        ("Number Words", [
            ("The word for number 1 is", "one"),
            ("The word for number 2 is", "two"),
            ("The word for number 5 is", "five"),
        ]),
        
        ("Logic (Basic)", [
            ("The antonym of hot is", "cold"),
            ("Complete the logic: sky is blue, grass is", "green"),
        ]),
    ]
    
    overall_correct = 0
    overall_total = 0
    category_results = []
    
    for category_name, questions in test_categories:
        print("=" * 80)
        print(f"CATEGORY: {category_name}")
        print("=" * 80)
        print()
        
        category_correct = 0
        
        for question, expected in questions:
            answer, is_correct, predictions = test_question(orchestrator, question, expected)
            
            status = "✅" if is_correct else "❌"
            print(f"{status} Q: {question}")
            print(f"   Expected: '{expected}' | Got: '{answer}'")
            
            # Show top 3 predictions
            if len(predictions) > 0:
                pred_str = ", ".join([f"'{p[0]}' ({p[1]*100:.1f}%)" for p in predictions[:3]])
                print(f"   Top predictions: {pred_str}")
            
            print()
            
            if is_correct:
                category_correct += 1
                overall_correct += 1
            overall_total += 1
        
        accuracy = 100.0 * category_correct / len(questions)
        print(f"Category Score: {category_correct}/{len(questions)} ({accuracy:.1f}%)")
        print()
        
        category_results.append((category_name, category_correct, len(questions), accuracy))
    
    # Final summary
    print("=" * 80)
    print("SUMMARY BY CATEGORY")
    print("=" * 80)
    print()
    
    for cat_name, correct, total, acc in category_results:
        status = "✅" if acc >= 70 else "⚠️" if acc >= 40 else "❌"
        print(f"{status} {cat_name:45s} {correct:2d}/{total:2d} ({acc:5.1f}%)")
    
    print()
    print("=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    overall_accuracy = 100.0 * overall_correct / overall_total
    print(f"Total Correct: {overall_correct}/{overall_total}")
    print(f"Overall Accuracy: {overall_accuracy:.1f}%")
    print()
    
    # Verdict
    if overall_accuracy >= 60:
        print("🎉 GOOD: AI shows decent learning capabilities!")
    elif overall_accuracy >= 40:
        print("✅ OKAY: AI has learned some patterns")
    elif overall_accuracy >= 20:
        print("⚠️  LIMITED: AI shows minimal learning")
    else:
        print("❌ POOR: AI hasn't learned effectively")
    
    print()
    
    # Key insights
    print("KEY INSIGHTS:")
    print("-" * 80)
    
    # Find best and worst categories
    sorted_results = sorted(category_results, key=lambda x: x[3], reverse=True)
    best_cat = sorted_results[0]
    worst_cat = sorted_results[-1]
    
    print(f"✅ Best: {best_cat[0]} ({best_cat[3]:.1f}%)")
    print(f"❌ Worst: {worst_cat[0]} ({worst_cat[3]:.1f}%)")
    print()
    
    # Limitations
    print("LIMITATIONS:")
    print("-" * 80)
    print("• Can only predict single tokens (not '10', '25', 'cold', etc.)")
    print("• Struggles with multi-character answers")
    print("• Works best with single digit/letter outputs")
    print("=" * 80)


if __name__ == "__main__":
    run_comprehensive_tests()
