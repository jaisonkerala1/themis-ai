"""
Test the simple policy trained on large dataset
"""

import sys
import os
import torch

sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator


def test_question(orchestrator, question, expected_answer, max_tokens=20):
    """Test a single question"""
    device = orchestrator.device
    orchestrator.reset()
    
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
        
        # Get top predictions
        probs = torch.softmax(logits, dim=-1)
        top_ids = torch.topk(probs, k=min(5, probs.shape[-1]), dim=-1).indices[0]
        
        # Decode predictions
        tokenizer = orchestrator.markov_blanket.tokenizer
        generated = ""
        for pred_id in top_ids:
            token = tokenizer.decode([pred_id.item()])
            if token not in ['[PAD]', '[UNK]', '[SEP]', '[CLS]']:
                generated = token
                break
    
    # Check if correct
    is_correct = generated.strip() == expected_answer.strip()
    
    return generated, is_correct


def run_tests():
    print("=" * 80)
    print("SIMPLE POLICY EVALUATION - LARGE DATASET")
    print("=" * 80)
    print()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint_path = "checkpoint_simple_large.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Run training first: .venv\\Scripts\\python.exe scripts\\train_simple_large.py")
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
    
    # Test cases
    test_cases = [
        # Trained examples (should get high accuracy)
        ("Addition - Trained", [
            ("If A = 1 and B = 2, then A + B =", "3"),
            ("If A = 5 and B = 5, then A + B =", "10"),
            ("What is 2 + 3 =", "5"),
        ]),
        # New variations
        ("Addition - New", [
            ("If A = 3 and B = 4, then A + B =", "7"),
            ("If A = 6 and B = 2, then A + B =", "8"),
            ("What is 5 + 4 =", "9"),
        ]),
        # Sequences
        ("Sequences", [
            ("Write the next number: 1, 2, 3, 4,", "5"),
            ("Write the next number: 5, 10, 15, 20,", "25"),
            ("Write the next number: 10, 20, 30, 40,", "50"),
        ]),
        # Counting
        ("Counting", [
            ("If count of X in XXX is", "3"),
            ("If count of A in AAAA is", "4"),
            ("If count of B in BBB is", "3"),
        ]),
    ]
    
    total_correct = 0
    total_questions = 0
    
    for category, questions in test_cases:
        print("=" * 80)
        print(f"CATEGORY: {category}")
        print("=" * 80)
        print()
        
        category_correct = 0
        
        for question, expected in questions:
            answer, is_correct = test_question(orchestrator, question, expected)
            
            status = "✅" if is_correct else "❌"
            print(f"{status} Q: {question}")
            print(f"   Expected: '{expected}' | Got: '{answer}'")
            print()
            
            if is_correct:
                category_correct += 1
                total_correct += 1
            total_questions += 1
        
        print(f"Category Score: {category_correct}/{len(questions)} ({100*category_correct/len(questions):.1f}%)")
        print()
    
    # Final results
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    accuracy = 100.0 * total_correct / total_questions
    print(f"Total Correct: {total_correct}/{total_questions}")
    print(f"Overall Accuracy: {accuracy:.1f}%")
    print()
    
    if accuracy >= 70:
        print("🎉 EXCELLENT: AI shows good learning!")
    elif accuracy >= 50:
        print("✅ GOOD: AI is learning patterns!")
    elif accuracy >= 30:
        print("⚠️  OKAY: Some learning, needs improvement")
    else:
        print("❌ POOR: Minimal learning occurred")
    
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
