"""
Deep Diagnosis: Test each Active Inference component independently
to find what's causing the mode collapse (always predicting 'O' or '7')
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


def load_checkpoint(path="checkpoint_real_ai.pt"):
    """Load trained checkpoint"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    config.device = str(device)
    
    orchestrator = Orchestrator(config)
    orchestrator.load_state_dict(checkpoint['model_state'])
    orchestrator = orchestrator.to(device)
    orchestrator.eval()
    
    return orchestrator, device


def test_1_markov_blanket():
    """Test 1: Does the sensory encoder work properly?"""
    print("=" * 80)
    print("TEST 1: Markov Blanket (Sensory Encoder)")
    print("=" * 80)
    print()
    
    orchestrator, device = load_checkpoint()
    
    test_inputs = [
        "If A = 1 and B = 2, then A + B =",
        "If A = 5 and B = 5, then A + B =",
        "What is 2 + 2 =",
        "Write the next number: 1, 2, 3, 4,",
    ]
    
    print("Testing if different inputs produce different encodings...")
    print()
    
    encodings = []
    for text in test_inputs:
        obs_dist = orchestrator.markov_blanket.encode(text)
        encoding = obs_dist.mean
        encodings.append(encoding)
        
        print(f"Text: '{text}'")
        print(f"  Encoding shape: {encoding.shape}")
        print(f"  Encoding mean: {encoding.mean().item():.4f}")
        print(f"  Encoding std: {encoding.std().item():.4f}")
        print(f"  First 5 values: {encoding[0, :5].tolist()}")
        print()
    
    # Check if encodings are different
    print("Checking if encodings are distinct...")
    for i in range(len(encodings)):
        for j in range(i+1, len(encodings)):
            diff = (encodings[i] - encodings[j]).abs().mean().item()
            print(f"  Difference between input {i} and {j}: {diff:.4f}")
    
    print()
    if all([(encodings[i] - encodings[j]).abs().mean().item() > 0.1 
            for i in range(len(encodings)) for j in range(i+1, len(encodings))]):
        print("✅ PASS: Encodings are sufficiently different")
    else:
        print("❌ FAIL: Encodings are too similar - encoder might be broken")
    
    print()


def test_2_recognition_networks():
    """Test 2: Do recognition networks produce different posteriors?"""
    print("=" * 80)
    print("TEST 2: Recognition Networks")
    print("=" * 80)
    print()
    
    orchestrator, device = load_checkpoint()
    
    test_inputs = [
        "If A = 1 and B = 2, then A + B =",
        "If A = 5 and B = 5, then A + B =",
    ]
    
    print("Testing if recognition networks respond to different inputs...")
    print()
    
    for text in test_inputs:
        # Encode
        obs_dist = orchestrator.markov_blanket.encode(text)
        
        # Get initial states
        states = orchestrator.world_model.get_initial_states(1, device, torch.float32)
        
        # Compute priors
        h_states, priors = orchestrator.world_model.compute_priors(states, None)
        
        # Get posteriors via recognition networks
        posteriors = orchestrator.world_model.recognition(obs_dist.mean, h_states, priors)
        
        print(f"Text: '{text}'")
        print(f"  Prior L1 mean: {priors[0].mean[0, :5].tolist()}")
        print(f"  Posterior L1 mean: {posteriors[0].mean[0, :5].tolist()}")
        print(f"  Difference: {(posteriors[0].mean - priors[0].mean).abs().mean().item():.6f}")
        print()
    
    print("✅ If differences are > 0.001, recognition networks are working")
    print()


def test_3_policy_distribution():
    """Test 3: Does the policy produce a proper distribution?"""
    print("=" * 80)
    print("TEST 3: Policy Network Distribution")
    print("=" * 80)
    print()
    
    orchestrator, device = load_checkpoint()
    
    text = "If A = 1 and B = 2, then A + B ="
    
    # Get states
    obs_dist = orchestrator.markov_blanket.encode(text)
    states = orchestrator.world_model.get_initial_states(1, device, torch.float32)
    h_states, priors = orchestrator.world_model.compute_priors(states, None)
    posteriors = orchestrator.world_model.recognition(obs_dist.mean, h_states, priors)
    
    # Get policy output
    z1_sample = posteriors[0].sample()
    policy_dist = orchestrator.planning_engine.amortized_policy(z1_sample, h_states[0])
    
    logits = policy_dist.logits
    probs = torch.softmax(logits, dim=-1)
    
    print(f"Testing policy for: '{text}'")
    print(f"Logits shape: {logits.shape}")
    print(f"Logits min/max: {logits.min().item():.4f} / {logits.max().item():.4f}")
    print(f"Logits mean/std: {logits.mean().item():.4f} / {logits.std().item():.4f}")
    print()
    
    # Check top predictions
    top_k = 10
    top_probs, top_ids = torch.topk(probs[0], k=top_k)
    
    print(f"Top {top_k} predictions:")
    tokenizer = orchestrator.markov_blanket.tokenizer
    for i in range(top_k):
        token_id = top_ids[i].item()
        token = tokenizer.decode([token_id])
        prob = top_probs[i].item()
        print(f"  {i+1}. '{token}' (ID: {token_id}, prob: {prob*100:.2f}%)")
    
    print()
    
    # Check entropy
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    max_entropy = torch.log(torch.tensor(probs.shape[-1], dtype=torch.float32)).item()
    
    print(f"Distribution entropy: {entropy:.4f}")
    print(f"Max possible entropy: {max_entropy:.4f}")
    print(f"Entropy ratio: {entropy/max_entropy:.4f}")
    print()
    
    if entropy / max_entropy < 0.1:
        print("❌ PROBLEM: Distribution has very low entropy (mode collapse!)")
        print("   The policy is too confident in one answer")
    elif top_probs[0].item() > 0.9:
        print("⚠️  WARNING: Top prediction has >90% probability")
        print("   Policy might be overconfident")
    else:
        print("✅ Distribution looks reasonable")
    
    print()


def test_4_loss_gradients():
    """Test 4: Are gradients flowing properly during training?"""
    print("=" * 80)
    print("TEST 4: Loss and Gradient Flow")
    print("=" * 80)
    print()
    
    orchestrator, device = load_checkpoint()
    
    # Load one training example
    dataset = json.load(open("training_dataset_large.json"))
    task = random.choice(dataset)
    
    question = task['question']
    answer = task['answer']
    
    print(f"Testing with: '{question}' -> '{answer}'")
    print()
    
    # Simulate one training step
    from themis.training.trainer import ActiveInferenceTrainer
    
    trainer = ActiveInferenceTrainer(orchestrator, device=device)
    
    # Create batch
    obs_seq = [[question]]
    tokenizer = orchestrator.markov_blanket.tokenizer
    action_ids = tokenizer.encode(answer, add_special_tokens=False)
    action_seq = torch.tensor([action_ids], device=device)
    done_seq = torch.ones(1, 1, device=device)
    
    # Forward pass
    orchestrator.train()
    metrics = trainer.train_step(obs_seq, action_seq, done_seq)
    
    print("Training Metrics:")
    print(f"  Total Loss: {metrics['loss']:.4f}")
    print(f"  VFE: {metrics['vfe']:.4f}")
    print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
    print()
    
    # Check specific losses
    if abs(metrics['loss']) > 1000:
        print("❌ PROBLEM: Loss is exploding (>1000)")
    elif metrics['loss'] > 0:
        print("⚠️  WARNING: Positive loss (VFE should be negative for good fit)")
    else:
        print("✅ Loss magnitude looks reasonable")
    
    print()
    
    # Check if policy loss dominates
    if abs(metrics['policy_loss']) > abs(metrics['vfe']) * 10:
        print("⚠️  WARNING: Policy loss dominates VFE")
        print("   The model is focusing on policy, not perception")
    
    print()


def test_5_belief_states():
    """Test 5: Are belief states meaningful or collapsed?"""
    print("=" * 80)
    print("TEST 5: Belief State Analysis")
    print("=" * 80)
    print()
    
    orchestrator, device = load_checkpoint()
    
    test_inputs = [
        "If A = 1 and B = 2, then A + B =",
        "If A = 5 and B = 5, then A + B =",
        "What is 2 + 2 =",
    ]
    
    print("Analyzing belief states for different inputs...")
    print()
    
    all_posteriors = []
    
    for text in test_inputs:
        obs_dist = orchestrator.markov_blanket.encode(text)
        states = orchestrator.world_model.get_initial_states(1, device, torch.float32)
        h_states, priors = orchestrator.world_model.compute_priors(states, None)
        posteriors = orchestrator.world_model.recognition(obs_dist.mean, h_states, priors)
        
        all_posteriors.append(posteriors[0].mean)
        
        print(f"Text: '{text[:40]}...'")
        print(f"  Posterior L1 mean: {posteriors[0].mean.mean().item():.6f}")
        print(f"  Posterior L1 std: {posteriors[0].mean.std().item():.6f}")
        print()
    
    # Check if posteriors are collapsed (all the same)
    print("Checking if posteriors differ across inputs...")
    differences = []
    for i in range(len(all_posteriors)):
        for j in range(i+1, len(all_posteriors)):
            diff = (all_posteriors[i] - all_posteriors[j]).abs().mean().item()
            differences.append(diff)
            print(f"  Difference {i} vs {j}: {diff:.6f}")
    
    print()
    avg_diff = sum(differences) / len(differences)
    
    if avg_diff < 0.001:
        print("❌ PROBLEM: Posteriors are nearly identical!")
        print("   Recognition networks might not be learning")
    elif avg_diff < 0.01:
        print("⚠️  WARNING: Posteriors are very similar")
    else:
        print("✅ Posteriors show reasonable variation")
    
    print()


def test_6_compare_to_simple():
    """Test 6: Compare with working simple policy"""
    print("=" * 80)
    print("TEST 6: Comparison with Working Simple Policy")
    print("=" * 80)
    print()
    
    # Load both models
    print("Loading Active Inference model...")
    orchestrator_ai, device = load_checkpoint("checkpoint_real_ai.pt")
    
    print("Loading Simple Policy model...")
    checkpoint = torch.load("checkpoint_simple_large.pt", map_location=device, weights_only=False)
    config = checkpoint['config']
    config.device = str(device)
    orchestrator_simple = Orchestrator(config)
    orchestrator_simple.load_state_dict(checkpoint['model_state'])
    orchestrator_simple = orchestrator_simple.to(device)
    orchestrator_simple.eval()
    
    print()
    
    # Test same input
    text = "If A = 1 and B = 2, then A + B ="
    
    print(f"Testing: '{text}'")
    print()
    
    # Active Inference prediction
    with torch.no_grad():
        obs_dist = orchestrator_ai.markov_blanket.encode(text)
        states = orchestrator_ai.world_model.get_initial_states(1, device, torch.float32)
        h_states, priors = orchestrator_ai.world_model.compute_priors(states, None)
        posteriors = orchestrator_ai.world_model.recognition(obs_dist.mean, h_states, priors)
        z1 = posteriors[0].sample()
        policy_dist_ai = orchestrator_ai.planning_engine.amortized_policy(z1, h_states[0])
        probs_ai = torch.softmax(policy_dist_ai.logits, dim=-1)
    
    # Simple policy prediction
    with torch.no_grad():
        obs_dist = orchestrator_simple.markov_blanket.encode(text)
        obs_embed = obs_dist.mean
        z_dim = orchestrator_simple.config.world_model.state_dim_stochastic
        h_dim = orchestrator_simple.config.world_model.state_dim_deterministic
        z_dummy = torch.zeros(1, z_dim, device=device)
        h_dummy = obs_embed[:, :h_dim] if h_dim <= 128 else torch.zeros(1, h_dim, device=device)
        policy_dist_simple = orchestrator_simple.planning_engine.amortized_policy(z_dummy, h_dummy)
        probs_simple = torch.softmax(policy_dist_simple.logits, dim=-1)
    
    # Compare top predictions
    tokenizer = orchestrator_ai.markov_blanket.tokenizer
    
    print("Active Inference Top 5:")
    top_probs, top_ids = torch.topk(probs_ai[0], k=5)
    for i in range(5):
        token = tokenizer.decode([top_ids[i].item()])
        print(f"  {i+1}. '{token}' ({top_probs[i].item()*100:.1f}%)")
    
    print()
    print("Simple Policy Top 5:")
    top_probs, top_ids = torch.topk(probs_simple[0], k=5)
    for i in range(5):
        token = tokenizer.decode([top_ids[i].item()])
        print(f"  {i+1}. '{token}' ({top_probs[i].item()*100:.1f}%)")
    
    print()
    print("Expected answer: '3'")
    print()


def run_all_tests():
    """Run complete diagnostic suite"""
    print("=" * 80)
    print("DEEP ACTIVE INFERENCE DIAGNOSIS")
    print("=" * 80)
    print()
    print("This will test each component to find what's broken")
    print()
    
    tests = [
        ("Sensory Encoding", test_1_markov_blanket),
        ("Recognition Networks", test_2_recognition_networks),
        ("Policy Distribution", test_3_policy_distribution),
        ("Loss & Gradients", test_4_loss_gradients),
        ("Belief States", test_5_belief_states),
        ("Compare to Working Model", test_6_compare_to_simple),
    ]
    
    for test_name, test_func in tests:
        try:
            test_func()
            input("Press ENTER to continue to next test...")
            print("\n" * 2)
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            print()
            input("Press ENTER to continue...")
    
    print("=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
