"""
Check if gradients are flowing to recognition networks during training
"""

import sys
import os
import torch
import json
import random

sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator
from themis.training.trainer import ActiveInferenceTrainer


def check_recognition_gradients():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create fresh orchestrator
    config = ThemisConfig()
    config.device = str(device)
    orchestrator = Orchestrator(config)
    orchestrator = orchestrator.to(device)
    orchestrator.train()
    
    # Create trainer
    trainer = ActiveInferenceTrainer(config, orchestrator)
    
    # Load a training example
    dataset = json.load(open("training_dataset_large.json"))
    task = random.choice(dataset)
    
    question = task['question']
    answer = task['answer']
    
    print("=" * 80)
    print("GRADIENT FLOW CHECK")
    print("=" * 80)
    print()
    print(f"Testing with: '{question}' -> '{answer}'")
    print()
    
    # Prepare batch
    obs_seq = [[question]]
    tokenizer = orchestrator.markov_blanket.tokenizer
    action_ids = tokenizer.encode(answer, add_special_tokens=False)
    action_seq = torch.tensor([action_ids], device=device)
    done_seq = torch.ones(1, 1, device=device)
    
    # Forward pass
    metrics = trainer.train_step(obs_seq, action_seq, done_seq)
    
    print("Training Metrics:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  VFE: {metrics['vfe']:.4f}")
    print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
    print()
    
    # Check gradients on recognition networks
    print("Recognition Network Gradients:")
    print()
    
    recognition = orchestrator.world_model.recognition
    
    for level_name, recognition_net in [
        ("Level 1", recognition.level1_recognition),
        ("Level 2", recognition.level2_recognition),
        ("Level 3", recognition.level3_recognition),
    ]:
        print(f"{level_name}:")
        
        # Check first layer weights
        first_layer = recognition_net.network[0]
        if first_layer.weight.grad is not None:
            grad_mean = first_layer.weight.grad.abs().mean().item()
            grad_max = first_layer.weight.grad.abs().max().item()
            print(f"  First layer weight grad: mean={grad_mean:.8f}, max={grad_max:.8f}")
        else:
            print(f"  First layer weight grad: None")
        
        # Check last layer weights
        last_layer = recognition_net.network[-1]
        if last_layer.weight.grad is not None:
            grad_mean = last_layer.weight.grad.abs().mean().item()
            grad_max = last_layer.weight.grad.abs().max().item()
            print(f"  Last layer weight grad: mean={grad_mean:.8f}, max={grad_max:.8f}")
        else:
            print(f"  Last layer weight grad: None")
        
        print()
    
    # Compare to policy gradients
    print("Policy Network Gradients (for comparison):")
    policy = orchestrator.planning_engine.amortized_policy
    first_layer = list(policy.net.children())[0]
    if first_layer.weight.grad is not None:
        grad_mean = first_layer.weight.grad.abs().mean().item()
        grad_max = first_layer.weight.grad.abs().max().item()
        print(f"  Policy first layer grad: mean={grad_mean:.8f}, max={grad_max:.8f}")
    else:
        print(f"  Policy first layer grad: None")
    
    print()
    print("=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    if recognition.level1_recognition.network[0].weight.grad is None:
        print("❌ CRITICAL: No gradients flowing to recognition networks!")
        print("   They are not learning at all!")
    else:
        recog_grad = recognition.level1_recognition.network[0].weight.grad.abs().mean().item()
        policy_grad = first_layer.weight.grad.abs().mean().item() if first_layer.weight.grad is not None else 0
        
        ratio = recog_grad / policy_grad if policy_grad > 0 else 0
        
        print(f"Recognition/Policy gradient ratio: {ratio:.4f}")
        print()
        
        if ratio < 0.001:
            print("❌ PROBLEM: Recognition gradients are 1000x smaller than policy!")
            print("   Recognition networks barely learning")
        elif ratio < 0.1:
            print("⚠️  WARNING: Recognition gradients much smaller than policy")
        else:
            print("✅ Gradients look reasonable")


if __name__ == "__main__":
    check_recognition_gradients()
