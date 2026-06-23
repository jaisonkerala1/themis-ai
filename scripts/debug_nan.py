"""
NaN Debugger - Find exactly where NaN originates
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
from themis.training.replay_buffer import ReplayBuffer


def check_tensor(name, tensor):
    """Check if tensor has NaN or Inf"""
    if tensor is None:
        print(f"  ⚠️  {name}: None")
        return False
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        print(f"  ❌ {name}: NaN={has_nan}, Inf={has_inf}")
        print(f"     Min: {tensor.min().item()}, Max: {tensor.max().item()}")
        return False
    else:
        print(f"  ✅ {name}: OK (min={tensor.min().item():.4f}, max={tensor.max().item():.4f})")
        return True


def debug_training_step():
    """Debug a single training step"""
    
    print("="*80)
    print("NaN DEBUGGING - SINGLE TRAINING STEP")
    print("="*80)
    print()
    
    # Setup
    config = ThemisConfig()
    device = config.resolve_device()
    
    print(f"Device: {device}\n")
    
    # Load dataset
    with open("training_dataset_large.json", 'r') as f:
        dataset = json.load(f)
    
    # Initialize
    orchestrator = Orchestrator(config)
    trainer = ActiveInferenceTrainer(config, orchestrator)
    replay_buffer = ReplayBuffer(config)
    tokenizer = orchestrator.markov_blanket.tokenizer
    
    print("[1] Collecting a few episodes...")
    for episode in range(10):
        task = random.choice(dataset)
        question = task['question']
        answer = task['answer']
        
        obs_list = [question]
        actions_list = []
        dones_list = []
        
        orchestrator.reset(batch_size=1)
        
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        obs = question
        
        for token_id in answer_tokens:
            orchestrator.step(obs, target_text=answer)
            orchestrator.prev_action_ids = torch.tensor([token_id], dtype=torch.long, device=device)
            token_str = tokenizer.decode([token_id])
            obs = question + " " + token_str
            obs_list.append(obs)
            actions_list.append(token_id)
            dones_list.append(len(actions_list) >= len(answer_tokens))
        
        replay_buffer.add_trajectory(obs_list, actions_list, dones_list)
    
    print(f"✓ Buffer: {len(replay_buffer)} transitions\n")
    
    # Sample batch
    print("[2] Sampling training batch...")
    obs_b, action_b, done_b = replay_buffer.sample_batch(
        batch_size=4,
        seq_len=4,
        device=device
    )
    
    print(f"✓ Batch shape: obs={len(obs_b)}x{len(obs_b[0])}, actions={action_b.shape}, done={done_b.shape}\n")
    
    # Check inputs
    print("[3] Checking input data...")
    check_tensor("action_b", action_b)
    check_tensor("done_b", done_b)
    print()
    
    # Check model weights before training
    print("[4] Checking model weights BEFORE training...")
    all_ok = True
    for name, param in orchestrator.named_parameters():
        if not check_tensor(f"Weight: {name}", param.data):
            all_ok = False
            break
    
    if not all_ok:
        print("\n❌ WEIGHTS HAVE NaN BEFORE TRAINING!")
        return
    
    print("\n✅ All weights OK before training\n")
    
    # Try ONE training step with detailed monitoring
    print("[5] Running ONE training step (with monitoring)...")
    print()
    
    try:
        # Manual forward pass to catch NaN
        batch_size = len(obs_b)
        seq_len = len(obs_b[0])
        
        # Reset states
        states = orchestrator.world_model.get_initial_states(
            batch_size=batch_size,
            device=device,
            dtype=config.resolve_dtype()
        )
        
        print("  [a] Initial states:")
        for i, state in enumerate(states):
            check_tensor(f"    Level {i+1} h", state['h'])
            check_tensor(f"    Level {i+1} z", state['z'])
        print()
        
        # Loop through sequence
        for t in range(seq_len):
            print(f"  [b] Time step {t}:")
            
            obs_t = [obs_b[b][t] for b in range(batch_size)]
            prev_action_ids = action_b[:, t-1] if t > 0 else None
            
            # Encode observations
            obs_dist = orchestrator.markov_blanket.encode_batch(obs_t, device=device)
            check_tensor(f"    Obs encoding mean", obs_dist.mean)
            check_tensor(f"    Obs encoding log_var", obs_dist.log_var)
            
            # Compute priors
            h_states, priors = orchestrator.world_model.compute_priors(states, prev_action_ids)
            
            for i in range(3):
                check_tensor(f"    Prior {i+1} mean", priors[i].mean)
                check_tensor(f"    Prior {i+1} log_var", priors[i].log_var)
            
            # Check for NaN here
            has_nan = any([
                torch.isnan(priors[i].mean).any() or torch.isnan(priors[i].log_var).any()
                for i in range(3)
            ])
            
            if has_nan:
                print("\n❌ NaN DETECTED IN PRIORS!")
                print("This is where the problem starts!")
                return
            
            print()
            
            # Continue with posteriors
            posteriors, _ = orchestrator.perception_engine.update_beliefs(
                world_model=orchestrator.world_model,
                observation=obs_dist,
                prev_states=states,
                action=prev_action_ids
            )
            
            for i in range(3):
                has_nan = check_tensor(f"    Posterior {i+1} mean", posteriors[i].mean)
                check_tensor(f"    Posterior {i+1} log_var", posteriors[i].log_var)
                
                if not has_nan:
                    print("\n❌ NaN DETECTED IN POSTERIORS!")
                    print("Problem is in perception engine!")
                    return
            
            # Sample next states
            states = orchestrator.world_model.sample_posteriors(h_states, posteriors)
            states = [
                {"h": s["h"], "z": s["z"].detach()}
                for s in states
            ]
            
            print()
        
        print("✅ Forward pass completed without NaN!")
        print()
        
        # Now try full training step
        print("[6] Running full training step with gradients...")
        metrics = trainer.train_step(obs_b, action_b, done_b)
        
        print(f"\nMetrics:")
        print(f"  VFE: {metrics['vfe']}")
        print(f"  Policy Loss: {metrics['policy_loss']}")
        print(f"  Total Loss: {metrics['loss']}")
        
        if metrics['loss'] != metrics['loss']:  # NaN check
            print("\n❌ LOSS IS NaN!")
        else:
            print("\n✅ TRAINING STEP SUCCESSFUL!")
        
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_training_step()
