"""
Diagnose the current checkpoint to see if it has NaN values
"""
import torch
import sys
import os

sys.path.append(os.getcwd())

checkpoint_path = "checkpoint.pt"

if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    print(f"Checkpoint keys: {checkpoint.keys()}")
    print(f"Training step: {checkpoint.get('step', 'unknown')}")
    
    # Check for NaN values in model weights
    model_state = checkpoint['model_state']
    nan_count = 0
    total_params = 0
    
    for name, param in model_state.items():
        total_params += param.numel()
        if torch.isnan(param).any():
            nan_count += torch.isnan(param).sum().item()
            print(f"  ❌ NaN found in: {name}")
    
    print(f"\n{'='*60}")
    if nan_count > 0:
        print(f"❌ CHECKPOINT IS CORRUPTED!")
        print(f"   Total parameters: {total_params:,}")
        print(f"   NaN parameters: {nan_count:,}")
        print(f"   Corruption rate: {100*nan_count/total_params:.2f}%")
    else:
        print(f"✅ Checkpoint is clean (no NaN values)")
        print(f"   Total parameters: {total_params:,}")
    print(f"{'='*60}")
else:
    print(f"❌ Checkpoint not found: {checkpoint_path}")
