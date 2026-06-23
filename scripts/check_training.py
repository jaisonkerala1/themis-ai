"""
Quick script to check if training is complete and show results
"""
import os
import sys

# Check if final checkpoint exists
if os.path.exists("checkpoint_real_ai.pt"):
    print("=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print()
    print("Next step: Test the AI's generalization ability")
    print()
    print("Run this command:")
    print("  .venv\\Scripts\\python.exe scripts\\test_real_ai.py")
    print()
    print("This will show you if the AI can generalize to new questions!")
    print("=" * 70)
else:
    print("=" * 70)
    print("⏳ TRAINING IN PROGRESS...")
    print("=" * 70)
    print()
    print("The training is still running. Expected duration: ~3 hours")
    print()
    print("Checkpoints are saved every 1000 epochs:")
    
    checkpoints = [
        ("checkpoint_real_ai_epoch1000.pt", "Epoch 1000 (~36 min)"),
        ("checkpoint_real_ai_epoch2000.pt", "Epoch 2000 (~72 min)"),
        ("checkpoint_real_ai_epoch3000.pt", "Epoch 3000 (~108 min)"),
        ("checkpoint_real_ai_epoch4000.pt", "Epoch 4000 (~144 min)"),
        ("checkpoint_real_ai_epoch5000.pt", "Epoch 5000 (~180 min)"),
    ]
    
    for checkpoint, label in checkpoints:
        if os.path.exists(checkpoint):
            print(f"  ✅ {label}")
        else:
            print(f"  ⏳ {label}")
    
    print()
    print("Keep your laptop:")
    print("  • Plugged in (don't let battery die!)")
    print("  • Sleep disabled (must stay awake)")
    print("  • Windows updates paused (no restarts!)")
    print()
    print("Run this script again to check progress:")
    print("  .venv\\Scripts\\python.exe scripts\\check_training.py")
    print("=" * 70)
