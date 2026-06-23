"""
Training Data Analysis Script
Analyzes whether the training data is sufficient for the task.
"""

import sys
import os
sys.path.append(os.getcwd())

from environments.reasoning_env import ReasoningEnv
from themis.layers.markov_blanket import MarkovBlanketIO
from themis.config import ThemisConfig

def analyze_training_data():
    print("=" * 60)
    print("TRAINING DATA SUFFICIENCY ANALYSIS")
    print("=" * 60)
    
    # 1. Dataset Size Analysis
    env = ReasoningEnv()
    dataset_size = len(env.dataset)
    print(f"\n[1] DATASET STATISTICS:")
    print(f"  Total unique tasks: {dataset_size}")
    print(f"  Training episodes: 30 (configured in train.py)")
    
    # Calculate how many times each task is seen
    episodes_per_task = 30 / dataset_size
    print(f"  Episodes per task: {episodes_per_task:.1f}")
    
    # 2. Token-level analysis
    config = ThemisConfig()
    markov_blanket = MarkovBlanketIO(config)
    tokenizer = markov_blanket.tokenizer
    
    print(f"\n[2] TOKEN-LEVEL ANALYSIS:")
    total_tokens = 0
    unique_outputs = set()
    
    for i, (prefix, target) in enumerate(env.dataset):
        target_tokens = tokenizer.encode(target, add_special_tokens=False)
        total_tokens += len(target_tokens)
        unique_outputs.add(target)
        print(f"  Task {i}: '{prefix.strip()}' -> '{target.strip()}'")
        print(f"    Target tokens: {target_tokens} ({len(target_tokens)} tokens)")
    
    avg_tokens_per_task = total_tokens / dataset_size
    print(f"\n  Total target tokens across all tasks: {total_tokens}")
    print(f"  Average tokens per task: {avg_tokens_per_task:.1f}")
    print(f"  Unique output strings: {len(unique_outputs)}")
    
    # 3. Training iterations analysis
    batch_size = 8
    seq_len = 4
    epochs = 1200
    
    print(f"\n[3] TRAINING EXPOSURE ANALYSIS:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Training epochs: {epochs}")
    print(f"  Total gradient updates: {epochs}")
    
    # Estimate how many times each transition is seen
    # 30 episodes, avg ~2 tokens per task = ~60 transitions total
    estimated_transitions = 30 * avg_tokens_per_task
    samples_per_epoch = batch_size * seq_len
    times_seen = (epochs * samples_per_epoch) / estimated_transitions
    
    print(f"  Estimated total transitions: {estimated_transitions:.0f}")
    print(f"  Samples per epoch: {samples_per_epoch}")
    print(f"  Times each transition seen: {times_seen:.1f}x")
    
    # 4. Verdict
    print(f"\n[4] SUFFICIENCY VERDICT:")
    print("-" * 60)
    
    issues = []
    if dataset_size < 100:
        issues.append("❌ VERY SMALL dataset (only 7 tasks)")
    if episodes_per_task < 10:
        issues.append("❌ LOW COVERAGE per task (4.3 episodes per task)")
    if avg_tokens_per_task < 3:
        issues.append("⚠️  SHORT outputs (avg 1-2 tokens)")
    if times_seen < 100:
        issues.append("❌ INSUFFICIENT repetitions (each pattern seen <100x)")
    if unique_outputs != set([t for _, t in env.dataset]):
        issues.append("⚠️  Potential token collision issues")
    
    if issues:
        print("TRAINING DATA IS INSUFFICIENT!")
        for issue in issues:
            print(f"  {issue}")
        
        print(f"\n[5] RECOMMENDATIONS:")
        print("  1. Increase episodes from 30 to 500+ (cover each task 70+ times)")
        print("  2. Increase epochs from 1200 to 5000+ (more repetitions)")
        print("  3. Add more diverse tasks to dataset (current: 7 tasks)")
        print("  4. Increase batch_size to 16 or 32 for better gradient signal")
        print("  5. Use longer sequences (seq_len=8 instead of 4)")
    else:
        print("✅ Training data appears sufficient!")
    
    print("=" * 60)

if __name__ == "__main__":
    analyze_training_data()
