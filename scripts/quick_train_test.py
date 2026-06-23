"""Quick test - just 10 epochs to see if NaN is fixed"""

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

print("="*60)
print("QUICK TRAINING TEST - 10 EPOCHS")
print("="*60)

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

# Collect 10 episodes
print("[1] Collecting 10 episodes...")
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

print(f"Buffer: {len(replay_buffer)} transitions\n")

# Train for 10 epochs
print("[2] Training for 10 epochs...")
for epoch in range(1, 11):
    obs_b, action_b, done_b = replay_buffer.sample_batch(
        batch_size=4,
        seq_len=4,
        device=device
    )
    
    metrics = trainer.train_step(obs_b, action_b, done_b)
    
    if str(metrics['loss']) == 'nan':
        print(f"Epoch {epoch}: STILL NaN!")
        break
    else:
        print(f"Epoch {epoch}: Loss={metrics['loss']:.4f}, VFE={metrics['vfe']:.4f}, Policy={metrics['policy_loss']:.4f}")

print("\n" + "="*60)
if str(metrics['loss']) != 'nan':
    print("SUCCESS! No NaN in 10 epochs!")
else:
    print("FAILED: Still getting NaN")
print("="*60)
