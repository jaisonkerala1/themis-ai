import sys
import os
import torch

sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator
from environments.reasoning_env import ReasoningEnv

config = ThemisConfig()
orchestrator = Orchestrator(config)
orchestrator.load_checkpoint("checkpoint.pt")
orchestrator.eval()

env = ReasoningEnv()

# Let's check index 0: 'If A = 1 and B = 2, then A + B =' -> Target: '3'
prefix, target = env.dataset[0]
print(f"Task: '{prefix.strip()}' -> Target: '{target.strip()}'")

# Reset and process prefix
orchestrator.reset()
obs_dist = orchestrator.markov_blanket.encode(prefix)
posteriors, _ = orchestrator.perception_engine.update_beliefs(
    world_model=orchestrator.world_model,
    observation=obs_dist,
    prev_states=orchestrator.prev_states,
    action=None
)
h_states, _ = orchestrator.world_model.compute_priors(orchestrator.prev_states, None)
current_states = orchestrator.world_model.sample_posteriors(h_states, posteriors, use_mean=True)

# 1. Inspect Amortized Policy Output
z = current_states[0]["z"]
h = current_states[0]["h"]
policy_dist = orchestrator.planning_engine.amortized_policy(z, h)
probs = policy_dist.probs[0]

# Print top 15 proposed actions from Amortized Policy
top_probs, top_indices = torch.topk(probs, 15)
tokenizer = orchestrator.markov_blanket.tokenizer
print("\nTop 15 proposed actions from Amortized Policy:")
for prob, idx in zip(top_probs, top_indices):
    token = repr(tokenizer.inverse_vocab[int(idx.item())]) if int(idx.item()) in tokenizer.inverse_vocab else "[UNK]"
    print(f"  ID {idx.item():4d}: {token:<15} Prob: {prob.item():.4f}")

# Run planning engine
target_pref = orchestrator.markov_blanket.encode(prefix + target)
action_ids, metrics = orchestrator.planning_engine(current_states, preference=target_pref)
print(f"\nPlanning Engine chosen Action: {action_ids[0].item()} ({repr(tokenizer.inverse_vocab[action_ids[0].item()])})")
print(f"Metrics: {metrics}")
