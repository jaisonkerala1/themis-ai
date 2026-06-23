"""
Test the compact addition formats (1+1=, 1 + 1 =) that were added to training
"""
import sys
import os
import torch

sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator


def predict(model, device, question):
    with torch.no_grad():
        obs_dist = model.markov_blanket.encode_batch([question])
        states = model.world_model.get_initial_states(1, device, torch.float32)
        h_states, priors = model.world_model.compute_priors(states, None)
        posteriors = model.world_model.recognition(obs_dist.mean, h_states, priors)
        z1 = posteriors[0].sample()
        policy_dist = model.planning_engine.amortized_policy(z1, h_states[0])
        probs = policy_dist.probs[0]
        top = torch.topk(probs, k=3)
        preds = [(model.markov_blanket.tokenizer.decode([i.item()]), p.item())
                 for i, p in zip(top.indices, top.values)]
        return preds


def main():
    config = ThemisConfig()
    device = config.resolve_device()
    model = Orchestrator(config)
    checkpoint = torch.load("checkpoint_real_ai.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"Model loaded on {device}\n")

    tests = [
        ("1+1=", "2"),
        ("2+3=", "5"),
        ("4+3=", "7"),
        ("3+4=", "7"),
        ("1 + 2 =", "3"),
        ("2 + 2 =", "4"),
        ("5+5=", "10"),
        ("9+9=", "18"),
        ("0+1=", "1"),
        ("6+2=", "8"),
    ]

    correct = 0
    for q, expected in tests:
        preds = predict(model, device, q)
        got = preds[0][0]
        # correct if exact, or first digit matches for two-digit answers
        ok = got.strip() == expected or (len(expected) > 1 and got.strip() == expected[0])
        if ok:
            correct += 1
        status = "OK " if ok else "XX "
        pred_str = ", ".join([f"'{t}' {p*100:.0f}%" for t, p in preds])
        print(f"{status} '{q}' -> got '{got}' (expected '{expected}')   [{pred_str}]")

    print(f"\nScore: {correct}/{len(tests)} ({100*correct/len(tests):.0f}%)")
    print("(two-digit answers count as correct if first digit matches - single-token limit)")


if __name__ == "__main__":
    main()
