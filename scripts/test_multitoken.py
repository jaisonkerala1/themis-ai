"""
Test multi-token generation: full answers like "10", "Paris", "cold"
using autoregressive decoding (feed output back until [EOS]).
"""
import sys
import os
import torch

sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator


def generate(model, device, question, max_tokens=12):
    """Autoregressive generation that REPLICATES training:
    carry the recurrent state forward and feed the previous action,
    so the model knows how far along it is and when to stop ([EOS]).
    """
    tokenizer = model.markov_blanket.tokenizer
    generated = ""
    with torch.no_grad():
        states = model.world_model.get_initial_states(1, device, torch.float32)
        prev_action = None
        for _ in range(max_tokens):
            context = question + " " + generated
            obs_dist = model.markov_blanket.encode_batch([context])

            # Use previous state + previous action (matches training recurrence)
            h_states, priors = model.world_model.compute_priors(states, prev_action)
            posteriors = model.world_model.recognition(obs_dist.mean, h_states, priors)

            z1 = posteriors[0].mean  # deterministic
            policy_dist = model.planning_engine.amortized_policy(z1, h_states[0])
            probs = policy_dist.probs[0]
            token_id = int(torch.argmax(probs).item())

            if token_id in (tokenizer.eos_id, tokenizer.pad_id, tokenizer.bos_id):
                break
            tok = tokenizer.decode([token_id])
            if tok == "":
                break
            generated += tok

            # Carry recurrent state forward and remember the action taken
            states = model.world_model.sample_posteriors(h_states, posteriors, use_mean=True)
            prev_action = torch.tensor([token_id], dtype=torch.long, device=device)
    return generated.strip()


def main():
    config = ThemisConfig()
    device = config.resolve_device()
    model = Orchestrator(config)
    checkpoint = torch.load("checkpoint_real_ai.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"Model loaded on {device}\n")

    tests = [
        # multi-digit numbers
        ("1+1=", "2"),
        ("2+3=", "5"),
        ("5+5=", "10"),
        ("9+9=", "18"),
        ("6+6=", "12"),
        ("7+8=", "15"),
        ("If A = 5 and B = 5, then A + B =", "10"),
        # word facts
        ("Capital of France is", "Paris"),
        ("The antonym of hot is", "cold"),
        ("Complete the logic: sky is blue, grass is", "green"),
        ("The color of the sun is", "yellow"),
        ("A baby dog is called a", "puppy"),
    ]

    correct = 0
    for q, expected in tests:
        got = generate(model, device, q)
        ok = got.lower() == expected.lower()
        if ok:
            correct += 1
        status = "OK " if ok else "XX "
        print(f"{status} '{q}' -> '{got}'   (expected '{expected}')")

    print(f"\nFULL-ANSWER Score: {correct}/{len(tests)} ({100*correct/len(tests):.0f}%)")


if __name__ == "__main__":
    main()
