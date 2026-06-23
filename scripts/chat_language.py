"""
Chat with the language-trained Themis.
Generates English text autoregressively using a sliding context window
and carrying the recurrent state forward (matches training).
"""
import sys
import os
import torch

sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator

CONTEXT_WINDOW = 64


def generate(model, device, prompt, max_tokens=120, temperature=0.8):
    tokenizer = model.markov_blanket.tokenizer
    generated = prompt
    with torch.no_grad():
        states = model.world_model.get_initial_states(1, device, torch.float32)
        prev_action = None
        for _ in range(max_tokens):
            context = generated[-CONTEXT_WINDOW:] if generated else " "
            obs_dist = model.markov_blanket.encode_batch([context])
            h_states, priors = model.world_model.compute_priors(states, prev_action)
            posteriors = model.world_model.recognition(obs_dist.mean, h_states, priors)
            z1 = posteriors[0].mean
            policy_dist = model.planning_engine.amortized_policy(z1, h_states[0])
            logits = policy_dist.logits[0]

            # Temperature sampling for more natural text
            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                token_id = int(torch.multinomial(probs, 1).item())
            else:
                token_id = int(torch.argmax(logits).item())

            if token_id in (tokenizer.eos_id, tokenizer.pad_id, tokenizer.bos_id):
                break
            tok = tokenizer.decode([token_id])
            if tok == "":
                break
            generated += tok

            states = model.world_model.sample_posteriors(h_states, posteriors, use_mean=True)
            prev_action = torch.tensor([token_id], dtype=torch.long, device=device)
    return generated


def main():
    config = ThemisConfig()
    device = config.resolve_device()
    model = Orchestrator(config)
    ckpt = "checkpoint_language.pt"
    if not os.path.exists(ckpt):
        print(f"No {ckpt} found. Train first with scripts/train_language.py")
        return
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"Loaded {ckpt} on {device}\n")

    prompts = [
        "Once upon a time",
        "One day, a little",
        "The dog",
        "She was very happy because",
    ]
    for p in prompts:
        out = generate(model, device, p)
        print(f"PROMPT: {p}")
        print(f"  -> {out}")
        print()


if __name__ == "__main__":
    main()
