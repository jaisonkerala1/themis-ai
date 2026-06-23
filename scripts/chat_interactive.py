"""
Interactive chat with the language-trained Themis.
Type a prompt, the AI continues it in English. Type 'quit' to exit.
"""
import sys
import os
import torch

sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator

CONTEXT_WINDOW = 64


def generate(model, device, prompt, max_tokens=150, temperature=0.7):
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
    ckpt = "checkpoint_language.pt"
    if not os.path.exists(ckpt):
        print(f"'{ckpt}' not found. Download it from Google Drive (MyDrive/themis/) into this folder first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    # Use the checkpoint's own config so dimensions always match the weights
    config = checkpoint["config"] if checkpoint.get("config") is not None else ThemisConfig()
    model = Orchestrator(config)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()

    print("=" * 60)
    print("  THEMIS LANGUAGE CHAT  (type 'quit' to exit)")
    print("=" * 60)
    print(f"Loaded {ckpt} on {device}")
    print("Tip: start a sentence and the AI continues it.")
    print("Try: 'Once upon a time' or 'The little boy'")
    print()

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if prompt.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not prompt:
            continue
        out = generate(model, device, prompt)
        print(f"AI : {out}\n")


if __name__ == "__main__":
    main()
