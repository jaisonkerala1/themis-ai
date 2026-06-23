"""
Themis AI - Interactive Web Interface
Loads BOTH models (math/QA and language) and shows both answers so you can
test each in one place.
"""

import sys
import os
import torch
from flask import Flask, render_template, request, jsonify

sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator

app = Flask(__name__)

device = None
math_model = None      # checkpoint_real_ai.pt (math + word facts)
lang_model = None      # checkpoint_language.pt (English stories)

LANG_CONTEXT_WINDOW = 64


def _load_one(path):
    """Load a single model using the config stored in its checkpoint."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=dev, weights_only=False)
    cfg = checkpoint["config"] if checkpoint.get("config") is not None else ThemisConfig()
    m = Orchestrator(cfg)
    m.load_state_dict(checkpoint["model_state"])
    m = m.to(cfg.resolve_device())
    m.eval()
    n = sum(p.numel() for p in m.parameters())
    print(f"  loaded {path}  ({n/1e6:.1f}M params)")
    return m


def load_models():
    global device, math_model, lang_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading models...")
    if os.path.exists("checkpoint_real_ai.pt"):
        math_model = _load_one("checkpoint_real_ai.pt")
    if os.path.exists("checkpoint_language.pt"):
        lang_model = _load_one("checkpoint_language.pt")
    if math_model is None and lang_model is None:
        raise RuntimeError("No checkpoints found (checkpoint_real_ai.pt / checkpoint_language.pt)")
    print(f"Ready on {device}")


def generate_math(question, max_tokens=12):
    """Short-answer generation (math/QA): full context, deterministic, stop at EOS."""
    if math_model is None:
        return None
    m = math_model
    tok = m.markov_blanket.tokenizer
    generated = ""
    with torch.no_grad():
        states = m.world_model.get_initial_states(1, m.config.resolve_device(), torch.float32)
        prev_action = None
        for _ in range(max_tokens):
            context = question + " " + generated
            obs = m.markov_blanket.encode_batch([context])
            h_states, priors = m.world_model.compute_priors(states, prev_action)
            post = m.world_model.recognition(obs.mean, h_states, priors)
            z1 = post[0].mean
            pd = m.planning_engine.amortized_policy(z1, h_states[0])
            tid = int(torch.argmax(pd.probs[0]).item())
            if tid in (tok.eos_id, tok.pad_id, tok.bos_id):
                break
            s = tok.decode([tid])
            if s == "":
                break
            generated += s
            states = m.world_model.sample_posteriors(h_states, post, use_mean=True)
            prev_action = torch.tensor([tid], dtype=torch.long, device=m.config.resolve_device())
    return generated.strip() or "(no answer)"


def generate_language(prompt, max_tokens=140, temperature=0.7):
    """Story-style generation: sliding window, temperature sampling, stop at EOS."""
    if lang_model is None:
        return None
    m = lang_model
    tok = m.markov_blanket.tokenizer
    dev = m.config.resolve_device()
    generated = prompt
    with torch.no_grad():
        states = m.world_model.get_initial_states(1, dev, torch.float32)
        prev_action = None
        for _ in range(max_tokens):
            context = generated[-LANG_CONTEXT_WINDOW:] if generated else " "
            obs = m.markov_blanket.encode_batch([context])
            h_states, priors = m.world_model.compute_priors(states, prev_action)
            post = m.world_model.recognition(obs.mean, h_states, priors)
            z1 = post[0].mean
            pd = m.planning_engine.amortized_policy(z1, h_states[0])
            logits = pd.logits[0]
            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                tid = int(torch.multinomial(probs, 1).item())
            else:
                tid = int(torch.argmax(logits).item())
            if tid in (tok.eos_id, tok.pad_id, tok.bos_id):
                break
            s = tok.decode([tid])
            if s == "":
                break
            generated += s
            states = m.world_model.sample_posteriors(h_states, post, use_mean=True)
            prev_action = torch.tensor([tid], dtype=torch.long, device=dev)
    return generated.strip()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        question = data.get('question', '').strip()
        if not question:
            return jsonify({'error': 'Please enter a question'}), 400

        math_answer = generate_math(question)
        lang_answer = generate_language(question)

        return jsonify({
            'success': True,
            'question': question,
            'math_answer': math_answer,
            'language_answer': lang_answer
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/examples')
def examples():
    return jsonify({
        'examples': [
            "1+1=",
            "Capital of France is",
            "The antonym of hot is",
            "Once upon a time",
            "The little boy",
            "If A = 1 and B = 2, then A + B ="
        ]
    })


if __name__ == '__main__':
    load_models()
    print("\n" + "=" * 60)
    print("THEMIS AI WEB INTERFACE (math + language)")
    print("=" * 60)
    print("Open: http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
