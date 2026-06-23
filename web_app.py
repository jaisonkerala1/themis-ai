"""
Themis AI - Interactive Web Interface
A simple Flask web app to interact with your trained AI
"""

import sys
import os
import torch
from flask import Flask, render_template, request, jsonify

sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator

# Initialize Flask app
app = Flask(__name__)

# Global model variables
model = None
config = None
device = None

def load_model():
    """Load the trained model once at startup"""
    global model, config, device
    
    print("Loading Themis AI model...")
    config = ThemisConfig()
    device = config.resolve_device()
    
    model = Orchestrator(config)
    # Load the working Active Inference model (trained with AMP fix + end-to-end gradients)
    checkpoint_path = "checkpoint_real_ai.pt" if os.path.exists("checkpoint_real_ai.pt") else "checkpoint_simple.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print(f"✓ Model loaded successfully on {device}")
    print(f"✓ Checkpoint: {checkpoint_path}")
    print(f"✓ Parameters: ~3.5M")
    print("✓ Ready to answer questions!")

def predict_answer(question):
    """Generate a full multi-token answer using autoregressive decoding.

    Replicates training: carry the recurrent state forward and feed the
    previous action each step, so the model tracks progress and knows when
    to stop ([EOS]). This produces full answers like "10", "Paris", "cold".
    """
    tokenizer = model.markov_blanket.tokenizer
    max_tokens = 12
    generated = ""
    steps = []

    with torch.no_grad():
        states = model.world_model.get_initial_states(1, device, torch.float32)
        prev_action = None
        for _ in range(max_tokens):
            context = question + " " + generated

            obs_dist = model.markov_blanket.encode_batch([context])
            h_states, priors = model.world_model.compute_priors(states, prev_action)
            posteriors = model.world_model.recognition(obs_dist.mean, h_states, priors)

            # Deterministic decoding: use posterior mean
            z1 = posteriors[0].mean
            policy_dist = model.planning_engine.amortized_policy(z1, h_states[0])
            probs = policy_dist.probs[0]

            token_id = int(torch.argmax(probs).item())
            conf = float(probs[token_id].item())

            # Stop conditions
            if token_id in (tokenizer.eos_id, tokenizer.pad_id, tokenizer.bos_id):
                break

            token_str = tokenizer.decode([token_id])
            if token_str == "":
                break

            generated += token_str
            steps.append({'token': token_str, 'confidence': conf})

            # Carry recurrent state forward and remember the action taken
            states = model.world_model.sample_posteriors(h_states, posteriors, use_mean=True)
            prev_action = torch.tensor([token_id], dtype=torch.long, device=device)

    answer = generated.strip()
    if answer == "":
        answer = "(no answer)"

    return {
        'answer': answer,
        'steps': steps
    }

@app.route('/')
def home():
    """Main page"""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Handle question from user"""
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please enter a question'}), 400
        
        # Get AI prediction (full multi-token answer)
        result = predict_answer(question)
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': result['answer'],
            'steps': result['steps'],
            # Keep 'predictions' for backward-compat with the existing UI
            'predictions': [{'token': result['answer'], 'confidence': result['steps'][0]['confidence'] if result['steps'] else 0.0}]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/examples')
def examples():
    """Return example questions"""
    return jsonify({
        'examples': [
            "If A = 1 and B = 2, then A + B =",
            "Write the next number in sequence: 2, 4, 6, 8,",
            "The antonym of hot is",
            "Complete the logic: sky is blue, grass is",
            "Write the next letters in sequence: A, C, E, G,",
            "If count of X in XXYXX is",
            "Capital of France is"
        ]
    })

if __name__ == '__main__':
    # Load model before starting server
    load_model()
    
    print("\n" + "="*60)
    print("🚀 THEMIS AI WEB INTERFACE")
    print("="*60)
    print("\n✓ Server starting...")
    print("✓ Open your browser and go to: http://localhost:5000")
    print("✓ Press Ctrl+C to stop the server\n")
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)
