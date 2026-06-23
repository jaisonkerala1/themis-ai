# ⚖️ Themis — Active Inference AI

> *"The brain is fundamentally an inference machine."* — Karl Friston

**Themis** — named after the Greek titaness of natural law — is a compact AI system built on Karl Friston's **Free Energy Principle** and **Active Inference**. Instead of scaling to trillions of parameters, Themis achieves powerful reasoning by discovering the causal laws governing its world through principled Bayesian inference.

## Core Thesis

A **10M–50M parameter** model with the right inductive biases — causality, uncertainty quantification, hierarchical structure learning — can outperform massive frontier models on:

- 🎯 **Causal Reasoning** — Built-in generative model, not just correlations
- 🔍 **Few-Shot Learning** — Bayesian updating from single examples
- 🧭 **Planning** — Explicit tree search in an internal world model
- 🛡️ **Robustness** — Knows what it doesn't know (calibrated uncertainty)
- 🌱 **Continual Learning** — Structure learning prevents catastrophic forgetting

## Architecture

7-layer hierarchical architecture based on the Active Inference framework:

```
Layer 7: Orchestrator        — Free energy budget management
Layer 6: Meta-Learning       — Structure learning (grow/prune model)
Layer 5: Action Engine       — Policy execution
Layer 4: Planning Engine     — Expected Free Energy minimization
Layer 3: World Model         — Hierarchical state-space generative model
Layer 2: Perception Engine   — Predictive coding (belief updating)
Layer 1: Markov Blanket I/O  — Text encoding/decoding
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Train on a simple environment
python scripts/train.py

# Interactive demo
python scripts/demo.py
```

## Hardware Requirements

- **Minimum**: 4GB VRAM GPU, 8GB RAM
- **Precision**: FP16 mixed precision throughout
- **Target**: Runs on a single consumer GPU

## Theoretical Background

- [Karl Friston — The Free Energy Principle](https://en.wikipedia.org/wiki/Free_energy_principle)
- [Active Inference — A Process Theory](https://doi.org/10.1162/neco_a_01357)
- [pymdp — Active Inference in Python](https://github.com/infer-actively/pymdp)
- [VERSES AI — AXIOM Architecture](https://arxiv.org/abs/2310.00603)

## License

MIT
