"""
Interactive demo showing the AI's step-by-step reasoning process
"""

import sys
import os
import torch

sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator
from environments.reasoning_env import ReasoningEnv


def interactive_demo():
    print("=" * 70)
    print("  THEMIS AI - INTERACTIVE REASONING DEMO")
    print("=" * 70)
    
    config = ThemisConfig()
    device = config.resolve_device()
    
    # Load the simple checkpoint
    orchestrator = Orchestrator(config)
    checkpoint = torch.load("checkpoint_simple.pt", map_location=device, weights_only=False)
    orchestrator.load_state_dict(checkpoint['model_state'])
    orchestrator.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"✓ Device: {device}")
    print(f"✓ Parameters: ~3.5M\n")
    
    # Test on the first 3 tasks with full reasoning display
    env = ReasoningEnv()
    
    test_cases = [
        (0, "Math Addition"),
        (4, "Letter Sequence"),
        (5, "Counting"),
    ]
    
    for task_idx, task_name in test_cases:
        prefix, target = env.dataset[task_idx]
        
        print("=" * 70)
        print(f"TASK: {task_name}")
        print("=" * 70)
        print(f"Question: '{prefix.strip()}'")
        print(f"Expected Answer: '{target}'")
        print()
        
        # Run the full orchestrator step
        orchestrator.reset(batch_size=1)
        
        with torch.no_grad():
            action_ids, action_tokens, metrics = orchestrator.step(
                prefix, 
                target_text=target
            )
        
        predicted_token = action_tokens[0]
        
        print("--- AI REASONING PROCESS ---")
        print()
        
        # Show perception metrics
        perc = metrics['loop']['perception']
        print(f"[Layer 2] Perception Engine:")
        print(f"  • Processed observation in {metrics['n_iterations']} iterations")
        print(f"  • Initial surprise (VFE): {perc['vfe_start']:.2f}")
        print(f"  • Final surprise (VFE): {perc['vfe_end']:.2f}")
        print(f"  • Surprise reduced by: {metrics['surprise']:.2f}")
        print()
        
        # Show planning metrics
        plan = metrics['loop']['planning']
        print(f"[Layer 4] Planning Engine:")
        print(f"  • Evaluated candidate policies")
        print(f"  • Best policy cost (G_min): {plan['G_min']:.2f}")
        print(f"  • Average policy cost (G_mean): {plan['G_mean']:.2f}")
        print(f"  • Selected action confidence: {plan['selected_policy_probs']:.2%}")
        print()
        
        # Show final answer
        print(f"[Layer 5] Action Selected:")
        print(f"  • Token ID: {action_ids[0].item()}")
        print(f"  • Predicted Answer: '{predicted_token}'")
        print()
        
        # Verdict
        is_correct = predicted_token.strip() == target.strip()
        if is_correct:
            print("✅ CORRECT! The AI got the right answer!")
        else:
            print(f"❌ INCORRECT. Expected '{target}' but got '{predicted_token}'")
        
        print()
    
    print("=" * 70)
    print("DEMO COMPLETED")
    print("=" * 70)
    print()
    print("Summary:")
    print("• The AI uses Active Inference to reason")
    print("• It minimizes surprise (Free Energy) to understand inputs")
    print("• It plans actions by minimizing Expected Free Energy")
    print("• All decisions are based on probabilistic inference")
    print()


if __name__ == "__main__":
    interactive_demo()
