"""
Themis Demonstration Client — Interactive CLI

Simulates a coordinated active inference loop in real-time, showing:
1. Sensory inputs matching the Markov Blanket.
2. Progressive VFE convergence in the Perception Engine.
3. Policy choices under the Expected Free Energy Planning Engine.
4. Final active step generation.
"""

import sys
import os
import argparse
import time
import torch

# Add current workspace to PYTHONPATH
sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator
from environments.reasoning_env import ReasoningEnv


def run_interactive_demo(auto: bool = False):
    print("====================================================")
    print("           THEMIS ACTIVE INFERENCE SYSTEM DEMO      ")
    print("====================================================")
    print("This demo showcases the 7-layer coordinated loop.")
    print("Perception updates minimize VFE (surprise) via localized")
    print("inner-loop optimization, and action outputs minimize expected VFE.")
    
    # 1. Initialize config and agent
    config = ThemisConfig()
    # Configure parameters for robust active inference
    config.perception.n_iterations = 8
    config.planning.n_candidate_policies = 32
    config.planning.n_policy_samples = 4
    config.planning.planning_horizon = 8
    config.planning.temperature = 0.01
    config.planning.efe_epistemic_weight = 0.0
    
    device = config.resolve_device()
    orchestrator = Orchestrator(config)
    if os.path.exists("checkpoint.pt"):
        print("Loading trained model checkpoint...")
        orchestrator.load_checkpoint("checkpoint.pt")
    orchestrator.reset()
    
    # 2. Select environment task
    env = ReasoningEnv()
    
    print("\nSelect a reasoning task to demonstrate:")
    for i, (prefix, target) in enumerate(env.dataset):
        print(f"  [{i}] Prefix: '{prefix.strip()}' -> Target: '{target.strip()}'")
        
    if auto:
        selected_idx = 0
        print(f"\nAuto mode selected task index: {selected_idx}")
    else:
        try:
            choice = input("\nEnter index choice (default 0): ").strip()
            selected_idx = int(choice) if choice else 0
            if selected_idx < 0 or selected_idx >= len(env.dataset):
                selected_idx = 0
        except ValueError:
            selected_idx = 0
            
    # Override dataset selection
    prefix, target = env.dataset[selected_idx]
    env.current_prefix = prefix
    env.current_target = target
    env.generated_text = ""
    env.step_count = 0
    
    print("\n----------------------------------------------------")
    print(f"STARTING EPISODE")
    print(f"Prompt Prefix:     '{prefix}'")
    print(f"Target Completion: '{target}'")
    print("----------------------------------------------------")
    
    done = False
    obs = prefix
    step_num = 1
    
    while not done:
        print(f"\n>>> AGENT STEP {step_num} <<<")
        print(f"Current Observation Context: '{obs}'")
        
        if not auto:
            input("[Press ENTER to trigger perception and planning]")
            
        # Get target preference text (full completed string)
        target_text = env.get_preference()
        
        # Run coordinated step
        action_ids, action_tokens, metrics = orchestrator.step(obs, target_text=target_text)
        action_str = action_tokens[0]
        
        # Display belief convergence
        loop_m = metrics["loop"]
        perc_m = loop_m["perception"]
        print("\n[Layer 2] Perception Belief Update Convergence:")
        print(f"  Perception Iterations: {metrics['n_iterations']}")
        print(f"  VFE Surprise Start:    {perc_m['vfe_start']:.4f}")
        print(f"  VFE Surprise End:      {perc_m['vfe_end']:.4f}")
        print(f"  Net Surprise Reduced:  {metrics['surprise']:.4f}")
        
        # Display Planning metrics
        plan_m = loop_m["planning"]
        print("\n[Layer 4] Expected Free Energy (EFE) Planning:")
        print(f"  Min G (Optimal policy cost):   {plan_m['G_min']:.4f}")
        print(f"  Mean G (Average policy cost):  {plan_m['G_mean']:.4f}")
        print(f"  Selected Action Token ID:      {action_ids[0].item()}")
        print(f"  Selected Action String Token:  '{action_str}'")
        
        # Step the environment
        obs, reward, done, info = env.step(action_str)
        
        print("\n[Layer 1] Environment Sensory Feedback:")
        print(f"  Step Reward / Penalty:      {reward:.2f}")
        print(f"  Accumulated Text Generated: '{info['generated']}'")
        
        step_num += 1
        
    print("\n====================================================")
    print("                EPISODE FINISHED                    ")
    print(f"Final Text:      '{obs}'")
    print(f"Matched Target?: {'SUCCESS [PASS]' if info['matched'] else 'FAIL'}")
    print("====================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Themis Interactive Demo Client")
    parser.add_argument("--auto", action="store_true", help="Run automatically without user input pauses")
    args = parser.parse_args()
    
    run_interactive_demo(auto=args.auto)
