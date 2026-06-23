import sys
import os
import torch

sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator
from themis.training.trainer import ActiveInferenceTrainer
from themis.training.replay_buffer import ReplayBuffer
from environments.reasoning_env import ReasoningEnv

def diagnose():
    print("=== Policy Training Diagnosis ===")
    config = ThemisConfig()
    device = torch.device("cpu")
    config.device = "cpu"
    config.training.batch_size = 8
    
    orchestrator = Orchestrator(config)
    trainer = ActiveInferenceTrainer(config, orchestrator)
    replay_buffer = ReplayBuffer(config)
    env = ReasoningEnv()
    
    # Collect expert trajectories
    for episode in range(15):
        obs = env.reset()
        obs_list = [obs]
        actions_list = []
        dones_list = []
        
        target_text = env.get_preference()
        target_completion = env.current_target
        
        tokenizer = orchestrator.markov_blanket.tokenizer
        expert_action_ids = tokenizer.encode(target_completion, add_special_tokens=False)
        
        done = False
        for expert_id in expert_action_ids:
            if done:
                break
            action_str = tokenizer.decode([expert_id])
            orchestrator.step(obs, target_text=target_text)
            orchestrator.prev_action_ids = torch.tensor([expert_id], dtype=torch.long, device=device)
            next_obs, _, done, info = env.step(action_str)
            obs_list.append(next_obs)
            actions_list.append(expert_id)
            dones_list.append(done)
            obs = next_obs
            
        replay_buffer.add_trajectory(obs_list, actions_list, dones_list)
        
    print(f"Replay buffer size: {len(replay_buffer)} transitions.")
    
    # Sample a SINGLE fixed batch
    obs_b, action_b, done_b = replay_buffer.sample_batch(
        batch_size=8,
        seq_len=4,
        device=device
    )
    
    print("\nTraining on a SINGLE fixed batch for 200 iterations...")
    for step in range(1, 201):
        metrics = trainer.train_step(obs_b, action_b, done_b)
        if step % 20 == 0 or step == 1:
            print(f"Step {step:3d} | VFE: {metrics['vfe']:10.4f} | Policy Loss: {metrics['policy_loss']:.6f} | Total Loss: {metrics['loss']:.4f}")
            
    print("\nDiagnostic complete.")

if __name__ == "__main__":
    diagnose()
