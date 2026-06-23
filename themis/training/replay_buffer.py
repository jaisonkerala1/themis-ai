"""
Themis Training — Replay Buffer

Implements a lightweight, sequence-based experience replay buffer.
Stores trajectories of interaction step sequences to allow batch sampling
for hierarchical RSSM and policy training. Optimized for 8GB RAM.
"""

from typing import List, Tuple, Dict, Any
import random
import torch
from torch import Tensor

from themis.config import ThemisConfig


class ReplayBuffer:
    """
    ReplayBuffer

    Stores complete trajectory histories (sequence of observations, actions, and dones)
    and samples continuous sub-sequence slices for training.
    """
    def __init__(self, config: ThemisConfig):
        self.config = config
        self.max_size = config.training.replay_buffer_size # e.g., 10000 transitions
        
        # Buffer containing list of trajectories
        # Each trajectory is a dict: {"obs": List[str], "actions": List[int], "dones": List[bool]}
        self.buffer: List[Dict[str, List[Any]]] = []
        self.total_steps = 0
        
    def add_trajectory(self, obs: List[str], actions: List[int], dones: List[bool]):
        """
        Adds a complete episode trajectory.
        """
        traj_len = len(actions)
        if traj_len == 0:
            return
            
        # If incoming trajectory is larger than max_size, truncate to last max_size transitions
        if traj_len > self.max_size:
            obs = obs[-self.max_size:]
            actions = actions[-self.max_size:]
            dones = dones[-self.max_size:]
            traj_len = self.max_size
            
        # Manage memory constraints: prune oldest trajectories if total steps exceed max_size
        while self.total_steps + traj_len > self.max_size and self.buffer:
            removed = self.buffer.pop(0)
            self.total_steps -= len(removed["actions"])
            
        trajectory = {
            "obs": obs,
            "actions": actions,
            "dones": dones
        }
        self.buffer.append(trajectory)
        self.total_steps += traj_len

    def sample_batch(self, batch_size: int, seq_len: int, device: torch.device) -> Tuple[List[List[str]], Tensor, Tensor]:
        """
        Samples a batch of sequence slices of length `seq_len`.
        
        Returns:
            obs_batch: List[List[str]] of shape [batch_size, seq_len]
            action_batch: Tensor [batch_size, seq_len] of token IDs
            done_batch: Tensor [batch_size, seq_len] of done flags (float)
        """
        obs_batch = []
        action_batch_list = []
        done_batch_list = []
        
        for _ in range(batch_size):
            # Select a random trajectory
            traj = random.choice(self.buffer)
            traj_len = len(traj["actions"])
            
            # If trajectory is shorter than seq_len, we slice the whole trajectory and pad it
            if traj_len <= seq_len:
                start_idx = 0
                slice_len = traj_len
            else:
                # Random starting point for slice
                start_idx = random.randint(0, traj_len - seq_len)
                slice_len = seq_len
                
            obs_slice = traj["obs"][start_idx : start_idx + slice_len]
            action_slice = traj["actions"][start_idx : start_idx + slice_len]
            done_slice = traj["dones"][start_idx : start_idx + slice_len]
            
            # Padding if shorter than seq_len
            if slice_len < seq_len:
                pad_len = seq_len - slice_len
                obs_slice += [obs_slice[-1]] * pad_len if obs_slice else [""] * pad_len
                action_slice += [0] * pad_len # index 0 is [PAD]
                done_slice += [True] * pad_len
                
            obs_batch.append(obs_slice)
            action_batch_list.append(torch.tensor(action_slice, dtype=torch.long, device=device))
            done_batch_list.append(torch.tensor(done_slice, dtype=torch.float32, device=device))
            
        actions_tensor = torch.stack(action_batch_list, dim=0) # [batch_size, seq_len]
        dones_tensor = torch.stack(done_batch_list, dim=0) # [batch_size, seq_len]
        
        return obs_batch, actions_tensor, dones_tensor

    def __len__(self) -> int:
        return self.total_steps
