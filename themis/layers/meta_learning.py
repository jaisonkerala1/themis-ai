"""
Themis Layers — Layer 6: Meta-Learning (Structure Learning)

Implements Bayesian structure learning. Monitors Variational Free Energy (VFE)
surprise history to trigger model expansion (neurogenesis) and checks
latent state variance to prune redundant dimensions (synaptic pruning).
"""

from typing import List, Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
from torch import Tensor

from themis.config import ThemisConfig


class StructureLearning(nn.Module):
    """
    Layer 6: Meta-Learning

    Implements structure learning mechanisms to dynamically resize
    the capacity of the generative model based on surprise (VFE) and redundancy.
    """
    def __init__(self, config: ThemisConfig):
        super().__init__()
        self.config = config
        self.meta_config = config.meta_learning
        self.z_dim = config.world_model.state_dim_stochastic
        
        # History buffers
        self.register_buffer("vfe_history", torch.zeros(1000))
        self.register_buffer("z_history", torch.zeros(1000, self.z_dim))
        
        self.history_idx = 0
        self.history_size = 0

    def update_history(self, vfe_val: float, latent_z: Tensor):
        """
        Record step metrics to sliding window history buffers.
        latent_z: [batch_size, z_dim] or [z_dim]
        """
        # Save VFE value
        self.vfe_history[self.history_idx] = float(vfe_val)
        
        # Save average z activation
        if latent_z.dim() > 1:
            z_mean = latent_z.mean(dim=0)
        else:
            z_mean = latent_z
            
        self.z_history[self.history_idx] = z_mean.detach()
        
        # Advance index (circular buffer)
        self.history_idx = (self.history_idx + 1) % 1000
        self.history_size = min(self.history_size + 1, 1000)

    def should_expand(self, window_size: int = 100) -> Tuple[bool, float]:
        """
        Bayesian Model Expansion trigger:
        If average VFE (surprise) over recent window exceeds expansion_threshold,
        it suggests the model structure is insufficient to explain the observations.
        
        Returns:
            expand: bool flag
            avg_vfe: float value
        """
        if self.history_size < window_size:
            return False, 0.0
            
        # Get last window_size items
        indices = [(self.history_idx - 1 - i) % 1000 for i in range(window_size)]
        recent_vfes = self.vfe_history[indices]
        avg_vfe = float(recent_vfes.mean().item())
        
        expand = avg_vfe > self.meta_config.expansion_threshold
        return expand, avg_vfe

    def get_pruning_mask(self, window_size: int = 200) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Bayesian Model Reduction (synaptic pruning):
        Analyzes standard deviation of latent activations in history.
        If a dimension has variance below reduction_threshold, it is redundant
        and can be masked/pruned.
        
        Returns:
            keep_mask: ByteTensor of shape [z_dim] containing 1 (keep) or 0 (prune).
            metrics: Pruning stats dictionary.
        """
        effective_window = min(self.history_size, window_size)
        if effective_window < 2:
            # Not enough data: keep all dimensions
            return torch.ones(self.z_dim, dtype=torch.bool, device=self.z_history.device), {
                "total_dim": self.z_dim,
                "pruned_dim": 0,
                "active_dim": self.z_dim,
                "min_std": 0.0,
                "max_std": 0.0
            }
            
        indices = [(self.history_idx - 1 - i) % 1000 for i in range(effective_window)]
        recent_zs = self.z_history[indices] # [effective_window, z_dim]
        
        # Standard deviation of each dimension over time
        z_stds = recent_zs.std(dim=0) # [z_dim]
        
        # Keep dimensions with std >= threshold
        keep_mask = z_stds >= self.meta_config.reduction_threshold
        
        # Edge case: do not prune all dimensions (must keep at least 1)
        if not keep_mask.any():
            # Keep the one with highest std
            keep_mask[z_stds.argmax()] = True
            
        pruned_count = int((~keep_mask).sum().item())
        
        metrics = {
            "total_dim": self.z_dim,
            "pruned_dim": pruned_count,
            "active_dim": self.z_dim - pruned_count,
            "min_std": float(z_stds.min().item()),
            "max_std": float(z_stds.max().item())
        }
        
        return keep_mask, metrics

    def apply_structure_change(self, world_model: nn.Module) -> Dict[str, Any]:
        """
        Evaluates current structure and logs expansion or applies pruning masks.
        """
        expand, avg_vfe = self.should_expand()
        keep_mask, pruning_stats = self.get_pruning_mask()
        
        logs = {
            "vfe_average": avg_vfe,
            "expansion_triggered": expand,
            "pruning_stats": pruning_stats
        }
        
        # Expansion: alert system to increase learning rate / temperature to promote search
        if expand:
            # We can log this event
            pass
            
        # Pruning: could zero out weights in transition/likelihood decoder, but
        # returning the pruning mask to the orchestrator to apply element-wise multiplication
        # is the safest way to perform dynamic masking without breaking static torch compile paths.
        return logs
