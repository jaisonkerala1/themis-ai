"""
Themis Models — Action & Transition Models

Provides action embedding modules to project discrete actions to continuous action vectors
used by the transition dynamics (equivalent to B matrix mapping).
"""

import torch
import torch.nn as nn
from torch import Tensor

from themis.config import ThemisConfig


class ActionEmbedding(nn.Module):
    """
    Action Embedding

    Maps discrete actions (token IDs) into a continuous action space (action_dim = 64).
    Supports mapping both single token IDs and batched IDs.
    """
    def __init__(self, config: ThemisConfig):
        super().__init__()
        self.config = config
        wm_config = config.world_model
        
        # Maps discrete token ID to action embedding vector
        self.embedding = nn.Embedding(
            num_embeddings=config.dims.vocab_size,
            embedding_dim=wm_config.action_dim
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        
    def forward(self, action_ids: Tensor) -> Tensor:
        """
        Args:
            action_ids: [batch_size] tensor of action token IDs (discrete).
        Returns:
            action_emb: [batch_size, action_dim] action embedding (continuous).
        """
        # If action_ids is a single scalar int / 0-dim tensor, add batch dimension
        if action_ids.dim() == 0:
            action_ids = action_ids.unsqueeze(0)
            return self.embedding(action_ids).squeeze(0)
            
        return self.embedding(action_ids)
