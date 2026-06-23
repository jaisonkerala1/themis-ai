"""
Themis Layers — Layer 3: Hierarchical World Model (Generative Model)

Brings together state space levels, transition embedding, and likelihood decoders
to coordinate generative dynamics at hierarchical timescales.
"""

from typing import List, Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
from torch import Tensor

from themis.config import ThemisConfig
from themis.core.distributions import GaussianDist
from themis.models.state_space import StateSpaceLevel
from themis.models.transition import ActionEmbedding
from themis.models.likelihood import LikelihoodDecoder
from themis.models.recognition import HierarchicalRecognition


class WorldModel(nn.Module):
    """
    Layer 3: Hierarchical World Model

    Coordinates the Hierarchical Recurrent State-Space Model (H-RSSM) across
    multiple timescales (timescales: Level 1 = 1, Level 2 = 4, Level 3 = 16).
    """
    def __init__(self, config: ThemisConfig):
        super().__init__()
        self.config = config
        wm_config = config.world_model
        dims = config.dims
        
        self.z_dim = wm_config.state_dim_stochastic
        self.h_dim = wm_config.state_dim_deterministic
        
        # 1. Action Embedding (projects action tokens to 64-dim continuous action space)
        self.action_embedding = ActionEmbedding(config)
        
        # 2. Hierarchical State Space Levels
        # Level 1 receives context from Level 2 (z_dim = 32)
        # Level 2 receives context from Level 3 (z_dim = 32)
        # Level 3 receives no top-down context (context_dim = 0)
        self.level1 = StateSpaceLevel(config, level_idx=0, context_dim=self.z_dim)
        self.level2 = StateSpaceLevel(config, level_idx=1, context_dim=self.z_dim)
        self.level3 = StateSpaceLevel(config, level_idx=2, context_dim=0)
        
        # 3. Observation Likelihood Decoder (maps Level 1 states to observation space)
        self.likelihood_decoder = LikelihoodDecoder(config)
        
        # 4. Recognition Networks (amortized bottom-up inference)
        self.recognition = HierarchicalRecognition(config)
        
        # 5. Goal Preference Distribution C (128-dim observation space)
        # Parameterized as a target mean and log-variance
        self.pref_mean = nn.Parameter(torch.zeros(1, dims.obs_embed_dim))
        self.pref_log_var = nn.Parameter(torch.zeros(1, dims.obs_embed_dim))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize preference parameters to default target
        nn.init.constant_(self.pref_mean, 0.0)
        nn.init.constant_(self.pref_log_var, 0.0) # standard normal-like preference by default

    def get_preference(self, batch_size: int = 1) -> GaussianDist:
        """Returns the target preference distribution C broadcasted to batch_size."""
        mean = self.pref_mean.expand(batch_size, -1)
        log_var = self.pref_log_var.expand(batch_size, -1)
        return GaussianDist(mean=mean, log_var=log_var)

    def get_initial_states(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> List[Dict[str, Tensor]]:
        """
        Creates zeroed initial state structures for all 3 levels.
        Returns:
            states: List[Dict[str, Tensor]] where index i corresponds to Level i+1 state.
        """
        h1, z1 = self.level1.initial_state(batch_size, device, dtype)
        h2, z2 = self.level2.initial_state(batch_size, device, dtype)
        h3, z3 = self.level3.initial_state(batch_size, device, dtype)
        
        return [
            {"h": h1, "z": z1},
            {"h": h2, "z": z2},
            {"h": h3, "z": z3}
        ]

    def compute_priors(
        self,
        prev_states: List[Dict[str, Tensor]],
        action_ids: Optional[Tensor] = None
    ) -> Tuple[List[Tensor], List[GaussianDist]]:
        """
        Top-down hierarchical prediction prior step.
        Computes transition deterministic state update and predicts prior z distributions.
        
        Args:
            prev_states: List of dictionaries containing "h" and "z" tensors per level.
            action_ids: Discrete token actions [batch_size] from the previous step.
            
        Returns:
            h_states: List of updated h tensors per level.
            priors: List of GaussianDist priors per level.
        """
        # Embed action if discrete action IDs are provided
        action_emb = None
        if action_ids is not None:
            action_emb = self.action_embedding(action_ids)
            
        # 1. Update Level 3 (Top level, no context)
        h3, prior3 = self.level3(
            prev_h=prev_states[2]["h"],
            prev_z=prev_states[2]["z"]
        )
        
        # 2. Update Level 2 (using Level 3 prior mean as top-down context)
        h2, prior2 = self.level2(
            prev_h=prev_states[1]["h"],
            prev_z=prev_states[1]["z"],
            context=prior3.mean
        )
        
        # 3. Update Level 1 (using Level 2 prior mean as context + action embedding)
        h1, prior1 = self.level1(
            prev_h=prev_states[0]["h"],
            prev_z=prev_states[0]["z"],
            action=action_emb,
            context=prior2.mean
        )
        
        return [h1, h2, h3], [prior1, prior2, prior3]

    def sample_posteriors(
        self,
        h_states: List[Tensor],
        posteriors: List[GaussianDist],
        use_mean: bool = False
    ) -> List[Dict[str, Tensor]]:
        """
        Packs updated h states and posterior beliefs q(z) into a state dictionary list,
        sampling z from the posteriors.
        """
        sampled_states = []
        for i, (h, post) in enumerate(zip(h_states, posteriors)):
            z = post.mean if use_mean else post.sample()
            sampled_states.append({"h": h, "z": z})
        return sampled_states

    def imagine_step(
        self,
        prev_states: List[Dict[str, Tensor]],
        action_ids: Optional[Tensor] = None,
        use_mean: bool = False
    ) -> Tuple[List[Dict[str, Tensor]], List[GaussianDist]]:
        """
        Performs a single step of generative simulation (imagination rollout) under a policy.
        Unlike compute_priors, it immediately samples z from the transition priors
        to build the next state dictionaries without looking at real observations.
        
        Args:
            prev_states: List of state dictionaries.
            action_ids: Discrete action IDs [batch_size].
            use_mean: If True, uses mean of priors instead of sampling.
            
        Returns:
            next_states: List of simulated state dictionaries.
            priors: List of predicted GaussianDist priors per level.
        """
        h_states, priors = self.compute_priors(prev_states, action_ids)
        
        # Sample z from priors to complete the state update
        next_states = []
        for h, prior in zip(h_states, priors):
            z = prior.mean if use_mean else prior.sample()
            next_states.append({"h": h, "z": z})
            
        return next_states, priors
