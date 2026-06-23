"""
Themis Configuration — Central hyperparameters for all components.

Hardware target: 8GB RAM, 4GB VRAM (use ≤3GB).
All dimensions are kept small to fit within budget.
FP16 mandatory throughout.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch


# =============================================================================
# Hardware Detection
# =============================================================================

def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype() -> torch.dtype:
    """FP32 everywhere for stability (FP16 causes issues with GRU cells)."""
    return torch.float32


DEVICE = get_device()
DTYPE = get_dtype()


# =============================================================================
# Core Dimensions — Everything flows from these
# =============================================================================

@dataclass
class CoreDims:
    """
    The fundamental dimensions of the system.
    Kept small to fit in 4GB VRAM.
    """
    # Latent state dimensions per hierarchy level
    state_dim: int = 256            # Hidden state size (s) [SCALED UP]
    state_dim_stochastic: int = 64  # Stochastic part of state (z) [SCALED UP]
    state_dim_deterministic: int = 192  # Deterministic part (h) [SCALED UP]

    # Observation / embedding dimensions
    obs_embed_dim: int = 256       # Encoded observation dimension [SCALED UP]
    action_dim: int = 128          # Action embedding dimension [SCALED UP]

    # Text-specific
    vocab_size: int = 8192         # Small BPE vocabulary (text-first, compact)
    max_seq_len: int = 512         # Maximum sequence length
    token_embed_dim: int = 256     # Token embedding dimension [SCALED UP]

    # Hierarchy
    n_hierarchy_levels: int = 3    # Number of world model levels
    timescales: tuple = (1, 4, 16) # Timescale multiplier per level


# =============================================================================
# Layer Configs
# =============================================================================

@dataclass
class PerceptionConfig:
    """Layer 2: Predictive coding engine."""
    n_levels: int = 3              # Hierarchy depth
    n_iterations: int = 8          # Message passing iterations per step
    learning_rate_mu: float = 0.1  # Belief update step size
    learning_rate_pi: float = 0.01 # Precision update step size
    state_dim: int = 128
    precision_floor: float = 1e-4  # Minimum precision (avoid division by zero)


@dataclass
class WorldModelConfig:
    """Layer 3: Hierarchical RSSM generative model."""
    n_levels: int = 3
    state_dim: int = 256
    state_dim_stochastic: int = 64
    state_dim_deterministic: int = 192
    action_dim: int = 128
    obs_embed_dim: int = 256
    hidden_dim: int = 512          # MLP hidden layer size [SCALED UP]
    n_categories: int = 16         # For categorical latent (if used)
    min_std: float = 0.1           # Minimum std for Gaussian posteriors
    timescales: tuple = (1, 4, 16)


@dataclass
class PlanningConfig:
    """Layer 4: Expected Free Energy planning."""
    planning_horizon: int = 8      # How far ahead to plan
    n_candidate_policies: int = 32 # Number of policies to evaluate
    n_policy_samples: int = 4      # Monte Carlo samples per policy
    temperature: float = 1.0       # Softmax temperature for policy selection
    efe_extrinsic_weight: float = 1.0  # Weight for goal-directed component
    efe_epistemic_weight: float = 1.0  # Weight for curiosity component
    use_amortized_policy: bool = True  # Use learned policy network


@dataclass
class ActionConfig:
    """Layer 5: Action execution."""
    action_dim: int = 64
    output_vocab_size: int = 8192  # For text generation
    max_output_len: int = 256


@dataclass
class MetaLearningConfig:
    """Layer 6: Structure learning."""
    expansion_threshold: float = 0.5    # Free energy threshold to trigger growth
    reduction_threshold: float = 0.01   # Redundancy threshold for pruning
    consolidation_interval: int = 100   # Steps between consolidation phases
    max_components: int = 64            # Maximum mixture components


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    learning_rate: float = 3e-4
    batch_size: int = 2            # Small due to VRAM constraints
    grad_accumulation_steps: int = 4  # Effective batch = 8
    max_grad_norm: float = 1.0
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    total_steps: int = 100_000
    replay_buffer_size: int = 10_000  # Small due to RAM constraints
    checkpoint_interval: int = 1000
    log_interval: int = 50
    use_amp: bool = False          # DISABLED: float16 overflows in variance division → NaN gradients
    # Loss weighting: lets us emphasize next-token (policy) learning over
    # observation reconstruction (VFE). Defaults preserve original behavior.
    vfe_weight: float = 1.0
    policy_loss_weight: float = 1.0


# =============================================================================
# Master Config
# =============================================================================

@dataclass
class ThemisConfig:
    """Top-level configuration bundling all sub-configs."""
    dims: CoreDims = field(default_factory=CoreDims)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    meta_learning: MetaLearningConfig = field(default_factory=MetaLearningConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "auto"
    seed: int = 42

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return get_device()
        return torch.device(self.device)

    def resolve_dtype(self) -> torch.dtype:
        return torch.float32

    def estimated_param_count(self) -> str:
        """Rough estimate of total parameters."""
        d = self.dims
        # Very rough: encoder + world model + decoder
        text_encoder = d.vocab_size * d.token_embed_dim  # ~1M
        h_dim = self.world_model.hidden_dim
        world_model_per_level = (
            d.state_dim * h_dim * 2 +  # transition MLP
            h_dim * d.state_dim +       # likelihood
            d.state_dim * d.obs_embed_dim      # posterior
        )
        world_model = world_model_per_level * d.n_hierarchy_levels
        decoder = d.state_dim * h_dim + h_dim * d.vocab_size
        total = text_encoder + world_model + decoder
        if total < 1_000_000:
            return f"~{total / 1000:.0f}K"
        return f"~{total / 1_000_000:.1f}M"
