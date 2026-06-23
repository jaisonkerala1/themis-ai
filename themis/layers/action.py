"""
Themis Layers — Layer 5: Action Engine

Coordinates the full perception-planning-action cycle of the active inference loop.
Processes sensory inputs, updates internal states, evaluates policies,
and selects actions.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
from torch import Tensor

from themis.config import ThemisConfig
from themis.layers.markov_blanket import MarkovBlanketIO
from themis.layers.perception import PerceptionEngine
from themis.layers.planning import PlanningEngine


class ActionEngine(nn.Module):
    """
    Layer 5: Action Engine

    Manages and executes the active inference step:
    observe -> infer (perception) -> plan -> act (action selection).
    """
    def __init__(
        self,
        config: ThemisConfig,
        markov_blanket: MarkovBlanketIO,
        perception_engine: PerceptionEngine,
        world_model: nn.Module,
        planning_engine: PlanningEngine
    ):
        super().__init__()
        self.config = config
        self.markov_blanket = markov_blanket
        self.perception_engine = perception_engine
        self.world_model = world_model
        self.planning_engine = planning_engine

    def step(
        self,
        observation_text: Union[str, List[str]],
        prev_states: List[Dict[str, Tensor]],
        prev_action_ids: Optional[Tensor] = None,
        target_text: Optional[Union[str, List[str]]] = None
    ) -> Tuple[Tensor, List[str], List[Dict[str, Tensor]], Dict[str, Any]]:
        """
        Executes a single active inference step.
        
        Args:
            observation_text: Raw input text (string or list of strings).
            prev_states: List of state dictionaries representing [Level 1, Level 2, Level 3] states from previous step.
            prev_action_ids: Action token IDs [batch_size] executed in the previous step.
            target_text: Optional target text (string or list of strings) representing the goal.
            
        Returns:
            action_ids: Selected action token IDs [batch_size]
            action_tokens: Decoded action string tokens for each batch item.
            current_states: Updated state dictionaries list.
            metrics: Accumulated step metrics.
        """
        # Ensure list representation of texts
        if isinstance(observation_text, str):
            observation_texts = [observation_text]
        else:
            observation_texts = observation_text
            
        # 1. Encode sensory input (Layer 1 Input)
        obs_dist = self.markov_blanket.encode_batch(observation_texts)
        
        # 2. Update belief posteriors via recognition network (single-step amortized inference)
        # PROFESSIONAL SOLUTION: Use trained recognition networks like VAE encoders
        h_states, priors = self.world_model.compute_priors(prev_states, prev_action_ids)
        
        # Bottom-up inference through recognition networks
        posteriors = self.world_model.recognition(obs_dist.mean, h_states, priors)
        
        # Create perception metrics
        perception_metrics = {
            'vfe_start': 0.0,
            'vfe_end': 0.0,
            'iterations': 1
        }
        
        # 3. h_states already computed above
        
        # 4. Construct current states (Internal States) by combining h and sampled z
        current_states = self.world_model.sample_posteriors(h_states, posteriors, use_mean=True)
        
        # Encode target preference if target_text is provided
        preference = None
        if target_text is not None:
            if isinstance(target_text, str):
                target_texts = [target_text]
            else:
                target_texts = target_text
            preference = self.markov_blanket.encode_batch(target_texts)

        # 5. Plan next action using Expected Free Energy (Layer 4)
        action_ids, planning_metrics = self.planning_engine(current_states, preference=preference)
        
        # 6. Decode selected actions to strings (Layer 1 Output)
        action_tokens = []
        for idx in action_ids:
            action_tokens.append(self.markov_blanket.tokenizer.decode([idx.item()]))
            
        # Accumulate metrics
        metrics = {
            "perception": perception_metrics,
            "planning": planning_metrics,
            "vfe": perception_metrics["vfe_end"],
            "surprise": perception_metrics["vfe_start"] - perception_metrics["vfe_end"]
        }
        
        return action_ids, action_tokens, current_states, metrics
