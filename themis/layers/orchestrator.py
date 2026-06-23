"""
Themis Layers — Layer 7: Orchestrator

The top-level executive system coordinating the full Themis agent loop.
Manages the Markov Blanket, Perception, World Model, Planning, Action, and Meta-Learning
layers, and controls adaptive compute budgeting based on global surprise levels.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
from torch import Tensor

from themis.config import ThemisConfig
from themis.layers.markov_blanket import MarkovBlanketIO
from themis.layers.perception import PerceptionEngine
from themis.layers.world_model import WorldModel
from themis.layers.planning import PlanningEngine
from themis.layers.action import ActionEngine
from themis.layers.meta_learning import StructureLearning


class Orchestrator(nn.Module):
    """
    Layer 7: Orchestrator

    The CEO of the agent. Bundles and coordinates all layers of the Themis
    Active Inference stack.
    """
    def __init__(self, config: ThemisConfig):
        super().__init__()
        self.config = config
        self.device = config.resolve_device()
        self.dtype = config.resolve_dtype()
        
        # 1. Instantiate Core Sub-components
        self.markov_blanket = MarkovBlanketIO(config)
        self.world_model = WorldModel(config)
        self.perception_engine = PerceptionEngine(config)
        self.planning_engine = PlanningEngine(config, self.world_model)
        
        self.action_engine = ActionEngine(
            config=config,
            markov_blanket=self.markov_blanket,
            perception_engine=self.perception_engine,
            world_model=self.world_model,
            planning_engine=self.planning_engine
        )
        
        self.meta_learning = StructureLearning(config)
        
        # 2. Running State Tracking
        self.prev_states: Optional[List[Dict[str, Tensor]]] = None
        self.prev_action_ids: Optional[Tensor] = None
        self.global_surprise = 0.0
        self.step_counter = 0
        
        self.to(device=self.device)

    def reset(self, batch_size: int = 1):
        """Resets the temporal context (beliefs and action memory) for a new episode."""
        self.prev_states = self.world_model.get_initial_states(
            batch_size=batch_size,
            device=self.device,
            dtype=self.dtype
        )
        self.prev_action_ids = None
        self.global_surprise = 0.0
        self.step_counter = 0

    def step(
        self,
        observation_text: Union[str, List[str]],
        target_text: Optional[Union[str, List[str]]] = None
    ) -> Tuple[Tensor, List[str], Dict[str, Any]]:
        """
        Runs a single step of the full coordinated Agent Loop.
        
        Args:
            observation_text: Raw input observation (string or list of strings).
            target_text: Optional target text (string or list of strings) representing the goal.
            
        Returns:
            action_ids: Selected action token IDs [batch_size]
            action_tokens: Decoded action string tokens.
            metrics: Execution metrics and states.
        """
        # Auto-initialize states if step is called without reset
        batch_size = 1 if isinstance(observation_text, str) else len(observation_text)
        if self.prev_states is None:
            self.reset(batch_size=batch_size)
            
        # 1. Adaptive Compute: Adjust perception iterations based on surprise
        # If the last step had high surprise, increase computation budget to think deeper
        default_iter = self.config.perception.n_iterations
        if self.global_surprise > 0.5:
            # High surprise: double iterations up to max cap
            n_iterations = min(default_iter * 2, 24)
            compute_boost = True
        else:
            n_iterations = default_iter
            compute_boost = False
            
        # 2. Delegate to Action Engine for core inference & action loop
        action_ids, action_tokens, current_states, loop_metrics = self.action_engine.step(
            observation_text=observation_text,
            prev_states=self.prev_states,
            prev_action_ids=self.prev_action_ids,
            target_text=target_text
        )
        
        # 3. Update local state history
        self.prev_states = current_states
        self.prev_action_ids = action_ids
        
        # 4. Extract surprise and update Meta-Learning history
        step_vfe = loop_metrics["vfe"]
        step_surprise = loop_metrics["surprise"]
        self.global_surprise = step_surprise
        
        # Sample Level 1 state from current_states to store in meta-history
        # current_states[0]["z"] shape: [batch_size, z_dim]
        self.meta_learning.update_history(step_vfe, current_states[0]["z"])
        
        # 5. Periodically trigger Bayesian Model Reduction / Structure check
        self.step_counter += 1
        meta_metrics = {}
        if self.step_counter % self.config.meta_learning.consolidation_interval == 0:
            meta_metrics = self.meta_learning.apply_structure_change(self.world_model)
            
        # Build comprehensive step metrics
        metrics = {
            "vfe": step_vfe,
            "surprise": step_surprise,
            "n_iterations": n_iterations,
            "compute_boost": compute_boost,
            "loop": loop_metrics,
            "meta": meta_metrics,
            "step": self.step_counter
        }
        
        return action_ids, action_tokens, metrics

    def save_checkpoint(self, path: str) -> Dict[str, Any]:
        """Saves weights and states to target path."""
        state = {
            "model_state": self.state_dict(),
            "config": self.config,
            "step": self.step_counter
        }
        torch.save(state, path)
        return {"saved_path": path, "step": self.step_counter}

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Loads weights and state from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(checkpoint["model_state"])
        self.step_counter = checkpoint["step"]
        return {"loaded_path": path, "step": self.step_counter}
