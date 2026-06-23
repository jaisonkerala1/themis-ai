"""
Themis Training — Active Inference Trainer

Implements the online and offline variational training loop for Themis.
Trains the World Model (H-RSSM) to minimize Variational Free Energy,
and distills the Planning Engine into the Amortized Policy actor network.
"""

from typing import List, Dict, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator
from themis.core.distributions import GaussianDist


class ActiveInferenceTrainer:
    """
    ActiveInferenceTrainer

    Manages variational optimization for all trainable parameters in the stack.
    Handles gradient accumulation, mixed precision, and norm clipping.
    """
    def __init__(self, config: ThemisConfig, orchestrator: Orchestrator):
        self.config = config
        self.orchestrator = orchestrator
        self.device = config.resolve_device()
        self.dtype = config.resolve_dtype()
        
        # Optionally freeze sensory encoder (prevents representation collapse for
        # the VFE-driven math model). For language we WANT the encoder/transformer
        # to train, so this is configurable.
        if getattr(config.training, "freeze_encoder", True):
            for param in self.orchestrator.markov_blanket.encoder.parameters():
                param.requires_grad = False
            
        # Setup Optimizer over trainable orchestrator parameters
        trainable_params = [p for p in self.orchestrator.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Setup GradScaler for float16 mixed precision
        self.use_amp = config.training.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        
    def train_step(
        self,
        obs_seq: List[List[str]],
        action_seq: Tensor,
        done_seq: Tensor
    ) -> Dict[str, float]:
        """
        Runs a single multi-step BPTT training update on a batch of sequences.
        
        Args:
            obs_seq: List[List[str]] of shape [batch_size, seq_len]
            action_seq: Token action IDs [batch_size, seq_len]
            done_seq: Done flags [batch_size, seq_len]
            
        Returns:
            loss_dict: Metrics dictionary.
        """
        batch_size = len(obs_seq)
        seq_len = len(obs_seq[0])
        
        # 1. Reset temporal state for start of batch training sequence
        states = self.orchestrator.world_model.get_initial_states(
            batch_size=batch_size,
            device=self.device,
            dtype=self.dtype
        )
        
        # Initialize running losses
        total_vfe = 0.0
        total_complexity = 0.0
        total_accuracy = 0.0
        total_policy_loss = 0.0
        
        # We run the forward step-by-step to compute sequential losses
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            for t in range(seq_len):
                # Get observation text slice for current time step
                obs_t = [obs_seq[b][t] for b in range(batch_size)]
                
                # Get previous actions: actions at t-1 (if t > 0)
                prev_action_ids = action_seq[:, t-1] if t > 0 else None
                
                # A. Prior transition dynamics (computed WITH gradients to train transition weights)
                h_states, priors = self.orchestrator.world_model.compute_priors(states, prev_action_ids)
                
                # B. Encode raw observations
                obs_dist = self.orchestrator.markov_blanket.encode_batch(obs_t, device=self.device)
                
                # C. Inferred beliefs q(z) via Recognition Networks
                # PROFESSIONAL SOLUTION: Amortized single-step inference via trained encoders
                posteriors = self.orchestrator.world_model.recognition(obs_dist.mean, h_states, priors)
                
                # Get element-wise mask for the current time step
                mask = (1.0 - (done_seq[:, t-1] if t > 0 else torch.zeros(batch_size, device=self.device)))

                # D. Variational Free Energy Loss calculation
                # Complexity KL: pulls priors p(z^i_t | h^i_t) towards posteriors q(z^i_t)
                kl_terms = []
                for i in range(3):
                    # We detach posteriors to train priors to predict posteriors
                    kl = posteriors[i].detach().kl_divergence(priors[i])
                    # NUMERICAL STABILITY: Clamp KL to prevent explosion
                    kl = torch.clamp(kl, min=0.0, max=100.0)
                    kl_masked = (kl * mask).mean()
                    kl_terms.append(kl_masked)
                step_complexity = torch.stack(kl_terms).sum()
                # NUMERICAL STABILITY: Prevent complexity from exploding
                step_complexity = torch.clamp(step_complexity, min=-100.0, max=100.0)
                
                # Accuracy: Likelihood reconstruction of observations from posteriors
                # Sample from Level 1 posterior: z^1_t ~ q(z^1_t)
                z1_sample = posteriors[0].sample() # propagates grads to posteriors
                pred_obs = self.orchestrator.world_model.likelihood_decoder(z1_sample, h_states[0])
                log_prob = pred_obs.log_prob_total(obs_dist.mean)
                # NUMERICAL STABILITY: Clamp log probability
                log_prob = torch.clamp(log_prob, min=-100.0, max=100.0)
                step_accuracy = (log_prob * mask).mean()
                
                # Step VFE
                step_vfe = step_complexity - step_accuracy
                # NUMERICAL STABILITY: Clamp VFE
                step_vfe = torch.clamp(step_vfe, min=-100.0, max=100.0)
                
                # E. Amortized Policy Distillation Loss
                # Train Amortized Policy to output selected actions action_seq[:, t]
                # END-TO-END FIX: Do NOT detach inputs. The policy's supervised signal
                # must flow back through the posterior to train the recognition network
                # to actually encode the observation. Detaching here caused the policy to
                # receive observation-independent input → same answer for every question.
                policy_dist = self.orchestrator.planning_engine.amortized_policy(
                    posteriors[0].sample(),
                    h_states[0]
                )
                
                # CrossEntropy loss (element-wise masked before mean)
                policy_log_prob = policy_dist.log_prob_index(action_seq[:, t])
                # NUMERICAL STABILITY: Clamp policy loss
                policy_log_prob = torch.clamp(policy_log_prob, min=-100.0, max=0.0)
                step_policy_loss = (- policy_log_prob * mask).mean()
                
                # F. Accumulate losses
                total_vfe += step_vfe
                total_complexity += step_complexity
                total_accuracy += step_accuracy
                total_policy_loss += step_policy_loss
                
                # G. Transition states for next time step
                states = self.orchestrator.world_model.sample_posteriors(h_states, posteriors)
                # Keep GRU states h connected for BPTT, detach stochastic z to block variance
                states = [
                    {"h": s["h"], "z": s["z"].detach()}
                    for s in states
                ]
                
            # Combine total training loss (weighted so language signal isn't drowned)
            loss = (self.config.training.vfe_weight * total_vfe
                    + self.config.training.policy_loss_weight * total_policy_loss)
            
        # 2. Optimization step (with AMP and Gradient Scaling)
        self.optimizer.zero_grad()
        
        self.scaler.scale(loss).backward()
        
        # Unscale gradients before clipping
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.orchestrator.parameters(), self.config.training.max_grad_norm)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return {
            "loss": float(loss.item()),
            "vfe": float(total_vfe.item()) / seq_len,
            "complexity": float(total_complexity.item()) / seq_len,
            "accuracy": float(total_accuracy.item()) / seq_len,
            "policy_loss": float(total_policy_loss.item()) / seq_len
        }
