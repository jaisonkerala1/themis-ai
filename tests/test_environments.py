"""
Tests for Phase 3 environments (TextCompletionEnv and ReasoningEnv).
"""

import pytest
import torch

from environments.text_env import TextCompletionEnv
from environments.reasoning_env import ReasoningEnv
from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator
from themis.core.distributions import GaussianDist


def test_text_completion_env():
    env = TextCompletionEnv()
    
    # 1. Reset
    prefix = env.reset()
    assert isinstance(prefix, str)
    assert len(prefix) > 0
    assert env.current_target in [t[1] for t in env.dataset]
    
    # 2. Step with correct overlap
    first_char_of_target = env.current_target[0]
    next_obs, reward, done, info = env.step(first_char_of_target)
    
    assert isinstance(next_obs, str)
    assert next_obs.startswith(prefix)
    assert reward >= -1.0
    assert not done or reward == 10.0 # unless target is 1 char
    assert info["generated"] == first_char_of_target
    
    # 3. Preference check
    pref_text = env.get_preference()
    assert isinstance(pref_text, str)
    assert pref_text.startswith(prefix)


def test_reasoning_env():
    env = ReasoningEnv()
    prefix = env.reset()
    assert isinstance(prefix, str)
    
    # Step with incorrect input to trigger done/fail path
    next_obs, reward, done, info = env.step("INVALID_TOKEN_ABC_123")
    
    assert done
    assert reward == -1.0
    assert not info["matched"]


def test_preference_with_encoder_projection():
    config = ThemisConfig()
    # Mock CPU/standard normal config for test speed
    config.perception.n_iterations = 2
    orchestrator = Orchestrator(config)
    
    env = TextCompletionEnv()
    env.reset()
    
    # Define a helper mapping target string to GaussianDist
    def encode_text_helper(text: str) -> GaussianDist:
        return orchestrator.markov_blanket.encode(text)
        
    pref_dist = env.get_preference(encoder=encode_text_helper)
    
    assert isinstance(pref_dist, GaussianDist)
    assert pref_dist.mean.shape == (1, config.dims.obs_embed_dim)
    assert pref_dist.log_var.shape == (1, config.dims.obs_embed_dim)
