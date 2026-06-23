"""
Themis Layers — Layer 1: Markov Blanket I/O

Defines the sensory-motor boundary for the Themis agent.
- Inputs (Sensory): Encodes text strings into distributional latent representations.
- Outputs (Active): Decodes latent action states into categorical distributions over tokens.
"""

from typing import Union, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from themis.config import ThemisConfig
from themis.core.distributions import GaussianDist, CategoricalDist
from themis.encoders.text_encoder import TextEncoder, SimpleTokenizer


class MarkovBlanketIO(nn.Module):
    """
    Layer 1: Markov Blanket I/O

    Acts as the statistical boundary separating the internal states of the agent
    from the external environment.
    """
    def __init__(self, config: ThemisConfig):
        super().__init__()
        self.config = config
        dims = config.dims

        # Sensory Encoder (Text Comprehension)
        self.encoder = TextEncoder(config)
        self.tokenizer: SimpleTokenizer = self.encoder.tokenizer

        # Active Decoder (Action / Text Generation)
        # Projects action latent vectors (action_dim) back to vocabulary logits
        self.decoder_head = nn.Sequential(
            nn.Linear(dims.action_dim, dims.token_embed_dim),
            nn.GELU(),
            nn.LayerNorm(dims.token_embed_dim),
            nn.Linear(dims.token_embed_dim, dims.vocab_size)
        )

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize decoder weights
        for layer in self.decoder_head:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.constant_(layer.bias, 0.0)

    def encode(self, text: str, device: Optional[torch.device] = None) -> GaussianDist:
        """
        Sensory interface: string -> GaussianDist representing q(z_sensory | text).
        """
        if device is None:
            device = self.config.resolve_device()

        # Tokenize and create tensor batch of size 1
        ids = self.tokenizer.encode(text)
        ids_tensor = torch.tensor([ids], dtype=torch.long, device=device)

        # Run text encoder
        sensory_dist = self.encoder(ids_tensor)
        return sensory_dist

    def encode_batch(self, texts: List[str], device: Optional[torch.device] = None) -> GaussianDist:
        """
        Batch sensory interface: list of strings -> batched GaussianDist.
        Handles padding automatically.
        """
        if device is None:
            device = self.config.resolve_device()

        # Tokenize all strings
        token_lists = [self.tokenizer.encode(text) for text in texts]
        max_len = max(len(t) for t in token_lists)
        pad_id = self.tokenizer.pad_id

        # Pad sequences
        padded_tokens = []
        attention_mask = []
        for tokens in token_lists:
            padded = tokens + [pad_id] * (max_len - len(tokens))
            mask = [False] * len(tokens) + [True] * (max_len - len(tokens))
            padded_tokens.append(padded)
            attention_mask.append(mask)

        tokens_tensor = torch.tensor(padded_tokens, dtype=torch.long, device=device)
        mask_tensor = torch.tensor(attention_mask, dtype=torch.bool, device=device)

        return self.encoder(tokens_tensor, attention_mask=mask_tensor)

    def decode(self, action_latent: Tensor) -> CategoricalDist:
        """
        Active interface: action_latent -> CategoricalDist over vocabulary.
        action_latent: Tensor of shape [batch_size, action_dim] or [action_dim]
        """
        # Projects actions to vocabulary logits
        logits = self.decoder_head(action_latent)
        return CategoricalDist(logits=logits)

    def sample_action_token(self, action_latent: Tensor, temperature: float = 1.0) -> Tuple[Tensor, int]:
        """
        Decodes the action latent and samples a token.
        Returns:
            sampled_one_hot: Tensor [batch_size, vocab_size] (differentiable Gumbel-Softmax sample)
            token_id: int (non-differentiable index) for the first element in batch.
        """
        dist = self.decode(action_latent)
        
        # Differentiable sampling
        if temperature != 1.0:
            # Scale logits by temperature before Gumbel-Softmax sampling
            dist_scaled = CategoricalDist(logits=dist.logits / temperature)
            sampled_one_hot = dist_scaled.sample()
        else:
            sampled_one_hot = dist.sample()
            
        token_id = int(dist.sample_index()[0].item())
        return sampled_one_hot, token_id
