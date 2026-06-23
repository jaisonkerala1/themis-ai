"""
Themis Encoders — Text Encoder

Implements a lightweight subword/char tokenizer and a Transformer encoder
that projects text sequences to a diagonal Gaussian latent observation distribution.
"""

import re
import math
from typing import List, Union, Tuple, Optional
import torch
import torch.nn as nn
from torch import Tensor

from themis.config import ThemisConfig
from themis.core.distributions import GaussianDist

# =============================================================================
# Custom Lightweight Subword/Character Tokenizer
# =============================================================================

class SimpleTokenizer:
    """
    A lightweight, zero-dependency subword/character tokenizer.
    Populated with ASCII characters and common English words up to vocab_size.
    Falls back to characters for unknown words, ensuring 100% coverage.
    """
    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        
        # 1. Special tokens
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for token in self.special_tokens:
            self._add_to_vocab(token)
            
        # 2. Add common characters (ASCII printable)
        for i in range(32, 127):
            self._add_to_vocab(chr(i))
        # Add basic whitespace/newlines
        for ws in ["\n", "\t", "\r"]:
            self._add_to_vocab(ws)
            
        # 3. Add a curated list of common English words/syllables to bootstrap subwords
        common_words = [
            "the", "and", "ing", "ion", "ent", "for", "that", "this", "with", "you",
            "have", "not", "but", "they", "are", "was", "him", "his", "her", "she",
            "you", "your", "what", "which", "their", "there", "about", "would",
            "will", "can", "one", "all", "out", "from", "had", "has", "been", "were",
            "who", "more", "some", "time", "than", "other", "into", "only", "them",
            "how", "then", "its", "our", "could", "first", "two", "new", "any",
            "about", "these", "make", "like", "into", "over", "also", "back", "even",
            "only", "your", "when", "here", "work", "well", "down", "may", "should",
            "very", "many", "most", "such", "than", "must", "much", "each", "both",
            "good", "make", "just", "know", "take", "year", "your", "them", "some"
        ]
        for word in common_words:
            self._add_to_vocab(word)
            self._add_to_vocab(" " + word) # Add space-prefixed version
            
        # Regex pattern for splitting text: words, spaces, or single punctuation/characters
        self.split_pattern = re.compile(r"(\w+|[^\w\s]|\s+)")
        
    def _add_to_vocab(self, token: str):
        if token not in self.vocab and len(self.vocab) < self.vocab_size:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token

    @property
    def pad_id(self) -> int: return self.vocab[self.pad_token]
    @property
    def unk_id(self) -> int: return self.vocab[self.unk_token]
    @property
    def bos_id(self) -> int: return self.vocab[self.bos_token]
    @property
    def eos_id(self) -> int: return self.vocab[self.eos_token]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.split_pattern.findall(text)
        ids = []
        if add_special_tokens:
            ids.append(self.bos_id)
            
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                # Fallback: break into characters
                for char in token:
                    if char in self.vocab:
                        ids.append(self.vocab[char])
                    else:
                        ids.append(self.unk_id)
                        
        if add_special_tokens:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Union[List[int], Tensor]) -> str:
        """Decode token IDs back to text."""
        if isinstance(ids, Tensor):
            ids = ids.tolist()
            
        tokens = []
        for idx in ids:
            if idx in self.inverse_vocab:
                token = self.inverse_vocab[idx]
                # Skip special formatting tokens in final decoded text
                if token not in self.special_tokens:
                    tokens.append(token)
            else:
                tokens.append(self.unk_token)
        return "".join(tokens)


# =============================================================================
# Transformer-Based Text Encoder Module
# =============================================================================

class TextEncoder(nn.Module):
    """
    Transformer Encoder that maps tokenized text sequences to
    a Gaussian observation distribution.
    """
    def __init__(self, config: ThemisConfig):
        super().__init__()
        self.config = config
        dims = config.dims
        
        self.tokenizer = SimpleTokenizer(vocab_size=dims.vocab_size)
        
        # Embedding Layers
        self.token_embeddings = nn.Embedding(
            num_embeddings=dims.vocab_size,
            embedding_dim=dims.token_embed_dim,
            padding_idx=self.tokenizer.pad_id
        )
        
        self.positional_embeddings = nn.Parameter(
            torch.zeros(1, dims.max_seq_len, dims.token_embed_dim)
        )
        
        # Transformer Layers (scaled with embedding dim for larger models)
        n_heads = 8 if dims.token_embed_dim % 8 == 0 else 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dims.token_embed_dim,
            nhead=n_heads,
            dim_feedforward=dims.token_embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=8
        )
        
        # Projection head to mean and log-variance of observation embedding
        self.projection = nn.Linear(dims.token_embed_dim, dims.obs_embed_dim * 2)
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize positional embeddings with sine/cosine or small normal
        nn.init.normal_(self.positional_embeddings, std=0.02)
        # Initialize projection weights to avoid large initial log_var
        nn.init.normal_(self.projection.weight, std=0.02)
        nn.init.constant_(self.projection.bias, 0.0)
        # Fill the second half of projection bias (corresponding to log_var) with negative values
        # so initial log_var starts small (e.g. -2.0, std = e^-1 = 0.36)
        with torch.no_grad():
            self.projection.bias[self.config.dims.obs_embed_dim:].fill_(-2.0)

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> GaussianDist:
        """
        Args:
            input_ids: [batch_size, seq_len] tensor of token IDs.
            attention_mask: [batch_size, seq_len] boolean tensor where True means pad/mask out.
        Returns:
            dist: GaussianDist representing q(z_sensory | text) of shape [batch_size, obs_embed_dim]
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. Embed tokens & add positional encoding
        x = self.token_embeddings(input_ids) # [batch_size, seq_len, token_embed_dim]
        pos = self.positional_embeddings[:, :seq_len, :]
        x = x + pos
        
        # 2. Pass through Transformer Encoder
        # PyTorch Transformer expects src_key_padding_mask [batch, seq_len] where True elements are padded out
        if attention_mask is None:
            # Mask out PAD tokens automatically
            attention_mask = (input_ids == self.tokenizer.pad_id)
            
        # If all items are padded (empty sequence edge case), avoid NaN
        if attention_mask.all():
            attention_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
            
        feats = self.transformer(x, src_key_padding_mask=attention_mask) # [batch_size, seq_len, token_embed_dim]
        
        # 3. Pooling: Mean pooling over non-masked tokens
        # Create mask multiplier
        mask_mult = (~attention_mask).unsqueeze(-1).to(feats.dtype) # [batch_size, seq_len, 1]
        sum_feats = (feats * mask_mult).sum(dim=1) # [batch_size, token_embed_dim]
        lengths = mask_mult.sum(dim=1).clamp(min=1.0) # [batch_size, 1]
        pooled = sum_feats / lengths # [batch_size, token_embed_dim]
        
        # 4. Project to Gaussian distribution parameters
        proj = self.projection(pooled) # [batch_size, obs_embed_dim * 2]
        
        return GaussianDist.from_params(proj)
