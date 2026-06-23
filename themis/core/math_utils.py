"""
Themis Core — Mathematical Utilities

Provides numerically stable functions for probability, information theory,
and active inference computations.
All functions support batched tensors and are designed to be safe with FP16/mixed precision.
"""

import math
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor


def logsumexp(x: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    """
    Numerically stable log-sum-exp trick wrapper.
    Avoids overflow/underflow when computing log of sum of exponentials.
    """
    return torch.logsumexp(x, dim=dim, keepdim=keepdim)


def softmax_with_temp(logits: Tensor, temperature: float = 1.0, dim: int = -1) -> Tensor:
    """
    Softmax with temperature scaling.
    If temperature is very low (<= 1e-5), returns a hard one-hot approximation (argmax).
    """
    if temperature <= 1e-5:
        # Hard argmax: create zeros and scatter 1.0 at the argmax index
        idx = torch.argmax(logits, dim=dim, keepdim=True)
        result = torch.zeros_like(logits)
        result.scatter_(dim, idx, 1.0)
        return result
    return F.softmax(logits / temperature, dim=dim)


def precision_weighted_error(prediction: Tensor, target: Tensor, precision: Tensor) -> Tensor:
    """
    Computes the precision-weighted prediction error:
    error = precision * (target - prediction)
    
    This is central to predictive coding (Layer 2) and represents
    how much we update beliefs in response to sensory mismatch.
    """
    return precision * (target - prediction)


def precision_weighted_mse(prediction: Tensor, target: Tensor, precision: Tensor, dim: int = -1) -> Tensor:
    """
    Computes the precision-weighted squared error (summed or averaged over a dimension):
    weighted_mse = sum(precision * (target - prediction)^2)
    
    Acts as the Accuracy term in the Variational Free Energy.
    """
    error_sq = (target - prediction).pow(2)
    return (precision * error_sq).sum(dim=dim)


def kl_categorical(p_logits: Tensor, q_logits: Tensor, dim: int = -1) -> Tensor:
    """
    KL Divergence between two categorical distributions parameterized by logits:
    KL(p || q) = sum( p_probs * (log_p_probs - log_q_probs) )
    """
    p_log_probs = F.log_softmax(p_logits, dim=dim)
    q_log_probs = F.log_softmax(q_logits, dim=dim)
    p_probs = torch.exp(p_log_probs)
    return (p_probs * (p_log_probs - q_log_probs)).sum(dim=dim)


def kl_gaussian(mean_q: Tensor, logvar_q: Tensor, mean_p: Tensor, logvar_p: Tensor, dim: int = -1) -> Tensor:
    """
    KL Divergence between two diagonal Gaussian distributions:
    q ~ N(mean_q, diag(exp(logvar_q)))
    p ~ N(mean_p, diag(exp(logvar_p)))
    
    KL(q || p) = 0.5 * sum( logvar_p - logvar_q + (exp(logvar_q) + (mean_q - mean_p)^2) / exp(logvar_p) - 1 )
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    return 0.5 * (
        logvar_p - logvar_q
        + (var_q + (mean_q - mean_p).pow(2)) / var_p
        - 1.0
    ).sum(dim=dim)


def entropy_categorical(logits: Tensor, dim: int = -1) -> Tensor:
    """
    Entropy of a categorical distribution:
    H = -sum( p * log(p) )
    """
    log_probs = F.log_softmax(logits, dim=dim)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=dim)


def entropy_gaussian(logvar: Tensor, dim: int = -1) -> Tensor:
    """
    Differential entropy of a diagonal Gaussian distribution:
    H = 0.5 * d * (1 + ln(2pi)) + 0.5 * sum( log(var) )
    """
    d = logvar.shape[dim]
    return 0.5 * (d * (1.0 + math.log(2 * math.pi)) + logvar.sum(dim=dim))
