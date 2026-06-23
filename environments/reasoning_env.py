"""
Themis Environments — Reasoning Puzzle Environment

Exposes logic and pattern-reasoning puzzles for the agent to solve.
Puzzles require the agent to generate reasoning tokens to complete the pattern.
"""

from typing import Tuple, Dict, Any, List, Optional
from environments.text_env import TextCompletionEnv


class ReasoningEnv(TextCompletionEnv):
    """
    ReasoningEnv

    Extends TextCompletionEnv with a default corpus of reasoning, logic,
    and sequence completion puzzles.
    """
    def __init__(self, dataset: Optional[List[Tuple[str, str]]] = None):
        if dataset is None:
            dataset = [
                ("If A = 1 and B = 2, then A + B = ", "3"),
                ("Write the next number in sequence: 2, 4, 6, 8, ", "10"),
                ("The antonym of hot is ", "cold"),
                ("Complete the logic: sky is blue, grass is ", "green"),
                ("Write the next letters in sequence: A, C, E, G, ", "I"),
                ("If count of X in XXYXX is ", "4"),
                ("Capital of France is ", "Paris")
            ]
        super().__init__(dataset=dataset)
