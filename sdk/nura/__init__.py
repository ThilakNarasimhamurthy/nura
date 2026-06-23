"""
Nura SDK — open-source reinforcement learning for business AI.

Connect your data, train a custom model, and get a managed API that
improves automatically from real-world outcomes.

Quick start
-----------
>>> from nura import OllamaAdapter, ResolutionReward, OllamaBrain
>>> adapter = OllamaAdapter()
>>> reward  = ResolutionReward()
>>> brain   = OllamaBrain()
"""

from __future__ import annotations

from .adapters.base import BaseLLMAdapter
from .adapters.ollama_adapter import OllamaAdapter
from .brain.base import BaseBrain
from .brain.ollama_brain import OllamaBrain
from .rewards.base import BaseReward
from .rewards.resolution import ResolutionReward

__version__ = "0.1.0"
__author__ = "Nura Contributors"
__license__ = "MIT"

__all__ = [
    # Adapters
    "BaseLLMAdapter",
    "OllamaAdapter",
    # Rewards
    "BaseReward",
    "ResolutionReward",
    # Brain
    "BaseBrain",
    "OllamaBrain",
]
