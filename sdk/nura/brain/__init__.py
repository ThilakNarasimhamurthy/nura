"""Brain implementations — plain-English explainers for training results."""

from .base import BaseBrain
from .ollama_brain import OllamaBrain

__all__ = ["BaseBrain", "OllamaBrain"]
