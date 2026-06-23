"""Abstract base class for Nura's AI assistant (the "brain")."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseBrain(ABC):
    """
    The Brain interprets raw data and model results in plain English.

    It acts as an AI-powered explainer layer between the technical training
    pipeline and the non-technical business owner who commissioned the model.
    All output should be jargon-free, specific, and actionable.

    Three responsibilities:

    1. **Dataset analysis** — read a sample of training data and summarise
       what the model will learn from it.
    2. **Reward recommendation** — given a dataset summary, suggest the most
       appropriate reward function for the business goal.
    3. **Result explanation** — after a training run, explain in plain English
       what improved, by how much, and what it means for the business.
    """

    @abstractmethod
    async def analyze_dataset(self, samples: list[dict]) -> str:
        """
        Summarise a dataset sample in plain English.

        Parameters
        ----------
        samples:
            A list of raw data records (e.g. chat transcripts, support
            tickets, form submissions).  Each record is an arbitrary dict
            whose schema depends on the connector that produced it.

        Returns
        -------
        str
            A concise, jargon-free summary of what the data contains and
            what patterns the model is likely to learn from it.
        """

    @abstractmethod
    async def recommend_reward(self, summary: str) -> str:
        """
        Recommend a reward function based on a dataset summary.

        Parameters
        ----------
        summary:
            Plain-English description of the dataset, typically produced
            by :meth:`analyze_dataset`.

        Returns
        -------
        str
            A recommendation (e.g. ``"ResolutionReward"``) with a short
            explanation of why it fits this use-case, written so a
            non-technical reader can act on it.
        """

    @abstractmethod
    async def explain_result(
        self,
        before_score: float,
        after_score: float,
        metrics: dict,
    ) -> str:
        """
        Explain a completed training run in plain English.

        Parameters
        ----------
        before_score:
            The model's average reward *before* training (baseline).
        after_score:
            The model's average reward *after* training.
        metrics:
            Additional metrics from the training run, e.g.
            ``{"episodes": 500, "avg_tokens": 128, "reward_std": 0.12}``.

        Returns
        -------
        str
            A plain-English narrative: what changed, by how much, and what
            that means for the business.  Should include a concrete
            real-world implication (e.g. "roughly X more issues resolved
            per 100 conversations").
        """
