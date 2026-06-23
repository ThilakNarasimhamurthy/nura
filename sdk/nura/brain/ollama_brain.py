"""Ollama-backed Brain: explains AI training in plain English."""

from __future__ import annotations

import json

from ..adapters.ollama_adapter import OllamaAdapter
from .base import BaseBrain

_SYSTEM_PROMPT = (
    "You are Nura's AI assistant. "
    "You explain AI training results in plain English to business owners "
    "who are not technical. No jargon. Be specific and helpful."
)

_DEFAULT_MODEL = "llama3.2:1b"


class OllamaBrain(BaseBrain):
    """
    Brain implementation that uses a local Ollama model to generate explanations.

    All three methods build a focused prompt, prepend the system instruction,
    and call the underlying :class:`~nura.adapters.ollama_adapter.OllamaAdapter`
    to produce a plain-English response.

    Parameters
    ----------
    model:
        Ollama model to use for explanations.  Defaults to ``"llama3.2:1b"``.
    temperature:
        Sampling temperature.  Lower values produce more consistent wording;
        higher values produce more varied phrasing.  Defaults to ``0.3``.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        temperature: float = 0.3,
    ) -> None:
        self._adapter = OllamaAdapter(model=model, temperature=temperature)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    async def _ask(self, prompt: str) -> str:
        """Prepend the system prompt and call the adapter."""
        return await self._adapter.generate(
            prompt=prompt,
            context=_SYSTEM_PROMPT,
        )

    # ------------------------------------------------------------------
    # BaseBrain interface
    # ------------------------------------------------------------------

    async def analyze_dataset(self, samples: list[dict]) -> str:
        """
        Summarise a batch of raw data records in plain English.

        The first 10 samples are serialised to JSON and embedded in the prompt
        so the model can identify patterns without seeing the full dataset.

        Parameters
        ----------
        samples:
            List of raw data records from a connector.

        Returns
        -------
        str
            A plain-English summary of the dataset.
        """
        preview = samples[:10]
        preview_text = json.dumps(preview, indent=2, default=str)
        total = len(samples)

        prompt = (
            f"I have a dataset with {total} records that I want to use to train "
            f"an AI assistant for my business.\n\n"
            f"Here are the first {len(preview)} records:\n"
            f"```json\n{preview_text}\n```\n\n"
            "Please summarise:\n"
            "1. What kind of data this is (in one sentence).\n"
            "2. What the AI will learn to do from this data.\n"
            "3. Any obvious gaps or quality issues I should know about.\n\n"
            "Keep the answer under 150 words and avoid technical terms."
        )

        return await self._ask(prompt)

    async def recommend_reward(self, summary: str) -> str:
        """
        Suggest the best reward function for a given dataset.

        Parameters
        ----------
        summary:
            Plain-English dataset summary (from :meth:`analyze_dataset`).

        Returns
        -------
        str
            Reward function recommendation with a plain-English rationale.
        """
        prompt = (
            "I'm setting up an AI training system for my business and I need "
            "to choose how to measure success.\n\n"
            f"Here is a summary of my data:\n{summary}\n\n"
            "The available reward options are:\n"
            "- ResolutionReward: use this when you have a clear yes/no outcome "
            "(e.g. issue resolved, sale made, question answered correctly).\n"
            "- Custom: tell me to use a custom approach only if none of the above fit.\n\n"
            "Which reward should I use, and why? Explain it as if I have never "
            "heard of AI training before. Keep it under 100 words."
        )

        return await self._ask(prompt)

    async def explain_result(
        self,
        before_score: float,
        after_score: float,
        metrics: dict,
    ) -> str:
        """
        Turn raw training metrics into a plain-English business narrative.

        Parameters
        ----------
        before_score:
            Average reward before training (baseline).
        after_score:
            Average reward after training.
        metrics:
            Additional training statistics (episodes, token counts, etc.).

        Returns
        -------
        str
            A plain-English explanation of what changed and why it matters.
        """
        improvement_pct = (
            ((after_score - before_score) / before_score * 100)
            if before_score > 0
            else 0.0
        )
        direction = "improved" if after_score >= before_score else "declined"
        metrics_text = json.dumps(metrics, indent=2, default=str)

        prompt = (
            f"My AI assistant just finished a training session. Here are the results:\n\n"
            f"- Before training: {before_score:.2%} success rate\n"
            f"- After training:  {after_score:.2%} success rate\n"
            f"- Change: {direction} by {abs(improvement_pct):.1f}%\n"
            f"- Other stats: {metrics_text}\n\n"
            "Please explain:\n"
            "1. What this result means for my business in one sentence.\n"
            "2. A concrete example of what this improvement looks like in real life "
            "(e.g. 'For every 100 customer questions, your AI will now correctly "
            "handle X more than before').\n"
            "3. Whether I should be happy with this result or if more training is needed.\n\n"
            "No jargon. Under 150 words."
        )

        return await self._ask(prompt)
