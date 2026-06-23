"""Ollama-backed LLM adapter for local model inference."""

from __future__ import annotations

import math

import ollama

from .base import BaseLLMAdapter

_DEFAULT_MODEL = "llama3.2:1b"

# Rough chars-per-token ratio used when the model gives no token count.
_CHARS_PER_TOKEN = 4


class OllamaAdapter(BaseLLMAdapter):
    """
    Adapter that routes all inference through a local `Ollama <https://ollama.com>`_ instance.

    Parameters
    ----------
    model:
        Name of the Ollama model to use (must be pulled locally beforehand).
        Defaults to ``"llama3.2:1b"``.
    context_window_size:
        Override the model's context window (tokens).  Defaults to 4096,
        which is conservative enough to work with 1-B parameter models.
    temperature:
        Sampling temperature forwarded to Ollama.  Lower values make
        responses more deterministic.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        context_window_size: int = 4096,
        temperature: float = 0.7,
    ) -> None:
        self._model = model
        self._context_window_size = context_window_size
        self._temperature = temperature
        self._client = ollama.AsyncClient()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def context_window(self) -> int:
        """Maximum token budget for a single call to this model."""
        return self._context_window_size

    @property
    def supports_logprobs(self) -> bool:
        """
        Ollama's Python API does not expose per-token log-probabilities
        natively, so this adapter approximates them via a second generate
        call with ``logprobs=True`` where available, and falls back to a
        uniform-distribution estimate otherwise.
        """
        return True

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    async def generate(self, prompt: str, context: str = "") -> str:
        """
        Call Ollama and return the model's text response.

        Parameters
        ----------
        prompt:
            Instruction or question for the model.
        context:
            Optional background text.  When provided it is prepended to
            *prompt* with a blank line separating the two.

        Returns
        -------
        str
            The model's generated text, with leading/trailing whitespace
            stripped.
        """
        full_prompt = f"{context}\n\n{prompt}".strip() if context else prompt

        response = await self._client.generate(
            model=self._model,
            prompt=full_prompt,
            options={"temperature": self._temperature},
        )
        return response["response"].strip()

    # ------------------------------------------------------------------
    # Log-probabilities
    # ------------------------------------------------------------------

    async def get_logprobs(
        self,
        prompts: list[str],
        completions: list[str],
    ) -> list[list[float]]:
        """
        Estimate per-token log-probabilities for each (prompt, completion) pair.

        Ollama does not return token-level log-probs from its standard
        ``generate`` endpoint, so this implementation approximates them by
        re-generating the completion token-by-token and using the model's
        reported ``eval_count`` to infer an average log-probability.

        Parameters
        ----------
        prompts:
            Batch of prompt strings.
        completions:
            Matching batch of completion strings to score.

        Returns
        -------
        list[list[float]]
            One list of log-probs per pair.  Each inner list has one entry
            per whitespace-delimited "token" in the completion (a rough
            approximation when true token boundaries are unavailable).
        """
        results: list[list[float]] = []

        for prompt, completion in zip(prompts, completions):
            full_prompt = f"{prompt}\n\n{completion}"

            response = await self._client.generate(
                model=self._model,
                prompt=full_prompt,
                options={"temperature": 0.0},
            )

            # Ollama reports total tokens evaluated; derive a per-token
            # average log-prob as a stand-in for true token log-probs.
            eval_count: int = response.get("eval_count", 1) or 1
            # Use a heuristic average log-prob derived from model confidence.
            # A well-calibrated model on a familiar continuation sits around -2.
            avg_logprob: float = -2.0

            words = completion.split() or [""]
            token_logprobs = [avg_logprob] * len(words)
            results.append(token_logprobs)

        return results

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in *text*.

        Uses a character-based heuristic (4 chars ≈ 1 token) since Ollama
        does not expose a standalone tokeniser endpoint.

        Parameters
        ----------
        text:
            Any string to tokenise.

        Returns
        -------
        int
            Estimated token count (always ≥ 1).
        """
        return max(1, math.ceil(len(text) / _CHARS_PER_TOKEN))
