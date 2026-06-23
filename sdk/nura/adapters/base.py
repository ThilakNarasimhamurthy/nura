"""Abstract base class for all LLM adapters in the Nura SDK."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMAdapter(ABC):
    """
    Contract every LLM backend must fulfil.

    Concrete subclasses wrap a specific model provider (Ollama, OpenAI, etc.)
    and expose a uniform interface so the rest of the SDK stays provider-agnostic.
    """

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    @abstractmethod
    async def generate(self, prompt: str, context: str = "") -> str:
        """
        Generate a text completion for *prompt*.

        Parameters
        ----------
        prompt:
            The instruction or question sent to the model.
        context:
            Optional background text prepended to the prompt (e.g. a
            customer conversation history).

        Returns
        -------
        str
            The model's raw text response.
        """

    # ------------------------------------------------------------------
    # Log-probabilities (needed for RL training)
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_logprobs(
        self,
        prompts: list[str],
        completions: list[str],
    ) -> list[list[float]]:
        """
        Return per-token log-probabilities for each (prompt, completion) pair.

        Parameters
        ----------
        prompts:
            A batch of prompt strings.
        completions:
            A matching batch of completion strings whose tokens are scored.

        Returns
        -------
        list[list[float]]
            Outer list has one entry per pair; inner list has one log-prob
            per token in the corresponding completion.
        """

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Return the number of tokens the model would use to represent *text*.

        Used by the training loop to avoid exceeding the context window.

        Parameters
        ----------
        text:
            Any string to tokenise.

        Returns
        -------
        int
            Approximate token count.
        """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Maximum number of tokens this model accepts in a single call."""

    @property
    @abstractmethod
    def supports_logprobs(self) -> bool:
        """
        Whether this adapter can return log-probabilities.

        When ``False``, RL algorithms that require log-probs will fall back
        to a compatible alternative or raise a clear error.
        """
