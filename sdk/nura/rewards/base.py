"""Abstract base class for all reward functions in the Nura SDK."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseReward(ABC):
    """
    Contract every reward function must fulfil.

    A reward function maps a batch of (prompt, completion, outcome) triples
    to a scalar reward signal that the RL training loop uses to reinforce or
    discourage the model's behaviour.

    Implement :meth:`score` to define *how* outcomes translate to rewards,
    and :meth:`validate` to assert that the reward function is correctly
    configured before training begins.
    """

    @abstractmethod
    def score(
        self,
        prompts: list[str],
        completions: list[str],
        outcomes: list[float],
    ) -> list[float]:
        """
        Compute a reward scalar for each (prompt, completion, outcome) triple.

        Parameters
        ----------
        prompts:
            The input prompts shown to the model during a rollout.
        completions:
            The model's generated responses, one per prompt.
        outcomes:
            The real-world business outcome for each response.
            Convention: ``1.0`` = success, ``0.0`` = failure.
            Values outside ``[0, 1]`` are allowed for richer signal
            (e.g. customer satisfaction on a 0–5 scale normalised to 0–1).

        Returns
        -------
        list[float]
            One reward scalar per triple.  Higher is better.

        Raises
        ------
        ValueError
            If the three lists have different lengths.
        """

    @abstractmethod
    def validate(self) -> bool:
        """
        Check that this reward function is ready to use.

        Called automatically by the training loop before the first update.
        Raise a descriptive :class:`ValueError` or return ``False`` to
        signal misconfiguration.

        Returns
        -------
        bool
            ``True`` when the reward function is correctly set up.
        """
