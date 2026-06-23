"""Resolution reward: uses the raw business outcome as the reward signal."""

from __future__ import annotations

from .base import BaseReward


class ResolutionReward(BaseReward):
    """
    The simplest possible reward function: the real-world outcome **is** the reward.

    Use this when you already have a binary or continuous success signal from
    your business system — for example:

    * ``1.0`` — the customer's issue was resolved on the first response.
    * ``0.0`` — the customer escalated to a human agent.
    * ``0.72`` — a post-chat satisfaction survey score normalised to ``[0, 1]``.

    Because the outcome is passed through unchanged, this reward function adds
    no transformation bias.  It is the recommended starting point for new
    projects; switch to a more complex reward only when you have evidence that
    a different shape would improve training.

    Parameters
    ----------
    clip_min:
        Clip rewards below this value.  Defaults to ``0.0``.
    clip_max:
        Clip rewards above this value.  Defaults to ``1.0``.
    """

    def __init__(self, clip_min: float = 0.0, clip_max: float = 1.0) -> None:
        if clip_min >= clip_max:
            raise ValueError(
                f"clip_min ({clip_min}) must be strictly less than clip_max ({clip_max})."
            )
        self.clip_min = clip_min
        self.clip_max = clip_max

    # ------------------------------------------------------------------
    # BaseReward interface
    # ------------------------------------------------------------------

    def score(
        self,
        prompts: list[str],
        completions: list[str],
        outcomes: list[float],
    ) -> list[float]:
        """
        Return *outcomes* clipped to ``[clip_min, clip_max]`` as the reward batch.

        The *prompts* and *completions* arguments are accepted for API
        compatibility but are not used — the outcome alone carries the signal.

        Parameters
        ----------
        prompts:
            Input prompts (unused by this reward, but required by the interface).
        completions:
            Model responses (unused by this reward, but required by the interface).
        outcomes:
            Real-world success signals, one per (prompt, completion) pair.

        Returns
        -------
        list[float]
            Clipped outcome values in the same order as the inputs.

        Raises
        ------
        ValueError
            If the three lists have different lengths.
        """
        if not (len(prompts) == len(completions) == len(outcomes)):
            raise ValueError(
                f"prompts ({len(prompts)}), completions ({len(completions)}), and "
                f"outcomes ({len(outcomes)}) must all have the same length."
            )

        return [max(self.clip_min, min(self.clip_max, float(o))) for o in outcomes]

    def validate(self) -> bool:
        """
        Confirm the clip bounds are sane.

        Returns
        -------
        bool
            Always ``True`` for a correctly constructed instance (the
            constructor already enforces the invariant).
        """
        return self.clip_min < self.clip_max
