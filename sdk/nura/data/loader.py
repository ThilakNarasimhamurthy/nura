"""NuraDataLoader — loads, prepares, and exports training data."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

_CONTEXT = "You are a helpful customer support agent."

_RESOLUTION_KEYWORDS = {
    "track",
    "refund",
    "cancel",
    "process",
    "resolve",
    "help",
    "update",
    "confirm",
    "assist",
}


def _outcome_signal(response: str) -> float:
    """
    Compute a binary outcome signal for a single response.

    Returns ``1.0`` when the response is substantive (> 50 words) **and**
    contains at least one resolution keyword, ``0.0`` otherwise.

    Parameters
    ----------
    response:
        The agent's reply text.

    Returns
    -------
    float
        ``1.0`` (resolved) or ``0.0`` (not resolved).
    """
    words = response.lower().split()
    long_enough = len(words) > 50
    has_keyword = bool(_RESOLUTION_KEYWORDS & set(words))
    return 1.0 if (long_enough and has_keyword) else 0.0


class NuraDataLoader:
    """
    Load, prepare, and export training data for Nura's RL pipeline.

    Supports three source types:

    * ``"bitext"`` — loads the Bitext customer-support dataset from the
      HuggingFace Hub (cached locally after the first download).
    * A path ending in ``.jsonl`` — loads a newline-delimited JSON file.
    * A path ending in ``.csv`` — loads a CSV file (must have ``instruction``
      and ``response`` columns).

    Parameters
    ----------
    source:
        Either the string ``"bitext"`` or a file path to a ``.jsonl`` /
        ``.csv`` file.

    Examples
    --------
    >>> loader = NuraDataLoader("bitext")
    >>> data = loader.prepare(n=200)
    >>> print(loader.summary(data))
    200 examples · 71% resolution rate · 11 intents
    """

    def __init__(self, source: str) -> None:
        self.source = source
        self._raw: list[dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load raw records from the configured source into ``self._raw``."""
        src = self.source.strip().lower()

        if src == "bitext":
            from datasets import load_dataset  # type: ignore[import]

            ds = load_dataset(
                "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
            )
            self._raw = [dict(row) for row in ds["train"]]

        elif self.source.endswith(".jsonl"):
            path = Path(self.source)
            with path.open(encoding="utf-8") as fh:
                self._raw = [json.loads(line) for line in fh if line.strip()]

        elif self.source.endswith(".csv"):
            import csv

            path = Path(self.source)
            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                self._raw = [dict(row) for row in reader]

        else:
            raise ValueError(
                f"Unknown source {self.source!r}. "
                "Pass 'bitext', a '.jsonl' path, or a '.csv' path."
            )

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def prepare(self, n: int = 200) -> list[dict[str, Any]]:
        """
        Return *n* training samples balanced across intents.

        Each sample is converted to a standard dict with ``prompt``,
        ``context``, ``ideal_response``, and ``outcome_signal`` fields.

        Balancing strategy: samples are grouped by ``intent`` (or
        ``category`` if ``intent`` is absent).  Up to ``ceil(n / k)``
        samples are drawn from each of the *k* groups, then the result is
        truncated to exactly *n* and shuffled.

        Parameters
        ----------
        n:
            Total number of samples to return.

        Returns
        -------
        list[dict]
            Each dict has the keys: ``prompt``, ``context``,
            ``ideal_response``, ``outcome_signal``.

        Raises
        ------
        ValueError
            If the dataset has no records.
        """
        if not self._raw:
            raise ValueError("Dataset is empty — nothing to prepare.")

        # Group by intent (fall back to category, then "unknown")
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in self._raw:
            key = row.get("intent") or row.get("category") or "unknown"
            groups[key].append(row)

        k = len(groups)
        per_group = max(1, -(-n // k))  # ceiling division

        selected: list[dict] = []
        for rows in groups.values():
            sample = random.sample(rows, min(per_group, len(rows)))
            selected.extend(sample)

        random.shuffle(selected)
        selected = selected[:n]

        return [self._convert(row) for row in selected]

    def _convert(self, row: dict[str, Any]) -> dict[str, Any]:
        """Convert a raw dataset row to Nura's standard training format."""
        instruction = str(row.get("instruction") or row.get("prompt") or "")
        response = str(row.get("response") or row.get("completion") or "")

        return {
            "prompt": instruction,
            "context": _CONTEXT,
            "ideal_response": response,
            "outcome_signal": _outcome_signal(response),
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save(self, data: list[dict[str, Any]], path: str) -> None:
        """
        Save *data* as a newline-delimited JSON file.

        Parameters
        ----------
        data:
            List of prepared sample dicts.
        path:
            Destination file path (created along with any missing parent
            directories).
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for record in data:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def to_trl_format(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert prepared samples to the format expected by TRL's reward trainer.

        Each output dict has:

        * ``prompt`` — the customer message.
        * ``completion`` — the ideal agent response.
        * ``reward`` — the outcome signal as a float.

        Parameters
        ----------
        data:
            List of dicts produced by :meth:`prepare`.

        Returns
        -------
        list[dict]
            TRL-compatible records.
        """
        return [
            {
                "prompt": record["prompt"],
                "completion": record["ideal_response"],
                "reward": record["outcome_signal"],
            }
            for record in data
        ]

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def baseline_score(self, data: list[dict[str, Any]]) -> float:
        """
        Compute the average outcome signal across *data*.

        This is Nura's "before" number — the resolution rate the model
        achieves before any RL fine-tuning.

        Parameters
        ----------
        data:
            List of dicts produced by :meth:`prepare`.

        Returns
        -------
        float
            Mean ``outcome_signal`` in ``[0.0, 1.0]``.  Returns ``0.0`` for
            an empty dataset.
        """
        if not data:
            return 0.0
        return sum(r["outcome_signal"] for r in data) / len(data)

    def summary(self, data: list[dict[str, Any]]) -> str:
        """
        Return a one-line human-readable summary of the prepared dataset.

        Format: ``"200 examples · 71% resolution rate · 11 intents"``

        Parameters
        ----------
        data:
            List of dicts produced by :meth:`prepare`.

        Returns
        -------
        str
            Plain-English summary string.
        """
        n = len(data)
        if n == 0:
            return "0 examples · 0% resolution rate · 0 intents"

        rate = round(self.baseline_score(data) * 100)

        # Re-derive intent count from the raw source via the prompt text
        # (intents are not stored in the prepared dict, so we approximate
        # by counting distinct prompts that share the same prefix word).
        unique_prompts = {r["prompt"].split()[0].lower() for r in data if r["prompt"]}
        intent_count = len(unique_prompts)

        return f"{n} examples · {rate}% resolution rate · {intent_count} intents"
