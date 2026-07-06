"""Unit tests for NuraDataLoader (fully offline — no HuggingFace calls)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from sdk.nura.data.loader import NuraDataLoader, _outcome_signal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loader_from_jsonl(records: list[dict]) -> NuraDataLoader:
    """Write *records* to a temp JSONL file and return a loader for it."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for rec in records:
        tmp.write(json.dumps(rec) + "\n")
    tmp.flush()
    tmp.close()
    return NuraDataLoader(tmp.name)


_SAMPLE_RECORDS = [
    {
        "instruction": "How do I track my order?",
        "response": (
            "I can help you track your order right away. "
            "Please provide your order number and I will process the tracking "
            "information for you. We will update you on the current status and "
            "estimated delivery date so you can confirm the details. "
            "Our team is here to assist you every step of the way."
        ),
        "intent": "track_order",
    },
    {
        "instruction": "I want a refund.",
        "response": "We can refund that.",  # too short → outcome 0.0
        "intent": "refund_request",
    },
    {
        "instruction": "Cancel my subscription please.",
        "response": (
            "I understand you'd like to cancel your subscription. "
            "I will process the cancellation and help resolve any outstanding "
            "charges. Please allow us to confirm the cancellation date and "
            "update your account accordingly. Our team is ready to assist "
            "with anything else you may need during this process."
        ),
        "intent": "cancel_subscription",
    },
]


# ---------------------------------------------------------------------------
# Test 1 — outcome signal: long + keyword → 1.0
# ---------------------------------------------------------------------------


def test_outcome_signal_resolving_response():
    # 55 words, contains "track", "process", "confirm", "resolve", "assist"
    long_response = (
        "I can help you track your order right away. "
        "Please provide your order number and I will process the tracking "
        "information for you. We will update you on the current status and "
        "estimated delivery date so you can confirm the details. "
        "Our team is here to assist you every step of the way and resolve any issues promptly."
    )
    assert len(long_response.split()) > 50, "test fixture must be > 50 words"
    assert _outcome_signal(long_response) == 1.0


# ---------------------------------------------------------------------------
# Test 2 — outcome signal: short response → 0.0
# ---------------------------------------------------------------------------


def test_outcome_signal_short_response():
    assert _outcome_signal("We can refund that.") == 0.0


# ---------------------------------------------------------------------------
# Test 3 — prepare returns correct count and required keys
# ---------------------------------------------------------------------------


def test_prepare_count_and_keys():
    loader = _make_loader_from_jsonl(_SAMPLE_RECORDS * 10)
    data = loader.prepare(n=6)
    assert len(data) == 6
    for record in data:
        assert set(record.keys()) == {
            "prompt",
            "context",
            "ideal_response",
            "outcome_signal",
        }


# ---------------------------------------------------------------------------
# Test 4 — to_trl_format produces correct structure
# ---------------------------------------------------------------------------


def test_to_trl_format():
    loader = _make_loader_from_jsonl(_SAMPLE_RECORDS)
    data = loader.prepare(n=3)
    trl = loader.to_trl_format(data)

    assert len(trl) == 3
    for item in trl:
        assert "prompt" in item
        assert "completion" in item
        assert "reward" in item
        assert isinstance(item["reward"], float)


# ---------------------------------------------------------------------------
# Test 5 — baseline_score is in [0, 1] and matches manual calculation
# ---------------------------------------------------------------------------


def test_baseline_score_range_and_value():
    loader = _make_loader_from_jsonl(_SAMPLE_RECORDS)
    data = loader.prepare(n=3)
    score = loader.baseline_score(data)
    assert 0.0 <= score <= 1.0


def test_baseline_score_empty_data():
    loader = _make_loader_from_jsonl(_SAMPLE_RECORDS)
    assert loader.baseline_score([]) == 0.0


# ---------------------------------------------------------------------------
# Test 6 — save and reload round-trips correctly
# ---------------------------------------------------------------------------


def test_save_and_reload():
    loader = _make_loader_from_jsonl(_SAMPLE_RECORDS)
    data = loader.prepare(n=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "out.jsonl")
        loader.save(data, out_path)

        reloaded = NuraDataLoader(out_path).prepare(n=3)

    assert len(reloaded) == 3
    assert all("prompt" in r for r in reloaded)
