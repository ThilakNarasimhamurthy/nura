"""Background retraining logic for the Nura API."""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import AppSettings, OutcomeRecord, RetrainHistory, engine

logger = logging.getLogger(__name__)

# Prevents two retraining jobs from running at the same time.
retrain_lock = threading.Lock()

_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_ADAPTERS_DIR = Path("./adapters")
_LATEST_SYMLINK = _ADAPTERS_DIR / "latest"

# ------------------------------------------------------------------
# Settings helpers (next_retrain_at persisted across restarts)
# ------------------------------------------------------------------

_RETRAIN_INTERVAL = 500  # outcomes between automatic retrains


def get_next_retrain_at(db: Session) -> int:
    """
    Return the outcomes-count threshold for the next retrain.

    Reads from ``AppSettings`` so the value survives server restarts.
    Defaults to ``_RETRAIN_INTERVAL`` if not yet set.
    """
    row = db.get(AppSettings, "next_retrain_at")
    return int(row.value) if row else _RETRAIN_INTERVAL


def bump_next_retrain_at(db: Session, current_count: int) -> int:
    """
    Advance ``next_retrain_at`` by one interval and persist it.

    Returns the new threshold.
    """
    new_threshold = current_count + _RETRAIN_INTERVAL
    row = db.get(AppSettings, "next_retrain_at")
    if row:
        row.value = str(new_threshold)
    else:
        db.add(AppSettings(key="next_retrain_at", value=str(new_threshold)))
    return new_threshold


# ------------------------------------------------------------------
# Symlink management
# ------------------------------------------------------------------


def _update_latest_symlink(adapter_path: str) -> None:
    """
    Atomically update ``./adapters/latest`` to point at *adapter_path*.

    Uses a temp-then-replace pattern so the symlink is never in a broken
    intermediate state.
    """
    _ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    resolved = str(Path(adapter_path).resolve())
    tmp = _ADAPTERS_DIR / f"latest_tmp_{os.getpid()}"

    try:
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        tmp.symlink_to(resolved)
        # os.replace is atomic on POSIX for symlinks
        os.replace(tmp, _LATEST_SYMLINK)
        logger.info("Updated latest adapter symlink → %s", resolved)
    except Exception:
        logger.exception("Failed to update latest adapter symlink")
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink(missing_ok=True)


# ------------------------------------------------------------------
# Core retrain function
# ------------------------------------------------------------------


def run_retrain(_db_hint: Session | None = None) -> None:
    """
    Pull the latest 500 labelled outcomes and fine-tune the model.

    Creates its own SQLAlchemy session — the ``_db_hint`` parameter is
    accepted for call-site compatibility but a fresh session is always
    opened inside the thread to avoid cross-thread SQLAlchemy issues.

    Steps
    -----
    1. Acquire ``retrain_lock`` (non-blocking — exits if already held).
    2. Insert a ``RetrainHistory`` row with ``status="running"``.
    3. Pull the last 500 ``OutcomeRecord`` rows that have a labelled signal.
    4. Convert to ``NuraTrainer``-compatible dicts.
    5. Run ``NuraTrainer.train()``.
    6. Persist before/after scores and update status to ``"complete"``.
    7. Update the ``./adapters/latest`` symlink.
    8. On any exception: set status to ``"failed"`` and log the error.
    """
    if not retrain_lock.acquire(blocking=False):
        logger.info("Retrain already in progress — skipping.")
        return

    db = Session(engine)
    history_id: int | None = None

    try:
        # ── Count labelled outcomes ────────────────────────────────────
        from sqlalchemy import func

        outcomes_count: int = db.execute(
            select(func.count(OutcomeRecord.id)).where(
                OutcomeRecord.outcome_signal.is_not(None)
            )
        ).scalar_one()

        # ── Insert RetrainHistory row ─────────────────────────────────
        history = RetrainHistory(
            triggered_at=datetime.now(timezone.utc),
            outcomes_at_trigger=outcomes_count,
            status="running",
        )
        db.add(history)
        db.commit()
        db.refresh(history)
        history_id = history.id
        logger.info("Retrain job #%d started (%d outcomes)", history_id, outcomes_count)

        # ── Pull last 500 labelled records ────────────────────────────
        records = (
            db.execute(
                select(OutcomeRecord)
                .where(OutcomeRecord.outcome_signal.is_not(None))
                .order_by(OutcomeRecord.timestamp.desc())
                .limit(500)
            )
            .scalars()
            .all()
        )

        if not records:
            raise ValueError("No labelled outcomes found — cannot retrain.")

        # ── Convert to NuraTrainer format ─────────────────────────────
        training_data = [
            {
                "prompt": r.prompt,
                "context": "You are a helpful customer support agent.",
                "ideal_response": r.response,
                "outcome_signal": float(r.outcome_signal),  # type: ignore[arg-type]
            }
            for r in records
        ]

        # ── Run training ──────────────────────────────────────────────
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from sdk.nura.rewards.resolution import ResolutionReward
        from sdk.nura.training.trainer import NuraTrainer

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = str(_ADAPTERS_DIR / f"retrain_{timestamp}")

        trainer = NuraTrainer(
            {
                "base_model": _BASE_MODEL,
                "output_dir": output_dir,
                "num_steps": 50,
                "batch_size": 2,
            }
        )

        metrics = trainer.train(training_data, ResolutionReward())
        logger.info("Retrain #%d complete: %s", history_id, metrics)

        # ── Update RetrainHistory ─────────────────────────────────────
        db.refresh(history)
        history.before_score = metrics["before_score"]
        history.after_score = metrics["after_score"]
        history.improvement_pct = metrics["improvement_pct"]
        history.status = "complete"
        db.commit()

        # ── Update latest symlink ─────────────────────────────────────
        _update_latest_symlink(metrics["adapter_path"])

    except Exception as exc:
        logger.exception("Retrain job #%s failed: %s", history_id, exc)
        if history_id is not None:
            try:
                history = db.get(RetrainHistory, history_id)
                if history:
                    history.status = "failed"
                    db.commit()
            except Exception:
                logger.exception("Could not update RetrainHistory status to failed")
    finally:
        db.close()
        retrain_lock.release()


def trigger_retrain(db: Session) -> None:
    """
    Start a background retrain thread if no run is already in progress.

    Parameters
    ----------
    db:
        The current request's DB session (used only to bump the threshold;
        the actual training thread opens its own session).
    """
    thread = threading.Thread(target=run_retrain, daemon=True)
    thread.start()
    logger.info("Background retrain thread started.")
