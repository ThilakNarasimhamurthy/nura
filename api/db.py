"""SQLAlchemy SQLite database setup for the Nura API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Generator

from sqlalchemy import DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

DATABASE_URL = "sqlite:///./nura.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # required for SQLite + FastAPI
)


class Base(DeclarativeBase):
    pass


class OutcomeRecord(Base):
    """
    Stores every prompt/response pair and its optional real-world outcome.

    ``outcome_signal`` is ``None`` until the client POSTs to ``/v1/outcomes``
    with the actual business result (e.g. whether the issue was resolved).
    """

    __tablename__ = "outcome_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prompt: Mapped[str] = mapped_column(String, nullable=False)
    response: Mapped[str] = mapped_column(String, nullable=False)
    outcome_signal: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class RetrainHistory(Base):
    """
    One record per automated retraining run.

    ``status`` transitions: ``"running"`` → ``"complete"`` | ``"failed"``.
    """

    __tablename__ = "retrain_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    triggered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    outcomes_at_trigger: Mapped[int] = mapped_column(Integer, nullable=False)
    before_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    after_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    improvement_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(
        String, nullable=False, default="running"
    )  # "running" | "complete" | "failed"


class AppSettings(Base):
    """
    Simple key-value settings table so state survives server restarts.

    Currently used to persist ``next_retrain_at``.
    """

    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(String, nullable=False)


def init_db() -> None:
    """Create all tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a SQLAlchemy session.

    The session is committed on success and rolled back + closed on any
    exception, ensuring every request gets a clean transaction.
    """
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
