from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, ForeignKeyConstraint, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Run(Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    split: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    agent_version: Mapped[str] = mapped_column(String(10), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(20), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(80), nullable=False)
    evaluator_version: Mapped[str] = mapped_column(String(80), nullable=False)
    catalogue_version: Mapped[str] = mapped_column(String(80), nullable=False)
    provider: Mapped[str] = mapped_column(String(40), nullable=False)
    model: Mapped[str] = mapped_column(String(120), nullable=False)
    settings: Mapped[dict] = mapped_column(JSON, nullable=False)
    dataset_hashes: Mapped[dict] = mapped_column(JSON, nullable=False)
    source_revision: Mapped[str | None] = mapped_column(String(80))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    counts: Mapped[dict | None] = mapped_column(JSON)
    metrics: Mapped[dict | None] = mapped_column(JSON)
    error_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class AnswerResult(Base):
    __tablename__ = "answer_results"
    __table_args__ = (UniqueConstraint("run_id", "answer_id", name="uq_answer_run_answer"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id", ondelete="CASCADE"), nullable=False)
    answer_id: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    error: Mapped[dict | None] = mapped_column(JSON)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    trace_id: Mapped[str | None] = mapped_column(String(80))


class ClaimResult(Base):
    __tablename__ = "claim_results"
    __table_args__ = (
        UniqueConstraint("run_id", "answer_id", "start_char", "end_char", name="uq_claim_run_answer_span"),
        ForeignKeyConstraint(
            ["run_id", "answer_id"],
            ["answer_results.run_id", "answer_results.answer_id"],
            ondelete="CASCADE",
            name="fk_claim_answer_result",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False)
    answer_id: Mapped[str] = mapped_column(String(40), nullable=False)
    claim_id: Mapped[str] = mapped_column(String(120), nullable=False)
    start_char: Mapped[int] = mapped_column(Integer, nullable=False)
    end_char: Mapped[int] = mapped_column(Integer, nullable=False)
    quote: Mapped[str] = mapped_column(String, nullable=False)
    label: Mapped[str] = mapped_column(String(20), nullable=False)
    fact_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    reason: Mapped[str] = mapped_column(String, nullable=False)
    gold_match: Mapped[dict | None] = mapped_column(JSON)