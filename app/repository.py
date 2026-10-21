from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AnswerResult, ClaimResult, Run


class RunRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def save_run(self, run_data: dict[str, Any], outcomes: list[dict[str, Any]]) -> None:
        with self.session_factory() as session:
            with session.begin():
                session.add(Run(**run_data))
                session.flush()
                for outcome in outcomes:
                    session.add(AnswerResult(
                        run_id=run_data["run_id"],
                        answer_id=outcome["answer_id"],
                        status=outcome["status"],
                        error=outcome.get("error"),
                        latency_ms=outcome.get("latency_ms", 0),
                        trace_id=outcome.get("trace_id"),
                    ))
                    for claim in outcome.get("claims", []):
                        session.add(ClaimResult(
                            run_id=run_data["run_id"],
                            answer_id=outcome["answer_id"],
                            claim_id=claim["claim_id"],
                            start_char=claim["start_char"],
                            end_char=claim["end_char"],
                            quote=claim["quote"],
                            label=claim["label"],
                            fact_ids=claim["fact_ids"],
                            reason=claim["reason"],
                            gold_match=claim.get("gold_match"),
                        ))

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.session_factory() as session:
            run = session.get(Run, run_id)
            if run is None:
                return None
            outcomes = session.scalars(
                select(AnswerResult).where(AnswerResult.run_id == run_id).order_by(AnswerResult.answer_id)
            ).all()
            claims = session.scalars(
                select(ClaimResult).where(ClaimResult.run_id == run_id).order_by(
                    ClaimResult.answer_id, ClaimResult.start_char, ClaimResult.end_char
                )
            ).all()
            claims_by_answer: dict[str, list[dict[str, Any]]] = {}
            for claim in claims:
                claims_by_answer.setdefault(claim.answer_id, []).append({
                    "claim_id": claim.claim_id,
                    "start_char": claim.start_char,
                    "end_char": claim.end_char,
                    "quote": claim.quote,
                    "label": claim.label,
                    "fact_ids": claim.fact_ids,
                    "reason": claim.reason,
                    "gold_match": claim.gold_match,
                })
            return {
                "run_id": run.run_id,
                "split": run.split,
                "status": run.status,
                "agent_version": run.agent_version,
                "prompt_version": run.prompt_version,
                "dataset_version": run.dataset_version,
                "evaluator_version": run.evaluator_version,
                "catalogue_version": run.catalogue_version,
                "provider": run.provider,
                "model": run.model,
                "settings": run.settings,
                "dataset_hashes": run.dataset_hashes,
                "source_revision": run.source_revision,
                "started_at": run.started_at,
                "completed_at": run.completed_at,
                "counts": run.counts,
                "metrics": run.metrics,
                "error_count": run.error_count,
                "answers": [
                    {
                        "answer_id": answer.answer_id,
                        "status": answer.status,
                        "error": answer.error,
                        "latency_ms": answer.latency_ms,
                        "trace_id": answer.trace_id,
                        "claims": claims_by_answer.get(answer.answer_id, []),
                    }
                    for answer in outcomes
                ],
            }