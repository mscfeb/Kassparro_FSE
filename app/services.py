from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from app.agent import AgentError, AgentInput, ClaimAgent
from app.catalogue import BrandCatalogue
from app.config import Settings
from app.data import DATASET_VERSION, EVALUATOR_VERSION, Answer, dataset_hashes, load_answers, load_gold
from app.evaluator import evaluate_split
from app.observability import LangfuseObserver, TraceSession
from app.provider import GeminiProvider
from app.reliability import RequestExecutor
from app.repository import RunRepository

logger = logging.getLogger(__name__)


class AssessmentService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        agent_factory: Callable[[str], ClaimAgent] | None = None,
        repository: RunRepository | None = None,
        observer: LangfuseObserver | None = None,
    ):
        self.settings = settings or Settings.from_environment()
        self.data_dir = Path(self.settings.data_dir)
        self.catalogue = BrandCatalogue(self.data_dir / "brand_facts.csv")
        self.agent_factory = agent_factory
        self.repository = repository
        self.observer = observer or LangfuseObserver(
            self.settings.langfuse_public_key,
            self.settings.langfuse_secret_key,
            self.settings.langfuse_host,
        )

    def make_agent(self, version: str) -> ClaimAgent:
        if self.agent_factory is not None:
            return self.agent_factory(version)
        if self.settings.provider != "gemini":
            raise RuntimeError("LLM_PROVIDER must be gemini")
        api_key = self.settings.gemini_api_key
        if not api_key:
            raise RuntimeError("Live provider unavailable: GEMINI_API_KEY is not configured")
        return ClaimAgent(
            GeminiProvider(
                model=self.settings.model,
                api_key=api_key,
                timeout_seconds=self.settings.provider_timeout_seconds,
            ),
            self.catalogue,
            request_executor=RequestExecutor(
                timeout_seconds=self.settings.provider_timeout_seconds,
                retry_delay_seconds=self.settings.retry_base_delay_seconds,
                max_attempts=1 + self.settings.max_retries_per_turn,
                semaphore=asyncio.Semaphore(self.settings.max_provider_concurrency),
            ),
            version=version,
            max_turns=self.settings.max_model_turns_per_answer,
        )

    async def classify(self, request: dict[str, Any]) -> dict[str, Any]:
        if request["brand_id"] not in self.catalogue.brands:
            from app.agent import UnknownPrimaryBrandError
            raise UnknownPrimaryBrandError("unknown primary brand")
        agent = self.make_agent(request["agent_version"])
        try:
            return await agent.classify(AgentInput(
            answer_id=request["answer_id"],
            brand_id=request["brand_id"],
            question=request["question"],
            ai_answer=request["ai_answer"],
        ))
        finally:
            await agent.aclose()

    async def run_evaluation(self, version: str, split: str) -> dict[str, Any]:
        started_at = datetime.now(timezone.utc)
        started_clock = time.perf_counter()
        run_id = str(uuid4())
        all_answers = load_answers(self.data_dir / "answers.csv")
        scheduled: list[Answer] = sorted(
            (answer for answer in all_answers if answer.split == split),
            key=lambda answer: answer.answer_id,
        )
        if not scheduled:
            raise ValueError(f"dataset has no scheduled answers for split {split}")
        agent = self.make_agent(version)
        prompt_version = version
        hashes = dataset_hashes(self.data_dir)
        source_revision = self._source_revision()
        trace_sessions: dict[str, TraceSession] = {}

        async def classify_one(answer: Answer) -> dict[str, Any]:
            answer_started = time.perf_counter()
            trace = self.observer.start_answer({
                "run_id": run_id,
                "answer_id": answer.answer_id,
                "split": split,
                "agent_version": version,
                "prompt_version": prompt_version,
                "dataset_version": DATASET_VERSION,
                "evaluator_version": EVALUATOR_VERSION,
                "catalogue_version": "speaker-catalogue-v4",
                "provider": self.settings.provider,
                "model": self.settings.model,
            })
            trace_sessions[answer.answer_id] = trace
            try:
                output = await agent.classify(
                    AgentInput(answer.answer_id, answer.brand_id, answer.question, answer.ai_answer),
                    event_sink=trace.record,
                )
                return {
                    "answer_id": answer.answer_id,
                    "status": "success",
                    "claims": output["claims"],
                    "error": None,
                    "latency_ms": round((time.perf_counter() - answer_started) * 1000),
                    "trace_id": trace.trace_id,
                }
            except Exception as exc:
                trace.record("error", {"error_type": getattr(exc, "error_type", "provider_error")})
                return {
                    "answer_id": answer.answer_id,
                    "status": "error",
                    "claims": [],
                    "error": self._safe_error(exc),
                    "latency_ms": round((time.perf_counter() - answer_started) * 1000),
                    "trace_id": trace.trace_id,
                }

        outcomes = await asyncio.gather(*(classify_one(answer) for answer in scheduled))
        await agent.aclose()
        outcomes.sort(key=lambda outcome: outcome["answer_id"])
        self._write_predictions({
            "run_id": run_id,
            "agent_version": version,
            "split": split,
            "dataset_version": DATASET_VERSION,
            "predictions": outcomes,
        }, version, split)
        answer_texts = {answer.answer_id: answer.ai_answer for answer in scheduled}
        if split == "development":
            gold = [row for row in load_gold(self.data_dir / "development_gold.csv") if row["answer_id"] in answer_texts]
        else:
            gold = None
        evaluation = evaluate_split(
            [answer.answer_id for answer in scheduled],
            gold,
            outcomes,
            answer_texts,
        )

        if split == "development":
            for answer_id, metrics in evaluation["per_answer"].items():
                trace = trace_sessions[answer_id]
                trace.score("extraction_f1", metrics["extraction_f1"])
                trace.score("joint_f1", metrics["joint_f1"])
            self._score_matched_labels(gold or [], outcomes, trace_sessions)
        self.observer.flush()

        run_status = "completed"
        finished_at = datetime.now(timezone.utc)
        counts = {
            "scheduled": len(scheduled),
            "successful": sum(item["status"] == "success" for item in outcomes),
            "errors": sum(item["status"] != "success" for item in outcomes),
            "predictions": sum(len(item["claims"]) for item in outcomes),
        }
        metrics = evaluation if split == "development" else None
        run_record = {
            "run_id": run_id,
            "split": split,
            "status": run_status,
            "agent_version": version,
            "prompt_version": prompt_version,
            "dataset_version": DATASET_VERSION,
            "evaluator_version": EVALUATOR_VERSION,
            "catalogue_version": "speaker-catalogue-v4",
            "provider": self.settings.provider,
            "model": self.settings.model,
            "settings": {
                "provider_timeout_seconds": self.settings.provider_timeout_seconds,
                "max_provider_concurrency": self.settings.max_provider_concurrency,
                "max_model_turns": 3,
                "max_attempts_per_turn": 2,
            },
            "dataset_hashes": hashes,
            "source_revision": source_revision,
            "started_at": started_at,
            "completed_at": finished_at,
            "counts": counts,
            "metrics": metrics,
            "error_count": counts["errors"],
        }
        report = {
            **run_record,
            "duration_ms": round((time.perf_counter() - started_clock) * 1000),
            "metrics": metrics,
            "outcomes": outcomes,
        }
        if self.repository is not None:
            try:
                self.repository.save_run(run_record, outcomes)
            except Exception:
                run_record["status"] = "failed"
                report["status"] = "failed"
                report["infrastructure_error"] = {"type": "PersistenceError", "message": "run evidence could not be persisted to the database"}
                logger.exception("Run persistence failed; preserving local report")
        self._write_report(report, version, split)
        return {
            "run_id": run_id,
            "status": report["status"],
            "counts": counts,
            "metrics": metrics,
        }

    @staticmethod
    def _safe_error(exc: Exception) -> dict[str, str]:
        if isinstance(exc, AgentError):
            return {"type": exc.error_type, "message": exc.message}
        if isinstance(exc, TimeoutError):
            return {"type": "timeout", "message": "provider request timed out"}
        return {"type": "provider_error", "message": "answer processing failed"}

    @staticmethod
    def _source_revision() -> str | None:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2, check=False
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    @staticmethod
    def _write_report(report: dict[str, Any], version: str, split: str) -> None:
        directory = Path("runs") / f"{version}_{split}"
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "report.json.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        temporary.replace(directory / "report.json")

    @staticmethod
    def _write_predictions(predictions: dict[str, Any], version: str, split: str) -> None:
        directory = Path("runs") / f"{version}_{split}"
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "predictions.json.tmp"
        temporary.write_text(json.dumps(predictions, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        temporary.replace(directory / "predictions.json")

    @staticmethod
    def _score_matched_labels(gold: list[dict[str, Any]], outcomes: list[dict[str, Any]], traces: dict[str, TraceSession]) -> None:
        prediction_by_span = {
            (outcome["answer_id"], claim["start_char"], claim["end_char"]): claim
            for outcome in outcomes
            for claim in outcome["claims"]
        }
        for item in gold:
            prediction = prediction_by_span.get((item["answer_id"], item["start_char"], item["end_char"]))
            if prediction is not None:
                traces[item["answer_id"]].score(
                    "label_correct",
                    1.0 if prediction["label"] == item["expected_label"] else 0.0,
                    comment=item["claim_id"],
                )