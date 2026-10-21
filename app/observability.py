from __future__ import annotations

import logging
import os
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class TraceSession:
    def __init__(self, trace: Any | None, trace_id: str | None):
        self.trace = trace
        self.trace_id = trace_id

    def record(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.trace is None:
            return
        try:
            if event_type == "generation":
                usage = payload.get("usage") or {}
                self.trace.generation(
                    name="model-turn",
                    input=payload.get("input"),
                    output={"content": payload.get("output"), "tool_calls": payload.get("tool_calls")},
                    model=payload.get("model"),
                    usage={"input": usage.get("prompt_tokens"), "output": usage.get("completion_tokens")}
                    if usage else None,
                    metadata={
                        "provider": payload.get("provider"),
                        "generation_id": payload.get("generation_id"),
                    },
                )
            elif event_type == "tool":
                self.trace.span(
                    name=payload.get("name", "catalogue-tool"),
                    input=payload.get("input"),
                    output=payload.get("output"),
                )
            elif event_type == "validation_failure":
                self.trace.span(
                    name="validation-failure",
                    level="ERROR",
                    status_message=payload.get("error_type"),
                )
            elif event_type == "accepted_claims":
                self.trace.update(output=payload.get("claims"))
            elif event_type == "error":
                self.trace.update(metadata={"error_type": payload.get("error_type")})
        except Exception:
            logger.warning("Langfuse observation could not be recorded")

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        if self.trace is None:
            return
        try:
            self.trace.score(name=name, value=value, comment=comment)
        except Exception:
            logger.warning("Langfuse score could not be recorded")


class LangfuseObserver:
    def __init__(self, public_key: str | None = None, secret_key: str | None = None, host: str | None = None):
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.client = None
        if self.public_key and self.secret_key:
            try:
                from langfuse import Langfuse
                self.client = Langfuse(public_key=self.public_key, secret_key=self.secret_key, host=self.host)
            except Exception:
                logger.warning("Langfuse is configured but unavailable")

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def start_answer(self, metadata: dict[str, Any]) -> TraceSession:
        trace_id = str(uuid4())
        if self.client is None:
            return TraceSession(None, None)
        try:
            trace = self.client.trace(
                id=trace_id,
                name="claim-classification-answer",
                metadata=metadata,
                input={"answer_id": metadata.get("answer_id")},
            )
            return TraceSession(trace, trace_id)
        except Exception:
            logger.warning("Langfuse trace could not be started")
            return TraceSession(None, None)

    def flush(self) -> None:
        if self.client is not None:
            try:
                self.client.flush()
            except Exception:
                logger.warning("Langfuse flush failed")