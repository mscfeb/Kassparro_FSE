from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from app.catalogue import BrandCatalogue, UnknownBrandError
from app.provider import ProviderReply
from app.reliability import (
    PermanentProviderError,
    ProviderTimeoutError,
    RequestExecutor,
    TransientProviderError,
)
from app.validation import InvalidAgentOutput, validate_model_claims

MAX_MODEL_TURNS = 3
TOOL_NAME = "get_brand_facts"
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Read the supplied catalogue rows for one known brand.",
        "parameters": {
            "type": "object",
            "properties": {"brand_id": {"type": "string"}},
            "required": ["brand_id"],
            "additionalProperties": False,
        },
    },
}


class Provider(Protocol):
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ProviderReply: ...


class AgentError(Exception):
    error_type = "agent_error"
    status_code = 502

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class IllegalToolError(AgentError):
    error_type = "illegal_tool"


class ModelTurnLimitError(AgentError):
    error_type = "model_turn_limit"


class ProviderCallError(AgentError):
    error_type = "provider_error"


class AnswerTimeoutError(AgentError):
    error_type = "timeout"
    status_code = 504


class RequestedBrandError(AgentError):
    error_type = "unknown_requested_brand"


class UnknownPrimaryBrandError(RequestedBrandError):
    status_code = 404


@dataclass(frozen=True)
class AgentInput:
    answer_id: str
    brand_id: str
    question: str
    ai_answer: str


def system_prompt(version: str) -> str:
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / f"{version}.txt"
    return prompt_path.read_text(encoding="utf-8")


class ClaimAgent:
    def __init__(
        self,
        provider: Provider,
        catalogue: BrandCatalogue,
        request_executor: RequestExecutor | None = None,
        version: str = "v1",
        max_turns: int = MAX_MODEL_TURNS,
    ):
        if version not in {"v1", "v2"}:
            raise ValueError("agent version must be v1 or v2")
        if not 1 <= max_turns <= MAX_MODEL_TURNS:
            raise ValueError("max_turns must be between one and three")
        self.provider = provider
        self.catalogue = catalogue
        self.request_executor = request_executor or RequestExecutor()
        self.version = version
        self.max_turns = max_turns

    async def aclose(self) -> None:
        close = getattr(self.provider, "aclose", None)
        if close is not None:
            await close()

    async def classify(
        self,
        agent_input: AgentInput,
        event_sink: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        def emit(event_type: str, payload: dict[str, Any]) -> None:
            if event_sink is not None:
                try:
                    event_sink(event_type, payload)
                except Exception:
                    pass

        if agent_input.brand_id not in self.catalogue.brands:
            raise UnknownPrimaryBrandError("unknown primary brand")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt(self.version)},
            {
                "role": "user",
                "content": json.dumps({
                    "answer_id": agent_input.answer_id,
                    "brand_id": agent_input.brand_id,
                    "question": agent_input.question,
                    "ai_answer": agent_input.ai_answer,
                }, ensure_ascii=False),
            },
        ]
        for _ in range(self.max_turns):
            try:
                reply = await self.request_executor.call(lambda: self.provider.complete(messages, [TOOL_SCHEMA]))
            except ProviderTimeoutError as exc:
                emit("error", {"error_type": "timeout"})
                raise AnswerTimeoutError("provider request timed out") from exc
            except (PermanentProviderError, TransientProviderError) as exc:
                emit("error", {"error_type": "provider_error"})
                raise ProviderCallError(str(exc)) from exc
            except AgentError:
                raise
            except Exception as exc:
                emit("error", {"error_type": "provider_error"})
                raise ProviderCallError("provider request failed") from exc
            trace_input = [
                {key: value for key, value in message.items() if key != "provider_state"}
                for message in messages
            ]
            emit("generation", {
                "input": trace_input,
                "output": reply.content,
                "tool_calls": [
                    {"id": call.call_id, "name": call.name, "arguments": call.arguments}
                    for call in reply.tool_calls
                ],
                "provider": reply.provider,
                "model": reply.model,
                "usage": reply.usage,
                "generation_id": reply.generation_id,
                "finish_reason": reply.finish_reason,
            })

            if reply.tool_calls:
                if reply.provider_state is not None:
                    messages.append({"role": "assistant", "provider_state": reply.provider_state})
                else:
                    messages.append({"role": "assistant", "content": reply.content})
                for call in reply.tool_calls:
                    if call.name != TOOL_NAME:
                        emit("validation_failure", {"error_type": "illegal_tool"})
                        raise IllegalToolError("model requested a prohibited tool")
                    try:
                        arguments = json.loads(call.arguments)
                    except (TypeError, json.JSONDecodeError) as exc:
                        raise IllegalToolError("catalogue tool arguments are invalid") from exc
                    if not isinstance(arguments, Mapping) or set(arguments) != {"brand_id"}:
                        emit("validation_failure", {"error_type": "invalid_tool_arguments"})
                        raise IllegalToolError("catalogue tool arguments are invalid")
                    try:
                        tool_result = self.catalogue.tool_payload(arguments["brand_id"])
                    except UnknownBrandError as exc:
                        emit("validation_failure", {"error_type": "unknown_requested_brand"})
                        raise RequestedBrandError("model requested an unknown brand") from exc
                    emit("tool", {"name": TOOL_NAME, "input": arguments, "output": tool_result})
                    messages.append({
                        "role": "tool",
                        "name": call.name,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    })
                continue

            try:
                payload = json.loads(reply.content or "")
                if not isinstance(payload, Mapping) or set(payload) != {"answer_id", "claims"}:
                    raise InvalidAgentOutput("final response must contain answer_id and claims only")
                if payload["answer_id"] != agent_input.answer_id:
                    raise InvalidAgentOutput("final answer_id does not match the request")
                claims = validate_model_claims(
                    agent_input.answer_id,
                    agent_input.ai_answer,
                    payload["claims"],
                    self.catalogue.fact_ids,
                )
            except (json.JSONDecodeError, InvalidAgentOutput, TypeError) as exc:
                emit("validation_failure", {"error_type": "invalid_output"})
                raise ProviderCallError("provider returned invalid structured output") from exc
            result = {"answer_id": agent_input.answer_id, "claims": claims}
            emit("accepted_claims", {"claims": claims})
            return result
        emit("error", {"error_type": "model_turn_limit"})
        raise ModelTurnLimitError("model turn limit reached without a valid final response")