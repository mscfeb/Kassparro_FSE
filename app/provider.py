from __future__ import annotations
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from app.reliability import PermanentProviderError, TransientProviderError

@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ProviderReply:
    content: str | None
    tool_calls: tuple[ToolCall, ...]
    model: str
    provider: str = "gemini"
    usage: dict[str, int] | None = None
    generation_id: str | None = None
    provider_state: Any | None = None
    finish_reason: str | None = None


class GeminiProvider:
    def __init__(self, model: str | None = None, api_key: str | None = None, timeout_seconds: float = 15.0):
        token = api_key or os.getenv("GEMINI_API_KEY")
        if not token:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("Install the google-genai dependency to use the live provider") from exc
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._types = types
        self._client = genai.Client(
            api_key=token,
            http_options=types.HttpOptions(
                timeout=int(timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        self._async_client = self._client.aio

    async def aclose(self) -> None:
        await self._async_client.aclose()
        client = getattr(self, "_client", None)
        if client is not None:
            client.close()

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ProviderReply:
        system_instruction = next((message["content"] for message in messages if message.get("role") == "system"), None)
        contents = []
        for message in messages:
            role = message.get("role")
            if role == "user":
                contents.append(self._types.Content(
                    role="user",
                    parts=[self._types.Part.from_text(text=message["content"])],
                ))
            elif role == "assistant" and message.get("provider_state") is not None:
                contents.append(message["provider_state"])
            elif role == "tool":
                contents.append(self._types.Content(
                    role="tool",
                    parts=[self._types.Part.from_function_response(
                        name=message["name"],
                        response=json.loads(message["content"]),
                    )],
                ))
            elif role == "assistant":
                text = message.get("content")
                if text:
                    contents.append(self._types.Content(
                        role="model",
                        parts=[self._types.Part.from_text(text=text)],
                    ))

        declarations = [
            self._types.FunctionDeclaration(
                name=tool["function"]["name"],
                description=tool["function"].get("description", ""),
                parameters_json_schema=tool["function"]["parameters"],
            )
            for tool in tools
        ]
        config = self._types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[self._types.Tool(function_declarations=declarations)],
            tool_config=self._types.ToolConfig(
                function_calling_config=self._types.FunctionCallingConfig(
                    mode="AUTO",
                )
            ),
            automatic_function_calling=self._types.AutomaticFunctionCallingConfig(disable=True),
        )
        try:
            response = await self._async_client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            self._raise_provider_error(exc)

        raw_content = response.candidates[0].content if response.candidates else None
        function_calls = list(response.function_calls or [])
        if not function_calls and raw_content is not None:
            function_calls = [
                part.function_call
                for part in (raw_content.parts or [])
                if part.function_call is not None
            ]
        calls = tuple(
            ToolCall(
                call_id=call.id or f"gemini-call-{index}",
                name=call.name or "",
                arguments=json.dumps(call.args or {}, ensure_ascii=False),
            )
            for index, call in enumerate(function_calls, start=1)
        )
        usage_metadata = response.usage_metadata
        usage = None
        if usage_metadata:
            usage = {
                "prompt_tokens": usage_metadata.prompt_token_count or 0,
                "completion_tokens": usage_metadata.candidates_token_count or 0,
                "total_tokens": usage_metadata.total_token_count or 0,
            }
        text = "".join(
            part.text or ""
            for part in (raw_content.parts if raw_content is not None else [])
            if part.text is not None and not getattr(part, "thought", False)
        ) or None
        return ProviderReply(
            content=text,
            tool_calls=calls,
            model=response.model_version or self.model,
            usage=usage,
            generation_id=response.response_id,
            provider_state=raw_content,
            finish_reason=str(getattr(response.candidates[0], "finish_reason", None))
            if response.candidates and getattr(response.candidates[0], "finish_reason", None) is not None
            else None,
        )

    @staticmethod
    def _raise_provider_error(exc: Exception) -> None:
        try:
            from google.genai.errors import APIError
        except ImportError:
            raise PermanentProviderError("Gemini provider request failed") from exc
        if isinstance(exc, APIError):
            code = exc.code
            if code in {408, 429} or (isinstance(code, int) and code >= 500):
                raise TransientProviderError(f"Gemini provider request failed transiently (HTTP {code})") from exc
            raise PermanentProviderError(f"Gemini provider request rejected (HTTP {code})") from exc
        raise PermanentProviderError("Gemini provider request failed") from exc