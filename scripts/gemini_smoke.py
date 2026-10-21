import asyncio
import json
from pathlib import Path

from app.agent import AgentError, AgentInput, ClaimAgent
from app.catalogue import BrandCatalogue
from app.config import Settings
from app.provider import GeminiProvider
from app.reliability import RequestExecutor


async def run() -> int:
    settings = Settings.from_environment()
    if not settings.gemini_api_key:
        print("Gemini smoke test blocked: GEMINI_API_KEY is not configured.")
        return 2
    catalogue = BrandCatalogue(Path(settings.data_dir) / "brand_facts.csv")
    provider = GeminiProvider(
        model=settings.model,
        api_key=settings.gemini_api_key,
        timeout_seconds=settings.provider_timeout_seconds,
    )
    tool_calls = []
    generations = []

    def record_event(kind, payload):
        if kind == "tool" and payload.get("name") == "get_brand_facts":
            tool_calls.append(payload)
        elif kind == "generation":
            output = payload.get("output") or ""
            stripped = output.strip()
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            generations.append({
                "text_present": bool(payload.get("output")),
                "text_length": len(output),
                "json_parseable": parsed is not None,
                "json_keys": sorted(parsed) if isinstance(parsed, dict) else None,
                "starts_with_object": stripped.startswith("{"),
                "starts_with_code_fence": stripped.startswith("```"),
                "first_character_codepoint": ord(stripped[0]) if stripped else None,
                "tool_call_count": len(payload.get("tool_calls") or []),
                "finish_reason": payload.get("finish_reason"),
            })

    try:
        agent = ClaimAgent(
            provider,
            catalogue,
            RequestExecutor(
                timeout_seconds=settings.provider_timeout_seconds,
                retry_delay_seconds=settings.retry_base_delay_seconds,
                max_attempts=1 + settings.max_retries_per_turn,
            ),
        )
        try:
            result = await agent.classify(
                AgentInput(
                    answer_id="SMOKE-01",
                    brand_id="cleartrail",
                    question="What is the current listed price?",
                    ai_answer="ClearTrail Mini is listed at INR 1,299.",
                ),
                event_sink=record_event,
            )
        except AgentError as exc:
            print(f"Gemini smoke test failed: {exc.error_type}: {exc.message}")
            print(f"catalogue_tool_calls_observed={len(tool_calls)}")
            print(f"generation_turns={generations}")
            if exc.__cause__ is not None:
                print(f"validation_cause={type(exc.__cause__).__name__}: {exc.__cause__}")
            return 1
        if not tool_calls:
            print("Gemini responded but did not request get_brand_facts; smoke test failed.")
            return 1
        print(f"Gemini tool-calling smoke test passed; accepted_claims={len(result['claims'])}.")
        return 0
    finally:
        await provider.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))