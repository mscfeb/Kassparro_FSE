from types import SimpleNamespace

import pytest
from google.genai import types

from app.agent import TOOL_SCHEMA
from app.provider import GeminiProvider


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeAsyncClient:
    def __init__(self, models):
        self.models = models

    async def aclose(self):
        return None


def response(content, function_calls=None, text=None):
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=content)],
        function_calls=function_calls or [],
        text=text,
        model_version="gemini-2.5-flash",
        response_id="response-1",
        usage_metadata=SimpleNamespace(
            prompt_token_count=40,
            candidates_token_count=9,
            total_token_count=49,
        ),
    )


@pytest.mark.asyncio
async def test_gemini_native_function_call_and_response_continuation():
    model_candidate = types.Content(
        role="model",
        parts=[types.Part.from_function_call(name="get_brand_facts", args={"brand_id": "cleartrail"})],
    )
    call = SimpleNamespace(id="call-1", name="get_brand_facts", args={"brand_id": "cleartrail"})
    final_content = '{"answer_id":"A01","claims":[]}'
    client_models = FakeModels([response(model_candidate, [call]), response(types.Content(role="model", parts=[types.Part.from_text(text=final_content)]), text=final_content)])
    provider = GeminiProvider.__new__(GeminiProvider)
    provider.model = "gemini-2.5-flash"
    provider._types = types
    provider._async_client = FakeAsyncClient(client_models)

    first = await provider.complete([
        {"role": "system", "content": "system"},
        {"role": "user", "content": '{"answer_id":"A01"}'},
    ], [TOOL_SCHEMA])
    assert first.provider == "gemini"
    assert first.tool_calls[0].name == "get_brand_facts"
    assert first.tool_calls[0].arguments == '{"brand_id": "cleartrail"}'
    assert first.provider_state is model_candidate
    first_request = client_models.calls[0]
    config = first_request["config"]
    assert config.automatic_function_calling.disable is True
    assert config.tool_config.function_calling_config.mode == "AUTO"
    assert config.tools[0].function_declarations[0].name == "get_brand_facts"
    assert config.response_mime_type is None
    assert config.response_json_schema is None
    assert first_request["contents"][0].parts[0].text == '{"answer_id":"A01"}'

    final = await provider.complete([
        {"role": "system", "content": "system"},
        {"role": "user", "content": '{"answer_id":"A01"}'},
        {"role": "assistant", "provider_state": first.provider_state},
        {"role": "tool", "name": "get_brand_facts", "content": '{"fact_version":"speaker-catalogue-v4","facts":[]}'},
    ], [TOOL_SCHEMA])
    second_contents = client_models.calls[1]["contents"]
    assert second_contents[1] is model_candidate
    function_response = second_contents[2].parts[0].function_response
    assert function_response.name == "get_brand_facts"
    assert function_response.response["fact_version"] == "speaker-catalogue-v4"
    assert final.content == final_content
    assert final.usage == {"prompt_tokens": 40, "completion_tokens": 9, "total_tokens": 49}
    await provider.aclose()


@pytest.mark.asyncio
async def test_gemini_falls_back_to_candidate_function_call_parts():
    candidate = types.Content(
        role="model",
        parts=[types.Part.from_function_call(name="get_brand_facts", args={"brand_id": "cleartrail"})],
    )
    models = FakeModels([response(candidate, function_calls=[])])
    provider = GeminiProvider.__new__(GeminiProvider)
    provider.model = "gemini-2.5-flash"
    provider._types = types
    provider._async_client = FakeAsyncClient(models)
    reply = await provider.complete(
        [{"role": "user", "content": "request"}],
        [TOOL_SCHEMA],
    )
    assert reply.content is None
    assert reply.tool_calls[0].name == "get_brand_facts"
    assert reply.provider_state is candidate
    await provider.aclose()


@pytest.mark.asyncio
async def test_gemini_does_not_include_thought_parts_in_final_text():
    candidate = types.Content(
        role="model",
        parts=[
            types.Part(text="internal reasoning", thought=True),
            types.Part.from_text(text='{"answer_id":"A01","claims":[]}'),
        ],
    )
    models = FakeModels([response(candidate, function_calls=[])])
    provider = GeminiProvider.__new__(GeminiProvider)
    provider.model = "gemini-2.5-flash"
    provider._types = types
    provider._async_client = FakeAsyncClient(models)
    reply = await provider.complete([{"role": "user", "content": "request"}], [TOOL_SCHEMA])
    assert reply.content == '{"answer_id":"A01","claims":[]}'
    await provider.aclose()