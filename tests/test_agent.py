import json

import pytest

from app.agent import AgentInput, ClaimAgent, IllegalToolError, ModelTurnLimitError, ProviderCallError, RequestedBrandError
from app.catalogue import BrandCatalogue
from app.provider import ProviderReply, ToolCall
from app.reliability import RequestExecutor


class FakeProvider:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def complete(self, messages, tools):
        self.calls.append((messages, tools))
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


@pytest.fixture
def catalogue():
    from pathlib import Path
    return BrandCatalogue(Path("data/brand_facts.csv"))


@pytest.mark.asyncio
async def test_agent_uses_requested_catalogue_tool_and_keeps_gold_out(catalogue):
    source = "ClearTrail Mini is listed at INR 1,299."
    fact = next(f for f in catalogue.get_brand_facts("cleartrail") if f.field == "price_2026")
    provider = FakeProvider([
        ProviderReply(None, (ToolCall("t1", "get_brand_facts", '{"brand_id":"cleartrail"}'),), "fake"),
        ProviderReply(json.dumps({"answer_id": "A01", "claims": [{
            "quote": source, "label": "supported", "fact_ids": [fact.fact_id], "reason": "The 2026 catalogue lists INR 1299."
        }]}), (), "fake"),
    ])
    agent = ClaimAgent(provider, catalogue, RequestExecutor(max_attempts=1))
    result = await agent.classify(AgentInput("A01", "cleartrail", "Question?", source))
    assert result["claims"][0]["quote"] == source
    assert "development_gold" not in json.dumps(provider.calls)
    assert "expected_label" not in json.dumps(provider.calls)
    assert provider.calls[0][1][0]["function"]["name"] == "get_brand_facts"
    tool_message = provider.calls[1][0][-1]
    assert tool_message["role"] == "tool"
    assert json.loads(tool_message["content"])["fact_version"] == "speaker-catalogue-v4"


@pytest.mark.asyncio
async def test_agent_rejects_illegal_tool(catalogue):
    provider = FakeProvider([ProviderReply(None, (ToolCall("t1", "delete_database", "{}"),), "fake")])
    with pytest.raises(IllegalToolError):
        await ClaimAgent(provider, catalogue, RequestExecutor(max_attempts=1)).classify(AgentInput("A01", "cleartrail", "q", ""))


@pytest.mark.asyncio
async def test_agent_rejects_unknown_requested_brand(catalogue):
    provider = FakeProvider([ProviderReply(None, (ToolCall("t1", "get_brand_facts", '{"brand_id":"unknown"}'),), "fake")])
    with pytest.raises(RequestedBrandError):
        await ClaimAgent(provider, catalogue, RequestExecutor(max_attempts=1)).classify(AgentInput("A01", "cleartrail", "q", ""))


@pytest.mark.asyncio
async def test_agent_invalid_output_is_answer_error_with_no_claims(catalogue):
    provider = FakeProvider([ProviderReply("not json", (), "fake")])
    with pytest.raises(ProviderCallError):
        await ClaimAgent(provider, catalogue, RequestExecutor(max_attempts=1)).classify(AgentInput("A01", "cleartrail", "q", ""))


@pytest.mark.asyncio
async def test_agent_turn_limit_is_enforced(catalogue):
    replies = [ProviderReply(None, (ToolCall(str(i), "get_brand_facts", '{"brand_id":"cleartrail"}'),), "fake") for i in range(3)]
    provider = FakeProvider(replies)
    with pytest.raises(ModelTurnLimitError):
        await ClaimAgent(provider, catalogue, RequestExecutor(max_attempts=1)).classify(AgentInput("A01", "cleartrail", "q", ""))
    assert len(provider.calls) == 3


@pytest.mark.asyncio
async def test_primary_unknown_brand_fails_before_provider_call(catalogue):
    provider = FakeProvider([])
    with pytest.raises(RequestedBrandError):
        await ClaimAgent(provider, catalogue).classify(AgentInput("A01", "missing", "q", ""))
    assert provider.calls == []