from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AgentVersion = Literal["v1", "v2"]
Split = Literal["development", "check"]
Label = Literal["supported", "contradicted", "unknown"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClassifyRequest(StrictModel):
    answer_id: str = Field(min_length=1)
    brand_id: str = Field(min_length=1)
    question: str
    ai_answer: str
    agent_version: AgentVersion


class RunRequest(StrictModel):
    agent_version: AgentVersion
    split: Split


class ClaimOutput(StrictModel):
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    label: Label
    fact_ids: list[str]
    reason: str


class ClassificationOutput(StrictModel):
    answer_id: str
    claims: list[ClaimOutput]


class ErrorOutput(StrictModel):
    error_type: str
    message: str