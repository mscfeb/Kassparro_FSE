from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agent import AgentError, UnknownPrimaryBrandError
from app.api import create_app


class StubRepository:
    def __init__(self, result=None):
        self.result = result
        self.requested_id = None

    def get_run(self, run_id):
        self.requested_id = run_id
        return self.result


class StubService:
    def __init__(self, classify_result=None, classify_error=None, run_result=None, repository=None):
        self.classify_result = classify_result or {"answer_id": "A01", "claims": []}
        self.classify_error = classify_error
        self.run_result = run_result or {"run_id": "id", "status": "completed", "counts": {}, "metrics": None}
        self.repository = repository
        self.classify_request = None

    async def classify(self, request):
        self.classify_request = request
        if self.classify_error:
            raise self.classify_error
        return self.classify_result

    async def run_evaluation(self, version, split):
        return self.run_result


def test_classify_success_and_empty_claims():
    service = StubService()
    client = TestClient(create_app(service=service))
    response = client.post("/classify", json={
        "answer_id": "A01", "brand_id": "cleartrail", "question": "Q?",
        "ai_answer": "No factual claims.", "agent_version": "v1",
    })
    assert response.status_code == 200
    assert response.json() == {"answer_id": "A01", "claims": []}
    assert service.classify_request["ai_answer"] == "No factual claims."


def test_classify_rejects_gold_and_caller_claim_fields():
    client = TestClient(create_app(service=StubService()))
    response = client.post("/classify", json={
        "answer_id": "A01", "brand_id": "cleartrail", "question": "Q?",
        "ai_answer": "Claim.", "agent_version": "v1", "claims": [], "expected_label": "supported",
    })
    assert response.status_code == 422


@pytest.mark.parametrize("error,status", [
    (UnknownPrimaryBrandError("unknown primary brand"), 404),
    (AgentError("provider request failed"), 502),
])
def test_classify_sanitizes_agent_errors(error, status):
    client = TestClient(create_app(service=StubService(classify_error=error)))
    response = client.post("/classify", json={
        "answer_id": "A01", "brand_id": "cleartrail", "question": "Q?",
        "ai_answer": "Claim.", "agent_version": "v1",
    })
    assert response.status_code == status
    assert "traceback" not in response.text.lower()


def test_evaluation_run_returns_created_and_check_metrics_null():
    client = TestClient(create_app(service=StubService()))
    response = client.post("/evaluation-runs", json={"agent_version": "v1", "split": "check"})
    assert response.status_code == 201
    assert response.json()["metrics"] is None


def test_run_retrieval_uuid_and_missing_run_errors():
    repository = StubRepository()
    client = TestClient(create_app(service=StubService(repository=repository)))
    assert client.get("/evaluation-runs/not-a-uuid").status_code == 422
    response = client.get("/evaluation-runs/123e4567-e89b-12d3-a456-426614174000")
    assert response.status_code == 404
    assert repository.requested_id == "123e4567-e89b-12d3-a456-426614174000"


def test_retrieval_reports_unavailable_persistence():
    client = TestClient(create_app(service=StubService()))
    response = client.get("/evaluation-runs/123e4567-e89b-12d3-a456-426614174000")
    assert response.status_code == 503