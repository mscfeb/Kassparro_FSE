import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy import event
from sqlalchemy.orm import sessionmaker

from app.models import AnswerResult, Base, ClaimResult, Run
from app.repository import RunRepository


@pytest.fixture
def repository(tmp_path):
    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    yield RunRepository(sessionmaker(bind=engine, expire_on_commit=False))
    engine.dispose()


def run_record(run_id):
    return {
        "run_id": run_id, "split": "development", "status": "completed", "agent_version": "v1",
        "prompt_version": "v1", "dataset_version": "long-brand-answers-50-v4",
        "evaluator_version": "sentence-spans-and-labels-v4", "catalogue_version": "speaker-catalogue-v4",
        "provider": "fake", "model": "test", "settings": {}, "dataset_hashes": {},
        "counts": {}, "metrics": {}, "error_count": 1,
    }


def test_persistence_retrieves_stable_answer_and_claim_order(repository):
    repository.save_run(run_record("run-1"), [
        {"answer_id": "A02", "status": "success", "latency_ms": 1, "claims": []},
        {"answer_id": "A01", "status": "error", "error": {"type": "ProviderError"}, "latency_ms": 4, "claims": [
            {"claim_id": "A01:5:9", "start_char": 5, "end_char": 9, "quote": "Two.", "label": "unknown", "fact_ids": [], "reason": "No fact."},
            {"claim_id": "A01:0:4", "start_char": 0, "end_char": 4, "quote": "One.", "label": "supported", "fact_ids": ["f1"], "reason": "Known."},
        ]},
    ])
    retrieved = repository.get_run("run-1")
    assert [row["answer_id"] for row in retrieved["answers"]] == ["A01", "A02"]
    assert [row["start_char"] for row in retrieved["answers"][0]["claims"]] == [0, 5]
    assert retrieved["answers"][1]["claims"] == []


def test_unique_constraints_and_foreign_keys(repository):
    session_factory = repository.session_factory
    repository.save_run(run_record("run-2"), [{"answer_id": "A01", "status": "success", "latency_ms": 0, "claims": []}])
    with session_factory() as session, pytest.raises(IntegrityError):
        session.add(AnswerResult(run_id="run-2", answer_id="A01", status="success", latency_ms=0))
        session.commit()
    with session_factory() as session, pytest.raises(IntegrityError):
        session.add(ClaimResult(run_id="missing", answer_id="A01", claim_id="x", start_char=0, end_char=1,
                                quote="x", label="unknown", fact_ids=[], reason="x"))
        session.commit()