from __future__ import annotations

from uuid import UUID

from fastapi import FastAPI, HTTPException

from app.agent import AgentError, UnknownPrimaryBrandError
from app.config import Settings
from app.database import create_database
from app.repository import RunRepository
from app.schemas import ClassifyRequest, RunRequest
from app.services import AssessmentService


def create_app(
    service: AssessmentService | None = None,
    repository: RunRepository | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    app = FastAPI(title="KASPARRO FSE I Assessment", version="1.0.0")
    resolved_settings = settings or Settings.from_environment()
    if service is not None:
        assessment = service
    else:
        if repository is None and resolved_settings.database_url:
            _engine, session_factory = create_database(resolved_settings, create_tables=True)
            repository = RunRepository(session_factory)
        assessment = AssessmentService(settings=resolved_settings, repository=repository)

    @app.post("/classify")
    async def classify(request: ClassifyRequest):
        try:
            return await assessment.classify(request.model_dump())
        except UnknownPrimaryBrandError as exc:
            raise HTTPException(status_code=404, detail={"error_type": exc.error_type, "message": exc.message}) from exc
        except AgentError as exc:
            raise HTTPException(status_code=exc.status_code, detail={"error_type": exc.error_type, "message": exc.message}) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail={"error_type": "provider_unavailable", "message": "live provider is not configured"}) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail={"error_type": "classification_failed", "message": "classification failed"}) from exc

    @app.post("/evaluation-runs", status_code=201)
    async def create_evaluation_run(request: RunRequest):
        try:
            return await assessment.run_evaluation(request.agent_version, request.split)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail={"error_type": "provider_unavailable", "message": "live provider is not configured"}) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail={"error_type": "run_failed", "message": "evaluation run failed"}) from exc

    @app.get("/evaluation-runs/{run_id}")
    async def get_evaluation_run(run_id: str):
        try:
            normalized_id = str(UUID(run_id))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail={"error_type": "invalid_uuid", "message": "run_id must be a UUID"}) from exc
        if assessment.repository is None:
            raise HTTPException(status_code=503, detail={"error_type": "persistence_unavailable", "message": "database is not configured"})
        result = assessment.repository.get_run(normalized_id)
        if result is None:
            raise HTTPException(status_code=404, detail={"error_type": "run_not_found", "message": "evaluation run not found"})
        return result

    return app


app = create_app()