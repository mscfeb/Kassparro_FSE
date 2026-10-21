import argparse
import asyncio

from app.config import Settings
from app.database import create_database
from app.repository import RunRepository
from app.services import AssessmentService


async def run(version: str, split: str) -> int:
    settings = Settings.from_environment()
    repository = None
    engine = None
    if settings.database_url:
        engine, session_factory = create_database(settings, create_tables=True)
        repository = RunRepository(session_factory)
    try:
        result = await AssessmentService(settings, repository=repository).run_evaluation(version, split)
        print(result)
        return 0 if result["status"] == "completed" else 1
    finally:
        if engine is not None:
            engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=("v1", "v2"), required=True)
    parser.add_argument("--split", choices=("development", "check"), required=True)
    args = parser.parse_args()
    return asyncio.run(run(args.version, args.split))


if __name__ == "__main__":
    raise SystemExit(main())