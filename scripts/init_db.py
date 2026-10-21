from app.config import Settings
from app.database import create_database


def main() -> int:
    settings = Settings.from_environment()
    if not settings.database_url:
        print("DATABASE_URL is not configured; schema initialization skipped.")
        return 2
    engine, _ = create_database(settings, create_tables=True)
    try:
        with engine.connect():
            pass
    finally:
        engine.dispose()
    print("Database connection succeeded and assessment tables were initialized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())