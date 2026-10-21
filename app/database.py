from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.models import Base


def create_database(settings: Settings, *, create_tables: bool = False) -> tuple[Engine, sessionmaker]:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not configured")
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    if engine.dialect.name == "sqlite":
        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(connection, _record):
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    if create_tables:
        Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)