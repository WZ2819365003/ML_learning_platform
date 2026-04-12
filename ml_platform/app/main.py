"""
ML Training Platform -- FastAPI application entry point.
"""

from contextlib import asynccontextmanager
from typing import Iterable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models.database import Base, async_engine
from app.api.routes import data, training, logs, experiment, visualization, model_mgmt
from app.api.routes.deploy import deploy_router, inference_router
from app.api.routes.dl import router as dl_router
from app.api.routes.timesfm import router as timesfm_router, ts_router
from app.api.websocket import router as ws_router
from app.services.timeseries_service import resume_unfinished_ts_tasks


async def _ensure_sqlite_columns(table_name: str, columns: Iterable[tuple[str, str]]) -> None:
    if async_engine.url.get_backend_name() != "sqlite":
        return

    async with async_engine.begin() as conn:
        rows = await conn.exec_driver_sql(f"PRAGMA table_info({table_name})")
        existing = {row[1] for row in rows.fetchall()}
        for column_name, column_sql in columns:
            if column_name not in existing:
                await conn.exec_driver_sql(
                    f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"
                )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown lifecycle events."""
    settings = get_settings()

    # --- Startup ---
    # Ensure storage directories exist
    settings.ensure_storage_dirs()

    # Create all database tables
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await _ensure_sqlite_columns(
        "ts_forecast_tasks",
        (
            ("notes", "TEXT"),
            ("tags", "JSON"),
        ),
    )
    await resume_unfinished_ts_tasks()

    yield

    # --- Shutdown ---
    await async_engine.dispose()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="ML Training Platform",
        version="2.3.0",
        lifespan=lifespan,
    )

    # CORS -- allow all origins during development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers under /api prefix
    app.include_router(data.router, prefix="/api")
    app.include_router(training.router, prefix="/api")
    app.include_router(logs.router, prefix="/api")
    app.include_router(experiment.router, prefix="/api")
    app.include_router(visualization.router, prefix="/api")
    app.include_router(model_mgmt.router, prefix="/api")
    app.include_router(deploy_router, prefix="/api")   # → /api/deploy/...
    app.include_router(inference_router)               # → /inference/... (no /api prefix)
    app.include_router(dl_router, prefix="/api")        # → /api/dl/...
    app.include_router(timesfm_router, prefix="/api")  # → /api/timesfm/...
    app.include_router(ts_router, prefix="/api")
    app.include_router(ws_router)

    # ---- Root-level endpoints ----

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok", "version": "2.3.0"}

    return app


app = create_app()
