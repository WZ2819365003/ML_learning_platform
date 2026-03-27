"""
ML Training Platform -- FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models.database import Base, async_engine
from app.api.routes import data, training, logs, experiment, visualization, model_mgmt
from app.api.websocket import router as ws_router


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

    yield

    # --- Shutdown ---
    await async_engine.dispose()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="ML Training Platform",
        version="0.1.0",
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
    app.include_router(ws_router)

    # ---- Root-level endpoints ----

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok", "version": "0.1.0"}

    return app


app = create_app()
