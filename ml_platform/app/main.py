"""
ML Training Platform -- FastAPI application entry point.
"""

import shutil
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select

from app.config import get_settings
from app.core.auth import request_is_authorized
from app.models.database import Base, async_engine, Dataset, ModelTagLibrary, async_session_factory
from app.api.routes import data, training, logs, experiment, visualization, model_mgmt
from app.api.routes.auth import router as auth_router
from app.api.routes.deploy import deploy_router, inference_router
from app.api.routes.dl import router as dl_router
from app.api.routes.timesfm import router as timesfm_router, ts_router
from app.api.routes.platform_tasks import router as platform_tasks_router
from app.api.routes.platform_experiments import router as platform_experiments_router
from app.api.routes.modeling_tasks import router as modeling_tasks_router
from app.api.routes.platform_runs import router as platform_runs_router
from app.api.routes.v3_runs import router as v3_runs_router
from app.api.routes.training_plans import router as training_plans_router
from app.api.websocket import router as ws_router
from app.services.timeseries_service import resume_unfinished_ts_tasks
from app.services.object_storage import upload_dataset_file
from app.utils.file_utils import generate_unique_filename
from app.utils.storage_paths import to_portable_storage_path

# ---------------------------------------------------------------------------
# Preset tag definitions — seeded once on first startup
# ---------------------------------------------------------------------------
_PRESET_TAGS = [
    # 类别
    {"name": "分类模型",  "dimension": "类别", "color": "blue"},
    {"name": "回归模型",  "dimension": "类别", "color": "blue"},
    {"name": "聚类模型",  "dimension": "类别", "color": "blue"},
    {"name": "时序预测",  "dimension": "类别", "color": "blue"},
    {"name": "异常检测",  "dimension": "类别", "color": "blue"},
    # 规模
    {"name": "轻量级",   "dimension": "规模", "color": "green"},
    {"name": "中等规模", "dimension": "规模", "color": "green"},
    {"name": "大型",     "dimension": "规模", "color": "green"},
    # 目的
    {"name": "实验探索", "dimension": "目的", "color": "orange"},
    {"name": "生产部署", "dimension": "目的", "color": "orange"},
    {"name": "对比基准", "dimension": "目的", "color": "orange"},
    {"name": "数据分析", "dimension": "目的", "color": "orange"},
    # 领域
    {"name": "金融",     "dimension": "领域", "color": "purple"},
    {"name": "医疗",     "dimension": "领域", "color": "purple"},
    {"name": "电商",     "dimension": "领域", "color": "purple"},
    {"name": "工业",     "dimension": "领域", "color": "purple"},
    {"name": "教育",     "dimension": "领域", "color": "purple"},
    {"name": "通用",     "dimension": "领域", "color": "purple"},
    # 数据类型
    {"name": "数值型",   "dimension": "数据类型", "color": "cyan"},
    {"name": "文本型",   "dimension": "数据类型", "color": "cyan"},
    {"name": "图像型",   "dimension": "数据类型", "color": "cyan"},
    {"name": "混合型",   "dimension": "数据类型", "color": "cyan"},
    {"name": "时序型",   "dimension": "数据类型", "color": "cyan"},
]


async def _seed_tag_library() -> None:
    """Insert preset tags on first startup; skip names that already exist.

    Uses a SELECT-then-INSERT pattern so it works with both SQLite and MySQL.
    """
    async with async_session_factory() as db:
        result = await db.execute(select(ModelTagLibrary).limit(1))
        if result.scalar_one_or_none() is not None:
            return  # already seeded

        existing_names_result = await db.execute(select(ModelTagLibrary.name))
        existing_names = {row[0] for row in existing_names_result.fetchall()}

        for tag in _PRESET_TAGS:
            if tag["name"] not in existing_names:
                db.add(ModelTagLibrary(**tag))

        await db.commit()


def _build_columns_info(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    total = len(df)
    return {
        col: {
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "missing_rate": round(df[col].isna().sum() / total, 4) if total else 0.0,
            "unique_count": int(df[col].nunique(dropna=True)),
            "unique_rate": round(df[col].nunique(dropna=True) / total, 4) if total else 0.0,
            "min_class_count": (
                int(df[col].value_counts(dropna=True).min())
                if int(df[col].nunique(dropna=True)) > 0 else 0
            ),
        }
        for col in df.columns
    }


async def _seed_example_datasets() -> None:
    """On first startup, import example CSVs from examples/data/ into the DB."""
    settings = get_settings()
    examples_dir = settings.project_root.parent / "examples" / "data"
    if not examples_dir.exists():
        return

    async with async_session_factory() as db:
        result = await db.execute(select(Dataset).limit(1))
        if result.scalar_one_or_none() is not None:
            return  # already seeded

        for csv_path in sorted(examples_dir.glob("*.csv")):
            content = csv_path.read_bytes()
            unique_name = generate_unique_filename(csv_path.name)
            dest = settings.storage_uploads / unique_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv_path, dest)

            df = pd.read_csv(dest)
            dataset = Dataset(
                name=csv_path.name,
                file_path=to_portable_storage_path(dest),
                file_size=len(content),
                row_count=len(df),
                column_count=len(df.columns),
                columns_info=_build_columns_info(df),
            )
            db.add(dataset)
            await db.flush()
            upload_dataset_file(dataset.id, dest)

        await db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown lifecycle events."""
    settings = get_settings()

    # --- Startup ---
    # A0: fail fast (before any table create / seed) when production is about
    # to boot with development defaults — misconfig must not start silently.
    settings.validate_for_production()
    if settings.is_production and not settings.auth_enabled:
        import logging

        logging.getLogger(__name__).warning(
            "⚠️ 生产环境显式关闭了鉴权（AUTH_ENABLED=false）——平台对公网完全裸奔，请确认这是有意为之。"
        )
    # Ensure storage directories exist
    settings.ensure_storage_dirs()

    # Production schema changes are owned by the deployment-time Alembic
    # command. Local and test environments keep the legacy bootstrap path so
    # existing SQLite workflows remain unchanged.
    if not settings.is_production:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        from app.core.migrations import run_startup_migrations

        await run_startup_migrations(async_engine)
    # V3 Phase 2 — force-import services so their module-level
    # ``register_executor`` calls populate the Scheduler registry before any
    # task dispatch happens.  The router imports already cover training_service
    # and dl_service, but explain_service is only imported lazily inside
    # handlers — so pull it in explicitly here.
    import app.services.training_service  # noqa: F401
    import app.services.dl_service        # noqa: F401
    import app.services.explain_service   # noqa: F401
    await _seed_tag_library()
    await _seed_example_datasets()
    await resume_unfinished_ts_tasks()

    yield

    # --- Shutdown ---
    await async_engine.dispose()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="ML Training Platform",
        version="3.3.0",
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

    # Auth guard (single-admin). Everything under /api and /inference needs a
    # bearer token when AUTH_ENABLED=true; login itself is the only exception.
    # WebSocket routes are guarded separately at the handshake (?token=).
    _AUTH_PUBLIC_PATHS = {"/api/auth/login"}

    @app.middleware("http")
    async def auth_guard(request, call_next):
        path = request.url.path
        guarded = path.startswith("/api") or path.startswith("/inference")
        if guarded and path not in _AUTH_PUBLIC_PATHS and request.method != "OPTIONS":
            if not request_is_authorized(request.headers.get("authorization")):
                return JSONResponse(status_code=401, content={"detail": "未登录或登录已过期"})
        return await call_next(request)

    # Register routers under /api prefix
    app.include_router(auth_router, prefix="/api")          # → /api/auth/...
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
    app.include_router(platform_tasks_router, prefix="/api")        # → /api/platform/tasks
    app.include_router(platform_experiments_router, prefix="/api")  # → /api/platform/experiments
    app.include_router(modeling_tasks_router, prefix="/api")        # → /api/v3/tasks
    app.include_router(v3_runs_router,        prefix="/api")        # → /api/v3/runs  (flat cross-task list)
    app.include_router(platform_runs_router, prefix="/api")         # → /api/platform/runs/{run_id}/inspector
    app.include_router(training_plans_router, prefix="/api")        # → /api/platform/training-plans
    app.include_router(ws_router)

    # ---- Root-level endpoints ----

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {
            "status": "ok",
            "version": "3.3.0",
            "environment": get_settings().environment,
        }

    return app


app = create_app()
