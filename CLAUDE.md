# 傻子也会训练模型 — ML Training Platform

## Project Structure

```
ml_platform/              # FastAPI backend (port 8000)
  app/
    api/routes/           # REST endpoints: data, training, logs, viz, models, experiments
    api/websocket.py      # WebSocket: /ws/training/{id}, /ws/logs/{id}
    core/                 # trainer.py (6 ML trainers), logger.py (per-task logging)
    models/database.py    # SQLAlchemy async ORM (Dataset, TrainingTask, TrainingLog)
    models/schemas.py     # Pydantic v2 request/response schemas
    services/             # Business logic: training, prediction, data, viz, log
    config.py             # Settings from .env (ports, paths, DB URL)
    main.py               # FastAPI app factory, lifespan, router registration
  storage/                # [gitignored] Runtime: uploads/, models/, logs/, *.db
  mlruns/                 # [gitignored] MLflow experiment artifacts
  tests/                  # Backend unit tests (pytest)
  requirements.txt        # Python dependencies

ml_platform_web/          # React frontend (port 3000, Vite dev server)
  src/
    pages/                # Dashboard, DataManagement, TrainingConfig, TrainingMonitor,
                          # Results, ModelManagement, Settings
    services/api.js       # Axios client (base: /api, proxy to :8000)
    components/layout/    # Header, Sidebar
    utils/formatters.js   # Metric/date/byte formatting
    styles/global.css     # Tailwind + Ant Design overrides
  package.json            # React + Ant Design + ECharts + Redux Toolkit

tests/                    # Playwright E2E tests (.spec.js)
examples/data/            # Sample datasets for testing
scripts/                  # Dev startup/stop (PowerShell + Python)
doc/                      # Design documents and project TODO tracker
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + SQLAlchemy async + aiosqlite |
| ML | scikit-learn + XGBoost + LightGBM + SHAP |
| Experiments | MLflow (SQLite backend) |
| Frontend | React 18 + Vite + Ant Design 5 + ECharts |
| State | Redux Toolkit |
| Real-time | WebSocket + in-memory EventBus |
| Testing | Playwright (E2E) + pytest (unit) |

## Key Commands

```bash
# Backend
cd ml_platform && uvicorn app.main:app --reload --port 8000

# Frontend
cd ml_platform_web && npm run dev

# Both (PowerShell)
./scripts/start-dev.ps1
./scripts/stop-dev.ps1

# E2E Tests (auto-starts backend + frontend)
npx playwright test

# Backend unit tests
cd ml_platform && python -m pytest tests/

# API docs
open http://localhost:8000/docs
```

## Conventions

- API routes: all under `/api/` prefix
- Schemas: Pydantic v2 with `ConfigDict(from_attributes=True)`
- DB sessions: `async_session_factory()` with auto-commit/rollback
- Training: background coroutine → ThreadPoolExecutor (4 workers)
- Model files: `storage/models/{task_id}.joblib`
- Log files: `storage/logs/{task_id}.log` + `{task_id}_metrics.json`
- File naming: `{uuid_12}-{original_name}` for uploads
- UI language: simplified Chinese
- Git branch: `dev` for development, `main` for releases
