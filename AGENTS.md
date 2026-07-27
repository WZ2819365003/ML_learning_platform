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

## User Preferences For This Platform

### AI Task Reports

- Reports are single-task research reports, not generic model reports. They must help the reader understand the task objective, executed process, model behavior, and final effect.
- The main evidence scope should stay focused: dataset overview, parameter settings, training process data, and model evaluation. Avoid adding unrelated sections just to make the report longer.
- Use formal hierarchy: `第一章`, `1.1`, `1.1.1`. Start important paragraphs with a clear conclusion sentence, then explain the evidence in natural language.
- Keep report language professional and non-colloquial. Avoid mechanically listing JSON keys or metrics without interpretation.
- Put charts and tables between relevant paragraphs. Do not stack every table or chart at the top.
- Use ECharts for dense process evidence such as loss curves, prediction curves, training metric trends, or trial-level trajectories.
- Use tables for low-density comparison facts such as final accuracy, F1, ROC-AUC, parameter settings, and dataset field summaries.
- Do not render run success rate as a major chart. It can be mentioned briefly as context when all runs completed or failures affect interpretation.
- Translate metric meaning for readers while keeping source keys visible where helpful. Example: write "最终测试准确率（final_test_accuracy）为 0.9790".
- Map dataset fields to readable Chinese labels when possible, while keeping original names visible. Example: `Torque [Nm]` can be shown as `扭矩（Torque [Nm]）`.
- Do not translate model identifiers into awkward Chinese names. Preserve names such as `random_forest`, `logistic_regression`, `XGBoost`, `LightGBM`, and `ARIMA`.
- Explain input and output explicitly: input dataset, target column, task type, selected models/strategies, model predictions, evaluation metrics, generated report archive.
- AI reports must be archived and viewable from the task page. The report tab should prefer the latest AI archive, with legacy Markdown as fallback.
- When a new AI report is generated from the modal, the report tab should update dynamically without requiring page refresh.

### Modeling Workbench UX

- The task detail page should use progressive disclosure: task summary and key actions first, tabs for overview/experiments/model comparison/report, then deeper drill-down.
- The report tab should render the rich AI report directly in the page area, not only inside a modal.
- Use a large modal for generated report preview, but keep the same report reader component shared between modal and inline report display.
- Display source state clearly, such as `AI 报告` and archive id, so users know whether they are reading generated AI content or the legacy basic report.
- Keep information density balanced. Metric strips are acceptable for high-level scanning, but detailed evidence should live in the corresponding chapter.
- Avoid all-table pages. Charts, tables, and prose should each be used where they are strongest.
- Validate important UI changes with Playwright screenshots on the real local page.

### Training And Tuning Semantics

- Do not conflate tuning strategy with a detached later module. For future tuning work, strategy should be configured when setting up mixed training or single-model training.
- If baseline, grid search, and Bayesian search are all selected, interpret it as `n x 3` groups of model runs, where `n` is the number of selected models.
- This tuning semantics note is a future design preference; do not refactor it unless the user explicitly asks.
