# DL Explainability And Persistence Iteration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make cloud-trained DL runs durable, inspectable, downloadable, and explainable with SHAP across MLP-DL, LSTM, CNN-1D, and Transformer.

**Architecture:** Persist `/app/storage` in a named Docker volume before changing application behavior. Extend the existing domain-task resolution path to return `DLTrainingTask` context, then adapt each PyTorch trainer to the existing Kernel SHAP payload contract instead of creating a second explanation API. Keep epoch visualization in `RunInspector`; route DL-only diagnostic charts to the existing DL results page and surface actionable errors for missing ML artifacts.

**Tech Stack:** Docker Compose, FastAPI, SQLAlchemy async, PyTorch 2.6 CPU, SHAP 0.49, React 18, Ant Design 5, Vitest, pytest, Playwright.

**Status:** Completed on 2026-07-17. Local regression, focused Playwright coverage,
cloud deployment, four-model DL SHAP verification, and repeated-container
persistence verification all passed.

## Global Constraints

- Preserve existing ML SHAP behavior and response shape.
- Do not expose raw credentials, bearer tokens, or MinIO secrets in logs or tests.
- Kernel SHAP must cap background and explained samples so CPU-only production remains responsive.
- DL preprocessing must replay the persisted `.preprocessor.joblib` and trainer scaler; never refit preprocessing during explanation.
- Existing unrelated worktree changes under `.claude/`, `.deploy_tmp/`, and `screenshots/` must remain untouched.
- Migrate the current cloud storage backup into the persistent volume before recreating the backend container.

---

### Task 1: Persistent Backend Runtime Storage

**Files:**
- Modify: `docker/docker-compose.yml`
- Create: `ml_platform/tests/test_docker_compose_storage_contract.py`

**Interfaces:**
- Consumes: backend runtime path `/app/storage` from `app.config.Settings`.
- Produces: Docker volume `ml_platform_backend_storage` mounted at `/app/storage`.

- [x] **Step 1: Write the failing Compose contract test**

```python
from pathlib import Path
import yaml


def test_backend_storage_is_persistent():
    compose = yaml.safe_load(
        (Path(__file__).parents[2] / "docker" / "docker-compose.yml").read_text()
    )
    assert "ml_backend_storage:/app/storage" in compose["services"]["backend"]["volumes"]
    assert compose["volumes"]["ml_backend_storage"]["name"] == "ml_platform_backend_storage"
```

- [x] **Step 2: Run the test and verify RED**

Run: `cd ml_platform && python -m pytest tests/test_docker_compose_storage_contract.py -q`

Expected: FAIL because `backend.volumes` and `ml_backend_storage` do not exist.

- [x] **Step 3: Add the stable named volume**

Add to `backend`:

```yaml
    volumes:
      - ml_backend_storage:/app/storage
```

Add to the top-level `volumes` block:

```yaml
  ml_backend_storage:
    name: ml_platform_backend_storage
```

- [x] **Step 4: Verify Compose and the focused test**

Run: `docker compose -f docker/docker-compose.yml config --quiet`

Run: `cd ml_platform && python -m pytest tests/test_docker_compose_storage_contract.py -q`

Expected: both exit 0.

---

### Task 2: Unified DL Domain Context And Download

**Files:**
- Modify: `ml_platform/app/services/resolver.py`
- Modify: `ml_platform/app/api/routes/platform_runs.py`
- Modify: `ml_platform/app/api/routes/model_mgmt.py`
- Modify: `ml_platform/tests/v3/test_run_inspector.py`
- Create: `ml_platform/tests/v3/test_dl_domain_routes.py`

**Interfaces:**
- Consumes: `PlatformTask.payload_ref = "dl_train:{domain_task_id}"` and `DLTrainingTask.model_path`.
- Produces: `resolve_task_and_dataset()` support for DL tasks; Inspector `training_task.family == "dl"`; unified `/api/models/{domain_task_id}/download` for `.joblib` and `.pt`.

- [x] **Step 1: Add failing resolver, Inspector, and download tests**

Seed a successful `DLTrainingTask`, a `PlatformTask(kind="dl_train")`, and an `ExperimentRun`. Assert:

```python
resolved, dataset = await resolve_task_and_dataset(run.id, db)
assert resolved.id == dl_task.id
assert resolved.task_type == "classification"

response = await client.get(f"/api/platform/runs/{run.id}/inspector")
assert response.json()["training_task"]["family"] == "dl"
assert response.json()["training_task"]["task_type"] == "classification"

download = await client.get(f"/api/models/{dl_task.id}/download")
assert download.status_code == 200
assert "filename=mlp_dl_" in download.headers["content-disposition"]
```

- [x] **Step 2: Run focused tests and verify RED**

Run: `cd ml_platform && python -m pytest tests/v3/test_run_inspector.py tests/v3/test_dl_domain_routes.py -q`

Expected: DL resolution/Inspector/download assertions fail while existing ML Inspector tests remain green.

- [x] **Step 3: Extend domain resolution and serialization**

Add direct `DLTrainingTask` resolution and follow `dl_train:` payload references before the existing V3 `.joblib` synthesis. Add `_serialize_dl_training_task()` with this stable shape:

```python
{
    "id": task.id,
    "family": "dl",
    "model_type": task.model_type,
    "task_type": task.task_type,
    "status": task.status,
    "progress": task.progress,
    "current_epoch": task.current_epoch,
    "total_epochs": task.total_epochs,
    "model_path": task.model_path,
    "result_metrics": task.result_metrics or {},
    "dataset": {"id": dataset.id, "name": dataset.name, ...},
}
```

The download route must query `TrainingTask` first, then `DLTrainingTask`, preserve the existing ML filename, and use `.pt` for DL.

- [x] **Step 4: Run focused tests and verify GREEN**

Run: `cd ml_platform && python -m pytest tests/v3/test_run_inspector.py tests/v3/test_dl_domain_routes.py -q`

Expected: all pass.

---

### Task 3: Kernel SHAP Adapter For PyTorch DL Runs

**Files:**
- Create: `ml_platform/app/services/dl_shap_adapter.py`
- Modify: `ml_platform/app/services/shap_service.py`
- Modify: `ml_platform/app/services/tuning_service.py`
- Modify: `ml_platform/tests/v3/test_shap_service.py`
- Modify: `ml_platform/tests/v3/test_tuning_service.py`

**Interfaces:**
- Consumes: resolved `DLTrainingTask`, `get_dl_trainer(model_type)`, `load_for_inference(path)`, persisted preprocessing/scaler sidecars.
- Produces: `build_dl_shap_context(task, dataset, max_background, max_samples) -> DLShapContext`; existing `compute_shap_summary()` payload with `method="kernel"` for DL.

- [x] **Step 1: Write failing DL adapter and SHAP service tests**

Create a tiny two-class MLP checkpoint with persisted preprocessing. Assert:

```python
context = build_dl_shap_context(dl_task, dataset, max_background=12, max_samples=6)
assert context.task_kind == "classification"
assert context.X_background.shape == (12, 3)
assert context.X_sample.shape == (6, 3)
assert context.model.predict_proba(context.X_sample).shape == (6, 2)

payload = await compute_shap_summary(run.id, db, max_samples=6)
assert payload["method"] == "kernel"
assert payload["task_kind"] == "classification"
assert payload["feature_names"] == ["a", "b", "c"]
assert len(payload["shap_values"]) == 6
```

Also assert `_schedule_shap_for_top_runs()` accepts a successful `dl_train` candidate instead of filtering it out.

- [x] **Step 2: Run focused tests and verify RED**

Run: `cd ml_platform && python -m pytest tests/v3/test_shap_service.py tests/v3/test_tuning_service.py -q`

Expected: import/behavior failures because the adapter and DL scheduling do not exist.

- [x] **Step 3: Implement the bounded DL inference adapter**

`DLShapContext` must hold `model`, `X_background`, `X_sample`, `y_sample`, `feature_names`, and `task_kind`. Classification adapters expose `predict_proba`; regression adapters expose `predict`. Both accept NumPy arrays or DataFrames and delegate to `BaseDLTrainer.predict()` after the persisted preprocessing transform has already run.

Use the canonical outer split, but replay the saved preprocessor:

```python
raw_train, raw_holdout, _, raw_y_holdout, task_kind = _outer_split(...)
X_background = artifact.transform_features(raw_train.sample(...))
X_sample = artifact.transform_features(raw_holdout.sample(...))
```

Do not call `fit_dl_preprocessing_artifact()`.

- [x] **Step 4: Branch the existing SHAP ladder by resolved family**

For DL tasks, call `_compute_kernel()` with the adapter and bounded arrays, then serialize through `_build_payload()`. Keep ML Tree/Kernel/Permutation behavior unchanged. Add `dl_train` to automatic explanation candidates.

- [x] **Step 5: Run focused tests and verify GREEN**

Run: `cd ml_platform && python -m pytest tests/v3/test_shap_service.py tests/v3/test_tuning_service.py -q`

Expected: all pass, including one real bounded Kernel SHAP computation.

---

### Task 4: DL-Aware Run Inspector And Actionable Visualization Errors

**Files:**
- Modify: `ml_platform_web/src/components/workbench/RunInspector.jsx`
- Modify: `ml_platform_web/src/components/workbench/TrainingViz.jsx`
- Create: `ml_platform_web/src/components/workbench/TrainingViz.test.js`
- Modify: `tests/v3-results-training-tab.spec.js`

**Interfaces:**
- Consumes: Inspector `training_task.family`, `training_task.task_type`, DL domain task ID, existing `/dl/results?taskId=` page.
- Produces: correct classification/regression epoch chart mode; no false “未绑定训练任务” for DL; visible error summary when ML visualization endpoints fail.

- [x] **Step 1: Write failing settled-request helper tests**

Export a pure helper `settleVizRequest(label, promise)` and assert:

```javascript
await expect(settleVizRequest('混淆矩阵', Promise.reject({ response: { data: { detail: 'Model file not found' } } })))
  .resolves.toEqual({ data: null, error: '混淆矩阵：Model file not found' })
```

Update the Playwright contract so a DL Inspector training-visualization tab expects an epoch chart plus a “查看完整 DL 结果” link, not the generic ML visualization grid.

- [x] **Step 2: Run tests and verify RED**

Run: `cd ml_platform_web && npm run test:unit -- TrainingViz.test.js`

Run: `npx playwright test tests/v3-results-training-tab.spec.js --project=chromium`

Expected: helper import/expectation and DL Inspector contract fail.

- [x] **Step 3: Implement DL-aware rendering and error aggregation**

Derive task kind from `training_task.task_type`, falling back to `run.params.task_type`. For `family === "dl"`, render `TrainingHistoryChart` and a link to `/dl/results?taskId={training_task.id}`; do not invoke generic `/api/viz/*` endpoints. For ML, aggregate endpoint errors into an `Alert` while preserving any charts that did load.

- [x] **Step 4: Run frontend verification**

Run: `cd ml_platform_web && npm run test:unit`

Run: `cd ml_platform_web && npm run lint`

Run: `cd ml_platform_web && npm run build`

Expected: all exit 0.

---

### Task 5: Full Regression, Cloud Migration, And Production Verification

**Files:**
- Modify on server: `/home/opsadmin/ml_platform/docker/docker-compose.yml`
- Consume backup: `/home/opsadmin/ml_platform/runtime/storage-backup-20260717`

**Interfaces:**
- Consumes: Tasks 1-4 artifacts and the existing cloud backup.
- Produces: recreated backend with durable storage and verified DL SHAP/download/visualization behavior.

- [x] **Step 1: Run complete local regression**

Run: `cd ml_platform && python -m pytest tests/ -q`

Run: `cd ml_platform_web && npm run test:unit && npm run lint && npm run build`

Expected: backend and frontend suites exit 0.

- [x] **Step 2: Back up and seed the stable Docker volume**

On the cloud server:

```bash
docker volume create ml_platform_backend_storage
docker run --rm \
  -v ml_platform_backend_storage:/dest \
  -v /home/opsadmin/ml_platform/runtime/storage-backup-20260717:/src:ro \
  alpine sh -c 'cp -a /src/. /dest/'
```

Verify the uploaded diabetes CSV and four new `dl_*.pt` files exist in the volume before recreation.

- [x] **Step 3: Deploy and recreate only affected services**

Update the active Compose file with the named volume, deploy source changes, rebuild backend/frontend, then recreate them without touching MySQL, Redis, or MinIO.

- [x] **Step 4: Verify production persistence and APIs**

Assert:

```text
docker inspect ml_platform_backend -> /app/storage mount present
GET /health -> 200
GET /api/dl/{id}/status -> SUCCESS
GET /api/dl/{id}/epochs -> 6 rows
GET /api/models/{id}/download -> 200, .pt attachment
POST explain -> QUEUED or already_computed
GET explain -> method=kernel, feature_count=8
```

- [x] **Step 5: Verify production UI**

Open the cloud V3 task, verify four-model comparison, open the best DL Run, confirm the epoch canvas is non-blank, the false unbound-task empty state is absent, the full DL results link works, and the SHAP tab renders feature importance after computation.

- [x] **Step 6: Recreate backend once more and verify persistence**

Recreate only `backend`, then repeat dataset, epoch, download, and SHAP checks. This is the acceptance test that prevents another database/filesystem split.

---

## Plan Self-Review

- Coverage: persistent data, DL domain resolution, download, SHAP, Inspector behavior, error visibility, local regression, and cloud recreation are each mapped to a task.
- Scope: Deep/Gradient explainers and extraction of all DL chart builders are intentionally deferred; Kernel SHAP and the existing DL results page satisfy this iteration.
- Type consistency: DL context uses `task_type` for classification/regression and `family` only for runtime routing.
- Safety: migration copies the existing backup into a new named volume before any backend recreation; stateful MySQL/MinIO services are not recreated.
