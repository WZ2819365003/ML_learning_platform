# Evaluation Integrity Phase 5 Sealed Final Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent ML tuning trials from observing the outer hold-out and evaluate only the current global selection winner once per evaluation version.

**Architecture:** Persist `evaluation_mode="selection"` in each ML tuning run's `search_meta`. The existing `train` executor resolves that context through `platform_task_id` and passes no hold-out to the trainer, while preserving the deterministic outer split for later use. A focused final-evaluation service loads the winning run's saved artifact, reproduces the split, computes requested metrics on the untouched hold-out, and writes only `final_test_*` plus audit metadata; ranking continues to use selection CV values.

**Tech Stack:** Python 3, FastAPI services, SQLAlchemy async, pandas, scikit-learn, joblib, pytest.

## Global Constraints

- Do not add or migrate database columns; use existing `ExperimentRun.search_meta` JSON.
- Keep standalone classic training behavior unchanged: it still evaluates its hold-out immediately.
- Apply selection-only execution only to ML runs created by V3 tuning; DL baseline retains its existing validation behavior and keeps B1 partially open.
- Never rank, tune, or trigger Top-K actions by `final_test_*`.
- Final evaluation must be idempotent for the same split/evaluation version.
- Preserve unrelated working-tree changes.

---

### Task 1: Selection-only training context

**Files:**
- Modify: `ml_platform/app/services/tuning_service.py`
- Modify: `ml_platform/app/services/training_service.py`
- Test: `ml_platform/tests/test_training_service_prepare_data.py`
- Test: `ml_platform/tests/v3/test_tuning_service.py`

**Interfaces:**
- Produces: `_resolve_evaluation_mode(platform_task_id) -> "selection" | "standard"`.
- Consumes: `ExperimentRun.search_meta` and existing deterministic `_prepare_data`.

- [x] Add failing tests proving ML tuning runs persist `evaluation_mode="selection"` while DL runs do not.
- [x] Add a failing training test proving selection mode calls the trainer with `X_val=None` and emits no raw/final-test objective metric.
- [x] Resolve execution mode from the linked run and pass `evaluation_mode` through `_run_training_sync` and `_run_training_sync_inner`.
- [x] Keep the raw split unchanged, but pass `None, None` as trainer validation inputs in selection mode.

### Task 2: Versioned final evaluation service

**Files:**
- Create: `ml_platform/app/services/final_evaluation_service.py`
- Create: `ml_platform/tests/v3/test_final_evaluation_service.py`

**Interfaces:**
- Produces: `evaluate_task_winner(db, modeling_task_id) -> dict`.
- Consumes: `task_leaderboard`, `PlatformTask.payload_ref`, `TrainingTask.model_path`, `_prepare_data`, and persisted tabular artifacts.

- [x] Add failing tests proving only the selection winner is evaluated, metrics are written as `final_test_*`, and `objective_value`/selection fields remain unchanged.
- [x] Add a failing idempotency test proving the same `split_seed=42`, `test_size`, dataset and artifact version are evaluated once.
- [x] Implement ML artifact loading, deterministic hold-out prediction, classification/regression metric calculation, and transactional run/search-meta write-back.
- [x] Offload dataset hashing, artifact loading and prediction from the async event-loop thread.
- [x] Return a structured `evaluated | skipped | unsupported` result; treat a DL winner as unsupported without mutating its metrics.

### Task 3: Batch finalisation integration

**Files:**
- Modify: `ml_platform/app/services/tuning_service.py`
- Modify: `ml_platform/tests/v3/test_tuning_service.py`

**Interfaces:**
- Consumes: `evaluate_task_winner` from Task 2.
- Produces: one final-evaluation attempt after selection ranking is refreshed.

- [x] Add a failing test proving `_finalise_batch` invokes final evaluation only after all runs are terminal and task summary is refreshed.
- [x] Invoke final evaluation before SHAP scheduling; failures are logged and do not change successful training status.
- [x] Refresh task summary again only if final evaluation changed run metrics; selection winner must remain unchanged.

### Task 4: Verification and roadmap

- [x] Run sealed-evaluation, tuning, training, modeling and artifact tests.
- [x] Run the complete backend suite, frontend unit tests, scoped lint and production build.
- [x] Run `git diff --check` and inspect the scoped diff.
- [x] Update `doc/优化蓝图.md`: record the completed ML isolation/evaluator work and keep B1 open for task-level sealing plus DL selection/final isolation.

### Review fixes

- [x] Evaluate versioned tabular artifacts against encoded targets so boolean/string labels produce valid final metrics.
- [x] Make the evaluation fingerprint storage-path independent and include the canonical requested metric set.
- [x] Refresh and lock the winner row after long-running evaluation work before merging JSON metrics/audit metadata.

**Remaining method boundary:** Batch completion evaluates the current global winner, but a later batch can produce another winner and therefore another hold-out evaluation. A strict task-level seal needs an explicit finalize/version lifecycle and concurrency guard. B1 remains open for that lifecycle and for DL selection/final isolation.
