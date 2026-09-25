# Evaluation Integrity Phase 6 Task Finalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make sealed hold-out evaluation an explicit, task-level, one-version action and prevent later experiment batches from turning the hold-out into another selection surface.

**Architecture:** Store a reserved finalization state machine in `ModelingTask.config`, persist a claim before model evaluation, and serialize experiment dispatch against the same task row. Reuse the phase-5 winner evaluator for artifact/content evaluation, remove its automatic batch hook, and expose one shared finalization control through `ModelComparison`.

**Tech Stack:** FastAPI, SQLAlchemy async, React 18, Ant Design 5, Vitest, pytest.

## Global Constraints

- No schema migration; use `ModelingTask.config._final_evaluation` and existing run audit JSON.
- Classic ML selection-only runs only; DL returns an explicit unsupported error.
- Finalized, evaluating and failed tasks reject new experiment batches.
- Repeated finalization of the same version must not read the hold-out again.
- Preserve selection-based ranking and standalone training behavior.
- Preserve unrelated working-tree changes.

---

### Task 1: Task finalization state contract

**Files:**
- Modify: `ml_platform/app/services/final_evaluation_service.py`
- Modify: `ml_platform/app/services/modeling_task_service.py`
- Test: `ml_platform/tests/v3/test_final_evaluation_service.py`

**Interfaces:**
- Produces: `task_final_evaluation_state(task) -> dict`.
- Produces: `finalize_task_winner(db, modeling_task_id) -> dict`.
- Reuses: `evaluate_task_winner(db, modeling_task_id)` for run-level evaluation.

- [x] Add failing tests for OPEN defaults, active-run/experiment rejection, claim persistence, FINALIZED idempotency, stale EVALUATING conflict, DL/legacy rejection and FAILED retry state.
- [x] Implement the reserved JSON state serializer with bounded error payloads and version 1.
- [x] Lock the task row, validate the global winner and active-run count, persist a UUID claim, then commit before evaluation.
- [x] Re-lock after evaluation, verify claim ownership and write FINALIZED/FAILED audit data.
- [x] Expose `final_evaluation` from `serialize_modeling_task` without exposing the reserved storage key as a second public contract.

### Task 2: Explicit API and dispatch lock

**Files:**
- Modify: `ml_platform/app/api/routes/modeling_tasks.py`
- Modify: `ml_platform/app/services/tuning_service.py`
- Modify: `ml_platform/tests/v3/test_tuning_service.py`
- Create: `ml_platform/tests/v3/test_task_finalization_route.py`

**Interfaces:**
- Produces: `POST /api/v3/tasks/{task_id}/final-evaluation`.
- Produces: `_lock_task_for_experiment_dispatch(db, task_id) -> ModelingTask`.

- [x] Add a failing route test for the POST response and GET task DTO state.
- [x] Add failing dispatch tests proving EVALUATING, FINALIZED and FAILED tasks reject new work while OPEN accepts it.
- [x] Add the POST route and map service validation to actionable 409/422 responses.
- [x] Put the dispatch guard at the shared tuning-service boundary used by single, bulk and code-config submissions.
- [x] Remove `evaluate_task_winner` from `_finalise_batch`; keep summary refresh and SHAP scheduling unchanged.
- [x] Replace the old automatic-call-order test with a test proving batch finalization never opens the hold-out.
- [x] Freeze objective metric/direction once finalization starts so the sealed winner cannot diverge from task semantics.
- [x] Serialize finalization and dispatch with a shared process-local lifecycle lock, supplementing database row locks for the single-process SQLite runtime.

### Task 3: Shared finalization UI

**Files:**
- Modify: `ml_platform_web/src/services/api.js`
- Modify: `ml_platform_web/src/utils/comparison.js`
- Modify: `ml_platform_web/src/utils/comparison.test.js`
- Modify: `ml_platform_web/src/components/workbench/ModelComparison.jsx`
- Modify: `ml_platform_web/src/pages/ModelingWorkflow.jsx`

**Interfaces:**
- Produces: `modelingTaskApi.finalize(taskId)`.
- Produces: `buildFinalizationVM(task, bestRun) -> {state, disabled, reason, finalValue}`.
- Consumes: `ModelComparison.onRefresh`, which must reload both task and runs after finalization.

- [x] Add failing unit tests for OPEN, active-run, DL, EVALUATING, FINALIZED and FAILED view states.
- [x] Add the API client and pure finalization view-model helper.
- [x] Render confirmation, loading, finalized metric and retry states in `ModelComparison`.
- [x] Change the workflow refresh callback to reload task and runs together; the detail page already uses `refreshAll`.
- [x] Disable new-batch controls after a task leaves OPEN, matching the backend guard.
- [x] Refresh task state on both success and failure so FAILED retry/error state appears immediately.

### Task 4: Verification and roadmap

- [x] Run phase-6 service, route, tuning and frontend unit tests.
- [x] Request an independent read-only review of lifecycle, locking and retry semantics; fix important findings.
- [x] Run the complete backend suite, frontend unit tests, scoped lint, production build and `git diff --check`.
- [x] Update `doc/优化蓝图.md` and this plan with the exact completed scope; keep B1 open only for DL if task-level sealing is complete.

## Review Corrections

- [x] Reject finalization while a parent experiment remains `RUNNING`, including Bayesian gaps between sequential trials.
- [x] Never auto-take over a stale claim; strict one-time hold-out access takes precedence over unattended recovery.
- [x] Share a process-local lifecycle lock between dispatch and finalization for the supported single-process SQLite runtime.
- [x] Refresh task state after failed frontend requests so retry/error state is immediately visible.
