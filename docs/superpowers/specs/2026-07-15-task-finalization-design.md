# Task-Level Final Evaluation Design

## Goal

Replace automatic per-batch hold-out evaluation with one explicit, task-level finalization action. A finalized task exposes the sealed hold-out to exactly one classic-ML selection winner for the current evaluation version and then rejects further experiment dispatch.

## Scope

- Classic ML V3 runs with `search_meta.evaluation_mode="selection"` only.
- No database migration. Task-level lifecycle metadata is stored under the reserved `ModelingTask.config._final_evaluation` key and exposed as top-level `final_evaluation` in task DTOs.
- DL remains unsupported and B1 remains open for DL selection/final isolation.
- Existing run-level `search_meta.final_evaluation` remains the detailed artifact/content audit.

## Lifecycle

The public task state is `OPEN | EVALUATING | FINALIZED | FAILED` with `version=1`.

1. `OPEN`: experiments can be launched. Finalization requires at least one successful selection-only ML run and no active runs.
2. `EVALUATING`: an explicit request has claimed the task. New experiment dispatch is rejected. A concurrent finalization request returns conflict instead of evaluating again.
3. `FINALIZED`: the winner run id, evaluation id, final metrics and timestamps are immutable for this version. Repeated finalization calls return the stored result without reading the hold-out again. New experiment dispatch is rejected.
4. `FAILED`: the claim records a bounded error string. New experiments remain blocked, but an explicit finalization retry may replace the failed claim.

The claim is persisted before file/model work. An `EVALUATING` claim older than 30 minutes is reported as stale but is never taken over automatically, because the original evaluator may still be running. Recovery requires checking the original request/run audit first; a run-level `already_evaluated` result remains recoverable through a `FAILED` retry.

## API

- `POST /api/v3/tasks/{task_id}/final-evaluation`
  - `200`: newly finalized or already finalized (idempotent).
  - `409`: active runs, live non-stale claim, or task locked against new work.
  - `422`: no successful run, DL winner, or winner was not trained selection-only.
- Existing `GET /api/v3/tasks/{task_id}` adds `final_evaluation`.

The endpoint returns `{status, final_evaluation}`. The task DTO always returns an `OPEN` default when no claim exists.

## Dispatch Guard

All experiment creation paths pass through tuning-service dispatch. Before creating an experiment, dispatch locks the `ModelingTask` row and rejects `EVALUATING`, `FINALIZED` and `FAILED` states. Finalization also rejects any `RUNNING` experiment, covering the gap between sequential Bayesian trials when no child Run is active. A shared process-local lifecycle lock serializes dispatch/finalization for the default single-process SQLite runtime; database row locks provide the cross-process boundary on databases that support `SELECT ... FOR UPDATE`. Multi-worker SQLite is outside the supported concurrency model.

## Frontend

`ModelComparison` owns the shared finalization control used by both `ModelingWorkflow` and `ModelingTaskDetail`:

- `OPEN`: show `确认最终模型`; confirmation text states that the hold-out is opened once and the task stops accepting new batches.
- `EVALUATING`: show a loading state and disable the action.
- `FINALIZED`: show the confirmed run and final metric; no second action.
- `FAILED`: show a compact error and a retry action.
- Active runs or a DL winner disable the action with a reason.

On success the component calls its existing refresh callback so task and leaderboard data are reloaded in both hosts.

## Failure And Concurrency Rules

- The claim token must still match before the task is marked `FINALIZED` or `FAILED`.
- A stale claim is not replaced automatically; strict one-time hold-out access takes priority over unattended takeover.
- Run JSON is refreshed under lock before final metrics are merged, preserving the phase-5 concurrency fix.
- A run-level `already_evaluated` result is treated as successful recovery when its audit matches the claimed winner.
- Finalization errors never change run or experiment success states.

## Tests

- Service tests: explicit-only behavior, active-run rejection, DL/legacy rejection, finalized idempotency, stale-claim takeover, and persisted failure state.
- Dispatch tests: every sealed state rejects a new batch; `OPEN` remains unchanged.
- Route test: task DTO and POST contract.
- Frontend unit tests: pure finalization view-model states and disable reasons.
- Existing full backend, frontend unit, scoped lint and production build remain green.
