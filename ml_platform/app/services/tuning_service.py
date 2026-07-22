"""
V3 Tuning Engine — baseline / grid_search / bayesian_search.

Responsibilities
----------------
1. Expand a user-submitted experiment batch into concrete training trials
   (one ``ExperimentRun`` per trial).
2. Launch them through the existing platform task pipeline so they show up
   in TaskCenter the same way as a manual training run.
3. Write trial metadata (``trial_no``, ``search_meta``, ``source_experiment_type``)
   so the Run Inspector can trace every run back to its tuning strategy.
4. Drive Optuna studies for ``bayesian_search`` — V1 is sequential (one
   trial at a time) so the TPE sampler always sees the latest result.

Dependencies:
  - sklearn ParameterGrid for cartesian expansion
  - optuna (TPE sampler, in-memory study) for Bayesian search
  - training_service.create_training_task_record + _run_training_sync_by_id
  - modeling_task_service.refresh_task_summary for best-run roll-up

Design notes
------------
- baseline & grid_search submit persisted runs through the per-kind scheduler.
- bayesian_search runs one trial at a time because Optuna's TPE needs the
  last result before suggesting the next set of hyperparameters. Its outer
  in-process orchestrator remains until M2d moves study state to RDBStorage.
- ``search_space`` shape:
    grid_search:
      {"<model_type>": {"param_name": [v1, v2, ...], ...}, ...}
    bayesian_search:
      {"<model_type>": {"param_name": {"type": "float", "low": ..., "high": ...}, ...}}
  Missing models fall back to ``registry/tuning_spaces.yaml`` defaults.
- ``budget_config``:
    max_trials      (grid/bayesian): cap on how many runs per model
    timeout_minutes (future)        : hard walltime
    random_state                    : seed used everywhere for reproducibility
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from itertools import product
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dl_registry import build_default_dl_config, get_dl_model_spec
from app.core.evaluation_metrics import resolve_objective_metrics
from app.core.model_registry import resolve_model_family
from app.models.database import (
    DLTrainingTask,
    ExperimentRun,
    ExperimentRunLog,
    ModelingTask,
    PlatformTask,
    PlatformExperiment,
    TrainingLog,
    async_session_factory,
)
from app.scheduler.task_runner import (
    register_domain_task,
    update_platform_task_status,
)
from app.scheduler.scheduler import get_scheduler
from app.services.modeling_task_service import (
    load_tuning_spaces,
    refresh_task_summary,
    serialize_experiment,
    task_final_evaluation_state,
)
from app.services.training_service import create_training_task_record
from app.services.task_lifecycle_lock import task_lifecycle_guard

logger = logging.getLogger(__name__)


_REGRESSION_METRICS = {"rmse", "mae", "mse", "mape", "r2"}

_VALID_BAYESIAN_DIST_TYPES = {"float", "int", "categorical"}


# ---------------------------------------------------------------------------
# Search-space validation — catches shape mistakes from the workbench UI
# before we spend cycles persisting PENDING trials that can never run.
# ---------------------------------------------------------------------------

def _validate_search_space(
    strategy_type: str,
    search_space: dict[str, Any] | None,
    selected_models: list[str],
) -> None:
    """Reject malformed search_space payloads with an actionable 422.

    Shape contracts (must match what the expanders in this module consume):

    - baseline:
        {model_type: {param: scalar | list | dict}}   (overrides, free-form)

    - grid_search:
        {model_type: {param: [v1, v2, ...]}}          (every leaf MUST be a list;
        single-value scalars get auto-wrapped by ``_expand_grid_search`` but
        that's a convenience only — we insist on list form to keep the UI
        payload legible and to surface "you forgot the comma" mistakes here.)

    - bayesian_search:
        {model_type: {param: {type: float|int|categorical, ...}}}
        * float/int require low + high
        * categorical requires a non-empty choices list
    """
    if search_space in (None, {}):
        return

    if not isinstance(search_space, dict):
        raise HTTPException(
            status_code=422,
            detail=f"search_space must be a dict keyed by model_type, got {type(search_space).__name__}",
        )

    for model_type, model_space in search_space.items():
        if model_type not in selected_models:
            # Harmless extra key — log but don't fail the batch.  (Plan snapshot
            # can carry stale entries if the user edited selected_models later.)
            logger.warning(
                "search_space has entry for %r but that model isn't in selected_models; ignoring",
                model_type,
            )
            continue
        if model_space in (None, {}):
            continue
        if not isinstance(model_space, dict):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"search_space[{model_type!r}] must be a dict of "
                    f"{{param: spec}}, got {type(model_space).__name__}"
                ),
            )

        for param_name, spec in model_space.items():
            if strategy_type == "grid_search":
                if not isinstance(spec, list):
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"grid_search: search_space[{model_type!r}][{param_name!r}] "
                            f"must be a list of candidate values, got {type(spec).__name__}. "
                            f"Example: {{\"{param_name}\": [50, 100, 200]}}"
                        ),
                    )
                if len(spec) == 0:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"grid_search: search_space[{model_type!r}][{param_name!r}] "
                            "is empty — remove it or provide at least one candidate."
                        ),
                    )

            elif strategy_type == "bayesian_search":
                if not isinstance(spec, dict):
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"bayesian_search: search_space[{model_type!r}][{param_name!r}] "
                            f"must be a distribution dict, got {type(spec).__name__}. "
                            f"Example: {{\"{param_name}\": {{\"type\": \"float\", \"low\": 0.01, \"high\": 1.0}}}}"
                        ),
                    )
                dist_type = spec.get("type")
                if dist_type not in _VALID_BAYESIAN_DIST_TYPES:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"bayesian_search: search_space[{model_type!r}][{param_name!r}].type "
                            f"must be one of {sorted(_VALID_BAYESIAN_DIST_TYPES)}, got {dist_type!r}"
                        ),
                    )
                if dist_type in ("float", "int"):
                    if "low" not in spec or "high" not in spec:
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                f"bayesian_search: {dist_type} distribution for "
                                f"{model_type!r}[{param_name!r}] requires both 'low' and 'high'"
                            ),
                        )
                    if spec["low"] >= spec["high"]:
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                f"bayesian_search: {model_type!r}[{param_name!r}] "
                                f"'low' ({spec['low']}) must be < 'high' ({spec['high']})"
                            ),
                        )
                elif dist_type == "categorical":
                    choices = spec.get("choices")
                    if not isinstance(choices, list) or len(choices) == 0:
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                f"bayesian_search: categorical distribution for "
                                f"{model_type!r}[{param_name!r}] requires non-empty 'choices' list"
                            ),
                        )

            # baseline: free-form overrides; no structural check.


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _lock_task_for_experiment_dispatch(
    db: AsyncSession,
    modeling_task_id: str,
) -> ModelingTask:
    task = (
        await db.execute(
            select(ModelingTask)
            .where(ModelingTask.id == modeling_task_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"ModelingTask {modeling_task_id!r} not found",
        )
    final_state = task_final_evaluation_state(task)
    if final_state.get("state") != "OPEN":
        raise HTTPException(
            status_code=409,
            detail=(
                "任务已进入最终确认流程，不能再启动新批次；"
                "需要继续调参时请创建新的建模任务。"
            ),
        )
    return task



# ---------------------------------------------------------------------------
# Shared batch pre-flight
# ---------------------------------------------------------------------------
# Every *pure* validation for a batch lives here — nothing in this function
# touches the database or launches work. That is the whole point: a bundle
# commits and launches each batch as it goes, so any check left inside the
# per-batch path can reject strategy #2 *after* strategy #1 is already
# training. The client would see a 422 while work runs on regardless, which is
# the worst contract we can offer. Running this over the entire bundle first
# makes rejection all-or-nothing.
#
# ``_dispatch_experiment_batch_locked`` consumes the same result, so the two
# callers can never drift apart.


@dataclass
class _PreflightedBatch:
    strategy_type: str
    selected_models: list[str]
    search_space: dict[str, Any]
    budget_config: dict[str, Any]
    eval_metrics: list[str]
    dl_config: dict[str, Any]
    model_family: str | None
    ml_models: list[str] = field(default_factory=list)
    dl_models: list[str] = field(default_factory=list)
    tuning_defaults: dict[str, Any] = field(default_factory=dict)
    trials: list[dict[str, Any]] = field(default_factory=list)
    dl_trials: list[dict[str, Any]] = field(default_factory=list)
    total_trials: int = 0
    test_size: float = 0.2
    cv_folds: int = 5
    max_trials: Any = None


def _preflight_batch(
    task: Any,
    *,
    strategy_type: str | None,
    selected_models: list[str],
    search_space: dict[str, Any] | None,
    budget_config: dict[str, Any] | None,
    eval_metrics: list[str] | None = None,
    model_family: str | None = None,
    dl_config: dict[str, Any] | None = None,
    where: str = "",
) -> _PreflightedBatch:
    """Validate and expand one batch without any side effects.

    ``where`` prefixes every error (e.g. ``"strategy #2: "``) so a bundle
    rejection points at the offending item.
    """
    def _reject(detail: str, status_code: int = 422):
        raise HTTPException(status_code=status_code, detail=f"{where}{detail}")

    selected_models = list(selected_models or [])
    search_space = dict(search_space or {})
    budget_config = dict(budget_config or {})

    # --- snapshot-first defaulting (authoritative; see the locked dispatcher) --
    snapshot_payload: dict[str, Any] = {}
    snapshot = getattr(task, "training_plan_snapshot", None) or {}
    if isinstance(snapshot, dict):
        snapshot_payload = snapshot.get("payload") or {}

    if not selected_models and snapshot_payload.get("selected_models"):
        selected_models = list(snapshot_payload["selected_models"])
    if (not search_space) and snapshot_payload.get("search_space"):
        search_space = dict(snapshot_payload["search_space"])
    if (not budget_config) and snapshot_payload.get("budget_config"):
        budget_config = dict(snapshot_payload["budget_config"])
    if (not eval_metrics) and snapshot_payload.get("eval_metrics"):
        eval_metrics = list(snapshot_payload["eval_metrics"])
    if dl_config is None and snapshot_payload.get("dl_config"):
        dl_config = dict(snapshot_payload["dl_config"])
    if model_family is None and snapshot_payload.get("model_family"):
        model_family = snapshot_payload["model_family"]
    if (not strategy_type) and snapshot_payload.get("strategy_type"):
        strategy_type = snapshot_payload["strategy_type"]

    if strategy_type not in ("baseline", "grid_search", "bayesian_search", "automl"):
        _reject(f"Unsupported strategy_type: {strategy_type!r}")
    if not selected_models and strategy_type != "automl":
        _reject(f"{strategy_type} requires at least one selected model")

    task_type = task.task_type or "classification"
    try:
        tuning_defaults = load_tuning_spaces(task_type)
    except (ValueError, FileNotFoundError) as exc:
        # An unsupported task_type is a client-side mistake, not a server
        # fault. Letting the raw ValueError escape turns it into a 500 with no
        # indication of which task types *are* supported.
        _reject(f"不支持的任务类型 {task_type!r}: {exc}")

    # Re-raise with the bundle prefix: these helpers raise their own
    # HTTPException, so without this "strategy #N" would be missing on exactly
    # the errors a bundle caller most needs to locate.
    try:
        _validate_search_space(strategy_type, search_space, selected_models)
    except HTTPException as exc:
        _reject(str(exc.detail), status_code=exc.status_code)

    # --- split tokens by registry family ------------------------------------
    ml_models: list[str] = []
    dl_models: list[str] = []
    unknown_models: list[str] = []
    for token in selected_models:
        family = resolve_model_family(token)
        if family == "ml":
            (ml_models if token in tuning_defaults else unknown_models).append(token)
        elif family == "dl":
            spec = get_dl_model_spec(token)
            if spec is None or task_type not in spec.get("task_types", []):
                unknown_models.append(token)
            else:
                dl_models.append(token)
        else:
            unknown_models.append(token)
    if unknown_models:
        _reject(
            f"Unknown or incompatible model_type(s): {unknown_models}. "
            f"Available ML for {task_type}: {sorted(tuning_defaults.keys())}"
        )

    if strategy_type != "baseline" and dl_models:
        _reject(
            "Deep-learning models are currently supported in baseline batches only. "
            f"Remove from {strategy_type}: {dl_models}"
        )

    # M2c guard: the Optuna study is an in-process ask/tell loop. Under Celery
    # the orchestrator submits trial 1 and returns, so the study never advances
    # and the experiment would sit RUNNING forever. Refuse until M2d.
    if strategy_type == "bayesian_search" and ml_models:
        try:
            _reject_bayesian_under_celery()
        except HTTPException as exc:
            _reject(str(exc.detail), status_code=exc.status_code)

    # --- budget values ------------------------------------------------------
    # `or <default>` would swallow an explicit 0 and skip the range check
    # below, so an invalid test_size=0 / cv_folds=0 would silently become the
    # default instead of a 422. Only a missing key may default.
    raw_test_size = budget_config.get("test_size")
    raw_cv_folds = budget_config.get("cv_folds")
    max_trials = budget_config.get("max_trials")
    try:
        test_size = 0.2 if raw_test_size is None else float(raw_test_size)
        cv_folds = 5 if raw_cv_folds is None else int(raw_cv_folds)
        if max_trials is not None:
            max_trials = int(max_trials)
    except (TypeError, ValueError):
        _reject(
            "budget_config test_size/cv_folds/max_trials must be numeric, got "
            f"{ {k: budget_config.get(k) for k in ('test_size', 'cv_folds', 'max_trials')} }"
        )
    if not 0.0 < test_size < 1.0:
        _reject(f"budget_config.test_size must be between 0 and 1, got {test_size}")
    if cv_folds < 2:
        _reject(f"budget_config.cv_folds must be at least 2, got {cv_folds}")
    if max_trials is not None and max_trials < 1:
        _reject(f"budget_config.max_trials must be at least 1, got {max_trials}")

    # Write the coerced values back. Consumers downstream read the *raw*
    # budget_config (``_launch_bayesian`` compares ``global_trial_no >=
    # max_trials``), so leaving a string "5" here would pass pre-flight and
    # then blow up with a TypeError inside the background study — an API that
    # reports success while the experiment dies.
    budget_config["test_size"] = test_size
    budget_config["cv_folds"] = cv_folds
    if max_trials is not None:
        budget_config["max_trials"] = max_trials

    eval_metrics = list(
        eval_metrics or _default_eval_metrics(task_type, task.objective_metric)
    )
    dl_config = dict(dl_config or {})
    dl_trials = _expand_dl_baseline(dl_models, dl_config, task_type)

    # --- expand trials (pure) so "zero trials" is caught before any launch ---
    trials: list[dict[str, Any]] = []
    if strategy_type == "automl":
        try:
            trials = _expand_automl(task_type, max_trials)
        except (FileNotFoundError, ValueError) as exc:
            _reject(f"AutoML 候选清单不可用: {exc}")
        if not trials:
            _reject(f"AutoML 注册表中没有适用于 {task_type} 的候选模型")
        total_trials = len(trials)
        selected_models = sorted({t["model_type"] for t in trials})
    elif strategy_type == "baseline":
        merged_overrides = dict(search_space)
        for token in dl_models:
            if token not in merged_overrides and dl_config.get(token):
                merged_overrides[token] = dl_config[token]
        trials = _expand_baseline(selected_models, tuning_defaults, merged_overrides)
        if not trials:
            _reject("Baseline produced no trials — check selected_models")
        total_trials = len(trials)
    elif strategy_type == "grid_search":
        ml_trials = _expand_grid_search(ml_models, tuning_defaults, search_space, max_trials)
        trials = ml_trials + _renumber_trials(dl_trials, start=len(ml_trials) + 1)
        if not trials:
            _reject(
                "Grid search produced no trials — provide search_space or pick "
                "models with grid_values defined"
            )
        total_trials = len(trials)
    else:  # bayesian_search
        total_trials = _count_bayesian_trials(ml_models, budget_config, max_trials) + len(dl_trials)

    return _PreflightedBatch(
        strategy_type=strategy_type,
        selected_models=selected_models,
        search_space=search_space,
        budget_config=budget_config,
        eval_metrics=eval_metrics,
        dl_config=dl_config,
        model_family=model_family,
        ml_models=ml_models,
        dl_models=dl_models,
        tuning_defaults=tuning_defaults,
        trials=trials,
        dl_trials=dl_trials,
        total_trials=total_trials,
        test_size=test_size,
        cv_folds=cv_folds,
        max_trials=max_trials,
    )


async def dispatch_experiment_batch(
    db: AsyncSession,
    *,
    modeling_task_id: str,
    name: str,
    strategy_type: str,
    selected_models: list[str],
    search_space: dict[str, Any],
    budget_config: dict[str, Any],
    eval_metrics: list[str] | None = None,
    description: str | None = None,
    model_family: str | None = None,
    dl_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    async with task_lifecycle_guard(modeling_task_id):
        return await _dispatch_experiment_batch_locked(
            db,
            modeling_task_id=modeling_task_id,
            name=name,
            strategy_type=strategy_type,
            selected_models=selected_models,
            search_space=search_space,
            budget_config=budget_config,
            eval_metrics=eval_metrics,
            description=description,
            model_family=model_family,
            dl_config=dl_config,
        )


async def _dispatch_experiment_batch_locked(
    db: AsyncSession,
    *,
    modeling_task_id: str,
    name: str,
    strategy_type: str,
    selected_models: list[str],
    search_space: dict[str, Any],
    budget_config: dict[str, Any],
    eval_metrics: list[str] | None = None,
    description: str | None = None,
    model_family: str | None = None,
    dl_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a PlatformExperiment for this batch, expand trials, and launch them.

    Returns a serialised experiment plus trial counts.  Actual training runs
    execute asynchronously through asyncio.create_task — the HTTP response
    returns immediately after runs are persisted in PENDING state.

    V3 Phase 1 — DL integration
    ----------------------------
    ``selected_models`` may contain both ML and DL tokens.  Each token is
    resolved against the model registries; unknown tokens raise 422.
    DL tokens are dispatched as ``PlatformTask(kind='dl_train')`` and executed
    by ``dl_service._run_dl_training_by_id``.  DL models are baseline-only in
    Phase 1; grid/bayesian requests that include DL tokens fail fast with 422.
    """
    task = await _lock_task_for_experiment_dispatch(db, modeling_task_id)
    if not task.dataset_id or not task.target_column:
        raise HTTPException(
            status_code=400,
            detail="Modeling task must have dataset_id and target_column before dispatch",
        )

    # All pure validation + trial expansion happens up front and is shared with
    # ``dispatch_experiment_bundle`` — see ``_preflight_batch``.
    pf = _preflight_batch(
        task,
        strategy_type=strategy_type,
        selected_models=selected_models,
        search_space=search_space,
        budget_config=budget_config,
        eval_metrics=eval_metrics,
        model_family=model_family,
        dl_config=dl_config,
    )
    strategy_type = pf.strategy_type
    selected_models = pf.selected_models
    search_space = pf.search_space
    budget_config = pf.budget_config
    eval_metrics = pf.eval_metrics
    tuning_defaults = pf.tuning_defaults
    ml_models, dl_models = pf.ml_models, pf.dl_models
    test_size, cv_folds, max_trials = pf.test_size, pf.cv_folds, pf.max_trials
    total_trials = pf.total_trials

    # Create the experiment shell (RUNNING immediately so UI polls see it live).
    exp = PlatformExperiment(
        modeling_task_id=modeling_task_id,
        name=name,
        description=description,
        dataset_id=task.dataset_id,
        dataset_name=task.dataset_name,
        dataset_version_id=task.dataset_version_id,
        objective_metric=task.objective_metric,
        objective_direction=task.objective_direction,
        kind=strategy_type,                 # keep legacy kind in sync
        strategy_type=strategy_type,
        selected_models=selected_models,
        search_space=search_space or None,
        budget_config=budget_config or None,
        status="RUNNING",
        config={"submitted_from": "v3_workbench"},
    )
    db.add(exp)
    await db.flush()
    await db.refresh(exp)

    # Bump the modeling task to RUNNING on every new batch — not just the first
    # one — so a COMPLETED/FAILED task immediately reflects that new work is in
    # flight.  ``refresh_task_summary`` at the end of the batch will settle the
    # final status based on aggregate experiment state.
    if task.status != "RUNNING":
        task.status = "RUNNING"
        task.finished_at = None  # clear previous completion timestamp
        await db.flush()

    if strategy_type in ("baseline", "grid_search", "automl"):
        await _persist_trials(
            db, exp, task, pf.trials, eval_metrics,
            test_size=test_size, cv_folds=cv_folds,
        )
        await db.commit()
        await _launch_concurrent(exp.id, modeling_task_id)
    else:  # bayesian_search — ML part runs the study; DL part runs baseline
        if pf.dl_trials:
            await _persist_trials(
                db, exp, task, pf.dl_trials, eval_metrics,
                test_size=test_size, cv_folds=cv_folds,
            )
        await db.commit()
        if pf.dl_trials:
            await _launch_concurrent(exp.id, modeling_task_id)
        if ml_models:
            _launch_bayesian(
                experiment_id=exp.id,
                modeling_task_id=modeling_task_id,
                selected_models=ml_models,
                search_space=search_space,
                tuning_defaults=tuning_defaults,
                budget_config=budget_config,
                eval_metrics=eval_metrics,
                test_size=test_size,
                cv_folds=cv_folds,
            )

    return {
        "experiment": serialize_experiment(exp),
        "trials_planned": total_trials,
        "strategy_type": strategy_type,
    }


async def dispatch_experiment_bundle(
    db: AsyncSession,
    *,
    modeling_task_id: str,
    name: str,
    strategies: list[dict[str, Any]],
    description: str | None = None,
) -> dict[str, Any]:
    """Submit several strategy batches for the same modeling task in one call."""
    if not strategies:
        raise HTTPException(status_code=422, detail="At least one strategy is required")

    # Pre-flight the WHOLE bundle before launching anything. Each batch commits
    # and starts its trials immediately, so a rejection discovered mid-loop
    # would leave earlier batches running while the client only sees an error —
    # "failed request, work already started" is the worst possible contract.
    #
    # This runs the *same* ``_preflight_batch`` the per-batch dispatcher uses,
    # so every pure check (search_space shape, unknown/incompatible models,
    # DL outside baseline, budget ranges, zero-trial expansions, bayesian under
    # Celery) is enforced across the bundle before batch #1 commits. A partial
    # check here would silently reintroduce partial launches.
    task = (
        await db.execute(select(ModelingTask).where(ModelingTask.id == modeling_task_id))
    ).scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail=f"Modeling task {modeling_task_id} not found")
    if not task.dataset_id or not task.target_column:
        raise HTTPException(
            status_code=400,
            detail="Modeling task must have dataset_id and target_column before dispatch",
        )

    for idx, spec in enumerate(strategies, start=1):
        _preflight_batch(
            task,
            strategy_type=spec.get("strategy_type"),
            selected_models=list(spec.get("selected_models") or []),
            search_space=spec.get("search_space") or {},
            budget_config=spec.get("budget_config") or {},
            eval_metrics=spec.get("eval_metrics"),
            dl_config=spec.get("dl_config"),
            where=f"strategy #{idx}: ",
        )

    submitted: list[dict[str, Any]] = []
    strategy_types: list[str] = []
    total_trials = 0
    for idx, spec in enumerate(strategies, start=1):
        strategy_type = spec.get("strategy_type")
        selected_models = list(spec.get("selected_models") or [])

        label = {
            "baseline": "基线",
            "grid_search": "网格",
            "bayesian_search": "贝叶斯",
        }.get(strategy_type, strategy_type or "批次")
        result = await dispatch_experiment_batch(
            db,
            modeling_task_id=modeling_task_id,
            name=spec.get("name") or f"{name}-{idx}-{label}",
            strategy_type=strategy_type,
            selected_models=selected_models,
            search_space=spec.get("search_space") or {},
            budget_config=spec.get("budget_config") or {},
            eval_metrics=spec.get("eval_metrics"),
            description=spec.get("description") or description,
        )
        submitted.append(result)
        strategy_types.append(strategy_type)
        total_trials += int(result.get("trials_planned") or 0)

    return {
        "batch_count": len(submitted),
        "strategy_types": strategy_types,
        "total_trials_planned": total_trials,
        "items": submitted,
    }


# ---------------------------------------------------------------------------
# Trial expansion
# ---------------------------------------------------------------------------

def _expand_baseline(
    selected_models: list[str],
    tuning_defaults: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """One trial per model, using fixed defaults (+ user overrides).

    Handles both ML and DL tokens.  ML pulls ``fixed`` hyperparameters from
    ``tuning_defaults``; DL builds ``arch_config`` / ``opt_config`` /
    ``train_config`` blocks via ``build_default_dl_config`` and merges any
    per-model overrides from the caller (supporting both the ``arch`` short
    form and the ``arch_config`` suffixed form for backward-compat).

    Baseline DL trials are capped at ``epochs <= 10`` so a mixed ML+DL batch
    finishes in seconds rather than minutes during a baseline sweep.
    """
    overrides = overrides or {}
    trials: list[dict[str, Any]] = []
    for idx, model_type in enumerate(selected_models, start=1):
        family = resolve_model_family(model_type) or "ml"
        if family == "dl":
            defaults = build_default_dl_config(model_type)
            per_model = overrides.get(model_type) or {}
            arch_override = per_model.get("arch_config") or per_model.get("arch") or {}
            opt_override = per_model.get("opt_config") or per_model.get("opt") or {}
            train_override = per_model.get("train_config") or per_model.get("train") or {}
            train_config = {**defaults["train"], **train_override}
            # Cap baseline epochs — full sweeps go through grid/bayesian paths.
            if int(train_config.get("epochs", 10) or 10) > 10:
                train_config["epochs"] = 10
            hyperparameters = {
                "arch_config": {**defaults["arch"], **arch_override},
                "opt_config": {**defaults["opt"], **opt_override},
                "train_config": train_config,
            }
            trials.append({
                "family": "dl",
                "model_type": model_type,
                "hyperparameters": hyperparameters,
                "trial_no": idx,
                "search_meta": {"strategy": "baseline", "grid_index": None, "family": "dl"},
            })
            continue

        template = tuning_defaults.get(model_type) or {}
        params = dict(template.get("fixed") or {})
        params.update((overrides.get(model_type) or {}))
        trials.append({
            "family": template.get("family", "ml"),
            "model_type": model_type,
            "hyperparameters": params,
            "trial_no": idx,
            "search_meta": {"strategy": "baseline", "grid_index": None},
        })
    return trials


def _expand_automl(task_type: str, max_trials: int | None = None) -> list[dict[str, Any]]:
    """One trial per *candidate*, straight from the AutoML registry.

    Deliberately not expressed as a baseline batch. ``_expand_baseline`` emits
    one trial per ``model_type`` and keys its overrides the same way, but the
    registry intentionally lists the same model several times with different
    hyperparameters (five of the eleven classification candidates share a
    model_type). Routing AutoML through baseline would silently collapse those
    to one trial each and quietly halve the search. Grid search is no better:
    it takes the cartesian product of a model's values, so two candidate
    *configurations* would become four trials rather than two.
    """
    from app.services.automl_service import load_candidates

    candidates = load_candidates(task_type)
    if max_trials is not None:
        candidates = candidates[:max_trials]

    trials: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        trials.append({
            "family": "ml",
            "model_type": candidate["model_type"],
            "hyperparameters": dict(candidate.get("hyperparameters") or {}),
            "trial_no": idx,
            "search_meta": {
                "strategy": "automl",
                "grid_index": None,
                "candidate_index": idx,
                "candidate_description": candidate.get("description"),
            },
        })
    return trials


def _expand_grid_search(
    selected_models: list[str],
    tuning_defaults: dict[str, Any],
    search_space: dict[str, Any] | None,
    max_trials: int | None,
) -> list[dict[str, Any]]:
    """Cartesian product per model, clipped to ``max_trials`` total."""
    from sklearn.model_selection import ParameterGrid  # local import — heavy

    search_space = search_space or {}
    trials: list[dict[str, Any]] = []
    trial_no = 0

    for model_type in selected_models:
        template = tuning_defaults[model_type]
        family = template.get("family", "ml")
        if family == "dl":
            logger.warning("Grid search: DL model %s is baseline-only, skipping", model_type)
            continue
        user_grid = search_space.get(model_type) or {}
        # Use user grid if present, otherwise fall back to registry defaults.
        grid = user_grid or (template.get("grid_values") or {})
        if not grid:
            # Model has no grid defined (e.g. ridge in classification) — skip.
            logger.warning("Grid search: no grid_values for %s, skipping", model_type)
            continue
        fixed = dict(template.get("fixed") or {})

        # ParameterGrid wants {"key": [values, ...]}; single values must be wrapped.
        normalised = {
            k: (v if isinstance(v, list) else [v])
            for k, v in grid.items()
        }

        for grid_idx, combo in enumerate(ParameterGrid(normalised)):
            trial_no += 1
            params = {**fixed, **combo}
            trials.append({
                "family": family,
                "model_type": model_type,
                "hyperparameters": params,
                "trial_no": trial_no,
                "search_meta": {
                    "strategy": "grid_search",
                    "grid_index": grid_idx,
                    "combo": combo,
                },
            })
            if max_trials and trial_no >= max_trials:
                return trials
    return trials


def _expand_dl_baseline(
    dl_models: list[str],
    dl_config: dict[str, Any],
    task_type: str,
) -> list[dict[str, Any]]:
    """
    One trial per DL model.  Thin wrapper over ``_expand_baseline`` that
    preserves the older ``(dl_models, dl_config, task_type)`` signature used
    by grid/bayesian downgrade paths; emits trials with the canonical
    ``arch_config``/``opt_config``/``train_config`` hyperparameter keys and
    ``family='dl'`` so ``_persist_trials`` picks the DL branch.
    """
    trials = _expand_baseline(
        selected_models=list(dl_models),
        tuning_defaults={},
        overrides=dl_config or {},
    )
    # Tag task_type for downstream DL trainer selection (classification / regression).
    return [{**t, "_task_type": task_type} for t in trials]


def _renumber_trials(trials: list[dict[str, Any]], *, start: int) -> list[dict[str, Any]]:
    """Rewrite trial_no in-place starting at ``start`` so ML+DL trial indices stay dense."""
    out = []
    for offset, t in enumerate(trials):
        out.append({**t, "trial_no": start + offset})
    return out


def _count_bayesian_trials(
    selected_models: list[str], budget_config: dict[str, Any] | None, max_trials: int | None
) -> int:
    per_model = (budget_config or {}).get("n_trials_per_model", 10)
    total = per_model * len(selected_models)
    if max_trials:
        total = min(total, max_trials)
    return total


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

async def _persist_trials(
    db: AsyncSession,
    exp: PlatformExperiment,
    task: ModelingTask,
    trials: list[dict[str, Any]],
    eval_metrics: list[str],
    *,
    test_size: float = 0.2,
    cv_folds: int = 5,
) -> None:
    """Create domain task + ExperimentRun + PlatformTask for each trial.

    ML trials create a ``TrainingTask`` + ``PlatformTask(kind='train')``;
    DL trials create a ``DLTrainingTask`` + ``PlatformTask(kind='dl_train')``.
    The run's ``params['family']`` is set so the concurrent executor can
    dispatch to the right service.
    """
    for trial in trials:
        # ``trial["family"]`` is set by the expander (_expand_baseline / _expand_grid_search /
        # _expand_dl_baseline).  DL trials use the helper that drops their hyperparameter
        # tree into arch/opt/train sections; ML trials go through create_training_task_record
        # which handles ID-leakage feature dropping downstream.
        family = trial.get("family", "ml")
        if family == "dl":
            domain_task = await _create_dl_training_task_record(
                db=db,
                dataset_id=task.dataset_id,
                target_column=task.target_column,
                task_type=task.task_type,
                model_type=trial["model_type"],
                config=trial["hyperparameters"],
                test_size=test_size,
            )
            kind = "dl_train"
            payload_ref = f"dl_train:{domain_task.id}"
        else:
            domain_task = await create_training_task_record(
                db,
                {
                    "dataset_id": task.dataset_id,
                    "model_type": trial["model_type"],
                    "target_column": task.target_column,
                    "hyperparameters": trial["hyperparameters"],
                    "test_size": test_size,
                    "cv_folds": cv_folds,
                    "eval_metrics": eval_metrics,
                },
            )
            kind = "train"
            payload_ref = f"train:{domain_task.id}"

        search_meta = dict(trial["search_meta"])
        # B1: every V3 run (ML and DL alike) trains in selection mode — the
        # outer hold-out stays sealed until POST /final-evaluation opens it
        # once for the confirmed winner.
        if family in ("ml", "dl"):
            search_meta["evaluation_mode"] = "selection"
        run = ExperimentRun(
            experiment_id=exp.id,
            params={
                "family": family,
                "model_type": trial["model_type"],
                "hyperparameters": trial["hyperparameters"],
                "dataset_id": task.dataset_id,
                "target_column": task.target_column,
                "task_type": task.task_type,
                "eval_metrics": eval_metrics,
            },
            status="PENDING",
            trial_no=trial["trial_no"],
            search_meta=search_meta,
            source_experiment_type=exp.strategy_type,
        )
        db.add(run)
        await db.flush()
        await db.refresh(run)

        platform_task = await register_domain_task(
            db=db,
            kind=kind,
            payload_ref=payload_ref,
        )
        run.task_id = platform_task.id
        await db.flush()

        trial["_domain_task_id"] = domain_task.id
        trial["_platform_task_id"] = platform_task.id
        trial["_run_id"] = run.id


async def _create_dl_training_task_record(
    *,
    db: AsyncSession,
    dataset_id: str,
    target_column: str,
    task_type: str,
    model_type: str,
    config: dict[str, Any],
    test_size: float,
) -> DLTrainingTask:
    """Create a DLTrainingTask row without auto-launching it."""
    from app.core.dl_registry import get_dl_trainer_registry

    available = list(get_dl_trainer_registry().keys())
    if model_type not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown DL model {model_type!r}. Available: {available}",
        )

    import uuid as _uuid_mod

    train_config = dict(config.get("train_config") or {})
    train_config.setdefault("test_size", test_size)
    short_id = str(_uuid_mod.uuid4())[:8]
    task = DLTrainingTask(
        dataset_id=dataset_id,
        name=f"{model_type}_{short_id}",
        target_column=target_column,
        model_type=model_type,
        task_type=task_type or "classification",
        arch_config=dict(config.get("arch_config") or {}),
        opt_config=dict(config.get("opt_config") or {}),
        train_config=train_config,
        status="PENDING",
    )
    db.add(task)
    await db.flush()
    await db.refresh(task)
    return task


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------

async def _launch_concurrent(experiment_id: str, modeling_task_id: str) -> None:
    """Submit every persisted PENDING run without waiting for execution."""
    await _run_concurrent_batch(experiment_id, modeling_task_id)


async def _run_concurrent_batch(experiment_id: str, modeling_task_id: str) -> None:
    """Submit all PENDING runs; their scheduler wrappers finalise the batch."""
    async with async_session_factory() as db:
        rows = await db.execute(
            select(ExperimentRun).where(
                ExperimentRun.experiment_id == experiment_id,
                ExperimentRun.status == "PENDING",
            )
        )
        runs = rows.scalars().all()
        triples = [
            (run.task_id, run.id, _parse_domain_task_id_from_payload_ref(run))
            for run in runs
        ]

    # Retrieve payload_ref → (kind, domain_task_id) via PlatformTask lookup.
    # ``kind`` decides which executor the trial runs through (ML train vs DL train).
    async with async_session_factory() as db:
        full_entries: list[tuple[str, str, str, str]] = []  # (kind, domain_id, platform_id, run_id)
        for platform_task_id, run_id, domain_fallback in triples:
            pt = (
                await db.execute(
                    select(PlatformTask).where(PlatformTask.id == platform_task_id)
                )
            ).scalar_one_or_none()
            if pt and pt.payload_ref and ":" in pt.payload_ref:
                # ``pt.kind`` is authoritative for executor-registry lookup; fall back
                # to parsing payload_ref for older rows that only recorded the prefix.
                payload_kind, _, domain_task_id = pt.payload_ref.partition(":")
                resolved_kind = (pt.kind or payload_kind or "train")
                full_entries.append((resolved_kind, domain_task_id, platform_task_id, run_id))
            elif domain_fallback:
                full_entries.append(("train", domain_fallback, platform_task_id, run_id))

    submissions = await asyncio.gather(
        *(
            get_scheduler(kind).submit(platform_task_id)
            for kind, _domain_id, platform_task_id, _run_id in full_entries
        ),
        return_exceptions=True,
    )
    for submission in submissions:
        if isinstance(submission, BaseException):
            logger.error(
                "Failed to submit experiment %s trial: %s",
                experiment_id,
                submission,
            )


def _parse_domain_task_id_from_payload_ref(run: ExperimentRun) -> str | None:
    # Best-effort fallback if PlatformTask lookup fails later.  ExperimentRun
    # itself does not carry the domain task id, so this always returns None;
    # kept as a placeholder so the lookup stays explicit.
    return None


async def _execute_single_trial(
    domain_task_id: str,
    platform_task_id: str,
    run_id: str,
    experiment_id: str,
    *,
    kind: str = "train",
) -> dict[str, Any]:
    """Run one trial to completion and write metrics + PlatformTask state back.

    V3 Phase 2 — no more ``if kind == "dl_train"`` branching.  The
    ``Executor Registry`` holds the (domain_id, platform_task_id) → dict
    callable for every supported kind; ML and DL are now peers.  Adding a
    new kind (prepare_data, evaluate, …) is a one-line registration in the
    owning service module.

    Codex's ``_normalise_run_metrics`` is still applied after the executor
    returns so the leaderboard sees ``accuracy``/``rmse`` aliases regardless
    of whether the underlying run was ML or DL.
    """
    from app.scheduler.executors import get_executor
    from app.services.run_writeback import claim_for_execution, complete_platform_task

    # Claim first: this is the only place RUNNING is written, and it refuses a
    # task whose Run is already terminal. Writing RUNNING before claiming would
    # drag a finished task back to RUNNING on a duplicate delivery.
    if not await claim_for_execution(platform_task_id):
        logger.info("Trial %s already terminal; skipping duplicate execution", run_id)
        return {"run_id": run_id, "status": "SKIPPED"}

    # Executor failure and write-back failure are NOT the same thing and must
    # not share an except block: a transient DB error while committing a
    # *successful* result would otherwise be re-reported as a failed trial.
    try:
        result = await get_executor(kind)(domain_task_id, platform_task_id)
        if not isinstance(result, dict):
            raise TypeError(
                f"executor for kind={kind!r} returned {type(result).__name__}, expected dict"
            )
        metrics = result.get("metrics") or {}
        evaluation_mode = result.get("evaluation_mode", "standard")
    except Exception as exc:  # noqa: BLE001 — genuine trial failure
        logger.error("Tuning trial %s failed: %s", run_id, exc, exc_info=True)
        # In-process has no retry budget, so this attempt is terminal. If the
        # write itself fails we let it propagate — a silently swallowed
        # write-back is exactly the RUNNING-forever bug M2c exists to kill.
        await complete_platform_task(
            platform_task_id,
            status="FAILED",
            error=str(exc),
            domain_task_id=domain_task_id,
            final_attempt=True,
        )
        return {"run_id": run_id, "status": "FAILED"}

    # Training succeeded. Any exception from here is a bookkeeping failure and
    # propagates untouched — never downgraded to "the trial failed".
    outcome = await complete_platform_task(
        platform_task_id,
        status="SUCCESS",
        metrics=metrics,
        evaluation_mode=evaluation_mode,
        domain_task_id=domain_task_id,
        final_attempt=True,
    )
    # Report what was actually committed, not what we asked for.
    return {
        "run_id": run_id,
        "status": outcome.status,
        "metrics": outcome.metrics or {},
    }


async def _mirror_logs_to_v3(*, domain_task_id: str, run_id: str) -> None:
    """Copy ``training_logs`` rows for a legacy task into ``experiment_run_logs``
    keyed by V3 ``run_id``.

    Called once per trial after the executor finishes (success or failure).
    Idempotent: if there's already a row in experiment_run_logs for this
    run_id, we skip — multiple calls for the same trial (e.g. retries) won't
    duplicate.

    Failures are swallowed and logged; observability data must never block a
    successful (or failed) run from being committed.
    """
    try:
        async with async_session_factory() as db:
            # Skip if already mirrored — keeps this safe to call multiple times.
            existing = (
                await db.execute(
                    select(ExperimentRunLog.id)
                    .where(ExperimentRunLog.run_id == run_id)
                    .limit(1)
                )
            ).first()
            if existing is not None:
                return

            legacy_rows = (
                await db.execute(
                    select(TrainingLog)
                    .where(TrainingLog.task_id == domain_task_id)
                    .order_by(TrainingLog.created_at)
                )
            ).scalars().all()
            if not legacy_rows:
                return  # nothing to mirror — trainer wrote no DB logs

            for ll in legacy_rows:
                db.add(ExperimentRunLog(
                    run_id=run_id,
                    level=ll.level,
                    message=ll.message,
                    extra=ll.extra,
                    created_at=ll.created_at,
                ))
            await db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to mirror logs to experiment_run_logs for run %s "
            "(domain_task_id=%s): %s", run_id, domain_task_id, exc
        )


def _normalise_run_metrics(
    metrics: dict[str, Any],
    evaluation_mode: str = "standard",
) -> dict[str, Any]:
    """Expose common leaderboard metric keys for both ML and DL runs.

    B1 semantics: in ``selection`` mode the DL val metrics come from an INNER
    validation split (the outer hold-out is sealed), so they map to
    ``selection_val_{metric}`` — never pre-stamped as ``final_test_*``.
    ``final_test_*`` for selection runs is written exclusively by
    final_evaluation_service after the winner is confirmed.
    """
    normalised: dict[str, Any] = {}
    for key, value in (metrics or {}).items():
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        normalised[key] = value

    aliases = {
        "val_acc": "accuracy",
        "val_f1_macro": "f1",
        "val_auc_roc": "roc_auc",
        "val_rmse": "rmse",
        "val_mae": "mae",
        "val_r2": "r2",
    }
    for source, target in aliases.items():
        source_value = normalised.get(source)
        if not isinstance(source_value, (int, float)):
            continue
        if evaluation_mode == "selection":
            normalised.setdefault(f"selection_val_{target}", float(source_value))
        else:
            if target not in normalised:
                normalised[target] = float(source_value)
            normalised.setdefault(f"final_test_{target}", float(source_value))
    return normalised


async def _finalise_batch(experiment_id: str, modeling_task_id: str) -> None:
    """After a batch completes, update experiment + refresh modeling task summary."""
    from sqlalchemy import func

    async with async_session_factory() as db:
        counts = await db.execute(
            select(ExperimentRun.status, func.count(ExperimentRun.id))
            .where(ExperimentRun.experiment_id == experiment_id)
            .group_by(ExperimentRun.status)
        )
        status_map = {s: int(c) for s, c in counts.all()}
        total = sum(status_map.values())
        # CANCELLED counts as done: a cancelled trial will never report again,
        # so excluding it would hang the batch at done < total forever.
        done = (
            status_map.get("SUCCESS", 0)
            + status_map.get("FAILED", 0)
            + status_map.get("CANCELLED", 0)
        )

        triggered_explain = False
        if total > 0 and done >= total:
            exp = (
                await db.execute(
                    select(PlatformExperiment)
                    .where(PlatformExperiment.id == experiment_id)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if exp and exp.status not in {"COMPLETED", "FAILED"}:
                if status_map.get("SUCCESS", 0) == 0:
                    exp.status = "FAILED"
                else:
                    exp.status = "COMPLETED"
                    triggered_explain = True
                exp.finished_at = datetime.now(timezone.utc)
                await db.commit()

        await refresh_task_summary(db, modeling_task_id)
        await db.commit()

    # ── Auto-trigger SHAP for the top-3 runs ──────────────────────────────
    # Runs happen outside this finalisation path, but after the experiment
    # has settled we kick off explain tasks asynchronously so the UI sees
    # SHAP results pop in shortly after the leaderboard stabilises. Failures
    # here are logged but never propagate — SHAP is a nice-to-have.
    if triggered_explain:
        try:
            await _schedule_shap_for_top_runs(experiment_id, top_k=3)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SHAP auto-trigger failed for experiment %s: %s", experiment_id, exc
            )


async def _schedule_shap_for_top_runs(experiment_id: str, *, top_k: int = 3) -> None:
    """Enqueue ``explain`` platform tasks for the top-K SUCCESS runs of an experiment.

    Ranking respects the experiment's objective_metric + objective_direction.
    Skips runs that already have a SHAP result (``metrics["shap_importances"]``).
    """
    async with async_session_factory() as db:
        exp = (
            await db.execute(
                select(PlatformExperiment).where(PlatformExperiment.id == experiment_id)
            )
        ).scalar_one_or_none()
        if exp is None:
            return

        objective_metric = (exp.objective_metric or "").strip() or None
        direction = (exp.objective_direction or "max").lower()
        reverse = direction != "min"  # "max" → sort descending

        runs = (
            await db.execute(
                select(ExperimentRun)
                .where(ExperimentRun.experiment_id == experiment_id)
                .where(ExperimentRun.status == "SUCCESS")
            )
        ).scalars().all()

        def _score(run: ExperimentRun) -> float:
            metrics = run.metrics or {}
            resolved = resolve_objective_metrics(metrics, objective_metric)
            if resolved.selection_value is not None:
                return resolved.selection_value
            # Fallback: lowest trial_no first so order is deterministic.
            return float(run.trial_no or 0)

        # Filter out runs already explained so re-runs of the finaliser are idempotent.
        candidates = [
            r for r in runs if not (r.metrics or {}).get("shap_importances")
        ]
        candidates.sort(key=_score, reverse=reverse)
        explainable_candidates: list[ExperimentRun] = []
        for run in candidates:
            if not run.task_id:
                continue
            platform_task = (
                await db.execute(
                    select(PlatformTask).where(PlatformTask.id == run.task_id)
                )
            ).scalar_one_or_none()
            if platform_task and platform_task.kind in {"train", "dl_train"}:
                explainable_candidates.append(run)

        top_runs = explainable_candidates[: max(0, int(top_k))]
        if not top_runs:
            return

        dispatches: list[tuple[str, str, int]] = []
        for run in top_runs:
            platform_task = await register_domain_task(
                db=db,
                kind="explain",
                payload_ref=f"explain:{run.id}",
                priority=3,
            )
            dispatches.append((platform_task.id, "explain", f"explain:{run.id}"))
        await db.commit()

    # Dispatch outside the DB transaction so Celery never races an uncommitted row.
    for platform_task_id, kind, payload_ref in dispatches:
        try:
            await get_scheduler(kind).submit(platform_task_id)
            logger.info(
                "SHAP explain dispatched: experiment=%s platform_task=%s payload=%s",
                experiment_id,
                platform_task_id,
                payload_ref,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to dispatch explain task %s: %s", platform_task_id, exc)
            try:
                await update_platform_task_status(
                    platform_task_id, "FAILED", error=str(exc)
                )
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Bayesian search (Optuna)
# ---------------------------------------------------------------------------

def _reject_bayesian_under_celery() -> None:
    """Fail fast when bayesian_search would be routed to a Celery worker.

    Better a clear 422 at dispatch than an experiment that silently sits in
    RUNNING forever. Lifted until M2d ships the cross-process ask/tell
    continuation (Optuna RDBStorage + completion-driven ``tell``).
    """
    from app.scheduler.scheduler import CeleryScheduler, get_scheduler

    if isinstance(get_scheduler("train"), CeleryScheduler):
        raise HTTPException(
            status_code=422,
            detail=(
                "贝叶斯调参暂不支持 Celery 执行（CELERY_KINDS 含 train）："
                "Optuna 的 ask/tell 目前是进程内顺序循环，跨进程会导致实验永久停在"
                "运行中。请改用 baseline/grid_search，或将 train 移出 CELERY_KINDS。"
            ),
        )


def _launch_bayesian(
    *,
    experiment_id: str,
    modeling_task_id: str,
    selected_models: list[str],
    search_space: dict[str, Any] | None,
    tuning_defaults: dict[str, Any],
    budget_config: dict[str, Any] | None,
    eval_metrics: list[str],
    test_size: float = 0.2,
    cv_folds: int = 5,
) -> None:
    asyncio.create_task(
        _run_bayesian_search_guarded(
            experiment_id=experiment_id,
            modeling_task_id=modeling_task_id,
            selected_models=selected_models,
            search_space=search_space or {},
            tuning_defaults=tuning_defaults,
            budget_config=budget_config or {},
            eval_metrics=eval_metrics,
            test_size=test_size,
            cv_folds=cv_folds,
        )
    )


async def _run_bayesian_search_guarded(**kwargs: Any) -> None:
    """Run Bayesian search and persist fatal background-worker failures."""
    experiment_id = kwargs["experiment_id"]
    modeling_task_id = kwargs["modeling_task_id"]
    try:
        await _run_bayesian_search(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Bayesian worker failed for experiment %s", experiment_id)
        async with async_session_factory() as db:
            exp = await db.get(PlatformExperiment, experiment_id)
            if exp:
                exp.status = "FAILED"
                exp.finished_at = datetime.now(timezone.utc)
                exp.config = {
                    **(exp.config or {}),
                    "worker_error": str(exc),
                }
                await db.flush()
            await refresh_task_summary(db, modeling_task_id)
            await db.commit()


def _tpe_startup_trials(n_trials: int, budget_config: dict[str, Any] | None = None) -> int:
    """How many trials TPE spends on random exploration before it models.

    Optuna's default is 10 — and this platform's default ``n_trials_per_model``
    is also 10. Left alone, that means the default "bayesian search" spends
    100% of its budget in the random startup phase and never reaches the TPE
    modelling stage at all: measured on optuna 4.1.0, a default run completes
    10 trials with **zero** TPE-sampled suggestions. Users pick 「贝叶斯搜索」
    and get random search, only slower (trials run sequentially).

    Scaling startup with the budget fixes that without making the default run
    any longer: ~1/3 of trials explore, the rest exploit. The floor of 3 keeps
    TPE from modelling on almost nothing; the ceiling of 10 preserves Optuna's
    default behaviour once the budget is large enough to afford it.

    An explicit ``n_startup_trials`` in budget_config always wins.
    """
    explicit = (budget_config or {}).get("n_startup_trials")
    if explicit is not None:
        return max(1, int(explicit))
    return max(3, min(10, n_trials // 3))


async def _run_bayesian_search(
    *,
    experiment_id: str,
    modeling_task_id: str,
    selected_models: list[str],
    search_space: dict[str, Any],
    tuning_defaults: dict[str, Any],
    budget_config: dict[str, Any],
    eval_metrics: list[str],
    test_size: float = 0.2,
    cv_folds: int = 5,
) -> None:
    """
    Run one Optuna study *per model*, sequentially.

    For each model:
      - build a study with TPE sampler
      - for each trial: sample params → train → feed value back to study
      - persist the trial as an ExperimentRun with optuna metadata
    """
    import optuna
    from optuna.samplers import TPESampler

    n_trials_per_model = int(budget_config.get("n_trials_per_model", 10))
    random_state = int(budget_config.get("random_state", 42))
    n_startup_trials = _tpe_startup_trials(n_trials_per_model, budget_config)
    max_trials = budget_config.get("max_trials")

    async with async_session_factory() as db:
        task = await db.get(ModelingTask, modeling_task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Modeling task not found")
        direction = "maximize" if (task.objective_direction or "max") == "max" else "minimize"
        objective_metric = task.objective_metric or "accuracy"
        dataset_id = task.dataset_id
        target_column = task.target_column
        task_type = task.task_type

    global_trial_no = 0
    budget_exhausted = False

    for model_type in selected_models:
        if budget_exhausted:
            break
        template = tuning_defaults[model_type]
        dist_space = search_space.get(model_type) or (template.get("distribution") or {})
        if not dist_space:
            logger.warning("Bayesian search: no distribution for %s, skipping", model_type)
            continue
        fixed = dict(template.get("fixed") or {})

        study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=random_state, n_startup_trials=n_startup_trials),
            study_name=f"{experiment_id}-{model_type}",
        )

        for trial_idx in range(n_trials_per_model):
            # `max_trials` (when set) is the GLOBAL budget across all models.
            # Use `n_trials_per_model` to control per-model trial count.
            # We must `break` (not `return`) so the function still falls
            # through to `_finalise_batch` below; otherwise the
            # PlatformExperiment row is stranded in status=RUNNING even
            # after every child run has reached SUCCESS/FAILED.
            if max_trials is not None and global_trial_no >= max_trials:
                budget_exhausted = True
                break
            global_trial_no += 1
            try:
                trial = study.ask()
                params = _sample_from_distribution(trial, dist_space)
                full_params = {**fixed, **params}

                run_id, platform_task_id, domain_task_id = await _persist_single_bayesian_trial(
                    experiment_id=experiment_id,
                    model_type=model_type,
                    hyperparameters=full_params,
                    trial_no=global_trial_no,
                    optuna_trial_id=trial.number,
                    dataset_id=dataset_id,
                    target_column=target_column,
                    task_type=task_type,
                    eval_metrics=eval_metrics,
                    test_size=test_size,
                    cv_folds=cv_folds,
                )

                scheduled = await get_scheduler("train").submit(platform_task_id)
                if not isinstance(scheduled, asyncio.Task):
                    logger.warning(
                        "Bayesian trial %s submitted to Celery; study continuation "
                        "is deferred to M2d",
                        run_id,
                    )
                    return
                outcome = await scheduled
                resolved = resolve_objective_metrics(
                    outcome.get("metrics") or {}, objective_metric
                )
                value = resolved.selection_value
                if value is None:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                else:
                    study.tell(trial, float(value))

                # Persist optuna state into the run's search_meta for the inspector.
                async with async_session_factory() as db:
                    run = (
                        await db.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))
                    ).scalar_one_or_none()
                    if run:
                        meta = dict(run.search_meta or {})
                        meta["optuna_state"] = trial.state.name if hasattr(trial, "state") else "COMPLETE"
                        meta["objective_value"] = value
                        meta["selection_metric_key"] = resolved.selection_metric_key
                        run.search_meta = meta
                        await db.commit()

            except Exception as exc:  # noqa: BLE001
                logger.error("Bayesian trial %d for %s failed: %s", trial_idx, model_type, exc)
                continue

    await _finalise_batch(experiment_id, modeling_task_id)


def _sample_from_distribution(trial: Any, dist_space: dict[str, Any]) -> dict[str, Any]:
    """Translate a {param: {type, ...}} spec into Optuna suggest_* calls."""
    params: dict[str, Any] = {}
    for name, spec in dist_space.items():
        spec_type = (spec or {}).get("type")
        if spec_type == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=bool(spec.get("log", False))
            )
        elif spec_type == "int":
            params[name] = trial.suggest_int(
                name, spec["low"], spec["high"], step=int(spec.get("step", 1))
            )
        elif spec_type == "categorical":
            # Optuna requires hashable choices; lists are not hashable so we
            # index into the list of choices and return the actual value.
            choices = spec["choices"]
            if all(isinstance(c, (str, int, float, bool)) or c is None for c in choices):
                params[name] = trial.suggest_categorical(name, choices)
            else:
                idx = trial.suggest_int(f"{name}__idx", 0, len(choices) - 1)
                params[name] = choices[idx]
        else:
            raise ValueError(f"Unknown distribution type {spec_type!r} for {name!r}")
    return params


async def _persist_single_bayesian_trial(
    *,
    experiment_id: str,
    model_type: str,
    hyperparameters: dict[str, Any],
    trial_no: int,
    optuna_trial_id: int,
    dataset_id: str,
    target_column: str,
    task_type: str,
    eval_metrics: list[str],
    test_size: float = 0.2,
    cv_folds: int = 5,
) -> tuple[str, str, str]:
    """Persist one Optuna trial to DB → return (run_id, platform_task_id, domain_task_id)."""
    async with async_session_factory() as db:
        domain_task = await create_training_task_record(
            db,
            {
                "dataset_id": dataset_id,
                "model_type": model_type,
                "target_column": target_column,
                "hyperparameters": hyperparameters,
                "test_size": test_size,
                "cv_folds": cv_folds,
                "eval_metrics": eval_metrics,
            },
        )

        run = ExperimentRun(
            experiment_id=experiment_id,
            params={
                "model_type": model_type,
                "hyperparameters": hyperparameters,
                "dataset_id": dataset_id,
                "target_column": target_column,
                "task_type": task_type,
                "eval_metrics": eval_metrics,
            },
            status="PENDING",
            trial_no=trial_no,
            search_meta={
                "strategy": "bayesian_search",
                "optuna_trial_id": optuna_trial_id,
                "evaluation_mode": "selection",
            },
            source_experiment_type="bayesian_search",
        )
        db.add(run)
        await db.flush()
        await db.refresh(run)

        platform_task = await register_domain_task(
            db=db,
            kind="train",
            payload_ref=f"train:{domain_task.id}",
        )
        run.task_id = platform_task.id
        await db.commit()

        return run.id, platform_task.id, domain_task.id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_eval_metrics(task_type: str, objective_metric: str | None) -> list[str]:
    if task_type == "regression" or (objective_metric or "").lower() in _REGRESSION_METRICS:
        return ["rmse", "mae", "r2"]
    return ["accuracy", "f1", "roc_auc"]
