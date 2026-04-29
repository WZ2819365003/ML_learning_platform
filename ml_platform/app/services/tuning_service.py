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
- baseline & grid_search fire runs concurrently via ``asyncio.create_task``.
- bayesian_search runs one trial at a time because Optuna's TPE needs the
  last result before suggesting the next set of hyperparameters.
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
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dl_registry import build_default_dl_config, get_dl_model_spec
from app.core.model_registry import resolve_model_family
from app.core.ts_registry import get_ts_model_spec
from app.models.database import (
    DLTrainingTask,
    ExperimentRun,
    ExperimentRunLog,
    ModelingTask,
    PlatformTask,
    PlatformExperiment,
    TrainingLog,
    TrainingTask,
    async_session_factory,
)
from app.scheduler.task_runner import (
    dispatch_platform_task,
    register_domain_task,
    update_platform_task_status,
)
from app.services.modeling_task_service import (
    _get_task_or_404,
    load_tuning_spaces,
    refresh_task_summary,
    serialize_experiment,
)
from app.services.training_service import create_training_task_record

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
    task = await _get_task_or_404(db, modeling_task_id)
    if not task.dataset_id or not task.target_column:
        raise HTTPException(
            status_code=400,
            detail="Modeling task must have dataset_id and target_column before dispatch",
        )

    task_type = task.task_type or "classification"
    tuning_defaults = load_tuning_spaces(task_type)

    # -----------------------------------------------------------------------
    # V3 Phase 2 — snapshot-first defaulting
    # If the caller omitted selected_models / search_space / dl_config /
    # budget_config / eval_metrics, fall back to the plan snapshot frozen at task creation.
    # Snapshot is authoritative; editing the live plan after bind does NOT
    # change what the task runs (reproducibility).
    # -----------------------------------------------------------------------
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

    # Validate search_space shape BEFORE we touch the DB or spawn any tasks.
    # Shape errors surface as a 422 with an actionable "use [50, 100, 200]"
    # hint so the workbench UI can fix its payload without guesswork.
    _validate_search_space(strategy_type, search_space, selected_models)

    # Split selected_models by family via registry lookup.  Any token that
    # belongs to neither registry is rejected.
    ml_models: list[str] = []
    dl_models: list[str] = []
    ts_models: list[str] = []
    unknown_models: list[str] = []
    for token in selected_models:
        family = resolve_model_family(token)
        if family == "ml":
            if token not in tuning_defaults:
                unknown_models.append(token)
            else:
                ml_models.append(token)
        elif family == "dl":
            spec = get_dl_model_spec(token)
            if spec is None or task_type not in spec.get("task_types", []):
                unknown_models.append(token)
            else:
                dl_models.append(token)
        elif family == "ts":
            spec = get_ts_model_spec(token)
            if spec is None or task_type not in spec.get("task_types", []):
                unknown_models.append(token)
            else:
                ts_models.append(token)
        else:
            unknown_models.append(token)
    if unknown_models:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown or incompatible model_type(s): {unknown_models}. "
                f"Available ML for {task_type}: {sorted(tuning_defaults.keys())}"
            ),
        )

    if strategy_type != "baseline" and dl_models:
        raise HTTPException(
            status_code=422,
            detail=(
            "Deep-learning models are currently supported in baseline batches only. "
                f"Remove from {strategy_type}: {dl_models}"
            ),
        )

    if strategy_type != "baseline" and ts_models:
        raise HTTPException(
            status_code=422,
            detail=(
                "Time-series models are currently supported in baseline batches only. "
                f"Remove from {strategy_type}: {ts_models}"
            ),
        )

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

    # Expand trials → list of concrete hyperparameter dicts per model.
    eval_metrics = list(eval_metrics or _default_eval_metrics(task_type, task.objective_metric))
    max_trials = budget_config.get("max_trials") if budget_config else None
    test_size = float((budget_config or {}).get("test_size") or 0.2)
    cv_folds = int((budget_config or {}).get("cv_folds") or 5)

    # DL trials are always baseline in Phase 1; non-baseline strategies with DL
    # tokens have already failed fast above.
    dl_trials = _expand_dl_baseline(dl_models, dl_config or {}, task_type)
    # TS trials are likewise baseline-only in Phase 1 (M2); the guard above
    # ensures non-baseline strategies never reach this point with ts_models.
    ts_trials = _expand_ts_baseline(ts_models, task_type)

    if strategy_type == "baseline":
        # ``_expand_baseline`` is family-aware so a single call covers both
        # ML and DL tokens; the ``search_space`` acts as a per-model override
        # (ML: hyperparameter overrides; DL: arch/opt/train section overrides).
        merged_overrides = dict(search_space or {})
        for token in dl_models:
            if token not in merged_overrides and (dl_config or {}).get(token):
                merged_overrides[token] = dl_config[token]
        trials = _expand_baseline(selected_models, tuning_defaults, merged_overrides)
        total_trials = len(trials)
        if total_trials == 0:
            raise HTTPException(status_code=422, detail="Baseline produced no trials — check selected_models")
        await _persist_trials(db, exp, task, trials, eval_metrics, test_size=test_size, cv_folds=cv_folds)
        await db.commit()
        _launch_concurrent(exp.id, modeling_task_id)
    elif strategy_type == "grid_search":
        ml_trials = _expand_grid_search(ml_models, tuning_defaults, search_space, max_trials)
        dl_numbered = _renumber_trials(dl_trials, start=len(ml_trials) + 1)
        ts_numbered = _renumber_trials(ts_trials, start=len(ml_trials) + len(dl_numbered) + 1)
        trials = ml_trials + dl_numbered + ts_numbered
        total_trials = len(trials)
        if total_trials == 0:
            raise HTTPException(
                status_code=422,
                detail="Grid search produced no trials — provide search_space or pick models with grid_values defined",
            )
        await _persist_trials(db, exp, task, trials, eval_metrics, test_size=test_size, cv_folds=cv_folds)
        await db.commit()
        _launch_concurrent(exp.id, modeling_task_id)
    elif strategy_type == "bayesian_search":
        # ML part runs bayesian; DL+TS parts run baseline concurrently (persisted now).
        non_ml_trials = dl_trials + _renumber_trials(ts_trials, start=len(dl_trials) + 1)
        if non_ml_trials:
            await _persist_trials(db, exp, task, non_ml_trials, eval_metrics, test_size=test_size, cv_folds=cv_folds)
        total_trials = _count_bayesian_trials(ml_models, budget_config, max_trials) + len(non_ml_trials)
        await db.commit()
        if non_ml_trials:
            _launch_concurrent(exp.id, modeling_task_id)
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
    else:
        raise HTTPException(status_code=422, detail=f"Unsupported strategy_type: {strategy_type!r}")

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

    submitted: list[dict[str, Any]] = []
    strategy_types: list[str] = []
    total_trials = 0
    for idx, spec in enumerate(strategies, start=1):
        strategy_type = spec.get("strategy_type")
        if strategy_type not in ("baseline", "grid_search", "bayesian_search"):
            raise HTTPException(
                status_code=422,
                detail="strategy_type must be baseline|grid_search|bayesian_search",
            )
        selected_models = list(spec.get("selected_models") or [])
        if not selected_models:
            raise HTTPException(
                status_code=422,
                detail=f"{strategy_type} requires at least one selected model",
            )

        label = {
            "baseline": "基线",
            "grid_search": "网格",
            "bayesian_search": "贝叶斯",
        }[strategy_type]
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
        elif family == "ts":
            spec = get_ts_model_spec(model_type) or {}
            per_model = overrides.get(model_type) or {}
            # Build default hyperparameters from registry param defaults.
            defaults = {p["name"]: p["default"] for p in spec.get("params", []) if "default" in p}
            hyperparameters = {**defaults, **per_model}
            trials.append({
                "family": "ts",
                "model_type": model_type,
                "hyperparameters": hyperparameters,
                "trial_no": idx,
                "search_meta": {"strategy": "baseline", "grid_index": None, "family": "ts"},
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
        template = tuning_defaults.get(model_type) or {}
        family = template.get("family") or resolve_model_family(model_type) or "ml"
        if family == "dl":
            logger.warning("Grid search: DL model %s is baseline-only, skipping", model_type)
            continue
        elif family == "ts":
            logger.warning("Grid search: TS model %s is baseline-only, skipping", model_type)
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


def _expand_ts_baseline(
    ts_models: list[str],
    task_type: str,
) -> list[dict[str, Any]]:
    """One trial per TS model using registry param defaults.

    Mirrors ``_expand_dl_baseline`` — emits trials with ``family='ts'`` so
    ``_persist_trials`` creates a ``PlatformTask(kind='ts_train')``.
    Full trainer execution arrives in M6/Task 18.
    """
    trials = _expand_baseline(
        selected_models=list(ts_models),
        tuning_defaults={},
        overrides={},
    )
    # Tag task_type for downstream TS trainer selection.
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
    DL trials create a ``DLTrainingTask`` + ``PlatformTask(kind='dl_train')``;
    TS trials reuse the ``TrainingTask`` shape as a stub domain record (no ``TSTrainingTask`` ORM yet, see M6) + ``PlatformTask(kind='ts_train')``.
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
        elif family == "ts":
            # M6/Task 18 Option A fix: bypass create_training_task_record() validation
            # (which only knows ML/DL tokens) by inserting a TrainingTask stub directly.
            # Now uses _create_ts_training_task_record — symmetric with the DL helper.
            domain_task = await _create_ts_training_task_record(
                db=db,
                dataset_id=task.dataset_id,
                target_column=task.target_column or "y",
                task_type=task.task_type,
                model_type=trial["model_type"],
                config=trial["hyperparameters"],
                test_size=test_size,
                cv_folds=cv_folds,
                eval_metrics=eval_metrics,
            )
            kind = "ts_train"
            payload_ref = f"ts_train:{domain_task.id}"
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

        # Build base run params shared across all families.
        run_params: dict[str, Any] = {
            "family": family,
            "model_type": trial["model_type"],
            "hyperparameters": trial["hyperparameters"],
            "dataset_id": task.dataset_id,
            "target_column": task.target_column,
            "task_type": task.task_type,
            "eval_metrics": eval_metrics,
        }

        # For ts family, propagate per-task forecasting metadata from
        # ModelingTask.config["time_series"] into run.params so that
        # ts_service.run_ts_executor can build TSMeta via its primary path
        # (run.params["time_series"]) without re-querying the task.
        if family == "ts":
            ts_config = (task.config or {}).get("time_series")
            if ts_config is None:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "ModelingTask.config['time_series'] is required for ts family "
                        "experiments. Set timestamp_col, target_col, freq, horizon, "
                        "lookback, validation, and interval_levels on the ModelingTask "
                        "before dispatching a ts batch."
                    ),
                )
            run_params["time_series"] = ts_config

        run = ExperimentRun(
            experiment_id=exp.id,
            params=run_params,
            status="PENDING",
            trial_no=trial["trial_no"],
            search_meta=trial["search_meta"],
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


async def _create_ts_training_task_record(
    *,
    db: AsyncSession,
    dataset_id: str,
    target_column: str,
    task_type: str,
    model_type: str,
    config: dict[str, Any],
    test_size: float,
    cv_folds: int = 5,
    eval_metrics: list[str] | None = None,
) -> TrainingTask:
    """Create a TrainingTask stub row for a ts family trial without validation.

    Mirrors ``_create_dl_training_task_record`` but inserts a ``TrainingTask``
    row directly (bypassing ``create_training_task_record``'s model-type guard,
    which only knows ML/DL tokens and would reject arima/ets/etc.).

    The returned row serves as the domain stub referenced by
    ``PlatformTask.payload_ref = "ts_train:{id}"``.  The ts executor
    (``ts_service.run_ts_executor``) locates the active ``ExperimentRun`` via
    ``ExperimentRun.task_id == platform_task_id`` and does not use this stub
    row directly — it is kept for auditability and symmetry with the ML/DL path.
    """
    import uuid as _uuid_mod

    short_id = str(_uuid_mod.uuid4())[:8]
    stub = TrainingTask(
        name=f"{model_type}_{short_id}",
        dataset_id=dataset_id,
        model_type=model_type,
        target_column=target_column,
        hyperparameters=dict(config) if config else {},
        test_size=float(test_size),
        cv_folds=int(cv_folds),
        eval_metrics=list(eval_metrics) if eval_metrics else [],
        status="PENDING",
    )
    db.add(stub)
    await db.flush()
    await db.refresh(stub)
    return stub


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------

def _launch_concurrent(experiment_id: str, modeling_task_id: str) -> None:
    """Fire one asyncio coroutine per persisted PENDING run."""
    asyncio.create_task(_run_concurrent_batch(experiment_id, modeling_task_id))


async def _run_concurrent_batch(experiment_id: str, modeling_task_id: str) -> None:
    """Launch all PENDING runs of an experiment in parallel and wait for completion."""
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

    await asyncio.gather(
        *(
            _execute_single_trial(dti, pti, rid, experiment_id, kind=kind)
            for kind, dti, pti, rid in full_entries
        ),
        return_exceptions=True,
    )
    await _finalise_batch(experiment_id, modeling_task_id)


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
    from app.services.experiment_service import update_run_metrics
    from app.scheduler.executors import get_executor

    await update_platform_task_status(platform_task_id, "RUNNING")

    try:
        async with async_session_factory() as db:
            run = (
                await db.execute(select(ExperimentRun).where(ExperimentRun.id == run_id))
            ).scalar_one_or_none()
            if run and run.started_at is None:
                run.started_at = datetime.now(timezone.utc)
                run.status = "RUNNING"
                await db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not mark run %s RUNNING: %s", run_id, exc)

    try:
        executor = get_executor(kind)
        result = await executor(domain_task_id, platform_task_id)
        metrics = _normalise_run_metrics(result.get("metrics") or {})
        async with async_session_factory() as db:
            await update_run_metrics(db, run_id, metrics, status="SUCCESS")
            await db.commit()
        await update_platform_task_status(platform_task_id, "SUCCESS", metrics=metrics)
        # Mirror legacy → V3 native logs so the Run Inspector keeps working
        # even if the legacy training_tasks row is later purged (CASCADE).
        await _mirror_logs_to_v3(domain_task_id=domain_task_id, run_id=run_id)
        return {"run_id": run_id, "status": "SUCCESS", "metrics": metrics}
    except Exception as exc:  # noqa: BLE001
        logger.error("Tuning trial %s failed: %s", run_id, exc, exc_info=True)
        try:
            async with async_session_factory() as db:
                await update_run_metrics(db, run_id, {}, status="FAILED")
                await db.commit()
        except Exception:
            pass
        await update_platform_task_status(platform_task_id, "FAILED", error=str(exc))
        # Even on failure: capture whatever logs the trainer wrote before it
        # blew up — failure diagnosis is the inspector's primary job.
        await _mirror_logs_to_v3(domain_task_id=domain_task_id, run_id=run_id)
        return {"run_id": run_id, "status": "FAILED"}


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


def _normalise_run_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Expose common leaderboard metric keys for both ML and DL runs."""
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
        if target not in normalised and isinstance(normalised.get(source), (int, float)):
            normalised[target] = float(normalised[source])
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
        done = status_map.get("SUCCESS", 0) + status_map.get("FAILED", 0)

        triggered_explain = False
        if total > 0 and done >= total:
            exp = (
                await db.execute(
                    select(PlatformExperiment).where(PlatformExperiment.id == experiment_id)
                )
            ).scalar_one_or_none()
            if exp:
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
            if objective_metric and objective_metric in metrics:
                try:
                    return float(metrics[objective_metric])
                except (TypeError, ValueError):
                    pass
            # Fallback: lowest trial_no first so order is deterministic.
            return float(run.trial_no or 0)

        # Filter out runs already explained so re-runs of the finaliser are idempotent.
        candidates = [
            r for r in runs if not (r.metrics or {}).get("shap_importances")
        ]
        candidates.sort(key=_score, reverse=reverse)
        ml_candidates: list[ExperimentRun] = []
        for run in candidates:
            if not run.task_id:
                continue
            platform_task = (
                await db.execute(
                    select(PlatformTask).where(PlatformTask.id == run.task_id)
                )
            ).scalar_one_or_none()
            if platform_task and platform_task.kind == "train":
                ml_candidates.append(run)

        top_runs = ml_candidates[: max(0, int(top_k))]
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

    # Dispatch outside the DB transaction so asyncio.create_task() fires cleanly.
    for platform_task_id, kind, payload_ref in dispatches:
        try:
            await dispatch_platform_task(platform_task_id, kind, payload_ref, 3)
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
        _run_bayesian_search(
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
    max_trials = budget_config.get("max_trials")

    async with async_session_factory() as db:
        task = await _get_task_or_404(db, modeling_task_id)
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
            sampler=TPESampler(seed=random_state),
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

                outcome = await _execute_single_trial(
                    domain_task_id,
                    platform_task_id,
                    run_id,
                    experiment_id,
                    kind="train",
                )
                value = (outcome.get("metrics") or {}).get(objective_metric)
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
