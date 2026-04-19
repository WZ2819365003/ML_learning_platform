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
from itertools import product
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import (
    ExperimentRun,
    ModelingTask,
    PlatformExperiment,
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
    description: str | None = None,
) -> dict[str, Any]:
    """
    Create a PlatformExperiment for this batch, expand trials, and launch them.

    Returns a serialised experiment plus trial counts.  Actual training runs
    execute asynchronously through asyncio.create_task — the HTTP response
    returns immediately after runs are persisted in PENDING state.
    """
    task = await _get_task_or_404(db, modeling_task_id)
    if not task.dataset_id or not task.target_column:
        raise HTTPException(
            status_code=400,
            detail="Modeling task must have dataset_id and target_column before dispatch",
        )

    task_type = task.task_type or "classification"
    tuning_defaults = load_tuning_spaces(task_type)

    unknown_models = [m for m in selected_models if m not in tuning_defaults]
    if unknown_models:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown model_type(s): {unknown_models}. "
                f"Available for {task_type}: {sorted(tuning_defaults.keys())}"
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
    eval_metrics = _default_eval_metrics(task_type, task.objective_metric)
    max_trials = budget_config.get("max_trials") if budget_config else None
    test_size = float((budget_config or {}).get("test_size") or 0.2)

    if strategy_type == "baseline":
        trials = _expand_baseline(selected_models, tuning_defaults, search_space)
        total_trials = len(trials)
        if total_trials == 0:
            raise HTTPException(status_code=422, detail="Baseline produced no trials — check selected_models")
        await _persist_trials(db, exp, task, trials, eval_metrics, test_size=test_size)
        await db.commit()
        _launch_concurrent(exp.id, modeling_task_id)
    elif strategy_type == "grid_search":
        trials = _expand_grid_search(selected_models, tuning_defaults, search_space, max_trials)
        total_trials = len(trials)
        if total_trials == 0:
            raise HTTPException(
                status_code=422,
                detail="Grid search produced no trials — provide search_space or pick models with grid_values defined",
            )
        await _persist_trials(db, exp, task, trials, eval_metrics, test_size=test_size)
        await db.commit()
        _launch_concurrent(exp.id, modeling_task_id)
    elif strategy_type == "bayesian_search":
        total_trials = _count_bayesian_trials(selected_models, budget_config, max_trials)
        await db.commit()
        _launch_bayesian(
            experiment_id=exp.id,
            modeling_task_id=modeling_task_id,
            selected_models=selected_models,
            search_space=search_space,
            tuning_defaults=tuning_defaults,
            budget_config=budget_config,
            eval_metrics=eval_metrics,
            test_size=test_size,
        )
    else:
        raise HTTPException(status_code=422, detail=f"Unsupported strategy_type: {strategy_type!r}")

    return {
        "experiment": serialize_experiment(exp),
        "trials_planned": total_trials,
        "strategy_type": strategy_type,
    }


# ---------------------------------------------------------------------------
# Trial expansion
# ---------------------------------------------------------------------------

def _expand_baseline(
    selected_models: list[str],
    tuning_defaults: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """One trial per model, using fixed defaults (+ user overrides)."""
    overrides = overrides or {}
    trials = []
    for idx, model_type in enumerate(selected_models, start=1):
        template = tuning_defaults[model_type]
        params = dict(template.get("fixed") or {})
        params.update((overrides.get(model_type) or {}))
        trials.append({
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
        template = tuning_defaults[model_type]
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
) -> None:
    """Create TrainingTask + ExperimentRun + PlatformTask for each trial."""
    for trial in trials:
        domain_task = await create_training_task_record(
            db,
            {
                "dataset_id": task.dataset_id,
                "model_type": trial["model_type"],
                "target_column": task.target_column,
                "hyperparameters": trial["hyperparameters"],
                "test_size": test_size,
                "eval_metrics": eval_metrics,
            },
        )

        run = ExperimentRun(
            experiment_id=exp.id,
            params={
                "model_type": trial["model_type"],
                "hyperparameters": trial["hyperparameters"],
                "dataset_id": task.dataset_id,
                "target_column": task.target_column,
                "task_type": task.task_type,
                "eval_metrics": eval_metrics,
            },
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
            kind="train",
            payload_ref=f"train:{domain_task.id}",
        )
        run.task_id = platform_task.id
        await db.flush()

        trial["_domain_task_id"] = domain_task.id
        trial["_platform_task_id"] = platform_task.id
        trial["_run_id"] = run.id


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

    # Retrieve payload_ref → domain_task_id via PlatformTask lookup.
    async with async_session_factory() as db:
        full_triples: list[tuple[str, str, str]] = []
        for platform_task_id, run_id, domain_fallback in triples:
            from app.models.database import PlatformTask
            pt = (
                await db.execute(
                    select(PlatformTask).where(PlatformTask.id == platform_task_id)
                )
            ).scalar_one_or_none()
            if pt and pt.payload_ref and ":" in pt.payload_ref:
                _, _, domain_task_id = pt.payload_ref.partition(":")
                full_triples.append((domain_task_id, platform_task_id, run_id))
            elif domain_fallback:
                full_triples.append((domain_fallback, platform_task_id, run_id))

    await asyncio.gather(
        *(_execute_single_trial(dti, pti, rid, experiment_id) for dti, pti, rid in full_triples),
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
) -> dict[str, Any]:
    """Run one trial to completion and write metrics + PlatformTask state back."""
    from app.services.experiment_service import update_run_metrics
    from app.services.training_service import _run_training_sync_by_id

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
        result = await _run_training_sync_by_id(domain_task_id, platform_task_id)
        metrics = result.get("metrics") or {}
        async with async_session_factory() as db:
            await update_run_metrics(db, run_id, metrics, status="SUCCESS")
            await db.commit()
        await update_platform_task_status(platform_task_id, "SUCCESS", metrics=metrics)
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
        return {"run_id": run_id, "status": "FAILED"}


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
        top_runs = candidates[: max(0, int(top_k))]
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

    for model_type in selected_models:
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
            if max_trials is not None and global_trial_no >= max_trials:
                return
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
                )

                outcome = await _execute_single_trial(
                    domain_task_id=domain_task_id,
                    platform_task_id=platform_task_id,
                    run_id=run_id,
                    experiment_id=experiment_id,
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
