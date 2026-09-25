"""
AutoML Service — batch-submit training candidates from the registry YAML.

Flow (asyncio / dev mode, no Celery worker required):
  1. Load candidate list from registry/automl_candidates.yaml
  2. For each candidate, create:
       - TrainingTask (domain record)
       - ExperimentRun (for leaderboard)
       - PlatformTask (unified visibility)
  3. Fire asyncio tasks concurrently — each runs _run_training_sync_by_id
     and on completion writes back to ExperimentRun + refreshes leaderboard.

When Celery workers are available (Stage 5), replace the asyncio launch
with submit_task() and drop the fire_and_forget_automl_run calls.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import ExperimentRun, PlatformExperiment, async_session_factory
from app.services.training_service import create_training_task_record
from app.scheduler.task_runner import register_domain_task, update_platform_task_status

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).parent.parent.parent / "registry" / "automl_candidates.yaml"

# Models whose training direction can be inferred from the objective metric
_REGRESSION_METRICS = {"rmse", "mae", "mse", "mape", "r2"}


def load_candidates(task_type: str = "classification") -> list[dict]:
    """Load and return AutoML candidates for the given task type."""
    from app.core.trainer import detect_task_type, list_available_models

    if not _REGISTRY_PATH.exists():
        raise FileNotFoundError(f"AutoML registry not found: {_REGISTRY_PATH}")
    with open(_REGISTRY_PATH) as f:
        data = yaml.safe_load(f)
    candidates = data.get(task_type, [])
    valid_models = set(list_available_models())
    filtered_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        model_type = candidate.get("model_type")
        if model_type not in valid_models:
            logger.warning("Skipping AutoML candidate %r: model is not registered", model_type)
            continue
        if detect_task_type(model_type) != task_type:
            logger.warning(
                "Skipping AutoML candidate %r: expected %s model, got %s",
                model_type, task_type, detect_task_type(model_type),
            )
            continue
        filtered_candidates.append(candidate)
    if not filtered_candidates:
        raise ValueError(f"No candidates defined for task_type={task_type!r}")
    return filtered_candidates


def warm_start_capable() -> list[str]:
    """Return model types that support incremental training via warm_start."""
    if not _REGISTRY_PATH.exists():
        return []
    with open(_REGISTRY_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("warm_start_capable", [])


# ---------------------------------------------------------------------------
# The dispatch half of this module was removed when AutoML moved onto the V3
# batch pipeline (M3-3). It ran its own asyncio loop and wrote ExperimentRun and
# PlatformTask in two separate steps — no claim, no atomic terminal state — and
# created runs without ``evaluation_mode``, so its results sat outside the
# selection/final separation and could never be compared or finalised.
#
# AutoML is now ``strategy_type="automl"`` in tuning_service, which reuses the
# same persist → schedule → M2c write-back path as every other batch. What
# survives here is the part that was always worth keeping: reading the
# candidate registry.
# ---------------------------------------------------------------------------
