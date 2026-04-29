# ml_platform/app/services/ts_service.py
"""TS Service — orchestrates ts family training within V3 platform.

Responsibilities (full implementation lands in M6):
  * Load Dataset → DataFrame
  * Validate time_series payload
  * Slice by validation strategy
  * Instantiate trainer (via ts_registry)
  * fit → predict on val → compute FORECAST_EVAL_METRICS
  * Persist model + write ExperimentRunLog + metrics_snapshot
  * Register executor("ts_train", ...) for V3 dispatch chain
"""
from __future__ import annotations

import logging
from typing import Any

from app.scheduler.executors import register_executor

logger = logging.getLogger(__name__)


async def run_ts_executor(domain_id: str, platform_task_id: str) -> dict[str, Any]:
    """V3 executor for ts family.

    Contract: domain_id is ExperimentRun.id (matches ml/dl convention).
    Stub — full impl arrives in M6 (Task 18).
    """
    raise NotImplementedError(
        "ts_service.run_ts_executor: stub. Real implementation arrives in M6/Task 18."
    )


register_executor("ts_train", run_ts_executor)
