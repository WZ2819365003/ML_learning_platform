"""Amazon Chronos time-series forecasting service.

Uses `chronos-forecasting` (python 3.13 compatible) as the zero-shot
foundation model.  The pipeline is lazily initialised on first use and
cached at module level for the process lifetime.

Model sizes available (set MODEL_NAME env var or pass model_name param):
    amazon/chronos-t5-tiny   ~8M  params — fastest
    amazon/chronos-t5-small  ~46M params — default (good accuracy/speed)
    amazon/chronos-t5-base   ~200M params
    amazon/chronos-t5-large  ~710M params

On first call the weights are downloaded from HuggingFace Hub (~200–800 MB
depending on size) and cached under ~/.cache/huggingface/.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from app.utils.storage_paths import resolve_runtime_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("CHRONOS_MODEL", "amazon/chronos-t5-small")

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chronos")
_pipeline = None          # lazy-loaded, cached for process lifetime
_pipeline_model: str = "" # which model is loaded


# ---------------------------------------------------------------------------
# Availability / status
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """Return True if chronos-forecasting is importable."""
    try:
        import chronos  # noqa: F401
        return True
    except ImportError:
        return False


def get_status() -> dict[str, Any]:
    return {
        "available": is_available(),
        "loaded": _pipeline is not None,
        "model": _pipeline_model or DEFAULT_MODEL,
        "backend": "chronos-forecasting (CPU)",
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_pipeline(model_name: str = DEFAULT_MODEL):
    """Load (or return cached) ChronosPipeline."""
    global _pipeline, _pipeline_model

    if _pipeline is not None and _pipeline_model == model_name:
        return _pipeline

    from chronos import ChronosPipeline  # type: ignore
    import torch

    logger.info("Loading Chronos model '%s' — first call may take 1-2 min…", model_name)
    _pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )
    _pipeline_model = model_name
    logger.info("Chronos model '%s' loaded successfully.", model_name)
    return _pipeline


# ---------------------------------------------------------------------------
# Synchronous forecast (runs inside ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def _forecast_sync(
    file_path: str,
    value_column: str,
    time_column: str | None,
    horizon: int,
    frequency: str,
    model_name: str,
) -> dict[str, Any]:
    """Blocking forecast — called from run_in_executor."""
    import torch
    import pandas as pd

    # ---- Load data ----------------------------------------------------------
    runtime_file_path = resolve_runtime_path(file_path)
    suffix = runtime_file_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(runtime_file_path)
    else:
        df = pd.read_csv(runtime_file_path)

    if value_column not in df.columns:
        raise ValueError(
            f"Column '{value_column}' not found. Available: {list(df.columns)}"
        )

    series = df[value_column].dropna().to_numpy(dtype=float)
    if len(series) < 10:
        raise ValueError(f"Need ≥ 10 non-null values in '{value_column}', got {len(series)}.")

    # ---- Build time axis for display ----------------------------------------
    hist_axis: list[str] | None = None
    if time_column and time_column in df.columns:
        raw = df[time_column].astype(str).tolist()
        hist_axis = raw[: len(series)]

    # ---- Run Chronos --------------------------------------------------------
    pipeline = _load_pipeline(model_name)
    context = torch.tensor(series[np.newaxis, :], dtype=torch.float32)

    forecast = pipeline.predict(
        context,
        prediction_length=horizon,
        num_samples=20,
        limit_prediction_length=False,
    )
    # forecast: (batch=1, num_samples, horizon) → squeeze to (num_samples, horizon)
    fc_np = forecast.squeeze(0).numpy().astype(float)

    point_forecast = fc_np.mean(axis=0).tolist()
    q10 = np.quantile(fc_np, 0.1, axis=0).tolist()
    q90 = np.quantile(fc_np, 0.9, axis=0).tolist()

    return {
        "historical": series.tolist(),
        "point_forecast": point_forecast,
        "q10": q10,
        "q90": q90,
        "horizon": horizon,
        "frequency": frequency,
        "value_column": value_column,
        "model_name": model_name,
        "time_axis": {
            "historical": hist_axis,
            "forecast": None,
        },
    }


# ---------------------------------------------------------------------------
# Async public API
# ---------------------------------------------------------------------------

async def run_forecast_async(
    file_path: str,
    value_column: str,
    time_column: str | None,
    horizon: int,
    frequency: str,
    model_name: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Submit forecast to thread pool and await result."""
    if not is_available():
        raise RuntimeError(
            "chronos-forecasting is not installed. "
            "Run: pip install chronos-forecasting torch"
        )
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        _forecast_sync,
        file_path,
        value_column,
        time_column,
        horizon,
        frequency,
        model_name,
    )


async def preload_model_async(model_name: str = DEFAULT_MODEL) -> None:
    """Trigger model download / warm-up in background thread."""
    if not is_available():
        return
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _load_pipeline, model_name)
