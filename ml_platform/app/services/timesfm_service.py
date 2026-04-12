"""TimesFM service — Google Research foundation model for time series forecasting.

The real TimesFM 1.3.0 package requires Python 3.10-3.11. The main FastAPI
process runs Python 3.13, so we invoke the model through a dedicated Python
3.10 virtual environment (ml_platform/timesfm_env/) via subprocess.

This keeps the main event loop fully non-blocking: the subprocess call runs
in a ThreadPoolExecutor and the result is returned as a plain dict.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from app.utils.storage_paths import resolve_runtime_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SERVICE_DIR = Path(__file__).parent          # ml_platform/app/services/
_RUNNER_SCRIPT = _SERVICE_DIR / "timesfm_runner.py"  # standalone inference script

# Python 3.10 venv created during setup
_VENV_PYTHON = (
    Path(__file__).parent.parent.parent        # ml_platform/
    / "timesfm_env" / "Scripts" / "python.exe"
)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="timesfm")


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _check_available() -> bool:
    """Return True if the timesfm venv Python and runner script both exist."""
    return _VENV_PYTHON.exists() and _RUNNER_SCRIPT.exists()


# ---------------------------------------------------------------------------
# Status query  (dynamic — no module-level cache so hot-reload works)
# ---------------------------------------------------------------------------

def get_timesfm_status() -> dict[str, Any]:
    available = _check_available()
    return {
        "available": available,
        "loaded": False,   # subprocess model — cold-start each call (model cached by process)
        "install_command": (
            None if available
            else (
                "py -3.10 -m venv ml_platform/timesfm_env && "
                "ml_platform/timesfm_env/Scripts/pip install timesfm[torch] torch "
                "--index-url https://download.pytorch.org/whl/cpu"
            )
        ),
        "backend": "google/timesfm-1.0-200m-pytorch (PyTorch 3.10 venv)",
    }


# ---------------------------------------------------------------------------
# Synchronous subprocess call (runs in executor)
# ---------------------------------------------------------------------------

def _run_forecast_subprocess(payload: dict) -> dict:
    """Call timesfm_runner.py in the 3.10 venv and parse its stdout JSON."""
    payload_json = json.dumps(payload)
    try:
        proc = subprocess.run(
            [str(_VENV_PYTHON), str(_RUNNER_SCRIPT)],
            input=payload_json,
            capture_output=True,
            text=True,
            timeout=300,   # 5 min max (model load on first call ~90s)
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("TimesFM inference timed out (>5 min).") from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"TimesFM venv Python not found at {_VENV_PYTHON}. "
            "Run the setup steps in README."
        ) from exc

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        # Parse error from stdout JSON if present
        try:
            out = json.loads(stdout)
            if not out.get("ok"):
                raise ValueError(out.get("error", "Unknown error from TimesFM runner"))
        except (json.JSONDecodeError, ValueError):
            raise RuntimeError(
                f"TimesFM runner exited with code {proc.returncode}. "
                f"stderr: {stderr[:400]}"
            )

    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"TimesFM runner returned invalid JSON. stdout: {proc.stdout[:200]}"
        ) from exc

    if not out.get("ok"):
        raise ValueError(out.get("error", "TimesFM runner returned an error"))

    return out["result"]


# ---------------------------------------------------------------------------
# Public async entry point
# ---------------------------------------------------------------------------

async def run_forecast(
    file_path: str,
    value_column: str,
    time_column: str | None,
    horizon: int,
    frequency: str,
) -> dict[str, Any]:
    """Run a TimesFM zero-shot forecast asynchronously via subprocess."""
    if not _check_available():
        raise RuntimeError(
            "TimesFM venv not found. "
            "Create it with: py -3.10 -m venv ml_platform/timesfm_env && "
            "ml_platform/timesfm_env/Scripts/pip install timesfm[torch] torch"
        )

    payload = {
        "file_path":    str(resolve_runtime_path(file_path)),
        "value_column": value_column,
        "time_column":  time_column,
        "horizon":      horizon,
        "frequency":    frequency,
    }

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        _run_forecast_subprocess,
        payload,
    )
