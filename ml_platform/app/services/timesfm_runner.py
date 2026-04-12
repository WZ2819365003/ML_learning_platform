"""Standalone TimesFM inference script.

Executed by the FastAPI service as a subprocess using the Python 3.10 venv
where timesfm[torch] is installed. Reads a JSON payload from stdin, runs
the forecast, and writes a JSON result to stdout.

Usage (internal only):
    python timesfm_runner.py < input.json

Note: timesfm prints banner messages to stdout at import time; we redirect
stdout to stderr during import so only the final JSON lands on stdout.
"""

import json
import os
import sys

# Suppress timesfm's stdout banner by temporarily redirecting stdout → stderr
_real_stdout = sys.stdout
sys.stdout = sys.stderr
try:
    import timesfm as _timesfm_preload  # noqa: F401 — triggers banner now
finally:
    sys.stdout = _real_stdout  # restore before any JSON output

import pandas as pd

from app.utils.storage_paths import resolve_runtime_path

# ----- helpers ---------------------------------------------------------------

_FREQ_INT_MAP = {"high": 0, "medium": 1, "low": 2}
_FREQ_PD_MAP  = {"high": "h", "medium": "W", "low": "MS"}


def _load_df(file_path: str) -> pd.DataFrame:
    p = resolve_runtime_path(file_path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(p)
    try:
        return pd.read_csv(p)
    except Exception as exc:
        raise ValueError(f"Unsupported file format '{suffix}'.") from exc


def run(payload: dict) -> dict:
    import timesfm  # already imported and banner suppressed at module top

    file_path    = payload["file_path"]
    value_column = payload["value_column"]
    time_column  = payload.get("time_column")
    horizon      = int(payload.get("horizon", 24))
    frequency    = payload.get("frequency", "high")

    df = _load_df(file_path)

    if value_column not in df.columns:
        raise ValueError(
            f"Column '{value_column}' not found. Available: {list(df.columns)}"
        )

    series = df[value_column].dropna().astype(float)
    if len(series) == 0:
        raise ValueError(f"Column '{value_column}' has no valid numeric values.")

    history_len   = min(512, len(series))
    series_trimmed = series.iloc[-history_len:]
    freq_int      = _FREQ_INT_MAP.get(frequency, 0)

    # Load model (cached by process lifetime — subprocess is reused if possible)
    if not hasattr(run, "_model"):
        run._model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend="pytorch", horizon_len=128),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
            ),
        )

    point_forecast, quantile_forecast = run._model.forecast(
        [series_trimmed.values.tolist()],
        freq=[freq_int],
    )

    pf = point_forecast[0][:horizon].tolist()
    qf = quantile_forecast[0]  # shape (horizon, n_quantiles)
    q10 = qf[:horizon, 0].tolist() if qf is not None else None
    q90 = qf[:horizon, -1].tolist() if qf is not None else None

    historical = series_trimmed.values.tolist()

    # Build time axis
    time_axis = None
    if time_column:
        if time_column not in df.columns:
            raise ValueError(
                f"time_column '{time_column}' not found. Available: {list(df.columns)}"
            )
        time_series = pd.to_datetime(df[time_column], errors="coerce")
        time_trimmed = time_series.iloc[series_trimmed.index].reset_index(drop=True)
        hist_dates   = time_trimmed.dt.strftime("%Y-%m-%d %H:%M").tolist()
        pd_freq      = _FREQ_PD_MAP.get(frequency, "h")
        last_date    = time_trimmed.dropna().iloc[-1]
        future_idx   = pd.date_range(start=last_date, periods=horizon + 1, freq=pd_freq)[1:]
        time_axis = {
            "historical": hist_dates,
            "forecast":   future_idx.strftime("%Y-%m-%d %H:%M").tolist(),
        }

    return {
        "historical":    [float(v) for v in historical],
        "point_forecast": [float(v) for v in pf],
        "q10":           [float(v) for v in q10] if q10 else None,
        "q90":           [float(v) for v in q90] if q90 else None,
        "horizon":       horizon,
        "frequency":     frequency,
        "value_column":  value_column,
        "time_axis":     time_axis,
    }


if __name__ == "__main__":
    payload = json.load(sys.stdin)
    try:
        result = run(payload)
        json.dump({"ok": True, "result": result}, sys.stdout)
    except Exception as exc:
        json.dump({"ok": False, "error": str(exc)}, sys.stdout)
        sys.exit(1)
