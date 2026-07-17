"""Child-process entry for user-code execution (blueprint A3).

Executed as ``python -I -E user_code_runner.py`` by app.core.user_code_executor
— NEVER imported by the FastAPI process. Deliberately self-contained (stdlib +
pandas/numpy only) so ``-I`` isolated mode works without the app package.

Protocol (single JSON object on stdin → single JSON line on stdout):

    {"mode": "config",   "code": "...", "timeout_s": 5}
    {"mode": "pipeline", "code": "...", "timeout_s": 60,
     "input_path": "/abs/src.csv", "output_path": "/abs/out.csv"}

Response: ``{"ok": true, ...}`` or ``{"ok": false, "error": "..."}``. The
parent enforces the wall-clock kill; this process additionally applies
best-effort POSIX rlimits (CPU / file size / open files). Linux containers
remain the production isolation boundary — rlimits are defence in depth,
not the guarantee.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import sys

_SAFE_BUILTIN_NAMES = [
    "range", "len", "list", "dict", "tuple", "set", "str", "int", "float",
    "bool", "min", "max", "sum", "sorted", "enumerate", "zip", "round", "abs",
    "map", "filter", "any", "all", "print", "True", "False", "None",
]
_MAX_CAPTURED_STDOUT = 8 * 1024


def _safe_builtins() -> dict:
    import builtins

    return {n: getattr(builtins, n) for n in _SAFE_BUILTIN_NAMES if hasattr(builtins, n)}


def _apply_rlimits(timeout_s: float) -> None:
    """Best-effort resource caps; every limit is optional by design."""
    try:
        import resource
    except ImportError:  # non-POSIX
        return
    cpu = max(1, math.ceil(timeout_s)) + 1
    for limit, value in (
        (getattr(resource, "RLIMIT_CPU", None), (cpu, cpu)),
        # Output files capped at 1 GB — a runaway to_csv can't fill the disk.
        (getattr(resource, "RLIMIT_FSIZE", None), (1 << 30, 1 << 30)),
        (getattr(resource, "RLIMIT_NOFILE", None), (64, 64)),
    ):
        if limit is None:
            continue
        try:
            resource.setrlimit(limit, value)
        except (ValueError, OSError):
            pass  # platform refused (e.g. macOS); parent timeout still rules


def _exec_user_code(code: str, extra_names: dict) -> dict:
    """Run code under restricted builtins; return the resulting namespace."""
    sandbox_globals: dict = {"__builtins__": _safe_builtins(), **extra_names}
    sandbox_locals: dict = dict(extra_names)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(code, "<user-code>", "exec"), sandbox_globals, sandbox_locals)  # noqa: S102
    sandbox_locals["__captured_stdout__"] = buffer.getvalue()[:_MAX_CAPTURED_STDOUT]
    return sandbox_locals


def _run_config(request: dict) -> dict:
    ns = _exec_user_code(request["code"], {})
    cfg = ns.get("config")
    if not isinstance(cfg, dict):
        return {"ok": False, "error": "代码需定义一个名为 config 的 dict（至少含 selected_models 字段）"}
    try:
        cfg = json.loads(json.dumps(cfg))  # schema gate: JSON-serializable only
    except (TypeError, ValueError):
        return {"ok": False, "error": "config 必须是可 JSON 序列化的 dict（不要放函数/对象等）"}
    return {"ok": True, "config": cfg, "stdout": ns["__captured_stdout__"]}


def _load_dataframe(path: str):
    import pandas as pd

    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    if lower.endswith(".xlsx"):
        return pd.read_excel(path)
    raise ValueError(f"不支持的数据格式: {path.rsplit('.', 1)[-1]}")


def _run_pipeline(request: dict) -> dict:
    import numpy as np
    import pandas as pd

    df = _load_dataframe(request["input_path"])
    ns = _exec_user_code(request["code"], {"pd": pd, "np": np, "df": df})
    out = ns.get("result", ns.get("df"))
    if not isinstance(out, pd.DataFrame):
        return {"ok": False, "error": "Pipeline 需产出一个 DataFrame（重新赋值 df 或定义 result）"}
    if out.empty:
        return {"ok": False, "error": "Pipeline 产出的数据为空"}
    out.to_csv(request["output_path"], index=False)
    return {
        "ok": True,
        "rows": int(len(out)),
        "cols": int(len(out.columns)),
        "stdout": ns["__captured_stdout__"],
    }


def main() -> None:
    try:
        request = json.loads(sys.stdin.read())
        _apply_rlimits(float(request.get("timeout_s") or 5))
        if request.get("mode") == "config":
            response = _run_config(request)
        elif request.get("mode") == "pipeline":
            response = _run_pipeline(request)
        else:
            response = {"ok": False, "error": f"未知执行模式: {request.get('mode')!r}"}
    except BaseException as exc:  # noqa: BLE001 — everything becomes a structured error
        response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write("\n" + json.dumps(response, ensure_ascii=False))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
