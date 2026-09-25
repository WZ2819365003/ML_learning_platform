"""Parent-side sandbox driver for user code (blueprint A3).

Every execution gets a fresh short-lived subprocess (``python -I -E``) so a
``while True`` in user code can never block the FastAPI event loop — the
wall-clock timeout SIGKILLs the whole process group and the API keeps
serving. Both 代码配置 (config_exec_service) and 数据 Pipeline
(data_pipeline_service) funnel through this single entry point.

Isolation notes: ``-I`` drops script dir / PYTHONPATH / user site; the child
applies best-effort rlimits; results come back only as JSON over stdout.
Per the blueprint, Linux containers are the production isolation target —
this is the process-level floor, portable to dev machines.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from contextlib import suppress
from pathlib import Path


class UserCodeError(ValueError):
    """User code failed (syntax/runtime/contract violation)."""


class UserCodeTimeout(UserCodeError):
    """User code exceeded its wall-clock budget and was killed."""


_RUNNER_PATH = Path(__file__).with_name("user_code_runner.py")
# Hard cap on what we read back from the child — oversized output is truncated
# rather than buffered into API memory.
_MAX_STDOUT_BYTES = 256 * 1024
_MAX_STDERR_CHARS = 2000


async def run_user_code(
    *,
    mode: str,
    code: str,
    timeout_s: float,
    input_path: str | None = None,
    output_path: str | None = None,
) -> dict:
    """Execute ``code`` in a fresh sandbox subprocess and return its payload.

    Raises UserCodeTimeout on wall-clock overrun (process group killed) and
    UserCodeError for any structured failure from the child. Both subclass
    ValueError so existing route handlers keep mapping them to HTTP 400.
    """
    request = json.dumps(
        {
            "mode": mode,
            "code": code,
            "timeout_s": timeout_s,
            "input_path": input_path,
            "output_path": output_path,
        },
        ensure_ascii=False,
    ).encode()

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-E",
        str(_RUNNER_PATH),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,  # own process group → killpg reaps children too
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(request), timeout=timeout_s
        )
    except (asyncio.TimeoutError, TimeoutError):
        _kill_process_group(proc)
        await proc.wait()
        raise UserCodeTimeout(
            f"代码执行超时（>{timeout_s:g}s），已强制终止。请缩小数据量或简化逻辑。"
        ) from None

    stdout = stdout[-_MAX_STDOUT_BYTES:]
    payload = _parse_response(stdout)
    if payload is None:
        err = stderr.decode(errors="replace")[-_MAX_STDERR_CHARS:].strip()
        raise UserCodeError(
            f"代码执行进程异常退出（exit {proc.returncode}）"
            + (f": {err}" if err else "")
        )
    if not payload.get("ok"):
        raise UserCodeError(str(payload.get("error") or "代码执行失败"))
    return payload


def _parse_response(stdout: bytes) -> dict | None:
    """The child's response is the LAST JSON line on stdout."""
    for line in reversed(stdout.decode(errors="replace").strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        with suppress(json.JSONDecodeError):
            parsed = json.loads(line)
            if isinstance(parsed, dict) and "ok" in parsed:
                return parsed
    return None


def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    with suppress(ProcessLookupError, PermissionError, OSError):
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)  # pgid == pid (new session)
        else:  # Windows fallback — kills the direct child only
            proc.kill()
