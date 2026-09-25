"""A3 — subprocess sandbox behaviour shared by config-exec and data-pipeline."""
import asyncio
import time

import pandas as pd
import pytest

from app.core.user_code_executor import (
    UserCodeError,
    UserCodeTimeout,
    run_user_code,
)


async def test_event_loop_stays_responsive_during_kill():
    """While a malicious loop burns in the sandbox, the event loop must keep
    scheduling other coroutines — the exact regression A3 fixes."""
    ticks = 0

    async def heartbeat():
        nonlocal ticks
        for _ in range(20):
            await asyncio.sleep(0.05)
            ticks += 1

    hb = asyncio.create_task(heartbeat())
    with pytest.raises(UserCodeTimeout):
        await run_user_code(mode="config", code="while True: pass", timeout_s=1)
    await hb
    # ~1s of sandbox burn → the 50ms heartbeat must have kept firing.
    assert ticks == 20


async def test_pipeline_transforms_dataframe(tmp_path):
    src = tmp_path / "src.csv"
    out = tmp_path / "out.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}).to_csv(src, index=False)

    payload = await run_user_code(
        mode="pipeline",
        code="df['c'] = df['a'] * 10\nresult = df[df['a'] > 1]",
        timeout_s=30,
        input_path=str(src),
        output_path=str(out),
    )
    assert payload["rows"] == 2 and payload["cols"] == 3
    roundtrip = pd.read_csv(out)
    assert list(roundtrip.columns) == ["a", "b", "c"]
    assert roundtrip["c"].tolist() == [20, 30]


async def test_pipeline_rejects_non_dataframe_result(tmp_path):
    src = tmp_path / "src.csv"
    pd.DataFrame({"a": [1]}).to_csv(src, index=False)
    with pytest.raises(UserCodeError, match="DataFrame"):
        await run_user_code(
            mode="pipeline", code="result = 42", timeout_s=30,
            input_path=str(src), output_path=str(tmp_path / "out.csv"),
        )


async def test_pipeline_rejects_empty_result(tmp_path):
    src = tmp_path / "src.csv"
    pd.DataFrame({"a": [1]}).to_csv(src, index=False)
    with pytest.raises(UserCodeError, match="为空"):
        await run_user_code(
            mode="pipeline", code="result = df[df['a'] > 99]", timeout_s=30,
            input_path=str(src), output_path=str(tmp_path / "out.csv"),
        )


async def test_user_print_does_not_corrupt_protocol():
    payload = await run_user_code(
        mode="config",
        code="print('hello {\"ok\": false}')\nconfig = {'selected_models': ['a']}",
        timeout_s=10,
    )
    assert payload["config"]["selected_models"] == ["a"]
    assert "hello" in payload.get("stdout", "")


async def test_filesystem_access_blocked():
    with pytest.raises(UserCodeError):
        await run_user_code(
            mode="config",
            code="config = {'selected_models': [open('/etc/passwd').read()]}",
            timeout_s=10,
        )


async def test_timeout_kill_is_prompt():
    start = time.monotonic()
    with pytest.raises(UserCodeTimeout):
        await run_user_code(mode="config", code="while True: pass", timeout_s=1)
    assert time.monotonic() - start < 5


async def test_unknown_mode_is_structured_error():
    with pytest.raises(UserCodeError, match="未知执行模式"):
        await run_user_code(mode="nope", code="config = {}", timeout_s=10)
