#!/usr/bin/env python3
"""Local MCP server for Doubao/Ark ML report generation.

The server uses the same report prompt and Ark client as the FastAPI backend.
It reads ARK_API_KEY or DOUBAO_API_KEY from the environment; no key is stored
in this file.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_PLATFORM = REPO_ROOT / "ml_platform"
if str(ML_PLATFORM) not in sys.path:
    sys.path.insert(0, str(ML_PLATFORM))

from fastapi import HTTPException  # noqa: E402

from app.services.ai_report_service import (  # noqa: E402
    check_doubao_reachability,
    generate_ai_report_from_context,
)

SERVER_NAME = "doubao-report"
SERVER_VERSION = "0.1.0"


TOOLS = [
    {
        "name": "check_doubao_reachability",
        "description": "Check whether the configured Doubao/Volcengine Ark API key can reach the chat completion endpoint.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "generate_ml_report",
        "description": "Generate a Chinese ML modeling report with conclusion, explanation, and suggestions from a provided task summary.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "context": {
                    "description": "Modeling task summary as an object or a string. Include metrics, winner model, data notes, and SHAP/importance when available.",
                    "anyOf": [
                        {"type": "object"},
                        {"type": "array"},
                        {"type": "string"},
                    ],
                },
                "task_id": {
                    "type": "string",
                    "description": "Optional task id carried into the response metadata.",
                },
            },
            "required": ["context"],
            "additionalProperties": False,
        },
    },
]


def _read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if line == b"":
            return None
        if line in (b"\r\n", b"\n"):
            break
        name, _, value = line.decode("ascii", errors="replace").partition(":")
        headers[name.lower()] = value.strip()

    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = sys.stdin.buffer.read(length)
    return json.loads(body.decode("utf-8"))


def _write_message(message: dict[str, Any]) -> None:
    body = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def _result(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _text_result(text: str, *, is_error: bool = False) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
    }


async def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "check_doubao_reachability":
        payload = await check_doubao_reachability()
        return _text_result(json.dumps(payload, ensure_ascii=False, indent=2))

    if name == "generate_ml_report":
        context = arguments.get("context")
        task_id = arguments.get("task_id")
        payload = await generate_ai_report_from_context(context, task_id=task_id)
        return _text_result(payload["markdown"])

    raise ValueError(f"Unknown tool: {name}")


def _handle_request(message: dict[str, Any]) -> dict[str, Any] | None:
    request_id = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}

    if request_id is None and isinstance(method, str) and method.startswith("notifications/"):
        return None

    try:
        if method == "initialize":
            requested_version = params.get("protocolVersion") or "2024-11-05"
            return _result(
                request_id,
                {
                    "protocolVersion": requested_version,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                },
            )
        if method == "ping":
            return _result(request_id, {})
        if method == "tools/list":
            return _result(request_id, {"tools": TOOLS})
        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            tool_result = asyncio.run(_call_tool(name, arguments))
            return _result(request_id, tool_result)
        return _error(request_id, -32601, f"Method not found: {method}")
    except HTTPException as exc:
        return _result(request_id, _text_result(str(exc.detail), is_error=True))
    except Exception as exc:  # noqa: BLE001 - MCP tools should report errors as tool output
        return _result(request_id, _text_result(f"{type(exc).__name__}: {exc}", is_error=True))


def main() -> None:
    while True:
        message = _read_message()
        if message is None:
            break
        response = _handle_request(message)
        if response is not None:
            _write_message(response)


if __name__ == "__main__":
    main()
