"""Run user Python to transform a dataset — "data pipeline as code".

Powers the workflow 导入数据 step's 「数据 Pipeline（代码）」 button: the user's code
receives the source dataset as ``df`` (pandas.DataFrame) with ``pd``/``np``
available, transforms it, and the resulting DataFrame (either the reassigned
``df`` or a ``result`` variable) is saved as a NEW dataset for training.

Restricted builtins; ``import`` / filesystem / process access are blocked
(pandas & numpy are pre-injected). Dev-tool convenience on the user's own
platform — not a hardened sandbox.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import Dataset
from app.services.data_service import _build_columns_info
from app.services.prediction_service import load_dataframe
from app.utils.storage_paths import to_portable_storage_path

_SAFE_BUILTIN_NAMES = [
    "range", "len", "list", "dict", "tuple", "set", "str", "int", "float", "bool",
    "min", "max", "sum", "sorted", "enumerate", "zip", "round", "abs", "map",
    "filter", "any", "all", "print", "True", "False", "None",
]


def _safe_builtins() -> dict[str, Any]:
    import builtins

    return {n: getattr(builtins, n) for n in _SAFE_BUILTIN_NAMES if hasattr(builtins, n)}


async def run_data_pipeline(
    db: AsyncSession,
    dataset_id: str,
    code: str,
    save_as: str | None = None,
) -> Dataset:
    """Execute ``code`` against the dataset's DataFrame and save the result as a
    new Dataset. Returns the new Dataset ORM row."""
    if not code or not code.strip():
        raise HTTPException(status_code=400, detail="代码为空")

    src = await db.get(Dataset, dataset_id)
    if src is None:
        raise HTTPException(status_code=404, detail="源数据集不存在")

    try:
        df = load_dataframe(src.file_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"无法读取源数据集: {exc}") from exc

    sandbox_globals: dict[str, Any] = {"__builtins__": _safe_builtins(), "pd": pd, "np": np}
    sandbox_locals: dict[str, Any] = {"df": df.copy()}
    try:
        exec(compile(code, "<data-pipeline>", "exec"), sandbox_globals, sandbox_locals)  # noqa: S102
    except Exception as exc:  # surface real error to the editor
        raise HTTPException(status_code=400, detail=f"Pipeline 执行失败: {type(exc).__name__}: {exc}") from exc

    out = sandbox_locals.get("result", sandbox_locals.get("df"))
    if not isinstance(out, pd.DataFrame):
        raise HTTPException(status_code=400, detail="Pipeline 需产出一个 DataFrame（重新赋值 df 或定义 result）")
    if out.empty:
        raise HTTPException(status_code=400, detail="Pipeline 产出的数据为空")

    settings = get_settings()
    base = save_as or f"{Path(src.name).stem}_pipeline"
    name = base if base.endswith(".csv") else f"{base}.csv"
    unique_name = f"{uuid.uuid4().hex[:12]}-{name}"
    dest = settings.storage_uploads / unique_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)

    dataset = Dataset(
        name=name,
        file_path=to_portable_storage_path(dest),
        file_size=dest.stat().st_size,
        row_count=len(out),
        column_count=len(out.columns),
        columns_info=_build_columns_info(out),
    )
    db.add(dataset)
    await db.flush()
    await db.refresh(dataset)
    return dataset
