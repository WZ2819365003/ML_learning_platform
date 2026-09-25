"""Run user Python to transform a dataset — "data pipeline as code".

Powers the workflow 导入数据 step's 「数据 Pipeline（代码）」 button: the user's code
receives the source dataset as ``df`` (pandas.DataFrame) with ``pd``/``np``
available, transforms it, and the resulting DataFrame (either the reassigned
``df`` or a ``result`` variable) is saved as a NEW dataset for training.

A3: execution happens in a fresh short-lived subprocess (restricted builtins,
no import, wall-clock timeout with SIGKILL) via app.core.user_code_executor.
The child loads the source file and writes the transformed CSV itself, so
DataFrames never cross the process boundary — only a small JSON summary does.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.user_code_executor import run_user_code
from app.models.database import Dataset
from app.services.data_service import _build_columns_info
from app.services.object_storage import restore_dataset_file, upload_dataset_file
from app.services.prediction_service import load_dataframe
from app.utils.storage_paths import to_portable_storage_path


async def run_data_pipeline(
    db: AsyncSession,
    dataset_id: str,
    code: str,
    save_as: str | None = None,
    owner_username: str | None = None,
) -> Dataset:
    """Execute ``code`` against the dataset's DataFrame and save the result as a
    new Dataset. Returns the new Dataset ORM row."""
    if not code or not code.strip():
        raise HTTPException(status_code=400, detail="代码为空")

    src = await db.get(Dataset, dataset_id)
    if src is None or (owner_username and src.owner_username != owner_username):
        raise HTTPException(status_code=404, detail="源数据集不存在")
    source_path = restore_dataset_file(src.id, src.file_path)
    if source_path is None:
        raise HTTPException(status_code=404, detail="源数据集文件不存在")

    settings = get_settings()
    base = save_as or f"{Path(src.name).stem}_pipeline"
    name = base if base.endswith(".csv") else f"{base}.csv"
    unique_name = f"{uuid.uuid4().hex[:12]}-{name}"
    dest = settings.storage_uploads / unique_name
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        await run_user_code(
            mode="pipeline",
            code=code,
            timeout_s=settings.pipeline_code_timeout_s,
            input_path=str(source_path),
            output_path=str(dest),
        )
    except ValueError as exc:  # UserCodeError/UserCodeTimeout included
        dest.unlink(missing_ok=True)  # never leave a half-written dataset
        raise HTTPException(status_code=400, detail=f"Pipeline 执行失败: {exc}") from exc

    try:
        out = load_dataframe(dest)
    except Exception as exc:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Pipeline 输出无法读取: {exc}") from exc

    dataset = Dataset(
        owner_username=owner_username,
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
    upload_dataset_file(dataset.id, dest)
    return dataset
