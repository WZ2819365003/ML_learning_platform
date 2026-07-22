"""Model deployment and inference routes.

Two routers:
- deploy_router (prefix="/deploy") — registered under /api → /api/deploy/...
- inference_router (prefix="/inference") — registered at root → /inference/...
"""

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.models.schemas import (
    DeployRequest,
    DeploymentListResponse,
    DeploymentResponse,
    InferenceJobResponse,
    InferenceRequest,
    UnifiedDeploymentListResponse,
)
from app.services.deploy_service import (
    create_deployment,
    delete_deployment,
    get_inference_result,
    list_deployments,
    run_inference,
    update_deployment_status,
)
from app.services.model_asset_service import list_unified_deployments

deploy_router = APIRouter(prefix="/deploy", tags=["Model Deployment"])
inference_router = APIRouter(prefix="/inference", tags=["Inference"])


@deploy_router.get("/assets", response_model=UnifiedDeploymentListResponse)
async def list_unified_deployments_route(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    runtime_type: str | None = Query(default=None, pattern="^(ml|dl)$"),
    db: AsyncSession = Depends(get_db),
):
    """List ML and DL deployments through one standardized view."""
    return await list_unified_deployments(
        db=db,
        page=page,
        page_size=page_size,
        runtime_type=runtime_type,
    )


@deploy_router.post("/{task_id}", response_model=DeploymentResponse)
async def deploy_model(
    task_id: str,
    body: DeployRequest,
    db: AsyncSession = Depends(get_db),
):
    """Deploy a trained model and get prediction URLs."""
    result = await create_deployment(
        task_id=task_id,
        name=body.name,
        description=body.description,
        max_batch_size=body.max_batch_size,
        db=db,
    )
    return DeploymentResponse(**result)


# NOTE: /list must be declared before /{deployment_id} to prevent FastAPI
# from matching the literal string "list" as a deployment_id path parameter.
@deploy_router.get("/list", response_model=DeploymentListResponse)
async def list_deployments_route(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List all model deployments with pagination."""
    return await list_deployments(page=page, page_size=page_size, db=db)


@deploy_router.delete("/{deployment_id}")
async def delete_deployment_route(
    deployment_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a deployment and evict it from the model cache."""
    await delete_deployment(deployment_id=deployment_id, db=db)
    return {"message": "Deployment deleted", "id": deployment_id}


@deploy_router.patch("/{deployment_id}/status")
async def update_status_route(
    deployment_id: str,
    status: str = Query(..., pattern="^(active|paused)$"),
    db: AsyncSession = Depends(get_db),
):
    """Pause or resume a deployment."""
    await update_deployment_status(deployment_id=deployment_id, status=status, db=db)
    return {"message": f"Status updated to {status}"}


@inference_router.post("/{deployment_id}/predict", response_model=InferenceJobResponse)
async def predict_route(
    deployment_id: str,
    body: InferenceRequest,
    db: AsyncSession = Depends(get_db),
):
    """Submit prediction request (url1). Returns result immediately."""
    result = await run_inference(
        deployment_id=deployment_id,
        rows=body.rows,
        include_probabilities=body.include_probabilities,
        db=db,
    )
    return InferenceJobResponse(**result)


@inference_router.post("/{deployment_id}/batch-predict")
async def batch_predict_route(
    deployment_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Submit a CSV for asynchronous batch prediction.

    Returns immediately with a job id; poll the status endpoint. Unlike the
    synchronous route this never materialises the whole file in memory.
    """
    from app.services.batch_prediction_service import create_batch_job

    content = await file.read()
    return await create_batch_job(
        db,
        deployment_id=deployment_id,
        filename=file.filename or "upload.csv",
        content=content,
    )


@inference_router.get("/{deployment_id}/batch-predict/{job_id}")
async def batch_predict_status_route(
    deployment_id: str,
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    from app.services.batch_prediction_service import get_batch_job, serialize_batch_job

    return serialize_batch_job(await get_batch_job(db, deployment_id, job_id))


@inference_router.get("/{deployment_id}/batch-predict/{job_id}/download")
async def batch_predict_download_route(
    deployment_id: str,
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Stream the result CSV. 409 while the job is still running so the caller
    never receives a truncated file that looks complete."""
    from pathlib import Path

    from app.services.batch_prediction_service import get_batch_job

    job = await get_batch_job(db, deployment_id, job_id)
    if job.status != "completed" or not job.result_path:
        raise HTTPException(
            status_code=409,
            detail=f"预测任务尚未完成（当前状态：{job.status}），结果暂不可下载",
        )
    path = Path(job.result_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="结果文件不存在")
    return FileResponse(
        path, media_type="text/csv", filename=f"predictions-{job_id[:8]}.csv"
    )


@inference_router.get("/{deployment_id}/result/{job_id}", response_model=InferenceJobResponse)
async def get_result_route(
    deployment_id: str,
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get prediction result by job ID (url2)."""
    result = await get_inference_result(
        deployment_id=deployment_id, job_id=job_id, db=db
    )
    return InferenceJobResponse(**result)
