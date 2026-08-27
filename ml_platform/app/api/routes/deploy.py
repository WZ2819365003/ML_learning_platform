"""Model deployment and inference routes.

Two routers:
- deploy_router (prefix="/deploy") — registered under /api → /api/deploy/...
- inference_router (prefix="/inference") — registered at root → /inference/...
"""

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import current_username_from_authorization, owner_scope_username
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
from app.services import ensemble_service

deploy_router = APIRouter(prefix="/deploy", tags=["Model Deployment"])
inference_router = APIRouter(prefix="/inference", tags=["Inference"])


@deploy_router.get("/assets", response_model=UnifiedDeploymentListResponse)
async def list_unified_deployments_route(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    runtime_type: str | None = Query(default=None, pattern="^(ml|dl)$"),
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """List ML and DL deployments through one standardized view."""
    return await list_unified_deployments(
        db=db,
        page=page,
        page_size=page_size,
        runtime_type=runtime_type,
        owner_username=owner_scope_username(username),
    )


# ---------------------------------------------------------------------------
# Ensemble (multi-model) deployments
# ---------------------------------------------------------------------------
# Separate endpoints rather than a mode flag on the single-model ones: an
# ensemble has members and weights instead of a task_id, and the response
# carries which members actually contributed to a prediction.
#
# These MUST stay above `POST /deploy/{task_id}`. FastAPI matches routes in
# registration order, so a literal path declared after a single-segment path
# parameter is unreachable: the request lands on the parameterised handler
# with task_id="ensembles" and 404s looking for a model by that name.

class EnsembleMemberSpec(BaseModel):
    """One member of a proposed ensemble."""

    domain_task_id: str
    family: str = "ml"
    weight: float = 0.0
    run_id: str | None = None
    model_type: str | None = None


class EnsembleCreateRequest(BaseModel):
    modeling_task_id: str
    name: str
    description: str | None = None
    members: list[EnsembleMemberSpec]


@deploy_router.post("/ensembles", status_code=201, summary="Create a weighted ensemble deployment")
async def create_ensemble_route(
    body: EnsembleCreateRequest,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
) -> dict:
    return await ensemble_service.create_ensemble(
        db,
        body.modeling_task_id,
        name=body.name,
        description=body.description,
        members=[m.model_dump() for m in body.members],
        owner_username=owner_scope_username(username),
    )


@deploy_router.get("/ensembles", summary="List ensemble deployments")
async def list_ensembles_route(
    modeling_task_id: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
) -> dict:
    items = await ensemble_service.list_ensembles(
        db,
        modeling_task_id=modeling_task_id,
        owner_username=owner_scope_username(username),
    )
    return {"items": items, "total": len(items)}


@deploy_router.delete("/ensembles/{ensemble_id}", summary="Delete an ensemble deployment")
async def delete_ensemble_route(
    ensemble_id: str,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
) -> dict:
    return await ensemble_service.delete_ensemble(
        db, ensemble_id, owner_username=owner_scope_username(username)
    )


@inference_router.post("/ensembles/{ensemble_id}/predict", summary="Predict with an ensemble")
async def ensemble_predict_route(
    ensemble_id: str,
    request: InferenceRequest,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
) -> dict:
    return await ensemble_service.run_ensemble_inference(
        db,
        ensemble_id,
        request.rows,
        owner_username=owner_scope_username(username),
    )


@deploy_router.post("/{task_id}", response_model=DeploymentResponse)
async def deploy_model(
    task_id: str,
    body: DeployRequest,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """Deploy a trained model and get prediction URLs."""
    result = await create_deployment(
        task_id=task_id,
        name=body.name,
        description=body.description,
        max_batch_size=body.max_batch_size,
        db=db,
        owner_username=owner_scope_username(username),
    )
    return DeploymentResponse(**result)


# NOTE: /list must be declared before /{deployment_id} to prevent FastAPI
# from matching the literal string "list" as a deployment_id path parameter.
@deploy_router.get("/list", response_model=DeploymentListResponse)
async def list_deployments_route(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """List all model deployments with pagination."""
    return await list_deployments(
        page=page,
        page_size=page_size,
        db=db,
        owner_username=owner_scope_username(username),
    )


@deploy_router.delete("/{deployment_id}")
async def delete_deployment_route(
    deployment_id: str,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """Delete a deployment and evict it from the model cache."""
    await delete_deployment(
        deployment_id=deployment_id,
        db=db,
        owner_username=owner_scope_username(username),
    )
    return {"message": "Deployment deleted", "id": deployment_id}


@deploy_router.patch("/{deployment_id}/status")
async def update_status_route(
    deployment_id: str,
    status: str = Query(..., pattern="^(active|paused)$"),
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """Pause or resume a deployment."""
    await update_deployment_status(
        deployment_id=deployment_id,
        status=status,
        db=db,
        owner_username=owner_scope_username(username),
    )
    return {"message": f"Status updated to {status}"}


@inference_router.post("/{deployment_id}/predict", response_model=InferenceJobResponse)
async def predict_route(
    deployment_id: str,
    body: InferenceRequest,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """Submit prediction request (url1). Returns result immediately."""
    result = await run_inference(
        deployment_id=deployment_id,
        rows=body.rows,
        include_probabilities=body.include_probabilities,
        db=db,
        owner_username=owner_scope_username(username),
    )
    return InferenceJobResponse(**result)


@inference_router.post("/{deployment_id}/batch-predict")
async def batch_predict_route(
    deployment_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
) -> dict:
    """Submit a CSV for asynchronous batch prediction.

    Returns immediately with a job id; poll the status endpoint. Unlike the
    synchronous route this never materialises the whole file in memory.
    """
    from app.services.batch_prediction_service import create_batch_job_from_upload

    return await create_batch_job_from_upload(
        db,
        deployment_id=deployment_id,
        file=file,
        owner_username=owner_scope_username(username),
    )


@inference_router.get("/{deployment_id}/batch-predict/{job_id}")
async def batch_predict_status_route(
    deployment_id: str,
    job_id: str,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
) -> dict:
    from app.services.batch_prediction_service import get_batch_job, serialize_batch_job

    return serialize_batch_job(
        await get_batch_job(
            db,
            deployment_id,
            job_id,
            owner_username=owner_scope_username(username),
        )
    )


@inference_router.get("/{deployment_id}/batch-predict/{job_id}/download")
async def batch_predict_download_route(
    deployment_id: str,
    job_id: str,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """Stream the result CSV. 409 while the job is still running so the caller
    never receives a truncated file that looks complete."""
    from pathlib import Path

    from app.services.batch_prediction_service import get_batch_job

    job = await get_batch_job(
        db,
        deployment_id,
        job_id,
        owner_username=owner_scope_username(username),
    )
    if job.status != "completed" or not job.result_path:
        raise HTTPException(
            status_code=409,
            detail=f"预测任务尚未完成（当前状态：{job.status}），结果暂不可下载",
        )
    from app.services.batch_prediction_service import restore_batch_file

    path = Path(job.result_path)
    if not path.exists():
        # The row says the job completed, so the result existed. A rebuilt
        # storage volume must not turn that into a 404 when object storage
        # still holds the copy we uploaded.
        if restore_batch_file(path, job_id, kind="result") is None:
            raise HTTPException(
                status_code=404, detail="结果文件不存在，且对象存储中无可恢复副本"
            )
    return FileResponse(
        path, media_type="text/csv", filename=f"predictions-{job_id[:8]}.csv"
    )


@inference_router.get("/{deployment_id}/result/{job_id}", response_model=InferenceJobResponse)
async def get_result_route(
    deployment_id: str,
    job_id: str,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(current_username_from_authorization),
):
    """Get prediction result by job ID (url2)."""
    result = await get_inference_result(
        deployment_id=deployment_id,
        job_id=job_id,
        db=db,
        owner_username=owner_scope_username(username),
    )
    return InferenceJobResponse(**result)
