"""Experiment routes -- query MLflow experiments and runs."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.config import get_settings

router = APIRouter(prefix="/experiments", tags=["Experiments"])


def _get_mlflow_client():
    """Get MLflow tracking client. Raises 503 if unavailable."""
    try:
        from mlflow.tracking import MlflowClient
        settings = get_settings()
        return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    except ImportError:
        raise HTTPException(status_code=503, detail="MLflow is not installed")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {e}")


@router.get("/list")
async def list_experiments() -> list[dict[str, Any]]:
    """List all MLflow experiments."""
    client = _get_mlflow_client()
    try:
        experiments = client.search_experiments()
        return [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage,
            }
            for exp in experiments
        ]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to query experiments: {e}")


@router.get("/runs")
async def list_runs(
    experiment_name: str = Query(default="ml_platform", description="Experiment name"),
    max_results: int = Query(default=50, ge=1, le=200),
) -> list[dict[str, Any]]:
    """List runs in an experiment with params and metrics."""
    client = _get_mlflow_client()
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            return []

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=max_results,
            order_by=["start_time DESC"],
        )
        return [
            {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
            }
            for run in runs
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to query runs: {e}")


@router.get("/runs/{run_id}")
async def get_run_detail(run_id: str) -> dict[str, Any]:
    """Get detailed info about a specific MLflow run."""
    client = _get_mlflow_client()
    try:
        run = client.get_run(run_id)
        return {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
            "tags": dict(run.data.tags),
            "artifact_uri": run.info.artifact_uri,
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Run not found: {e}")
