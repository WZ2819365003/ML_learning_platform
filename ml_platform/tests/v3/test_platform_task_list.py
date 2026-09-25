from app.api.routes.platform_tasks import list_platform_tasks
from app.models.database import ExperimentRun, PlatformExperiment, PlatformTask


async def test_orphan_only_excludes_tasks_linked_to_runs(db):
    linked = PlatformTask(kind="train", status="SUCCESS", payload_ref="train:linked")
    orphan = PlatformTask(kind="predict", status="SUCCESS", payload_ref="predict:orphan")
    db.add_all([linked, orphan])
    await db.flush()

    experiment = PlatformExperiment(name="linked-experiment", strategy_type="baseline")
    db.add(experiment)
    await db.flush()
    db.add(ExperimentRun(
        experiment_id=experiment.id,
        task_id=linked.id,
        status="SUCCESS",
    ))
    await db.flush()

    result = await list_platform_tasks(
        page=1,
        page_size=20,
        kind=None,
        status=None,
        orphan_only=True,
        db=db,
    )

    assert result["total"] == 1
    assert [item["id"] for item in result["items"]] == [orphan.id]
