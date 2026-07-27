"""HTTP contract for explicit task-level final evaluation."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.models.database import ModelingTask, get_db
from app.services import final_evaluation_service


@pytest.fixture
def app_with_db(session_factory):
    app = create_app()

    async def _override_get_db():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = _override_get_db
    return app


async def test_task_get_exposes_public_final_evaluation_state(
    session_factory, app_with_db
):
    async with session_factory() as db:
        task = ModelingTask(
            name="finalized-task",
            task_type="classification",
            objective_metric="accuracy",
            config={
                "user_setting": True,
                "_final_evaluation": {
                    "state": "FINALIZED",
                    "version": 1,
                    "winner_run_id": "winner-1",
                },
            },
        )
        db.add(task)
        await db.commit()
        task_id = task.id

    async with AsyncClient(
        transport=ASGITransport(app=app_with_db), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/v3/tasks/{task_id}")

    assert response.status_code == 200
    assert response.json()["config"] == {"user_setting": True}
    assert response.json()["final_evaluation"]["state"] == "FINALIZED"


async def test_task_finalize_route_returns_service_result(
    session_factory, app_with_db, monkeypatch
):
    async with session_factory() as db:
        task = ModelingTask(
            name="open-task",
            task_type="classification",
            objective_metric="accuracy",
        )
        db.add(task)
        await db.commit()
        task_id = task.id

    async def fake_finalize(db, modeling_task_id, **_kwargs):
        assert modeling_task_id == task_id
        return {
            "status": "finalized",
            "final_evaluation": {
                "state": "FINALIZED",
                "version": 1,
                "winner_run_id": "winner-1",
            },
        }

    monkeypatch.setattr(
        final_evaluation_service, "finalize_task_winner", fake_finalize
    )
    async with AsyncClient(
        transport=ASGITransport(app=app_with_db), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/api/v3/tasks/{task_id}/final-evaluation"
        )

    assert response.status_code == 200
    assert response.json()["final_evaluation"]["state"] == "FINALIZED"
