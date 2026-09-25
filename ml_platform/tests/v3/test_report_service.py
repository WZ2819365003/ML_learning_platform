"""M3-1 — the finalized-task Markdown report.

The load-bearing test here is ``test_selection_and_final_never_share_a_table``.
Everything else is presentation; that one guards a correctness property. See
docs/superpowers/specs/2026-07-22-modeling-task-report-design.md.
"""
from __future__ import annotations

import re

import pytest
from fastapi import HTTPException

from app.models.database import (
    Dataset,
    ExperimentRun,
    ModelingTask,
    PlatformExperiment,
)
from app.services import report_service
from app.services.modeling_task_service import set_task_final_evaluation_state


async def _seed(
    db,
    *,
    finalized=True,
    final_metrics=None,
    shap=None,
    dataset_version_id="dsv-1",
    with_leaderboard=True,
    row_count=500,
):
    ds = Dataset(name="churn.csv", file_path="/tmp/churn.csv", file_size=99, row_count=row_count)
    db.add(ds)
    await db.flush()

    task = ModelingTask(
        name="客户流失预测",
        dataset_id=ds.id,
        dataset_name=ds.name,
        dataset_version_id=dataset_version_id,
        target_column="churn",
        task_type="classification",
        objective_metric="accuracy",
        objective_direction="max",
    )
    db.add(task)
    await db.flush()

    exp = PlatformExperiment(
        modeling_task_id=task.id, name="baseline-1", strategy_type="baseline",
        dataset_id=ds.id, objective_metric="accuracy", status="COMPLETED",
    )
    db.add(exp)
    await db.flush()

    winner_metrics = {
        "selection_cv_mean_accuracy": 0.8412,
        "final_test_accuracy": 0.8003,
        "final_test_f1": 0.7719,
    }
    if shap is not None:
        winner_metrics["shap_importances"] = shap

    winner = ExperimentRun(
        experiment_id=exp.id,
        status="SUCCESS",
        trial_no=1,
        params={
            "model_type": "random_forest",
            "hyperparameters": {"n_estimators": 200, "max_depth": 8},
            "cv_folds": 5,
            "test_size": 0.2,
            "random_state": 42,
        },
        metrics=winner_metrics,
        search_meta={"evaluation_mode": "selection"},
    )
    db.add(winner)
    await db.flush()

    if with_leaderboard:
        db.add(
            ExperimentRun(
                experiment_id=exp.id,
                status="SUCCESS",
                trial_no=2,
                params={"model_type": "logistic_regression", "hyperparameters": {"C": 1.0}},
                metrics={"selection_cv_mean_accuracy": 0.7955},
                search_meta={"evaluation_mode": "selection"},
            )
        )
        await db.flush()

    if finalized:
        set_task_final_evaluation_state(
            task,
            {
                "state": "FINALIZED",
                "winner_run_id": winner.id,
                "evaluation_id": "eval-abc",
                "finalized_at": "2026-07-22T03:00:00+00:00",
                "final_metrics": (
                    final_metrics
                    if final_metrics is not None
                    else {"final_test_accuracy": 0.8003, "final_test_f1": 0.7719}
                ),
            },
        )
    await db.commit()
    return task.id, winner.id


def _tables(markdown: str) -> list[str]:
    """Split the document into contiguous Markdown table blocks."""
    blocks, current = [], []
    for line in markdown.splitlines():
        if line.strip().startswith("|"):
            current.append(line)
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return blocks


# ---------------------------------------------------------------------------
# The property that matters
# ---------------------------------------------------------------------------

async def test_selection_and_final_never_share_a_table(db):
    """Selection metrics and final-test metrics are different measurements.

    Rendering them in one table invites the reader to compare them and conclude
    the model "got worse on test" — the exact misreading the sealed hold-out
    design exists to prevent. No table may carry both.
    """
    task_id, _ = await _seed(db)
    md = await report_service.build_task_report(db, task_id)

    for table in _tables(md):
        has_final = "0.8003" in table or "0.7719" in table
        has_selection = "0.8412" in table or "0.7955" in table
        assert not (has_final and has_selection), (
            "a table mixes final-test and selection metrics:\n" + table
        )


async def test_candidate_section_warns_against_comparing(db):
    """The warning is what stops a reader who only skims the tables."""
    task_id, _ = await _seed(db)
    md = await report_service.build_task_report(db, task_id)

    section = md.split("## 3. 候选模型对比")[1].split("##")[0]
    assert "不可与" in section and "直接比较" in section
    assert "选择阶段" in section


async def test_final_section_states_the_holdout_was_opened_once(db):
    task_id, _ = await _seed(db)
    md = await report_service.build_task_report(db, task_id)
    section = md.split("## 2. 最终评估")[1].split("## 3.")[0]
    assert "封存测试集" in section
    assert "仅开启一次" in section


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

async def test_unfinalized_task_is_refused_with_guidance(db):
    task_id, _ = await _seed(db, finalized=False)
    with pytest.raises(HTTPException) as exc:
        await report_service.build_task_report(db, task_id)
    assert exc.value.status_code == 409
    # Assert on wording unique to THIS refusal. "最终评估" alone also appears in
    # the broken-data 409, so a loose check passes even when the finalize gate
    # is removed entirely — the test would then be green for the wrong reason.
    detail = str(exc.value.detail)
    assert "尚未执行最终评估" in detail, "409 must name the missing precondition"
    assert "不完整" not in detail, "refused for the wrong reason (data integrity, not gating)"


async def test_missing_task_is_404(db):
    with pytest.raises(HTTPException) as exc:
        await report_service.build_task_report(db, "no-such-task")
    assert exc.value.status_code == 404


async def test_deleted_winner_run_is_refused(db, session_factory):
    """A report whose subject no longer exists must fail loudly, not silently
    fall back to whatever is currently ranked first."""
    task_id, winner_id = await _seed(db)
    winner = await db.get(ExperimentRun, winner_id)
    await db.delete(winner)
    await db.commit()

    with pytest.raises(HTTPException) as exc:
        await report_service.build_task_report(db, task_id)
    assert exc.value.status_code == 409
    assert "不完整" in str(exc.value.detail)


async def test_winner_comes_from_finalize_not_leaderboard_rank(db, session_factory):
    """The frozen winner wins even when a later run scores higher.

    Otherwise an already-published report would change its subject as new
    experiments land — the same document describing a different model.
    """
    task_id, winner_id = await _seed(db)
    winner = await db.get(ExperimentRun, winner_id)

    # A newcomer beats the winner on the selection metric.
    db.add(
        ExperimentRun(
            experiment_id=winner.experiment_id,
            status="SUCCESS",
            trial_no=3,
            params={"model_type": "xgboost"},
            metrics={"selection_cv_mean_accuracy": 0.9999},
            search_meta={"evaluation_mode": "selection"},
        )
    )
    await db.commit()

    md = await report_service.build_task_report(db, task_id)
    headline = md.split("\n\n")[1]
    assert "random_forest" in headline, "report followed the leaderboard instead of the frozen winner"
    assert "xgboost" not in headline


# ---------------------------------------------------------------------------
# Degradation — a missing part must not kill the whole report
# ---------------------------------------------------------------------------

async def test_missing_shap_degrades_to_a_note(db):
    task_id, _ = await _seed(db, shap=None)
    md = await report_service.build_task_report(db, task_id)
    assert "未生成特征重要性" in md
    assert "## 2. 最终评估" in md, "the rest of the report must still render"


async def test_shap_is_ranked_high_to_low(db):
    """``shap_importances`` holds mean *absolute* SHAP values (shap_service
    builds it from ``mean_abs_shap``), so entries are non-negative and the
    report's 「平均绝对 SHAP 值」 column label is accurate. Ranking is simply
    descending magnitude."""
    task_id, _ = await _seed(
        db, shap={"tenure": 0.05, "monthly_charges": 0.42, "gender": 0.01}
    )
    md = await report_service.build_task_report(db, task_id)
    section = md.split("### 6. 特征重要性")[1]
    order = re.findall(r"`(\w+)`", section)
    assert order[:3] == ["monthly_charges", "tenure", "gender"], order
    assert "-" not in section.split("|")[-3], "absolute values must not render negative"


async def test_missing_dataset_version_shows_em_dash(db):
    task_id, _ = await _seed(db, dataset_version_id=None)
    md = await report_service.build_task_report(db, task_id)
    assert "数据集版本 ID | —" in md


async def test_empty_leaderboard_does_not_crash(db):
    task_id, _ = await _seed(db, with_leaderboard=False)
    md = await report_service.build_task_report(db, task_id)
    assert "## 3. 候选模型对比" in md


async def test_empty_final_metrics_does_not_crash(db):
    task_id, _ = await _seed(db, final_metrics={})
    md = await report_service.build_task_report(db, task_id)
    assert "最终评估未产生指标" in md


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

async def test_report_carries_the_essentials(db):
    task_id, _ = await _seed(db)
    md = await report_service.build_task_report(db, task_id)

    assert "客户流失预测" in md
    assert "random_forest" in md
    assert "0.8003" in md            # final metric
    assert "churn.csv" in md
    assert "| 样本量 | 500 |" in md
    assert "n_estimators" in md      # winner hyperparameters
    assert "eval-abc" in md          # evaluation id for traceability


async def test_report_route_returns_markdown(db, client_factory=None):
    """The route contract: text/markdown plus a download filename."""
    from httpx import ASGITransport, AsyncClient

    from app.main import create_app
    from app.models.database import get_db

    task_id, _ = await _seed(db)
    app = create_app()

    async def _override():
        yield db

    app.dependency_overrides[get_db] = _override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(f"/api/v3/tasks/{task_id}/report.md")

    assert resp.status_code == 200
    assert "text/markdown" in resp.headers["content-type"]
    assert "attachment" in resp.headers.get("content-disposition", "")
    assert resp.text.startswith("# 建模任务报告")
