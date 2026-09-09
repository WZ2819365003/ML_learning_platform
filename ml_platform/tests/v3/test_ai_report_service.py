"""AI-assisted task report generation.

The end-to-end checks here run the real pipeline against a seeded task and
assert the payload contract the page consumes: `markdown` with {{chart:id}}
markers, `charts` as semantic specs, `headline`, `meta`, `appendix_tables`,
`run_reports` — and none of the retired `report_blocks` / `tables` /
`headline_metrics`.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.main import create_app
from app.models.database import (
    AIReportArchive,
    Dataset,
    ExperimentRun,
    ModelingTask,
    PlatformExperiment,
    get_db,
)
from app.services import ai_report_service, report_charts
from app.services.modeling_task_service import set_task_final_evaluation_state

_CONTRACT_KEYS = {
    "task_id", "archive_id", "archived_at", "generated_at", "model", "source",
    "markdown", "run_reports", "runs_total", "runs_reported", "best_run_id",
    "report_schema_version", "headline", "meta", "charts", "appendix_tables", "evidence",
}
_RETIRED_KEYS = ("report_blocks", "tables", "headline_metrics")
_CHART_KINDS = {"hbar", "dots", "hist", "stacked", "lines", "scatter_pair"}


def _settings(**overrides):
    base = {
        "doubao_api_key": "unit-test-key",
        "doubao_base_url": "https://unit.test/api/v3",
        "doubao_model": "doubao-test",
        "doubao_timeout_s": 3.0,
        "doubao_max_tokens": 1200,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


async def _seed_task(db):
    ds = Dataset(
        name="churn.csv",
        file_path="/tmp/churn.csv",
        file_size=99,
        row_count=500,
        column_count=4,
        columns_info={
            "age": {
                "dtype": "int64",
                "missing_count": 0,
                "missing_rate": 0.0,
                "unique_count": 70,
            },
            "tenure": {
                "dtype": "float64",
                "missing_count": 3,
                "missing_rate": 0.006,
                "unique_count": 120,
            },
            "monthly_charges": {
                "dtype": "float64",
                "missing_count": 0,
                "missing_rate": 0.0,
                "unique_count": 210,
            },
            "churn": {
                "dtype": "int64",
                "missing_count": 0,
                "missing_rate": 0.0,
                "unique_count": 2,
            },
        },
    )
    db.add(ds)
    await db.flush()

    task = ModelingTask(
        name="客户流失预测",
        dataset_id=ds.id,
        dataset_name=ds.name,
        target_column="churn",
        task_type="classification",
        objective_metric="accuracy",
        objective_direction="max",
        status="COMPLETED",
        summary_snapshot={"data_quality": {"missing_rate": 0.03}},
    )
    db.add(task)
    await db.flush()

    exp = PlatformExperiment(
        modeling_task_id=task.id,
        name="baseline",
        strategy_type="baseline",
        selected_models=["random_forest", "logistic_regression"],
        budget_config={"cv_folds": 5, "test_size": 0.2, "max_trials": 2},
        objective_metric="accuracy",
        objective_direction="max",
        status="COMPLETED",
    )
    db.add(exp)
    await db.flush()

    winner = ExperimentRun(
        experiment_id=exp.id,
        status="SUCCESS",
        trial_no=1,
        params={
            "model_type": "random_forest",
            "hyperparameters": {"n_estimators": 200, "max_depth": 8},
            "cv_folds": 5,
            "test_size": 0.2,
        },
        metrics={
            "selection_cv_mean_accuracy": 0.84,
            "final_test_accuracy": 0.81,
            "accuracy": 0.81,
            "f1": 0.79,
            "roc_auc": 0.86,
            "train_accuracy": 0.88,
            "validation_accuracy": 0.84,
            "history": [
                {"epoch": 1, "train_loss": 0.72, "val_loss": 0.78, "val_acc": 0.68},
                {"epoch": 2, "train_loss": 0.58, "val_loss": 0.63, "val_acc": 0.77},
                {"epoch": 3, "train_loss": 0.49, "val_loss": 0.55, "val_acc": 0.84},
            ],
            "val_roc_fpr": [0.0, 0.08, 0.24, 1.0],
            "val_roc_tpr": [0.0, 0.61, 0.83, 1.0],
            "y_true": [0, 1, 1, 0, 1],
            "y_pred": [0, 1, 0, 0, 1],
            "shap_importances": {"tenure": 0.38, "monthly_charges": 0.22},
        },
        search_meta={"evaluation_mode": "selection"},
        source_experiment_type="baseline",
    )
    db.add(winner)
    contender = ExperimentRun(
        experiment_id=exp.id,
        status="SUCCESS",
        trial_no=2,
        params={
            "model_type": "logistic_regression",
            "hyperparameters": {"C": 1.0, "penalty": "l2"},
            "cv_folds": 5,
            "test_size": 0.2,
        },
        metrics={
            "selection_cv_mean_accuracy": 0.79,
            "final_test_accuracy": 0.77,
            "accuracy": 0.77,
            "f1": 0.75,
            "roc_auc": 0.82,
            "train_accuracy": 0.8,
            "validation_accuracy": 0.79,
        },
        search_meta={"evaluation_mode": "selection"},
        source_experiment_type="baseline",
    )
    db.add(contender)
    await db.flush()

    set_task_final_evaluation_state(
        task,
        {
            "state": "FINALIZED",
            "version": 1,
            "winner_run_id": winner.id,
            "final_metrics": {"final_test_accuracy": 0.81},
        },
    )
    await db.commit()
    return task.id


def _assert_spec(chart: dict) -> None:
    assert chart["kind"] in _CHART_KINDS, chart["id"]
    assert chart["title"] and chart["caption"], chart["id"]
    assert chart["tooltip_fields"] and chart["rows"], chart["id"]
    assert report_charts.renderer_leaks(chart) == [], chart["id"]


async def test_generate_ai_report_requires_configured_key(db):
    with pytest.raises(HTTPException) as exc:
        await ai_report_service.generate_ai_task_report(
            db,
            "any-task",
            settings=_settings(doubao_api_key=""),
        )

    assert exc.value.status_code == 503
    assert "ARK_API_KEY" in str(exc.value.detail)


async def test_generate_ai_report_renders_facts_and_asks_only_for_the_gaps(db, monkeypatch):
    task_id = await _seed_task(db)
    captured = {}

    async def fake_request(settings, messages):
        # The overview is the first call; the per-model reports follow it and
        # would otherwise overwrite what this test is about.
        captured.setdefault("settings", settings)
        captured.setdefault("messages", messages)
        return (
            "# AI 建模报告\n\n"
            "## 第一章 结论\n\n"
            "总分：82/100。本任务可以进入小流量验证。\n\n"
            "|模型|分数|\n|---|---|\n|random_forest|0.81|\n\n"
            "## 第三章 建议\n\n建议补充稳定性验证。"
        )

    monkeypatch.setattr(ai_report_service, "_request_chat_completion", fake_request)

    result = await ai_report_service.generate_ai_task_report(
        db,
        task_id,
        settings=_settings(),
    )

    prompt_text = "\n".join(message["content"] for message in captured["messages"])
    assert captured["settings"].doubao_model == "doubao-test"

    # The prompt asks for judgement, not for a skeleton to be filled. It used to
    # paste a literal 第一章/1.1/1.1.1 outline and demand "每个小节至少包含一个
    # 多自然段说明", which produced uniform padded prose; those are gone.
    assert "## 第一章 结论" not in prompt_text
    assert "#### 1.1.1" not in prompt_text
    assert "多自然段" not in prompt_text
    assert "总分：xx/100" not in prompt_text

    # The task-report path no longer sends a brief. It renders the document
    # from computed facts and asks only for the <<…>> sentences, so what the
    # model receives is the finished report plus a numbered list of gaps.
    assert "===== 报告 =====" in prompt_text
    assert "只回复一个 JSON 对象" in prompt_text
    assert "不可更改" in captured["messages"][0]["content"]

    # The task's own facts reach it — inside the rendered document, as text it
    # may read but not rewrite.
    assert "客户流失预测" in prompt_text
    assert "random_forest" in prompt_text

    assert result["task_id"] == task_id
    assert result["model"] == "doubao-test"
    assert result["report_schema_version"] == "ai_report.rich.v2"
    assert result["archive_id"]

    # The payload contract: nothing more, and none of the retired shapes.
    assert set(result) == _CONTRACT_KEYS
    for gone in _RETIRED_KEYS:
        assert gone not in result, gone

    # The markdown is rendered, not returned by the model: the stub reply above
    # is not JSON, so no slot is filled, every sentence is one the backend
    # computed, and the model's table never reaches the document.
    markdown = result["markdown"]
    assert markdown.startswith("# 客户流失预测 · 建模报告")
    assert "## 结论" in markdown
    assert "<<" not in markdown, "no unfilled slot is printed"
    assert "|模型|分数|" not in markdown
    assert not re.search(r"^\|", markdown, flags=re.M), "no tables in the body"
    assert "第一章" not in markdown and "总分" not in markdown

    # One cover sentence, separate from the conclusion's opening.
    assert result["headline"].startswith("random_forest 胜出")
    assert result["headline"] not in markdown

    assert result["meta"] == {
        "task_name": "客户流失预测",
        "dataset_name": "churn.csv",
        "target_column": "churn",
        "task_type": "classification",
        "objective_metric": "accuracy",
        "run_count": 2,
        "model_count": 2,
    }

    # Every marker in the document is backed by a spec, in document order, and
    # every spec is a placed marker. This task has no folds and no target
    # histogram, so those two figures are dropped rather than shipped empty.
    placed = re.findall(r"\{\{chart:([a-z0-9_]+)\}\}", markdown)
    assert placed == ["leaderboard_bars", "field_composition", "shap_bars"]
    assert [chart["id"] for chart in result["charts"]] == placed
    for chart in result["charts"]:
        _assert_spec(chart)
    leaderboard = result["charts"][0]
    assert leaderboard["kind"] == "hbar"
    assert leaderboard["categories"] == ["random_forest", "logistic_regression"]
    assert leaderboard["series"][0]["values"] == [0.84, 0.79]
    assert {f["key"] for f in leaderboard["tooltip_fields"]} >= {"accuracy", "r2", "std", "scheme"}

    # The two wide tables live in the appendix, in the shape they always had.
    assert [table["id"] for table in result["appendix_tables"]] == ["data_profile", "parameter_settings"]
    parameter_table = result["appendix_tables"][1]
    assert "key_params" in [column["key"] for column in parameter_table["columns"]]
    assert any("random_forest" in row["model_type"] for row in parameter_table["rows"])
    assert any(row["column"] == "使用时长（tenure）" for row in result["appendix_tables"][0]["rows"])

    assert any("最终测试指标：final_test_accuracy = 0.8100" in line for line in result["evidence"])

    # One sub-report per run, each carrying only the figures it has data for.
    assert result["runs_total"] == 2 and result["runs_reported"] == 2
    by_model = {r["model_type"]: r for r in result["run_reports"]}
    assert set(by_model) == {"random_forest", "logistic_regression"}
    for report in result["run_reports"]:
        assert {"run_id", "model_type", "markdown", "charts"} <= set(report)
        assert report["markdown"].startswith(f"# {report['model_type']} · 分报告")
        assert "<<" not in report["markdown"]
        assert not re.search(r"^\|", report["markdown"], flags=re.M)
        assert not re.search(r"不值得|建议|应当|优先", report["markdown"])
        for chart in report["charts"]:
            _assert_spec(chart)
    assert [c["id"] for c in by_model["random_forest"]["charts"]] == ["loss_history"]
    assert by_model["random_forest"]["charts"][0]["y_log"] is True
    assert by_model["logistic_regression"]["charts"] == []
    assert "本模型即本次最优" in by_model["random_forest"]["markdown"]
    assert "与最优的 random_forest" in by_model["logistic_regression"]["markdown"]

    archived = (
        await db.execute(
            select(AIReportArchive).where(AIReportArchive.id == result["archive_id"])
        )
    ).scalar_one()
    assert archived.task_id == task_id
    assert archived.payload["archive_id"] == result["archive_id"]


async def test_ai_report_archive_list_and_detail(db, monkeypatch):
    task_id = await _seed_task(db)

    async def fake_request(settings, messages):
        return '{"1": "下一步：在封存测试集上复核 random_forest。"}'

    monkeypatch.setattr(ai_report_service, "_request_chat_completion", fake_request)

    generated = await ai_report_service.generate_ai_task_report(
        db,
        task_id,
        settings=_settings(),
    )
    archives = await ai_report_service.list_ai_report_archives(db, task_id)
    restored = await ai_report_service.get_ai_report_archive(db, task_id, generated["archive_id"])

    assert len(archives) == 1
    assert archives[0]["id"] == generated["archive_id"]
    assert archives[0]["task_id"] == task_id
    # Titled after the task now, not with a generic label; the archive stores
    # whatever heading the rendered report carries.
    assert "建模报告" in archives[0]["title"]
    # The spliced sentence is in the archived document, and nothing else moved.
    assert "下一步：在封存测试集上复核 random_forest。" in generated["markdown"]
    assert restored["archive_id"] == generated["archive_id"]
    assert restored["task_id"] == task_id
    assert restored["markdown"] == generated["markdown"]
    assert restored["headline"] == generated["headline"]
    assert [c["id"] for c in restored["charts"]] == [c["id"] for c in generated["charts"]]
    assert len(restored["run_reports"]) == 2
    assert restored["report_schema_version"] == "ai_report.rich.v2"
    for gone in _RETIRED_KEYS:
        assert gone not in restored, gone


async def test_ai_report_keeps_model_identifiers_untranslated(monkeypatch):
    async def fake_request(settings, messages):
        return (
            "# AI 建模报告\n\n"
            "## 一、结论\n\n总分：70/100。阿里玛 当前证据不足。\n\n"
            "## 二、解释\n\n阿里玛 的曲线需要补充。\n\n"
            "## 三、建议\n\n继续验证。"
        )

    monkeypatch.setattr(ai_report_service, "_request_chat_completion", fake_request)

    result = await ai_report_service.generate_ai_report_from_context(
        {
            "task": {"id": "task-1", "name": "销量预测", "task_type": "regression"},
            "experiments": [{"selected_models": ["ARIMA"], "strategy_type": "baseline"}],
            "leaderboard": [{"model_type": "ARIMA", "metrics": {}}],
        },
        task_id="task-1",
        settings=_settings(),
    )

    assert "阿里玛" not in result["markdown"]
    assert "ARIMA" in result["markdown"]


def test_rich_report_evidence_uses_run_level_final_metric_when_task_final_state_is_open():
    context = {
        "task": {
            "name": "糖尿病预测",
            "target_column": "Outcome",
            "task_type": "classification",
            "objective_metric": "accuracy",
            "objective_direction": "max",
            "final_evaluation": {"state": "OPEN", "version": 1},
        },
        "dataset": {"name": "diabetes.csv", "row_count": 768, "column_count": 9},
        "experiments": [],
        "run_status_counts": {"SUCCESS": 1},
        "leaderboard": [
            {
                "rank": 1,
                "run_id": "run-1",
                "model_type": "logistic_regression",
                "strategy_type": "grid_search",
                "trial_no": 1,
                "selection_metric_key": "cv_avg_accuracy",
                "selection_value": 0.7778,
                "final_test_metric_key": "accuracy",
                "final_test_value": 0.7045,
                "metrics": {"cv_avg_accuracy": 0.7778, "accuracy": 0.7045},
                "params": {"model_type": "logistic_regression"},
            },
            {
                "rank": 2,
                "run_id": "run-2",
                "model_type": "random_forest",
                "strategy_type": "grid_search",
                "trial_no": 2,
                "selection_metric_key": "cv_avg_accuracy",
                "selection_value": 0.7578,
                "final_test_metric_key": "accuracy",
                "final_test_value": 0.7597,
                "metrics": {"cv_avg_accuracy": 0.7578, "accuracy": 0.7597},
                "params": {"model_type": "random_forest"},
            }
        ],
        "successful_run_examples": [],
    }

    payload = ai_report_service.build_rich_report_payload(
        context,
        "# 糖尿病预测 · 建模报告\n\n## 结论\n\n有 Run 级测试指标。\n",
    )

    # The run-level final score is a fact the evidence list states, keyed to
    # the run it belongs to; the leaderboard order (selection score) still
    # decides the headline.
    assert any("Run 级最终测试指标：accuracy = 0.7597" in item for item in payload["evidence"])
    assert any("当前最终测试最佳：random_forest" in item for item in payload["evidence"])
    assert not any("尚未执行最终评估" in item for item in payload["evidence"])
    assert payload["headline"].startswith("logistic_regression 胜出")
    for gone in _RETIRED_KEYS:
        assert gone not in payload, gone


def test_payload_charts_are_the_specs_the_markdown_places():
    context = {
        "task": {
            "name": "设备故障预测",
            "target_column": "Target",
            "task_type": "classification",
            "objective_metric": "accuracy",
            "objective_direction": "max",
            "final_evaluation": {"state": "FINALIZED", "version": 1},
        },
        "dataset": {"name": "predictive_maintenance.csv", "row_count": 10000, "column_count": 7},
        "experiments": [],
        "run_status_counts": {"SUCCESS": 3},
        "leaderboard": [
            {
                "rank": 1,
                "run_id": "run-1",
                "model_type": "random_forest",
                "strategy_type": "grid_search",
                "trial_no": 1,
                "selection_metric_key": "selection_cv_mean_accuracy",
                "selection_value": 0.96,
                "final_test_metric_key": "final_test_accuracy",
                "final_test_value": 0.97,
                "metrics": {"selection_cv_mean_accuracy": 0.96, "final_test_accuracy": 0.97},
                "params": {"model_type": "random_forest"},
            },
            {
                "rank": 2,
                "run_id": "run-2",
                "model_type": "random_forest",
                "strategy_type": "grid_search",
                "trial_no": 2,
                "selection_metric_key": "selection_cv_mean_accuracy",
                "selection_value": 0.95,
                "final_test_metric_key": "final_test_accuracy",
                "final_test_value": 0.965,
                "metrics": {"selection_cv_mean_accuracy": 0.95, "final_test_accuracy": 0.965},
                "params": {"model_type": "random_forest"},
            },
            {
                "rank": 3,
                "run_id": "run-3",
                "model_type": "random_forest",
                "strategy_type": "bayesian_search",
                "trial_no": 1,
                "selection_metric_key": "selection_cv_mean_accuracy",
                "selection_value": 0.955,
                "final_test_metric_key": "final_test_accuracy",
                "final_test_value": 0.968,
                "metrics": {"selection_cv_mean_accuracy": 0.955, "final_test_accuracy": 0.968},
                "params": {"model_type": "random_forest"},
            },
        ],
        "successful_run_examples": [],
    }

    # No marker, no chart: a spec the page never places is payload for nothing.
    bare = ai_report_service.build_rich_report_payload(context, "# 报告\n\n## 结论\n\n三次训练。\n")
    assert bare["charts"] == []

    placed = ai_report_service.build_rich_report_payload(
        context, "# 报告\n\n## 结论\n\n三次训练。\n\n{{chart:leaderboard_bars}}\n",
    )
    assert [chart["id"] for chart in placed["charts"]] == ["leaderboard_bars"]
    chart = placed["charts"][0]
    _assert_spec(chart)
    assert chart["kind"] == "hbar"
    # Three trials of one model are three bars, told apart by a suffix.
    assert chart["categories"] == ["random_forest", "random_forest #2", "random_forest #3"]
    assert [s["name"] for s in chart["series"]] == ["交叉验证"]
    assert chart["series"][0]["values"] == [0.96, 0.95, 0.955]
    # A score metric has no "1% of the target mean" line.
    assert chart["reference_lines"] == []
    assert "option" not in chart


def test_appendix_data_profile_uses_reader_facing_field_labels():
    context = {
        "task": {
            "name": "客户流失预测",
            "target_column": "churn",
            "task_type": "classification",
            "objective_metric": "accuracy",
            "objective_direction": "max",
            "final_evaluation": {
                "final_metrics": {"final_test_accuracy": 0.81},
            },
        },
        "dataset": {
            "name": "churn.csv",
            "row_count": 500,
            "column_count": 3,
            "columns_info": {
                "Air temperature [K]": {"dtype": "float64", "missing_count": 0, "missing_rate": 0, "unique_count": 110},
                "monthly_charges": {"dtype": "float64", "missing_count": 0, "missing_rate": 0, "unique_count": 210},
                "tenure": {"dtype": "float64", "missing_count": 3, "missing_rate": 0.006, "unique_count": 120},
                "churn": {"dtype": "int64", "missing_count": 0, "missing_rate": 0, "unique_count": 2},
            },
        },
        "experiments": [],
        "run_status_counts": {"SUCCESS": 1},
        "leaderboard": [
            {
                "rank": 1,
                "run_id": "run-1",
                "model_type": "random_forest",
                "strategy_type": "baseline",
                "trial_no": 1,
                "selection_metric_key": "selection_cv_mean_accuracy",
                "selection_value": 0.84,
                "final_test_metric_key": "final_test_accuracy",
                "final_test_value": 0.81,
                "metrics": {"final_test_accuracy": 0.81, "f1": 0.79, "roc_auc": 0.86},
            }
        ],
        "successful_run_examples": [],
    }

    payload = ai_report_service.build_rich_report_payload(
        context,
        "# 客户流失预测 · 建模报告\n\n## 结论\n\n一个模型。\n",
    )

    data_table = next(table for table in payload["appendix_tables"] if table["id"] == "data_profile")
    assert any(row["column"] == "月费用（monthly_charges）" for row in data_table["rows"])
    assert any(row["column"] == "空气温度（Air temperature [K]）" for row in data_table["rows"])
    assert any(row["column"] == "是否流失（churn）" for row in data_table["rows"])
    assert any("最终测试指标：final_test_accuracy = 0.8100" in item for item in payload["evidence"])


def test_highlighted_lead_sentences_are_split_into_readable_paragraphs():
    markdown = (
        "# AI 建模报告\n\n"
        "## 第三章 建议\n\n"
        "**建议先处理类别不平衡问题。** 后续说明第一条建议。"
        "**建议扩大参数搜索空间。** 后续说明第二条建议。"
    )

    result = ai_report_service._separate_repeated_bold_leads(markdown)

    assert "后续说明第一条建议。\n\n**建议扩大参数搜索空间。**" in result


async def test_ai_report_route_returns_markdown_payload(
    session_factory,
    monkeypatch,
):
    app = create_app()

    async def _override_get_db():
        async with session_factory() as session:
            yield session

    async def fake_generate(db, task_id, **_kwargs):
        return {
            "task_id": task_id,
            "archive_id": "report-1",
            "model": "doubao-test",
            "source": "doubao",
            "generated_at": "2026-07-26T00:00:00+00:00",
            "archived_at": "2026-07-26T00:00:00+00:00",
            "markdown": "# 客户流失预测 · 建模报告\n\n## 结论\n\nrandom_forest 表现最好。\n\n{{chart:leaderboard_bars}}\n",
            "report_schema_version": "ai_report.rich.v2",
            "headline": "random_forest 胜出，准确率 0.84；各项验证检查已通过。",
            "meta": {"task_name": "客户流失预测", "dataset_name": "churn.csv", "target_column": "churn",
                     "run_count": 2, "model_count": 2},
            "charts": [{"id": "leaderboard_bars", "kind": "hbar", "title": "两个模型的准确率",
                        "caption": "random_forest 以 0.84 领先。", "unit": "准确率",
                        "tooltip_fields": [{"key": "accuracy", "label": "准确率"}],
                        "rows": [{"category": "random_forest", "accuracy": 0.84}],
                        "categories": ["random_forest"],
                        "series": [{"name": "交叉验证", "values": [0.84], "error": None, "color_role": "primary"}],
                        "reference_lines": []}],
            "appendix_tables": [],
            "evidence": [],
            "run_reports": [],
            "runs_total": 0,
            "runs_reported": 0,
            "best_run_id": None,
        }

    async def fake_list(db, task_id, **_kwargs):
        return [{
            "id": "report-1",
            "task_id": task_id,
            "title": "客户流失预测 · 建模报告",
            "model": "doubao-test",
            "source": "doubao",
            "generated_at": "2026-07-26T00:00:00+00:00",
            "ai_score": None,
        }]

    async def fake_get(db, task_id, report_id, **_kwargs):
        assert report_id == "report-1"
        payload = await fake_generate(db, task_id)
        payload["archive_id"] = report_id
        return payload

    app.dependency_overrides[get_db] = _override_get_db
    monkeypatch.setattr(ai_report_service, "generate_ai_task_report", fake_generate)
    monkeypatch.setattr(ai_report_service, "list_ai_report_archives", fake_list)
    monkeypatch.setattr(ai_report_service, "get_ai_report_archive", fake_get)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/api/v3/tasks/task-1/ai-report")
        list_response = await client.get("/api/v3/tasks/task-1/ai-reports")
        detail_response = await client.get("/api/v3/tasks/task-1/ai-reports/report-1")

    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == "task-1"
    assert body["source"] == "doubao"
    assert body["markdown"].startswith("# 客户流失预测 · 建模报告")
    assert body["report_schema_version"] == "ai_report.rich.v2"
    assert body["headline"].startswith("random_forest 胜出")
    assert body["meta"]["task_name"] == "客户流失预测"
    assert body["charts"][0]["kind"] == "hbar"
    assert "option" not in body["charts"][0]
    for gone in _RETIRED_KEYS:
        assert gone not in body, gone
    assert body["archive_id"] == "report-1"
    assert list_response.status_code == 200
    assert list_response.json()["items"][0]["id"] == "report-1"
    assert detail_response.status_code == 200
    assert detail_response.json()["archive_id"] == "report-1"
