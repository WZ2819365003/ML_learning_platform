"""AI-assisted task report generation."""
from __future__ import annotations

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
from app.services import ai_report_service
from app.services.modeling_task_service import set_task_final_evaluation_state


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


async def test_generate_ai_report_requires_configured_key(db):
    with pytest.raises(HTTPException) as exc:
        await ai_report_service.generate_ai_task_report(
            db,
            "any-task",
            settings=_settings(doubao_api_key=""),
        )

    assert exc.value.status_code == 503
    assert "ARK_API_KEY" in str(exc.value.detail)


async def test_generate_ai_report_prompt_asks_for_judgement_not_a_skeleton(db, monkeypatch):
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
            "### 1.1 综合判断\n\n"
            "#### 1.1.1 任务结论\n\n"
            "总分：82/100。本任务可以进入小流量验证。\n\n"
            "#### 1.1.2 主要依据\n\n"
            "最终测试 accuracy 是核心依据。\n\n"
            "## 第二章 过程与评价\n\n"
            "### 2.1 训练过程\n\n"
            "#### 2.1.1 过程结论\n\n"
            "训练曲线显示 loss 下降。\n\n"
            "|模型|分数|\n|---|---|\n|random_forest|0.81|\n\n"
            "## 第三章 建议\n\n"
            "### 3.1 后续工作\n\n"
            "#### 3.1.1 验证建议\n\n"
            "建议补充稳定性验证。"
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

    # What it asks for instead.
    assert "===== 范本开始 =====" in prompt_text
    assert "## 数据集概况" in prompt_text
    assert "参照系" in prompt_text
    assert "多数类基线" in prompt_text
    assert "选择分" in prompt_text          # selection vs final-test separation

    # Facts computed server-side and handed over rather than left to the model.
    assert "reference_frames" in prompt_text
    assert "readiness" in prompt_text

    # Still enforced: the model writes prose, the frontend renders the tables
    # and charts, and model identifiers stay untranslated.
    assert "不要输出表格、代码块或图表" in prompt_text
    assert "模型名保留原始标识" in prompt_text

    # The task's own facts still reach it.
    assert "客户流失预测" in prompt_text
    assert "random_forest" in prompt_text
    assert "tenure" in prompt_text

    assert result["task_id"] == task_id
    assert result["model"] == "doubao-test"
    assert "## 第三章 建议" in result["markdown"]
    assert "**总分：82/100。**" in result["markdown"]
    assert result["report_schema_version"] == "ai_report.rich.v1"
    assert result["archive_id"]
    assert any(item["key"] == "ai_score" for item in result["headline_metrics"])
    assert not any(chart["id"] == "feature_importance" for chart in result["charts"])
    assert any(chart["id"] == "training_curves" for chart in result["charts"])
    assert any(chart["id"] == "roc_curve" for chart in result["charts"])
    assert any(chart["id"] == "prediction_curve" for chart in result["charts"])
    assert all(chart["id"] != "leaderboard_top_runs" for chart in result["charts"])
    assert all(chart["id"] != "run_status_distribution" for chart in result["charts"])
    table_ids = {table["id"] for table in result["tables"]}
    assert table_ids == {"data_profile", "parameter_settings", "metric_comparison"}
    assert not any(table["id"] == "model_training_summary" for table in result["tables"])
    assert not any(table["id"] == "metric_glossary" for table in result["tables"])
    assert not any(table["id"] == "model_training_process" for table in result["tables"])
    assert not any(table["id"] == "recommendation_plan" for table in result["tables"])
    assert not any(table["id"] == "experiments" for table in result["tables"])
    parameter_table = next(table for table in result["tables"] if table["id"] == "parameter_settings")
    parameter_columns = [column["key"] for column in parameter_table["columns"]]
    assert "key_params" in parameter_columns
    assert any("random_forest" in row["model_type"] for row in parameter_table["rows"])
    metric_table = next(table for table in result["tables"] if table["id"] == "metric_comparison")
    metric_columns = [column["key"] for column in metric_table["columns"]]
    assert "test_accuracy" in metric_columns
    assert "test_f1" in metric_columns
    assert "test_roc_auc" in metric_columns
    assert "test_rmse" not in metric_columns
    assert "test_mae" not in metric_columns
    blocks = result["report_blocks"]
    assert blocks[0]["type"] == "markdown"
    assert blocks[0]["id"] == "conclusion"
    assert any(block["id"] == "task_scope" for block in blocks)
    assert any(block["id"] == "process_chapter" for block in blocks)
    assert any(block["id"] == "data_profile_explanation" for block in blocks)
    assert any(block["id"] == "model_training_process_explanation" for block in blocks)
    assert any(block["id"] == "effect_summary" for block in blocks)
    assert not any(block["id"] == "ai_explanation" for block in blocks)
    assert any(block["id"] == "parameter_explanation" for block in blocks)
    assert any(block["type"] == "table" and block["table_id"] == "data_profile" for block in blocks)
    assert any(block["type"] == "table" and block["table_id"] == "parameter_settings" for block in blocks)
    assert any(block["type"] == "table" and block["table_id"] == "metric_comparison" for block in blocks)
    assert any(block["type"] == "chart" and block["chart_id"] == "training_curves" for block in blocks)
    assert any(block["type"] == "chart" and block["chart_id"] == "roc_curve" for block in blocks)
    assert any(block["type"] == "chart" and block["chart_id"] == "prediction_curve" for block in blocks)
    assert not any(block["type"] == "metric_strip" for block in blocks)
    assert not any(block["type"] == "evidence" for block in blocks)
    assert not any(block["id"] == "input_output_explanation" for block in blocks)
    assert not any(block["id"] == "metric_visual_explanation" for block in blocks)
    assert not any(block["type"] == "table" and block["table_id"] == "model_training_summary" for block in blocks)
    assert not any(block["type"] == "table" and block["table_id"] == "model_training_process" for block in blocks)
    assert not any(block["type"] == "table" and block["table_id"] == "metric_glossary" for block in blocks)
    assert not any(block["type"] == "table" and block["table_id"] == "recommendation_plan" for block in blocks)
    assert not any(block["type"] == "table" and block["table_id"] == "experiments" for block in blocks)
    assert not any(block.get("chart_id") == "leaderboard_top_runs" for block in blocks)
    assert not any(block.get("chart_id") == "run_status_distribution" for block in blocks)
    assert not any(block.get("chart_id") == "feature_importance" for block in blocks)
    report_text = "\n".join(
        block.get("markdown", "")
        for block in blocks
        if block["type"] == "markdown"
    )
    assert "|模型|分数|" not in report_text
    assert "### 1.2 任务范围" in report_text
    assert "## 第二章 过程与评价" in report_text
    assert "### 2.1 数据集概况" in report_text
    assert "### 2.2 参数设置" in report_text
    assert "### 2.3 训练过程" in report_text
    assert "### 2.4 模型评价" in report_text
    assert "## 任务目标与完成情况" not in report_text
    assert "## 数据概况与字段解释" not in report_text
    assert "## 模型训练过程" not in report_text
    # The report is a reading flow: text, then tables/charts, then more text.
    block_types = [block["type"] for block in blocks[:10]]
    assert block_types.count("markdown") >= 5
    assert "table" in block_types
    data_profile_index = next(i for i, block in enumerate(blocks) if block.get("table_id") == "data_profile")
    parameter_index = next(i for i, block in enumerate(blocks) if block.get("table_id") == "parameter_settings")
    training_chart_index = next(i for i, block in enumerate(blocks) if block.get("chart_id") == "training_curves")
    metric_table_index = next(i for i, block in enumerate(blocks) if block.get("table_id") == "metric_comparison")
    suggestions_index = next(i for i, block in enumerate(blocks) if block["id"] == "suggestions")
    assert data_profile_index < parameter_index < training_chart_index < metric_table_index < suggestions_index

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
        return (
            "# AI 建模报告\n\n"
            "## 一、结论\n\n总分：82/100。random_forest 可以进入小流量验证。\n\n"
            "## 二、解释\n\n训练曲线显示 loss 下降。\n\n"
            "## 三、建议\n\n继续验证。"
        )

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
    assert archives[0]["title"] == "AI 建模报告"
    assert restored["archive_id"] == generated["archive_id"]
    assert restored["task_id"] == task_id
    assert restored["report_blocks"][0]["id"] == "conclusion"


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


def test_rich_report_uses_run_level_final_metric_when_task_final_state_is_open():
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
        "# AI 建模报告\n\n## 一、结论\n\n总分：80/100。\n\n## 二、解释\n\n有 Run 级测试指标。\n\n## 三、建议\n\n继续验证。",
    )

    final_metric = next(item for item in payload["headline_metrics"] if item["key"] == "final_test")
    assert final_metric["value"] == "0.7597"
    best_model = next(item for item in payload["headline_metrics"] if item["key"] == "best_model")
    assert best_model["value"] == "random_forest"
    report_text = "\n".join(
        block.get("markdown", "")
        for block in payload["report_blocks"]
        if block["type"] == "markdown"
    )
    assert "尚未执行最终测试" not in report_text
    assert "Run 级最终测试指标" in report_text
    assert any("Run 级最终测试指标" in item for item in payload["evidence"])


def test_training_chart_falls_back_to_trial_level_metrics_without_history():
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

    payload = ai_report_service.build_rich_report_payload(
        context,
        "# AI 建模报告\n\n## 第一章 结论\n\n总分：90/100。\n\n## 第二章 过程与评价\n\n训练过程稳定。\n\n## 第三章 建议\n\n继续验证。",
    )

    chart = next(chart for chart in payload["charts"] if chart["id"] == "training_curves")
    assert "Trial" in chart["option"]["xAxis"]["name"]
    assert "交叉验证平均准确率" in {series["name"] for series in chart["option"]["series"]}
    assert "最终测试准确率" in {series["name"] for series in chart["option"]["series"]}
    assert any(block.get("chart_id") == "training_curves" for block in payload["report_blocks"])


def test_rich_report_uses_reader_facing_field_and_metric_labels():
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
        "# AI 建模报告\n\n## 第一章 结论\n\n总分：88/100。\n\n## 第三章 建议\n\n继续验证。",
    )

    data_table = next(table for table in payload["tables"] if table["id"] == "data_profile")
    metric_table = next(table for table in payload["tables"] if table["id"] == "metric_comparison")
    report_text = "\n".join(
        block.get("markdown", "")
        for block in payload["report_blocks"]
        if block["type"] == "markdown"
    )

    assert any(row["column"] == "月费用（monthly_charges）" for row in data_table["rows"])
    assert any(row["column"] == "空气温度（Air temperature [K]）" for row in data_table["rows"])
    assert any(row["column"] == "是否流失（churn）" for row in data_table["rows"])
    assert metric_table["rows"][0]["selection_metric"] == "交叉验证平均准确率（selection_cv_mean_accuracy）=0.8400"
    assert "random_forest 最终测试准确率为 0.8100" in report_text


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
            "markdown": "# AI 建模报告\n\n## 一、结论\n\n总分：80/100。",
            "report_schema_version": "ai_report.rich.v1",
            "headline_metrics": [{"key": "ai_score", "label": "AI 总分", "value": "80/100"}],
            "charts": [],
            "tables": [],
            "evidence": [],
            "report_blocks": [{"type": "markdown", "id": "conclusion", "markdown": "## 一、结论\n\n总分：80/100。"}],
        }

    async def fake_list(db, task_id, **_kwargs):
        return [{
            "id": "report-1",
            "task_id": task_id,
            "title": "AI 建模报告",
            "model": "doubao-test",
            "source": "doubao",
            "generated_at": "2026-07-26T00:00:00+00:00",
            "ai_score": "80/100",
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
    assert body["markdown"].startswith("# AI 建模报告")
    assert body["report_schema_version"] == "ai_report.rich.v1"
    assert body["headline_metrics"][0]["label"] == "AI 总分"
    assert body["report_blocks"][0]["id"] == "conclusion"
    assert body["archive_id"] == "report-1"
    assert list_response.status_code == 200
    assert list_response.json()["items"][0]["id"] == "report-1"
    assert detail_response.status_code == 200
    assert detail_response.json()["archive_id"] == "report-1"
