"""Two-tier report: one verdict plus one narrative per model.

The chart-placeholder tests are the ones that matter. Charts stay
backend-generated and the model only picks *where* one goes; if it could emit
chart data it would happily draw a loss curve that never happened — an error
that raises nothing and cannot be caught by reading the report.
"""
import asyncio

import pytest

from app.services import ai_report_narrative as narrative


class TestAvailableRunCharts:
    def test_offers_epoch_charts_only_when_there_is_a_history(self):
        with_history = narrative.available_run_charts(
            {"metrics": {"history": [{"epoch": 1}]}}, "regression")
        without = narrative.available_run_charts({"metrics": {}}, "regression")
        assert {"loss_history", "lr_history"} <= {c["id"] for c in with_history}
        assert {"loss_history", "lr_history"} & {c["id"] for c in without} == set()

    def test_offers_fold_scores_only_when_folds_were_persisted(self):
        # Runs trained before cv_folds was kept must not be offered this.
        assert "fold_scores" not in {
            c["id"] for c in narrative.available_run_charts({"metrics": {}}, "regression")
        }
        assert "fold_scores" in {
            c["id"] for c in narrative.available_run_charts(
                {"metrics": {"cv_folds": [{"fold": 1}]}}, "regression")
        }

    def test_prediction_curve_is_regression_only(self):
        assert "prediction_curve" in {
            c["id"] for c in narrative.available_run_charts({"metrics": {}}, "regression")}
        assert "prediction_curve" not in {
            c["id"] for c in narrative.available_run_charts({"metrics": {}}, "classification")}


class TestSelectRuns:
    def test_reads_the_leaderboard_which_is_what_the_context_actually_has(self):
        # build_task_report_context emits `leaderboard`, never `runs`. Reading
        # the wrong key is silent — an absent key is an empty list, so every
        # report came back with no sub-reports and nothing said why.
        picked = narrative.select_runs_for_reports({"leaderboard": [
            {"run_id": "b", "rank": 2}, {"run_id": "a", "rank": 1},
        ]})
        assert [r["run_id"] for r in picked] == ["a", "b"]

    def test_leaderboard_entries_need_no_status_field(self):
        # They are successful by construction; requiring status would drop them.
        assert len(narrative.select_runs_for_reports(
            {"leaderboard": [{"run_id": "a", "rank": 1}]})) == 1

    def test_keeps_only_successful_runs_best_first(self):
        picked = narrative.select_runs_for_reports({"runs": [
            {"run_id": "b", "status": "SUCCESS", "rank": 2},
            {"run_id": "f", "status": "FAILED", "rank": 1},
            {"run_id": "a", "status": "SUCCESS", "rank": 1},
        ]})
        assert [r["run_id"] for r in picked] == ["a", "b"]

    def test_caps_the_count(self):
        # A grid search can produce dozens; nobody reads dozens, and each one
        # is a model call.
        runs = [{"run_id": str(i), "status": "SUCCESS", "rank": i} for i in range(30)]
        assert len(narrative.select_runs_for_reports({"runs": runs})) == narrative._MAX_RUN_REPORTS

    def test_unranked_runs_sort_last_rather_than_crashing(self):
        picked = narrative.select_runs_for_reports({"runs": [
            {"run_id": "x", "status": "SUCCESS"},
            {"run_id": "a", "status": "SUCCESS", "rank": 1},
        ]})
        assert [r["run_id"] for r in picked] == ["a", "x"]


class TestTaskBrief:
    def test_carries_only_what_a_sub_report_needs(self):
        brief = narrative.build_task_brief({
            "task": {"target_column": "load", "task_type": "regression",
                     "objective_metric": "rmse", "dataset_name": "d.csv"},
            "target_stats": {"mean": 8897},
            "leaderboard": [
                {"rank": 1, "model_type": "xgboost", "objective_value": 72.4},
                {"rank": 2, "model_type": "lstm", "objective_value": 122.9},
            ],
            "runs": [{"run_id": "x", "metrics": {"secret": 1}}],
        })
        assert brief["target_column"] == "load"
        assert brief["target_stats"] == {"mean": 8897}
        assert brief["best_run"]["model_type"] == "xgboost"
        # Every other run's metrics would be N times the tokens in N prompts,
        # and invite drift into comparing everything with everything.
        assert "runs" not in brief

    def test_tolerates_a_leaderboard_with_no_rank_one(self):
        brief = narrative.build_task_brief({"task": {}, "leaderboard": [{"rank": 3}]})
        assert brief["best_run"] is None

    def test_tolerates_an_empty_context(self):
        assert narrative.build_task_brief({})["target_column"] is None


class TestGenerateNarrativeReport:
    @pytest.fixture
    def context(self):
        return {"runs": [
            {"run_id": "a", "status": "SUCCESS", "rank": 1, "model_type": "xgboost",
             "metrics": {"history": [{"epoch": 1}]}},
            {"run_id": "b", "status": "SUCCESS", "rank": 2, "model_type": "lstm",
             "metrics": {}},
        ]}

    async def test_overall_report_is_generated_before_the_run_reports(self, context):
        order = []

        async def call(messages):
            first = messages[1]["content"]
            order.append("overview" if "建模总报告" in first else "run")
            return "文本"  # placement pass parses this as "no charts"

        await narrative.generate_narrative_report(context, call_model=call)
        assert order[0] == "overview", "the verdict is what the page shows first"
        # Two calls per run: prose, then chart placement on the finished text.
        assert order.count("run") >= 2

    async def test_run_reports_run_concurrently(self, context):
        active = concurrent_peak = 0

        async def call(messages):
            nonlocal active, concurrent_peak
            active += 1
            concurrent_peak = max(concurrent_peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return "文本"

        await narrative.generate_narrative_report(context, call_model=call)
        assert concurrent_peak > 1

    async def test_concurrency_is_bounded(self):
        # Doubao is rate-limited; an unbounded gather over a grid search's worth
        # of runs would stampede it.
        ctx = {"runs": [{"run_id": str(i), "status": "SUCCESS", "rank": i,
                         "model_type": "m", "metrics": {}} for i in range(8)]}
        active = peak = 0

        async def call(messages):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return "文本"

        await narrative.generate_narrative_report(ctx, call_model=call)
        assert peak <= narrative._MAX_CONCURRENT_RUN_REPORTS + 1  # +1 for the overview

    async def test_one_failed_run_report_does_not_sink_the_rest(self, context):
        async def call(messages):
            body = messages[1]["content"]
            # A run prompt is the only one that opens "请为模型 <name> 写";
            # the overview embeds the whole context JSON, so matching on a bare
            # model name would fire there too and take the overview down with it.
            if "请为模型 lstm" in body:
                raise RuntimeError("上游超时")
            return "文本"

        out = await narrative.generate_narrative_report(context, call_model=call)
        assert out["overview"] == "文本"
        by_model = {r["model_type"]: r for r in out["runs"]}
        assert by_model["xgboost"]["markdown"] == "文本"
        assert by_model["lstm"]["markdown"] is None
        assert "上游超时" in by_model["lstm"]["error"]

    async def test_reports_how_many_runs_were_covered(self, context):
        async def call(messages):
            return "文本"

        out = await narrative.generate_narrative_report(context, call_model=call)
        assert out["runs_total"] == 2
        assert out["runs_reported"] == 2


class TestBuildRunCharts:
    HISTORY = [
        {"epoch": 1, "train_loss": 1.0, "val_loss": 1.2, "lr": 0.001},
        {"epoch": 2, "train_loss": 0.6, "val_loss": 0.8, "lr": 0.0005},
    ]

    def test_builds_only_the_charts_that_were_placed(self):
        # An unplaced chart is a payload shipped to the browser for nothing.
        run = {"metrics": {"history": self.HISTORY}}
        ids = [c["id"] for c in narrative.build_run_charts(run, ["loss_history"])]
        assert ids == ["loss_history"]

    def test_builds_nothing_when_the_data_is_absent(self):
        # The menu said a chart was available; if the data vanished between
        # then and now, an empty option would render as a broken frame.
        assert narrative.build_run_charts({"metrics": {}}, ["loss_history"]) == []

    def test_loss_chart_carries_both_series(self):
        run = {"metrics": {"history": self.HISTORY}}
        chart = narrative.build_run_charts(run, ["loss_history"])[0]
        assert [s["name"] for s in chart["option"]["series"]] == ["训练损失", "验证损失"]
        assert chart["option"]["series"][0]["data"] == [1.0, 0.6]

    def test_learning_rate_uses_a_log_axis(self):
        run = {"metrics": {"history": self.HISTORY}}
        chart = narrative.build_run_charts(run, ["lr_history"])[0]
        assert chart["option"]["yAxis"]["type"] == "log"

    def test_fold_chart_draws_a_mean_line(self):
        run = {"metrics": {"cv_folds": [{"fold": 1, "rmse": 70}, {"fold": 2, "rmse": 80}]}}
        chart = narrative.build_run_charts(run, ["fold_scores"])[0]
        assert chart["option"]["series"][0]["data"] == [70, 80]
        assert chart["option"]["series"][0]["markLine"]["data"][0]["yAxis"] == 75

    def test_prediction_curve_needs_at_least_two_points(self):
        one = {"metrics": {"val_scatter": {"actual": [1], "predicted": [1]}}}
        two = {"metrics": {"val_scatter": {"actual": [1, 2], "predicted": [1, 2]}}}
        assert narrative.build_run_charts(one, ["prediction_curve"]) == []
        assert len(narrative.build_run_charts(two, ["prediction_curve"])) == 1




class TestSplitRunSections:
    """The sub-report shape is fixed in code, not negotiated with the model."""

    def test_splits_on_the_two_headings(self):
        out = narrative.split_run_sections("训练过程\n收敛于第 30 轮。\n训练结果\nRMSE 约为均值的 0.8%。")
        assert out["process"] == "收敛于第 30 轮。"
        assert out["result"] == "RMSE 约为均值的 0.8%。"

    def test_tolerates_markdown_and_bold_headings(self):
        # 豆包 dresses headings inconsistently across calls; all of these are
        # the same heading and must not end up as body text.
        for heading in ("## 训练结果", "**训练结果**", "训练结果：", "### **训练结果**"):
            out = narrative.split_run_sections(f"训练过程\na\n{heading}\nb")
            assert out["result"] == "b", heading
            assert "训练结果" not in out["process"], heading

    def test_text_before_any_heading_is_kept_not_dropped(self):
        out = narrative.split_run_sections("开场白。\n训练结果\nb")
        assert "开场白。" in out["process"]


class TestBuildRunSections:
    def _charts(self, *ids):
        return [{"id": i, "option": {"series": []}, "title": i} for i in ids]

    def test_each_chart_lands_in_its_own_section(self):
        out = narrative.build_run_sections(
            "训练过程\na\n训练结果\nb",
            self._charts("loss_history", "fold_scores", "prediction_curve"),
        )
        assert [s["key"] for s in out] == ["process", "result"]
        assert [c["id"] for c in out[0]["charts"]] == ["loss_history", "fold_scores"]
        assert [c["id"] for c in out[1]["charts"]] == ["prediction_curve"]

    def test_sections_keep_their_fixed_order(self):
        # Even if the model writes 训练结果 first, the reader sees process first.
        out = narrative.build_run_sections("训练结果\nb\n训练过程\na", [])
        assert [s["title"] for s in out] == ["训练过程", "训练结果"]

    def test_an_empty_section_is_dropped_rather_than_rendered_bare(self):
        out = narrative.build_run_sections("训练过程\n只写了这一段。", [])
        assert [s["key"] for s in out] == ["process"]

    def test_a_chart_without_an_option_is_not_offered(self):
        # build_run_charts returns nothing for absent data; an empty frame reads
        # as a broken chart rather than an absent one.
        out = narrative.build_run_sections(
            "训练过程\na", [{"id": "loss_history", "option": None}],
        )
        assert out[0]["charts"] == []


class TestRunPromptAsksForInterpretationOnly:
    def test_forbids_advice_and_a_third_section(self):
        user = narrative.build_run_messages({"model_type": "xgboost"}, {})[1]["content"]
        assert "不要给任何建议" in user
        assert "不要写第三段" in user

    def test_never_mentions_charts_to_the_model(self):
        # Placement is a code decision now; mentioning figures only tempted the
        # model to shape the argument around them.
        user = narrative.build_run_messages({"model_type": "xgboost"}, {})[1]["content"]
        assert "不要提到任何图表" in user
