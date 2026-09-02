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




class TestChartAxesHaveRoomToRender:
    """ECharts clips what does not fit the grid inset; it never widens it."""

    def _loss_chart(self):
        run = {"metrics": {"history": [
            {"epoch": i, "train_loss": 8_000_000 - i, "val_loss": 8_100_000 - i}
            for i in range(1, 6)
        ]}}
        return narrative.build_run_charts(run, ["loss_history"])[0]

    def test_the_y_axis_gap_fits_a_seven_digit_loss(self):
        # 8,000,000 is eleven characters; at the old 56px inset the axis read
        # ",000,000" with the leading digits cut off.
        assert self._loss_chart()["option"]["grid"]["left"] >= 80

    def test_the_x_axis_name_sits_under_the_ticks_not_past_them(self):
        # Left at the axis end it collided with the plot edge and rendered as
        # a single letter.
        axis = self._loss_chart()["option"]["xAxis"]
        assert axis["nameLocation"] == "middle"
        assert self._loss_chart()["option"]["grid"]["bottom"] > axis["nameGap"]

    def test_every_chart_reserves_the_same_room(self):
        run = {"metrics": {
            "history": [{"epoch": 1, "train_loss": 1.0, "lr": 0.1},
                        {"epoch": 2, "train_loss": 0.5, "lr": 0.05}],
            "cv_folds": [{"fold": 1, "rmse": 1.0}, {"fold": 2, "rmse": 1.1}],
            "val_scatter": {"actual": [1, 2, 3], "predicted": [1, 2, 3]},
        }}
        charts = narrative.build_run_charts(
            run, ["loss_history", "lr_history", "fold_scores", "prediction_curve"],
        )
        assert len(charts) == 4
        for chart in charts:
            assert chart["option"]["grid"]["left"] >= 80, chart["id"]


class TestExemplarDrivenPrompts:
    """The brief is a worked example, not a rule list.

    Two revisions of prescribed structure produced padding — sections written
    to be filled rather than because there was something to say. The exemplar
    carries length, depth and register instead.
    """

    def _user(self):
        return narrative.build_run_messages({"model_type": "xgboost"}, {})[1]["content"]

    def test_the_sub_report_brief_contains_a_worked_example(self):
        user = self._user()
        assert "===== 范本开始 =====" in user
        assert "## 训练过程" in user and "## 训练结果" in user

    def test_the_example_is_marked_as_another_task_and_not_to_be_copied(self):
        # An exemplar full of concrete numbers is an invitation to copy them
        # into the report as facts.
        user = self._user()
        assert "其他任务" in user
        assert "一个都不要照抄" in user

    def test_it_still_demands_a_reference_frame_for_every_metric(self):
        # The one guard that is about correctness rather than shape.
        assert "参照系" in self._user()

    def test_the_first_pass_never_mentions_figures(self):
        # Told about charts, the model bends the argument toward what can be
        # drawn; placement runs afterwards, on finished text.
        assert "不要提到任何图表" in self._user()


class TestChartPlacementSeesRenderedCharts:
    def _charts(self):
        return [{
            "id": "loss_history", "title": "训练/验证损失",
            "description": "逐轮损失。",
            "option": {"series": [{"name": "训练损失"}, {"name": "验证损失"}]},
        }]

    def test_the_placement_pass_is_shown_what_was_actually_drawn(self):
        # It used to be handed a menu of ids that *could* be built, and placed
        # figures that then rendered as nothing.
        described = narrative.describe_built_charts(self._charts())
        assert described[0]["id"] == "loss_history"
        assert described[0]["画的是"] == "训练损失、验证损失"

    def test_declining_to_place_a_chart_is_an_allowed_answer(self):
        user = narrative.build_chart_placement_messages("正文。", self._charts())[1]["content"]
        assert "就不要插它" in user
        assert "[]" in user

    def test_paragraphs_are_numbered_from_one_for_the_model(self):
        user = narrative.build_chart_placement_messages(
            "第一段。\n\n第二段。", self._charts(),
        )[1]["content"]
        assert "[第1段]" in user and "[第2段]" in user

    def test_an_empty_placement_reply_leaves_the_report_untouched(self):
        out, dropped = narrative.apply_chart_placements("正文。", [], {"loss_history"})
        assert out == "正文。"
        assert dropped == []


class TestStripLeadingTitle:
    def test_drops_the_report_title_the_model_adds_unasked(self):
        # 豆包 opens most sub-reports with "# AI 建模报告" however plainly it is
        # told not to; kept, it renders as a second title inside the page's own.
        out = narrative.strip_leading_title("# AI 建模报告\n\n## 训练过程\n正文。")
        assert out.startswith("## 训练过程")

    def test_keeps_the_section_headings(self):
        out = narrative.strip_leading_title("## 训练过程\n正文。")
        assert out.startswith("## 训练过程")

    def test_tolerates_leading_blank_lines_and_empty_input(self):
        assert narrative.strip_leading_title("\n\n# 标题\n正文。") == "正文。"
        assert narrative.strip_leading_title("") == ""
        assert narrative.strip_leading_title(None) == ""
