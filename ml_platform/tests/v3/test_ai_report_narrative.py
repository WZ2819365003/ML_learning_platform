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


class TestApplyChartPlacements:
    MD = "训练过程\n\n第一段正文。\n\n结果解读\n\n第二段正文。"

    def test_inserts_a_marker_after_the_chosen_paragraph(self):
        out, dropped = narrative.apply_chart_placements(
            self.MD, [{"chart": "loss_history", "after_paragraph": 2}], {"loss_history"})
        parts = out.split("\n\n")
        assert parts[2] == "{{chart:loss_history}}"
        assert dropped == []

    def test_drops_a_chart_the_run_does_not_have(self):
        # An empty frame reads as a broken chart rather than as the model
        # having asked for something that does not exist.
        out, dropped = narrative.apply_chart_placements(
            self.MD, [{"chart": "loss_history", "after_paragraph": 1}], set())
        assert "chart:" not in out
        assert dropped == ["loss_history"]

    def test_drops_an_out_of_range_paragraph(self):
        out, dropped = narrative.apply_chart_placements(
            self.MD, [{"chart": "loss_history", "after_paragraph": 99}], {"loss_history"})
        assert "chart:" not in out
        assert dropped == ["loss_history"]

    def test_never_places_the_same_chart_twice(self):
        out, _ = narrative.apply_chart_placements(
            self.MD,
            [{"chart": "loss_history", "after_paragraph": 1},
             {"chart": "loss_history", "after_paragraph": 2}],
            {"loss_history"})
        assert out.count("{{chart:loss_history}}") == 1

    def test_a_malformed_reply_costs_the_figures_not_the_report(self):
        for junk in (None, "抱歉，我无法完成", {"chart": "x"}, 42):
            out, _ = narrative.apply_chart_placements(self.MD, junk, {"loss_history"})
            assert out == self.MD


class TestParsePlacements:
    def test_reads_a_bare_array(self):
        assert narrative._parse_placements('[{"chart":"a","after_paragraph":1}]') == [
            {"chart": "a", "after_paragraph": 1}]

    def test_unwraps_a_fenced_reply(self):
        # Models fence JSON however often they are told not to, and a
        # wrapped-but-correct answer should not cost the report its figures.
        assert narrative._parse_placements('```json\n[{"chart":"a","after_paragraph":1}]\n```') == [
            {"chart": "a", "after_paragraph": 1}]

    def test_finds_an_array_buried_in_prose(self):
        assert narrative._parse_placements('好的，结果是 [{"chart":"a","after_paragraph":2}] 。') == [
            {"chart": "a", "after_paragraph": 2}]

    def test_returns_empty_for_unparseable_text(self):
        assert narrative._parse_placements("我觉得不需要图") == []
        assert narrative._parse_placements("") == []


class TestSelectRuns:
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
            # "分报告" appears only in a run prompt; the overview embeds the
            # whole context JSON, so matching on a model name would fire there
            # too and take the overview down with it.
            if "分报告" in body and "lstm" in body:
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
