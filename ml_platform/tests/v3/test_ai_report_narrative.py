"""Two-tier report: one verdict plus one narrative per model.

The chart-placement tests are the ones that matter. Charts stay
backend-computed and reach the page as semantic specs; if the model could emit
chart data it would happily draw a loss curve that never happened — an error
that raises nothing and cannot be caught by reading the report.
"""
import asyncio

import pytest

from app.services import ai_report_narrative as narrative


class TestPlacedCharts:
    def test_reads_markers_in_document_order(self):
        doc = "a\n\n{{chart:fold_dots}}\n\nb\n\n{{chart:leaderboard_bars}}\n"
        assert narrative.placed_chart_ids(doc) == ["fold_dots", "leaderboard_bars"]

    def test_keeps_only_placed_specs_in_document_order(self):
        specs = [{"id": "leaderboard_bars"}, {"id": "fold_dots"}, {"id": "target_hist"}]
        doc = "{{chart:fold_dots}}\n\n{{chart:leaderboard_bars}}\n"
        assert [c["id"] for c in narrative.keep_placed(specs, doc)] == ["fold_dots", "leaderboard_bars"]

    def test_an_unbuilt_marker_is_ignored(self):
        assert narrative.keep_placed([], "{{chart:fold_dots}}") == []


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


class TestGenerateNarrativeReport:
    @pytest.fixture
    def context(self):
        runs = [
            {"run_id": "a", "status": "SUCCESS", "rank": 1, "model_type": "xgboost",
             "objective_value": 72.0,
             "metrics": {"cv_avg_rmse": 72.0, "cv_std_rmse": 1.0,
                         "cv_folds": [{"fold": i, "rmse": 72.0 + (i % 2)} for i in range(1, 6)]}},
            {"run_id": "b", "status": "SUCCESS", "rank": 2, "model_type": "lstm",
             "objective_value": 90.0,
             "metrics": {"selection_val_rmse": 90.0,
                         "history": [{"epoch": 1, "val_loss": 2.0}, {"epoch": 2, "val_loss": 1.5}]}},
        ]
        return {
            "runs": runs,
            "leaderboard": runs,
            "task": {"name": "T", "objective_metric": "rmse", "target_column": "y"},
            "dataset": {"row_count": 10, "column_count": 2, "column_names": ["y", "x_lag_1"]},
            "run_status_counts": {"SUCCESS": 2},
            "_target_stats": {"mean": 100.0, "min": 1.0, "max": 2.0},
        }

    async def test_overall_report_is_generated_before_the_run_reports(self, context):
        order = []

        async def call(messages):
            body = messages[1]["content"]
            # Only the overall report carries the model-gap section.
            order.append("overview" if "## 模型差距" in body else "run")
            return "{}"

        await narrative.generate_narrative_report(context, call_model=call)
        assert order[0] == "overview", "the verdict is what the page shows first"
        assert order.count("run") == 2, "one call per model, not two"

    async def test_run_reports_run_concurrently(self, context):
        active = concurrent_peak = 0

        async def call(messages):
            nonlocal active, concurrent_peak
            active += 1
            concurrent_peak = max(concurrent_peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return "{}"

        await narrative.generate_narrative_report(context, call_model=call)
        assert concurrent_peak > 1

    async def test_concurrency_is_bounded(self):
        # Doubao is rate-limited; an unbounded gather over a grid search's worth
        # of runs would stampede it.
        ctx = {"runs": [{"run_id": str(i), "status": "SUCCESS", "rank": i, "model_type": "m",
                         "objective_value": 1.0 + i,
                         "metrics": {"history": [{"epoch": 1, "val_loss": 2.0}]}} for i in range(8)]}
        active = peak = 0

        async def call(messages):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return "{}"

        await narrative.generate_narrative_report(ctx, call_model=call)
        assert peak <= narrative._MAX_CONCURRENT_RUN_REPORTS + 1  # +1 for the overview

    async def test_a_failed_call_costs_prose_not_the_report(self, context):
        # The document is fully rendered from computed facts before the model is
        # asked for anything, so an upstream failure loses the sentences it was
        # going to write — never the numbers, the figures or the other models.
        async def call(messages):
            if "lstm · 分报告" in messages[1]["content"]:
                raise RuntimeError("上游超时")
            return '{"1": "补写的句子。"}'

        out = await narrative.generate_narrative_report(context, call_model=call)
        by_model = {r["model_type"]: r for r in out["runs"]}
        assert "lstm" in by_model["lstm"]["markdown"], "facts survive the failure"
        assert "<<" not in by_model["lstm"]["markdown"], "no unfilled slot is printed"
        assert "补写的句子。" in by_model["xgboost"]["markdown"]

    async def test_a_reply_cannot_alter_a_rendered_number(self, context):
        async def call(messages):
            return '{"1": "其实最优模型是 ARIMA，RMSE 是 99999。"}'

        out = await narrative.generate_narrative_report(context, call_model=call)
        # The reply is spliced into its slot; everything else is what the
        # backend rendered. This is the whole point of the JSON round trip.
        assert "xgboost" in out["overview"]
        assert "xgboost 表现最好" in out["overview"]

    async def test_charts_are_specs_placed_by_marker(self, context):
        async def call(messages):
            return "{}"

        out = await narrative.generate_narrative_report(context, call_model=call)
        placed = narrative.placed_chart_ids(out["overview"])
        assert [c["id"] for c in out["overview_charts"]] == placed
        assert "leaderboard_bars" in placed and "fold_dots" in placed
        by_model = {r["model_type"]: r for r in out["runs"]}
        assert [c["id"] for c in by_model["xgboost"]["charts"]] == ["fold_scores"]
        assert [c["id"] for c in by_model["lstm"]["charts"]] == ["loss_history"]
        for report in out["runs"]:
            for chart in report["charts"]:
                assert "kind" in chart and "option" not in chart

    async def test_reports_how_many_runs_were_covered(self, context):
        async def call(messages):
            return "{}"

        out = await narrative.generate_narrative_report(context, call_model=call)
        assert out["runs_total"] == 2
        assert out["runs_reported"] == 2
