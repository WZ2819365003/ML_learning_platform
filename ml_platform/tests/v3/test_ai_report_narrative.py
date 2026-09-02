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


class TestGenerateNarrativeReport:
    @pytest.fixture
    def context(self):
        runs = [
            {"run_id": "a", "status": "SUCCESS", "rank": 1, "model_type": "xgboost",
             "objective_value": 72.0,
             "metrics": {"history": [{"epoch": 1, "val_loss": 1.0}],
                         "cv_avg_rmse": 72.0, "cv_std_rmse": 1.0}},
            {"run_id": "b", "status": "SUCCESS", "rank": 2, "model_type": "lstm",
             "objective_value": 90.0, "metrics": {}},
        ]
        return {
            "runs": runs,
            "leaderboard": runs,
            "task": {"name": "T", "objective_metric": "rmse", "target_column": "y"},
            "dataset": {"row_count": 10, "column_count": 2, "column_names": ["y", "x_lag_1"]},
            "run_status_counts": {"SUCCESS": 2},
            "_target_stats": {"mean": 100.0, "min": 1.0, "max": 2.0},
            "_readiness": {"score": 60, "checks": []},
        }

    async def test_overall_report_is_generated_before_the_run_reports(self, context):
        order = []

        async def call(messages):
            body = messages[1]["content"]
            # Only the overall report carries a leaderboard section.
            order.append("overview" if "## 模型表现" in body else "run")
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
        ctx = {"runs": [{"run_id": str(i), "status": "SUCCESS", "rank": i,
                         "model_type": "m", "metrics": {}} for i in range(8)]}
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
        # going to write — never the numbers, the tables or the other models.
        async def call(messages):
            if "lstm" in messages[1]["content"]:
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

    async def test_reports_how_many_runs_were_covered(self, context):
        async def call(messages):
            return "{}"

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




