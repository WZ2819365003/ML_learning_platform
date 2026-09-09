"""Chart specs say what a figure shows; the renderer decides how it looks.

Every builder is fed the task-0bc692d9 fixture or a minimal run and checked
for two things: the data a hover needs is in `rows`, and nothing that belongs
to a charting library has leaked in.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.services import report_charts as rc
from tests.v3 import report_fixture

SERVICES = Path(rc.__file__).parent
_FORBIDDEN_SOURCE = re.compile(r'"grid"|nameGap|itemStyle|"#[0-9a-f]{6}"')


@pytest.fixture
def ctx():
    # Function-scoped on purpose: several tests below edit the context to
    # provoke a caption, and a shared copy leaked those edits into the
    # reading-order test that ran after them.
    return report_fixture.context()


class TestSourcePurity:
    @pytest.mark.parametrize("name", [
        "report_charts.py", "report_facts.py", "report_template.py", "ai_report_narrative.py",
    ])
    def test_no_renderer_vocabulary_in_the_backend(self, name):
        assert not _FORBIDDEN_SOURCE.search((SERVICES / name).read_text(encoding="utf-8")), name

    def test_the_leak_detector_catches_a_planted_option(self):
        assert rc.renderer_leaks({"id": "x", "series": [{"values": [1], "item" + "Style": {}}]})
        assert rc.renderer_leaks({"id": "x", "colour": "#2563eb"})
        assert rc.renderer_leaks({"id": "x", "rows": [{"category": "a", "rmse": 1.0}]}) == []


class TestLeaderboardBars:
    def test_one_series_per_validation_scheme_aligned_to_the_categories(self, ctx):
        spec = rc.leaderboard_bars(ctx)
        assert spec["kind"] == "hbar"
        assert len(spec["categories"]) == 7
        assert [s["name"] for s in spec["series"]] == ["交叉验证", "留出验证"]
        cv, holdout = spec["series"]
        assert cv["values"][:4] == [72.4673, 72.4673, 72.7856, 79.284]
        assert cv["values"][4:] == [None, None, None]
        assert holdout["values"][:4] == [None] * 4
        assert holdout["values"][4] == 132.0422
        assert cv["error"][0] == 0.8539 and holdout["error"] is None
        assert cv["color_role"] == "primary" and holdout["color_role"] == "muted"

    def test_a_rerun_gets_a_distinct_category(self, ctx):
        spec = rc.leaderboard_bars(ctx)
        assert spec["categories"][:2] == ["xgboost_regressor", "xgboost_regressor #2"]
        assert spec["categories"][5:] == ["lstm", "lstm #2"]

    def test_reference_line_is_one_percent_of_the_target_mean(self, ctx):
        spec = rc.leaderboard_bars(ctx)
        assert spec["reference_lines"] == [{"value": 88.9659, "label": "均值的 1%"}]

    def test_rows_carry_what_the_old_table_did(self, ctx):
        row = rc.leaderboard_bars(ctx)["rows"][0]
        assert row["category"] == "xgboost_regressor"
        assert row["rmse"] == 72.4673 and row["pct"] == "0.81%"
        assert row["r2"] == 0.9974 and row["std"] == 0.8539 and row["scheme"] == "交叉验证"
        keys = {f["key"] for f in rc.leaderboard_bars(ctx)["tooltip_fields"]}
        assert {"rmse", "pct", "r2", "std", "scheme"} <= keys

    def test_caption_reads_overlap_and_the_other_scheme(self, ctx):
        caption = rc.leaderboard_bars(ctx)["caption"]
        assert "全在 1% 线以内" in caption
        assert "xgboost_regressor 与 lightgbm_regressor 的误差条重叠" in caption
        assert "留出验证的三个模型在 1.5%–2.1%" in caption

    def test_non_overlapping_error_bars_are_said_so(self, ctx):
        ctx["leaderboard"][2]["metrics"]["cv_avg_rmse"] = 76.0
        ctx["leaderboard"][2]["metrics"]["selection_cv_mean_rmse"] = 76.0
        ctx["leaderboard"][2]["objective_value"] = 76.0
        assert "不相交" in rc.leaderboard_bars(ctx)["caption"]

    def test_score_metrics_get_no_percent_of_mean(self):
        ctx = {"task": {"objective_metric": "accuracy", "target_column": "y"},
               "leaderboard": [{"run_id": "a", "model_type": "rf", "objective_value": 0.84,
                                "metrics": {"selection_cv_mean_accuracy": 0.84}}]}
        spec = rc.leaderboard_bars(ctx)
        assert spec["reference_lines"] == []
        assert spec["rows"][0]["pct"] == "—"
        assert "准确率" in spec["title"]


class TestFoldDots:
    def test_one_row_per_model_from_the_run_that_kept_its_folds(self, ctx):
        spec = rc.fold_dots(ctx)
        assert spec["kind"] == "dots" and spec["mean_marker"] is True
        assert spec["categories"] == ["xgboost_regressor", "lightgbm_regressor", "random_forest_regressor"]
        # The rank-1 xgboost run has no folds; its rerun does, and is drawn once.
        assert len(spec["points"][0]["values"]) == 5
        assert spec["points"][0]["rows"][0] == {"category": "xgboost_regressor", "fold": 1,
                                                "rmse": 72.9, "mae": 52.488, "r2": 0.9974}

    def test_caption_says_who_overlaps_and_who_is_apart(self, ctx):
        caption = rc.fold_dots(ctx)["caption"]
        assert "xgboost_regressor 与 lightgbm_regressor 的五折范围重叠" in caption
        assert "random_forest_regressor 整体右移，与前两者不相交" in caption

    def test_absent_without_any_folds(self):
        assert rc.fold_dots({"task": {}, "leaderboard": [{"metrics": {"cv_avg_rmse": 1.0}}]}) is None


class TestTargetHist:
    def test_parses_the_stringified_histogram(self, ctx):
        spec = rc.target_hist(ctx)
        assert spec["kind"] == "hist"
        assert len(spec["bins"]) == 20
        assert sum(b["count"] for b in spec["bins"]) == sum(report_fixture.HIST_COUNTS)
        assert spec["bins"][0]["from"] == pytest.approx(5498.36, abs=0.01)
        assert spec["rows"][0]["count"] == 120 and spec["rows"][0]["pct"].endswith("%")

    def test_markers_carry_mean_and_quartiles(self, ctx):
        labels = [m["label"] for m in rc.target_hist(ctx)["markers"]]
        assert labels == ["均值", "P25", "中位数", "P75"]

    def test_caption_reads_shape_body_and_tail(self, ctx):
        caption = rc.target_hist(ctx)["caption"]
        assert caption.startswith("单峰")
        assert "主体在" in caption
        assert "最大值 14274 在长尾末端，只有 9 个样本" in caption

    def test_a_left_skewed_histogram_is_called_left_skewed(self, ctx):
        info = ctx["dataset"]["columns_info"]["load"]
        info["histogram"]["counts"] = json.dumps(list(reversed(report_fixture.HIST_COUNTS)))
        assert "左偏" in rc.target_hist(ctx)["caption"]

    def test_absent_without_a_histogram(self, ctx):
        ctx["dataset"]["columns_info"]["load"]["histogram"] = None
        assert rc.target_hist(ctx) is None


class TestFieldComposition:
    def test_segments_account_for_every_column(self, ctx):
        spec = rc.field_composition(ctx)
        assert spec["kind"] == "stacked"
        assert [s["name"] for s in spec["segments"]] == [
            "原始采集与日历", "周期三角变换", "滞后项", "滚动统计", "气象与交互衍生"]
        assert [s["count"] for s in spec["segments"]] == [11, 8, 4, 3, 9]
        assert "load_lag_336" in spec["segments"][2]["items"]
        assert "24 列（69%）是训练流程构造出来的" in spec["caption"]

    def test_hover_lists_the_column_names(self, ctx):
        rows = rc.field_composition(ctx)["rows"]
        assert rows[2]["items"] == "load_lag_1、load_lag_2、load_lag_48、load_lag_336"


class TestShapBars:
    def test_top_eight_with_share_of_the_leader(self, ctx):
        spec = rc.shap_bars(ctx["leaderboard"][0], "load")
        assert spec["kind"] == "hbar" and len(spec["categories"]) == 8
        assert spec["categories"][0] == "load_lag_1"
        assert spec["rows"][1] == {"category": "load_lag_2", "shap": 95.6, "pct_of_top": "9.79%"}
        assert spec["series"][0]["color_role"] == "accent"

    def test_caption_counts_the_targets_own_lags_and_rolls(self, ctx):
        caption = rc.shap_bars(ctx["leaderboard"][0], "load")["caption"]
        assert "load_lag_1 一个特征的贡献是第二名的 10 倍" in caption
        assert "前 8 里有 5 个是 load 自身的滞后或滚动项" in caption

    def test_needs_two_features(self):
        assert rc.shap_bars({"metrics": {"top_shap_importances": [{"feature": "a", "mean_abs_shap": 1}]}}) is None


class TestOverviewCharts:
    def test_five_in_reading_order(self, ctx):
        assert [c["id"] for c in rc.build_overview_charts(ctx)] == [
            "leaderboard_bars", "fold_dots", "target_hist", "field_composition", "shap_bars"]

    def test_the_shap_chart_belongs_to_the_leading_cohorts_winner(self, ctx):
        assert rc.build_overview_charts(ctx)[-1]["title"].startswith("xgboost_regressor")

    def test_no_training_curves_overlay(self, ctx):
        assert "training_curves" not in {c["id"] for c in rc.build_overview_charts(ctx)}


class TestRunCharts:
    def test_a_tree_run_gets_folds_and_shap(self, ctx):
        ids = [c["id"] for c in rc.build_run_charts(ctx["leaderboard"][2], ctx)]
        assert ids == ["fold_scores", "shap_bars"]

    def test_a_deep_run_gets_loss_and_predictions(self, ctx):
        ids = [c["id"] for c in rc.build_run_charts(ctx["leaderboard"][4], ctx)]
        assert ids == ["loss_history", "pred_vs_actual"]

    def test_fold_scores_plot_the_objective_with_a_mean_line(self, ctx):
        spec = rc.fold_scores(ctx["leaderboard"][2], "rmse")
        assert spec["series"][0]["values"][0] == 72.8147
        assert spec["reference_lines"][0]["value"] == pytest.approx(72.7856, abs=1e-3)
        assert spec["reference_lines"][0]["label"].startswith("均值")
        assert "RMSE" in spec["title"] and "rmse" not in spec["title"]
        assert set(spec["rows"][0]) == {"category", "fold", "rmse", "mae", "r2"}

    def test_fold_scores_fall_back_when_the_objective_is_absent(self, ctx):
        assert rc.fold_scores(ctx["leaderboard"][2], "mape") is not None

    def test_fold_caption_names_an_outlier_fold(self):
        run = {"metrics": {"cv_folds": [{"fold": i, "rmse": v} for i, v in
                                        enumerate([70.0, 70.1, 69.9, 70.05, 90.0], 1)]}}
        assert "第 5 折明显离群" in rc.fold_scores(run, "rmse")["caption"]

    def test_pred_vs_actual_carries_pairs_residuals_and_stats(self, ctx):
        spec = rc.pred_vs_actual(ctx["leaderboard"][4], "load")
        assert spec["kind"] == "scatter_pair"
        assert len(spec["pair"]["x"]) == len(spec["pair"]["actual"]) == 500
        assert sum(b["count"] for b in spec["residual_bins"]) == 500
        assert spec["stats"]["rmse"] > 0 and spec["stats"]["ratio"] >= 1
        assert set(spec["rows"][0]) == {"category", "x", "actual", "predicted", "residual"}
        assert "500 个点" in spec["caption"]

    def test_pred_vs_actual_needs_two_points(self):
        one = {"metrics": {"val_scatter": {"actual": [1], "predicted": [1]}}}
        two = {"metrics": {"val_scatter": {"actual": [1, 2], "predicted": [1, 2]}}}
        assert rc.pred_vs_actual(one) is None
        assert rc.pred_vs_actual(two) is not None

    def test_loss_history_marks_the_best_epoch_and_shades_the_wait(self):
        history = [{"epoch": i, "train_loss": 30 - i, "val_loss": 30 - i if i <= 20 else 12, "lr": 0.001}
                   for i in range(1, 29)]
        spec = rc.loss_history({"metrics": {"history": history}}, "rmse")
        assert spec["kind"] == "lines" and spec["y_log"] is True
        assert [s["name"] for s in spec["series"]] == ["训练损失", "验证损失"]
        assert spec["markers"] == [{"x": 20, "label": "最优轮 20"}]
        assert spec["shade"] == {"from": 20, "to": 28, "label": "早停等待"}
        assert "第 20 轮最低，之后 8 轮未再刷新" in spec["caption"]
        assert "两条线在这里分开" in spec["caption"]

    def test_a_flat_validation_loss_under_a_falling_training_loss_is_a_plateau(self):
        # Training still easing down, validation within 1% of its minimum:
        # spare capacity, not divergence.
        history = [{"epoch": i, "train_loss": 30.0 - i, "val_loss": 10.0 if i <= 20 else 10.05}
                   for i in range(1, 29)]
        caption = rc.loss_history({"metrics": {"history": history}}, "rmse")["caption"]
        assert "训练损失仍在下降，验证损失持平" in caption
        assert "分开" not in caption

    def test_the_fixture_deep_runs_peak_where_their_early_stop_says(self, ctx):
        # Early stop at 38 with patience 10 means the best epoch is 28; a
        # plateau of equal minima made it 14.
        spec = rc.loss_history(ctx["leaderboard"][4], "rmse")
        assert spec["markers"] == [{"x": 28, "label": "最优轮 28"}]
        assert spec["shade"] == {"from": 28, "to": 38, "label": "早停等待"}

    def test_a_run_still_improving_has_no_shade(self):
        history = [{"epoch": i, "train_loss": 30 - i, "val_loss": 31 - i} for i in range(1, 6)]
        assert rc.loss_history({"metrics": {"history": history}}, "rmse")["shade"] is None

    def test_a_flat_learning_rate_is_not_plotted(self):
        run = {"metrics": {"history": [{"epoch": i, "lr": 0.001} for i in range(1, 6)]}}
        assert rc.lr_history(run) is None

    def test_a_changing_learning_rate_is_plotted_on_a_log_axis(self):
        run = {"metrics": {"history": [{"epoch": 1, "lr": 0.01}, {"epoch": 2, "lr": 0.005}]}}
        spec = rc.lr_history(run)
        assert spec["y_log"] is True and spec["series"][0]["values"] == [0.01, 0.005]
        assert "共变动 1 次" in spec["caption"]
