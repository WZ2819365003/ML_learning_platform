"""Report context enrichment: reference frames and the readiness score.

Both used to be the model's job. It was asked to write "总分：xx/100", which a
regex then scraped back out — so the same task could score differently on a
rerun and nothing defined what the number meant — and to relate metrics to the
data itself, which meant doing arithmetic in prose.
"""
import pytest

from app.services.ai_report_service import (
    build_ai_report_messages,
    build_reference_frames,
    compute_readiness_score,
)


REGRESSION_CTX = {
    "task": {"task_type": "regression", "target_column": "load"},
    "dataset": {"columns_info": {"load": {"dtype": "float64", "mean": 8896.6, "min": 5498.4, "max": 14274.2}}},
    "metrics": {"rmse": 72.4673},
}


class TestReferenceFrames:
    def test_expresses_error_as_a_share_of_the_target(self):
        frames = build_reference_frames(REGRESSION_CTX)
        assert frames["rmse"]["pct_of_mean"] == pytest.approx(0.81, abs=0.02)
        assert "均值" in frames["rmse"]["plain"]

    def test_reports_nothing_when_the_target_has_no_stats(self):
        # Better an absent frame than one built on a guessed mean.
        ctx = {**REGRESSION_CTX, "dataset": {"columns_info": {"load": {"dtype": "float64"}}}}
        assert build_reference_frames(ctx) == {}

    def test_does_not_divide_by_a_zero_mean(self):
        ctx = {**REGRESSION_CTX, "dataset": {"columns_info": {"load": {"mean": 0.0}}}}
        assert build_reference_frames(ctx) == {}

    def test_classification_gets_the_majority_class_baseline(self):
        # 92% accuracy means something quite different at a 90% majority.
        ctx = {
            "task": {"task_type": "classification", "target_column": "y"},
            "dataset": {"columns_info": {"y": {"value_counts": {"a": 900, "b": 100}}}},
        }
        frames = build_reference_frames(ctx)
        assert frames["majority_baseline"]["share"] == pytest.approx(0.9)
        assert "多数类" in frames["majority_baseline"]["plain"]


class TestReadinessScore:
    def test_full_marks_when_every_check_passes(self):
        out = compute_readiness_score({
            "task": {"final_evaluation_state": "FINALIZED"},
            "runs": [{"status": "SUCCESS"}, {"status": "SUCCESS"}],
            "metrics": {"cv_avg_rmse": 100, "cv_std_rmse": 5},
        })
        assert out["score"] == 100

    def test_missing_final_evaluation_costs_its_weight(self):
        out = compute_readiness_score({
            "task": {},
            "runs": [{"status": "SUCCESS"}],
            "metrics": {"cv_avg_rmse": 100, "cv_std_rmse": 5},
        })
        assert out["score"] == 60          # 30 stability + 30 success

    def test_absent_cv_data_scores_as_unknown_not_as_a_pass(self):
        # Scoring an unknown as a pass would overstate readiness.
        out = compute_readiness_score({
            "task": {"final_evaluation_state": "FINALIZED"},
            "runs": [{"status": "SUCCESS"}],
            "metrics": {},
        })
        assert out["score"] == 70
        stability = next(c for c in out["checks"] if c["key"] == "cross_fold_stability")
        assert stability["passed"] is False
        assert "无交叉验证" in stability["detail"]

    def test_a_failed_run_fails_the_success_check(self):
        out = compute_readiness_score({
            "task": {}, "runs": [{"status": "SUCCESS"}, {"status": "FAILED"}], "metrics": {},
        })
        run_check = next(c for c in out["checks"] if c["key"] == "run_success")
        assert run_check["passed"] is False
        assert run_check["detail"] == "1/2 成功"

    def test_every_check_carries_its_weight_and_label(self):
        """The rubric is published with the score so it can be argued with."""
        out = compute_readiness_score({"task": {}, "runs": [], "metrics": {}})
        assert sum(c["weight"] for c in out["checks"]) == 100
        assert all(c["label"] for c in out["checks"])


class TestPrompt:
    def test_hands_the_computed_facts_to_the_model(self):
        messages = build_ai_report_messages(REGRESSION_CTX)
        user = messages[-1]["content"]
        assert "reference_frames" in user
        assert "readiness" in user

    def test_asks_for_a_verdict_first_and_drops_the_chapter_skeleton(self):
        user = build_ai_report_messages(REGRESSION_CTX)[-1]["content"]
        assert "结论先行" in user
        # The old prompt pasted a literal skeleton for the model to fill; the
        # words still appear, but only in the instruction not to use them.
        assert "## 第一章 结论" not in user
        assert "#### 1.1.1" not in user
        assert "章节按内容出现" in user

    def test_no_longer_asks_the_model_to_invent_a_score(self):
        user = build_ai_report_messages(REGRESSION_CTX)[-1]["content"]
        assert "总分：xx/100" not in user
