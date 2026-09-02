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
    _build_headline_metrics,
    _build_report_blocks,
    _compact_metrics,
    _highlight_report_lead_sentences,
    _context_for_llm,
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

    def test_briefs_with_a_worked_example_rather_than_a_structure(self):
        # Two revisions of prescribed shape — a 三章 skeleton, then a numbered
        # requirement list — both produced padding: sections written to be
        # filled rather than because there was something to say.
        user = build_ai_report_messages(REGRESSION_CTX)[-1]["content"]
        assert "===== 范本开始 =====" in user
        assert "## 第一章 结论" not in user
        assert "#### 1.1.1" not in user

    def test_the_example_is_marked_as_another_task_and_not_to_be_copied(self):
        # An exemplar full of concrete numbers invites copying them in as facts.
        user = build_ai_report_messages(REGRESSION_CTX)[-1]["content"]
        assert "其他任务" in user
        assert "一个都不要照抄" in user

    def test_asks_for_a_verdict_and_the_dataset_only(self):
        # Per-model detail belongs to the sub-reports; repeated here it was the
        # bulk of the length, and it contradicted them as often as not.
        user = build_ai_report_messages(REGRESSION_CTX)[-1]["content"]
        assert "## 结论" in user
        assert "## 数据集概况" in user
        assert "不要写建议" in user

    def test_still_demands_a_reference_frame_and_the_selection_caveat(self):
        # The guards that are about correctness rather than shape survive the
        # move to an exemplar.
        user = build_ai_report_messages(REGRESSION_CTX)[-1]["content"]
        assert "参照系" in user
        assert "选择分" in user

    def test_no_longer_asks_the_model_to_invent_a_score(self):
        user = build_ai_report_messages(REGRESSION_CTX)[-1]["content"]
        assert "总分：xx/100" not in user


class TestReadinessReadsTheRealContext:
    """The rubric must read the keys build_task_report_context really emits.

    It previously read context["runs"] and context["metrics"], neither of which
    exists. Both checks failed for every task, so a task with 7/7 successful
    runs and 1.2% cross-fold variation scored 0/100 — silently, because a
    missing key is an empty list, not an error.
    """

    def _ctx(self):
        # Shapes copied from build_task_report_context: leaderboard entries
        # carry metrics but no status; statuses live in run_status_counts.
        return {
            "task": {},
            "run_status_counts": {"SUCCESS": 7},
            "leaderboard": [
                {"run_id": "a", "rank": 1,
                 "metrics": {"cv_avg_rmse": 72.4673, "cv_std_rmse": 0.8539}},
            ],
            "successful_run_examples": [{"run_id": "a", "status": "SUCCESS"}],
        }

    def test_counts_successes_from_run_status_counts(self):
        out = compute_readiness_score(self._ctx())
        check = next(c for c in out["checks"] if c["key"] == "run_success")
        assert check["passed"] is True
        assert check["detail"] == "7/7 成功"

    def test_finds_cross_fold_metrics_inside_each_run(self):
        out = compute_readiness_score(self._ctx())
        check = next(c for c in out["checks"] if c["key"] == "cross_fold_stability")
        # 0.8539 / 72.4673 = 1.2%, comfortably inside the 15% bar.
        assert check["passed"] is True

    def test_a_failed_run_fails_the_check(self):
        ctx = self._ctx()
        ctx["run_status_counts"] = {"SUCCESS": 6, "FAILED": 1}
        check = next(c for c in compute_readiness_score(ctx)["checks"]
                     if c["key"] == "run_success")
        assert check["passed"] is False
        assert check["detail"] == "6/7 成功"

    def test_unfinished_final_evaluation_still_costs_its_weight(self):
        # 40 points for the sealed test set are genuinely unearned here; the
        # fix must not hand them out.
        out = compute_readiness_score(self._ctx())
        assert out["score"] == 60
        assert next(c for c in out["checks"]
                    if c["key"] == "final_evaluation")["passed"] is False


class TestHeadlineScore:
    def test_ai_score_card_shows_the_computed_score(self):
        # The card read "—" on every report: it called the legacy regex scrape
        # of "总分：xx/100" that the prompt no longer asks the model to write.
        metrics = _build_headline_metrics(
            {
                "task": {},
                "run_status_counts": {"SUCCESS": 7},
                "leaderboard": [
                    {"run_id": "a", "rank": 1,
                     "metrics": {"cv_avg_rmse": 72.0, "cv_std_rmse": 0.85}},
                ],
                "successful_run_examples": [{"run_id": "a", "status": "SUCCESS"}],
            },
            "# 报告\n正文里没有任何总分字样。",
        )
        card = next(m for m in metrics if m["key"] == "ai_score")
        assert card["value"] == "60/100"
        assert "最终评估" in card["detail"] or "封存" in card["detail"]


class TestCurveMetricsSurviveCompaction:
    """Chart data must survive the trip that shrinks the prompt.

    _compact_metrics keeps at most ten non-scalar values, chosen in alphabetical
    order. val_scatter sorts near the end, so it was dropped from every run and
    the 实际值 vs 预测值 chart had no data to draw — silently, since an absent
    key just means "no chart for this model".
    """

    def _metrics(self, n=500):
        return {
            **{f"filler_{i}": [1, 2] for i in range(12)},   # eat the ten slots
            "val_scatter": {"actual": list(range(n)), "predicted": list(range(n))},
        }

    def test_val_scatter_is_not_dropped(self):
        out = _compact_metrics(self._metrics())
        assert "val_scatter" in out

    def test_parallel_series_keep_enough_points_to_plot(self):
        out = _compact_metrics(self._metrics())
        actual = out["val_scatter"]["actual"]
        # The generic path cut these to twelve, which is not a curve.
        assert len(actual) == 120
        assert len(out["val_scatter"]["predicted"]) == len(actual)

    def test_a_short_series_is_left_alone(self):
        out = _compact_metrics(
            {"val_scatter": {"actual": [1, 2, 3], "predicted": [1, 2, 3]}}
        )
        assert out["val_scatter"]["actual"] == [1, 2, 3]

    def test_the_prompt_still_only_sees_a_summary(self):
        # 120 points per run across eight runs would bloat the prompt for no
        # gain; the model is told how many points there are, not what they are.
        out = _context_for_llm({"val_scatter": {"actual": list(range(500))}})
        assert out["val_scatter"] == {"actual": {"points": 500}}


class TestModelCountIsAFactNotAnEstimate:
    """Counting is handed over, like the reference frames and the readiness score.

    Asked how many models were trained, the model answered "3" for a task with
    seven runs — the number of experiment batches, which is also in the context
    and is not the same thing. A wrong count in the opening sentence discredits
    everything after it.
    """

    def test_the_count_and_the_winner_come_from_the_leaderboard(self):
        ctx = {"leaderboard": [{"run_id": str(i), "model_type": f"m{i}"} for i in range(7)],
               "experiments": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}
        user = build_ai_report_messages(ctx)[-1]["content"]
        assert "本次共训练了 7 个模型" in user
        assert "表现最好的是 m0" in user

    def test_no_example_model_name_appears_anywhere_in_the_prompt(self):
        # "写 random_forest、ARIMA，不要写成随机森林" taught the model that ARIMA
        # was in play; the report then opened with "其中 ARIMA 表现最好" on a task
        # that never trained one. Inside a prompt, an example name and a fact
        # are indistinguishable.
        user = build_ai_report_messages({})[-1]["content"]
        for invented in ("ARIMA", "阿里玛"):
            assert invented not in user, invented

    def test_it_says_which_number_not_to_use(self):
        user = build_ai_report_messages(REGRESSION_CTX)[-1]["content"]
        assert "实验批次数，不是模型数" in user

    def test_an_empty_leaderboard_does_not_crash_the_prompt(self):
        user = build_ai_report_messages({})[-1]["content"]
        assert "本次共训练了 0 个模型" in user
        assert "表现最好的是 暂无" in user

    def test_a_prebuilt_context_string_still_works(self):
        # The archive path passes context already serialised.
        assert build_ai_report_messages("已经序列化好的上下文")[-1]["content"]


class TestLeadSentenceBoldingRespectsDecimals:
    """A decimal point is not the end of a sentence.

    Every metric in these reports is a decimal, and the opening sentence is the
    one place the bolding always applies — so "RMSE 为 72.47" rendered as
    "RMSE 为 72.**47", with the bold closing inside the number.
    """

    def test_a_decimal_does_not_end_the_lead_sentence(self):
        out = _highlight_report_lead_sentences("本次 RMSE 为 72.47，误差很小。后面还有一句。")
        assert "72.47" in out
        assert "72.**47" not in out

    def test_the_full_sentence_is_still_bolded(self):
        out = _highlight_report_lead_sentences("本次 RMSE 为 72.47，误差很小。后面还有一句。")
        assert out.startswith("**本次 RMSE 为 72.47，误差很小。**")

    def test_an_ascii_period_still_ends_a_sentence(self):
        out = _highlight_report_lead_sentences("The RMSE is fine. More text here.")
        assert out.startswith("**The RMSE is fine.**")

    def test_several_decimals_in_one_sentence(self):
        out = _highlight_report_lead_sentences("A 是 0.81%，B 是 1.6%，都可以。第二句。")
        assert "0.81%" in out and "1.6%" in out
        assert "**A 是 0.81%，B 是 1.6%，都可以。**" in out


class TestReportBlocksCarryNoChapterScaffold:
    """The overall report is prose plus artifacts, not a filled-in template.

    Six server-written chapters used to be appended after the model's prose —
    任务范围, 过程与评价, 数据集概况, 参数说明, 训练过程, 效果小结. They were
    written when the report was one long document; now the overall report gives
    the verdict and the dataset and a sub-report per model covers the rest, so
    they said everything a third time. Their headings also leaked into the table
    of contents, which listed "第二章 过程与评价" for a report with no chapters.
    """

    def _blocks(self, markdown="## 结论\n\n可以用。\n\n## 数据集概况\n\n八万行。"):
        return _build_report_blocks(
            markdown,
            [{"id": "training_curves"}],
            [{"id": "data_profile"}],
        )

    def test_the_models_prose_is_kept_whole(self):
        # It used to be sliced on 第一章/第三章 headings the prompt no longer
        # asks for, so the fallback quietly kept everything anyway.
        body = self._blocks()[0]["markdown"]
        assert "## 结论" in body and "## 数据集概况" in body

    def test_no_chapter_boilerplate_is_appended(self):
        joined = " ".join(b.get("markdown", "") for b in self._blocks())
        for phrase in ("第二章", "1.2 任务范围", "过程与评价"):
            assert phrase not in joined, phrase

    def test_real_tables_and_charts_survive(self):
        # They are computed from data and are the only place some facts appear.
        ids = [b["id"] for b in self._blocks()]
        assert "data_profile_block" in ids
        assert "training_curves_block" in ids

    def test_the_title_is_not_doubled(self):
        # The body used to be a slice of the prose, so prepending a title was
        # safe; the whole document goes in now, title included.
        body = _build_report_blocks("# AI 建模报告\n\n## 结论\n\n可以用。", [], [])[0]["markdown"]
        assert body.count("# AI 建模报告") == 1

    def test_a_body_without_a_title_still_gets_one(self):
        body = _build_report_blocks("## 结论\n\n可以用。", [], [])[0]["markdown"]
        assert body.startswith("# AI 建模报告")

    def test_an_artifact_that_was_not_built_is_not_referenced(self):
        ids = [b["id"] for b in _build_report_blocks("正文。", [], [])]
        assert ids == ["conclusion"]
