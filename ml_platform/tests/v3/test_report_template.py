"""The renderer's job is that no model output can touch a computed number."""

import pytest

from app.services import ai_report_narrative as narrative
from app.services import report_facts as rf
from app.services import report_template as rt


class TestRender:
    def test_substitutes_dotted_paths(self):
        out = rt.render("最优 {{best.model}}，RMSE {{best.value}}。",
                        {"best": {"model": "xgboost", "value": "72.4673"}})
        assert out.strip() == "最优 xgboost，RMSE 72.4673。"

    def test_a_missing_fact_leaves_no_dangling_text(self):
        # The failure mode this guards: "误差量级为均值的 " with nothing after.
        out = rt.render("值 {{a.b}}。", {})
        assert out.strip() == "值 。"

    def test_zero_is_a_fact_not_an_absence(self):
        out = rt.render("{{#if n}}失败 {{n}} 个{{/if}}", {"n": 0})
        assert "失败 0 个" in out

    def test_if_block_drops_when_the_fact_is_absent(self):
        assert rt.render("A{{#if x}}\nB\n{{/if}}C", {}).strip() == "AC"

    def test_if_block_keeps_when_present(self):
        assert "B" in rt.render("A{{#if x}}\nB\n{{/if}}", {"x": "y"})

    def test_nested_if_blocks_pair_correctly(self):
        # A non-greedy pattern alone mispairs these, silently dropping the tail.
        tpl = "{{#if a}}外{{#if b}}内{{/if}}尾{{/if}}"
        assert "外内尾" in rt.render(tpl, {"a": 1, "b": 1})
        assert "外尾" in rt.render(tpl, {"a": 1})
        assert rt.render(tpl, {}).strip() == ""


class TestChartSlots:
    def test_keeps_a_chart_the_run_has(self):
        out = rt.render("正文\n\n{{chart:fold_scores}}\n", {}, {"fold_scores"})
        assert "{{chart:fold_scores}}" in out

    def test_drops_the_whole_line_when_there_is_no_data(self):
        # Not a decision the model is asked to make any more, and not an empty
        # frame in the page either.
        out = rt.render("正文\n\n{{chart:prediction_curve}}\n\n后文\n", {}, set())
        assert "chart" not in out
        assert "正文" in out and "后文" in out


class TestWritingSlots:
    def test_lists_instructions_in_document_order(self):
        assert rt.writing_slots("a<<第一>>b<<第二>>") == ["第一", "第二"]

    def test_splices_answers_by_index(self):
        out, n = rt.apply_writing("建议：<<写建议>>结束", {"1": "先做最终评估。"})
        assert out == "建议：先做最终评估。结束"
        assert n == 1

    def test_an_unanswered_slot_leaves_no_trace(self):
        # A report one sentence short beats a report printing "<<写建议>>".
        out, n = rt.apply_writing("建议：<<写建议>>", {})
        assert out == "建议："
        assert n == 0

    def test_a_non_dict_reply_costs_the_sentences_not_the_report(self):
        out, n = rt.apply_writing("正文 <<写一句>>", "模型答非所问")
        assert out.strip() == "正文"
        assert n == 0

    def test_the_model_cannot_reach_anything_but_its_own_slots(self):
        # The whole point: answers are spliced in, never merged with a rewrite,
        # so no reply can change a number, a model name, or a heading.
        doc = "最优 xgboost，RMSE 72.4673。<<写建议>>"
        out, _ = rt.apply_writing(doc, {"1": "RMSE 其实是 99999，模型是 ARIMA。"})
        assert out.startswith("最优 xgboost，RMSE 72.4673。")


class TestReportIntegrity:
    @pytest.mark.parametrize("broken", [
        "五折 RMSE 极差 ，为均值的。",
        "最差折（ ）与最好折（第 2 折）之间。",
        "采用 baseline、。",
        "{{missing.fact}}",
        "<<还没填>>",
    ])
    def test_rejects_structurally_incomplete_reports(self, broken):
        assert rt.integrity_issues(broken)
        with pytest.raises(ValueError, match="结构校验失败"):
            rt.validate_integrity(broken)

    def test_accepts_a_complete_report(self):
        rt.validate_integrity("# 报告\n\n五折 RMSE 极差 2.1，变异系数 1.2%。\n\n{{chart:fold_scores}}")


class TestParseAnswers:
    def test_reads_a_bare_object(self):
        assert rt.parse_answers('{"1": "甲"}') == {"1": "甲"}

    def test_unwraps_a_fenced_reply(self):
        assert rt.parse_answers('```json\n{"1": "甲"}\n```') == {"1": "甲"}

    def test_finds_an_object_buried_in_prose(self):
        assert rt.parse_answers('好的：\n{"2": "乙"}\n以上') == {"2": "乙"}

    def test_returns_empty_for_unparseable_text(self):
        assert rt.parse_answers("我不知道") == {}
        assert rt.parse_answers(None) == {}

    def test_drops_non_string_values(self):
        assert rt.parse_answers('{"1": "甲", "2": 42}') == {"1": "甲"}


class TestTemplatesOnDisk:
    @pytest.mark.parametrize("name", ["overview", "run_ml", "run_dl"])
    def test_each_template_loads_and_asks_the_model_for_something(self, name):
        tpl = rt.load_template(name)
        assert tpl.strip()
        assert rt.writing_slots(tpl), f"{name} 没有留给模型的位置"

    @pytest.mark.parametrize("name", ["overview", "run_ml", "run_dl"])
    def test_no_template_hardcodes_a_model_name_or_a_number(self, name):
        # "写 random_forest、ARIMA" in a prompt put ARIMA into a verdict for a
        # task that never trained one; an example number did the same thing.
        body = rt.load_template(name)
        for invented in ("ARIMA", "random_forest", "xgboost", "lightgbm", "8897"):
            assert invented not in body, invented


class TestFillMessages:
    def test_hands_the_document_over_read_only(self):
        msgs = rt.build_fill_messages("最优 xgboost，RMSE 72.4673。<<写建议>>", ["写建议"])
        user = msgs[1]["content"]
        assert "72.4673" in user
        assert "不可更改" in msgs[0]["content"]

    def test_numbers_the_slots_for_the_reply_to_key_on(self):
        msgs = rt.build_fill_messages("a<<第一>>b<<第二>>", ["第一", "第二"])
        assert "1. 第一" in msgs[1]["content"]
        assert "2. 第二" in msgs[1]["content"]

    def test_asks_for_json_only(self):
        msgs = rt.build_fill_messages("a<<x>>", ["x"])
        assert "只回复一个 JSON 对象" in msgs[1]["content"]

    def test_names_no_example_model_and_no_example_number(self):
        # The two ways a prompt has already leaked fiction into a verdict.
        msgs = rt.build_fill_messages("a<<x>>", ["x"])
        blob = msgs[0]["content"] + msgs[1]["content"].split("===== 报告 =====")[0]
        for invented in ("ARIMA", "random_forest", "8897"):
            assert invented not in blob, invented


class TestOverviewFactsHandleDuplicateRuns:
    """Two runs of the same model read as a two-horse race until someone looks."""

    def _ctx(self):
        entry = lambda rank, model, value: {
            "rank": rank, "model_type": model, "objective_value": value,
            "run_id": f"r{rank}",
            "metrics": {"cv_avg_rmse": value, "cv_std_rmse": 1.0, "cv_avg_r2": 0.99},
        }
        return {
            "task": {"name": "T", "objective_metric": "rmse", "target_column": "y"},
            "dataset": {"row_count": 10, "column_count": 2, "column_names": ["y", "x_lag_1"]},
            "leaderboard": [entry(1, "A", 72.0), entry(2, "A", 72.0),
                            entry(3, "B", 72.3), entry(4, "C", 80.0)],
            "run_status_counts": {"SUCCESS": 4},
            "_target_stats": {"mean": 8896.59, "min": 1.0, "max": 2.0},
            "_readiness": {"score": 60, "checks": []},
        }

    def test_the_duplicate_is_called_out(self):
        facts = rf.build_overview_facts(self._ctx())
        assert "同一模型" in facts["duplicates"]["note"]

    def test_the_runner_up_skips_the_duplicate(self):
        facts = rf.build_overview_facts(self._ctx())
        assert facts["runner_up"]["model"] == "B"

    def test_the_third_model_is_the_third_distinct_one(self):
        # board[2] is still B here; reporting it as third had B "落后" itself.
        facts = rf.build_overview_facts(self._ctx())
        assert "C" in facts["third"]["sentence"]
        assert "B" not in facts["third"]["sentence"]

    def test_no_duplicate_section_when_every_model_is_distinct(self):
        ctx = self._ctx()
        ctx["leaderboard"] = [e for i, e in enumerate(ctx["leaderboard"]) if i != 1]
        assert "duplicates" not in rf.build_overview_facts(ctx)


class TestValidationCohorts:
    def _ctx(self):
        cv = {
            "run_id": "cv", "rank": 1, "model_type": "xgboost", "objective_value": 72.0,
            "metrics": {"selection_cv_mean_rmse": 72.0, "cv_std_rmse": 0.8},
        }
        holdout = {
            "run_id": "holdout", "rank": 2, "model_type": "lstm", "objective_value": 132.0,
            "metrics": {"selection_val_rmse": 132.0, "history": [{"val_loss": 10.0}]},
        }
        return {
            "task": {"name": "T", "objective_metric": "rmse", "target_column": "y"},
            "dataset": {"row_count": 10, "column_count": 2, "column_names": ["y", "x"]},
            "leaderboard": [cv, holdout],
            "run_status_counts": {"SUCCESS": 2},
            "_target_stats": {"mean": 1000.0, "min": 1.0, "max": 2.0},
            "_readiness": {"score": 60, "checks": []},
        }

    def test_mixed_validation_has_no_global_winner_claim(self):
        facts = rf.build_overview_facts(self._ctx())
        assert "不存在可直接认定的全局最优模型" in facts["conclusion"]["sentence"]
        assert "不形成全局排名" in facts["families"]["caveat"]

    def test_table_ranks_within_each_validation_cohort(self):
        table = rf.build_overview_facts(self._ctx())["tables"]["leaderboard"]
        assert "验证口径" in table and "组内排名" in table
        assert "| 交叉验证 | 1 | xgboost" in table
        assert "| 留出验证 | 1 | lstm" in table

    def test_a_holdout_run_is_not_compared_to_a_cv_run(self):
        ctx = self._ctx()
        _, facts = rf.build_run_facts(ctx["leaderboard"][1], ctx, ctx["leaderboard"][0])
        assert "相差 60" not in facts["gap"]["sentence"]
        assert "本模型即本次最优" in facts["gap"]["sentence"]
        assert "两种口径不直接比较" in facts["gap"]["caveat"]

    def test_cv_summary_without_folds_renders_no_blank_fold_sentence(self):
        ctx = self._ctx()
        run = ctx["leaderboard"][0]
        name, facts = rf.build_run_facts(run, ctx, run)
        rendered = rt.render(rt.load_template(name), facts, set())
        assert "未保存逐折明细" in rendered
        assert "极差 ，" not in rendered
        assert "（ ）" not in rendered

    def test_temporal_features_warn_when_old_runs_lack_split_provenance(self):
        ctx = self._ctx()
        ctx["dataset"]["column_names"] = ["load", "load_lag_48", "hour_sin"]
        warning = rf.build_overview_facts(ctx)["validation"]["risk_sentence"]
        assert "不能证明模型对未来时段" in warning
        assert "时间顺序" in warning

    def test_time_aware_runs_name_their_actual_validation_scheme(self):
        entry = self._ctx()["leaderboard"][0]
        entry["metrics"]["validation_strategy"] = "time_series_expanding"
        assert rf.validation_scheme(entry) == "时间序列交叉验证"


class TestChartDefects:
    """Three ways a chart was drawn wrong rather than not drawn at all."""

    def _folds(self):
        # lightgbm's real r2 values: two folds sit exactly on the minimum.
        return [{"fold": i + 1, "r2": r, "rmse": m} for i, (r, m) in enumerate(
            [(0.9974, 72.8147), (0.9974, 71.8756), (0.9972, 73.5246),
             (0.9972, 74.0774), (0.9974, 71.6359)])]

    def _chart(self, cid, run, objective="rmse"):
        built = narrative.build_run_charts(run, [cid], objective)
        return built[0] if built else None

    def test_the_fold_chart_plots_the_objective_metric(self):
        # It took the first numeric key in the fold dict, which is r2 — the
        # prose and the ranking both talk about rmse.
        chart = self._chart("fold_scores", {"metrics": {"cv_folds": self._folds()}})
        assert "RMSE" in chart["title"].upper()
        assert chart["option"]["series"][0]["data"][0] == 72.8147

    def test_the_fold_chart_falls_back_when_the_objective_is_absent(self):
        chart = self._chart("fold_scores", {"metrics": {"cv_folds": self._folds()}}, "mape")
        assert chart is not None

    def test_the_lowest_bar_is_not_zero_pixels_high(self):
        # scale:true puts the axis floor on the data minimum, so the two folds
        # sitting at 0.9972 rendered as nothing at all.
        chart = self._chart("fold_scores", {"metrics": {"cv_folds": self._folds()}}, "r2")
        axis = chart["option"]["yAxis"]
        assert axis.get("scale") is not True
        assert axis["min"] < 0.9972, axis

    def test_the_axis_bounds_are_round_numbers(self):
        # ECharts prints an explicit min and max verbatim, so an unrounded
        # padded bound became the tick label "74.32155".
        axis = self._chart("fold_scores", {"metrics": {"cv_folds": self._folds()}})["option"]["yAxis"]
        for bound in (axis["min"], axis["max"]):
            assert len(str(bound).split(".")[-1]) <= 2, bound

    def test_the_metric_name_is_capitalised_in_the_title(self):
        chart = self._chart("fold_scores", {"metrics": {"cv_folds": self._folds()}})
        assert "RMSE" in chart["title"]
        assert "rmse" not in chart["title"]

    def test_the_mean_line_label_sits_inside_the_plot(self):
        # At the default right-hand end the plot edge clipped it to "均值".
        chart = self._chart("fold_scores", {"metrics": {"cv_folds": self._folds()}})
        mark = chart["option"]["series"][0]["markLine"]
        assert mark["label"]["position"].startswith("inside")

    def test_identical_fold_values_still_get_a_usable_axis(self):
        flat = [{"fold": i + 1, "rmse": 5.0} for i in range(3)]
        axis = self._chart("fold_scores", {"metrics": {"cv_folds": flat}})["option"]["yAxis"]
        assert axis["min"] < 5.0 < axis["max"]

    def test_the_loss_chart_uses_a_log_axis(self):
        # Epoch 1 is orders of magnitude above the rest; on a linear axis the
        # whole curve flattens onto zero and the divergence point vanishes.
        run = {"metrics": {"history": [
            {"epoch": 1, "train_loss": 4.2e7, "val_loss": 4.3e7},
            {"epoch": 2, "train_loss": 300.0, "val_loss": 320.0},
            {"epoch": 3, "train_loss": 140.0, "val_loss": 150.0},
        ]}}
        assert self._chart("loss_history", run)["option"]["yAxis"]["type"] == "log"


class TestDeepLearningTrainingFacts:
    def _hist(self, n=38, best=28):
        # A loss curve, which is what these runs actually record.
        return [{"epoch": i + 1, "val_loss": 20000 - (i * 100 if i <= best else 50),
                 "train_loss": 19000 - i * 120} for i in range(n)]

    def test_the_curve_is_labelled_loss_not_the_objective_metric(self):
        # It was labelled "验证 RMSE 17435.13" next to an RMSE of 132.04 — the
        # square of one presented as the other, a thousandfold apparent error.
        _, facts = rf.build_run_facts(
            {"model_type": "m", "metrics": {"history": self._hist()}}, {"leaderboard": [{}]},
        )
        assert facts["train"]["metric"] == "损失"

    def test_the_objective_metric_is_used_when_the_curve_really_is_that(self):
        hist = [{"epoch": i, "val_rmse": 200 - i} for i in range(1, 6)]
        ctx = {"task": {"objective_metric": "rmse"}, "leaderboard": [{}]}
        _, facts = rf.build_run_facts({"model_type": "m", "metrics": {"history": hist}}, ctx)
        assert facts["train"]["metric"] == "RMSE"

    def test_no_epoch_plan_is_claimed_when_none_was_configured(self):
        # planned was len(history), so it always equalled the actual count and
        # the sentence read "计划训练 38 轮，实际在第 38 轮触发早停".
        _, facts = rf.build_run_facts(
            {"model_type": "m", "metrics": {"history": self._hist()}}, {"leaderboard": [{}]},
        )
        assert facts["train"]["plan_note"] == ""
        # Nor is a completed run claimed: without the configured budget there is
        # no way to tell one from an early stop, and this run's best epoch sits
        # ten short of its last — the signature of the opposite.
        assert facts["train"]["stop_reason"] == "结束"

    def test_early_stopping_is_claimed_only_against_the_configured_budget(self):
        run = {"model_type": "m", "params": {"hyperparameters": {"epochs": 50}},
               "metrics": {"history": self._hist(38)}}
        _, facts = rf.build_run_facts(run, {"leaderboard": [{}]})
        assert facts["train"]["plan_note"] == "计划训练 50 轮，"
        assert facts["train"]["stop_reason"] == "触发早停"
        assert facts["train"]["actual_epochs"] == 38

    def test_a_held_out_score_carries_the_comparability_caveat(self):
        # Ranked against a champion scored by cross-validation, which is a
        # different measurement, not a worse one.
        best = {"model_type": "b", "objective_value": 72.0,
                "metrics": {"cv_avg_rmse": 72.0, "cv_std_rmse": 0.9}}
        run = {"model_type": "m", "objective_value": 132.0, "metrics": {"history": self._hist()}}
        ctx = {"task": {"objective_metric": "rmse"}, "leaderboard": [best]}
        _, facts = rf.build_run_facts(run, ctx, best)
        assert "口径不同" in facts["gap"]["caveat"]


class TestRunFactsDoNotMisstateProvenance:
    """Four things the sub-reports asserted that were not true.

    All four were visible only by reading the rendered page: none raises, and
    each reads as a confident statement of fact.
    """

    def _ctx(self, runs):
        return {
            "task": {"name": "T", "objective_metric": "rmse", "target_column": "y"},
            "dataset": {"row_count": 9, "column_count": 2, "column_names": ["y", "x"]},
            "leaderboard": runs,
            "run_status_counts": {"SUCCESS": len(runs)},
            "_target_stats": {"mean": 8896.59, "min": 1.0, "max": 2.0},
            "_readiness": {"score": 60, "checks": []},
        }

    def _cv_run(self, run_id, rank, value, folds=True):
        metrics = {"cv_avg_rmse": value, "cv_std_rmse": 0.85}
        if folds:
            metrics["cv_folds"] = [{"fold": i, "rmse": value + i * 0.1} for i in range(1, 6)]
        return {"run_id": run_id, "rank": rank, "model_type": "A",
                "objective_value": value, "metrics": metrics}

    def test_a_cross_validated_score_is_not_called_留出验证(self):
        # The winning run has a CV mean but no per-fold detail persisted, and
        # the label was derived from the detail rather than from the score.
        best = self._cv_run("r1", 1, 72.4673, folds=False)
        _, facts = rf.build_run_facts(best, self._ctx([best]))
        assert "交叉验证" in facts["headline"]["sentence"]
        assert "留出验证" not in facts["headline"]["sentence"]

    def test_a_rerun_of_the_winner_is_not_compared_with_itself(self):
        # Produced "与最优模型 A（72.4673）相差 0，相对差 0%".
        best = self._cv_run("r1", 1, 72.4673)
        dup = self._cv_run("r2", 2, 72.4673)
        _, facts = rf.build_run_facts(dup, self._ctx([best, dup]), best)
        assert "重复训练" in facts["gap"]["sentence"]
        assert "相差 0" not in facts["gap"]["sentence"]

    def test_a_model_without_cross_validation_is_not_judged_against_cv_noise(self):
        best = self._cv_run("r1", 1, 72.4673)
        dl = {"run_id": "r9", "rank": 5, "model_type": "D", "objective_value": 132.04,
              "metrics": {"history": [{"val_loss": 5.0}, {"val_loss": 4.0}]}}
        _, facts = rf.build_run_facts(dl, self._ctx([best, dl]), best)
        assert "交叉验证噪声" not in facts["gap"]["sentence"]

    def test_an_unknown_epoch_plan_does_not_become_训练结束(self):
        # best epoch ten short of the last is the signature of an early stop;
        # calling it a completed run states the opposite of what happened.
        history = [{"val_loss": 100 - i} for i in range(28)] + [{"val_loss": 80} for _ in range(10)]
        dl = {"run_id": "r9", "rank": 2, "model_type": "D", "objective_value": 72.0,
              "params": {}, "metrics": {"history": history}}
        best = self._cv_run("r1", 1, 70.0)
        _, facts = rf.build_run_facts(dl, self._ctx([best, dl]), best)
        assert facts["train"]["stop_reason"] != "训练结束"


class TestChartCaptionsDoNotAssert:
    def test_the_fold_caption_states_no_finding(self):
        # "个别折偏低说明数据划分不均" printed under a chart whose own verdict
        # two lines above says no fold is an outlier.
        from app.services import ai_report_narrative as narrative
        run = {"metrics": {"cv_folds": [{"fold": i, "rmse": 1.0 + i} for i in range(1, 4)]}}
        chart = narrative.build_run_charts(run, ["fold_scores"])[0]
        assert "说明" not in chart["description"]

    def test_the_loss_caption_states_no_finding(self):
        from app.services import ai_report_narrative as narrative
        run = {"metrics": {"history": [
            {"epoch": 1, "train_loss": 9.0, "val_loss": 9.5},
            {"epoch": 2, "train_loss": 4.0, "val_loss": 5.0},
        ]}}
        chart = narrative.build_run_charts(run, ["loss_history"])[0]
        assert "就是过拟合的起点" not in chart["description"]


class TestDeepLearningConfigIsRead:
    """The DL trainers nest their config; reading only the flat level missed it."""

    def _run(self, epochs=50, patience=10, ran=38):
        history = [{"epoch": i + 1, "val_loss": max(1.0, 30 - i), "lr": 0.001, "train_loss": 30 - i}
                   for i in range(ran)]
        return {
            "run_id": "r", "rank": 2, "model_type": "lstm", "objective_value": 132.0,
            "params": {"hyperparameters": {
                "train_config": {"epochs": epochs, "batch_size": 32,
                                 "early_stopping_patience": patience, "scheduler": "none"},
                "arch_config": {"num_layers": 2, "hidden_size": 128, "dropout": 0.3},
            }},
            "metrics": {"history": history},
        }

    def _facts(self, run):
        ctx = {"task": {"objective_metric": "rmse"}, "dataset": {},
               "leaderboard": [{"run_id": "b", "model_type": "A", "objective_value": 72.0,
                                "metrics": {"cv_avg_rmse": 72.0, "cv_std_rmse": 0.85}}],
               "_target_stats": {"mean": 8896.59}}
        return rf.build_run_facts(run, ctx, ctx["leaderboard"][0])[1]

    def test_the_epoch_budget_comes_from_train_config(self):
        # 38 of a configured 50 is an early stop; without the nested lookup it
        # was reported as a completed run.
        assert self._facts(self._run())["train"]["plan_note"] == "计划训练 50 轮，"

    def test_early_stopping_names_its_patience(self):
        assert "早停耐心 10 轮" in self._facts(self._run())["train"]["stop_reason"]

    def test_the_architecture_is_stated(self):
        note = self._facts(self._run())["run"]["arch_note"]
        assert "2 层" in note and "128 维" in note and "批量 32" in note

    def test_running_the_full_budget_is_not_an_early_stop(self):
        facts = self._facts(self._run(epochs=38, ran=38))
        assert "早停" not in facts["train"]["stop_reason"]


class TestConstantLearningRateHasNoChart:
    def test_a_flat_rate_is_not_plotted(self):
        # These runs set scheduler "none", so the chart was a horizontal line
        # captioned "调度器折半降速" — a schedule that never ran.
        from app.services import ai_report_narrative as narrative
        run = {"metrics": {"history": [{"epoch": i, "lr": 0.001} for i in range(1, 6)]}}
        assert narrative.build_run_charts(run, ["lr_history"]) == []

    def test_a_changing_rate_is_plotted(self):
        from app.services import ai_report_narrative as narrative
        run = {"metrics": {"history": [{"epoch": 1, "lr": 0.01}, {"epoch": 2, "lr": 0.005}]}}
        assert len(narrative.build_run_charts(run, ["lr_history"])) == 1


class TestStringifiedConfigIsStillRead:
    """Nested config reaches the report as strings, and must still be read.

    _compact_value turns every leaf past depth three into str(), so the run's
    own hyperparameters arrive as {"epochs": "50", "hidden_layers": "[256, 128]"}
    — and an isinstance check declines all of it without a word.
    """

    def _run(self):
        return {
            "run_id": "r", "rank": 2, "model_type": "mlp_dl", "objective_value": 132.0,
            "params": {"hyperparameters": {
                "train_config": {"epochs": "50", "batch_size": "32",
                                 "early_stopping_patience": "10"},
                "arch_config": {"hidden_layers": "[256, 128]", "dropout": "0.3"},
            }},
            "metrics": {"history": [
                {"epoch": i + 1, "val_loss": max(1.0, 30 - i), "train_loss": 30 - i}
                for i in range(38)
            ]},
        }

    def _facts(self):
        ctx = {"task": {"objective_metric": "rmse"}, "dataset": {},
               "leaderboard": [{"run_id": "b", "model_type": "A", "objective_value": 72.0,
                                "metrics": {"cv_avg_rmse": 72.0, "cv_std_rmse": 0.85}}],
               "_target_stats": {"mean": 8896.59}}
        return rf.build_run_facts(self._run(), ctx, ctx["leaderboard"][0])[1]

    def test_a_stringified_epoch_budget_is_read(self):
        assert self._facts()["train"]["plan_note"] == "计划训练 50 轮，"

    def test_a_stringified_patience_is_read(self):
        assert "早停耐心 10 轮" in self._facts()["train"]["stop_reason"]

    def test_layer_widths_are_not_joined_character_by_character(self):
        # '×'.join over the string "[256, 128]" rendered "[×2×5×6×,× ×1×2×8×]".
        note = self._facts()["run"]["arch_note"]
        assert "隐藏层 256×128" in note
        assert "×2×5×6" not in note

    def test_the_batch_size_survives_as_a_number(self):
        assert "批量 32" in self._facts()["run"]["arch_note"]
