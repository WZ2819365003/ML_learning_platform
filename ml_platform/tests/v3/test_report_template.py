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
