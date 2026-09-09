"""The overview report reads the way doc/report-mock-overview.md reads.

Rendered against the task-0bc692d9 fixture with the model's slots left empty,
so every sentence measured here is one the backend wrote. The seven checks are
the plan's readability gate: short sentences, short paragraphs, no tables, a
figure for every marker, a caption that reads the figure, a verdict that
commits, and specs that carry no renderer configuration.
"""
from __future__ import annotations

import re

import pytest

from app.services import ai_report_narrative, ai_report_service, report_charts, report_facts, report_template
from tests.v3 import report_fixture

_SENTENCE_END = re.compile(r"[。；！？]")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_NUMBER = re.compile(r"\d+(?:\.\d+)?%?")
_FORBIDDEN_VERDICT = ("不存在", "无法认定")


def _render_overview() -> tuple[str, list[dict], dict]:
    context = report_fixture.context()
    charts = report_charts.build_overview_charts(context)
    facts = report_facts.build_overview_facts(context)
    doc = report_template.render(
        report_template.load_template("overview"), facts, {c["id"] for c in charts},
    )
    # The model answered nothing: what remains is entirely computed prose.
    doc, _ = report_template.apply_writing(doc, {})
    return doc, ai_report_narrative.keep_placed(charts, doc), facts


def _prose_paragraphs(markdown: str) -> list[str]:
    """Body paragraphs: no headings, no chart markers, no blank runs."""
    out = []
    for block in re.split(r"\n\s*\n", markdown):
        text = block.strip()
        if not text or text.startswith("#") or text.startswith("{{chart:"):
            continue
        out.append(text)
    return out


def _sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(paragraph) if s.strip()]


def _numbers(sentence: str) -> list[str]:
    # load_lag_1 and xgboost_regressor are names, not numbers.
    return _NUMBER.findall(_IDENTIFIER.sub("", sentence))


@pytest.fixture(scope="module")
def rendered():
    return _render_overview()


def test_every_sentence_carries_at_most_two_numbers(rendered):
    doc, _, _ = rendered
    offenders = [
        (s, _numbers(s))
        for p in _prose_paragraphs(doc) for s in _sentences(p)
        if len(_numbers(s)) > 2
    ]
    assert not offenders, offenders


def test_every_paragraph_has_at_most_three_sentences(rendered):
    doc, _, _ = rendered
    offenders = [p for p in _prose_paragraphs(doc) if len(_sentences(p)) > 3]
    assert not offenders, offenders


def test_the_body_holds_no_markdown_table(rendered):
    doc, _, _ = rendered
    assert not re.search(r"^\|", doc, flags=re.M)


def test_five_chart_markers_each_backed_by_a_spec(rendered):
    doc, charts, _ = rendered
    placed = re.findall(r"\{\{chart:([a-z0-9_]+)\}\}", doc)
    assert placed == ["leaderboard_bars", "fold_dots", "target_hist", "field_composition", "shap_bars"]
    assert [c["id"] for c in charts] == placed
    for chart in charts:
        assert chart["kind"] in {"hbar", "dots", "hist", "stacked", "lines", "scatter_pair"}
        assert chart["tooltip_fields"] and chart["rows"], chart["id"]


def test_every_caption_reads_its_own_data(rendered):
    _, charts, _ = rendered
    for chart in charts:
        caption = chart["caption"]
        names = {str(v) for row in chart["rows"] for v in row.values() if isinstance(v, str)}
        numbers = set()
        for row in chart["rows"]:
            for v in row.values():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    numbers.add(v)
        caption_numbers = {float(n.rstrip("%")) for n in _numbers(caption)}
        mentions_name = any(name and name in caption for name in names)
        mentions_number = any(
            abs(n - m) < 1e-9 or (m and abs(n - m) / abs(m) < 0.01)
            for n in caption_numbers for m in numbers
        )
        assert mentions_name or mentions_number, f"{chart['id']}: {caption}"


def test_the_verdict_commits_to_a_winner(rendered):
    doc, _, facts = rendered
    conclusion = doc.split("## 结论")[1].split("##")[0]
    first = _sentences(_prose_paragraphs(conclusion)[0])[0]
    assert "xgboost_regressor" in first
    assert re.search(r"\d+(?:\.\d+)?%", first), first
    for phrase in _FORBIDDEN_VERDICT:
        assert phrase not in conclusion, phrase
    assert len(_sentences(_prose_paragraphs(conclusion)[0])) <= 3
    # The cover line is a different sentence from the opening of the conclusion.
    assert facts["headline"]["sentence"] != first
    assert "xgboost" in facts["headline"]["sentence"]


def test_specs_carry_no_renderer_configuration(rendered):
    _, charts, _ = rendered
    for chart in charts:
        assert report_charts.renderer_leaks(chart) == [], chart["id"]
    context = report_fixture.context()
    for run in context["leaderboard"]:
        for chart in report_charts.build_run_charts(run, context):
            assert report_charts.renderer_leaks(chart) == [], (run["model_type"], chart["id"])


def test_the_payload_follows_the_contract(rendered):
    doc, charts, _ = rendered
    payload = ai_report_service.build_rich_report_payload(report_fixture.context(), doc)
    assert set(payload) == {"report_schema_version", "headline", "meta", "charts",
                            "appendix_tables", "evidence"}
    assert payload["headline"].startswith("xgboost 胜出")
    assert payload["meta"] == {
        "task_name": "测试1-电力负荷预测", "dataset_name": "电力负荷预测数据.csv",
        "target_column": "load", "task_type": "regression", "objective_metric": "rmse",
        "run_count": 7, "model_count": 5,
    }
    assert [c["id"] for c in payload["charts"]] == [c["id"] for c in charts]
    assert [t["id"] for t in payload["appendix_tables"]] == ["data_profile", "parameter_settings"]
    for gone in ("report_blocks", "tables", "headline_metrics"):
        assert gone not in payload


def test_no_sub_report_recommends_or_prioritises():
    context = report_fixture.context()
    best = context["leaderboard"][0]
    for run in context["leaderboard"]:
        charts = report_charts.build_run_charts(run, context)
        name, facts = report_facts.build_run_facts(run, context, best)
        doc = report_template.render(report_template.load_template(name), facts, {c["id"] for c in charts})
        doc, _ = report_template.apply_writing(doc, {})
        assert not re.search(r"不值得|建议|应当|优先", doc), (run["model_type"], doc)
        assert not re.search(r"^\|", doc, flags=re.M)
        report_template.validate_integrity(doc)
