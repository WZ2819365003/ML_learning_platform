"""Two-tier AI report: one task-level verdict plus one narrative per model.

Replaces a single long document that tried to be both. The previous prompt
mandated a fixed chapter skeleton (第一章/1.1/1.1.1), demanded "每个小节至少
包含一个多自然段说明", and forbade roughly ten things — which together produce
padding in a uniform voice, and, because it also said 不要机械地逐个模型罗列,
suppressed exactly the per-model detail a reader wants.

So it is split:

    总报告 — what the dataset holds and which model to use, in prose, short
    分报告 — one per run: how it trained, how it scored, what to watch

Each document is rendered from computed facts (report_facts) around figure
slots (report_charts) before the model sees it; the model only fills the
<<…>> sentences. Charts are always backend-computed from real data and reach
the page as semantic specs, never as renderer options. Letting the model emit
chart data would hand it a way to draw a plausible loss curve that never
happened — an error that does not raise, does not crash, and cannot be spotted
by reading the report.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Doubao is rate-limited and each call is seconds-to-a-minute; unbounded gather
# over a grid search's worth of runs would stampede it.
_MAX_CONCURRENT_RUN_REPORTS = 4

# Nobody reads twenty narratives, and generating them costs twenty calls. The
# leaderboard is ordered, so the cut keeps the ones worth reading.
_MAX_RUN_REPORTS = 8

# Markers the renderer keeps and the frontend splits on.
_CHART_PLACEHOLDER = re.compile(r"\{\{\s*chart\s*:\s*([a-z0-9_]+)\s*\}\}", re.I)


def placed_chart_ids(markdown: str) -> list[str]:
    """Chart ids in document order, as they appear in the rendered markdown."""
    return [m.group(1).lower() for m in _CHART_PLACEHOLDER.finditer(markdown or "")]


def keep_placed(charts: list[dict[str, Any]], markdown: str) -> list[dict[str, Any]]:
    """Only the specs the document actually places, in the document's order.

    An unplaced chart is a payload shipped to the browser for nothing.
    """
    by_id = {c["id"]: c for c in charts}
    return [by_id[cid] for cid in placed_chart_ids(markdown) if cid in by_id]


def select_runs_for_reports(context: dict[str, Any]) -> list[dict[str, Any]]:
    """The runs worth narrating: best first, capped.

    Reads `leaderboard`, which is what build_task_report_context actually
    produces — it holds only successful runs, already ranked, with the metrics
    and params a sub-report needs. `runs` is accepted as a fallback for callers
    that assemble a context themselves.

    Getting this key wrong is silent: an absent key yields an empty list, so
    every report simply came back with no sub-reports at all and nothing
    anywhere said why.
    """
    entries = context.get("leaderboard") or context.get("runs") or []
    runs = [
        r for r in entries
        # Leaderboard entries carry no status field; they are successful by
        # construction. Only filter when a status is actually present.
        if str(r.get("status", "SUCCESS")).upper() == "SUCCESS"
    ]
    runs.sort(key=lambda r: (r.get("rank") is None, r.get("rank") or 0))
    return runs[:_MAX_RUN_REPORTS]


async def generate_narrative_report(
    context: dict[str, Any],
    *,
    call_model: Callable[[list[dict[str, str]]], Any],
    task_type: str = "regression",
) -> dict[str, Any]:
    """Overall report first, then the per-run ones concurrently.

    Order matters: the overall verdict is what the page shows immediately, so it
    is not made to wait behind a batch of per-model calls.

    Each document is fully rendered from computed facts before the model sees
    it; the call only fills the <<…>> sentences. A failed call therefore costs
    prose, not the report — the numbers and figures are already in place.
    """
    from app.services import report_charts, report_facts, report_template

    async def _write(doc: str, label: str) -> str:
        slots = report_template.writing_slots(doc)
        if not slots:
            return doc
        try:
            raw = await call_model(report_template.build_fill_messages(doc, slots))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Slot fill failed for %s (facts kept): %s", label, exc)
            return report_template.apply_writing(doc, {})[0]
        filled, count = report_template.apply_writing(
            doc, report_template.parse_answers(raw)
        )
        if count < len(slots):
            logger.info("%s: %d/%d slots answered", label, count, len(slots))
        return filled

    # Built before rendering, so the template can drop the slot for a figure
    # that cannot be drawn rather than leaving a marker the page renders as
    # nothing.
    overview_charts = report_charts.build_overview_charts(context)
    overview = await _write(
        report_template.render(
            report_template.load_template("overview"),
            report_facts.build_overview_facts(context),
            {c["id"] for c in overview_charts},
        ),
        "overview",
    )
    report_template.validate_integrity(overview, label="总报告")

    runs = select_runs_for_reports(context)
    best = (context.get("leaderboard") or [{}])[0]
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_RUN_REPORTS)

    async def _one(run: dict[str, Any]) -> dict[str, Any]:
        charts = report_charts.build_run_charts(run, context)
        template_name, facts = report_facts.build_run_facts(run, context, best)
        doc = report_template.render(
            report_template.load_template(template_name),
            facts,
            {c["id"] for c in charts},
        )
        async with semaphore:
            markdown = await _write(doc, str(run.get("model_type")))
        report_template.validate_integrity(
            markdown, label=f"Run {run.get('run_id') or run.get('model_type')} 分报告",
        )
        return {
            "run_id": run.get("run_id"),
            "model_type": run.get("model_type"),
            "strategy_type": run.get("strategy_type"),
            "trial_no": run.get("trial_no"),
            "validation_scheme": report_facts.validation_scheme(run),
            "markdown": markdown,
            "charts": keep_placed(charts, markdown),
        }

    run_reports = await asyncio.gather(*(_one(r) for r in runs))
    return {
        "overview": overview,
        "overview_charts": keep_placed(overview_charts, overview),
        "runs": list(run_reports),
        "runs_total": len(context.get("leaderboard") or context.get("runs") or []),
        "runs_reported": len(run_reports),
    }
