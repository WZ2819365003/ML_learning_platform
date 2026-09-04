"""Template-driven report rendering.

The model used to be handed a brief and asked for a whole report. Every number
in it was then something it had produced: it counted models by looking at the
wrong list, divided one metric by another to get a percentage, and copied a
model name out of the prompt's own example into the verdict. None of those
errors raise, and none are visible without checking the report against the
database.

So the shape and the arithmetic move here. A template is markdown holding:

    {{a.b}}            a fact the backend computed — substituted before the
                       model ever sees the document
    {{#if a.b}}…{{/if}} a section that only exists when the fact does
    {{chart:id}}       a figure slot; the line is dropped when the run has no
                       data for that chart, so nothing asks the model whether
                       to illustrate
    <<写什么>>          the one thing left for the model: a sentence of
                       interpretation, written against facts already on the page

The model answers with JSON keyed by slot number, and the backend splices the
sentences in. It never edits the rendered markdown, so no reply can alter a
number, a model name, or a heading.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "report_templates"

_SLOT = re.compile(r"\{\{\s*([a-z0-9_]+(?:\.[a-z0-9_]+)*)\s*\}\}", re.I)
_IF_OPEN = re.compile(r"\{\{#if\s+([a-z0-9_.]+)\s*\}\}", re.I)
_IF_CLOSE = re.compile(r"\{\{/if\}\}", re.I)
_CHART = re.compile(r"^[ \t]*\{\{\s*chart\s*:\s*([a-z0-9_]+)\s*\}\}[ \t]*$\n?", re.I | re.M)
_WRITE = re.compile(r"<<(.+?)>>", re.S)
_DANGLING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("未解析模板标记", re.compile(
        r"\{\{(?!\s*chart\s*:)\s*(?:#if|/if|[a-z0-9_.]+)", re.I,
    )),
    ("未填写写作槽位", re.compile(r"<<.+?>>", re.S)),
    ("空括号", re.compile(r"[（(]\s*[）)]")),
    ("空枚举项", re.compile(r"、\s*[，。；]")),
    ("缺失极差数值", re.compile(r"极差\s*[，。；]")),
    ("缺失变异系数", re.compile(r"变异系数\s*[，。；]")),
)


def load_template(name: str) -> str:
    return (TEMPLATE_DIR / f"{name}.md").read_text(encoding="utf-8")


def _lookup(facts: dict[str, Any], path: str) -> Any:
    node: Any = facts
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _truthy(value: Any) -> bool:
    # A computed 0 or 0.0 is a real fact, not an absent one; only None and the
    # empty string/collection mean "this section has nothing to say".
    if value is None:
        return False
    if isinstance(value, (str, list, tuple, dict)):
        return len(value) > 0
    return True


def render(template: str, facts: dict[str, Any], chart_ids: set[str] | None = None) -> str:
    """Substitute facts, resolve conditionals, and drop unbacked chart slots."""
    charts = chart_ids or set()

    def _resolve_ifs(text: str) -> str:
        """Resolve {{#if}} blocks, matching close tags by depth.

        A non-greedy regex pairs an outer open tag with the *inner* close tag,
        which drops everything between the inner close and the real one — the
        tail of the outer block just disappears, with nothing to show for it.
        """
        out: list[str] = []
        pos = 0
        while True:
            opened = _IF_OPEN.search(text, pos)
            if not opened:
                out.append(text[pos:])
                return "".join(out)
            out.append(text[pos:opened.start()])
            depth = 1
            cursor = opened.end()
            while depth:
                nxt_open = _IF_OPEN.search(text, cursor)
                nxt_close = _IF_CLOSE.search(text, cursor)
                if not nxt_close:
                    # Unbalanced template: keep the body rather than silently
                    # swallowing the rest of the document.
                    logger.warning("Template has an unclosed {{#if}} block")
                    out.append(text[opened.end():])
                    return "".join(out)
                if nxt_open and nxt_open.start() < nxt_close.start():
                    depth += 1
                    cursor = nxt_open.end()
                    continue
                depth -= 1
                cursor = nxt_close.end()
                body_end = nxt_close.start()
            body = text[opened.end():body_end]
            if _truthy(_lookup(facts, opened.group(1))):
                out.append(_resolve_ifs(body))
            pos = cursor

    text = _resolve_ifs(template)
    text = _CHART.sub(
        lambda m: f"{{{{chart:{m.group(1).lower()}}}}}\n"
        if m.group(1).lower() in charts else "",
        text,
    )

    missing: list[str] = []

    def _fill(match: re.Match[str]) -> str:
        value = _lookup(facts, match.group(1))
        if value is None:
            missing.append(match.group(1))
            return ""
        return str(value)

    text = _SLOT.sub(_fill, text)
    if missing:
        # Loud, because a silently empty slot is how a report ends up asserting
        # "误差量级为目标列均值的 " with nothing after it.
        logger.warning("Template slots had no fact: %s", ", ".join(sorted(set(missing))))
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def writing_slots(markdown: str) -> list[str]:
    """The instructions left for the model, in document order."""
    return [m.group(1).strip() for m in _WRITE.finditer(markdown)]


def apply_writing(markdown: str, answers: Any) -> tuple[str, int]:
    """Splice the model's sentences into the slots; drop any it did not answer.

    Returns the filled markdown and how many slots were actually answered. An
    unanswered slot leaves no trace — a report one sentence short beats a report
    with "<<写一句建议>>" printed in it.
    """
    if not isinstance(answers, dict):
        answers = {}
    filled = 0
    index = 0

    def _sub(_match: re.Match[str]) -> str:
        nonlocal filled, index
        index += 1
        value = answers.get(str(index)) or answers.get(index)
        if not isinstance(value, str) or not value.strip():
            return ""
        filled += 1
        return value.strip()

    return _WRITE.sub(_sub, markdown), filled


def integrity_issues(markdown: str) -> list[str]:
    """Find structural defects that must never reach a report archive."""
    text = str(markdown or "")
    return [label for label, pattern in _DANGLING_PATTERNS if pattern.search(text)]


def validate_integrity(markdown: str, *, label: str = "report") -> None:
    issues = integrity_issues(markdown)
    if issues:
        raise ValueError(f"{label} 结构校验失败：{'、'.join(issues)}")


def parse_answers(raw: Any) -> dict[str, str]:
    """Unwrap the model's JSON object from a reply that may be fenced or chatty."""
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items() if isinstance(v, str)}
    text = str(raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    for candidate in (text, (re.search(r"\{.*\}", text, re.S) or _Empty()).group(0)):
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError, AttributeError):
            continue
        if isinstance(parsed, dict):
            return {str(k): v for k, v in parsed.items() if isinstance(v, str)}
    return {}


class _Empty:
    def group(self, _i: int) -> str:
        return ""


# ---------------------------------------------------------------------------
# The one call to the model
# ---------------------------------------------------------------------------

_SYSTEM = (
    "你是机器学习结果解读助手。你会看到一份已经写好的报告，其中的数字、模型名、判定"
    "全部由系统算出，不可更改。你的唯一工作是补写文中标注的空位。"
    "只依据报告里已有的事实写作，绝不引入新的数字、模型名或结论。"
)


def build_fill_messages(markdown: str, slots: list[str]) -> list[dict[str, str]]:
    """Ask for the missing sentences, and nothing else.

    The model is given the finished document as read-only context and answers
    with JSON keyed by slot number. It never returns the document, so no reply
    can alter a number: the worst a bad answer can do is cost one sentence.
    """
    listing = "\n".join(f"{i}. {text}" for i, text in enumerate(slots, 1))
    user = (
        "下面是一份报告，其中 <<…>> 是留给你的空位，尖括号里是这一句要写什么。\n\n"
        "===== 报告 =====\n"
        f"{markdown}\n"
        "===== 报告结束 =====\n\n"
        f"需要补写的空位共 {len(slots)} 处：\n{listing}\n\n"
        "要求：\n"
        "- 只回复一个 JSON 对象，键是空位编号的字符串，值是补写的句子，不要任何其他文字\n"
        "- 每处按括号里的字数要求写，写完即止，不要复述已有的数字\n"
        "- 中文，书面语，与报告其余部分语气一致，直接接在前文后面读得通\n"
        "- 报告里没有的事实一律不写。某一处实在无话可说就给空字符串\n\n"
        '例如：{"1": "……。", "2": "……。"}'
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]
