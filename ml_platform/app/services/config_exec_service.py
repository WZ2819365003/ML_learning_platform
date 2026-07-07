"""Execute user Python code to produce a V3 experiment-batch config.

Powers the workflow 模型配置 step's 「代码配置」 button (esp. 混合策略). The user's
code runs in a restricted namespace and must assign a dict named ``config``
describing the experiment batch, e.g.::

    config = {
        "name": "我的混合实验",
        "strategy_type": "baseline",          # baseline | grid_search | bayesian_search
        "model_family": "mixed",              # ml | dl | mixed
        "selected_models": ["random_forest", "xgboost", "mlp_dl"],
        "search_space": {"random_forest": {"n_estimators": 300}},
        "dl_config": {"mlp_dl": {"arch": {"hidden_layers": [256, 128]}}},
        "eval_metrics": ["accuracy", "f1"],
        "budget_config": {"cv_folds": 5, "test_size": 0.2},
    }

Only a safe subset of builtins is exposed; ``import``, filesystem and process
access are blocked. This is a dev-tool convenience on the user's own platform,
not a hardened sandbox — do not expose it to untrusted callers.
"""

from __future__ import annotations

from typing import Any

# Whitelisted builtins — enough to build configs (comprehensions, ranges,
# arithmetic) without exposing import/open/eval/exec/getattr etc.
_SAFE_BUILTIN_NAMES = [
    "range", "len", "list", "dict", "tuple", "set", "str", "int", "float",
    "bool", "min", "max", "sum", "sorted", "enumerate", "zip", "round", "abs",
    "map", "filter", "any", "all", "print", "True", "False", "None",
]


def _safe_builtins() -> dict[str, Any]:
    import builtins

    return {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES if hasattr(builtins, name)}


def execute_config_code(code: str) -> dict[str, Any]:
    """Run ``code`` in a restricted namespace and return the ``config`` dict.

    Raises ValueError with an actionable message on empty code, execution
    error, or a missing/mistyped ``config`` variable.
    """
    if not code or not code.strip():
        raise ValueError("代码为空")

    sandbox_globals: dict[str, Any] = {"__builtins__": _safe_builtins()}
    sandbox_locals: dict[str, Any] = {}
    try:
        compiled = compile(code, "<config-code>", "exec")
        exec(compiled, sandbox_globals, sandbox_locals)  # noqa: S102 — intentional, restricted builtins
    except Exception as exc:  # surface the real error to the editor
        raise ValueError(f"代码执行失败: {type(exc).__name__}: {exc}") from exc

    cfg = sandbox_locals.get("config", sandbox_globals.get("config"))
    if not isinstance(cfg, dict):
        raise ValueError("代码需定义一个名为 config 的 dict（至少含 selected_models 字段）")
    if not cfg.get("selected_models"):
        raise ValueError("config 缺少 selected_models（要训练的模型列表）")
    return cfg
