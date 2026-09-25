"""Execute user Python code to produce a V3 experiment-batch config.

Powers the workflow 模型配置 step's 「代码配置」 button (esp. 混合策略). The user's
code must assign a dict named ``config`` describing the experiment batch, e.g.::

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

A3: the code runs in a fresh short-lived subprocess (restricted builtins, no
import, wall-clock timeout with SIGKILL) via app.core.user_code_executor —
a ``while True`` can no longer block the FastAPI event loop.
"""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.core.user_code_executor import run_user_code


async def execute_config_code(code: str, timeout_s: float | None = None) -> dict[str, Any]:
    """Run ``code`` in the sandbox subprocess and return the ``config`` dict.

    Raises ValueError (incl. UserCodeTimeout/UserCodeError subclasses) with an
    actionable message on empty code, execution error, timeout, or a
    missing/mistyped ``config`` variable.
    """
    if not code or not code.strip():
        raise ValueError("代码为空")

    payload = await run_user_code(
        mode="config",
        code=code,
        timeout_s=timeout_s if timeout_s is not None else get_settings().user_code_timeout_s,
    )
    cfg = payload.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("代码需定义一个名为 config 的 dict（至少含 selected_models 字段）")
    if not cfg.get("selected_models"):
        raise ValueError("config 缺少 selected_models（要训练的模型列表）")
    return cfg
