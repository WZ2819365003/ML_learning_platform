"""TS Model Registry — single source of truth for time-series model metadata.

Mirrors dl_registry.py / model_registry.py shape. 5 models in scope:
  arima              — statsmodels SARIMAX
  ets                — statsmodels Holt-Winters / ETS
  lstm_forecaster    — pytorch direct multi-step
  tcn_forecaster     — pytorch causal conv stack
  timesfm_1          — google foundation model via 3.10 venv subprocess
"""
from __future__ import annotations
from pathlib import Path

TS_CATEGORY_REGISTRY: list[dict] = [
    {"id": "statistical", "display_name": "统计模型", "icon": "line-chart"},
    {"id": "neural",      "display_name": "神经网络", "icon": "deployment-unit"},
    {"id": "foundation",  "display_name": "基础模型", "icon": "cloud"},
]

TS_MODEL_REGISTRY: list[dict] = [
    {
        "id": "arima", "display_name": "ARIMA",
        "category": "statistical", "task_types": ["forecasting"],
        "supports_intervals": True, "supports_exogenous": True,
        "params": [
            {"name": "p", "display_name": "AR 阶数 p", "type": "int",
             "default": 1, "min": 0, "max": 10, "required": True, "advanced": False,
             "description": "自回归阶数"},
            {"name": "d", "display_name": "差分阶数 d", "type": "int",
             "default": 1, "min": 0, "max": 2, "required": True, "advanced": False,
             "description": "差分次数（去趋势）"},
            {"name": "q", "display_name": "MA 阶数 q", "type": "int",
             "default": 1, "min": 0, "max": 10, "required": True, "advanced": False,
             "description": "移动平均阶数"},
            {"name": "seasonal_periods", "display_name": "季节周期", "type": "int",
             "default": 0, "min": 0, "max": 365, "required": False, "advanced": True,
             "description": "0 关闭季节性；7=周；12=月；52=年"},
        ],
    },
    {
        "id": "ets", "display_name": "ETS / Holt-Winters",
        "category": "statistical", "task_types": ["forecasting"],
        "supports_intervals": True, "supports_exogenous": False,
        "params": [
            {"name": "trend", "display_name": "趋势", "type": "str",
             "default": "add", "options": ["add", "mul", "none"],
             "required": True, "advanced": False,
             "description": "additive/multiplicative/none"},
            {"name": "seasonal", "display_name": "季节性", "type": "str",
             "default": "none", "options": ["add", "mul", "none"],
             "required": True, "advanced": False, "description": ""},
            {"name": "seasonal_periods", "display_name": "季节周期", "type": "int",
             "default": 0, "min": 0, "max": 365, "required": False, "advanced": False,
             "description": "0 关闭"},
        ],
    },
    {
        "id": "lstm_forecaster", "display_name": "LSTM Forecaster",
        "category": "neural", "task_types": ["forecasting"],
        "supports_intervals": False, "supports_exogenous": True,
        "params": [
            {"name": "hidden_size", "display_name": "隐藏维度", "type": "int",
             "default": 64, "min": 8, "max": 512, "required": True, "advanced": False,
             "description": ""},
            {"name": "num_layers", "display_name": "层数", "type": "int",
             "default": 2, "min": 1, "max": 6, "required": True, "advanced": False,
             "description": ""},
            {"name": "dropout", "display_name": "Dropout", "type": "float",
             "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05,
             "required": False, "advanced": False, "description": ""},
            {"name": "epochs", "display_name": "训练轮数", "type": "int",
             "default": 50, "min": 1, "max": 500, "required": True, "advanced": False,
             "description": ""},
            {"name": "batch_size", "display_name": "批大小", "type": "int",
             "default": 32, "min": 8, "max": 512, "required": True, "advanced": False,
             "description": ""},
            {"name": "learning_rate", "display_name": "学习率", "type": "float",
             "default": 0.001, "min": 1e-5, "max": 0.1, "step": 0.0001,
             "required": True, "advanced": False, "description": ""},
        ],
    },
    {
        "id": "tcn_forecaster", "display_name": "TCN Forecaster",
        "category": "neural", "task_types": ["forecasting"],
        "supports_intervals": False, "supports_exogenous": True,
        "params": [
            {"name": "channels", "display_name": "通道数", "type": "int",
             "default": 32, "min": 8, "max": 256, "required": True, "advanced": False,
             "description": ""},
            {"name": "kernel_size", "display_name": "卷积核大小", "type": "int",
             "default": 3, "min": 2, "max": 7, "required": True, "advanced": False,
             "description": ""},
            {"name": "num_layers", "display_name": "层数", "type": "int",
             "default": 4, "min": 2, "max": 10, "required": True, "advanced": False,
             "description": ""},
            {"name": "dropout", "display_name": "Dropout", "type": "float",
             "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05,
             "required": False, "advanced": False, "description": ""},
            {"name": "epochs", "display_name": "训练轮数", "type": "int",
             "default": 50, "min": 1, "max": 500, "required": True, "advanced": False,
             "description": ""},
            {"name": "batch_size", "display_name": "批大小", "type": "int",
             "default": 32, "min": 8, "max": 512, "required": True, "advanced": False,
             "description": ""},
            {"name": "learning_rate", "display_name": "学习率", "type": "float",
             "default": 0.001, "min": 1e-5, "max": 0.1, "step": 0.0001,
             "required": True, "advanced": False, "description": ""},
        ],
    },
    {
        "id": "timesfm_1", "display_name": "TimesFM 1.0 (Google)",
        "category": "foundation", "task_types": ["forecasting"],
        "supports_intervals": False, "supports_exogenous": False,
        "params": [
            {"name": "context_len", "display_name": "上下文长度", "type": "int",
             "default": 512, "min": 32, "max": 2048, "required": True, "advanced": False,
             "description": "TimesFM 输入序列长度"},
        ],
        "availability_check": "timesfm_env",
    },
]


def get_ts_model_spec(model_id: str) -> dict | None:
    return next((m for m in TS_MODEL_REGISTRY if m["id"] == model_id), None)


def get_ts_model_ids() -> list[str]:
    return [m["id"] for m in TS_MODEL_REGISTRY]


def list_available_models() -> list[dict]:
    """Return registry entries with `available: bool` flag.

    For models requiring external resources (timesfm_env), checks at call time.
    """
    out = []
    for m in TS_MODEL_REGISTRY:
        avail = True
        check = m.get("availability_check")
        if check == "timesfm_env":
            venv_root = Path(__file__).parent.parent.parent / "timesfm_env"
            venv_python = venv_root / "Scripts" / "python.exe"  # Windows
            if not venv_python.exists():
                venv_python = venv_root / "bin" / "python"       # Linux/macOS
            avail = venv_python.exists()
        out.append({**m, "available": avail})
    return out
