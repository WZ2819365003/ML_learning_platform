"""DL Model Registry — single source of truth for deep-learning model metadata.

Mirrors the ML model_registry.py pattern:
  DL_CATEGORY_REGISTRY     → UI category tabs
  DL_MODEL_REGISTRY        → per-model arch param specs
  DL_OPTIMIZER_PARAMS      → shared optimizer params (all models)
  DL_TRAIN_PARAMS          → shared training-control params (all models)

Each ParamSpec dict has the same fields as the ML registry:
  name / display_name / type / default / min / max / step /
  options / required / advanced / description
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Category Registry
# ---------------------------------------------------------------------------

DL_CATEGORY_REGISTRY: list[dict] = [
    {"id": "feedforward", "display_name": "前馈网络",   "icon": "deployment-unit"},
    {"id": "recurrent",   "display_name": "循环网络",   "icon": "sync"},
    {"id": "conv",        "display_name": "卷积网络",   "icon": "block"},
    {"id": "attention",   "display_name": "注意力机制", "icon": "eye"},
]

# ---------------------------------------------------------------------------
# Shared Optimizer Params (rendered in a dedicated "优化器" section)
# ---------------------------------------------------------------------------

DL_OPTIMIZER_PARAMS: list[dict] = [
    {
        "name": "optimizer", "display_name": "优化器", "type": "str",
        "default": "adam", "options": ["adam", "adamw", "sgd", "rmsprop"],
        "required": True, "advanced": False,
        "description": "选择梯度下降优化算法。Adam 适合大多数任务；SGD + 动量适合 CV。",
    },
    {
        "name": "learning_rate", "display_name": "学习率", "type": "float",
        "default": 0.001, "min": 1e-5, "max": 0.1, "step": 0.0001,
        "required": True, "advanced": False,
        "description": "控制参数更新步长。建议从 1e-3 开始，若不收敛则调小。",
    },
    {
        "name": "weight_decay", "display_name": "权重衰减 (L2)", "type": "float",
        "default": 0.0, "min": 0.0, "max": 0.1, "step": 0.0001,
        "required": False, "advanced": False,
        "description": "L2 正则化强度，防止过拟合。AdamW 使用解耦权重衰减。",
    },
    {
        "name": "beta1", "display_name": "Beta1 (Adam)", "type": "float",
        "default": 0.9, "min": 0.5, "max": 0.999, "step": 0.01,
        "required": False, "advanced": True,
        "description": "Adam/AdamW 一阶矩估计的衰减率。",
    },
    {
        "name": "beta2", "display_name": "Beta2 (Adam)", "type": "float",
        "default": 0.999, "min": 0.9, "max": 0.9999, "step": 0.0001,
        "required": False, "advanced": True,
        "description": "Adam/AdamW 二阶矩估计的衰减率。",
    },
    {
        "name": "momentum", "display_name": "动量 (SGD/RMSProp)", "type": "float",
        "default": 0.9, "min": 0.0, "max": 0.99, "step": 0.01,
        "required": False, "advanced": True,
        "description": "SGD 或 RMSProp 的动量系数，加速收敛。",
    },
]

# ---------------------------------------------------------------------------
# Shared Training Control Params
# ---------------------------------------------------------------------------

DL_TRAIN_PARAMS: list[dict] = [
    {
        "name": "epochs", "display_name": "最大训练轮数", "type": "int",
        "default": 50, "min": 1, "max": 100, "step": 10,
        "required": True, "advanced": False,
        "description": "完整遍历训练集的最大次数。Early Stopping 可提前终止。",
    },
    {
        "name": "batch_size", "display_name": "批大小", "type": "int",
        "default": 32, "min": 8, "max": 1024, "step": 8,
        "required": True, "advanced": False,
        "description": "每次梯度更新使用的样本数。越大越稳定，但内存占用更高。",
    },
    {
        "name": "test_size", "display_name": "验证集比例", "type": "float",
        "default": 0.2, "min": 0.05, "max": 0.5, "step": 0.05,
        "required": True, "advanced": False,
        "description": "划分为验证集的数据比例，用于 Early Stopping 和指标评估。",
    },
    {
        "name": "early_stopping_patience", "display_name": "早停耐心值", "type": "int",
        "default": 10, "min": 1, "max": 100, "step": 1,
        "required": False, "advanced": False,
        "description": "验证集损失连续多少 epoch 未改善时停止训练。",
    },
    {
        "name": "grad_clip", "display_name": "梯度裁剪", "type": "float",
        "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
        "required": False, "advanced": True,
        "description": "梯度范数上限，防止梯度爆炸。设为 0 禁用。",
    },
    {
        "name": "scheduler", "display_name": "学习率调度", "type": "str",
        "default": "none", "options": ["none", "step", "cosine", "plateau"],
        "required": False, "advanced": True,
        "description": "none: 固定 lr。step: 每 epochs/5 衰减。cosine: 余弦退火。plateau: 停滞时衰减。",
    },
]

# ---------------------------------------------------------------------------
# Per-model Architecture Params
# ---------------------------------------------------------------------------

DL_MODEL_REGISTRY: list[dict] = [

    # ── MLP-DL ───────────────────────────────────────────────────────────────
    {
        "id": "mlp_dl",
        "display_name": "MLP（深度）",
        "category": "feedforward",
        "task_types": ["classification", "regression"],
        "description": "多层全连接网络，支持任意层数、Dropout 正则化和批归一化。"
                       "适合中等规模表格数据，训练速度快。",
        "arch_params": [
            {
                "name": "hidden_layers", "display_name": "隐藏层大小", "type": "list",
                "default": [256, 128], "required": True, "advanced": False,
                "description": "各隐藏层神经元数，逗号分隔。例: 256,128,64。",
            },
            {
                "name": "dropout", "display_name": "Dropout 率", "type": "float",
                "default": 0.3, "min": 0.0, "max": 0.9, "step": 0.05,
                "required": False, "advanced": False,
                "description": "每层后随机置零的神经元比例，防止过拟合。",
            },
            {
                "name": "batch_norm", "display_name": "批归一化", "type": "bool",
                "default": True, "required": False, "advanced": False,
                "description": "在激活函数前对每层输出做 Batch Normalization，稳定训练。",
            },
            {
                "name": "activation", "display_name": "激活函数", "type": "str",
                "default": "relu", "options": ["relu", "gelu", "tanh", "leaky_relu"],
                "required": False, "advanced": True,
                "description": "隐藏层激活函数。GELU 在 Transformer 任务中表现更好。",
            },
        ],
    },

    # ── LSTM/GRU ─────────────────────────────────────────────────────────────
    {
        "id": "lstm",
        "display_name": "LSTM / GRU",
        "category": "recurrent",
        "task_types": ["classification", "regression"],
        "description": "长短期记忆网络（LSTM）或门控循环单元（GRU）。"
                       "将特征向量视为单步序列，捕捉特征间的顺序关系。",
        "arch_params": [
            {
                "name": "cell_type", "display_name": "循环单元类型", "type": "str",
                "default": "lstm", "options": ["lstm", "gru"],
                "required": True, "advanced": False,
                "description": "LSTM 含遗忘门，参数更多但表达能力更强；GRU 更轻量。",
            },
            {
                "name": "hidden_size", "display_name": "隐藏状态维度", "type": "int",
                "default": 128, "min": 16, "max": 1024, "step": 16,
                "required": True, "advanced": False,
                "description": "RNN 每层的隐藏状态向量维度。",
            },
            {
                "name": "num_layers", "display_name": "堆叠层数", "type": "int",
                "default": 2, "min": 1, "max": 8, "step": 1,
                "required": False, "advanced": False,
                "description": "RNN 堆叠层数。多层可以学习更抽象的特征。",
            },
            {
                "name": "bidirectional", "display_name": "双向", "type": "bool",
                "default": False, "required": False, "advanced": False,
                "description": "启用双向 RNN，将前向和后向输出拼接，输出维度翻倍。",
            },
            {
                "name": "dropout", "display_name": "Dropout 率", "type": "float",
                "default": 0.3, "min": 0.0, "max": 0.9, "step": 0.05,
                "required": False, "advanced": False,
                "description": "多层 RNN 层间的 Dropout 概率（单层无效）。",
            },
            {
                "name": "fc_hidden", "display_name": "输出头隐藏维度", "type": "int",
                "default": 64, "min": 16, "max": 512, "step": 16,
                "required": False, "advanced": True,
                "description": "RNN 输出后全连接头的中间维度。",
            },
        ],
    },

    # ── CNN-1D ────────────────────────────────────────────────────────────────
    {
        "id": "cnn1d",
        "display_name": "CNN-1D",
        "category": "conv",
        "task_types": ["classification", "regression"],
        "description": "一维卷积网络，将特征向量视为通道数为 1 的序列。"
                       "通过局部感受野捕捉相邻特征间的交互，适合有序特征的数据。",
        "arch_params": [
            {
                "name": "num_filters", "display_name": "卷积核数量", "type": "int",
                "default": 64, "min": 8, "max": 512, "step": 8,
                "required": True, "advanced": False,
                "description": "每层卷积的输出通道数（特征图数量）。",
            },
            {
                "name": "kernel_size", "display_name": "卷积核大小", "type": "int",
                "default": 3, "min": 1, "max": 15, "step": 2,
                "required": True, "advanced": False,
                "description": "一维卷积核的长度。建议使用奇数以保持对称 padding。",
            },
            {
                "name": "num_conv_layers", "display_name": "卷积层数", "type": "int",
                "default": 2, "min": 1, "max": 6, "step": 1,
                "required": False, "advanced": False,
                "description": "卷积块的重复次数，更多层可学习更复杂的特征组合。",
            },
            {
                "name": "dropout", "display_name": "Dropout 率", "type": "float",
                "default": 0.3, "min": 0.0, "max": 0.9, "step": 0.05,
                "required": False, "advanced": False,
                "description": "每个卷积块后的 Dropout 概率。",
            },
            {
                "name": "fc_hidden", "display_name": "全连接头隐藏维度", "type": "int",
                "default": 128, "min": 16, "max": 1024, "step": 16,
                "required": False, "advanced": True,
                "description": "自适应池化后全连接分类/回归头的隐藏维度。",
            },
        ],
    },

    # ── Transformer ───────────────────────────────────────────────────────────
    {
        "id": "transformer",
        "display_name": "Transformer",
        "category": "attention",
        "task_types": ["classification", "regression"],
        "description": "轻量级 Transformer Encoder。将每个特征投影为 token，"
                       "通过多头自注意力捕捉全局特征交互。适合高维稠密特征数据。",
        "arch_params": [
            {
                "name": "d_model", "display_name": "嵌入维度 (d_model)", "type": "int",
                "default": 64, "min": 16, "max": 512, "step": 16,
                "required": True, "advanced": False,
                "description": "每个 token 的嵌入维度，必须能被注意力头数整除。",
            },
            {
                "name": "nhead", "display_name": "注意力头数", "type": "int",
                "default": 4, "min": 1, "max": 16, "step": 1,
                "required": True, "advanced": False,
                "description": "多头注意力的头数。d_model 必须能被 nhead 整除。",
            },
            {
                "name": "num_encoder_layers", "display_name": "Encoder 层数", "type": "int",
                "default": 2, "min": 1, "max": 12, "step": 1,
                "required": False, "advanced": False,
                "description": "堆叠的 TransformerEncoderLayer 数量。",
            },
            {
                "name": "dim_feedforward", "display_name": "FFN 中间维度", "type": "int",
                "default": 256, "min": 64, "max": 2048, "step": 64,
                "required": False, "advanced": False,
                "description": "每个 Encoder 层内前馈网络的中间维度，通常为 d_model 的 4 倍。",
            },
            {
                "name": "dropout", "display_name": "Dropout 率", "type": "float",
                "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05,
                "required": False, "advanced": False,
                "description": "Attention 权重和 FFN 输出的 Dropout 概率。",
            },
        ],
    },
]

# ---------------------------------------------------------------------------
# Trainer Registry (id → class)  — imported lazily to avoid circular imports
# ---------------------------------------------------------------------------

def get_dl_trainer_registry() -> dict:
    from app.core.dl_models.mlp_dl      import MLPDLTrainer
    from app.core.dl_models.lstm        import LSTMTrainer
    from app.core.dl_models.cnn1d       import CNN1DTrainer
    from app.core.dl_models.transformer import TransformerTrainer
    return {
        "mlp_dl":      MLPDLTrainer,
        "lstm":        LSTMTrainer,
        "cnn1d":       CNN1DTrainer,
        "transformer": TransformerTrainer,
    }


def get_dl_trainer(model_type: str):
    registry = get_dl_trainer_registry()
    cls = registry.get(model_type)
    if cls is None:
        raise ValueError(f"Unknown DL model '{model_type}'. Available: {list(registry.keys())}")
    return cls()


# ---------------------------------------------------------------------------
# Metadata helpers (mirror model_registry.py)
# ---------------------------------------------------------------------------

def get_dl_model_spec(model_id: str) -> dict | None:
    """Return the DL ModelSpec for a given model ID, or None if not found."""
    return next((m for m in DL_MODEL_REGISTRY if m["id"] == model_id), None)


def get_dl_models_by_task(task_type: str) -> list[dict]:
    """Return all DL models that support the given task type."""
    return [m for m in DL_MODEL_REGISTRY if task_type in m["task_types"]]


def get_dl_model_ids() -> list[str]:
    """Return all registered DL model IDs."""
    return [m["id"] for m in DL_MODEL_REGISTRY]


def build_default_dl_config(model_id: str) -> dict:
    """
    Build a baseline DL config (arch + opt + train) from registry defaults.
    Used by training_plan_service when the user picks DL models without
    customising hyperparams.
    """
    spec = get_dl_model_spec(model_id)
    if spec is None:
        raise ValueError(f"Unknown DL model '{model_id}'")
    arch = {p["name"]: p["default"] for p in spec.get("arch_params", [])}
    opt = {p["name"]: p["default"] for p in DL_OPTIMIZER_PARAMS}
    train = {p["name"]: p["default"] for p in DL_TRAIN_PARAMS}
    return {"arch": arch, "opt": opt, "train": train}


def clamp_train_config(train_config: dict) -> dict:
    """Clamp numeric train-control params into their registry min/max range.

    ``DL_TRAIN_PARAMS`` carries ``min``/``max`` so the config form can bound its
    inputs, but those bounds are only UI metadata — a request that skips the form
    (direct API call, code-config, a stale frontend) would otherwise submit any
    value at all. Baseline used to be protected by a blunt ``epochs > 10 → 10``
    override in ``tuning_service``; that silently discarded what the user asked
    for. Enforcing the *same* bounds the form advertises replaces that surprise
    with a predictable ceiling.

    Returns a new dict; unknown keys and non-numeric params pass through as-is.
    """
    spec_by_name = {p["name"]: p for p in DL_TRAIN_PARAMS}
    clamped = dict(train_config or {})
    for name, value in list(clamped.items()):
        spec = spec_by_name.get(name)
        if spec is None or spec.get("type") not in ("int", "float"):
            continue
        low, high = spec.get("min"), spec.get("max")
        if low is None and high is None:
            continue
        try:
            numeric = int(value) if spec["type"] == "int" else float(value)
        except (TypeError, ValueError):
            continue  # leave malformed input for the trainer/schema to reject
        if low is not None:
            numeric = max(numeric, low)
        if high is not None:
            numeric = min(numeric, high)
        clamped[name] = numeric
    return clamped
