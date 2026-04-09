"""
Model Registry — single source of truth for all ML model metadata.

Each ModelSpec describes:
  - id: matches TRAINER_REGISTRY / REGRESSION_TRAINER_REGISTRY key
  - taxonomy: category, task_types
  - params: full ParamSpec list (basic + advanced)
  - class_weight_support: whether the model accepts imbalanced-data handling

No imports from other app modules — pure data.
"""
from __future__ import annotations
from typing import Any, Literal

TaskType = Literal["classification", "regression"]
ParamType = Literal["int", "float", "str", "bool", "list"]


# ---------------------------------------------------------------------------
# Category Registry
# ---------------------------------------------------------------------------

CATEGORY_REGISTRY: list[dict] = [
    {"id": "trees",    "display_name": "树模型",   "icon": "apartment"},
    {"id": "boosting", "display_name": "集成提升", "icon": "thunderbolt"},
    {"id": "linear",   "display_name": "线性模型", "icon": "line-chart"},
    {"id": "kernel",   "display_name": "核方法",   "icon": "cluster"},
    {"id": "neural",   "display_name": "神经网络", "icon": "robot"},
]


# ---------------------------------------------------------------------------
# Common params (rendered separately in frontend, not per-model)
# ---------------------------------------------------------------------------

COMMON_PARAMS: list[dict] = [
    {
        "name": "random_state",
        "display_name": "随机种子",
        "type": "int",
        "default": 42,
        "min": 0,
        "max": 99999,
        "step": 1,
        "required": False,
        "advanced": False,
        "description": "随机数种子，相同值可复现结果。",
    },
    {
        "name": "test_size",
        "display_name": "测试集比例",
        "type": "float",
        "default": 0.2,
        "min": 0.05,
        "max": 0.5,
        "step": 0.05,
        "required": True,
        "advanced": False,
        "description": "划分为测试集的数据比例。",
    },
    {
        "name": "cv_folds",
        "display_name": "交叉验证折数",
        "type": "int",
        "default": 5,
        "min": 2,
        "max": 20,
        "step": 1,
        "required": False,
        "advanced": False,
        "description": "K折交叉验证的折数，越多越稳定但更耗时。",
    },
]

# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

CLASSIFICATION_EVAL_METRICS: list[dict] = [
    {"value": "accuracy",  "label": "准确率 (Accuracy)",           "description": "分类正确样本占总样本的比例"},
    {"value": "f1",        "label": "F1 分数",                     "description": "精确率与召回率的调和平均值（加权）"},
    {"value": "precision", "label": "精确率 (Precision)",          "description": "预测为正类中真正正类的比例（加权）"},
    {"value": "recall",    "label": "召回率 (Recall)",             "description": "真正正类中被正确预测的比例（加权）"},
    {"value": "roc_auc",   "label": "ROC AUC",                     "description": "ROC曲线下面积，范围0-1，越高越好"},
    {"value": "log_loss",  "label": "对数损失 (Log Loss)",         "description": "交叉熵损失，越低越好"},
]

REGRESSION_EVAL_METRICS: list[dict] = [
    {"value": "rmse",  "label": "均方根误差 (RMSE)",              "description": "预测误差的标准差，量纲与目标相同"},
    {"value": "mae",   "label": "平均绝对误差 (MAE)",             "description": "预测误差绝对值的平均，对异常值鲁棒"},
    {"value": "mse",   "label": "均方误差 (MSE)",                  "description": "预测误差平方的平均值"},
    {"value": "r2",    "label": "R² 决定系数",                    "description": "模型解释方差的比例，1.0 为完美拟合"},
    {"value": "mape",  "label": "平均绝对百分比误差 (MAPE)",       "description": "误差相对于真实值的百分比"},
]


# ---------------------------------------------------------------------------
# Shared param templates
# ---------------------------------------------------------------------------

def _n_estimators(default: int = 100) -> dict:
    return {
        "name": "n_estimators", "display_name": "迭代轮数/树的数量", "type": "int",
        "default": default, "min": 10, "max": 1000, "step": 10, "advanced": False,
        "description": "集成中树的数量（提升轮数）。越多越稳定，但训练更慢。",
    }


def _learning_rate(default: float = 0.1) -> dict:
    return {
        "name": "learning_rate", "display_name": "学习率", "type": "float",
        "default": default, "min": 0.001, "max": 1.0, "step": 0.01, "advanced": False,
        "description": "每一步收缩权重的步长，防止过拟合。建议 0.01–0.3。",
    }


def _max_depth(default: int | None = None, min_: int = 1, max_: int = 50) -> dict:
    return {
        "name": "max_depth", "display_name": "最大深度", "type": "int",
        "default": default, "min": min_, "max": max_, "step": 1, "advanced": False,
        "description": f"树的最大深度。{'None 表示不限制。' if default is None else ''}",
    }


def _reg_alpha() -> dict:
    return {
        "name": "reg_alpha", "display_name": "L1 正则系数", "type": "float",
        "default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1, "advanced": True,
        "description": "L1 正则化权重，促进权重稀疏。",
    }


def _reg_lambda() -> dict:
    return {
        "name": "reg_lambda", "display_name": "L2 正则系数", "type": "float",
        "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "advanced": True,
        "description": "L2 正则化权重，控制权重大小。",
    }


def _subsample() -> dict:
    return {
        "name": "subsample", "display_name": "样本采样率", "type": "float",
        "default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05, "advanced": True,
        "description": "每棵树训练时随机采样的样本比例，防止过拟合。",
    }


def _colsample_bytree() -> dict:
    return {
        "name": "colsample_bytree", "display_name": "特征采样率", "type": "float",
        "default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05, "advanced": True,
        "description": "每棵树随机采样的特征比例。",
    }


def _class_weight_param() -> dict:
    return {
        "name": "class_weight", "display_name": "类别权重", "type": "str",
        "default": None, "options": ["None", "balanced"], "advanced": False,
        "description": "balanced 自动按频率反比调整权重，处理类别不平衡问题。",
    }


# ---------------------------------------------------------------------------
# Model Registry — all 15 models
# ---------------------------------------------------------------------------

MODEL_REGISTRY: list[dict] = [

    # -----------------------------------------------------------------------
    # TREES
    # -----------------------------------------------------------------------
    {
        "id": "random_forest",
        "display_name": "随机森林",
        "category": "trees",
        "task_types": ["classification"],
        "description": "基于多棵决策树的集成方法，通过投票降低过拟合，适合高维数据，可解释性较好。",
        "class_weight_support": True,
        "params": [
            _n_estimators(100),
            _max_depth(None),
            {"name": "min_samples_split", "display_name": "最小分裂样本数", "type": "int",
             "default": 2, "min": 2, "max": 50, "step": 1, "advanced": True,
             "description": "分裂内部节点所需的最小样本数。"},
            {"name": "min_samples_leaf", "display_name": "叶节点最小样本数", "type": "int",
             "default": 1, "min": 1, "max": 50, "step": 1, "advanced": True,
             "description": "叶节点所需的最小样本数，增大可防止过拟合。"},
            _class_weight_param(),
        ],
    },
    {
        "id": "random_forest_regressor",
        "display_name": "随机森林回归",
        "category": "trees",
        "task_types": ["regression"],
        "description": "随机森林的回归版本，通过多棵树的平均预测连续值，对噪声鲁棒。",
        "class_weight_support": False,
        "params": [
            _n_estimators(100),
            _max_depth(None),
            {"name": "min_samples_split", "display_name": "最小分裂样本数", "type": "int",
             "default": 2, "min": 2, "max": 50, "step": 1, "advanced": True,
             "description": "分裂内部节点所需的最小样本数。"},
            {"name": "min_samples_leaf", "display_name": "叶节点最小样本数", "type": "int",
             "default": 1, "min": 1, "max": 50, "step": 1, "advanced": True,
             "description": "叶节点所需的最小样本数。"},
        ],
    },

    # -----------------------------------------------------------------------
    # BOOSTING
    # -----------------------------------------------------------------------
    {
        "id": "xgboost",
        "display_name": "XGBoost",
        "category": "boosting",
        "task_types": ["classification"],
        "description": "梯度提升树高效实现，速度快、精度高，支持正则化，适合表格数据竞赛场景。",
        "class_weight_support": True,
        "params": [
            _n_estimators(100),
            _learning_rate(0.1),
            _max_depth(6, 1, 20),
            _subsample(),
            _colsample_bytree(),
            _reg_alpha(),
            _reg_lambda(),
            _class_weight_param(),
        ],
    },
    {
        "id": "xgboost_regressor",
        "display_name": "XGBoost 回归",
        "category": "boosting",
        "task_types": ["regression"],
        "description": "XGBoost 回归版本，适合非线性连续值预测任务，精度高、速度快。",
        "class_weight_support": False,
        "params": [
            _n_estimators(100),
            _learning_rate(0.1),
            _max_depth(6, 1, 20),
            _subsample(),
            _colsample_bytree(),
            _reg_alpha(),
            _reg_lambda(),
        ],
    },
    {
        "id": "lightgbm",
        "display_name": "LightGBM",
        "category": "boosting",
        "task_types": ["classification"],
        "description": "微软开源的高效梯度提升框架，基于叶子分裂策略，速度极快，内存友好。",
        "class_weight_support": True,
        "params": [
            _n_estimators(100),
            _learning_rate(0.1),
            {"name": "num_leaves", "display_name": "叶子节点数", "type": "int",
             "default": 31, "min": 4, "max": 256, "step": 1, "advanced": False,
             "description": "每棵树的最大叶子数，LightGBM 核心参数，建议 < 2^max_depth。"},
            {"name": "max_depth", "display_name": "最大深度", "type": "int",
             "default": -1, "min": -1, "max": 20, "step": 1, "advanced": True,
             "description": "-1 表示不限制深度，通过 num_leaves 控制复杂度。"},
            {"name": "min_child_samples", "display_name": "叶节点最小样本数", "type": "int",
             "default": 20, "min": 1, "max": 200, "step": 1, "advanced": True,
             "description": "叶节点所需的最小样本数，防止过拟合。"},
            {"name": "colsample_bytree", "display_name": "特征采样率", "type": "float",
             "default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05, "advanced": True,
             "description": "每次迭代随机选取的特征比例（feature_fraction）。"},
            _class_weight_param(),
        ],
    },
    {
        "id": "lightgbm_regressor",
        "display_name": "LightGBM 回归",
        "category": "boosting",
        "task_types": ["regression"],
        "description": "LightGBM 回归版本，训练速度快，适合大数据量连续值预测。",
        "class_weight_support": False,
        "params": [
            _n_estimators(100),
            _learning_rate(0.1),
            {"name": "num_leaves", "display_name": "叶子节点数", "type": "int",
             "default": 31, "min": 4, "max": 256, "step": 1, "advanced": False,
             "description": "每棵树的最大叶子数。"},
            {"name": "max_depth", "display_name": "最大深度", "type": "int",
             "default": -1, "min": -1, "max": 20, "step": 1, "advanced": True,
             "description": "-1 表示不限制深度。"},
            {"name": "min_child_samples", "display_name": "叶节点最小样本数", "type": "int",
             "default": 20, "min": 1, "max": 200, "step": 1, "advanced": True,
             "description": "叶节点所需的最小样本数。"},
        ],
    },

    # -----------------------------------------------------------------------
    # LINEAR
    # -----------------------------------------------------------------------
    {
        "id": "logistic_regression",
        "display_name": "逻辑回归",
        "category": "linear",
        "task_types": ["classification"],
        "description": "经典线性分类器，可解释性强，系数可直接作为特征重要性，适合基线模型。",
        "class_weight_support": True,
        "params": [
            {"name": "C", "display_name": "正则化强度 (C)", "type": "float",
             "default": 1.0, "min": 0.001, "max": 100.0, "step": 0.1, "advanced": False,
             "description": "正则化强度的倒数，C 越小正则化越强，越防止过拟合。"},
            {"name": "max_iter", "display_name": "最大迭代次数", "type": "int",
             "default": 1000, "min": 100, "max": 5000, "step": 100, "advanced": False,
             "description": "求解器最大迭代次数，不收敛时应增大。"},
            {"name": "solver", "display_name": "求解器", "type": "str",
             "default": "lbfgs", "options": ["lbfgs", "saga", "liblinear", "newton-cg"],
             "advanced": True,
             "description": "优化算法。saga 支持 L1/ElasticNet，lbfgs 适合多分类。"},
            {"name": "l1_ratio", "display_name": "L1 比例 (ElasticNet)", "type": "float",
             "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, "advanced": True,
             "description": "仅 saga+elasticnet 时有效，0=Ridge，1=Lasso。"},
            _class_weight_param(),
        ],
    },
    {
        "id": "linear_regression",
        "display_name": "线性回归",
        "category": "linear",
        "task_types": ["regression"],
        "description": "最基础的线性回归，无正则化，适合线性关系明显的问题，可解释性最强。",
        "class_weight_support": False,
        "params": [
            {"name": "fit_intercept", "display_name": "拟合截距", "type": "bool",
             "default": True, "advanced": False,
             "description": "是否计算截距项 b，设为 False 时假设数据已中心化。"},
        ],
    },
    {
        "id": "ridge",
        "display_name": "岭回归 (Ridge)",
        "category": "linear",
        "task_types": ["regression"],
        "description": "带 L2 正则化的线性回归，防止多重共线性，适合特征间高度相关的场景。",
        "class_weight_support": False,
        "params": [
            {"name": "alpha", "display_name": "正则化系数 (α)", "type": "float",
             "default": 1.0, "min": 0.0001, "max": 100.0, "step": 0.1, "advanced": False,
             "description": "L2 正则化强度，α 越大正则化越强。"},
            {"name": "max_iter", "display_name": "最大迭代次数", "type": "int",
             "default": 2000, "min": 100, "max": 10000, "step": 100, "advanced": True,
             "description": "求解器最大迭代次数。"},
        ],
    },
    {
        "id": "lasso",
        "display_name": "套索回归 (Lasso)",
        "category": "linear",
        "task_types": ["regression"],
        "description": "带 L1 正则化的线性回归，能将不重要特征权重压为 0，实现自动特征选择。",
        "class_weight_support": False,
        "params": [
            {"name": "alpha", "display_name": "正则化系数 (α)", "type": "float",
             "default": 1.0, "min": 0.0001, "max": 100.0, "step": 0.1, "advanced": False,
             "description": "L1 正则化强度，α 越大特征越稀疏。"},
            {"name": "max_iter", "display_name": "最大迭代次数", "type": "int",
             "default": 2000, "min": 100, "max": 10000, "step": 100, "advanced": True,
             "description": "求解器最大迭代次数。"},
        ],
    },
    {
        "id": "elasticnet",
        "display_name": "弹性网络 (ElasticNet)",
        "category": "linear",
        "task_types": ["regression"],
        "description": "结合 L1 和 L2 正则化，兼顾特征选择与系数稳定性，适合高维稀疏场景。",
        "class_weight_support": False,
        "params": [
            {"name": "alpha", "display_name": "正则化系数 (α)", "type": "float",
             "default": 1.0, "min": 0.0001, "max": 100.0, "step": 0.1, "advanced": False,
             "description": "总正则化强度。"},
            {"name": "l1_ratio", "display_name": "L1 比例", "type": "float",
             "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": False,
             "description": "L1 正则在总正则中的比例，0=纯 Ridge，1=纯 Lasso。"},
            {"name": "max_iter", "display_name": "最大迭代次数", "type": "int",
             "default": 2000, "min": 100, "max": 10000, "step": 100, "advanced": True,
             "description": "求解器最大迭代次数。"},
        ],
    },

    # -----------------------------------------------------------------------
    # KERNEL
    # -----------------------------------------------------------------------
    {
        "id": "svm",
        "display_name": "支持向量机 (SVM)",
        "category": "kernel",
        "task_types": ["classification"],
        "description": "通过最大化分类间隔找到最优决策边界，适合中小型高维数据集。",
        "class_weight_support": True,
        "params": [
            {"name": "C", "display_name": "惩罚系数 (C)", "type": "float",
             "default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.5, "advanced": False,
             "description": "误分类惩罚强度，C 越大间隔越小但训练误差越小。"},
            {"name": "kernel", "display_name": "核函数", "type": "str",
             "default": "rbf", "options": ["rbf", "linear", "poly", "sigmoid"],
             "advanced": False,
             "description": "核函数类型。rbf 适合大多数场景，linear 适合线性可分问题。"},
            {"name": "gamma", "display_name": "核系数 (γ)", "type": "str",
             "default": "scale", "options": ["scale", "auto"],
             "advanced": True,
             "description": "核函数系数，scale=1/(n_features×Var)，auto=1/n_features。"},
            _class_weight_param(),
        ],
    },
    {
        "id": "svr",
        "display_name": "支持向量回归 (SVR)",
        "category": "kernel",
        "task_types": ["regression"],
        "description": "SVM 的回归版本，寻找在 ε 管道内拟合最多样本的超平面，对异常值鲁棒。",
        "class_weight_support": False,
        "params": [
            {"name": "C", "display_name": "惩罚系数 (C)", "type": "float",
             "default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.5, "advanced": False,
             "description": "误差惩罚强度，C 越大对误差容忍度越低。"},
            {"name": "kernel", "display_name": "核函数", "type": "str",
             "default": "rbf", "options": ["rbf", "linear", "poly", "sigmoid"],
             "advanced": False,
             "description": "核函数类型。"},
            {"name": "gamma", "display_name": "核系数 (γ)", "type": "str",
             "default": "scale", "options": ["scale", "auto"],
             "advanced": True,
             "description": "核函数系数。"},
            {"name": "epsilon", "display_name": "ε 管道宽度", "type": "float",
             "default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01, "advanced": True,
             "description": "ε-不敏感管道宽度，管道内误差不计入损失。"},
        ],
    },

    # -----------------------------------------------------------------------
    # NEURAL
    # -----------------------------------------------------------------------
    {
        "id": "mlp",
        "display_name": "多层感知机 (MLP)",
        "category": "neural",
        "task_types": ["classification"],
        "description": "全连接神经网络分类器，通过多层非线性变换学习复杂模式，适合非线性问题。",
        "class_weight_support": False,
        "params": [
            {"name": "hidden_layer_sizes", "display_name": "隐藏层结构", "type": "list",
             "default": [100, 50], "advanced": False,
             "description": "每个元素代表一个隐藏层的神经元数，如 [128, 64, 32]。"},
            {"name": "activation", "display_name": "激活函数", "type": "str",
             "default": "relu", "options": ["relu", "tanh", "logistic", "identity"],
             "advanced": False,
             "description": "隐藏层激活函数，relu 通常效果最好。"},
            {"name": "learning_rate_init", "display_name": "初始学习率", "type": "float",
             "default": 0.001, "min": 0.0001, "max": 0.1, "step": 0.0001, "advanced": False,
             "description": "权重更新的初始步长。"},
            {"name": "learning_rate", "display_name": "学习率策略", "type": "str",
             "default": "adaptive", "options": ["constant", "invscaling", "adaptive"],
             "advanced": True,
             "description": "adaptive 在 loss 不下降时自动降低学习率。"},
            {"name": "max_iter", "display_name": "最大迭代次数", "type": "int",
             "default": 500, "min": 50, "max": 2000, "step": 50, "advanced": True,
             "description": "最大训练迭代轮次。"},
            {"name": "alpha", "display_name": "L2 正则系数", "type": "float",
             "default": 0.0001, "min": 0.0, "max": 1.0, "step": 0.0001, "advanced": True,
             "description": "L2 正则化强度，防止权重过大。"},
            {"name": "early_stopping", "display_name": "早停", "type": "bool",
             "default": False, "advanced": True,
             "description": "验证集 loss 不再下降时提前停止训练。"},
        ],
    },
    {
        "id": "mlp_regressor",
        "display_name": "MLP 回归",
        "category": "neural",
        "task_types": ["regression"],
        "description": "全连接神经网络回归器，可拟合复杂非线性连续值关系。",
        "class_weight_support": False,
        "params": [
            {"name": "hidden_layer_sizes", "display_name": "隐藏层结构", "type": "list",
             "default": [100, 50], "advanced": False,
             "description": "每个元素代表一个隐藏层的神经元数。"},
            {"name": "activation", "display_name": "激活函数", "type": "str",
             "default": "relu", "options": ["relu", "tanh", "logistic", "identity"],
             "advanced": False,
             "description": "隐藏层激活函数。"},
            {"name": "learning_rate_init", "display_name": "初始学习率", "type": "float",
             "default": 0.001, "min": 0.0001, "max": 0.1, "step": 0.0001, "advanced": False,
             "description": "权重更新的初始步长。"},
            {"name": "learning_rate", "display_name": "学习率策略", "type": "str",
             "default": "adaptive", "options": ["constant", "invscaling", "adaptive"],
             "advanced": True,
             "description": "adaptive 在 loss 不下降时自动降低学习率。"},
            {"name": "max_iter", "display_name": "最大迭代次数", "type": "int",
             "default": 500, "min": 50, "max": 2000, "step": 50, "advanced": True,
             "description": "最大训练迭代轮次。"},
            {"name": "alpha", "display_name": "L2 正则系数", "type": "float",
             "default": 0.0001, "min": 0.0, "max": 1.0, "step": 0.0001, "advanced": True,
             "description": "L2 正则化强度。"},
            {"name": "early_stopping", "display_name": "早停", "type": "bool",
             "default": False, "advanced": True,
             "description": "验证集 loss 不再下降时提前停止训练。"},
        ],
    },
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_model_spec(model_id: str) -> dict | None:
    """Return the ModelSpec for a given model ID, or None if not found."""
    return next((m for m in MODEL_REGISTRY if m["id"] == model_id), None)


def get_models_by_task(task_type: str) -> list[dict]:
    """Return all models that support the given task type."""
    return [m for m in MODEL_REGISTRY if task_type in m["task_types"]]


def get_model_ids() -> list[str]:
    """Return all registered model IDs."""
    return [m["id"] for m in MODEL_REGISTRY]
