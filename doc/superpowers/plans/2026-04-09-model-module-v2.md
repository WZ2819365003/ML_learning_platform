# Model Module v2 — Professional ML System Design

**Branch:** `feat/model-module-v2`  
**Date:** 2026-04-09  
**Status:** Implementation Plan

---

## Overview

Upgrade the model module from a flat list of trainer strings into a professional-grade, self-describing registry. The registry is the **single source of truth** — backend, frontend, and API all derive from it. No hardcoded model lists anywhere in UI code after this change.

---

## Architecture

```
model_registry.py  ←→  schemas.py  →  training.py (GET /models)
       ↓                    ↓
training_service.py    TrainingRequest (class_weight field)
       ↓
trainer.py / regression_trainers.py (configure() reads class_weight)
       ↓
Frontend: ModelSelector + DynamicParamForm ← TrainingConfig.jsx
```

---

## 1. Model Taxonomy (Category Registry)

Five categories, unified in backend data and frontend tabs:

| Category ID | Display Name | Models |
|---|---|---|
| `trees` | 树模型 | random_forest, random_forest_regressor |
| `boosting` | 集成提升 | xgboost, xgboost_regressor, lightgbm, lightgbm_regressor |
| `linear` | 线性模型 | logistic_regression, linear_regression, ridge, lasso, elasticnet |
| `kernel` | 核方法 | svm, svr |
| `neural` | 神经网络 | mlp, mlp_regressor |

---

## 2. ParamSpec Data Structure

```python
# ml_platform/app/core/model_registry.py (new file)

from typing import Any, Literal, TypedDict

TaskType = Literal["classification", "regression"]
ParamType = Literal["int", "float", "str", "bool", "list"]


class ParamSpec(TypedDict, total=False):
    name: str           # required — key passed to trainer.configure()
    display_name: str   # required — UI label
    type: ParamType     # required
    default: Any        # required
    min: float | int    # numeric only
    max: float | int    # numeric only
    step: float | int   # numeric precision for UI control
    options: list[str]  # str enum only
    required: bool      # default False
    advanced: bool      # default False; hidden until "高级参数" toggle
    description: str    # tooltip in UI


class ModelSpec(TypedDict):
    id: str
    display_name: str
    category: str
    task_types: list[TaskType]
    description: str
    class_weight_support: bool
    params: list[ParamSpec]
```

---

## 3. Full Model Registry (all 15 models)

### 3.1 Trees

```python
# random_forest
{
    "id": "random_forest",
    "display_name": "随机森林",
    "category": "trees",
    "task_types": ["classification"],
    "description": "基于多棵决策树的集成方法，通过投票降低过拟合，适合高维数据。",
    "class_weight_support": True,
    "params": [
        {"name": "n_estimators", "display_name": "树的数量", "type": "int",
         "default": 100, "min": 10, "max": 1000, "step": 10, "advanced": False,
         "description": "集成中决策树的数量，越多越稳定但训练更慢。"},
        {"name": "max_depth", "display_name": "最大深度", "type": "int",
         "default": None, "min": 1, "max": 50, "step": 1, "advanced": False,
         "description": "树的最大深度，None 表示不限制。"},
        {"name": "min_samples_split", "display_name": "最小分裂样本数", "type": "int",
         "default": 2, "min": 2, "max": 50, "step": 1, "advanced": True,
         "description": "分裂内部节点所需的最小样本数。"},
        {"name": "min_samples_leaf", "display_name": "叶节点最小样本数", "type": "int",
         "default": 1, "min": 1, "max": 50, "step": 1, "advanced": True,
         "description": "叶节点所需的最小样本数，增大可防止过拟合。"},
        {"name": "class_weight", "display_name": "类别权重", "type": "str",
         "default": None, "options": ["None", "balanced"], "advanced": False,
         "description": "balanced 自动按频率反比调整权重，处理类别不平衡。"},
    ],
},

# random_forest_regressor — 同上去除 class_weight
{
    "id": "random_forest_regressor",
    "display_name": "随机森林回归",
    "category": "trees",
    "task_types": ["regression"],
    "description": "随机森林的回归版本，通过多棵树的平均预测连续值。",
    "class_weight_support": False,
    "params": [
        {"name": "n_estimators", "display_name": "树的数量", "type": "int",
         "default": 100, "min": 10, "max": 1000, "step": 10, "advanced": False,
         "description": "集成中决策树的数量。"},
        {"name": "max_depth", "display_name": "最大深度", "type": "int",
         "default": None, "min": 1, "max": 50, "step": 1, "advanced": False,
         "description": "树的最大深度，None 表示不限制。"},
        {"name": "min_samples_split", "display_name": "最小分裂样本数", "type": "int",
         "default": 2, "min": 2, "max": 50, "step": 1, "advanced": True,
         "description": "分裂内部节点所需的最小样本数。"},
        {"name": "min_samples_leaf", "display_name": "叶节点最小样本数", "type": "int",
         "default": 1, "min": 1, "max": 50, "step": 1, "advanced": True,
         "description": "叶节点所需的最小样本数。"},
    ],
},
```

### 3.2 Boosting

```python
# xgboost (clf)
{
    "id": "xgboost",
    "display_name": "XGBoost",
    "category": "boosting",
    "task_types": ["classification"],
    "description": "梯度提升树高效实现，速度快、精度高，支持正则化，适合表格数据。",
    "class_weight_support": True,
    "params": [
        {"name": "n_estimators", "display_name": "迭代轮数", "type": "int",
         "default": 100, "min": 10, "max": 1000, "step": 10, "advanced": False,
         "description": "提升轮数（树的数量）。"},
        {"name": "learning_rate", "display_name": "学习率", "type": "float",
         "default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01, "advanced": False,
         "description": "每一步收缩权重的步长，防止过拟合。建议 0.01–0.3。"},
        {"name": "max_depth", "display_name": "最大深度", "type": "int",
         "default": 6, "min": 1, "max": 20, "step": 1, "advanced": False,
         "description": "每棵树的最大深度，控制模型复杂度。"},
        {"name": "subsample", "display_name": "样本采样率", "type": "float",
         "default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05, "advanced": True,
         "description": "每棵树训练时随机采样的样本比例。"},
        {"name": "colsample_bytree", "display_name": "特征采样率", "type": "float",
         "default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05, "advanced": True,
         "description": "每棵树随机采样的特征比例。"},
        {"name": "reg_alpha", "display_name": "L1 正则系数", "type": "float",
         "default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1, "advanced": True,
         "description": "L1 正则化权重，促进权重稀疏。"},
        {"name": "reg_lambda", "display_name": "L2 正则系数", "type": "float",
         "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "advanced": True,
         "description": "L2 正则化权重，控制权重大小。"},
        {"name": "class_weight", "display_name": "类别权重", "type": "str",
         "default": None, "options": ["None", "balanced"], "advanced": False,
         "description": "balanced 自动计算 scale_pos_weight（仅二分类）。"},
    ],
},

# xgboost_regressor — 同上去除 class_weight
{
    "id": "xgboost_regressor",
    "display_name": "XGBoost 回归",
    "category": "boosting",
    "task_types": ["regression"],
    "description": "XGBoost 回归版本，适合非线性连续值预测任务。",
    "class_weight_support": False,
    "params": [
        # n_estimators, learning_rate, max_depth, subsample,
        # colsample_bytree, reg_alpha, reg_lambda — same specs
    ],
},

# lightgbm (clf)
{
    "id": "lightgbm",
    "display_name": "LightGBM",
    "category": "boosting",
    "task_types": ["classification"],
    "description": "微软开源的高效梯度提升框架，基于叶子分裂策略，速度极快。",
    "class_weight_support": True,
    "params": [
        {"name": "n_estimators", "display_name": "迭代轮数", "type": "int",
         "default": 100, "min": 10, "max": 1000, "step": 10, "advanced": False,
         "description": "提升轮数。"},
        {"name": "learning_rate", "display_name": "学习率", "type": "float",
         "default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01, "advanced": False,
         "description": "收缩步长，建议配合较多迭代轮数使用较小学习率。"},
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
        {"name": "class_weight", "display_name": "类别权重", "type": "str",
         "default": None, "options": ["None", "balanced"], "advanced": False,
         "description": "balanced 触发 is_unbalance=True，适合正负样本比例悬殊的场景。"},
    ],
},

# lightgbm_regressor — 同上去除 class_weight
```

### 3.3 Linear

```python
# logistic_regression
{
    "id": "logistic_regression",
    "display_name": "逻辑回归",
    "category": "linear",
    "task_types": ["classification"],
    "description": "经典线性分类器，可解释性强，适合线性可分问题和基线模型。",
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
         "description": "仅当 penalty=elasticnet 时有效，0=Ridge，1=Lasso。"},
        {"name": "class_weight", "display_name": "类别权重", "type": "str",
         "default": None, "options": ["None", "balanced"], "advanced": False,
         "description": "balanced 自动按频率反比调整权重。"},
    ],
},

# linear_regression
{
    "id": "linear_regression",
    "display_name": "线性回归",
    "category": "linear",
    "task_types": ["regression"],
    "description": "最基础的线性回归，无正则化，适合线性关系明显的问题。",
    "class_weight_support": False,
    "params": [
        {"name": "fit_intercept", "display_name": "拟合截距", "type": "bool",
         "default": True, "advanced": False,
         "description": "是否计算截距项 b，设为 False 时假设数据已中心化。"},
    ],
},

# ridge
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

# lasso
{
    "id": "lasso",
    "display_name": "套索回归 (Lasso)",
    "category": "linear",
    "task_types": ["regression"],
    "description": "带 L1 正则化的线性回归，能将不重要特征权重压为 0，实现特征选择。",
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

# elasticnet
{
    "id": "elasticnet",
    "display_name": "弹性网络 (ElasticNet)",
    "category": "linear",
    "task_types": ["regression"],
    "description": "结合 L1 和 L2 正则化，兼顾特征选择与系数稳定性。",
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
```

### 3.4 Kernel

```python
# svm
{
    "id": "svm",
    "display_name": "支持向量机 (SVM)",
    "category": "kernel",
    "task_types": ["classification"],
    "description": "通过最大化分类间隔找到最优决策边界，适合中小型高维数据集。",
    "class_weight_support": True,
    "params": [
        {"name": "C", "display_name": "惩罚系数 (C)", "type": "float",
         "default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.1, "advanced": False,
         "description": "误分类惩罚强度，C 越大间隔越小但训练误差越小。"},
        {"name": "kernel", "display_name": "核函数", "type": "str",
         "default": "rbf", "options": ["rbf", "linear", "poly", "sigmoid"],
         "advanced": False,
         "description": "核函数类型。rbf 适合大多数场景，linear 适合线性可分问题。"},
        {"name": "gamma", "display_name": "核系数 (γ)", "type": "str",
         "default": "scale", "options": ["scale", "auto"], "advanced": True,
         "description": "核函数系数，scale=1/(n_features×X.var())，auto=1/n_features。"},
        {"name": "class_weight", "display_name": "类别权重", "type": "str",
         "default": None, "options": ["None", "balanced"], "advanced": False,
         "description": "balanced 自动按频率反比调整权重。"},
    ],
},

# svr
{
    "id": "svr",
    "display_name": "支持向量回归 (SVR)",
    "category": "kernel",
    "task_types": ["regression"],
    "description": "SVM 的回归版本，寻找在 ε 管道内拟合最多样本的超平面。",
    "class_weight_support": False,
    "params": [
        {"name": "C", "display_name": "惩罚系数 (C)", "type": "float",
         "default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.1, "advanced": False,
         "description": "误差惩罚强度，C 越大对误差容忍度越低。"},
        {"name": "kernel", "display_name": "核函数", "type": "str",
         "default": "rbf", "options": ["rbf", "linear", "poly", "sigmoid"],
         "advanced": False,
         "description": "核函数类型。"},
        {"name": "gamma", "display_name": "核系数 (γ)", "type": "str",
         "default": "scale", "options": ["scale", "auto"], "advanced": True,
         "description": "核函数系数。"},
        {"name": "epsilon", "display_name": "ε 管道宽度", "type": "float",
         "default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01, "advanced": True,
         "description": "ε-不敏感管道宽度，管道内误差不计入损失。"},
    ],
},
```

### 3.5 Neural

```python
# mlp (clf)
{
    "id": "mlp",
    "display_name": "多层感知机 (MLP)",
    "category": "neural",
    "task_types": ["classification"],
    "description": "全连接神经网络分类器，通过多层非线性变换学习复杂模式。",
    "class_weight_support": False,
    "params": [
        {"name": "hidden_layer_sizes", "display_name": "隐藏层结构", "type": "list",
         "default": [100, 50], "advanced": False,
         "description": "每个元素代表一个隐藏层的神经元数，如 [128, 64, 32]。"},
        {"name": "activation", "display_name": "激活函数", "type": "str",
         "default": "relu", "options": ["relu", "tanh", "logistic", "identity"],
         "advanced": False,
         "description": "隐藏层激活函数。relu 通常效果最好。"},
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

# mlp_regressor — 同上，task_types=["regression"]
```

---

## 4. New API Response Schema

### 4.1 New Pydantic Models (schemas.py additions)

```python
class ParamSpecSchema(BaseModel):
    name: str
    display_name: str
    type: str
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[str] | None = None
    required: bool = False
    advanced: bool = False
    description: str = ""

class ModelMetadata(BaseModel):
    id: str
    display_name: str
    category: str
    task_types: list[str]
    description: str
    class_weight_support: bool
    params: list[ParamSpecSchema]

class CategoryMetadata(BaseModel):
    id: str
    display_name: str
    icon: str

class ModelsListResponse(BaseModel):
    categories: list[CategoryMetadata]
    models: list[ModelMetadata]
```

### 4.2 Updated Route (training.py)

```python
@router.get("/models", response_model=ModelsListResponse)
async def get_available_models():
    from app.core.model_registry import CATEGORY_REGISTRY, MODEL_REGISTRY
    return {"categories": CATEGORY_REGISTRY, "models": MODEL_REGISTRY}
```

### 4.3 Enhanced TrainingRequest

```python
class TrainingRequest(BaseModel):
    dataset_id: str
    target_column: str
    model_type: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    eval_metrics: list[str] = Field(default_factory=lambda: ["accuracy"])
    cross_validation: CrossValidationConfig | None = None
    # NEW: imbalanced data handling
    class_weight: str | None = Field(default=None,
        description="'balanced' or None. Applied only to models with class_weight_support.")
```

---

## 5. Common Params Spec

```python
COMMON_PARAMS = [
    {"name": "random_state", "display_name": "随机种子", "type": "int",
     "default": 42, "min": 0, "max": 99999, "advanced": False,
     "description": "随机数种子，相同值可复现结果。"},
    {"name": "test_size", "display_name": "测试集比例", "type": "float",
     "default": 0.2, "min": 0.05, "max": 0.5, "step": 0.05, "required": True,
     "advanced": False, "description": "划分为测试集的数据比例。"},
    {"name": "cv_folds", "display_name": "交叉验证折数", "type": "int",
     "default": 5, "min": 2, "max": 20, "advanced": False,
     "description": "K折交叉验证的折数。"},
]

CLASSIFICATION_EVAL_METRICS = [
    {"value": "accuracy",  "label": "准确率 (Accuracy)"},
    {"value": "f1",        "label": "F1 分数"},
    {"value": "precision", "label": "精确率"},
    {"value": "recall",    "label": "召回率"},
    {"value": "roc_auc",   "label": "ROC AUC"},
    {"value": "log_loss",  "label": "对数损失"},
]

REGRESSION_EVAL_METRICS = [
    {"value": "rmse",  "label": "均方根误差 (RMSE)"},
    {"value": "mae",   "label": "平均绝对误差 (MAE)"},
    {"value": "mse",   "label": "均方误差 (MSE)"},
    {"value": "r2",    "label": "R² 决定系数"},
    {"value": "mape",  "label": "平均绝对百分比误差 (MAPE)"},
]
```

---

## 6. Imbalanced Data Handling

`class_weight` is a **top-level field in `TrainingRequest`** (not inside `hyperparameters`).

### Training service injection

```python
# In training_service._run_training_sync():
hp = {**hyperparameters}

if class_weight and class_weight != "None":
    if model_type in ("xgboost", "xgboost_regressor"):
        # Compute scale_pos_weight for binary XGBoost
        if len(np.unique(y_train)) == 2:
            neg = np.sum(y_train == 0)
            pos = np.sum(y_train == 1)
            hp["_computed_scale_pos_weight"] = float(neg / pos) if pos > 0 else 1.0
    else:
        hp["class_weight"] = class_weight

trainer.configure(hp)
```

### Trainer updates (trainer.py configure methods)

```python
# RandomForestTrainer.configure():
cw = hyperparameters.get("class_weight")
params["class_weight"] = None if (cw is None or cw == "None") else cw

# XGBoostTrainer.configure():
spw = hyperparameters.get("_computed_scale_pos_weight")
if spw is not None:
    params["scale_pos_weight"] = spw

# LightGBMTrainer.configure():
cw = hyperparameters.get("class_weight")
if cw == "balanced":
    params["is_unbalance"] = True

# LogisticRegressionTrainer.configure():
cw = hyperparameters.get("class_weight")
params["class_weight"] = None if (cw is None or cw == "None") else cw

# SVMTrainer.configure():
cw = hyperparameters.get("class_weight")
params["class_weight"] = None if (cw is None or cw == "None") else cw
```

---

## 7. Frontend Components

### 7.1 ModelSelector.jsx

```jsx
// ml_platform_web/src/components/training/ModelSelector.jsx
// Props: modelRegistry, taskFilter, value, onChange
// - Category tabs (全部 + one per category)
// - Filtered model dropdown with task badge + description tooltip
// - Selected model description text below
```

### 7.2 DynamicParamForm.jsx

```jsx
// ml_platform_web/src/components/training/DynamicParamForm.jsx
// Props: modelSpec, advancedMode, form
// Renders Form.Items under name={['hyperparameters', paramName]}
//   int/float → InputNumber(min, max, step, precision)
//   bool      → Switch (valuePropName="checked")
//   str enum  → Select(options)
//   list      → Input with comma-split getValueFromEvent
```

### 7.3 TrainingConfig.jsx changes

- Remove hardcoded `CLASSIFICATION_MODELS`, `REGRESSION_MODELS` arrays
- State: `modelRegistry` (from API), `advancedMode` (bool)
- Task type detection: `isRegression = selectedModelSpec?.task_types?.includes('regression') && !selectedModelSpec?.task_types?.includes('classification')`
- Replace model `<Select>` → `<ModelSelector>`
- Replace static param fields → `<DynamicParamForm>` + `<CommonParamsSection>`
- Add `class_weight` field when `selectedModelSpec?.class_weight_support`
- `handleSubmit` sends `class_weight` top-level + `hyperparameters` nested from form

---

## 8. Implementation Plan (8 Tasks)

| # | Task | Files | Notes |
|---|---|---|---|
| 1 | Create `model_registry.py` | `ml_platform/app/core/model_registry.py` (new) | All 15 models, full param specs, category registry, COMMON_PARAMS, metric lists |
| 2 | Add Pydantic schemas | `ml_platform/app/models/schemas.py` | Add `ParamSpecSchema`, `ModelMetadata`, `CategoryMetadata`, `ModelsListResponse`; add `class_weight` to `TrainingRequest` |
| 3 | Update training route | `ml_platform/app/api/routes/training.py` | Replace `GET /models` → returns `ModelsListResponse` |
| 4 | Update training service | `ml_platform/app/services/training_service.py` | Thread `class_weight`, compute `scale_pos_weight` for XGBoost, inject into `hp` before `configure()` |
| 5 | Update trainer `configure()` | `ml_platform/app/core/trainer.py` + `regression_trainers.py` | Read `class_weight` / `_computed_scale_pos_weight` per model; add `reg_alpha`, `reg_lambda` to boosting regressors |
| 6 | Create `ModelSelector.jsx` | `ml_platform_web/src/components/training/ModelSelector.jsx` (new) | Category tabs + filtered dropdown + task badges |
| 7 | Create `DynamicParamForm.jsx` | `ml_platform_web/src/components/training/DynamicParamForm.jsx` (new) | All param types, advancedMode filter, `name={['hyperparameters', p.name]}` |
| 8 | Refactor `TrainingConfig.jsx` | `ml_platform_web/src/pages/TrainingConfig.jsx` | Wire ModelSelector + DynamicParamForm + class_weight; update handleSubmit |

### Dependency Graph

```
Task 1 (model_registry.py)
    ├── Task 2 (schemas.py)
    │       └── Task 3 (training route)
    │                └── Task 4 (training service)
    │                         └── Task 5 (trainer configure)
    └── Task 6 (ModelSelector.jsx)  ← parallel with 2-5
             └── Task 7 (DynamicParamForm.jsx)
                      └── Task 8 (TrainingConfig.jsx)
```
