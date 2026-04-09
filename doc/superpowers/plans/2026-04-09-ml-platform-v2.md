# ML Platform v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 ML 训练平台基础上，增加回归支持、相关性分析 API、回归可视化、模型部署服务（URL in/out）、以及相应的前端页面。

**Architecture:** 后端保持 FastAPI + SQLAlchemy async 架构，新增回归 Trainer、部署服务和推理路由；前端在现有 7 页面基础上新增 Deploy 页面，增强 TrainingConfig 和 Results 页面。

**Tech Stack:** FastAPI, SQLAlchemy async, scikit-learn, pandas, React 18, Ant Design 5, ECharts 5

---

## 文件变更清单

### 新建文件
- `ml_platform/app/services/deploy_service.py` — 部署管理 + ModelCache + 推理逻辑
- `ml_platform/app/api/routes/deploy.py` — 部署 + 推理路由
- `ml_platform_web/src/pages/ModelDeploy.jsx` — 模型部署管理前端页面
- `ml_platform/app/core/regression_trainers.py` — 回归模型训练器

### 修改文件
- `ml_platform/app/core/trainer.py` — 添加 KFold 支持（回归）、回归指标、任务类型检测
- `ml_platform/app/models/database.py` — 新增 ModelDeployment + InferenceJob 表
- `ml_platform/app/models/schemas.py` — 新增回归指标 schema、部署 schema
- `ml_platform/app/services/data_service.py` — 新增相关性分析、目标分布
- `ml_platform/app/services/viz_service.py` — 新增残差图、预测 vs 实际
- `ml_platform/app/services/training_service.py` — 适配任务类型检测
- `ml_platform/app/api/routes/data.py` — 新增 correlation + target_distribution 端点
- `ml_platform/app/api/routes/visualization.py` — 新增残差图端点
- `ml_platform/app/main.py` — 注册 deploy 路由
- `ml_platform_web/src/services/api.js` — 新增 deploy + correlation API
- `ml_platform_web/src/pages/TrainingConfig.jsx` — 新增回归模型选项
- `ml_platform_web/src/pages/Results.jsx` — 新增残差图 + 相关性热力图
- `ml_platform_web/src/pages/ModelManagement.jsx` — 新增"部署模型"按钮
- `ml_platform_web/src/components/layout/Sidebar.jsx` — 新增"模型部署"菜单项
- `ml_platform_web/src/App.jsx` — 注册 /deploy 路由

---

## Task 1: 回归 Trainer + 任务类型检测

**文件:**
- 新建: `ml_platform/app/core/regression_trainers.py`
- 修改: `ml_platform/app/core/trainer.py` (在 TRAINER_REGISTRY 注册回归模型 + 添加 `detect_task_type()`)
- 修改: `ml_platform/app/models/schemas.py` (新增回归指标枚举 + `task_type` 字段)
- 测试: `ml_platform/tests/test_regression_trainers.py`

### 详细步骤

- [ ] **Step 1.1: 新建 `regression_trainers.py`**

创建 `ml_platform/app/core/regression_trainers.py`，内容如下：

```python
"""Regression model trainers."""
from app.core.trainer import BaseTrainer


class RegressionMixin:
    """覆盖 BaseTrainer.train() 使用 KFold（不分层）并计算回归指标。"""

    def train(self, X_train, y_train, X_val, y_val,
              eval_metrics=None, cv_folds=5, callback=None):
        import numpy as np
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        eval_metrics = eval_metrics or ["rmse", "r2"]

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        X_full = np.vstack([X_train, X_val]) if X_val is not None and len(X_val) > 0 else X_train
        y_full = np.concatenate([y_train, y_val]) if y_val is not None and len(y_val) > 0 else y_train

        fold_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_full)):
            X_ft, X_fv = X_full[train_idx], X_full[val_idx]
            y_ft, y_fv = y_full[train_idx], y_full[val_idx]

            self.model.fit(X_ft, y_ft)
            y_pred = self.model.predict(X_fv)

            fold_metrics = self._compute_regression_metrics(y_fv, y_pred, eval_metrics)
            fold_metrics["fold"] = fold_idx + 1
            fold_results.append(fold_metrics)

            if callback:
                callback(fold_idx + 1, cv_folds, fold_metrics)

        # Final train on X_train, eval on X_val
        self.model.fit(X_train, y_train)
        final_metrics = {}
        if X_val is not None and len(X_val) > 0:
            y_val_pred = self.model.predict(X_val)
            final_metrics = self._compute_regression_metrics(y_val, y_val_pred, eval_metrics)

        avg_metrics = {}
        for key in fold_results[0]:
            if key == "fold":
                continue
            values = [fr[key] for fr in fold_results if fr.get(key) is not None]
            if values:
                avg_metrics[f"cv_avg_{key}"] = round(float(np.mean(values)), 4)
                avg_metrics[f"cv_std_{key}"] = round(float(np.std(values)), 4)

        final_metrics.update(avg_metrics)
        final_metrics["cv_folds"] = fold_results
        return final_metrics

    def _compute_regression_metrics(self, y_true, y_pred, metric_names):
        import numpy as np
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        metrics = {}
        for name in metric_names:
            try:
                if name == "mse":
                    metrics[name] = round(float(mean_squared_error(y_true, y_pred)), 4)
                elif name == "rmse":
                    metrics[name] = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)
                elif name == "mae":
                    metrics[name] = round(float(mean_absolute_error(y_true, y_pred)), 4)
                elif name == "r2":
                    metrics[name] = round(float(r2_score(y_true, y_pred)), 4)
                elif name == "mape":
                    mask = y_true != 0
                    if mask.any():
                        metrics[name] = round(float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100, 4)
                    else:
                        metrics[name] = None
            except Exception:
                metrics[name] = None
        return metrics


class RandomForestRegressorTrainer(RegressionMixin, BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "random_forest_regressor"

    def configure(self, hyperparameters: dict):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(
            n_estimators=hyperparameters.get("n_estimators", 100),
            max_depth=hyperparameters.get("max_depth", None),
            random_state=hyperparameters.get("random_state", 42),
            n_jobs=-1,
        )


class XGBoostRegressorTrainer(RegressionMixin, BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "xgboost_regressor"

    def configure(self, hyperparameters: dict):
        from xgboost import XGBRegressor
        self.model = XGBRegressor(
            n_estimators=hyperparameters.get("n_estimators", 100),
            learning_rate=hyperparameters.get("learning_rate", 0.1),
            max_depth=hyperparameters.get("max_depth", 6),
            random_state=hyperparameters.get("random_state", 42),
        )


class LightGBMRegressorTrainer(RegressionMixin, BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "lightgbm_regressor"

    def configure(self, hyperparameters: dict):
        from lightgbm import LGBMRegressor
        self.model = LGBMRegressor(
            n_estimators=hyperparameters.get("n_estimators", 100),
            learning_rate=hyperparameters.get("learning_rate", 0.1),
            num_leaves=hyperparameters.get("num_leaves", 31),
            random_state=hyperparameters.get("random_state", 42),
            verbose=-1,
        )


class LinearRegressionTrainer(RegressionMixin, BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "linear_regression"

    def configure(self, hyperparameters: dict):
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression(
            fit_intercept=hyperparameters.get("fit_intercept", True)
        )


class RidgeTrainer(RegressionMixin, BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "ridge"

    def configure(self, hyperparameters: dict):
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=hyperparameters.get("alpha", 1.0))


class LassoTrainer(RegressionMixin, BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "lasso"

    def configure(self, hyperparameters: dict):
        from sklearn.linear_model import Lasso
        self.model = Lasso(
            alpha=hyperparameters.get("alpha", 1.0),
            max_iter=hyperparameters.get("max_iter", 2000),
        )


class SVRTrainer(RegressionMixin, BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "svr"

    def configure(self, hyperparameters: dict):
        from sklearn.svm import SVR
        self.model = SVR(
            C=hyperparameters.get("C", 1.0),
            kernel=hyperparameters.get("kernel", "rbf"),
            epsilon=hyperparameters.get("epsilon", 0.1),
        )


class MLPRegressorTrainer(RegressionMixin, BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "mlp_regressor"

    def configure(self, hyperparameters: dict):
        from sklearn.neural_network import MLPRegressor
        hidden_layers = hyperparameters.get("hidden_layer_sizes", [100, 50])
        if isinstance(hidden_layers, list):
            hidden_layers = tuple(hidden_layers)
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=hyperparameters.get("activation", "relu"),
            max_iter=hyperparameters.get("max_iter", 500),
            random_state=hyperparameters.get("random_state", 42),
        )


class ElasticNetTrainer(RegressionMixin, BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "elasticnet"

    def configure(self, hyperparameters: dict):
        from sklearn.linear_model import ElasticNet
        self.model = ElasticNet(
            alpha=hyperparameters.get("alpha", 1.0),
            l1_ratio=hyperparameters.get("l1_ratio", 0.5),
            max_iter=hyperparameters.get("max_iter", 2000),
        )


REGRESSION_TRAINER_REGISTRY = {
    "random_forest_regressor": RandomForestRegressorTrainer,
    "xgboost_regressor": XGBoostRegressorTrainer,
    "lightgbm_regressor": LightGBMRegressorTrainer,
    "linear_regression": LinearRegressionTrainer,
    "ridge": RidgeTrainer,
    "lasso": LassoTrainer,
    "elasticnet": ElasticNetTrainer,
    "svr": SVRTrainer,
    "mlp_regressor": MLPRegressorTrainer,
}

REGRESSION_MODEL_TYPES = set(REGRESSION_TRAINER_REGISTRY.keys())
```

- [ ] **Step 1.2: 修改 `trainer.py` — 添加任务类型检测 + 注册回归模型**

在 `ml_platform/app/core/trainer.py` 末尾（`list_available_models()` 之后）添加：

```python
def detect_task_type(y) -> str:
    """Heuristic: classify vs regression based on target column dtype and cardinality."""
    import pandas as pd
    import numpy as np
    s = pd.Series(y) if not isinstance(y, pd.Series) else y
    if s.dtype == "object" or str(s.dtype) == "category":
        return "classification"
    unique_count = s.nunique()
    if s.dtype in [np.dtype("int64"), np.dtype("int32")] and unique_count <= 20:
        return "classification"
    if unique_count / len(s) > 0.05 and s.dtype in [np.dtype("float64"), np.dtype("float32")]:
        return "regression"
    return "classification"
```

并在文件顶部 import 后，更新 `get_trainer()` 函数：

```python
def get_trainer(model_type: str) -> BaseTrainer:
    from app.core.regression_trainers import REGRESSION_TRAINER_REGISTRY
    combined = {**TRAINER_REGISTRY, **REGRESSION_TRAINER_REGISTRY}
    trainer_cls = combined.get(model_type)
    if trainer_cls is None:
        raise ValueError(f"Unknown model type: '{model_type}'. Available: {list(combined.keys())}")
    return trainer_cls()

def list_available_models() -> list[str]:
    from app.core.regression_trainers import REGRESSION_TRAINER_REGISTRY
    return list({**TRAINER_REGISTRY, **REGRESSION_TRAINER_REGISTRY}.keys())
```

- [ ] **Step 1.3: 写测试文件 `tests/test_regression_trainers.py`**

```python
"""Tests for regression trainers."""
import numpy as np
import pytest
from sklearn.datasets import make_regression

from app.core.regression_trainers import (
    LinearRegressionTrainer, RidgeTrainer, RandomForestRegressorTrainer
)
from app.core.trainer import detect_task_type, get_trainer


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    split = 160
    return X[:split], X[split:], y[:split], y[split:]


def test_linear_regression_trainer(reg_data):
    X_train, X_val, y_train, y_val = reg_data
    trainer = LinearRegressionTrainer()
    trainer.configure({})
    results = trainer.train(X_train, y_train, X_val, y_val,
                            eval_metrics=["rmse", "r2"], cv_folds=3)
    assert "rmse" in results
    assert "r2" in results
    assert results["r2"] > 0.5  # should fit reasonably
    assert "cv_avg_rmse" in results


def test_ridge_trainer(reg_data):
    X_train, X_val, y_train, y_val = reg_data
    trainer = RidgeTrainer()
    trainer.configure({"alpha": 0.5})
    results = trainer.train(X_train, y_train, X_val, y_val,
                            eval_metrics=["mae", "r2"], cv_folds=3)
    assert "mae" in results
    assert "r2" in results


def test_rf_regressor(reg_data):
    X_train, X_val, y_train, y_val = reg_data
    trainer = RandomForestRegressorTrainer()
    trainer.configure({"n_estimators": 20})
    results = trainer.train(X_train, y_train, X_val, y_val,
                            eval_metrics=["rmse", "mae", "r2"], cv_folds=3)
    assert results["r2"] is not None


def test_detect_task_type_classification():
    import pandas as pd
    y_cls = pd.Series(["cat", "dog", "cat", "fish"] * 50)
    assert detect_task_type(y_cls) == "classification"

    y_int_few = pd.Series([0, 1, 2, 1, 0] * 40)
    assert detect_task_type(y_int_few) == "classification"


def test_detect_task_type_regression():
    import pandas as pd
    import numpy as np
    y_reg = pd.Series(np.random.uniform(0, 100, 500).astype(float))
    assert detect_task_type(y_reg) == "regression"


def test_get_trainer_regression():
    trainer = get_trainer("xgboost_regressor")
    assert trainer.model_type == "xgboost_regressor"

def test_get_trainer_unknown():
    with pytest.raises(ValueError, match="Unknown model type"):
        get_trainer("nonexistent_model")
```

- [ ] **Step 1.4: 运行测试，确认通过**

```bash
cd ml_platform && python -m pytest tests/test_regression_trainers.py -v
```

期望：6/6 PASS

- [ ] **Step 1.5: Commit**

```bash
git add ml_platform/app/core/regression_trainers.py ml_platform/app/core/trainer.py ml_platform/tests/test_regression_trainers.py
git commit -m "feat: add regression trainers and task type detection"
```

---

## Task 2: 相关性分析 + 目标分布 API

**文件:**
- 修改: `ml_platform/app/services/data_service.py` (新增 2 个函数)
- 修改: `ml_platform/app/api/routes/data.py` (新增 2 个端点)
- 修改: `ml_platform/app/models/schemas.py` (新增响应 schema)
- 测试: `ml_platform/tests/test_data_correlation.py`

### 详细步骤

- [ ] **Step 2.1: 新增 schemas**

在 `ml_platform/app/models/schemas.py` 添加：

```python
class CorrelationResponse(BaseModel):
    """Correlation matrix response."""
    feature_names: list[str]
    matrix: list[list[float | None]]    # N×N correlation matrix
    method: str
    top_pairs: list[dict]               # [{"f1": str, "f2": str, "corr": float}]

class TargetDistributionResponse(BaseModel):
    """Target column distribution."""
    column: str
    task_type: str          # "classification" | "regression"
    labels: list[str]       # For classification: unique values; regression: bin edges
    counts: list[int]
    percentages: list[float]
```

- [ ] **Step 2.2: 新增 data_service 函数**

在 `ml_platform/app/services/data_service.py` 末尾添加：

```python
async def get_correlation_matrix(
    dataset_id: str, method: str, db: AsyncSession
) -> dict:
    """Compute feature correlation matrix."""
    from app.models.database import Dataset
    from sqlalchemy import select
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    ext = Path(dataset.file_path).suffix.lower()
    df = _read_dataframe(Path(dataset.file_path), ext)

    # Only numeric columns
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        raise HTTPException(status_code=400, detail="No numeric columns found")

    if method not in ("pearson", "spearman", "kendall"):
        method = "pearson"

    corr = numeric_df.corr(method=method)
    feature_names = list(corr.columns)
    matrix = []
    for col in feature_names:
        row = []
        for col2 in feature_names:
            val = corr.loc[col, col2]
            row.append(None if pd.isna(val) else round(float(val), 4))
        matrix.append(row)

    # Top 20 correlated pairs (excluding self-correlation)
    pairs = []
    n = len(feature_names)
    for i in range(n):
        for j in range(i + 1, n):
            val = corr.iloc[i, j]
            if not pd.isna(val):
                pairs.append({
                    "f1": feature_names[i],
                    "f2": feature_names[j],
                    "corr": round(float(val), 4)
                })
    pairs.sort(key=lambda x: abs(x["corr"]), reverse=True)

    return {
        "feature_names": feature_names,
        "matrix": matrix,
        "method": method,
        "top_pairs": pairs[:20],
    }


async def get_target_distribution(
    dataset_id: str, target_column: str, db: AsyncSession
) -> dict:
    """Get distribution of target column."""
    from app.models.database import Dataset
    from sqlalchemy import select
    from app.core.trainer import detect_task_type

    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    ext = Path(dataset.file_path).suffix.lower()
    df = _read_dataframe(Path(dataset.file_path), ext)

    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{target_column}' not found")

    y = df[target_column].dropna()
    task_type = detect_task_type(y)

    if task_type == "classification":
        counts = y.value_counts()
        labels = [str(l) for l in counts.index.tolist()]
        count_list = counts.tolist()
        total = sum(count_list)
        percentages = [round(c / total * 100, 2) for c in count_list]
    else:
        # Regression: histogram bins
        import numpy as np
        counts_arr, bin_edges = np.histogram(y, bins=20)
        labels = [f"{round(float(bin_edges[i]), 2)}-{round(float(bin_edges[i+1]), 2)}"
                  for i in range(len(bin_edges) - 1)]
        count_list = counts_arr.tolist()
        total = sum(count_list)
        percentages = [round(c / total * 100, 2) if total > 0 else 0 for c in count_list]

    return {
        "column": target_column,
        "task_type": task_type,
        "labels": labels,
        "counts": count_list,
        "percentages": percentages,
    }
```

- [ ] **Step 2.3: 新增 data.py 端点**

在 `ml_platform/app/api/routes/data.py` 添加 2 个路由（在 `delete_dataset_route` 之后）：

```python
from app.models.schemas import CorrelationResponse, TargetDistributionResponse
from app.services.data_service import get_correlation_matrix, get_target_distribution

@router.get("/{dataset_id}/correlation", response_model=CorrelationResponse)
async def get_correlation_route(
    dataset_id: str,
    method: str = Query(default="pearson", pattern="^(pearson|spearman|kendall)$"),
    db: AsyncSession = Depends(get_db),
) -> CorrelationResponse:
    """Compute feature correlation matrix."""
    result = await get_correlation_matrix(dataset_id=dataset_id, method=method, db=db)
    return CorrelationResponse(**result)


@router.get("/{dataset_id}/target_distribution", response_model=TargetDistributionResponse)
async def get_target_distribution_route(
    dataset_id: str,
    target_column: str = Query(..., description="Target column name"),
    db: AsyncSession = Depends(get_db),
) -> TargetDistributionResponse:
    """Get distribution of target column (classification counts or regression histogram)."""
    result = await get_target_distribution(dataset_id=dataset_id, target_column=target_column, db=db)
    return TargetDistributionResponse(**result)
```

- [ ] **Step 2.4: 写测试**

新建 `ml_platform/tests/test_data_correlation.py`：

```python
"""Tests for correlation and target distribution endpoints."""
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Integration-style test using actual CSV
def make_test_csv(tmp_path, classification=True):
    np.random.seed(42)
    n = 200
    if classification:
        df = pd.DataFrame({
            "feat_a": np.random.randn(n),
            "feat_b": np.random.randn(n),
            "feat_c": np.random.randn(n) * 2,
            "target": np.random.choice(["A", "B", "C"], n),
        })
    else:
        df = pd.DataFrame({
            "feat_a": np.random.randn(n),
            "feat_b": np.random.randn(n),
            "target": np.random.uniform(0, 100, n),
        })
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return path


def test_detect_task_type_in_target_distribution(tmp_path):
    from app.core.trainer import detect_task_type
    import pandas as pd, numpy as np
    y_cls = pd.Series(["A", "B", "A", "C"] * 50)
    assert detect_task_type(y_cls) == "classification"
    y_reg = pd.Series(np.random.uniform(0, 100, 500).astype(float))
    assert detect_task_type(y_reg) == "regression"


def test_correlation_response_shape():
    """Test that correlation matrix is square and values are in [-1, 1]."""
    import pandas as pd, numpy as np
    df = pd.DataFrame(np.random.randn(100, 4), columns=["a", "b", "c", "d"])
    corr = df.corr(method="pearson")
    n = len(corr.columns)
    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            if not pd.isna(val):
                assert -1.0 <= val <= 1.0 + 1e-9
```

- [ ] **Step 2.5: 运行测试**

```bash
cd ml_platform && python -m pytest tests/test_data_correlation.py -v
```

期望：PASS

- [ ] **Step 2.6: Commit**

```bash
git add ml_platform/app/services/data_service.py ml_platform/app/api/routes/data.py ml_platform/app/models/schemas.py ml_platform/tests/test_data_correlation.py
git commit -m "feat: add correlation matrix and target distribution APIs"
```

---

## Task 3: 回归可视化 API（残差图 + 预测 vs 实际）

**文件:**
- 修改: `ml_platform/app/services/viz_service.py` (新增 2 个函数)
- 修改: `ml_platform/app/api/routes/visualization.py` (新增 2 个端点)
- 修改: `ml_platform/app/models/schemas.py` (新增响应 schema)
- 测试: `ml_platform/tests/test_viz_regression.py`

### 详细步骤

- [ ] **Step 3.1: 新增 schemas**

在 `schemas.py` 添加：

```python
class ResidualPlotResponse(BaseModel):
    predicted: list[float]
    residuals: list[float]
    actual: list[float]

class PredVsActualResponse(BaseModel):
    predicted: list[float]
    actual: list[float]
    r2: float | None
    rmse: float | None
```

- [ ] **Step 3.2: 在 `viz_service.py` 添加回归可视化函数**

在文件末尾添加：

```python
async def get_residual_plot(task_id: str, db: AsyncSession) -> dict:
    """Load model and compute residuals for regression tasks."""
    import numpy as np

    task, model, X_val, y_val = await _load_task_model_data(task_id, db)

    y_pred = model.predict(X_val)
    residuals = (y_val - y_pred).tolist()

    return {
        "predicted": y_pred.tolist(),
        "residuals": residuals,
        "actual": y_val.tolist(),
    }


async def get_pred_vs_actual(task_id: str, db: AsyncSession) -> dict:
    """Predicted vs actual scatter data for regression."""
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error

    task, model, X_val, y_val = await _load_task_model_data(task_id, db)
    y_pred = model.predict(X_val)

    try:
        r2 = round(float(r2_score(y_val, y_pred)), 4)
        rmse = round(float(np.sqrt(mean_squared_error(y_val, y_pred))), 4)
    except Exception:
        r2, rmse = None, None

    return {
        "predicted": y_pred.tolist(),
        "actual": y_val.tolist(),
        "r2": r2,
        "rmse": rmse,
    }
```

**注意（Issue 3 修复）：** `_load_task_model_data` 是一个**新增的**辅助函数，与 viz_service.py 中现有的 `_get_task_and_dataset` 和 `_load_and_split_data` 并存（不替换它们）。区别：现有的 `_load_and_split_data` 使用 `stratify=y.values`（适合分类），此处**故意不用 stratify**，因为回归目标是连续值，stratify 会报错。

在 `viz_service.py` 文件顶部已有的 imports 下面（不要重复导入），添加：

```python
async def _load_task_model_data(task_id: str, db: AsyncSession):
    """New helper for regression viz: load task + model + val split (no stratify).
    
    Note: viz_service.py already has _get_task_and_dataset and _load_and_split_data
    for classification. This helper is separate and intentionally omits stratify=
    because regression targets are continuous floats (stratify would raise ValueError).
    """
    from sklearn.model_selection import train_test_split

    result = await db.execute(select(TrainingTask).where(TrainingTask.id == task_id))
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != "SUCCESS":
        raise HTTPException(status_code=400, detail="Task not completed successfully")
    if not task.model_path:
        raise HTTPException(status_code=404, detail="Model file not found")

    model = _load_model(task.model_path)   # use existing _load_model helper

    result2 = await db.execute(select(Dataset).where(Dataset.id == task.dataset_id))
    dataset = result2.scalar_one_or_none()
    df = load_dataframe(dataset.file_path)
    X, y, _, _ = prepare_training_frame(df, task.target_column)
    # No stratify — works for both regression and classification
    _, X_val, _, y_val = train_test_split(
        X.values, y.values, test_size=task.test_size, random_state=42
    )
    return task, model, X_val, y_val
```

- [ ] **Step 3.3: 新增路由端点**

在 `visualization.py` 添加（仿照现有端点格式）：

```python
from app.models.schemas import ResidualPlotResponse, PredVsActualResponse
from app.services.viz_service import get_residual_plot, get_pred_vs_actual

@router.get("/{task_id}/residual_plot", response_model=ResidualPlotResponse)
async def residual_plot_route(task_id: str, db: AsyncSession = Depends(get_db)):
    """Residual plot data for regression tasks."""
    result = await get_residual_plot(task_id=task_id, db=db)
    return ResidualPlotResponse(**result)

@router.get("/{task_id}/pred_vs_actual", response_model=PredVsActualResponse)
async def pred_vs_actual_route(task_id: str, db: AsyncSession = Depends(get_db)):
    """Predicted vs actual scatter data for regression."""
    result = await get_pred_vs_actual(task_id=task_id, db=db)
    return PredVsActualResponse(**result)
```

- [ ] **Step 3.4: 写测试**

新建 `ml_platform/tests/test_viz_regression.py`：

```python
"""Test regression visualization helpers."""
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def test_residual_computation():
    """Unit test: residuals = actual - predicted."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    residuals = y_true - y_pred
    assert len(residuals) == 4
    assert abs(residuals[0] - (-0.1)) < 1e-6


def test_r2_rmse_computation():
    np.random.seed(42)
    y_true = np.random.uniform(0, 10, 100)
    model = LinearRegression()
    X = y_true.reshape(-1, 1) + np.random.randn(100, 1) * 0.1
    model.fit(X, y_true)
    y_pred = model.predict(X)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    assert r2 > 0.9
    assert rmse < 1.0
```

- [ ] **Step 3.5: 运行测试**

```bash
cd ml_platform && python -m pytest tests/test_viz_regression.py -v
```

- [ ] **Step 3.6: Commit**

```bash
git add ml_platform/app/services/viz_service.py ml_platform/app/api/routes/visualization.py ml_platform/app/models/schemas.py ml_platform/tests/test_viz_regression.py
git commit -m "feat: add regression visualization endpoints (residual plot, pred vs actual)"
```

---

## Task 4: 模型部署服务（ModelDeployment DB + Deploy Service + API）

**文件:**
- 修改: `ml_platform/app/models/database.py` (新增 ModelDeployment + InferenceJob)
- 修改: `ml_platform/app/models/schemas.py` (新增部署 schema)
- 新建: `ml_platform/app/services/deploy_service.py`
- 新建: `ml_platform/app/api/routes/deploy.py`
- 修改: `ml_platform/app/main.py` (注册 deploy 路由)
- 测试: `ml_platform/tests/test_deploy_service.py`

### 详细步骤

- [ ] **Step 4.1: 新增数据库模型**

在 `database.py` 的 `TrainingLog` 类之后添加：

```python
# ---------------------------------------------------------------------------
# ModelDeployment
# ---------------------------------------------------------------------------

class ModelDeployment(Base):
    __tablename__ = "model_deployments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    task_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("training_tasks.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), default="active")  # active/paused/error
    max_batch_size: Mapped[int] = mapped_column(default=100)
    request_count: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    task: Mapped["TrainingTask"] = relationship()
    jobs: Mapped[list["InferenceJob"]] = relationship(
        back_populates="deployment", cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# InferenceJob
# ---------------------------------------------------------------------------

class InferenceJob(Base):
    __tablename__ = "inference_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    deployment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("model_deployments.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending/processing/completed/failed
    input_rows: Mapped[int | None] = mapped_column(default=None)
    predictions: Mapped[list | None] = mapped_column(JSON, default=None)
    probabilities: Mapped[list | None] = mapped_column(JSON, default=None)
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    deployment: Mapped[ModelDeployment] = relationship(back_populates="jobs")
```

确保在 `create_all` 时这些表会被创建（lifespan 里已有 `Base.metadata.create_all`，无需额外改动）。

- [ ] **Step 4.2: 新增 deploy schemas**

在 `schemas.py` 添加：

```python
class DeployRequest(BaseModel):
    name: str
    description: str = ""
    max_batch_size: int = 100

class DeploymentEndpoints(BaseModel):
    predict: str
    result_template: str

class DeploymentResponse(BaseModel):
    deployment_id: str
    task_id: str
    name: str
    status: str
    request_count: int
    created_at: datetime
    endpoints: DeploymentEndpoints
    model_config = ConfigDict(from_attributes=True)

class DeploymentListResponse(BaseModel):
    deployments: list[DeploymentResponse]
    total: int

class InferenceRequest(BaseModel):
    rows: list[dict] = []
    include_probabilities: bool = False

class InferenceJobResponse(BaseModel):
    job_id: str
    deployment_id: str
    status: str
    predictions: list | None = None
    probabilities: list | None = None
    input_rows: int | None = None
    error_message: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
    model_config = ConfigDict(from_attributes=True)
```

- [ ] **Step 4.3: 新建 `deploy_service.py`**

新建 `ml_platform/app/services/deploy_service.py`：

```python
"""Model deployment service with LRU model cache."""
from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import joblib
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import InferenceJob, ModelDeployment, TrainingTask, _utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LRU Model Cache
# ---------------------------------------------------------------------------

class _ModelCache:
    """LRU cache for loaded joblib models."""

    def __init__(self, max_size: int = 10):
        self._cache: OrderedDict[str, object] = OrderedDict()
        self._max_size = max_size

    def get(self, deployment_id: str, model_path: str):
        if deployment_id in self._cache:
            self._cache.move_to_end(deployment_id)
            return self._cache[deployment_id]
        model = joblib.load(model_path)
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[deployment_id] = model
        return model

    def evict(self, deployment_id: str):
        self._cache.pop(deployment_id, None)


_cache = _ModelCache(max_size=10)


# ---------------------------------------------------------------------------
# Service Functions
# ---------------------------------------------------------------------------

def _make_endpoints(deployment_id: str, base_url: str = "") -> dict:
    return {
        "predict": f"{base_url}/inference/{deployment_id}/predict",
        "result_template": f"{base_url}/inference/{deployment_id}/result/{{job_id}}",
    }


async def create_deployment(
    task_id: str, name: str, description: str, max_batch_size: int, db: AsyncSession
) -> dict:
    # Verify task exists and succeeded
    result = await db.execute(select(TrainingTask).where(TrainingTask.id == task_id))
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Training task not found")
    if task.status != "SUCCESS":
        raise HTTPException(status_code=400, detail="Only successful training tasks can be deployed")
    if not task.model_path or not Path(task.model_path).exists():
        raise HTTPException(status_code=400, detail="Model file not found on disk")

    deployment = ModelDeployment(
        task_id=task_id,
        name=name,
        description=description,
        max_batch_size=max_batch_size,
        status="active",
        request_count=0,
    )
    db.add(deployment)
    await db.flush()  # get the generated id

    return {
        "deployment_id": deployment.id,
        "task_id": task_id,
        "name": deployment.name,
        "status": deployment.status,
        "request_count": deployment.request_count,
        "created_at": deployment.created_at,
        "endpoints": _make_endpoints(deployment.id),
    }


async def list_deployments(page: int, page_size: int, db: AsyncSession) -> dict:
    total_result = await db.execute(select(func.count()).select_from(ModelDeployment))
    total = total_result.scalar_one()

    offset = (page - 1) * page_size
    result = await db.execute(
        select(ModelDeployment).order_by(ModelDeployment.created_at.desc()).offset(offset).limit(page_size)
    )
    deployments = result.scalars().all()

    return {
        "deployments": [
            {
                "deployment_id": d.id,
                "task_id": d.task_id,
                "name": d.name,
                "status": d.status,
                "request_count": d.request_count,
                "created_at": d.created_at,
                "endpoints": _make_endpoints(d.id),
            }
            for d in deployments
        ],
        "total": total,
    }


async def delete_deployment(deployment_id: str, db: AsyncSession) -> None:
    result = await db.execute(select(ModelDeployment).where(ModelDeployment.id == deployment_id))
    deployment = result.scalar_one_or_none()
    if deployment is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    _cache.evict(deployment_id)
    await db.delete(deployment)


async def update_deployment_status(deployment_id: str, status: str, db: AsyncSession) -> None:
    if status not in ("active", "paused"):
        raise HTTPException(status_code=400, detail="Status must be 'active' or 'paused'")
    result = await db.execute(select(ModelDeployment).where(ModelDeployment.id == deployment_id))
    deployment = result.scalar_one_or_none()
    if deployment is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    deployment.status = status


async def run_inference(
    deployment_id: str, rows: list[dict], include_probabilities: bool, db: AsyncSession
) -> dict:
    """Synchronous small-batch inference. Returns predictions immediately."""
    from app.models.database import Dataset
    from app.services.prediction_service import load_dataframe, prepare_training_frame, prepare_prediction_frame

    result = await db.execute(
        select(ModelDeployment, TrainingTask)
        .join(TrainingTask, ModelDeployment.task_id == TrainingTask.id)
        .where(ModelDeployment.id == deployment_id)
    )
    row = result.first()
    if row is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    deployment, task = row

    if deployment.status != "active":
        raise HTTPException(status_code=400, detail="Deployment is paused")

    model = _cache.get(deployment_id, task.model_path)

    # Load training data to derive encoders for consistent preprocessing
    # (Issue 2 fix: prepare_prediction_frame signature is (training_df, rows, target_column))
    ds_result = await db.execute(select(Dataset).where(Dataset.id == task.dataset_id))
    dataset = ds_result.scalar_one_or_none()
    training_df = load_dataframe(dataset.file_path)

    # prepare_prediction_frame(training_df, rows_list, target_column) → pd.DataFrame
    X_pred = prepare_prediction_frame(training_df, rows, task.target_column)
    predictions_raw = model.predict(X_pred).tolist()

    # Decode classification target labels if applicable
    _, _, _, target_encoder = prepare_training_frame(training_df, task.target_column)

    # Decode if classification
    if target_encoder is not None:
        try:
            predictions = target_encoder.inverse_transform(
                [int(p) for p in predictions_raw]
            ).tolist()
        except Exception:
            predictions = predictions_raw
    else:
        predictions = predictions_raw

    probabilities = None
    if include_probabilities and hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X_pred).tolist()
        except Exception:
            pass

    # Update request count
    deployment.request_count = (deployment.request_count or 0) + 1

    # Create inference job record
    job = InferenceJob(
        deployment_id=deployment_id,
        status="completed",
        input_rows=len(rows),
        predictions=predictions,
        probabilities=probabilities,
        completed_at=_utcnow(),
    )
    db.add(job)
    await db.flush()

    return {
        "job_id": job.id,
        "deployment_id": deployment_id,
        "status": "completed",
        "predictions": predictions,
        "probabilities": probabilities,
        "input_rows": len(rows),
        "error_message": None,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
    }


async def get_inference_result(deployment_id: str, job_id: str, db: AsyncSession) -> dict:
    result = await db.execute(
        select(InferenceJob).where(
            InferenceJob.id == job_id,
            InferenceJob.deployment_id == deployment_id
        )
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Inference job not found")
    return {
        "job_id": job.id,
        "deployment_id": deployment_id,
        "status": job.status,
        "predictions": job.predictions,
        "probabilities": job.probabilities,
        "input_rows": job.input_rows,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
    }
```

- [ ] **Step 4.4: 新建 `deploy.py` 路由**

新建 `ml_platform/app/api/routes/deploy.py`。

**架构说明：** 使用两个独立 router，与现有路由模式一致（参考 `visualization.py` 中 `prefix="/viz"`）：
- `deploy_router` → 前缀 `/deploy`，在 `main.py` 注册时加 `prefix="/api"`，最终路径 `/api/deploy/...`
- `inference_router` → 前缀 `/inference`，在 `main.py` 注册时**不加** `/api` 前缀，最终路径 `/inference/...`（设计上 url1/url2 不走 `/api` 命名空间）

```python
"""Model deployment and inference routes."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.models.schemas import (
    DeployRequest, DeploymentResponse, DeploymentListResponse,
    InferenceJobResponse, InferenceRequest,
)
from app.services.deploy_service import (
    create_deployment, delete_deployment, get_inference_result,
    list_deployments, run_inference, update_deployment_status,
)

deploy_router = APIRouter(prefix="/deploy", tags=["Model Deployment"])
inference_router = APIRouter(prefix="/inference", tags=["Inference"])


@deploy_router.post("/{task_id}", response_model=DeploymentResponse)
async def deploy_model(
    task_id: str,
    body: DeployRequest,
    db: AsyncSession = Depends(get_db),
):
    """Deploy a trained model and get prediction URL."""
    result = await create_deployment(
        task_id=task_id,
        name=body.name,
        description=body.description,
        max_batch_size=body.max_batch_size,
        db=db,
    )
    return DeploymentResponse(
        deployment_id=result["deployment_id"],
        task_id=result["task_id"],
        name=result["name"],
        status=result["status"],
        request_count=result["request_count"],
        created_at=result["created_at"],
        endpoints=result["endpoints"],
    )


# 注意：/list 必须在 /{deployment_id} 之前声明，否则 FastAPI 会将 "list" 捕获为 deployment_id
@deploy_router.get("/list", response_model=DeploymentListResponse)
async def list_deployments_route(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    return await list_deployments(page=page, page_size=page_size, db=db)


@deploy_router.delete("/{deployment_id}")
async def delete_deployment_route(deployment_id: str, db: AsyncSession = Depends(get_db)):
    await delete_deployment(deployment_id=deployment_id, db=db)
    return {"message": "Deployment deleted", "id": deployment_id}


@deploy_router.patch("/{deployment_id}/status")
async def update_status_route(
    deployment_id: str,
    status: str = Query(..., pattern="^(active|paused)$"),
    db: AsyncSession = Depends(get_db),
):
    await update_deployment_status(deployment_id=deployment_id, status=status, db=db)
    return {"message": f"Status updated to {status}"}


@inference_router.post("/{deployment_id}/predict", response_model=InferenceJobResponse)
async def predict_route(
    deployment_id: str,
    body: InferenceRequest,
    db: AsyncSession = Depends(get_db),
):
    """Submit prediction request (url1). Returns result immediately for small batches."""
    result = await run_inference(
        deployment_id=deployment_id,
        rows=body.rows,
        include_probabilities=body.include_probabilities,
        db=db,
    )
    return InferenceJobResponse(**result)


@inference_router.get("/{deployment_id}/result/{job_id}", response_model=InferenceJobResponse)
async def get_result_route(
    deployment_id: str,
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get prediction result by job ID (url2)."""
    result = await get_inference_result(deployment_id=deployment_id, job_id=job_id, db=db)
    return InferenceJobResponse(**result)
```

- [ ] **Step 4.5: 在 `main.py` 注册 deploy 路由**

在 `main.py` 中 import 并注册两个路由：

```python
from app.api.routes.deploy import deploy_router, inference_router

# deploy 管理接口走 /api 前缀，与其他路由一致
app.include_router(deploy_router, prefix="/api")   # → /api/deploy/...
# inference 接口不走 /api 前缀（url1/url2 直接暴露）
app.include_router(inference_router)               # → /inference/...
```

- [ ] **Step 4.6: 写部署服务测试**

新建 `ml_platform/tests/test_deploy_service.py`：

```python
"""Tests for deploy service model cache and logic."""
from app.services.deploy_service import _ModelCache
import pytest


def test_model_cache_lru_eviction(tmp_path):
    """LRU cache evicts oldest when full."""
    import joblib
    from sklearn.linear_model import LinearRegression
    import numpy as np

    # Create 3 tiny model files
    paths = []
    for i in range(3):
        m = LinearRegression()
        m.fit([[1]], [1])
        p = str(tmp_path / f"m{i}.joblib")
        joblib.dump(m, p)
        paths.append(p)

    cache = _ModelCache(max_size=2)
    cache.get("d0", paths[0])
    cache.get("d1", paths[1])
    assert "d0" in cache._cache
    assert "d1" in cache._cache

    # Adding d2 should evict d0 (LRU)
    cache.get("d2", paths[2])
    assert "d0" not in cache._cache
    assert "d1" in cache._cache
    assert "d2" in cache._cache


def test_model_cache_evict():
    cache = _ModelCache(max_size=5)
    cache._cache["fake_id"] = "fake_model"
    cache.evict("fake_id")
    assert "fake_id" not in cache._cache


def test_model_cache_miss_returns_model(tmp_path):
    import joblib
    from sklearn.linear_model import LinearRegression
    m = LinearRegression()
    m.fit([[1], [2]], [1, 2])
    p = str(tmp_path / "model.joblib")
    joblib.dump(m, p)

    cache = _ModelCache(max_size=5)
    loaded = cache.get("dep1", p)
    assert hasattr(loaded, "predict")
```

- [ ] **Step 4.7: 运行测试**

```bash
cd ml_platform && python -m pytest tests/test_deploy_service.py -v
```

- [ ] **Step 4.8: Commit**

```bash
git add ml_platform/app/models/database.py ml_platform/app/models/schemas.py ml_platform/app/services/deploy_service.py ml_platform/app/api/routes/deploy.py ml_platform/app/main.py ml_platform/tests/test_deploy_service.py
git commit -m "feat: model deployment service with LRU cache and inference endpoints"
```

---

## Task 5: 前端 — 训练配置增强（回归模型选项）

**文件:**
- 修改: `ml_platform_web/src/pages/TrainingConfig.jsx`
- 修改: `ml_platform_web/src/services/api.js` (新增 correlation + deploy API)

### 详细步骤

- [ ] **Step 5.1: 更新 `api.js`**

在 `api.js` 中添加：

```javascript
// 在 api.js 顶部，现有 `const api = axios.create(...)` 之后，添加第二个实例：
// （inference 路由没有 /api 前缀，url: /inference/{id}/predict）
const inferenceApi = axios.create({
  baseURL: 'http://127.0.0.1:8000',
  timeout: 30000,
});
inferenceApi.interceptors.response.use((r) => r.data, (e) => Promise.reject(e));

// Correlation & distribution
export const dataEnhancedApi = {
  getCorrelation: (datasetId, method = 'pearson') =>
    api.get(`/data/${datasetId}/correlation?method=${method}`),
  getTargetDistribution: (datasetId, targetColumn) =>
    api.get(`/data/${datasetId}/target_distribution?target_column=${encodeURIComponent(targetColumn)}`),
};

// Deploy API
// 注意：predict / getResult 使用 inferenceApi（无 /api 前缀），其余使用 api（有 /api 前缀）
export const deployApi = {
  createDeployment: (taskId, payload) => api.post(`/deploy/${taskId}`, payload),
  listDeployments: (params = {}) => api.get('/deploy/list', { params }),
  deleteDeployment: (deploymentId) => api.delete(`/deploy/${deploymentId}`),
  updateStatus: (deploymentId, status) =>
    api.patch(`/deploy/${deploymentId}/status?status=${status}`),
  predict: (deploymentId, payload) =>
    inferenceApi.post(`/inference/${deploymentId}/predict`, payload),
  getResult: (deploymentId, jobId) =>
    inferenceApi.get(`/inference/${deploymentId}/result/${jobId}`),
};
```

- [ ] **Step 5.2: 更新 `TrainingConfig.jsx` — 添加回归模型选项**

找到模型选择部分（`Select` 组件），确保在 `listModels` 返回的列表中，回归模型也能正确显示。关键改动：

1. 在模型选择下拉旁显示任务类型标签（通过 `detect_task_type` 调用后端）
2. 将模型分组显示：
   - 分类模型：random_forest, xgboost, lightgbm, logistic_regression, svm, mlp
   - 回归模型：random_forest_regressor, xgboost_regressor, lightgbm_regressor, linear_regression, ridge, lasso, svr, mlp_regressor

在 `TrainingConfig.jsx` 中找到模型选择 `<Select>` 块，替换为：

```jsx
<Form.Item label="选择模型" name="modelType" rules={[{required: true}]}>
  <Select placeholder="请选择模型">
    <Select.OptGroup label="分类模型">
      {['random_forest','xgboost','lightgbm','logistic_regression','svm','mlp'].map(m => (
        <Select.Option key={m} value={m}>{m}</Select.Option>
      ))}
    </Select.OptGroup>
    <Select.OptGroup label="回归模型">
      {['random_forest_regressor','xgboost_regressor','lightgbm_regressor',
        'linear_regression','ridge','lasso','svr','mlp_regressor'].map(m => (
        <Select.Option key={m} value={m}>{m}</Select.Option>
      ))}
    </Select.OptGroup>
  </Select>
</Form.Item>
```

同时更新评估指标选项，根据模型类型（分类/回归）动态切换：

```jsx
// 在 form watch 或 onValuesChange 中检测模型类型
const isRegression = REGRESSION_MODELS.includes(selectedModel);
const metricOptions = isRegression
  ? ['rmse', 'mae', 'r2', 'mse', 'mape']
  : ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'log_loss'];
```

- [ ] **Step 5.3: 手动验证前端**

启动前后端：
```bash
./scripts/start-dev.ps1
```

访问 http://127.0.0.1:3000/training/config ，确认：
- 模型选择下拉有分类/回归分组
- 选择回归模型后，评估指标切换为 rmse/r2/mae

- [ ] **Step 5.4: Commit**

```bash
git add ml_platform_web/src/services/api.js ml_platform_web/src/pages/TrainingConfig.jsx
git commit -m "feat: frontend training config - add regression model grouping and dynamic metrics"
```

---

## Task 6: 前端 — 模型部署页面（新建 ModelDeploy.jsx）

**文件:**
- 新建: `ml_platform_web/src/pages/ModelDeploy.jsx`
- 修改: `ml_platform_web/src/components/layout/Sidebar.jsx`
- 修改: `ml_platform_web/src/App.jsx`

### 详细步骤

- [ ] **Step 6.1: 设计系统查询（ui-ux-pro-max）**

运行设计查询以获取适合 Dashboard/Admin 的样式建议：

```bash
python .trae/skills/ui-ux-pro-max/cli/assets/scripts/search.py "admin panel deployment management" --stack react -n 3
python .trae/skills/ui-ux-pro-max/cli/assets/scripts/search.py "table with copy url button" --stack react -n 3
```

- [ ] **Step 6.2: 新建 `ModelDeploy.jsx`**

创建 `ml_platform_web/src/pages/ModelDeploy.jsx`，实现以下功能：

**布局：**
- 顶部：统计卡片（部署总数、活跃数、总调用次数）
- 中部：部署列表 Table（名称、模型类型、状态、调用次数、操作）
- 底部：选中部署的详情面板（URL展示、在线测试、结果查看）

**关键 UI 元素：**

```jsx
import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Tag, Input, Space, message, Tooltip, Divider, 
         Statistic, Row, Col, Select, Badge, Spin, Typography, Modal } from 'antd';
import { CopyOutlined, PlayCircleOutlined, PauseCircleOutlined, 
         DeleteOutlined, LinkOutlined, CloudUploadOutlined } from '@ant-design/icons';
import { deployApi } from '../services/api';

const { Text, Paragraph } = Typography;

// URL 展示 + 一键复制组件
const CopyableUrl = ({ url, label }) => (
  <div style={{ marginBottom: 8 }}>
    <Text type="secondary">{label}:</Text>
    <Space style={{ width: '100%', marginTop: 4 }}>
      <Text code style={{ flex: 1, wordBreak: 'break-all' }}>{url}</Text>
      <Tooltip title="复制">
        <Button 
          icon={<CopyOutlined />} 
          size="small"
          onClick={() => { navigator.clipboard.writeText(url); message.success('已复制'); }}
        />
      </Tooltip>
    </Space>
  </div>
);

export default function ModelDeploy() {
  const [deployments, setDeployments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState(null);
  const [testInput, setTestInput] = useState('[\n  {"feature1": 1.0, "feature2": "A"}\n]');
  const [testResult, setTestResult] = useState(null);
  const [testLoading, setTestLoading] = useState(false);

  const fetchDeployments = async () => {
    setLoading(true);
    try {
      const data = await deployApi.listDeployments();
      setDeployments(data.deployments || []);
    } catch(e) {
      message.error('加载部署列表失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchDeployments(); }, []);

  const handleDelete = async (id) => {
    try {
      await deployApi.deleteDeployment(id);
      message.success('已删除');
      fetchDeployments();
      if (selected?.deployment_id === id) setSelected(null);
    } catch(e) { message.error('删除失败'); }
  };

  const handleToggleStatus = async (deployment) => {
    const newStatus = deployment.status === 'active' ? 'paused' : 'active';
    try {
      await deployApi.updateStatus(deployment.deployment_id, newStatus);
      message.success(`已${newStatus === 'active' ? '恢复' : '暂停'}`);
      fetchDeployments();
    } catch(e) { message.error('操作失败'); }
  };

  const handleTest = async () => {
    if (!selected) return;
    setTestLoading(true);
    setTestResult(null);
    try {
      let rows;
      try { rows = JSON.parse(testInput); } 
      catch(e) { message.error('JSON 格式错误'); setTestLoading(false); return; }
      if (!Array.isArray(rows)) rows = [rows];
      const result = await deployApi.predict(selected.deployment_id, {
        rows, include_probabilities: true
      });
      setTestResult(result);
    } catch(e) {
      message.error('预测失败: ' + (e?.detail || e?.message || '未知错误'));
    } finally { setTestLoading(false); }
  };

  const columns = [
    { title: '部署名称', dataIndex: 'name', key: 'name', render: (v, r) => (
      <a onClick={() => setSelected(r)}>{v}</a>
    )},
    { title: '状态', dataIndex: 'status', key: 'status', render: (s) => (
      <Badge status={s === 'active' ? 'success' : 'default'} text={s === 'active' ? '活跃' : '暂停'} />
    )},
    { title: '调用次数', dataIndex: 'request_count', key: 'request_count' },
    { title: '创建时间', dataIndex: 'created_at', key: 'created_at', 
      render: (v) => new Date(v).toLocaleString('zh-CN') },
    { title: '操作', key: 'action', render: (_, r) => (
      <Space>
        <Tooltip title={r.status === 'active' ? '暂停' : '恢复'}>
          <Button 
            size="small" 
            icon={r.status === 'active' ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
            onClick={() => handleToggleStatus(r)}
          />
        </Tooltip>
        <Tooltip title="删除">
          <Button size="small" danger icon={<DeleteOutlined />} onClick={() => handleDelete(r.deployment_id)} />
        </Tooltip>
      </Space>
    )},
  ];

  const baseUrl = 'http://127.0.0.1:8000';

  return (
    <div style={{ padding: 24 }}>
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={8}><Card><Statistic title="总部署数" value={deployments.length} /></Card></Col>
        <Col span={8}><Card><Statistic title="活跃部署" value={deployments.filter(d => d.status === 'active').length} /></Card></Col>
        <Col span={8}><Card><Statistic title="总调用次数" value={deployments.reduce((s, d) => s + (d.request_count || 0), 0)} /></Card></Col>
      </Row>

      <Card title="已部署模型" style={{ marginBottom: 24 }}>
        <Table
          columns={columns}
          dataSource={deployments}
          rowKey="deployment_id"
          loading={loading}
          size="small"
          rowClassName={(r) => r.deployment_id === selected?.deployment_id ? 'ant-table-row-selected' : ''}
        />
      </Card>

      {selected && (
        <Card title={`部署详情: ${selected.name}`} extra={<Button size="small" onClick={() => setSelected(null)}>关闭</Button>}>
          <CopyableUrl 
            label="预测接口 (url1) — POST" 
            url={`${baseUrl}/inference/${selected.deployment_id}/predict`} 
          />
          <CopyableUrl 
            label="结果查询 (url2) — GET" 
            url={`${baseUrl}/inference/${selected.deployment_id}/result/{job_id}`} 
          />

          <Divider>在线测试</Divider>
          <Text type="secondary" style={{ display: 'block', marginBottom: 8 }}>
            输入 JSON 数组（每个对象为一行特征数据）：
          </Text>
          <Input.TextArea
            value={testInput}
            onChange={e => setTestInput(e.target.value)}
            rows={6}
            style={{ fontFamily: 'monospace', marginBottom: 8 }}
          />
          <Button type="primary" icon={<PlayCircleOutlined />} onClick={handleTest} loading={testLoading}>
            发送预测请求
          </Button>

          {testResult && (
            <>
              <Divider>预测结果</Divider>
              <pre style={{ 
                background: '#f6f8fa', padding: 12, borderRadius: 6, 
                overflow: 'auto', maxHeight: 300, fontSize: 13
              }}>
                {JSON.stringify(testResult, null, 2)}
              </pre>
            </>
          )}

          <Divider>cURL 示例</Divider>
          <pre style={{ background: '#1e1e1e', color: '#d4d4d4', padding: 12, borderRadius: 6, fontSize: 12 }}>
{`curl -X POST ${baseUrl}/inference/${selected.deployment_id}/predict \\
  -H "Content-Type: application/json" \\
  -d '{"rows": [{"feat1": 1.5, "feat2": "A"}]}'`}
          </pre>
        </Card>
      )}
    </div>
  );
}
```

- [ ] **Step 6.3: 更新 Sidebar.jsx**

在 Sidebar 的菜单配置中，在"模型管理"之后添加"模型部署"入口：

```jsx
{ key: '/deploy', icon: <CloudUploadOutlined />, label: '模型部署' }
```

并确保 import `CloudUploadOutlined`。

- [ ] **Step 6.4: 更新 App.jsx**

在路由配置中添加：

```jsx
import ModelDeploy from './pages/ModelDeploy';
// ...在 routes 数组中:
{ path: '/deploy', element: <ModelDeploy /> }
```

- [ ] **Step 6.5: 手动验证**

访问 http://127.0.0.1:3000/deploy ，确认：
- 侧边栏有"模型部署"入口
- 页面显示统计卡片和空表格
- 从模型管理页部署一个模型后，表格出现该部署
- 点击部署行，展示 url1/url2 和在线测试面板
- 在线测试能正确返回预测结果

- [ ] **Step 6.6: Commit**

```bash
git add ml_platform_web/src/pages/ModelDeploy.jsx ml_platform_web/src/components/layout/Sidebar.jsx ml_platform_web/src/App.jsx ml_platform_web/src/services/api.js
git commit -m "feat: model deploy frontend page with URL display and online prediction tester"
```

---

## Task 7: 前端 — 模型管理页增加"部署"按钮

**文件:**
- 修改: `ml_platform_web/src/pages/ModelManagement.jsx`

### 详细步骤

- [ ] **Step 7.1: 在模型详情中添加"部署模型"按钮**

在 `ModelManagement.jsx` 的模型详情 Modal 或详情展开区域，找到操作按钮区域，添加：

```jsx
import { CloudUploadOutlined } from '@ant-design/icons';
import { deployApi } from '../services/api';
import { useNavigate } from 'react-router-dom';

// 在组件内：
const navigate = useNavigate();
const [deployModalVisible, setDeployModalVisible] = useState(false);
const [deployName, setDeployName] = useState('');

const handleDeploy = async (taskId) => {
  if (!deployName.trim()) { message.warning('请输入部署名称'); return; }
  try {
    await deployApi.createDeployment(taskId, { name: deployName, description: '' });
    message.success('部署成功！');
    setDeployModalVisible(false);
    navigate('/deploy');
  } catch(e) { message.error('部署失败: ' + (e?.detail || '未知错误')); }
};

// 在操作按钮区域：
<Button 
  icon={<CloudUploadOutlined />} 
  size="small"
  onClick={() => { setDeployName(model.model_type + '-deploy'); setDeployModalVisible(true); }}
>
  部署
</Button>

// Modal：
<Modal
  title="部署模型"
  open={deployModalVisible}
  onOk={() => handleDeploy(selectedTask?.id)}
  onCancel={() => setDeployModalVisible(false)}
>
  <Input 
    placeholder="部署名称（如: rf-prod）" 
    value={deployName} 
    onChange={e => setDeployName(e.target.value)} 
  />
</Modal>
```

- [ ] **Step 7.2: 手动验证**

从模型列表点击"部署"，输入名称，确认跳转到 /deploy 并显示新部署。

- [ ] **Step 7.3: Commit**

```bash
git add ml_platform_web/src/pages/ModelManagement.jsx
git commit -m "feat: add deploy button to model management page"
```

---

## 完成阶段（finishing-a-development-branch）

完成所有 Tasks 后，执行以下步骤：

- [ ] 运行所有后端测试：`cd ml_platform && python -m pytest tests/ -v`
- [ ] 运行 E2E 测试：`npx playwright test`（确保后端已启动）
- [ ] 提供 PR 选项

---

## 注意事项

1. **不要用 Celery**，当前架构用 asyncio + ThreadPoolExecutor，保持一致
2. **回归 Trainer 使用 KFold**（不是 StratifiedKFold），因为回归 y 无法分层
3. **inference 路由无 `/api/` 前缀**，main.py 注册时不加 prefix
4. **viz_service.py 中的辅助函数** `_load_task_model_data` 在 Task 3 中需要检查是否已存在类似逻辑，避免重复
5. **3次失败停下来**，汇报给用户具体问题
