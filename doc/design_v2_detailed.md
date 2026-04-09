# 傻子也会训练模型 — 详细设计文档 v2

> 版本：v2.0 | 日期：2026-04-08 | 作者：zhuow + Claude

---

## 一、项目定位与目标

### 1.1 项目名称
**傻子也会训练模型**（ML Training Platform）

### 1.2 核心理念
让没有深度机器学习背景的用户，也能通过**上传数据 → 选择模型 → 设置参数 → 一键训练 → 查看结果 → 部署推理**的简洁流程，完成从数据到模型到服务的全链路。

### 1.3 目标用户
- 数据分析师：有数据处理经验，但不熟悉 sklearn/PyTorch 代码
- 业务人员：需要快速验证数据假设，不想写代码
- ML 初学者：想通过可视化界面理解训练过程
- 小团队：需要快速部署预测 API，无专职 MLOps

### 1.4 核心能力矩阵

| 能力 | 描述 | 当前状态 |
|------|------|----------|
| 数据管理 | 上传CSV/Parquet/Excel，预览、统计、删除 | ✅ 已实现 |
| 模型训练 | 6种ML模型，超参数配置，交叉验证 | ✅ 已实现 |
| 实时监控 | WebSocket推送训练进度和日志 | ✅ 已实现 |
| 训练日志 | 按任务独立日志，支持导出 | ✅ 已实现 |
| 实验追踪 | MLflow 集成，参数/指标/模型自动记录 | ✅ 已实现 |
| 可视化 | 混淆矩阵、ROC、特征重要性、学习曲线 | ✅ 已实现 |
| SHAP解释 | TreeExplainer/KernelExplainer | ✅ 已实现 |
| 模型推理 | 加载模型 + 单条/批量预测 | ⚡ 基础实现 |
| URL部署 | 输入URL→模型→输出URL | ❌ 未实现 |
| 深度学习 | PyTorch CNN/RNN/Transformer | ❌ 未实现 |
| 回归支持 | 回归任务 + 回归指标 | ❌ 未实现 |
| 相关性分析 | 皮尔逊/斯皮尔曼相关矩阵 | ❌ 未实现 |
| 数据预处理 | 标准化、独热编码、缺失值策略 | ⚡ 基础实现 |

---

## 二、系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户浏览器                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  React 18 + Ant Design + ECharts                        │   │
│  │  ┌─────────┬──────────┬──────────┬────────┬──────────┐  │   │
│  │  │Dashboard│数据管理   │训练配置   │结果可视化│模型管理  │  │   │
│  │  └─────────┴──────────┴──────────┴────────┴──────────┘  │   │
│  └────────────────────────┬─────────────────────────────────┘   │
└───────────────────────────┼─────────────────────────────────────┘
                            │ HTTP / WebSocket
┌───────────────────────────┼─────────────────────────────────────┐
│  FastAPI Application      │                                      │
│  ┌────────────────────────┴──────────────────────────────────┐  │
│  │                    API Gateway Layer                       │  │
│  │  /api/data  /api/training  /api/viz  /api/models         │  │
│  │  /api/logs  /api/experiments  /api/deploy                │  │
│  │  /ws/training/{id}  /ws/logs/{id}                        │  │
│  └────────────────────────┬──────────────────────────────────┘  │
│  ┌────────────────────────┴──────────────────────────────────┐  │
│  │                    Service Layer                           │  │
│  │  TrainingService  DataService  VizService                 │  │
│  │  PredictionService  LogService  DeployService(新)         │  │
│  └────────────────────────┬──────────────────────────────────┘  │
│  ┌────────────────────────┴──────────────────────────────────┐  │
│  │                    Core Layer                              │  │
│  │  BaseTrainer → 6 ML Trainers + DeepLearning Trainers(新) │  │
│  │  TrainingLogger  EventBus  PreprocessPipeline(新)         │  │
│  └────────────────────────┬──────────────────────────────────┘  │
│  ┌────────────────────────┴──────────────────────────────────┐  │
│  │                    Storage Layer                           │  │
│  │  SQLite/PostgreSQL   File Storage   MLflow Tracking       │  │
│  │  (datasets, tasks)   (models, logs)  (experiments)        │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│  Model Serving Layer (新)  │                                     │
│  ┌────────────────────────┴──────────────────────────────────┐  │
│  │  /inference/{model_id}/predict   ← 输入数据URL             │  │
│  │  /inference/{model_id}/result    ← 获取预测结果URL          │  │
│  │  模型热加载 + 缓存  |  批量推理队列  |  结果持久化           │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 技术栈

| 层 | 技术 | 版本 | 用途 |
|---|---|---|---|
| 前端框架 | React | 18.2 | SPA |
| UI组件库 | Ant Design | 5.12 | 表单/表格/布局 |
| 图表库 | ECharts | 5.4 | 训练曲线/混淆矩阵/SHAP图 |
| 状态管理 | Redux Toolkit | 2.1 | 全局状态 |
| HTTP客户端 | Axios | 1.6 | API调用 |
| 后端框架 | FastAPI | 0.115 | REST API + WebSocket |
| ORM | SQLAlchemy | 2.0 | 异步数据库操作 |
| ML引擎 | scikit-learn | 1.6 | 传统ML模型 |
| 梯度提升 | XGBoost + LightGBM | latest | 高性能树模型 |
| 深度学习 | PyTorch | 2.x | CNN/RNN/Transformer (新增) |
| 可解释性 | SHAP | 0.46 | 模型解释 |
| 实验追踪 | MLflow | 2.19 | 参数/指标/模型记录 |
| E2E测试 | Playwright | 1.58 | 端到端自动化测试 |

---

## 三、功能模块详细设计

### 3.1 数据管理模块

#### 3.1.1 现有功能
- 文件上传（CSV/Parquet/Excel，最大200MB）
- 数据预览（前N行 + 列统计信息）
- 数据集列表（分页）
- 数据集删除

#### 3.1.2 新增设计：数据预处理配置

**目标**：让用户在训练前配置数据预处理策略，而不是硬编码在后端。

```
数据预处理配置 (PreprocessConfig)
├── 缺失值处理
│   ├── 数值型: 均值 / 中位数 / 众数 / 指定值 / 删除行
│   └── 类别型: 众数 / 指定值 / "Unknown" / 删除行
├── 特征编码
│   ├── 类别特征: LabelEncoding / OneHotEncoding / TargetEncoding
│   └── 有序特征: OrdinalEncoding (用户指定顺序)
├── 特征缩放
│   ├── StandardScaler (标准化, z-score)
│   ├── MinMaxScaler (归一化, 0-1)
│   └── RobustScaler (鲁棒缩放, 抗异常值)
├── 特征选择
│   ├── 手动选择/排除列
│   ├── 方差阈值过滤
│   └── 相关性过滤 (高相关特征自动提示)
└── 目标变量
    ├── X特征列选择
    ├── Y目标列选择
    └── 任务类型自动检测: 分类 / 回归 (根据Y的dtype和unique数)
```

**API 新增**:
```
POST /api/data/{dataset_id}/preprocess-preview
  Body: { preprocess_config }
  Response: { preview_rows, column_stats_after, warnings }

GET /api/data/{dataset_id}/correlation
  Query: ?method=pearson|spearman|kendall
  Response: { correlation_matrix, feature_names }
```

**Pydantic Schema**:
```python
class PreprocessConfig(BaseModel):
    missing_strategy_numeric: Literal["mean", "median", "mode", "constant", "drop"] = "median"
    missing_fill_value: Optional[float] = None
    missing_strategy_categorical: Literal["mode", "constant", "unknown", "drop"] = "unknown"
    encoding_strategy: Literal["label", "onehot", "target"] = "label"
    scaling_strategy: Optional[Literal["standard", "minmax", "robust"]] = None
    feature_columns: Optional[list[str]] = None      # None = 全部 (除target)
    exclude_columns: Optional[list[str]] = None
```

#### 3.1.3 新增设计：相关性分析

**前端**：ECharts 热力图展示相关性矩阵
- 颜色编码：红(正相关) → 白(无相关) → 蓝(负相关)
- 鼠标悬停显示精确值
- 支持筛选 Top-N 相关特征对
- 高相关性警告（r > 0.95 提示多重共线性）

---

### 3.2 模型训练模块

#### 3.2.1 现有模型

| 模型 | 类型 | 关键超参数 |
|------|------|-----------|
| RandomForest | 分类 | n_estimators, max_depth, min_samples_split |
| XGBoost | 分类 | n_estimators, learning_rate, max_depth, subsample |
| LightGBM | 分类 | n_estimators, learning_rate, num_leaves, max_depth |
| LogisticRegression | 分类 | C, solver, max_iter, penalty |
| SVM | 分类 | C, kernel, gamma, degree |
| MLP | 分类 | hidden_layer_sizes, activation, learning_rate, alpha |

#### 3.2.2 新增设计：回归模型支持

**核心改动**：训练器需要根据任务类型（分类/回归）自动切换。

| 新增模型 | 类型 | 关键超参数 |
|---------|------|-----------|
| LinearRegression | 回归 | fit_intercept |
| Ridge | 回归 | alpha |
| Lasso | 回归 | alpha |
| ElasticNet | 回归 | alpha, l1_ratio |
| RandomForestRegressor | 回归 | n_estimators, max_depth |
| XGBRegressor | 回归 | n_estimators, learning_rate, max_depth |
| LGBMRegressor | 回归 | n_estimators, learning_rate, num_leaves |
| SVR | 回归 | C, kernel, epsilon |
| MLPRegressor | 回归 | hidden_layer_sizes, activation, alpha |

**任务类型自动检测逻辑**:
```python
def detect_task_type(y: pd.Series) -> Literal["classification", "regression"]:
    if y.dtype == "object" or y.dtype.name == "category":
        return "classification"
    unique_ratio = y.nunique() / len(y)
    if y.dtype in ["int64", "int32"] and y.nunique() <= 20:
        return "classification"
    if unique_ratio > 0.05 and y.dtype in ["float64", "float32"]:
        return "regression"
    return "classification"  # default safe
```

**回归指标**:
- MSE (均方误差)
- RMSE (均方根误差)
- MAE (平均绝对误差)
- R² (决定系数)
- MAPE (平均绝对百分比误差)

#### 3.2.3 新增设计：深度学习模型

**Phase 2 新增**（PyTorch-based）:

| 模型 | 适用场景 | 关键超参数 |
|------|---------|-----------|
| DNN (全连接网络) | 表格数据通用 | layers, dropout, batch_norm, lr, epochs, batch_size |
| TabNet | 表格数据 (注意力机制) | n_d, n_a, n_steps, gamma, lr |

**深度学习通用超参数面板**:
```
训练参数
├── epochs: 10-1000 (默认100)
├── batch_size: 16/32/64/128/256 (默认64)
├── learning_rate: 1e-5 ~ 1e-1 (默认1e-3)
├── optimizer: Adam / SGD / AdamW / RMSProp
├── scheduler: StepLR / CosineAnnealing / ReduceOnPlateau / None
├── early_stopping: patience (默认10), min_delta
└── weight_decay: 0 ~ 0.1

网络结构
├── hidden_layers: [128, 64, 32] (可动态增减层)
├── dropout: 0.0 ~ 0.5 (每层独立或全局)
├── batch_normalization: true/false
└── activation: ReLU / LeakyReLU / GELU / Tanh
```

**PyTorch Trainer 设计**:
```python
class PyTorchBaseTrainer(BaseTrainer):
    """PyTorch模型训练基类"""
    
    def train(self, X_train, y_train, X_val, y_val, hyperparams, callback):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._build_model(X_train.shape[1], hyperparams).to(device)
        optimizer = self._build_optimizer(model, hyperparams)
        scheduler = self._build_scheduler(optimizer, hyperparams)
        
        train_loader = self._make_dataloader(X_train, y_train, hyperparams)
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(hyperparams.get("epochs", 100)):
            # Training loop
            train_loss = self._train_epoch(model, train_loader, optimizer, device)
            # Validation
            val_metrics = self._validate(model, X_val, y_val, device)
            # Scheduler step
            if scheduler: scheduler.step(val_metrics.get("val_loss"))
            # Early stopping
            if val_metrics["val_loss"] < best_val_loss - min_delta:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
                self._save_best(model)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            # Callback for real-time progress
            callback(step=epoch+1, total=total_epochs, metrics=val_metrics)
        
        return self._load_best(), final_metrics
```

#### 3.2.4 训练配置前端设计

```
┌─────────────────────────────────────────────────────────────┐
│  训练配置页                                                  │
│                                                             │
│  ① 选择数据集   [dropdown: 数据集列表]                       │
│     └─ 预览: 1000行 × 15列  |  查看统计  |  相关性分析       │
│                                                             │
│  ② 设置 X / Y                                               │
│     目标列(Y): [dropdown]  任务类型: [自动检测: 分类 ✓]       │
│     特征列(X): [全选] □col1 ☑col2 ☑col3 □col4(排除)        │
│     ⚠️ 提示: col4 与 Y 相关性=0.98, 建议排除(数据泄漏风险)   │
│                                                             │
│  ③ 数据预处理                                                │
│     缺失值(数值): [中位数 ▼]  缺失值(类别): [Unknown ▼]      │
│     特征编码: [LabelEncoding ▼]  特征缩放: [StandardScaler ▼]│
│                                                             │
│  ④ 选择模型   [tabs: 传统ML | 深度学习]                      │
│     传统ML:  ○RF  ●XGBoost  ○LightGBM  ○LR  ○SVM  ○MLP    │
│     深度学习: ○DNN  ○TabNet                                  │
│                                                             │
│  ⑤ 超参数设置                                                │
│     ┌─────────────────────────────────┐                     │
│     │  n_estimators: [100____] slider │                     │
│     │  learning_rate: [0.1____]       │                     │
│     │  max_depth: [6________] slider  │                     │
│     │  subsample: [0.8______]         │                     │
│     │  ☑ 交叉验证  折数: [5__]        │                     │
│     │  测试集比例: [0.2_____] slider  │                     │
│     └─────────────────────────────────┘                     │
│                                                             │
│  ⑥ 评估指标  ☑accuracy  ☑f1  □precision  □recall  ☑roc_auc │
│              (回归时显示: ☑rmse  ☑r2  □mae  □mape)          │
│                                                             │
│  [🚀 开始训练]                              [重置] [保存配置] │
└─────────────────────────────────────────────────────────────┘
```

---

### 3.3 训练监控模块

#### 3.3.1 现有功能
- 任务列表（状态筛选、分页）
- 实时进度（2秒轮询）
- WebSocket 日志流
- 停止训练

#### 3.3.2 增强设计：实时训练曲线

**前端增强**：
- **实时Loss/Metric曲线**：ECharts 动态折线图，每个fold/epoch一个数据点
- **多指标并排**：accuracy + f1 + loss 在同一图表中，双Y轴
- **折叠详情**：点击任务展开详细日志 + 当前最佳fold + 预计剩余时间
- **深度学习增强**：epoch级别粒度，train_loss vs val_loss 实时对比曲线

**WebSocket 消息格式增强**:
```json
{
  "type": "training_progress",
  "task_id": "uuid",
  "step": 3,
  "total_steps": 5,
  "progress": 60,
  "metrics": {
    "accuracy": 0.95,
    "f1": 0.93,
    "train_loss": 0.15,
    "val_loss": 0.18
  },
  "elapsed_seconds": 12.5,
  "eta_seconds": 8.3
}
```

---

### 3.4 可视化与解释性模块

#### 3.4.1 现有可视化

| 图表 | 数据来源 | 状态 |
|------|---------|------|
| 混淆矩阵 (Heatmap) | 预测 vs 真实 | ✅ |
| ROC曲线 | FPR/TPR/AUC | ✅ |
| 特征重要性 (Bar) | model.feature_importances_ | ✅ |
| 学习曲线 | per-fold metrics | ✅ |
| SHAP Summary | mean_abs_shap | ✅ |

#### 3.4.2 新增可视化

| 图表 | 类型 | 描述 | 适用场景 |
|------|------|------|---------|
| **相关性热力图** | Heatmap | 特征间皮尔逊/斯皮尔曼相关系数 | 数据探索阶段 |
| **残差图** | Scatter | 预测值 vs 残差 | 回归任务 |
| **预测 vs 实际散点图** | Scatter | y_pred vs y_true + 对角线 | 回归任务 |
| **SHAP Dependence Plot** | Scatter | 单特征SHAP值 vs 特征值 | 特征交互分析 |
| **SHAP Waterfall** | Bar | 单样本预测解释（基准值→最终预测）| 个案解释 |
| **Partial Dependence Plot** | Line | 特征值变化对预测的边际影响 | 特征效应分析 |
| **训练Loss对比曲线** | Line | train_loss vs val_loss (深度学习) | 过拟合诊断 |
| **类别分布图** | Bar/Pie | 目标变量分布 | 类别不平衡检测 |
| **指标雷达图** | Radar | 多模型多指标对比 | 模型选择 |

#### 3.4.3 可视化 API 新增

```
GET /api/viz/{task_id}/residual_plot
  → { predicted: [], actual: [], residuals: [] }

GET /api/viz/{task_id}/pred_vs_actual
  → { predicted: [], actual: [] }

GET /api/viz/{task_id}/shap_dependence?feature=col_name
  → { feature_values: [], shap_values: [], color_feature: str, color_values: [] }

GET /api/viz/{task_id}/shap_waterfall?sample_index=0
  → { base_value: float, features: [], shap_values: [], predicted_value: float }

GET /api/viz/{task_id}/partial_dependence?feature=col_name&grid_points=50
  → { feature_values: [], avg_predictions: [] }

GET /api/data/{dataset_id}/correlation?method=pearson
  → { matrix: [][], feature_names: [], top_pairs: [{f1, f2, corr}] }

GET /api/data/{dataset_id}/target_distribution?target_column=Y
  → { labels: [], counts: [], type: "classification"|"regression" }
```

#### 3.4.4 可视化页面布局

```
┌─────────────────────────────────────────────────────────────┐
│  结果可视化页                                                │
│                                                             │
│  [选择训练任务: dropdown]  模型: XGBoost  状态: ✅ 成功       │
│                                                             │
│  ┌─ 指标概览卡片 ──────────────────────────────────────────┐│
│  │ Accuracy: 95.2%  │ F1: 93.8%  │ ROC-AUC: 0.98  │ ...  ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  [Tabs: 模型评估 | 特征分析 | 可解释性 | 数据洞察]           │
│                                                             │
│  ── 模型评估 Tab ──                                         │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │  混淆矩阵    │  │  ROC 曲线    │                        │
│  │  (Heatmap)   │  │  (多类支持)  │                        │
│  └──────────────┘  └──────────────┘                        │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │  学习曲线    │  │  残差图(回归) │                        │
│  └──────────────┘  └──────────────┘                        │
│                                                             │
│  ── 特征分析 Tab ──                                         │
│  ┌──────────────────────┐  ┌──────────────────┐            │
│  │  特征重要性 (Top 20) │  │  相关性热力图    │            │
│  └──────────────────────┘  └──────────────────┘            │
│                                                             │
│  ── 可解释性 Tab ──                                         │
│  ┌──────────────────────┐  ┌──────────────────┐            │
│  │  SHAP Summary Plot   │  │  SHAP Dependence │            │
│  │  (蜂群图/Bar)        │  │  (选择特征)      │            │
│  └──────────────────────┘  └──────────────────┘            │
│  ┌──────────────────────┐  ┌──────────────────┐            │
│  │  SHAP Waterfall      │  │  PDP (偏依赖图)  │            │
│  │  (选择样本)          │  │  (选择特征)      │            │
│  └──────────────────────┘  └──────────────────┘            │
│                                                             │
│  [导出报告 PDF]  [下载图表 PNG]                              │
└─────────────────────────────────────────────────────────────┘
```

---

### 3.5 模型管理模块

#### 3.5.1 现有功能
- 模型列表（按类型筛选、分页）
- 模型详情（指标、超参数）
- 多模型对比
- 模型删除
- 单条预测（JSON输入）

#### 3.5.2 增强设计

**模型版本管理**:
```
模型管理
├── 模型列表 (按数据集分组)
│   ├── 最佳模型标记 ⭐
│   ├── 指标快速对比 (卡片/表格切换)
│   └── 批量操作 (删除/导出)
├── 模型详情
│   ├── 训练配置回溯 (数据集、参数、预处理)
│   ├── 完整指标面板
│   ├── 模型大小、训练耗时
│   └── 一键重训 (用相同配置)
└── 模型对比
    ├── 雷达图 (多指标)
    ├── 表格对比 (指标并排)
    └── 推荐最优模型 (加权评分)
```

---

### 3.6 模型部署与推理模块（核心新增）

#### 3.6.1 设计目标
用户训练好模型后，平台自动生成两个 URL：
- **Input URL** (`url1`): 提交预测数据
- **Output URL** (`url2`): 获取预测结果

#### 3.6.2 部署流程

```
用户点击"部署模型"
    ↓
平台生成 deployment_id
    ↓
模型加载到内存缓存 (LRU)
    ↓
生成两个端点:
    POST /inference/{deployment_id}/predict  ← url1 (输入)
    GET  /inference/{deployment_id}/result/{job_id}  ← url2 (输出)
    ↓
用户拿到 URL，可以集成到任何系统
```

#### 3.6.3 API 设计

```
# === 部署管理 ===

POST /api/deploy/{task_id}
  Body: { "name": "my-model-v1", "description": "...", "max_batch_size": 100 }
  Response: {
    "deployment_id": "dp-xxxx",
    "status": "active",
    "endpoints": {
      "predict": "http://host/inference/dp-xxxx/predict",
      "batch_predict": "http://host/inference/dp-xxxx/batch",
      "result": "http://host/inference/dp-xxxx/result/{job_id}"
    }
  }

GET /api/deploy/list
  → [{ deployment_id, name, model_type, status, created_at, request_count }]

DELETE /api/deploy/{deployment_id}
  → 卸载模型，回收资源

PATCH /api/deploy/{deployment_id}
  Body: { "status": "active"|"paused" }
  → 暂停/恢复服务

# === 推理端点 ===

POST /inference/{deployment_id}/predict
  Body: {
    "rows": [{"feature1": 1.5, "feature2": "A"}],  // 或
    "csv_url": "https://example.com/data.csv",       // 或
    "json_url": "https://example.com/data.json"
  }
  Response: {
    "job_id": "job-xxxx",                  // 批量时异步
    "predictions": [0, 1, 0],              // 同步小批量直接返回
    "probabilities": [[0.1, 0.9], ...],    // 可选
    "result_url": "/inference/dp-xxxx/result/job-xxxx"
  }

GET /inference/{deployment_id}/result/{job_id}
  Response: {
    "status": "completed",
    "predictions": [...],
    "probabilities": [...],
    "created_at": "...",
    "completed_at": "..."
  }

# === 批量推理 ===

POST /inference/{deployment_id}/batch
  Body: multipart/form-data (CSV file upload)
  Response: {
    "job_id": "job-xxxx",
    "status": "processing",
    "total_rows": 10000,
    "result_url": "/inference/dp-xxxx/result/job-xxxx"
  }

GET /inference/{deployment_id}/result/{job_id}/download
  → CSV file with predictions appended
```

#### 3.6.4 数据库设计

```python
class ModelDeployment(Base):
    __tablename__ = "model_deployments"
    
    id = Column(String, primary_key=True)         # dp-{uuid}
    task_id = Column(String, ForeignKey("training_tasks.id"))
    name = Column(String, nullable=False)
    description = Column(String, default="")
    status = Column(String, default="active")      # active / paused / error
    max_batch_size = Column(Integer, default=100)
    request_count = Column(Integer, default=0)     # 调用计数
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

class InferenceJob(Base):
    __tablename__ = "inference_jobs"
    
    id = Column(String, primary_key=True)          # job-{uuid}
    deployment_id = Column(String, ForeignKey("model_deployments.id"))
    status = Column(String, default="pending")     # pending / processing / completed / failed
    input_rows = Column(Integer)
    predictions = Column(JSON)                     # 结果JSON
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime)
    completed_at = Column(DateTime, nullable=True)
```

#### 3.6.5 模型缓存策略

```python
from functools import lru_cache
import joblib

class ModelCache:
    """LRU缓存已部署模型，避免重复加载"""
    
    def __init__(self, max_size=10):
        self._cache = OrderedDict()
        self._max_size = max_size
    
    def get_model(self, deployment_id: str, model_path: str):
        if deployment_id in self._cache:
            self._cache.move_to_end(deployment_id)
            return self._cache[deployment_id]
        
        model = joblib.load(model_path)
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)  # evict LRU
        self._cache[deployment_id] = model
        return model
```

#### 3.6.6 部署页面设计

```
┌─────────────────────────────────────────────────────────────┐
│  模型部署管理                                                │
│                                                             │
│  ┌─ 已部署模型 ────────────────────────────────────────────┐│
│  │ 名称          │ 模型    │ 状态  │ 调用次数 │ 操作       ││
│  │ my-model-v1  │ XGBoost │ 🟢活跃│ 1,234   │ [暂停][删除]││
│  │ rf-prod      │ RF      │ ⏸暂停 │ 567     │ [恢复][删除]││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  ┌─ 部署详情: my-model-v1 ─────────────────────────────────┐│
│  │                                                         ││
│  │  预测API (url1):                                        ││
│  │  POST http://localhost:8000/inference/dp-xxx/predict     ││
│  │  [📋 复制URL]                                           ││
│  │                                                         ││
│  │  请求示例:                                               ││
│  │  curl -X POST url1 \                                    ││
│  │    -H "Content-Type: application/json" \                ││
│  │    -d '{"rows": [{"col1": 1.5, "col2": "A"}]}'        ││
│  │                                                         ││
│  │  结果查询 (url2):                                        ││
│  │  GET http://localhost:8000/inference/dp-xxx/result/{id}  ││
│  │  [📋 复制URL]                                           ││
│  │                                                         ││
│  │  ── 在线测试 ──                                          ││
│  │  [JSON编辑器: 输入测试数据]                               ││
│  │  [🚀 发送请求]                                           ││
│  │  结果: { "predictions": [1], "probabilities": [...] }   ││
│  │                                                         ││
│  │  ── 批量预测 ──                                          ││
│  │  [拖拽上传CSV]  或  [输入数据URL]                         ││
│  │  [开始批量预测]                                           ││
│  │  进度: ████████░░ 80% (8000/10000)                      ││
│  │  [下载结果CSV]                                           ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## 四、API 全景路由表

### 4.1 现有路由 (已实现)

| 模块 | 方法 | 路径 | 说明 |
|------|------|------|------|
| Health | GET | `/health` | 健康检查 |
| Data | POST | `/api/data/upload` | 上传数据集 |
| Data | GET | `/api/data/list` | 数据集列表 |
| Data | GET | `/api/data/{id}/preview` | 数据预览 |
| Data | DELETE | `/api/data/{id}` | 删除数据集 |
| Training | POST | `/api/training/start` | 启动训练 |
| Training | GET | `/api/training/{id}/status` | 训练状态 |
| Training | POST | `/api/training/{id}/stop` | 停止训练 |
| Training | GET | `/api/training/list` | 任务列表 |
| Training | GET | `/api/training/models` | 可用模型列表 |
| Logs | GET | `/api/logs/{id}` | 训练日志 |
| Logs | GET | `/api/logs/{id}/download` | 导出日志 |
| Logs | GET | `/api/logs/{id}/metrics` | 指标数据 |
| Viz | GET | `/api/viz/{id}/confusion_matrix` | 混淆矩阵 |
| Viz | GET | `/api/viz/{id}/roc_curve` | ROC曲线 |
| Viz | GET | `/api/viz/{id}/feature_importance` | 特征重要性 |
| Viz | GET | `/api/viz/{id}/learning_curve` | 学习曲线 |
| Viz | GET | `/api/viz/{id}/shap_summary` | SHAP概览 |
| Models | GET | `/api/models/list` | 模型列表 |
| Models | GET | `/api/models/{id}/detail` | 模型详情 |
| Models | GET | `/api/models/compare` | 模型对比 |
| Models | DELETE | `/api/models/{id}` | 删除模型 |
| Models | POST | `/api/models/{id}/predict` | 模型预测 |
| Experiments | GET | `/api/experiments/list` | MLflow实验列表 |
| Experiments | GET | `/api/experiments/runs` | MLflow运行列表 |
| Experiments | GET | `/api/experiments/runs/{id}` | 运行详情 |
| WebSocket | WS | `/ws/training/{id}` | 训练进度流 |
| WebSocket | WS | `/ws/logs/{id}` | 日志流 |

### 4.2 新增路由 (待实现)

| 模块 | 方法 | 路径 | 说明 | 优先级 |
|------|------|------|------|--------|
| Data | POST | `/api/data/{id}/preprocess-preview` | 预处理预览 | P1 |
| Data | GET | `/api/data/{id}/correlation` | 相关性矩阵 | P1 |
| Data | GET | `/api/data/{id}/target_distribution` | 目标变量分布 | P1 |
| Viz | GET | `/api/viz/{id}/residual_plot` | 残差图(回归) | P1 |
| Viz | GET | `/api/viz/{id}/pred_vs_actual` | 预测vs实际(回归) | P1 |
| Viz | GET | `/api/viz/{id}/shap_dependence` | SHAP依赖图 | P2 |
| Viz | GET | `/api/viz/{id}/shap_waterfall` | SHAP瀑布图 | P2 |
| Viz | GET | `/api/viz/{id}/partial_dependence` | 偏依赖图 | P2 |
| Deploy | POST | `/api/deploy/{task_id}` | 部署模型 | P1 |
| Deploy | GET | `/api/deploy/list` | 部署列表 | P1 |
| Deploy | DELETE | `/api/deploy/{id}` | 卸载部署 | P1 |
| Deploy | PATCH | `/api/deploy/{id}` | 暂停/恢复 | P2 |
| Inference | POST | `/inference/{id}/predict` | 在线预测 | P1 |
| Inference | GET | `/inference/{id}/result/{job_id}` | 查询结果 | P1 |
| Inference | POST | `/inference/{id}/batch` | 批量推理 | P2 |
| Inference | GET | `/inference/{id}/result/{job_id}/download` | 下载结果 | P2 |

---

## 五、数据库设计

### 5.1 现有表

```
datasets
├── id (UUID, PK)
├── name (String)
├── file_path (String)
├── file_size (Integer)
├── row_count (Integer)
├── column_count (Integer)
├── columns_info (JSON)
├── created_at (DateTime)
└── updated_at (DateTime)

training_tasks
├── id (UUID, PK)
├── dataset_id (FK → datasets.id)
├── model_type (String)
├── target_column (String)
├── hyperparameters (JSON)
├── test_size (Float)
├── eval_metrics (JSON)
├── status (String: PENDING/RUNNING/SUCCESS/FAILED)
├── progress (Integer: 0-100)
├── result_metrics (JSON)
├── model_path (String)
├── celery_task_id (String, nullable)
├── error_message (String, nullable)
├── created_at (DateTime)
└── updated_at (DateTime)

training_logs
├── id (Integer, PK, autoincrement)
├── task_id (FK → training_tasks.id)
├── level (String)
├── message (String)
├── extra (JSON)
└── created_at (DateTime)
```

### 5.2 新增表

```
model_deployments (新)
├── id (String, PK)               # dp-{short_uuid}
├── task_id (FK → training_tasks.id)
├── name (String, unique)
├── description (String)
├── status (String: active/paused/error)
├── max_batch_size (Integer, default=100)
├── request_count (Integer, default=0)
├── created_at (DateTime)
└── updated_at (DateTime)

inference_jobs (新)
├── id (String, PK)               # job-{short_uuid}
├── deployment_id (FK → model_deployments.id)
├── status (String: pending/processing/completed/failed)
├── input_rows (Integer)
├── input_source (String)          # "json" / "csv_upload" / "url"
├── predictions (JSON)
├── probabilities (JSON, nullable)
├── error_message (String, nullable)
├── created_at (DateTime)
└── completed_at (DateTime, nullable)

preprocess_configs (新, 可选)
├── id (Integer, PK)
├── task_id (FK → training_tasks.id)
├── config (JSON)                  # PreprocessConfig 序列化
├── encoders (BLOB, nullable)      # pickled encoders for inference
└── created_at (DateTime)
```

---

## 六、前端页面规划

### 6.1 页面清单

| 页面 | 路由 | 当前状态 | 增强内容 |
|------|------|---------|---------|
| Dashboard | `/dashboard` | ✅ 已实现 | 增加部署统计、推理调用量 |
| 数据管理 | `/data` | ✅ 已实现 | 增加相关性分析、目标分布图、预处理配置 |
| 训练配置 | `/training/config` | ✅ 已实现 | 增加回归模型、DL模型、预处理面板、X/Y配置 |
| 训练监控 | `/training/monitor` | ✅ 已实现 | 增加实时Loss曲线、ETA、epoch级粒度 |
| 结果可视化 | `/results` | ✅ 已实现 | 增加SHAP依赖图、PDP、残差图、相关性热力图 |
| 模型管理 | `/models` | ✅ 已实现 | 增加一键部署、雷达图对比 |
| **模型部署** | `/deploy` | ❌ 新增 | 部署列表、URL管理、在线测试、批量推理 |
| 设置 | `/settings` | ✅ 已实现 | 无需改动 |

### 6.2 路由表

```javascript
const routes = [
  { path: "/",                  redirect: "/dashboard" },
  { path: "/dashboard",         component: Dashboard },
  { path: "/data",              component: DataManagement },
  { path: "/training/config",   component: TrainingConfig },
  { path: "/training/monitor",  component: TrainingMonitor },
  { path: "/results",           component: Results },
  { path: "/models",            component: ModelManagement },
  { path: "/deploy",            component: ModelDeploy },        // 新增
  { path: "/settings",          component: Settings },
];
```

---

## 七、实施计划

### Phase 2A：回归支持 + 数据增强（预计3天）

| 天 | 任务 | 产出 |
|----|------|------|
| Day 1 | 回归Trainer (9种) + 任务类型自动检测 + 回归指标 | `core/trainer.py` 扩展 |
| Day 2 | 相关性分析API + 目标分布API + 预处理配置API | 3个新端点 |
| Day 3 | 回归可视化 (残差图 + 预测vs实际) + 前端集成 | `viz_service.py` + 前端页面 |

### Phase 2B：深度学习模型（预计3天）

| 天 | 任务 | 产出 |
|----|------|------|
| Day 4 | PyTorchBaseTrainer + DNN Trainer | `core/pytorch_trainer.py` |
| Day 5 | 深度学习超参数面板 + epoch实时推送 | 前端 + WebSocket增强 |
| Day 6 | TabNet Trainer + 训练曲线可视化 | train/val loss对比图 |

### Phase 2C：高级可视化（预计2天）

| 天 | 任务 | 产出 |
|----|------|------|
| Day 7 | SHAP Dependence + Waterfall + PDP API | `viz_service.py` 扩展 |
| Day 8 | 前端可视化集成 (ECharts) + 相关性热力图 | 4个新图表组件 |

### Phase 3：模型部署服务（预计3天）

| 天 | 任务 | 产出 |
|----|------|------|
| Day 9 | 部署数据库模型 + Deploy API + Model Cache | `deploy_service.py` |
| Day 10 | 推理端点 (同步/异步) + 批量推理 | `inference/` 路由 |
| Day 11 | 部署管理前端页面 + URL展示 + 在线测试 | `ModelDeploy.jsx` |

### Phase 4：打磨与测试（预计2天）

| 天 | 任务 | 产出 |
|----|------|------|
| Day 12 | Playwright E2E 测试覆盖新功能 | 测试用例 |
| Day 13 | Dashboard 统计增强 + 整体联调 + Bug修复 | 最终版本 |

---

## 八、文件变更清单

### 新增文件

```
ml_platform/app/
├── core/
│   └── pytorch_trainer.py          # PyTorch训练器基类 + DNN + TabNet
├── api/routes/
│   └── deploy.py                   # 部署管理API
├── services/
│   └── deploy_service.py           # 部署业务逻辑 + ModelCache
└── models/
    └── (database.py中新增2张表)

ml_platform_web/src/
└── pages/
    └── ModelDeploy.jsx             # 模型部署管理页面
```

### 修改文件

```
ml_platform/app/
├── core/trainer.py                 # +9种回归Trainer + 任务类型检测
├── models/database.py              # +ModelDeployment + InferenceJob表
├── models/schemas.py               # +PreprocessConfig + DeployRequest + 回归指标
├── services/training_service.py    # +回归支持 + 预处理配置透传
├── services/prediction_service.py  # +标准化/编码策略可配置
├── services/viz_service.py         # +残差图 + SHAP依赖图 + PDP
├── services/data_service.py        # +相关性分析 + 目标分布
├── api/routes/data.py              # +correlation + target_distribution端点
├── api/routes/visualization.py     # +4个新可视化端点
├── main.py                         # +deploy路由 + inference路由注册

ml_platform_web/src/
├── services/api.js                 # +deploy + inference API
├── pages/DataManagement.jsx        # +相关性分析入口 + 预处理配置
├── pages/TrainingConfig.jsx        # +回归模型 + DL模型 + 预处理面板
├── pages/TrainingMonitor.jsx       # +实时Loss曲线
├── pages/Results.jsx               # +新增图表Tab
├── pages/ModelManagement.jsx       # +一键部署按钮
├── pages/Dashboard.jsx             # +部署统计
├── components/layout/Sidebar.jsx   # +部署菜单项
└── App.jsx                         # +/deploy路由
```

---

## 九、非功能需求

### 9.1 性能
- 模型推理延迟 < 100ms (缓存命中时)
- 批量推理支持 10,000+ 行
- 前端首屏加载 < 3s
- WebSocket消息延迟 < 500ms

### 9.2 可靠性
- 训练失败自动记录错误日志，不影响其他任务
- 模型缓存OOM保护（LRU淘汰 + 最大内存限制）
- 推理服务自动健康检查

### 9.3 安全性
- 文件上传类型白名单 (.csv, .parquet, .xlsx)
- 上传大小限制 (200MB)
- 推理端点限流（可选，后续增加）
- 无身份验证（当前为内部工具定位）

### 9.4 可观测性
- 所有训练任务全程日志可追溯
- MLflow 实验追踪
- 推理调用计数
- 部署状态监控

---

## 十、技术决策记录

| 决策 | 选择 | 原因 |
|------|------|------|
| 异步训练 | ThreadPoolExecutor (非Celery) | 简化部署，单机够用，Celery预留接口 |
| 数据库 | SQLite (async) | 开发简单，单文件，生产可换PostgreSQL |
| 模型序列化 | joblib | sklearn生态标准，比pickle更高效 |
| 深度学习框架 | PyTorch | 灵活性强，TabNet等高级模型生态好 |
| 前端图表 | ECharts | 比 Chart.js 功能更强，中文社区活跃 |
| 实时通信 | WebSocket (原生) | 比Socket.io轻量，FastAPI原生支持 |
| 模型缓存 | 自实现LRU | 比Redis简单，适合单机场景 |
| 任务类型检测 | 启发式规则 | 简单直觉，覆盖99%场景 |

---

## 附录A：超参数配置对照表

### 传统ML模型

| 模型 | 参数名 | 类型 | 范围 | 默认值 | 说明 |
|------|--------|------|------|--------|------|
| **RandomForest** | n_estimators | int | 10-1000 | 100 | 树的数量 |
| | max_depth | int | 1-50/None | None | 最大深度 |
| | min_samples_split | int | 2-20 | 2 | 内部节点最少样本数 |
| | min_samples_leaf | int | 1-20 | 1 | 叶节点最少样本数 |
| **XGBoost** | n_estimators | int | 10-1000 | 100 | 提升轮数 |
| | learning_rate | float | 0.001-1.0 | 0.1 | 学习率 |
| | max_depth | int | 1-15 | 6 | 最大深度 |
| | subsample | float | 0.5-1.0 | 0.8 | 行采样比例 |
| | colsample_bytree | float | 0.5-1.0 | 0.8 | 列采样比例 |
| | reg_alpha | float | 0-10 | 0 | L1正则 |
| | reg_lambda | float | 0-10 | 1 | L2正则 |
| **LightGBM** | n_estimators | int | 10-1000 | 100 | 提升轮数 |
| | learning_rate | float | 0.001-1.0 | 0.1 | 学习率 |
| | num_leaves | int | 8-256 | 31 | 最大叶子数 |
| | max_depth | int | -1~50 | -1 | 最大深度 |
| | min_child_samples | int | 5-100 | 20 | 叶节点最少样本 |
| **LogisticRegression** | C | float | 0.001-100 | 1.0 | 正则化强度(逆) |
| | solver | enum | lbfgs/saga/... | lbfgs | 优化算法 |
| | max_iter | int | 100-10000 | 100 | 最大迭代 |
| | penalty | enum | l1/l2/elasticnet | l2 | 正则化类型 |
| **SVM** | C | float | 0.001-100 | 1.0 | 正则化参数 |
| | kernel | enum | rbf/linear/poly | rbf | 核函数 |
| | gamma | float/enum | scale/auto/0.001-10 | scale | 核系数 |
| **MLP** | hidden_layer_sizes | tuple | 自定义 | (100,) | 隐藏层结构 |
| | activation | enum | relu/tanh/logistic | relu | 激活函数 |
| | learning_rate_init | float | 1e-5~0.1 | 0.001 | 初始学习率 |
| | alpha | float | 0-0.1 | 0.0001 | L2正则 |
| | max_iter | int | 100-2000 | 200 | 最大迭代 |

### 深度学习模型

| 模型 | 参数名 | 类型 | 范围 | 默认值 | 说明 |
|------|--------|------|------|--------|------|
| **DNN/TabNet** | epochs | int | 10-1000 | 100 | 训练轮数 |
| (通用) | batch_size | int | 16/32/64/128/256 | 64 | 批大小 |
| | learning_rate | float | 1e-5~0.1 | 1e-3 | 学习率 |
| | optimizer | enum | Adam/SGD/AdamW | Adam | 优化器 |
| | weight_decay | float | 0-0.1 | 0 | 权重衰减 |
| | dropout | float | 0-0.5 | 0.1 | Dropout率 |
| | early_stopping | bool | - | true | 早停 |
| | patience | int | 3-50 | 10 | 早停耐心值 |
| **DNN** | hidden_layers | list[int] | 自定义 | [128,64,32] | 层结构 |
| | batch_norm | bool | - | true | 批归一化 |
| | activation | enum | ReLU/GELU/... | ReLU | 激活函数 |
| **TabNet** | n_d | int | 8-64 | 16 | 决策层宽度 |
| | n_a | int | 8-64 | 16 | 注意力层宽度 |
| | n_steps | int | 3-10 | 5 | 决策步数 |
| | gamma | float | 1.0-2.0 | 1.5 | 注意力松弛系数 |

---

## 附录B：评估指标对照表

### 分类指标

| 指标 | 英文 | 计算方式 | 适用场景 |
|------|------|---------|---------|
| 准确率 | accuracy | 正确数/总数 | 类别均衡 |
| F1分数 | f1 | 2×P×R/(P+R) weighted | 通用 |
| 精确率 | precision | TP/(TP+FP) weighted | 关注误报 |
| 召回率 | recall | TP/(TP+FN) weighted | 关注漏报 |
| AUC | roc_auc | ROC曲线下面积 | 二分类/多分类 |
| 对数损失 | log_loss | 交叉熵 | 概率校准 |

### 回归指标

| 指标 | 英文 | 计算方式 | 适用场景 |
|------|------|---------|---------|
| 均方误差 | mse | mean((y-ŷ)²) | 对大误差敏感 |
| 均方根误差 | rmse | √mse | 同量纲 |
| 平均绝对误差 | mae | mean(|y-ŷ|) | 鲁棒 |
| 决定系数 | r2 | 1 - SS_res/SS_tot | 拟合优度 |
| 百分比误差 | mape | mean(|y-ŷ|/|y|) | 相对误差 |

---

*文档结束。实施时按 Phase 2A → 2B → 2C → 3 → 4 顺序推进。*
