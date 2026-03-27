# Step 1：后端核心搭建（第1周）

## 目标

搭建完整的后端骨架，实现数据上传、多模型异步训练、训练指标实时推送、训练日志系统。所有 API 可通过 Swagger 文档独立验证，不依赖前端。

---

## 1. 项目初始化与骨架搭建

### 1.1 项目结构

```
ml_platform/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI 入口
│   ├── config.py                # 配置管理（数据库、Redis、路径等）
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── data.py          # 数据上传/预览相关接口
│   │   │   ├── training.py      # 训练任务相关接口
│   │   │   ├── experiment.py    # 实验记录查询接口
│   │   │   └── logs.py          # 日志查询/导出接口
│   │   └── websocket.py         # WebSocket 实时推送
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py          # SQLAlchemy 模型定义
│   │   └── schemas.py           # Pydantic 请求/响应模型
│   ├── services/
│   │   ├── __init__.py
│   │   ├── data_service.py      # 数据处理业务逻辑
│   │   ├── training_service.py  # 训练任务编排
│   │   └── log_service.py       # 日志收集与导出
│   ├── tasks/
│   │   ├── __init__.py
│   │   └── train_task.py        # Celery 异步训练任务
│   ├── core/
│   │   ├── __init__.py
│   │   ├── trainer.py           # 统一训练器接口（适配多模型）
│   │   ├── metrics.py           # 指标计算与记录
│   │   └── logger.py            # 训练日志管理器
│   └── utils/
│       ├── __init__.py
│       └── file_utils.py        # 文件操作工具
├── storage/
│   ├── uploads/                 # 上传的数据文件
│   ├── models/                  # 保存的模型文件
│   └── logs/                    # 训练日志文件
├── tests/
│   ├── __init__.py
│   ├── test_data_api.py
│   ├── test_training_api.py
│   └── test_log_api.py
├── requirements.txt
├── docker-compose.yml           # Redis + PostgreSQL
├── .env                         # 环境变量
└── README.md
```

### 1.2 技术依赖

```txt
# requirements.txt
fastapi==0.115.*
uvicorn[standard]==0.34.*
sqlalchemy==2.0.*
alembic==1.14.*            # 数据库迁移
psycopg2-binary==2.9.*     # PostgreSQL 驱动（轻量版可用 aiosqlite）
celery==5.4.*
redis==5.2.*
python-multipart==0.0.*    # 文件上传
pandas==2.2.*
scikit-learn==1.6.*
pycaret==3.4.*
mlflow==2.19.*
shap==0.46.*
websockets==14.*
python-dotenv==1.0.*
loguru==0.7.*              # 日志库
```

### 1.3 Docker Compose（开发环境中间件）

```yaml
# docker-compose.yml
version: "3.9"
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: ml_platform
      POSTGRES_USER: ml_user
      POSTGRES_PASSWORD: ml_password
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

---

## 2. 数据上传与管理模块

### 2.1 功能点

- **POST /api/data/upload** — 上传 CSV/Parquet 文件
  - 校验文件格式、大小限制（默认 200MB）
  - 存储到 `storage/uploads/`，文件名加 UUID 防冲突
  - 解析列名、数据类型、行数，写入数据库元数据表
  - 返回 dataset_id

- **GET /api/data/{dataset_id}/preview** — 数据预览
  - 返回前 100 行数据
  - 返回基础统计信息（均值、缺失率、分布类型）

- **GET /api/data/list** — 数据集列表
  - 分页查询已上传的数据集

- **DELETE /api/data/{dataset_id}** — 删除数据集

### 2.2 数据库表设计

```sql
CREATE TABLE datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,          -- 原始文件名
    file_path VARCHAR(500) NOT NULL,     -- 存储路径
    file_size BIGINT,                    -- 文件大小(bytes)
    row_count INTEGER,                   -- 行数
    column_count INTEGER,                -- 列数
    columns_info JSONB,                  -- 列名、类型、缺失率等
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

---

## 3. 训练任务模块

### 3.1 功能点

- **POST /api/training/start** — 启动训练任务
  - 请求体包含：dataset_id、model_type、hyperparameters、target_column、eval_metrics
  - 创建 Celery 异步任务
  - 返回 task_id

- **GET /api/training/{task_id}/status** — 查询训练状态
  - 状态：PENDING / RUNNING / SUCCESS / FAILED
  - 返回当前 epoch、进度百分比

- **POST /api/training/{task_id}/stop** — 终止训练
  - 通过 Celery revoke 终止任务

- **GET /api/training/list** — 训练任务列表
  - 按状态筛选、分页

### 3.2 支持的模型类型（第一周先实现）

| 模型类型 | 框架 | 关键超参数 |
|---------|------|-----------|
| RandomForest | sklearn | n_estimators, max_depth, min_samples_split |
| XGBoost | xgboost/sklearn | n_estimators, learning_rate, max_depth |
| LightGBM | lightgbm | n_estimators, learning_rate, num_leaves |
| LogisticRegression | sklearn | C, penalty, max_iter |
| SVM | sklearn | C, kernel, gamma |
| MLP (多层感知机) | sklearn | hidden_layer_sizes, activation, learning_rate |

> 深度学习模型（PyTorch CNN/RNN）放到第二周。

### 3.3 统一训练器接口设计

```python
# app/core/trainer.py
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """所有训练器的基类"""

    @abstractmethod
    def configure(self, hyperparameters: dict):
        """配置超参数"""
        pass

    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val, callback=None):
        """
        执行训练
        callback: 每个epoch/step调用，用于推送实时指标
        callback(epoch, metrics_dict)
        """
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test) -> dict:
        """评估模型，返回指标字典"""
        pass

    @abstractmethod
    def save(self, path: str):
        """保存模型"""
        pass

    @abstractmethod
    def load(self, path: str):
        """加载模型"""
        pass
```

### 3.4 训练请求体示例

```json
{
    "dataset_id": "uuid-xxxx",
    "target_column": "label",
    "model_type": "random_forest",
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5
    },
    "test_size": 0.2,
    "eval_metrics": ["accuracy", "f1", "roc_auc"],
    "cross_validation": {
        "enabled": true,
        "folds": 5
    }
}
```

### 3.5 数据库表设计

```sql
CREATE TABLE training_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID REFERENCES datasets(id),
    model_type VARCHAR(50) NOT NULL,
    hyperparameters JSONB,
    target_column VARCHAR(100),
    test_size FLOAT DEFAULT 0.2,
    eval_metrics JSONB,                    -- ["accuracy", "f1"]
    status VARCHAR(20) DEFAULT 'PENDING',  -- PENDING/RUNNING/SUCCESS/FAILED
    celery_task_id VARCHAR(255),
    progress FLOAT DEFAULT 0,              -- 0-100
    result_metrics JSONB,                  -- 训练完成后的最终指标
    model_path VARCHAR(500),               -- 模型保存路径
    error_message TEXT,                    -- 失败时的错误信息
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 4. WebSocket 实时推送

### 4.1 设计

```
前端 ──WebSocket──> /ws/training/{task_id}
                          │
                    后端 WebSocket Manager
                          │
              Celery Worker ──Redis PubSub──> 推送到连接的客户端
```

### 4.2 推送数据格式

```json
{
    "type": "metrics",
    "task_id": "uuid-xxxx",
    "epoch": 15,
    "total_epochs": 100,
    "progress": 15.0,
    "metrics": {
        "train_loss": 0.342,
        "val_loss": 0.411,
        "train_accuracy": 0.891,
        "val_accuracy": 0.856
    },
    "timestamp": "2026-03-22T10:30:15Z"
}
```

状态变更推送：

```json
{
    "type": "status",
    "task_id": "uuid-xxxx",
    "status": "SUCCESS",
    "message": "训练完成",
    "result_metrics": {
        "accuracy": 0.923,
        "f1": 0.917,
        "roc_auc": 0.961
    }
}
```

### 4.3 实现要点

- 使用 Redis Pub/Sub 作为 Celery Worker 和 WebSocket 之间的桥梁
- Celery Worker 训练过程中每个 epoch 向 Redis channel 发布指标
- FastAPI WebSocket endpoint 订阅对应 channel，转发给前端
- 支持多客户端同时监听同一个训练任务

---

## 5. 训练日志系统

### 5.1 日志分层

| 层级 | 内容 | 存储方式 |
|------|------|---------|
| **系统日志** | API 请求日志、错误追踪、服务状态 | loguru → 文件轮转 |
| **训练日志** | 每个训练任务的详细过程日志（数据加载、预处理、每轮指标、异常信息） | 按 task_id 独立文件 |
| **指标日志** | 结构化的训练指标数据（loss、accuracy 等，按 epoch 记录） | 数据库 + JSON 文件 |

### 5.2 API

- **GET /api/logs/{task_id}** — 获取训练日志（支持分页、级别筛选）
- **GET /api/logs/{task_id}/download** — 导出训练日志（支持 txt/json 格式）
- **GET /api/logs/{task_id}/metrics** — 获取结构化指标数据（用于前端绘图）
- **WebSocket /ws/logs/{task_id}** — 实时日志流（训练中实时查看）

### 5.3 训练日志格式

每个训练任务生成独立日志文件 `storage/logs/{task_id}.log`：

```
2026-03-22 10:30:00 | INFO  | 训练任务启动 | model=random_forest | dataset=iris.csv
2026-03-22 10:30:01 | INFO  | 数据加载完成 | rows=1000 | cols=10 | target=label
2026-03-22 10:30:01 | INFO  | 数据集划分 | train=800 | test=200
2026-03-22 10:30:02 | INFO  | 模型初始化 | params={"n_estimators": 100, "max_depth": 10}
2026-03-22 10:30:03 | INFO  | [Epoch 1/100] train_loss=0.692 val_loss=0.701 accuracy=0.512
2026-03-22 10:30:04 | INFO  | [Epoch 2/100] train_loss=0.543 val_loss=0.589 accuracy=0.734
...
2026-03-22 10:31:45 | INFO  | 训练完成 | best_accuracy=0.923 | 耗时=105s
2026-03-22 10:31:46 | INFO  | 模型已保存 | path=storage/models/uuid-xxxx.joblib
```

### 5.4 指标日志结构（JSON，用于前端绘图）

```json
// storage/logs/{task_id}_metrics.json
{
    "task_id": "uuid-xxxx",
    "model_type": "random_forest",
    "epochs": [
        {
            "epoch": 1,
            "train_loss": 0.692,
            "val_loss": 0.701,
            "train_accuracy": 0.512,
            "val_accuracy": 0.498,
            "timestamp": "2026-03-22T10:30:03Z"
        }
    ]
}
```

### 5.5 日志管理器设计

```python
# app/core/logger.py
class TrainingLogger:
    """每个训练任务一个实例"""

    def __init__(self, task_id: str):
        self.task_id = task_id
        # 文本日志 → storage/logs/{task_id}.log
        # 指标日志 → storage/logs/{task_id}_metrics.json
        # 同时推送到 Redis PubSub

    def log(self, level: str, message: str, **extra):
        """写入文本日志 + 推送到 WebSocket"""
        pass

    def log_metrics(self, epoch: int, metrics: dict):
        """记录指标 + 追加到 JSON + 推送实时数据"""
        pass

    def export(self, format: str = "txt") -> str:
        """导出日志文件，返回文件路径"""
        pass
```

---

## 6. MLflow 集成

### 6.1 集成方式

- 每次训练自动创建一个 MLflow Run
- 记录：超参数、每轮指标、最终指标、模型文件
- MLflow Tracking Server 本地启动（`mlflow server --port 5001`）

### 6.2 集成位置

在 `train_task.py` 中，训练开始时：

```python
import mlflow

with mlflow.start_run(run_name=f"{model_type}_{task_id[:8]}"):
    mlflow.log_params(hyperparameters)

    for epoch in range(epochs):
        metrics = trainer.train_step(...)
        mlflow.log_metrics(metrics, step=epoch)

    mlflow.sklearn.log_model(model, "model")
```

---

## 7. 第一周每日计划

| 日期 | 任务 | 验收标准 |
|------|------|---------|
| Day 1 | 项目初始化、依赖安装、Docker Compose 启动 Redis+PG、FastAPI 骨架跑通 | `GET /health` 返回 200 |
| Day 2 | 数据上传模块完成（upload、preview、list、delete） | Swagger 中上传 CSV 并能预览 |
| Day 3 | Celery 任务框架搭建 + 第一个模型（RandomForest）训练跑通 | POST 启动训练，GET 查看状态=SUCCESS |
| Day 4 | 补充其余 5 个模型 + 统一训练器接口 | 6 种模型都能通过 API 启动训练 |
| Day 5 | WebSocket 实时推送 + Redis PubSub 桥梁 | 用 wscat 连接 WebSocket 能收到实时指标 |
| Day 6 | 训练日志系统（文件日志 + 指标日志 + 导出 API） | 下载训练日志文件，JSON 指标数据可查询 |
| Day 7 | MLflow 集成 + 整体联调 + 基础测试 | MLflow UI 中可看到实验记录 |

---

## 8. 注意事项

1. **第一周不涉及模型推理**，推理服务放到后续阶段
2. **sklearn 模型没有 epoch 概念**，用交叉验证的每一折作为"步骤"来推送进度
3. **文件存储先用本地**，后续可以切换到 MinIO/S3
4. **数据库轻量版可用 SQLite**，降低开发环境搭建成本
5. **所有 API 先写好 Pydantic Schema**，FastAPI 自动生成文档，方便后续前端对接
