# ML Learning Platform - Docker 交付包

> **v3.2.3 完整交付包** — 镜像 + MySQL 数据 dump + MinIO 数据 seed，接收方只需 `docker load` + `docker-compose up -d`。

---

## 数据清单

| 类别 | 内容 | 来源 |
|---|---|---|
| **MySQL 元数据** | 18 张表，全部历史记录 | `mysql_init/01_data.sql` (994 KB) |
| ├─ 数据集 | 3 条：predictive_maintenance / diabetes / ETTh1 | `datasets` |
| ├─ ML 训练任务 | 3 条（random_forest 系列） | `training_tasks` |
| ├─ DL 训练任务 | 6 条（transformer / lstm / mlp_dl / cnn1d） | `dl_training_tasks` |
| ├─ V3 平台任务 | **23 条**（17 SUCCESS / 6 FAILED） | `platform_tasks` |
| ├─ V3 实验 Run | **7 条**（mlp_dl / lstm / cnn1d / random_forest / lightgbm 等） | `experiment_runs` |
| ├─ V3 建模任务 | 1 条："V3.1.1 演示 - 4ML+3DL 混合基线" | `modeling_tasks` |
| ├─ V3 实验组 | 1 条："baseline-4ml-3dl" | `platform_experiments` |
| ├─ 训练方案 | 1 条 | `training_plans` |
| ├─ 训练日志 | 45 + 101 = 146 条 | `training_logs` / `dl_training_logs` |
| ├─ DL Epoch 记录 | 83 条 | `dl_training_epochs` |
| └─ 模型标签库 | 23 条 | `model_tag_library` |
| **MinIO 对象存储** | 36 MB SHAP/logs/models artifacts | `minio_seed/ml-platform/` |
| **后端镜像** | FastAPI + ML 库 + 上传文件 + 模型文件 + MLflow 数据 | `ml_platform_backend.tar` (480 MB) |
| **前端镜像** | React 生产构建 | `ml_platform_frontend.tar` (51 MB) |

**交付包总大小：~576 MB**

---

## 接收方使用方法

### 1. 加载镜像

```bash
cd docker
docker load -i ml_platform_backend.tar
docker load -i ml_platform_frontend.tar
```

> Redis / MinIO / MySQL / Nginx 镜像 docker-compose 会自动从公开仓库拉取。

### 2. 启动服务

```bash
docker-compose up -d
```

启动顺序（自动）：
1. MySQL 启动 → 自动加载 `mysql_init/01_data.sql`（首次启动 only）
2. Redis / MinIO 启动
3. **minio_init** 一次性容器：把 `minio_seed/` 镜像到 MinIO `ml-platform` bucket，完成后退出
4. Backend 启动（依赖 MySQL/Redis/MinIO 健康）
5. Frontend / Nginx 启动

首次启动约 30-60 秒（MySQL 导入数据 + MinIO seed）。

### 3. 访问应用

| 入口 | URL | 说明 |
|---|---|---|
| **主入口** | http://localhost | Nginx 反向代理（推荐） |
| 前端 | http://localhost:3000 | React 应用直连 |
| 后端 API | http://localhost:8000 | FastAPI |
| API 文档 | http://localhost:8000/docs | Swagger UI |
| MinIO 控制台 | http://localhost:9001 | 用户 `mlplatform` / 密码 `mlplatform123` |
| MySQL | localhost:3307 | 用户 `root` / 密码 `123456` / 库 `ml_platform` |

打开 http://localhost 即可看到所有历史数据 —— V3 建模工作台、ML/DL 训练记录、SHAP 解释一应俱全。

---

## 服务管理

```bash
# 查看状态
docker-compose ps

# 查看日志
docker-compose logs -f backend
docker-compose logs -f mysql

# 停止（保留数据）
docker-compose down

# 完全清除（删除 MySQL/Redis/MinIO 数据 volume，下次启动会重新 seed）
docker-compose down -v
```

> **重要**：`docker-compose down -v` 之后再 `up -d`，MySQL 会从 `mysql_init/01_data.sql` 重新初始化，
> MinIO 会从 `minio_seed/` 重新 seed，恢复到出厂状态。

---

## 架构说明

```
┌──────────────┐     ┌──────────────┐
│   Browser    │ ──> │   Nginx :80  │
└──────────────┘     └──────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
       Frontend:3000   Backend:8000    /ws/* (WebSocket)
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   MySQL:3306          Redis:6379          MinIO:9000
   (主元数据)          (cache/queue)      (artifacts)
```

### 数据存储分布
- **MySQL**：所有元数据（datasets / tasks / runs / V3 platform / 训练历史）
- **MinIO**：SHAP 解释 JSON / 训练 logs / 部分 model artifacts
- **后端镜像内本地文件**：
  - `/app/storage/uploads/` — 3 个 CSV 数据集（3 MB）
  - `/app/storage/models/` — 6 个训练好的模型文件 .joblib/.pt（3.5 MB）
  - `/app/storage/mlflow.db` — MLflow 元数据（220 KB）
  - `/app/mlruns/` — MLflow artifacts（37 MB）

---

## 常见问题

### Q: 端口被占用？
```bash
lsof -i :80     # 看占用
kill -9 <PID>   # 释放
```
或修改 `docker-compose.yml` 中的端口映射，例如 `"8080:80"`。

### Q: 接收方机器上有同名 volume 冲突？
若接收方之前用过本项目，可能存在残留 volume。删除：
```bash
docker volume rm docker_ml_mysql_data docker_ml_minio_data docker_ml_redis_data
```
然后重新 `docker-compose up -d`。

### Q: 想重置数据回到出厂状态？
```bash
docker-compose down -v && docker-compose up -d
```

### Q: 如何备份当前运行中的数据？
```bash
docker exec ml_platform_mysql mysqldump -uroot -p123456 \
  --databases ml_platform > backup_$(date +%Y%m%d).sql
docker exec ml_platform_minio mc mirror local/ml-platform/ ./minio_backup/
```

---

## 版本信息

- **项目版本**: v3.2.3
- **Python**: 3.11
- **Node.js**: 20
- **MySQL**: 8.0
- **Redis**: 7-alpine
- **MinIO**: latest
- **Nginx**: alpine
- **数据快照时间**: 2026-04-25
