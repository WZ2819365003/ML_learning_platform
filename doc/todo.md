# ML Training Platform — TODO

## Phase 1: 后端核心 (第1周) — 已完成

- [x] Day 1: 项目初始化 — 目录结构、依赖、配置、数据库、FastAPI 骨架
- [x] Day 2: 数据上传模块 — upload/preview/list/delete 4个API
- [x] Day 3: 训练任务框架 — asyncio + ThreadPool 异步训练、BaseTrainer、RandomForest
- [x] Day 4: 多模型支持 — 6种模型训练器 (RF/XGB/LGBM/LR/SVM/MLP) + 工厂模式
- [x] Day 5: WebSocket 实时推送 — EventBus 内存 pub/sub + /ws/training/{id} + /ws/logs/{id}
- [x] Day 6: 训练日志系统 — per-task 文件日志 + JSON metrics + 查询/导出 API
- [x] Day 7: MLflow 集成 — SQLite backend、实验追踪、参数/指标/模型自动记录

### 已实现的 API

| 模块 | 方法 | 路径 | 说明 |
|------|------|------|------|
| Health | GET | /health | 健康检查 |
| Data | POST | /api/data/upload | 上传 CSV/Parquet/Excel |
| Data | GET | /api/data/{id}/preview | 数据预览 + 统计 |
| Data | GET | /api/data/list | 数据集列表(分页) |
| Data | DELETE | /api/data/{id} | 删除数据集 |
| Training | POST | /api/training/start | 启动训练任务 |
| Training | GET | /api/training/{id}/status | 查询训练状态 |
| Training | POST | /api/training/{id}/stop | 终止训练 |
| Training | GET | /api/training/list | 训练任务列表(分页/筛选) |
| Training | GET | /api/training/models | 可用模型列表 |
| Logs | GET | /api/logs/{id} | 训练日志(分页/级别筛选) |
| Logs | GET | /api/logs/{id}/download | 导出日志(txt/json) |
| Logs | GET | /api/logs/{id}/metrics | 结构化指标数据 |
| Experiments | GET | /api/experiments/list | MLflow 实验列表 |
| Experiments | GET | /api/experiments/runs | MLflow run 列表 |
| Experiments | GET | /api/experiments/runs/{id} | MLflow run 详情 |
| WebSocket | WS | /ws/training/{id} | 实时训练指标推送 |
| WebSocket | WS | /ws/logs/{id} | 实时日志流 |

---

## Phase 2: 后端完善 (第2周) — 待开始

- [ ] SHAP 可解释性 API (特征重要性、SHAP summary plot 数据)
- [ ] 模型保存/加载管理 (版本管理、元数据)
- [ ] 深度学习模型支持 (PyTorch CNN/RNN)
- [ ] 多实验对比 API

## Phase 3: 前端开发 (第3-4周) — 待开始

- [ ] React + Vite 项目搭建
- [ ] 数据上传页 (表格预览、统计可视化)
- [ ] 模型配置页 (模型选择、超参数表单)
- [ ] 训练监控页 (实时 loss 曲线、进度条)
- [ ] 结果页 (混淆矩阵、ROC/AUC、SHAP 图)
- [ ] 模型管理页 (列表、版本对比)

## Phase 4: 模型推理 (第5周) — 待开始

- [ ] 推理服务 API (上传数据 → 加载模型 → 返回预测)
- [ ] 批量推理
- [ ] 推理结果可视化

---

## 技术栈

| 层 | 技术 | 状态 |
|---|---|---|
| 后端框架 | FastAPI + asyncio | 已集成 |
| 数据库 | SQLite (aiosqlite) | 已集成 |
| 训练引擎 | scikit-learn + XGBoost + LightGBM | 已集成 |
| 实验追踪 | MLflow (SQLite backend) | 已集成 |
| 实时通信 | WebSocket + EventBus | 已集成 |
| 日志系统 | loguru + per-task 文件日志 | 已集成 |
| 任务队列 | asyncio (Celery 预留) | 已集成 |
| 前端 | React + ECharts (计划) | 待开始 |
