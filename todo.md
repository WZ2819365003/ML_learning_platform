# ML Training Platform — TODO

## 整体进度

- [ ] **Step 1**：后端核心搭建（第1周）
- [ ] **Step 2**：后端完善 — SHAP可解释性 + 深度学习模型 + 模型保存管理（第2周）
- [ ] **Step 3**：前端 — 数据管理 + 模型配置 + 训练监控（第3-4周）
- [ ] **Step 4**：前端 — 结果可视化 + SHAP图 + 模型管理（第3-4周）
- [ ] **Step 5**：模型推理服务 + 联调优化（后续）

---

## Step 1 详细任务

### Day 1：项目初始化
- [ ] 创建项目目录结构
- [ ] 编写 requirements.txt 并安装依赖
- [ ] 编写 docker-compose.yml，启动 Redis + PostgreSQL
- [ ] FastAPI 骨架搭建，`GET /health` 跑通
- [ ] 配置管理（config.py + .env）
- [ ] SQLAlchemy 数据库连接 + Alembic 迁移初始化

### Day 2：数据上传模块
- [ ] `POST /api/data/upload` — 文件上传 + 元数据入库
- [ ] `GET /api/data/{id}/preview` — 数据预览（前100行 + 基础统计）
- [ ] `GET /api/data/list` — 数据集列表（分页）
- [ ] `DELETE /api/data/{id}` — 删除数据集
- [ ] Swagger 手动测试验证

### Day 3：训练任务框架
- [ ] Celery + Redis 任务队列搭建
- [ ] BaseTrainer 抽象类实现
- [ ] RandomForest 训练器实现
- [ ] `POST /api/training/start` — 启动训练
- [ ] `GET /api/training/{id}/status` — 查询状态
- [ ] `POST /api/training/{id}/stop` — 终止训练
- [ ] `GET /api/training/list` — 任务列表

### Day 4：多模型支持
- [ ] XGBoost 训练器
- [ ] LightGBM 训练器
- [ ] LogisticRegression 训练器
- [ ] SVM 训练器
- [ ] MLP 训练器
- [ ] 模型工厂（根据 model_type 自动选择训练器）
- [ ] 全模型 API 联调测试

### Day 5：WebSocket 实时推送
- [ ] WebSocket endpoint `/ws/training/{task_id}`
- [ ] Redis Pub/Sub 桥梁（Celery Worker → Redis → WebSocket）
- [ ] 训练过程中按 epoch/fold 推送指标
- [ ] 训练状态变更推送（开始/完成/失败）
- [ ] wscat 测试验证

### Day 6：训练日志系统
- [ ] TrainingLogger 类实现（文本日志 + 指标日志）
- [ ] 按 task_id 生成独立日志文件
- [ ] 结构化指标日志（JSON 格式）
- [ ] `GET /api/logs/{task_id}` — 查询日志
- [ ] `GET /api/logs/{task_id}/download` — 导出日志（txt/json）
- [ ] `GET /api/logs/{task_id}/metrics` — 查询指标数据
- [ ] `WebSocket /ws/logs/{task_id}` — 实时日志流

### Day 7：MLflow 集成 + 联调
- [ ] MLflow Tracking Server 本地部署
- [ ] 训练任务自动创建 MLflow Run
- [ ] 记录超参数、每轮指标、最终指标
- [ ] 模型文件记录到 MLflow
- [ ] 整体 API 联调
- [ ] 编写基础单元测试

---

## 备注

- 暂不涉及模型推理，放到 Step 5
- 日志系统是重点，训练过程全程可追溯、可导出
- 第一周以 sklearn 传统模型为主，PyTorch 深度学习模型放到第二周
- 开发环境可用 SQLite 替代 PostgreSQL 降低门槛
