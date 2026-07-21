# ML Training Platform — Roadmap / TODO

> 这份 TODO 以当前 V3 平台为基线，不再保留早期"前端待开始 / SHAP 待开始"式计划。已落地能力见 [功能说明](./功能说明.md) 和 [系统架构](./系统架构.md)。

## 当前状态

- [x] FastAPI + React + V3 工作台主链路已具备：数据集、训练方案、建模任务、实验批次、Run、Inspector、SHAP、模型资产与部署。
- [x] 训练策略已覆盖 `baseline`、`grid_search`、`bayesian_search`。
- [x] 模型族已覆盖传统 ML、Tabular DL、Chronos/TS 兼容通道。
- [x] Docker stack、MySQL、Redis、MinIO、WebSocket、MLflow 运行链路已具备。

## P0 发布门禁

- [ ] Docker 镜像版本、`/health.version`、前端 `package.json` 版本、git commit 在测试报告中一致可追溯。
- [ ] `playwright_test/test/08-v3-end-to-end.spec.js` 必须严格通过：`TrainingPlan -> ModelingTask -> ExperimentBatch -> ExperimentRun -> Inspector -> SHAP`。
- [ ] 后端核心测试在发布前通过：`cd ml_platform && python -m pytest tests/`。
- [ ] 全量 Playwright 巡检报告归档：`artifacts/results.json` + HTML report + 后端关键日志。
- [ ] 发布前确认种子数据集可用，至少包含一个分类数据集和一个回归数据集。

## P1 模型扩展契约

- [ ] 抽象并文档化模型注册契约：`token`、`family`、`task_type`、默认参数、参数 schema、训练入口、预测入口、可解释性支持级别。
- [ ] 为每个新增模型添加 registry contract test，确保 V3 modal、后端调度、模型资产页使用同一份 token。
- [ ] 新模型接入时强制提供最小 E2E：创建任务、跑一次 baseline、产生 metrics、可进入 Inspector。
- [ ] 把模型能力标签标准化：是否支持 `predict_proba`、是否支持 SHAP tree path、是否支持部署推理、是否支持批量预测。

## P2 策略扩展

- [ ] 将调参策略接口固定为 `validate -> plan_trials -> dispatch -> aggregate`，新增策略只实现策略层，不改 ModelingTask 主流程。
- [ ] 为 `grid_search`、`bayesian_search` 增加边界测试：空 search_space、非法参数、max_trials 截断、失败 run 聚合。
- [ ] 增加策略级预算控制：全局最大 run 数、单模型最大 trial、超时时间、并发度。
- [ ] 增加策略对比报告导出：不同策略的 best run、均值/方差、耗时、失败率。

## P3 可靠性与可观测性

- [x] 将 `InProcessScheduler` 的生产替代方案打通：Celery worker、Redis broker、任务重试、取消、恢复。
      基础设施已就绪（worker 容器、queues 单一来源、claim/CAS 写回、reconcile + stalled recovery）。
      **但生产尚未开启** —— 阻断项是 executor 的 attempt fencing，见 TECH_DEBT TD-2。
- [x] 为数据库 schema 引入正式 migration 流程，避免运行库和代码模型漂移。
      Alembic 已冻结 0001 基线并推进到 0004；生产走 Alembic，非生产走幂等启动迁移。
- [x] WebSocket 跨实例改为 Redis pub/sub，支持多 backend 实例部署。
      `EVENT_BUS_MODE=redis` 已实现；生产 compose 默认仍是 `memory`，多实例前需切换。
- [ ] 对训练失败补结构化错误分类：数据问题、模型参数问题、依赖问题、资源问题。
- [ ] 对长任务补资源指标：耗时、CPU/内存、训练数据规模、模型文件大小。

## P4 产品与运维

- [ ] 模型卡片补齐：训练数据、指标、参数、解释性、部署状态、风险说明。
- [ ] 增加模型漂移 / 数据漂移监控入口，部署模型能看到线上输入分布变化。
- [ ] 增加权限、审计、项目空间隔离，避免多人共享环境互相覆盖任务。
- [ ] 生产部署补 HTTPS、密钥管理、备份恢复、对象存储生命周期策略。
- [ ] 增加版本化发布检查清单，明确 build、test、tag、push、rollback 步骤。
