# 模型资产统一设计

**日期**: 2026-04-10
**状态**: Approved
**目标**: 在保持现有 ML / DL 接口兼容的前提下，修复 DL 日志实时回传缺失，统一模型管理与部署体验，优先保证功能可用和改动风险可控。

---

## 背景

当前项目的机器学习和深度学习链路已经分别可用，但在三个关键点上出现明显割裂：

1. `DLMonitor` 只有 epoch 指标实时推送，没有训练日志实时推送，日志区只能靠 REST 一次性读取。
2. 模型管理在前端被拆成 ML / DL / 通用三套交互，字段结构和可操作项不一致，用户需要理解内部实现差异。
3. 模型部署在后端和前端分别走两套接口与数据模型：
   - ML: `TrainingTask` + `ModelDeployment` + `/api/deploy/*` + `/inference/*`
   - DL: `DLTrainingTask` + `DLModelDeployment` + `/api/dl/deployments/*`

这导致“功能有了，但产品体验不连续”，也让后续扩展统一能力变得困难。

---

## 设计原则

1. **接口兼容优先**
   现有 `/api/deploy/*`、`/api/dl/deployments/*`、`/api/models/*`、`/api/dl/*` 保持可用，不做破坏式替换。

2. **统一视图优先于统一存储**
   本次不强制把 ML / DL 的数据库表合并成一张表，而是在服务层新增统一“模型资产视图”和“部署视图”，让前端先统一起来。

3. **最小改动修复实时性**
   DL 训练链路沿用现有 `TrainingLogger + EventBus`，只补齐日志推送与状态落库，不重写训练框架。

4. **用户体验优先**
   前端模型管理页和部署页统一成一套主交互，ML / DL 差异只保留在能力细节上，不再暴露为两套完全不同的页面逻辑。

---

## 问题根因

### 1. DL 日志未同步回传到前端

- `dl_service.py` 当前只向 `dl:{task_id}` 频道发布 epoch 事件。
- `DLMonitor.jsx` 只连接 `/api/dl/ws/{task_id}`，并只处理 `epoch` 和 `done` 消息。
- 日志面板只调用 `GET /api/dl/{task_id}/logs` 读取已有日志，没有订阅 `/ws/logs/{task_id}`。
- `TrainingLogger.log()` 已经会向 `logs:{task_id}` 广播，但 DL 页面没有接入该通道。

**结论**: 日志不是“没写入”，而是“没有建立实时订阅链路”。

### 2. 模型管理功能混乱

- ML 模型管理支持详情、对比、在线预测、部署。
- DL 模型管理只支持备注/标签和删除，能力入口分散在别的页面。
- `ModelManagement.jsx` 以三个 Tab 形式暴露内部实现，不符合用户按“模型资产”统一管理的直觉。

**结论**: 问题不是单点 bug，而是视图层没有统一领域模型。

### 3. DL 部署与 ML 部署不一致

- ML 部署返回 `deployment_id/endpoints`，支持结果查询 URL。
- DL 部署直接返回 ORM 风格字段，预测接口与返回结构也不同。
- 部署页被拆成 ML / DL 两块，URL 展示、测试入口、字段名都不一致。

**结论**: 后端契约未标准化，导致前端必须分叉实现。

---

## 目标方案

### A. 统一训练日志实时链路

保留现有两类 WebSocket：

- `/api/dl/ws/{task_id}`: DL epoch / done 事件
- `/ws/logs/{task_id}`: 通用文本日志事件

新增和调整：

1. DL 训练线程继续使用 `TrainingLogger.log()` 写日志。
2. `epoch_callback` 除了发指标，还要更新 `current_epoch/progress`，确保 REST 状态与实时图一致。
3. `DLMonitor` 首次进入时：
   - 先拉一次 REST 状态
   - 先拉一次历史日志
   - 再同时建立两个 WebSocket 连接：指标流 + 日志流
4. 日志消息到达前端后按时间顺序 append，不覆盖已有日志。

### B. 新增统一模型资产视图

新增后端统一接口，例如：

- `GET /api/models/assets`
- `GET /api/models/assets/{asset_id}`

统一返回字段包含：

- `asset_id`
- `runtime_type`: `ml | dl`
- `task_id`
- `name`
- `dataset_id`
- `dataset_name`
- `model_type`
- `task_type`
- `status`
- `created_at`
- `finished_at`
- `notes`
- `tags`
- `model_path`
- `deployable`
- `predictable`
- `metrics_summary`
- `deployment_count`

其中：

- `asset_id` 采用带前缀格式，避免 ML / DL ID 冲突，例如 `ml:<task_id>`、`dl:<task_id>`
- `metrics_summary` 只返回前端列表页需要的摘要，详情页再按运行时拉取详细数据

现有接口继续保留：

- ML 详情仍用 `/api/models/{task_id}/detail`
- DL 详情仍用 `/api/dl/{task_id}/status`

统一资产接口只做列表与标准化聚合，不抢占原有领域接口职责。

### C. 新增统一部署视图

新增后端统一部署接口，例如：

- `GET /api/deploy/assets`

标准化返回字段：

- `deployment_id`
- `runtime_type`
- `source_task_id`
- `source_asset_id`
- `name`
- `description`
- `status`
- `request_count`
- `created_at`
- `predict_url`
- `result_url`
- `supports_result_polling`

标准化策略：

- ML 部署从 `ModelDeployment` 映射，保留 `predict_url` 和 `result_url`
- DL 部署从 `DLModelDeployment` 映射，`result_url` 为空，`supports_result_polling=false`

这样前端部署页只需要一套表格和一套详情面板。

### D. 统一前端交互

#### 模型管理页

改为单列表视图，核心能力统一：

- 按运行时筛选：全部 / ML / DL
- 统一查看详情
- 统一备注/标签
- 统一删除
- 统一部署入口

按运行时差异化能力：

- ML 支持模型对比与在线预测
- DL 展示训练曲线摘要与任务类型指标摘要

#### 模型部署页

改为单列表：

- 展示运行时类型、部署名称、状态、调用次数、创建时间
- 统一详情侧栏显示 URL 和在线测试
- 根据 `supports_result_polling` 决定是否展示结果查询 URL

---

## 兼容策略

### 保留不变

- 现有 ML / DL 创建部署接口
- 现有 ML / DL 删除、暂停、预测接口
- 现有 ML / DL 详情接口

### 新增而不替换

- 新的统一模型资产查询接口
- 新的统一部署视图接口

### 前端迁移方式

- 页面默认使用统一接口渲染
- 原有分叉 API 保留给详情动作和能力动作
- 不删除旧 API 封装，只减少页面直接依赖它们的场景

---

## 影响文件

后端：

- `ml_platform/app/services/dl_service.py`
- `ml_platform/app/core/logger.py`
- `ml_platform/app/api/routes/dl.py`
- `ml_platform/app/api/routes/model_mgmt.py`
- `ml_platform/app/api/routes/deploy.py`
- `ml_platform/app/models/schemas.py`
- 可能新增统一模型资产/部署视图的 service 函数

前端：

- `ml_platform_web/src/services/api.js`
- `ml_platform_web/src/pages/DLMonitor.jsx`
- `ml_platform_web/src/pages/ModelManagement.jsx`
- `ml_platform_web/src/pages/ModelDeploy.jsx`

测试：

- `ml_platform/tests/` 下新增后端接口与服务测试
- `tests/` 下补前端或端到端关键流程测试

---

## 验证标准

1. DL 训练进行中时，监控页日志区无需刷新即可持续滚动新增日志。
2. 模型管理页能同时看到 ML 和 DL 模型，并可按运行时筛选。
3. 统一模型管理页能对 ML / DL 都执行删除、备注/标签、部署入口操作。
4. 模型部署页使用统一列表展示 ML / DL 部署，并能正确在线测试。
5. 旧接口调用不报错，现有页面路由仍能工作。
