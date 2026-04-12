# 时序任务模块重构设计

**日期**: 2026-04-12  
**状态**: Approved

---

## 目标

把当前 `TimesFM/Chronos` 页面从“基础模型控制台”重构为“独立的时序任务模块”，并在模型部署页提供可管理的 `TimesFM` 部署实例。

本轮只处理产品抽象、接口语义、页面结构和 UI 质量，不扩展复杂的数据预处理逻辑。

---

## 问题

当前实现存在 4 个核心问题：

1. [TSConfig.jsx](C:/Users/zhuow/cowork_workplace/gitLabCoding/project/Ai_learning/ml_platform_web/src/pages/TSConfig.jsx)、[TSMonitor.jsx](C:/Users/zhuow/cowork_workplace/gitLabCoding/project/Ai_learning/ml_platform_web/src/pages/TSMonitor.jsx)、[TSResults.jsx](C:/Users/zhuow/cowork_workplace/gitLabCoding/project/Ai_learning/ml_platform_web/src/pages/TSResults.jsx) 仍然沿用“模型状态 / 预加载 / 模型选择”的心智，和实际业务不符。
2. `TimesFM` 被当成“要管理的模型”，但用户要的是“基础时序能力 + 多部署实例 + 多预测任务”。
3. 模型部署页的通用模型 tab 仍然是占位，无法创建和管理 `TimesFM` 部署。
4. 时序页面的视觉结构混乱，列表、配置、详情、返回路径和专业信息暴露方式都不够稳定。

---

## 设计原则

1. 时序模块与 ML / DL 解耦，但保持平台整体风格一致。
2. `TimesFM` 不进入模型管理页。
3. `TimesFM` 可以在部署页创建多个部署实例，每个实例拥有独立 URL。
4. 时序任务创建时必须显式绑定 `deployment_id`。
5. 重要技术信息不隐藏，只重构排版和优先级。
6. 兼容现有 `/api/timesfm/*` 和旧前端路由，避免一次性打断现有功能。

---

## 模块边界

### 1. 时序任务

时序任务是独立模块，不归入训练任务中心，也不出现在模型管理中。

前端主流程：

- `任务列表`：分页查看和筛选
- `新建任务`：选择数据集、部署实例、预测参数
- `任务详情`：查看图表、结果表、技术参数、原始响应

### 2. TimesFM 部署

`TimesFM` 不作为训练产物存在，但作为“基础时序服务部署实例”存在于模型部署页。

每个部署实例具备：

- `deployment_id`
- `name`
- `description`
- `status`
- `request_count`
- `backend_label`
- `predict_url`
- `created_at`

### 3. 模型管理

模型管理保留 ML / DL / 通用模型三 tab，但通用模型不再直接承载 `TimesFM` 管理动作。它只保留说明和跳转，明确“TimesFM 请在模型部署中管理”。

---

## 页面结构

### A. 时序任务列表页

推荐新路由：`/ts/tasks`

页面结构：

- 顶部：页面标题、说明、主按钮“新建时序任务”
- 摘要卡：总任务数、运行中、成功数、失败数
- 筛选栏：状态、数据集、部署实例、时间范围
- 分页表格：任务名称、数据集、部署实例、状态、预测步长、创建时间、操作

行为：

- 点击行进入详情页 `/ts/tasks/{id}`
- 返回列表时保留筛选条件和分页

### B. 新建时序任务页

推荐新路由：`/ts/tasks/new`

页面结构：

- 左侧主表单：数据集、时间列、目标列、预测步长、频率、部署实例
- 右侧专业信息卡：
  - 部署 URL
  - 基础引擎说明
  - 当前参数摘要
  - 输入要求和注意事项

不再展示：

- 模型预加载按钮
- 模型下载状态控制台
- 用户手动切换基础模型版本

### C. 时序任务详情页

推荐新路由：`/ts/tasks/:id`

页面结构：

- 顶部：返回列表、任务标题、状态和主操作
- 第一屏：任务摘要卡
- 第二屏：预测图表
- 第三屏：预测结果表
- 第四屏：技术详情

技术详情必须展示：

- `deployment_id`
- 部署名称
- `predict_url`
- 基础引擎 / 后端标签
- 数据集、时间列、目标列
- 提交参数
- 原始响应 JSON
- 错误信息 / 任务日志

### D. 模型部署页中的 TimesFM Tab

保留 [ModelDeploy.jsx](C:/Users/zhuow/cowork_workplace/gitLabCoding/project/Ai_learning/ml_platform_web/src/pages/ModelDeploy.jsx) 的三 tab 结构：

- 机器学习部署
- 深度学习部署
- TimesFM 部署

TimesFM tab 采用“左侧列表 + 右侧详情组件”结构：

- 左：部署实例分页表格
- 右：部署详情、URL、状态切换、在线测试、调用统计、新建部署

---

## 路由与兼容

### 新前端路由

- `/ts/tasks`
- `/ts/tasks/new`
- `/ts/tasks/:id`

### 兼容旧前端路由

- `/ts/config` -> `/ts/tasks/new`
- `/ts/monitor` -> `/ts/tasks`
- `/ts/results?id={id}` -> `/ts/tasks/{id}`

### 新后端接口

任务接口：

- `POST /api/ts/tasks`
- `GET /api/ts/tasks`
- `GET /api/ts/tasks/{task_id}`
- `DELETE /api/ts/tasks/{task_id}`

部署接口：

- `POST /api/ts/deployments`
- `GET /api/ts/deployments`
- `GET /api/ts/deployments/{deployment_id}`
- `PATCH /api/ts/deployments/{deployment_id}/status`
- `DELETE /api/ts/deployments/{deployment_id}`
- `POST /api/ts/deployments/{deployment_id}/predict`

兼容旧接口：

- `/api/timesfm/start`
- `/api/timesfm/list`
- `/api/timesfm/{id}`

旧接口内部复用新的任务服务，不再作为主心智暴露。

---

## 数据模型

### 1. 新增 `TimeSeriesDeployment`

用途：保存可管理的 TimesFM 部署实例。

关键字段：

- `id`
- `name`
- `description`
- `backend_label`
- `status`
- `request_count`
- `config`
- `created_at`
- `updated_at`

### 2. 扩展 `TimeSeriesForecastTask`

新增：

- `deployment_id`
- `task_params`
- `runtime_info`

保留：

- `dataset_id`
- `dataset_name`
- `value_column`
- `time_column`
- `horizon`
- `frequency`
- `status`
- `result`
- `error_message`
- `created_at`
- `started_at`
- `finished_at`

### 3. 轻量迁移策略

项目当前没有 Alembic。本轮采用启动时 schema 补齐：

- 新表通过 `create_all` 创建
- 旧表缺失列时，在启动阶段执行 SQLite `ALTER TABLE ... ADD COLUMN`

---

## UI / UX 方向

本轮采用 `ui-ux-pro-max` 的“专业控制台”增量设计，不走花哨 demo 风格。

设计约束：

- 保持 Ant Design 基础交互稳定
- 统一卡片圆角、阴影、间距和状态色
- 表格全部分页
- 图表固定高度，避免拉伸变形
- 页面首屏优先显示最关键的操作和状态
- 技术细节可见，但通过分区和描述卡组织

色彩方向：

- 主色：深蓝
- 中性色：石墨灰 / 冷白
- 状态色：绿色、琥珀、红色

---

## 验证

后端：

- 新增 `pytest` 覆盖时序部署 CRUD、任务创建绑定部署、兼容旧接口

前端：

- `npm run build`
- Playwright 覆盖：
  - 时序任务列表 -> 详情 -> 返回
  - 新建任务提交流程
  - TimesFM 部署 tab 的列表和详情

---

## 风险

1. 现有 SQLite 数据库不会自动添加新列，需要补齐启动迁移逻辑。
2. 旧页面和旧接口仍需兼容，避免前端跳转和后端返回结构同时断裂。
3. TimesFM 运行依赖 Python 3.10 子环境，部署实例管理不能假装成“完全本地轻量 API”；页面上需要诚实展示后端引擎信息。
