# Model Asset Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 DL 日志实时回传，并在兼容现有接口的前提下统一模型管理和模型部署体验。

**Architecture:** 保留 ML / DL 现有存储与动作接口，在后端新增统一模型资产视图和统一部署视图，前端改为基于统一视图渲染。DL 监控页补齐通用日志 WebSocket 订阅与状态同步更新。

**Tech Stack:** FastAPI, SQLAlchemy async, React 18, Ant Design 5, Axios, pytest, Playwright

---

## File Map

- Modify: `ml_platform/app/services/dl_service.py`
- Modify: `ml_platform/app/api/routes/dl.py`
- Modify: `ml_platform/app/api/routes/model_mgmt.py`
- Modify: `ml_platform/app/api/routes/deploy.py`
- Modify: `ml_platform/app/models/schemas.py`
- Modify: `ml_platform_web/src/services/api.js`
- Modify: `ml_platform_web/src/pages/DLMonitor.jsx`
- Modify: `ml_platform_web/src/pages/ModelManagement.jsx`
- Modify: `ml_platform_web/src/pages/ModelDeploy.jsx`
- Create: `ml_platform/tests/test_unified_model_assets.py`
- Create: `ml_platform/tests/test_dl_log_streaming.py`

### Task 1: DL 日志实时回传

**Files:**
- Modify: `ml_platform/app/services/dl_service.py`
- Modify: `ml_platform_web/src/pages/DLMonitor.jsx`
- Test: `ml_platform/tests/test_dl_log_streaming.py`

- [ ] Step 1: 写失败用例，证明 DL 日志没有通过通用日志通道实时消费
- [ ] Step 2: 运行 `python -m pytest ml_platform/tests/test_dl_log_streaming.py -q`
- [ ] Step 3: 在 DL 训练链路里补齐日志广播与 `current_epoch/progress` 更新
- [ ] Step 4: 在 `DLMonitor.jsx` 同时订阅 epoch 和 log 两类 WebSocket
- [ ] Step 5: 再次运行 `python -m pytest ml_platform/tests/test_dl_log_streaming.py -q`

### Task 2: 统一模型资产接口

**Files:**
- Modify: `ml_platform/app/api/routes/model_mgmt.py`
- Modify: `ml_platform/app/models/schemas.py`
- Test: `ml_platform/tests/test_unified_model_assets.py`

- [ ] Step 1: 写失败用例，定义统一模型资产列表返回结构
- [ ] Step 2: 运行 `python -m pytest ml_platform/tests/test_unified_model_assets.py -q`
- [ ] Step 3: 新增统一模型资产列表/详情 schema 与路由实现
- [ ] Step 4: 保证 ML / DL 字段被标准化映射到统一结构
- [ ] Step 5: 再次运行 `python -m pytest ml_platform/tests/test_unified_model_assets.py -q`

### Task 3: 统一部署视图接口

**Files:**
- Modify: `ml_platform/app/api/routes/deploy.py`
- Modify: `ml_platform/app/models/schemas.py`
- Modify: `ml_platform/app/services/dl_service.py`

- [ ] Step 1: 先为统一部署视图补测试断言
- [ ] Step 2: 新增统一部署列表接口，标准化 ML / DL 返回字段
- [ ] Step 3: 保留现有创建/删除/状态切换/预测接口不变
- [ ] Step 4: 运行相关 pytest 用例，确认兼容接口仍可用

### Task 4: 统一前端模型管理页

**Files:**
- Modify: `ml_platform_web/src/services/api.js`
- Modify: `ml_platform_web/src/pages/ModelManagement.jsx`

- [ ] Step 1: 接入统一模型资产 API
- [ ] Step 2: 把列表改成单视图 + 运行时筛选
- [ ] Step 3: 统一删除、备注/标签、部署入口
- [ ] Step 4: 保留 ML 专有能力和 DL 专有摘要展示
- [ ] Step 5: 本地检查页面渲染和关键交互

### Task 5: 统一前端部署页

**Files:**
- Modify: `ml_platform_web/src/services/api.js`
- Modify: `ml_platform_web/src/pages/ModelDeploy.jsx`

- [ ] Step 1: 接入统一部署视图 API
- [ ] Step 2: 改成单表格和单详情测试面板
- [ ] Step 3: 按运行时分派实际测试请求
- [ ] Step 4: 保持旧的 ML / DL 创建部署动作入口可用
- [ ] Step 5: 本地检查 URL 展示、启停、删除、在线测试

### Task 6: 回归验证

**Files:**
- Test: `ml_platform/tests/test_dl_log_streaming.py`
- Test: `ml_platform/tests/test_unified_model_assets.py`
- Test: `tests/`

- [ ] Step 1: 运行新增 pytest 用例
- [ ] Step 2: 运行后端现有相关 pytest 用例
- [ ] Step 3: 运行至少一条覆盖模型管理/部署的 Playwright 用例
- [ ] Step 4: 记录未覆盖风险和兼容性结果
