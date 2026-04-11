# DL Monitor And Deploy Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 恢复部署详情组件、修复深度学习监控页布局与分页、并让 DL 训练过程状态真正持久化到数据库。

**Architecture:** 保持现有 ML/DL 路由和页面边界不变，只为 DL 增加 epoch/log 持久化表和查询接口。前端继续保留三 tab 布局，但把部署详情和监控页交互拉回到用户认可的模式。

**Tech Stack:** FastAPI, SQLAlchemy async, SQLite, React 18, Ant Design, Playwright, pytest

---

### Task 1: 持久化模型设计与测试

**Files:**
- Modify: `ml_platform/app/models/database.py`
- Modify: `ml_platform/app/models/schemas.py`
- Test: `ml_platform/tests/test_dl_log_streaming.py`

- [ ] 写 DL epoch/log 持久化的失败测试
- [ ] 运行测试确认失败
- [ ] 为 DL epoch/log 增加 ORM 模型和 schema
- [ ] 运行测试确认通过

### Task 2: DL 服务层与接口

**Files:**
- Modify: `ml_platform/app/services/dl_service.py`
- Modify: `ml_platform/app/api/routes/dl.py`
- Modify: `ml_platform/app/core/logger.py`
- Test: `ml_platform/tests/test_dl_log_streaming.py`

- [ ] 写失败测试，覆盖每个 epoch 持久化和分页查询
- [ ] 运行测试确认失败
- [ ] 实现 DL epoch/log 持久化、分页接口、每 epoch 日志输出
- [ ] 运行测试确认通过

### Task 3: 数据集去重

**Files:**
- Modify: `ml_platform/app/services/data_service.py`
- Test: `ml_platform/tests/test_dataset_dedup.py`

- [ ] 写失败测试，覆盖重复上传复用已有数据集
- [ ] 运行测试确认失败
- [ ] 实现按内容 hash 的上传去重
- [ ] 运行测试确认通过

### Task 4: 模型管理页

**Files:**
- Modify: `ml_platform/app/api/routes/model_mgmt.py`
- Modify: `ml_platform_web/src/pages/ModelManagement.jsx`
- Test: `tests/model-management-metadata.spec.js`

- [ ] 写失败测试，覆盖标签/备注展示与分页
- [ ] 运行测试确认失败
- [ ] 实现标签/备注展示与编辑入口
- [ ] 运行测试确认通过

### Task 5: 部署页回退

**Files:**
- Modify: `ml_platform_web/src/pages/ModelDeploy.jsx`
- Test: `tests/model-pages-tabs.spec.js`
- Test: `tests/model-deploy-detail.spec.js`

- [ ] 写失败测试，覆盖 `v2.0.0` 风格详情组件
- [ ] 运行测试确认失败
- [ ] 恢复部署详情组件交互
- [ ] 运行测试确认通过

### Task 6: DL 监控页修复

**Files:**
- Modify: `ml_platform_web/src/pages/DLMonitor.jsx`
- Modify: `ml_platform_web/src/services/api.js`
- Test: `tests/dl-monitor-realtime-logs.spec.js`
- Test: `tests/dl-monitor-persistence.spec.js`

- [ ] 写失败测试，覆盖图表宽度、分页、刷新恢复
- [ ] 运行测试确认失败
- [ ] 实现响应式图表、分页表格、历史回填
- [ ] 运行测试确认通过

### Task 7: 集成验证

**Files:**
- Modify: `tests/analytics-and-models.spec.js`

- [ ] 运行后端单元测试
- [ ] 运行前端构建
- [ ] 运行关键 Playwright 回归
- [ ] 用 Playwright 截图检查 DL 监控页布局
