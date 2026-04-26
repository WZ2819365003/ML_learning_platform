# V3 平台里程碑测试套件 (playwright_test)

独立于根目录 `tests/` 的"里程碑式"E2E 测试套件，覆盖：
- 平台冒烟 / 数据 / ML / DL / TS / 模型管理 / V3 工作台 / 全站导航 / WebSocket
- 后端 API + 前端页面 + 实时通道 三层全打通
- 巡检类用例会把异常记录到报告；发布门禁类用例必须失败即红

## 运行
```bash
cd playwright_test
npx playwright test --config=playwright.config.js
```

## 单测 / 发布门禁
```bash
npx playwright test --config=playwright.config.js test/07-v3-workbench.spec.js
npx playwright test --config=playwright.config.js test/08-v3-end-to-end.spec.js
```

推荐显式指定环境，避免本机端口与 Docker/Nginx 端口混用：
```bash
BASE_UI=http://127.0.0.1:3000 \
BASE_API=http://127.0.0.1:8000/api \
BASE_ROOT=http://127.0.0.1:8000 \
npx playwright test --config=playwright.config.js test/08-v3-end-to-end.spec.js
```

## 报告
- `playwright-testV1.md` — 主报告（人读）
- `artifacts/results.json` — 完整 JSON（含每条用例的 annotation/attachment）
- `artifacts/html/` — Playwright HTML 报告
- `artifacts/test-results/` — trace.zip & screenshot
- `artifacts/backend_log_errors.txt` — 测试时间窗内的后端错误日志
- `artifacts/mysql_log.txt` — MySQL 错误日志（应为空）

## 前置条件
- Docker stack 全部 healthy：`docker ps` 应看到 mysql/redis/minio/backend/frontend/nginx
- 前端 `http://127.0.0.1:3000` 可访问
- 后端 `http://127.0.0.1:8000/health` 返回 200
- `examples/data/predictive_maintenance.csv` 存在；08 门禁会在运行时上传一份新数据集，避免依赖历史 DB 记录或容器内旧上传文件

## 约定
- `01`、`02`、`04`、`06`、`09`、`10` 偏巡检：允许把非关键异常以 annotation / attachment 记录，保证一次运行尽量收集完整问题面。
- `08-v3-end-to-end.spec.js` 是发布门禁：必须真实完成 `TrainingPlan -> ModelingTask -> ExperimentBatch -> ExperimentRun -> Inspector -> SHAP`，任何 4xx/5xx、run FAILED、超时或 SHAP 缺失都应直接失败。
- 前端页面巡检记录 console error / 4xx / pageerror 到 attachments；门禁用例需要把这些异常收敛成明确断言。
- WebSocket 巡检只校验"能 open"，不校验消息内容（不存在的 task_id 也允许订阅）。
