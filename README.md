# 傻子也会训练模型 — ML Training Platform

面向表格数据建模的训练平台：后端提供数据集管理、训练调度、模型管理、部署预测、日志和可视化 API；前端提供简体中文的训练工作台和结果查看界面。

当前发布版本：V2.0.0。

## 目录结构

```text
mcp/                      # 本地 MCP 封装，当前包含豆包 AI 报告工具
ml_platform/              # FastAPI 后端，默认端口 8000
  app/                    # API、服务、模型、调度器、核心训练逻辑
  alembic/                # 数据库迁移
  registry/               # 模型候选、调参空间等配置
  tests/                  # 后端 pytest 单测
  storage/                # 本地运行数据，已 gitignore
  mlruns/                 # MLflow 实验产物，已 gitignore

ml_platform_web/          # React + Vite 前端，默认端口 3000
  src/                    # 页面、组件、服务、样式、工具函数
  tests-e2e/              # 前端侧 E2E 用例

tests/                    # 根目录 Playwright E2E 测试
examples/data/            # 示例数据集
scripts/                  # 本地开发、部署和种子数据脚本
docker/                   # Docker 镜像、nginx、部署说明
doc/                      # 产品/架构/技术债文档
docs/superpowers/         # 历史规格与实施计划
```

## 技术栈

| 层级 | 技术 |
| --- | --- |
| 后端 | FastAPI, SQLAlchemy async, aiosqlite |
| 机器学习 | scikit-learn, XGBoost, LightGBM, SHAP |
| 实验管理 | MLflow |
| 前端 | React 18, Vite, Ant Design 5, ECharts |
| 状态管理 | Redux Toolkit |
| 实时通信 | WebSocket, EventBus |
| 测试 | pytest, Playwright, Vitest |

## 本地启动

如果默认端口没有被占用：

```bash
cd ml_platform
uvicorn app.main:app --reload --port 8000
```

```bash
cd ml_platform_web
npm run dev
```

如果 8000/3000 已被 Docker 或其他服务占用，可以换端口：

```bash
cd ml_platform
DATABASE_URL='sqlite+aiosqlite:///./storage/codex-dev.db' \
S3_ENABLED=false \
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

```bash
cd ml_platform_web
VITE_API_TARGET='http://127.0.0.1:8001' \
npm run dev -- --host 127.0.0.1 --port 3001
```

## 常用检查

```bash
# 后端重点测试
cd ml_platform
python -m pytest tests/

# 前端构建
cd ml_platform_web
npm run build

# 根目录 E2E，配置会自动启动测试端口的前后端
npx playwright test
```

## 运行产物

这些目录是本地生成物或运行数据，不提交到 git：

- `ml_platform/storage/`
- `ml_platform/storage/backups/`
- `ml_platform/mlruns/`
- `ml_platform_web/dist/`
- `output/`、`playwright-report/`、`test-results/`、`playwright_test/artifacts/`
- `.pytest_cache/`、`__pycache__/`、`.DS_Store`
