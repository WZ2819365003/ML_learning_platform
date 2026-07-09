# V3 平台梳理 — 导航收敛 + 多模型对比重做

日期：2026-07-09 · 分支：`feat/v3-unified-workflow`

## Context（为什么做）
V3 平台的核心价值是「一个任务里跑多个模型、比出最优」。但外层套了太多重叠的页面与入口——`建模工作台 / 实验管理 / 任务中心 / 训练方案 / Run 诊断中心` 职责重叠，`V3 平台 BETA` 菜单与新的「建模」分组重复；多模型对比又分散在详情页的「Run 对比 + 策略对比」两个 tab 里，以「策略/Run」为轴而非「模型」为轴，看不清爽。用户诉求（已确认）：**收敛导航 + 把多模型对比重做成清晰的主视图**，不动底层数据模型。

## 决策（已与用户确认）
- 改造力度：**中**——导航收敛 + 对比重做，不改底层数据模型（Task→Experiment→Run→PlatformTask→TrainingTask 保持不变）。
- **策略配置（基线/网格/贝叶斯调参）必须保留**，作为**独立入口**（不是内联进直配、也不删）。
- **策略对比不降级**，作为对比视图里的正经一块保留。
- 任务主界面**统一为「工作流」**；带 4 个 tab 的旧详情页降级（其对比 tab 用新组件替换，页面暂留作高级视图，不本轮删除）。

## 非目标
- 不改数据库 schema / 抽象层；不迁移 `platformExperimentsApi`（仅下线其页面）；不重写训练/部署链路；不上云。

## Part 1 — 导航收敛（`V3 平台` 菜单整体解散，全并入「建模」）

新的「建模」分组（`Sidebar.jsx`）：
| 菜单 | 路由 | 说明 |
|---|---|---|
| 任务列表 | `/v3/tasks` | 唯一主入口（去掉 V3 里重复的「建模工作台」） |
| 模型管理 | `/models` | 复用现有 ModelManagement |
| 模型部署 | `/deploy` | 复用现有 ModelDeploy |
| 运行诊断 | `/v3/runs` | 合并「任务中心」：tab【全部 Run（V3Runs 现状）/ 孤立任务（TaskCenter 的 platformTasks 视图）】 |

处理掉：
- **实验管理**（`/experiments`）：导航移除；路由 `Navigate` 重定向到 `/v3/tasks`；`platformExperimentsApi` 暂留（`ShapView` 仍用），标为后续迁移债。
- **训练方案**（`/v3/training-plans`）：从顶层菜单移除；页面保留，入口改到「调参/策略」入口里的「套用方案 / 存为方案」。
- **任务中心**（`/tasks`）：并入「运行诊断」，顶层菜单移除（路由重定向到 `/v3/runs`）。
- **V3 平台 BETA** 分组：彻底删除。
- `Header.jsx` 面包屑 section：`/v3/runs`、`/experiments`(重定向前) 等归「建模」。

## Part 2 — 多模型对比重做（核心）

新增可复用组件 `components/workbench/ModelComparison.jsx`，**同时用在**：工作流「训练过程」步、旧详情页「模型对比」tab（替换原 Run 对比 + 策略对比两 tab）。以**模型维度**为主：

- **顶部**：🏆 最优模型高亮（模型名 + 目标指标值 + 一键「部署此模型」）。
- **排行榜表**：行=模型/Run，列=该 task_type 的关键指标并排（分类 `accuracy/f1/roc_auc`；回归 `rmse/mae/r2`）+ 策略标签(小) + 状态；按目标指标排序、rank1 高亮；行内操作 `详情 / 日志 / SHAP / 下载模型 / 部署此模型`（复用 `RunInspector` + `deployRun` + `runModelDownloadUrl`）。
- **对比图**：主指标柱状图（ECharts，项目已依赖），可切换指标；一眼看差距。
- **策略对比（保留）**：作为组件内一块「按策略看」（复用现有 `StrategyCompareTab` 的五数概括/箱线 + best），与模型对比并列、不删不降级。
- 数据源复用：`modelingTaskApi.leaderboard` / `runs` / `strategyComparison`。

工作流「训练过程」步：进度树 + `ModelComparison`（取代当前简版排行榜）。

## Part 3 — 策略配置保留为独立入口

在工作流「模型配置」步的 tab 组里，新增第 4 个 tab **「调参策略」**，与 `机器学习 / 深度学习 / 混合` 并列：
- 内容 = 复用 `ExperimentBatchModal` 的策略表单（策略选择 基线/网格/贝叶斯 + 参与模型 + 每模型搜索空间 + 预算），抽成内联组件在该 tab 渲染并派发批次。
- **训练方案接入这里**：本 tab 顶部提供「套用已保存方案 / 存为方案」（复用 `trainingPlansApi` + `TrainingPlanPicker`）。
- 直配 3 tab（基线直接训练）与「调参策略」tab 各自 `createExperimentBatch` 派发，走同一管线。

## 需要改动/新增的关键文件
- 导航：`components/layout/Sidebar.jsx`、`components/layout/Header.jsx`、`App.jsx`（路由重定向 `/experiments`→`/v3/tasks`、`/tasks`→`/v3/runs`）。
- 对比：新增 `components/workbench/ModelComparison.jsx`；`pages/ModelingWorkflow.jsx`（训练过程步用它）；`pages/ModelingTaskDetail.jsx`（Run对比+策略对比 两 tab 合并为「模型对比」）。
- 诊断合并：`pages/V3Runs.jsx` 加「孤立任务」tab（吸收 `TaskCenter` 的 platformTasks 视图 / 复用 `OrphanTaskDetailDrawer`）。
- 策略入口：`components/workbench/ModelConfigTabs.jsx`（加「调参策略」tab）；抽 `ExperimentBatchModal` 表单体为内联可复用（保留 Modal 供详情页「启动新批次」继续用）。
- 复用不改：`RunInspector`、`StrategyCompareTab`、`ProgressTree`、`deployRun`、`runModelDownloadUrl`、`ModelManagement`、`ModelDeploy`。

## 验证
1. 侧边栏只剩「建模(任务列表/模型管理/模型部署/运行诊断) / 时序任务 / 系统设置」；无「V3 平台」「实验管理」「任务中心」「训练方案」顶层项；`/experiments`、`/tasks` 深链重定向正确。
2. 任务里「模型对比」：指标并排表 + 柱状图 + 点模型开 RunInspector + 部署最优；策略对比仍在。
3. 「调参策略」tab：选网格/贝叶斯 + 搜索空间能派发一批调参 Run；「套用方案」能预填。
4. `运行诊断` 两 tab 都有数据；旧 `/models`、`/deploy` 正常。
5. `cd ml_platform && python -m pytest tests/`（无后端改动，回归绿）；`npm run lint` 改动文件 0 error；本地 `:5188`+`:8001` 逐项截图。

## 风险
- 动 `ModelingTaskDetail`（合并两 tab）→ 用共享 `ModelComparison` 降低重复；保留页面路由，仅换 tab 内容。
- 抽 `ExperimentBatchModal` 表单体供两处用（详情页 Modal + 配置步 tab）→ 用「表单体 + 外壳」拆分，Modal 行为不回归。
- 合并 `TaskCenter`→`V3Runs` 的孤立任务视图 → 直接搬 `OrphanTaskDetailDrawer` 与 platformTasksApi 调用，不重写。
