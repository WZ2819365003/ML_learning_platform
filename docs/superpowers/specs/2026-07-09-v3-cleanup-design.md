# V3 平台梳理 — 导航收敛 + 多模型对比重做

日期：2026-07-09 · 分支：`feat/v3-unified-workflow` · v2（已折入 Codex 复审意见）

## Context（为什么做）
V3 平台的核心价值是「一个任务里跑多个模型、比出最优」。但外层套了太多重叠的页面与入口——`建模工作台 / 实验管理 / 任务中心 / 训练方案 / Run 诊断中心` 职责重叠，`V3 平台 BETA` 菜单与新的「建模」分组重复；多模型对比又分散在详情页的「Run 对比 + 策略对比」两个 tab 里，以「策略/Run」为轴而非「模型」为轴，看不清爽。用户诉求（已确认）：**收敛导航 + 把多模型对比重做成清晰的主视图**，不动底层数据模型。

## 决策（已与用户确认）
- 改造力度：**中**——导航收敛 + 对比重做，不改底层数据模型（Task→Experiment→Run→PlatformTask→TrainingTask 保持不变）。
- **策略配置（基线/网格/贝叶斯调参）保留**，作为**独立入口**（配置步新增「调参策略」tab）。
- **策略对比不降级**，作为对比视图里的正经一块保留。
- 任务主界面**主推「工作流」**；带 4 个 tab 的旧详情页**保留但降为"高级视图"**（其对比 tab 用新组件替换），本轮**不删**。

## 非目标
- 不改数据库 schema / 抽象层；**不删除 `platformExperimentsApi`**（`ShapView`、`ExperimentRedirect` 仍用，仅下线 `/experiments` 页面）；不重写训练/部署链路；不上云；**「存为方案」不在本轮**（现无保存能力，属新功能，本轮只做「套用方案」）。

## 主/次入口与深链（回应 Codex #1）
- 任务列表行主操作 → **工作流** `/v3/tasks/:id/workflow`（主）。
- 「查看详情（Tab 视图）」保留为次级入口 `/v3/tasks/:id`（高级视图）。
- 深链保持稳定：`/v3/tasks`、`/v3/tasks/:id`、`/v3/tasks/:id/workflow` 均不变；仅新增重定向（下）。

## Part 1 — 导航收敛（`V3 平台` 菜单整体解散，全并入「建模」）

新的「建模」分组（`Sidebar.jsx`）：任务列表 `/v3/tasks`（去重，删 V3 的「建模工作台」）、模型管理 `/models`、模型部署 `/deploy`、**运行诊断** `/v3/runs`。

处理掉（页面下线只动导航+路由，**不动其 API 封装**）：
- **实验管理**（`/experiments`）：导航移除；`App.jsx` 加 `Navigate` 重定向到 `/v3/tasks`；**保留** `platformExperimentsApi`（`ShapView.jsx:312`、`ExperimentRedirect.jsx:25` 仍用）；同步更新 `ExperimentRedirect.jsx:44` 兜底文案（不再指向 `/experiments`）。
- **训练方案**（`/v3/training-plans`）：顶层菜单移除；页面保留，入口改到「调参策略」tab 的「套用方案」。
- **任务中心**（`/tasks`）：顶层菜单移除；路由重定向到 `/v3/runs`。其孤立任务视图并入运行诊断（见 Part 4，**排在最后做**）。
- **V3 平台 BETA** 分组彻底删除；`Header.jsx` 面包屑 section：`/v3/runs`、`/models`、`/deploy` 归「建模」。

## Part 2 — 多模型对比重做（核心）

### 2a. 归一化 adapter（回应 Codex #5，先做）
新增 `utils/comparison.js`：把三种数据形状归一到一个 `ComparisonVM`：
- 输入源：`modelingTaskApi.leaderboard`（`ModelingWorkflow.jsx:88`）、`runs`（`ModelingTaskDetail.jsx:107`）、`strategyComparison`（`StrategyCompareTab.jsx:52`）。
- 输出：`{ rows: [{run_id, model_type, strategy_type, status, metrics:{}, objective_value, domain_task_id, family, is_best}], metricKeys: string[], strategyStats }`。
- `metricKeys` **动态**取自任务 `eval_metrics` ∪ 目标指标（回应 Codex #7），渲染时**缺失回退**为 `-`。

### 2b. 共享组件 `components/workbench/ModelComparison.jsx`
以**模型维度**为主，集中处理 筛选/行操作/三态（空/加载/错误，对齐 `StrategyCompareTab.jsx:190`，回应 Codex #8）：
- 顶部：🏆 最优模型高亮 + 一键「部署最优」。
- 排行榜表：模型 × `metricKeys` 动态列并排 + 策略标签(小) + 状态；按目标指标排序、rank1 高亮；行内 `详情/日志/SHAP/下载/部署此模型`（复用 `RunInspector`；部署走 **shared deploy helper**，见 2c）。
- 对比图：主指标柱状图（ECharts），可切指标。
- 策略对比（**保留**）：内嵌一块「按策略看」，复用 `StrategyCompareTab`。

### 2c. 共享部署 helper（回应 Codex #6）
把 `DeployStep.jsx:21/47` 的护栏抽成 `useDeployRun()`（校验 `status==SUCCESS && domain_task_id`、命名、响应态、错误提示），供 `DeployStep` 与 `ModelComparison` 行操作共用，避免重复/漏护栏。

### 2d. 接入（先一处，回应 Codex 排序）
先只在 **工作流「训练过程」步** 用 `ModelComparison`（取代当前简版排行榜 `ModelingWorkflow.jsx:301`）；数据形状验证 OK 后，再替换 `ModelingTaskDetail.jsx` 的「Run 对比 + 策略对比」两 tab 为单个「模型对比」tab。

## Part 3 — 策略配置保留为独立入口（回应 Codex #3）
- 先把 `ExperimentBatchModal` 拆成 **`ExperimentBatchForm`（控制器+状态+展示：重置/加载调参空间/方案列表/payload 组装/提交）** + **`ExperimentBatchModal`（薄壳）**；保证详情页「启动新批次」Modal 行为不回归（保留其现有 Playwright 断言）。
- 工作流「模型配置」步新增第 4 个 tab **「调参策略」**，内嵌 `ExperimentBatchForm`（策略 基线/网格/贝叶斯 + 参与模型 + 每模型搜索空间 + 预算 + **套用方案**）。
- 直配 3 tab 与「调参策略」tab 各自 `createExperimentBatch` 派发，走同一管线。

## Part 4 — 运行诊断合并（最后做，回应 Codex #4）
`V3Runs.jsx`（`v3RunsApi` 扁平表）加第 2 个 tab「孤立任务」：**整块搬** `TaskCenter` 的 `FlatView` + `OrphanTaskDetailDrawer` + `platformTasksApi`（含 重试/取消/删除/批量/筛选/选择），不重写。排在所有改动之后。

## 实施顺序（采纳 Codex 建议）
1. **Part 1 IA**（侧栏/头/路由/重定向 + `ExperimentRedirect` 文案）——含重定向测试。
2. **2a adapter + 2b/2c ModelComparison**，**仅接工作流训练步**。
3. 验证数据形状后，**替换详情页两 tab** 为「模型对比」。
4. **Part 3**：抽 `ExperimentBatchForm`+壳（先保 Modal 测试），再加「调参策略」tab。
5. **Part 4**：孤立任务并入运行诊断。

## 需要改动/新增的关键文件
- 导航：`Sidebar.jsx`、`Header.jsx`、`App.jsx`、`ExperimentRedirect.jsx`（文案）。
- 对比：新增 `utils/comparison.js`、`components/workbench/ModelComparison.jsx`、`hooks/useDeployRun.js`；改 `ModelingWorkflow.jsx`、`ModelingTaskDetail.jsx`；复用 `StrategyCompareTab`、`RunInspector`、`DeployStep`。
- 策略：`ExperimentBatchModal.jsx` → 抽 `ExperimentBatchForm.jsx` + 壳；`ModelConfigTabs.jsx`（加「调参策略」tab）。
- 诊断：`V3Runs.jsx`（加孤立任务 tab）、搬用 `OrphanTaskDetailDrawer`。
- 不删：`platformExperimentsApi`、`platformTasksApi`、`trainingPlansApi`、ModelManagement、ModelDeploy。

## 验证
1. 侧边栏只剩「建模(任务列表/模型管理/模型部署/运行诊断) / 时序任务 / 系统设置」；`/experiments`→`/v3/tasks`、`/tasks`→`/v3/runs` 重定向正确；`/experiments/:id`（ExperimentRedirect）仍可用。
2. 「模型对比」：动态指标并排表 + 柱状图 + RunInspector 下钻 + 部署最优；策略对比仍在；空/加载/错误三态正常。
3. 「调参策略」tab：网格/贝叶斯 + 搜索空间派发调参 Run；「套用方案」预填；详情页「启动新批次」Modal 无回归。
4. 运行诊断两 tab（全部 Run / 孤立任务，含重试/取消/删除）有数据；`/models`、`/deploy` 正常。
5. `cd ml_platform && python -m pytest tests/`（无后端改动，回归绿）；**更新受影响的 Playwright**（现断言旧名 `建模工作台/Run 诊断中心/Run 对比/模型解释`，见 `tests/v3-workbench.spec.js:61`、`tests/v3-runs-page.spec.js:51`）；`npm run lint` 改动文件 0 error；本地 `:5188`+`:8001` 逐项截图。

## 风险与缓解
- 抽 `ExperimentBatchForm`：**控制器/状态与展示一起抽**（非纯 JSX），先固化 Modal 测试再改。
- `ModelComparison` 复用两处：靠 2a 归一化 adapter + 集中筛选/行操作/三态，避免形状分叉。
- 部署行操作：统一走 `useDeployRun` 护栏。
- `TaskCenter`→`V3Runs`：整块搬 FlatView+Drawer，排最后，单独回归。
