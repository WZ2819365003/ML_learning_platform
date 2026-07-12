# V3 平台梳理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 收敛 V3 平台导航到单一「建模」分组，并把分散的多模型对比重做成一个清晰、可复用的「模型对比」视图；策略配置保留为独立入口。

**Architecture:** 纯前端重排 + 组件抽取（不动后端与数据模型）。核心是一个归一化 adapter（`buildComparisonVM`）把 run 列表转成统一 VM，喂给共享 `ModelComparison` 组件；部署护栏抽成 `useDeployRun`；`ExperimentBatchModal` 拆成 `ExperimentBatchForm`（受控）+ 薄壳，让"调参策略"能内嵌进配置步。

**Tech Stack:** React 18 + Vite + Ant Design 5 + ECharts + react-router-dom v6；测试 Playwright（已用）+ 新增 Vitest（仅纯逻辑单测）。

## Global Constraints
- 分支 `feat/v3-unified-workflow`；本地验证用前端 `:5188`（`.env.local` 已指 `:8001` 后端），**不上云**。
- **不改**后端、DB schema、数据模型；**不删** `platformExperimentsApi` / `platformTasksApi` / `trainingPlansApi`。
- **不做**「存为方案」（本轮仅「套用方案」）。
- `npm run lint` 对改动文件必须 0 error。
- 复用既有组件：`RunInspector`、`StrategyCompareTab`、`ProgressTree`、`ModelManagement`、`ModelDeploy`、`OrphanTaskDetailDrawer`、`TrainingPlanPicker`。

### 共享接口（跨任务，务必一致）
```ts
// utils/comparison.js
Row = {
  run_id: string, model_type: string, strategy_type: string, status: string,
  metrics: Record<string, number>, objective_value: number|null,
  domain_task_id: string|null, family: 'ml'|'dl'|null, is_best: boolean
}
ComparisonVM = { rows: Row[], metricKeys: string[], objective_metric: string, objective_direction: 'max'|'min' }
buildComparisonVM(rawRows: any[], task: {objective_metric, objective_direction}): ComparisonVM

// hooks/useDeployRun.js
useDeployRun(task): { deploying: boolean, deployment: object|null, error: string|null, deploy(runId: string, opts?: {name?: string}): Promise<void> }
```

---

### Task 1: 导航收敛（IA）+ 重定向

**Files:**
- Modify: `ml_platform_web/src/components/layout/Sidebar.jsx`（删 V3 平台分组；建模加 运行诊断）
- Modify: `ml_platform_web/src/App.jsx`（加 `/experiments`→`/v3/tasks`、`/tasks`→`/v3/runs` 重定向；删对应旧 route 的菜单可达性，但保留 `/experiments/:experimentId`、`/v3/tasks/:id`）
- Modify: `ml_platform_web/src/components/layout/Header.jsx`（`/v3/runs` section 归「建模」；`PAGE_TITLES['/v3/runs']`→'运行诊断'；section 表去掉已删项）
- Modify: `ml_platform_web/src/pages/V3Runs.jsx`（页标题 `Run 诊断中心`→`运行诊断`）
- Modify: `ml_platform_web/src/pages/ExperimentRedirect.jsx`（顶部注释 `:8` 与兜底文案 `:44` 不再指向 `/experiments`）
- Test: `playwright_test/test/13-v3-ia-redirects.spec.js`（新）

**Interfaces:** Produces 无（纯路由/导航）。

- [ ] **Step 1: 写重定向失败测试**

⚠ 项目有两套 Playwright 配置：根 `playwright.config.js`（testDir `./tests`，baseURL `127.0.0.1:3000`，webServer 自动起前后端）与 `playwright_test/playwright.config.js`（testDir `./test`，baseURL `process.env.BASE_UI || 127.0.0.1:3000`）。本测试放 `playwright_test/test/`，**用相对路径吃所在套件的 baseURL，不得硬编码端口**（本地对着 :5188 跑时用 `BASE_UI=http://localhost:5188` 传入）。

```js
// playwright_test/test/13-v3-ia-redirects.spec.js
import { test, expect } from '@playwright/test'
test('legacy V3 routes redirect into 建模', async ({ page }) => {
  await page.goto('/experiments')
  await expect(page).toHaveURL(/\/v3\/tasks$/)
  await page.goto('/tasks')
  await expect(page).toHaveURL(/\/v3\/runs$/)
})
test('sidebar has no V3 平台 group; 建模 has 运行诊断', async ({ page }) => {
  await page.goto('/v3/tasks')
  const sider = page.locator('.ant-layout-sider')
  // 注意：V3 菜单标题带 BETA 徽章（文本为 "V3 平台BETA"），精确 name 匹配会漏 — 用包含匹配
  await expect(sider.getByText('V3 平台')).toHaveCount(0)
  await expect(sider.getByText('实验管理')).toHaveCount(0)
  await expect(sider.getByText('任务中心')).toHaveCount(0)
  await expect(sider.getByText('运行诊断')).toBeVisible()
})
```

- [ ] **Step 2: 跑测试确认失败**
Run: `cd playwright_test && BASE_UI=http://localhost:5188 npx playwright test test/13-v3-ia-redirects.spec.js`
Expected: FAIL（当前有 V3 平台菜单、无重定向）

- [ ] **Step 3: 改 Sidebar 建模分组**（`Sidebar.jsx`：`menuItems` 里 建模.children 追加、删除 V3 平台分组块）
```jsx
// 建模 children：
children: [
  { key: 'modeling-tasks', label: <Link to="/v3/tasks">任务列表</Link> },
  { key: 'models',         label: <Link to="/models">模型管理</Link> },
  { key: 'deploy',         label: <Link to="/deploy">模型部署</Link> },
  { key: 'v3-runs',        label: <Link to="/v3/runs">运行诊断</Link> },
],
// 删除整个 { key:'v3', label:'V3 平台'... } 分组对象。
// getSelectedKey: /v3/runs → 'v3-runs'；删除 tasks/experiments 分支或让其落到 modeling-tasks。
// defaultOpenKeys 去掉 'v3'。
```

- [ ] **Step 4: 改 App.jsx 路由**（保留详情/workflow/redirect 详情页；旧列表页重定向）
```jsx
<Route path="/experiments" element={<Navigate to="/v3/tasks" replace />} />
<Route path="/experiments/:experimentId" element={<ExperimentRedirect />} />
<Route path="/tasks" element={<Navigate to="/v3/runs" replace />} />
// TaskCenter/Experiments 的 import 可留（未用则删，避免 lint unused）。V3Runs 路由保持 /v3/runs。
```

- [ ] **Step 5: 改 Header section/标题 + V3Runs 页标题 + ExperimentRedirect 文案**
`Header.jsx`：SECTIONS `/v3/runs`→'建模'；删除 `/tasks`、`/experiments` 到「V3 平台」的映射（现重定向）；`PAGE_TITLES['/v3/runs']` 由 `'Run 诊断中心'` 改为 `'运行诊断'`。
`V3Runs.jsx:325`：页面 `<Title>` 由 `Run 诊断中心` 改为 `运行诊断`（与菜单一致，旧断言在 Step 7 同步更新）。
`ExperimentRedirect.jsx`：顶部注释（`:8`）与兜底跳转/文案（`:44`）改成指向 `/v3/tasks`（"返回任务列表"）。

- [ ] **Step 6: lint + 跑测试确认通过**
Run: `npx eslint src/components/layout/Sidebar.jsx src/App.jsx src/components/layout/Header.jsx src/pages/ExperimentRedirect.jsx`
Run: `npx playwright test playwright_test/test/13-v3-ia-redirects.spec.js`
Expected: lint 0 error；重定向测试 PASS

- [ ] **Step 7: 更新受影响的旧 Playwright 断言**
改 `tests/v3-workbench.spec.js`、`tests/v3-runs-page.spec.js`、`tests/v3-strategies-uiplus.spec.js`、`tests/v3_2-smoke.spec.js` 中断言旧菜单名（建模工作台/Run 诊断中心/V3 平台）的用例，指向新 IA。

- [ ] **Step 8: Commit**
```bash
git add ml_platform_web/src playwright_test/test/13-v3-ia-redirects.spec.js tests
git commit -m "refactor(v3): IA 收敛 — V3 平台并入建模 + /experiments,/tasks 重定向"
```

---

### Task 2: `buildComparisonVM` adapter（纯逻辑 + 单测）

**Files:**
- Create: `ml_platform_web/src/utils/comparison.js`
- Create: `ml_platform_web/src/utils/comparison.test.js`
- Modify: `ml_platform_web/package.json`（加 `"test:unit": "vitest run"`；devDep `vitest`）
- Create: `ml_platform_web/vitest.config.js`

**Interfaces:** Produces `buildComparisonVM(rawRows, task) => ComparisonVM`（见 Global 接口）。

- [ ] **Step 1: 装 vitest + 配置**
```bash
cd ml_platform_web && npm i -D vitest
```
```js
// vitest.config.js
import { defineConfig } from 'vitest/config'
export default defineConfig({ test: { environment: 'node', include: ['src/**/*.test.js'] } })
```
package.json scripts 加：`"test:unit": "vitest run"`。

- [ ] **Step 2: 写失败单测**
```js
// src/utils/comparison.test.js
import { describe, it, expect } from 'vitest'
import { buildComparisonVM } from './comparison'
const task = { objective_metric: 'accuracy', objective_direction: 'max' }
const rows = [
  { run_id: 'a', params: { model_type: 'random_forest' }, strategy_type: 'baseline', status: 'SUCCESS', metrics: { accuracy: 0.9, f1: 0.88 }, objective_value: 0.9, domain_task_id: 'd1', family: 'ml' },
  { run_id: 'b', params: { model_type: 'xgboost' }, strategy_type: 'baseline', status: 'SUCCESS', metrics: { accuracy: 0.96, roc_auc: 1.0 }, objective_value: 0.96, domain_task_id: 'd2', family: 'ml' },
]
describe('buildComparisonVM', () => {
  it('metricKeys = union of run metrics, objective first', () => {
    const vm = buildComparisonVM(rows, task)
    expect(vm.metricKeys[0]).toBe('accuracy')
    expect(new Set(vm.metricKeys)).toEqual(new Set(['accuracy', 'f1', 'roc_auc']))
  })
  it('is_best by direction=max', () => {
    const vm = buildComparisonVM(rows, task)
    expect(vm.rows.find(r => r.is_best).run_id).toBe('b')
  })
  it('is_best by direction=min', () => {
    const vm = buildComparisonVM(rows, { objective_metric: 'rmse', objective_direction: 'min' })
    // objective_value 越小越好：a(0.9) < b(0.96) → a best
    expect(vm.rows.find(r => r.is_best).run_id).toBe('a')
  })
  it('model_type from params, carries domain_task_id/family', () => {
    const vm = buildComparisonVM(rows, task)
    expect(vm.rows[0].model_type).toBe('random_forest')
    expect(vm.rows[0].domain_task_id).toBe('d1')
  })
  it('empty rows → empty vm', () => {
    const vm = buildComparisonVM([], task)
    expect(vm.rows).toEqual([]); expect(vm.metricKeys).toEqual(['accuracy'])
  })
})
```

- [ ] **Step 3: 跑测试确认失败**
Run: `npm run test:unit`
Expected: FAIL（comparison 未实现）

- [ ] **Step 4: 实现 adapter**
```js
// src/utils/comparison.js
export function buildComparisonVM(rawRows = [], task = {}) {
  const objective_metric = task.objective_metric || 'accuracy'
  const objective_direction = task.objective_direction === 'min' ? 'min' : 'max'
  const rows = (rawRows || []).map(r => ({
    run_id: r.run_id,
    model_type: r.params?.model_type || r.model_type || '-',
    strategy_type: r.strategy_type || 'baseline',
    status: r.status || 'SUCCESS',
    metrics: r.metrics || {},
    objective_value: typeof r.objective_value === 'number' ? r.objective_value : null,
    domain_task_id: r.domain_task_id ?? null,
    family: r.family ?? null,
    is_best: false,
  }))
  // metricKeys：objective 优先，其余来自 run.metrics 键并集
  const keys = new Set([objective_metric])
  rows.forEach(r => Object.keys(r.metrics).forEach(k => keys.add(k)))
  const metricKeys = [objective_metric, ...[...keys].filter(k => k !== objective_metric).sort()]
  // is_best：按 objective_value + direction 选唯一最优（忽略 null）
  const scored = rows.filter(r => r.objective_value != null)
  if (scored.length) {
    const best = scored.reduce((a, b) =>
      (objective_direction === 'max' ? b.objective_value > a.objective_value : b.objective_value < a.objective_value) ? b : a)
    best.is_best = true
  }
  return { rows, metricKeys, objective_metric, objective_direction }
}
```

- [ ] **Step 5: 跑测试确认通过**
Run: `npm run test:unit`
Expected: PASS（5 tests）

- [ ] **Step 6: Commit**
```bash
git add ml_platform_web/src/utils/comparison.js ml_platform_web/src/utils/comparison.test.js ml_platform_web/vitest.config.js ml_platform_web/package.json ml_platform_web/package-lock.json
git commit -m "feat(v3): buildComparisonVM 归一化 adapter + vitest 单测"
```

---

### Task 3: `useDeployRun` hook（从 DeployStep 抽护栏）

**Files:**
- Create: `ml_platform_web/src/hooks/useDeployRun.js`
- Modify: `ml_platform_web/src/components/workbench/DeployStep.jsx`（改用 hook，行为不变）

**Interfaces:**
- Consumes: `modelingTaskApi.deployRun`（已存在）
- Produces: `useDeployRun(task) => { deploying, deployment, error, deploy(runId, {name}) }`

- [ ] **Step 1: 实现 hook**
```js
// src/hooks/useDeployRun.js
import { useState } from 'react'
import { message } from 'antd'
import { modelingTaskApi } from '../services/api'
export function useDeployRun(task) {
  const [deploying, setDeploying] = useState(false)
  const [deployment, setDeployment] = useState(null)
  const [error, setError] = useState(null)
  const deploy = async (runId, { name } = {}) => {
    if (!runId) { message.warning('请先选择一个成功的 Run'); return }
    setDeploying(true); setError(null)
    try {
      const resp = await modelingTaskApi.deployRun(task.id, runId, {
        name: (name && name.trim()) || `${task.name}-部署`,
        description: `来自建模任务 ${task.name} 的模型`,
      })
      setDeployment(resp); message.success('部署成功，模型已上线')
    } catch (err) {
      setError(err?.response?.data?.detail || '部署失败')
    } finally { setDeploying(false) }
  }
  return { deploying, deployment, error, deploy }
}
```

- [ ] **Step 2: DeployStep 改用 hook**
把 `DeployStep.jsx` 里本地的 `deploying/deployment/handleDeploy` 换成 `const { deploying, deployment, error, deploy } = useDeployRun(task)`；`handleDeploy` → `deploy(runId, { name })`；错误展示读 `error`。快速预测块保持不变。

- [ ] **Step 3: 回归验证（本地浏览器）**
起 `:5188`+`:8001`，进已成功任务的 部署步 → 部署 → 出「推理接口」；预测返回。与改前一致。

- [ ] **Step 4: lint**
Run: `npx eslint src/hooks/useDeployRun.js src/components/workbench/DeployStep.jsx`
Expected: 0 error

- [ ] **Step 5: Commit**
```bash
git add ml_platform_web/src/hooks/useDeployRun.js ml_platform_web/src/components/workbench/DeployStep.jsx
git commit -m "refactor(v3): 抽 useDeployRun 共享部署护栏，DeployStep 复用"
```

---

### Task 4: `ModelComparison` 组件 + 接入工作流训练步

**Files:**
- Create: `ml_platform_web/src/components/workbench/ModelComparison.jsx`
- Modify: `ml_platform_web/src/pages/ModelingWorkflow.jsx`（训练过程步用 `ModelComparison` 取代当前简版排行榜）

**Interfaces:**
- Consumes: `buildComparisonVM`（Task 2）、`useDeployRun`（Task 3）、`RunInspector`、`StrategyCompareTab`、`runModelDownloadUrl`、`modelingTaskApi.leaderboard/runs`
- Produces: `<ModelComparison task={task} rows={rawRows} loading={bool} error={str} onRefresh={fn} />`

- [ ] **Step 1: 组件骨架（三态 + 表 + 图 + 策略节 + 部署）**
关键点：`const vm = useMemo(() => buildComparisonVM(rows, task), [rows, task])`。
- **表**：固定列(排名/模型/策略/状态) + `vm.metricKeys.map(k => 列: render r.metrics[k] ?? '-')`；`is_best` 行高亮。
- **行操作**：详情/解释(RunInspector)；下载 仅当 `domain_task_id` 存在；**部署**——按钮 `disabled = !(r.status==='SUCCESS' && r.domain_task_id)`（Task 5 详情页会喂进 RUNNING/FAILED 的 run，务必按此禁用），点击弹命名框（默认 `${task.name}-部署`，可改）→ `deploy(r.run_id, { name })`（`useDeployRun`，Task 3）。
- **顶部「部署最优」**：对 `vm.rows.find(r => r.is_best)`，同样受 `SUCCESS && domain_task_id` 约束。
- **对比图（ECharts 柱状）**：⚠**先按模型聚合**——同一 `model_type` 可能有多条 run（网格/贝叶斯多 trial），每模型只取**其最优 trial**的该指标值（按 `objective_direction` 选 max/min）画一根柱；可切 `metricKeys`。
- **策略节**：底部内嵌 `<StrategyCompareTab taskId={task.id} onInspect={rid => openInspector(rid,'shap')} />`（自取策略数据，不经 adapter）。
- **三态**：`loading`→Spin/骨架，`error`→Alert，空→Empty（对齐 `StrategyCompareTab` 风格）。

- [ ] **Step 2: 工作流训练步接入**
`ModelingWorkflow.jsx` 训练过程步：把 `runColumns`+排行榜 Card+`StrategyCompareTab` 三块替换为
`<ModelComparison task={task} rows={leaderboard} loading={runsLoading} error={null} onRefresh={loadRuns} />`（保留上方 ProgressTree + 状态计数 + 「再加一组」）。`RunInspector`/inspector 状态移入 ModelComparison（或保留在页面、通过回调）。

- [ ] **Step 3: 本地验证**
进 `本地e2e-iris` 工作流训练步：指标并排列（accuracy/f1/roc_auc）+ 柱状图 + 点模型开 RunInspector + 部署最优；策略对比仍在；空/加载/错误态正常。`preview_console_logs` 无报错。

- [ ] **Step 4: lint**
Run: `npx eslint src/components/workbench/ModelComparison.jsx src/pages/ModelingWorkflow.jsx`

- [ ] **Step 5: Commit**
```bash
git add ml_platform_web/src/components/workbench/ModelComparison.jsx ml_platform_web/src/pages/ModelingWorkflow.jsx
git commit -m "feat(v3): ModelComparison 共享对比视图，接入工作流训练步"
```

---

### Task 5: 详情页两 tab 合并为「模型对比」

**Files:**
- Modify: `ml_platform_web/src/pages/ModelingTaskDetail.jsx`（`Run 对比` + `策略对比` 两 tab → 单个「模型对比」用 `ModelComparison`）

**Interfaces:** Consumes `ModelComparison`（Task 4）。

- [ ] **Step 1: 替换 tab**
`tabItems` 里删 `runs` 与 `strategy-compare` 两项，新增 `{ key:'compare', label:<><LineChartOutlined/> 模型对比</>, children:<ModelComparison task={task} rows={runs.length?runs:leaderboard} loading={runsLoading||lbLoading} error={null} onRefresh={refreshAll} /> }`。删除 `runsTab`/`leaderboardColumns`/`runStrategyFilter` 等仅服务旧 tab 的代码。`概览`、`实验编排` 两 tab 保留。

- [ ] **Step 2: 本地验证**
`/v3/tasks/:id`（详情）→「模型对比」tab 与工作流一致；深链稳定；无报错。

- [ ] **Step 3: lint + 提交**
```bash
npx eslint src/pages/ModelingTaskDetail.jsx
git add ml_platform_web/src/pages/ModelingTaskDetail.jsx
git commit -m "refactor(v3): 详情页 Run对比+策略对比 合并为「模型对比」"
```

---

### Task 6: 抽 `ExperimentBatchForm` +薄壳（受控）

**Files:**
- Create: `ml_platform_web/src/components/workbench/ExperimentBatchForm.jsx`（控制器+状态+展示）
- Modify: `ml_platform_web/src/components/workbench/ExperimentBatchModal.jsx`（改为薄壳，包 Form）

**Interfaces:**
- Produces: `<ExperimentBatchForm task active resetKey onSubmitted />`；`active`(bool) 触发重置/加载调参空间/方案列表（替代原依赖 Modal `open` 的副作用）；`onSubmitted()` 提交成功回调。
- `ExperimentBatchModal` = `<Modal open ...><ExperimentBatchForm task active={open} resetKey={open} onSubmitted={()=>{onSubmitted?.();onClose?.()}}/></Modal>`

- [ ] **Step 1: 固化 Modal 回归断言**（若无则补最小 Playwright：详情页「启动新批次」→ 选模型 → 提交成功）
- [ ] **Step 2: 搬迁**：把 `ExperimentBatchModal.jsx` 现有全部表单 state/effects/handleSubmit/JSX 移进 `ExperimentBatchForm.jsx`，effects 依赖从 `open` 改为 `active`（`useEffect(...,[active, task?.task_type])`）。Modal 变薄壳。
- [ ] **Step 3: 回归**：详情页「启动新批次」行为不变（本地 + Playwright）。
- [ ] **Step 4: lint + 提交**
```bash
git add ml_platform_web/src/components/workbench/ExperimentBatchForm.jsx ml_platform_web/src/components/workbench/ExperimentBatchModal.jsx
git commit -m "refactor(v3): 抽 ExperimentBatchForm(受控)+薄壳，保 Modal 不回归"
```

---

### Task 7: 配置步新增「调参策略」tab

**Files:**
- Modify: `ml_platform_web/src/components/workbench/ModelConfigTabs.jsx`（加第 4 个 tab）

**Interfaces:** Consumes `ExperimentBatchForm`（Task 6）。

- [ ] **Step 1: 加 tab**
`Tabs.items` 追加 `{ key:'tune', label:'调参策略', children:<ExperimentBatchForm task={task} active={activeKey==='tune'} resetKey={task?.id} onSubmitted={onSubmitted}/> }`；用 `Tabs onChange` 维护 `activeKey`，只有当前是 `tune` 时 `active=true`（避免后台重复请求）。「套用方案」已在 ExperimentBatchForm 内（保留），「存为方案」不做。

⚠**训练方案管理页别成孤儿**：Task 1 删了 V3 菜单里「训练方案」入口，而「套用方案」只是下拉、到不了 `/v3/training-plans` 管理页。**在调参策略 tab 顶部加一个 `<Button type="link" onClick={()=>window.open('/v3/training-plans','_blank')}>管理训练方案 →</Button>`**（或 `navigate`），让用户仍能创建/编辑方案。`/v3/training-plans` 路由在 App.jsx 保留（Task 1 不删该 route，仅删菜单项）。
- [ ] **Step 2: 本地验证**：配置步出现 机器学习/深度学习/混合/**调参策略** 四 tab；调参策略里选网格/贝叶斯 + 搜索空间 → 派发调参 Run（进度树可见）；套用方案预填。
- [ ] **Step 3: lint + 提交**
```bash
git add ml_platform_web/src/components/workbench/ModelConfigTabs.jsx
git commit -m "feat(v3): 配置步新增「调参策略」tab（内嵌受控 ExperimentBatchForm）"
```

---

### Task 8: 孤立任务并入运行诊断（最后）

**Files:**
- Modify: `ml_platform_web/src/pages/V3Runs.jsx`（加「孤立任务」tab）
- 复用（不改）：`ml_platform_web/src/pages/TaskCenter.jsx` 的 `FlatView` 逻辑与 `OrphanTaskDetailDrawer`、`platformTasksApi`

- [ ] **Step 1: V3Runs 包一层 Tabs**
现有扁平表放进 tab「全部 Run」；新增 tab「孤立任务」：整块搬 `TaskCenter` 的孤立任务表（`platformTasksApi` 列表 + 重试/取消/删除/批量/筛选/选择 + `OrphanTaskDetailDrawer`）。若 TaskCenter 里该视图已是独立子组件则直接引用；否则抽成 `components/workbench/OrphanTasksPanel.jsx` 再在两处用。
- [ ] **Step 2: 本地验证**：`/v3/runs` 两 tab 都有数据；孤立任务的 重试/取消/删除/抽屉可用；`/tasks` 重定向到此页。
- [ ] **Step 3: lint + 提交**
```bash
git add ml_platform_web/src/pages/V3Runs.jsx ml_platform_web/src/components/workbench/OrphanTasksPanel.jsx
git commit -m "feat(v3): 运行诊断合并「孤立任务」tab（搬自任务中心）"
```

---

## 收尾（全部任务后）
- [ ] 全量回归：`cd ml_platform && python -m pytest tests/`（应绿，无后端改动）；`cd ml_platform_web && npm run test:unit`；`npm run lint`（改动文件 0 error）。
- [ ] 全量 Playwright 巡检（更新后的 spec）。
- [ ] 还原 `.env.local`/`:8001` 说明保持不变；分支不合并、不上云（等你确认）。

## Verification（端到端，本地 `:5188`+`:8001`）
1. 侧栏只剩 仪表盘/数据管理/**建模(任务列表·模型管理·模型部署·运行诊断)**/时序任务/系统设置；`/experiments`→`/v3/tasks`、`/tasks`→`/v3/runs` 重定向；`/experiments/:id` 仍工作。
2. 工作流训练步 & 详情页「模型对比」：动态指标并排 + 柱状图 + RunInspector + 部署最优 + 策略对比；三态正常。
3. 配置步四 tab；「调参策略」能派发网格/贝叶斯；详情页「启动新批次」Modal 无回归。
4. 运行诊断两 tab（全部 Run / 孤立任务，含重试/取消/删除）。
5. `/models`、`/deploy` 正常。
```
