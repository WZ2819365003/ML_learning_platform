# ML Platform Web — 前端工程

React 18 + Vite + Ant Design 5 + ECharts。界面语言简体中文。

这份 README 是给**接手改布局和样式**的前端工程师看的：怎么把本地前端连上服务器后端联调、样式改在哪里、哪些地方不能随手动。

---

## 一、跑起来（联调服务器后端）

后端已经部署在服务器上，你**不需要在本地跑后端**，只要把 Vite 的代理指过去。

```bash
npm install
cp .env.local.example .env.local     # 里面已填好服务器地址
npm run dev                          # http://localhost:3000
```

`.env.local` 只有一行：

```
VITE_API_TARGET=http://203.176.93.249:18081
```

`vite.config.js` 读这个变量，把 `/api`、`/inference`、`/ws` 三类请求都转发到它。**代理是在 Node 侧转发的，不走浏览器跨域**，所以不用管 CORS。

改完这个变量要重启 `npm run dev` 才生效。

### 登录

打开 `http://localhost:3000` 会跳到 `/login`。账号问后端同事要。

登录成功后 token 存在 `localStorage.ml_platform_token`，`services/api.js` 的拦截器会自动带上。调试时想直接看某个页面，可以在控制台确认：

```js
localStorage.getItem('ml_platform_token')   // 有值说明已登录
```

401 会被拦截器捕获并跳回登录页——如果你在改页面时突然被弹走，先看这个。

### 验证连通

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://203.176.93.249:18081/health         # 200
curl -s -o /dev/null -w "%{http_code}\n" http://203.176.93.249:18081/api/v3/tasks/  # 401（正常，要登录）
```

---

## 二、常用命令

```bash
npm run dev          # 开发服务器，端口 3000
npm run build        # 生产构建，输出 dist/
npm run preview      # 预览构建产物
npm run test:unit    # 单元测试（vitest）
npm run lint         # ESLint，--max-warnings 0
```

改完样式**至少跑一遍** `npm run test:unit` 和 `npm run build`。测试是 node 环境的纯函数测试（见第五节），跑得很快。

---

## 三、目录结构

```
src/
  App.jsx                    路由表（见下）
  main.jsx                   入口
  pages/                     19 个页面组件，一个路由一个
  components/
    layout/                  Header / Sidebar / ErrorBoundary
    workbench/               40 个文件 —— V3 建模工作台，本项目的主体
    viz/                     16 个文件 —— 可视化组件与注册表
    results/                 统一结果页的面板与注册表
  services/api.js            唯一的 axios 客户端，所有接口都在这里
  utils/                     格式化、路由推导、比较逻辑等纯函数
  styles/global.css          850 行，全部全局样式（见第四节）
  hooks/                     自定义 hook
```

### 路由表（`App.jsx`）

当前主线是 `/v3/*`，其余多为历史路由的重定向。

| 路由 | 页面 | 说明 |
|---|---|---|
| `/` `/dashboard` | → `/v3/tasks` | 仪表盘已删除，重定向保留给旧书签 |
| `/v3/tasks` | `ModelingTasks` | **落地页**，建模任务列表 |
| `/v3/tasks/:taskId` | `ModelingTaskDetail` | 任务详情：概览 / 实验编排 / 模型对比 / **任务报告** |
| `/v3/tasks/:taskId/workflow` | `ModelingWorkflow` | 四步工作流：导入数据 → 模型配置 → 训练过程 → 部署上线 |
| `/v3/runs` | `V3Runs` | 运行诊断，跨任务的 Run 平铺表；操作按钮开 `RunInspector` 抽屉 |
| `/v3/training-plans` | `TrainingPlans` | 训练方案管理 |
| `/data` | `DataManagement` | 数据集上传与预览 |
| `/models` `/deploy` | `ModelManagement` / `ModelDeploy` | 模型管理与部署（含多模型加权融合） |
| `/ts/*` | 时序任务 | 独立一套 |
| `/training/*` `/dl/*` | 旧流水线 | **已废弃**，勿在此投入 |

---

## 四、样式改在哪里

### 1. `src/styles/global.css`（850 行）

按注释分区，从上到下：

| 区块 | 行数附近 | 内容 |
|---|---|---|
| Design tokens | 5 | CSS 变量：色板、圆角、阴影、间距 |
| Base | 54 | 全局重置、滚动条 |
| Animations | 78 | 过渡与关键帧 |
| **AI report** | 88–304 | 报告阅读器：正文排版、封面、图表区、附录折叠 |
| **Ant Design overrides** | 305–476 | 卡片 / 按钮 / 标签 / 表格 / 表单 / 弹窗 / 布局 |
| Premium utility classes | 477+ | 项目自用工具类 |

**改全局观感优先动 Design tokens**，比逐个组件覆盖干净。

### 2. Tailwind

`tailwind.config.js` 存在，`global.css` 顶部有 `@tailwind base/components/utilities`。工具类可以直接用，但现有代码**主要靠 Ant Design + 手写 CSS**，Tailwind 用得很少。新增时保持一致，别在同一个组件里混两套。

### 3. 图表样式：`components/workbench/reportCharts.js`

报告里所有图表的样式**只在这一个文件里定义**——`REPORT_CHART_THEME` 里的配色、字号、边距、悬停格式、高度。

```js
export const REPORT_CHART_THEME = Object.freeze({
  colors: { primary, primaryLight, muted, accent, accentLight, ink, text, subtle, grid, axis, shade, onFill },
  palette: [...],
  fontSize, fontFamily, labelMargin, ...
})
```

这是刻意设计的：后端只发**语义图规格**（`{kind, title, caption, tooltip_fields, rows, ...}`，7 种 `kind`：`hbar` / `dots` / `hist` / `stacked` / `lines` / `scatter_pair` / `matrix`），前端 `renderReportChart(spec)` 把它映射成 ECharts option。

> **改图表样式请改这里，不要在组件里补 `grid.left` 或 `itemStyle`。**
> 之前就是一张图一张图地补，导致轴标签被裁、图例翻页、柱子高度为 0 之类的问题反复出现。现在改一处、全部图受益。
>
> 后端**不允许**出现任何 ECharts 字段，有测试卡着。

---

## 五、测试

`vitest.config.js` 用的是 `environment: 'node'`，**没有 jsdom、没有 testing-library**。所以现有测试全是**纯函数测试**：给一个规格，断言产出的 option / 视图模型 / 格式化结果。

26 个测试文件、228 个用例。渲染层的验证用 `react-dom/server` 的 `renderToStaticMarkup`（见 `ReportPrintDocument.test.js`），只验"能不能渲染出正确的标记"，不做交互测试。

改样式一般不会碰这些测试；改结构（比如动 `renderReportChart` 或视图模型）就会，跑一下就知道。

### E2E

包里带了 `tests-e2e/` 下的 4 个 Playwright 用例（任务中心层级、方案创建与套用、Run 抽屉、清理回归）和 `playwright.config.js`。

它们默认打 `http://localhost:3000`，需要你的 dev server 已经在跑（配置文件里写明了不自动拉起）。因为代理指向服务器后端，这些用例会**读写服务器上的真实数据**——跑之前先跟后端同事确认一声。

改样式一般用不上它们，跑单元测试就够。

---

## 六、动之前先知道的几件事

1. **报告链路是最近重写的，逻辑集中且有测试保护**：
   `ReportDocument.jsx`（统一渲染总报告与分报告）、`reportCharts.js`（图表主题与渲染器）、`ReportPrintDocument.jsx`（打印/导出用的离屏文档）、`AiReportModal.jsx`（封面与阅读器外壳）。
   改这几个文件的**结构**要留意打印文档也得跟着对，改**样式**基本安全。

2. **正文里的图位由标记决定**：后端返回的 markdown 含 `{{chart:id}}`，`ReportDocument` 按标记就地插图。别把它当普通 markdown 渲染。

3. **旧归档要能打开**：老报告的 `charts[]` 里带的是现成的 ECharts option（`spec.option`），渲染器直接透传。删这条兼容分支会让历史报告白屏。

4. **`services/api.js` 是唯一的接口层**，新增请求写在这里，别在组件里直接 `axios`。注意 `/inference` 路由**不带 `/api` 前缀**，是后端设计如此，代理里单独配了规则。

5. **`components/workbench/` 有 40 个文件**，是主体。`TrainingViz.jsx` / `ShapView.jsx` / `StrategyCompareTab.jsx` 里还有历史遗留的内联 ECharts 样式，没纳入统一主题——如果你要顺手统一，那是独立一件事，改动面不小。

6. **不要提交 `.env.local`**（已在 `.gitignore`）。

---

## 七、这个压缩包里没有什么

- `node_modules/`（跑 `npm install` 生成）
- `dist/`（跑 `npm run build` 生成）
- `.git/`（这是工程快照，不是仓库副本）
- `.env.local`（换成了 `.env.local.example`）
- 后端代码与任何密钥

改完之后把 `src/` 的 diff 或整个目录发回来即可。
