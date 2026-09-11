# ML Platform Web — 前端工程

React 18 + Vite + Ant Design 5 + ECharts。界面语言简体中文。

这份 README 是给**接手改布局和样式**的前端工程师看的：怎么把本地前端连上服务器后端联调、样式改在哪里、哪些地方不能随手动。

---

## 一、跑起来（联调服务器后端）

后端已经部署在服务器上，你**不需要在本地跑后端**，只要把 Vite 的代理指过去。

```bash
nvm use                              # Node 20.20.2（见 .nvmrc）
npm ci --include=optional
cp .env.local.example .env.local     # 里面已填好服务器地址
npm run dev                          # http://localhost:3000
```

`.env.local` 只有一行：

```
VITE_API_TARGET=http://203.176.93.249:18081
```

`vite.config.js` 读这个变量，把 `/api`、`/inference`、`/ws` 和 `/health` 请求都转发到它。**代理是在 Node 侧转发的，不走浏览器跨域**，所以不用管 CORS。

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
npm run dev:integrity # 隔离浏览器验证：前端 3300 + 内存模拟 API 3901
```

改完样式**至少跑一遍** `npm run test:unit` 和 `npm run build`。测试是 node 环境的纯函数测试（见第五节），跑得很快。

---

## 三、目录结构

```
src/
  App.jsx                    主题 Provider、登录路由与应用框架
  navigation/                页面路由、菜单、页签身份、缓存与状态上下文
  theme/                     dark tokens、Ant Design 配置与图表外观适配
  ui/                        Ant Design 兼容包装（表单与弹层归属）
  main.jsx                   入口
  pages/                     19 个页面组件，一个路由一个
  components/
    layout/                  Header / Sidebar
    workbench/               40 个文件 —— V3 建模工作台，本项目的主体
    viz/                     16 个文件 —— 可视化组件与注册表
    results/                 统一结果页的面板与注册表
  services/api.js            唯一的 axios 客户端，所有接口都在这里
  utils/                     格式化、路由推导、比较逻辑等纯函数
  styles/global.css          基础样式、组件几何、报告样式
  styles/workspace.css       应用框架、菜单页签、响应式与纸张配色
  hooks/                     自定义 hook
```

### 路由表（`src/navigation/PageRoutes.jsx`）

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
| `/training/*` `/dl/*` | ML/DL 配置、监控与结果 | 保留兼容，支持任务 ID 隔离页签 |

---

## 四、样式与页签

界面只使用 dark 风格，参考 `power-trade` 的真实组件和基座 tokens。

- `src/theme/tokens.js` / `antd.js`：颜色和 Ant Design 组件主题。
- `src/styles/global.css` / `workspace.css`：表格、弹窗尺寸、报告排版及 64px 顶栏、28px 页签栏等框架布局。
- `src/ui/index.jsx`：业务组件使用的 Ant Design 导出，保留 Form API，补充草稿保护和弹层归属。
- `src/navigation/routes.js`：统一菜单、标题、地址规范化、任务/family 缓存身份。
- `src/navigation/Workspace.jsx`：每个缓存页使用独立 location，保留滚动和组件状态；关闭时卸载。
- `src/hooks/useActiveEffect.js`：轮询、WebSocket 等在页签隐藏时清理，激活后恢复。

图表仍由原业务渲染器生成数据和 option。`src/utils/echarts.js` 统一初始化 dark 主题、尺寸观察和外观适配；`src/theme/chartOptions.js` 保留数据、格式化回调与业务语义。`reportCharts.js` 保留原来的 7 种语义图规格和旧归档透传；打印副本使用原纸张配色，且只在当前活动页签存在。

## 五、验证

当前共有 **29 个测试文件、250 个用例**，使用 Vitest node 环境，涵盖纯函数和 `react-dom/server` 渲染。改动后运行单元测试、ESLint 和构建；涉及缓存、表单、弹层或图表尺寸时也要做浏览器验证。

[本次改造与完整性验证报告](docs/DARK_UI_VALIDATION.md) 记录具体页面、状态、请求和验证边界。

`npm run dev:integrity` 启动隔离的本机模拟环境，创建/上传等操作不转发到真实后端。原有 `tests-e2e/` 下 4 个 Playwright 文件默认访问 3000 并写入其配置的后端；本次使用隔离模拟 API 和 CUA 浏览器交互验证，没有直接运行这组真实数据测试。

---

## 六、动之前先知道的几件事

1. **报告链路是最近重写的，逻辑集中且有测试保护**：
   `ReportDocument.jsx`（统一渲染总报告与分报告）、`reportCharts.js`（图表主题与渲染器）、`ReportPrintDocument.jsx`（打印/导出用的离屏文档）、`AiReportModal.jsx`（封面与阅读器外壳）。
   改这几个文件的**结构**要留意打印文档也得跟着对，改**样式**基本安全。

2. **正文里的图位由标记决定**：后端返回的 markdown 含 `{{chart:id}}`，`ReportDocument` 按标记就地插图。别把它当普通 markdown 渲染。

3. **旧归档要能打开**：老报告的 `charts[]` 里带的是现成的 ECharts option（`spec.option`），渲染器直接透传。删这条兼容分支会让历史报告白屏。

4. **`services/api.js` 是唯一的接口层**，新增请求写在这里，别在组件里直接 `axios`。注意 `/inference` 路由**不带 `/api` 前缀**，是后端设计如此，代理里单独配了规则。

5. **图表统一从 `utils/echarts.js` 初始化**。不要绕过此入口，否则 dark 适配和页签恢复后的尺寸观察不会生效。

6. **不要提交 `.env.local`**（已在 `.gitignore`）。

---

## 七、这个压缩包里没有什么

- `node_modules/`（跑 `npm install` 生成）
- `dist/`（跑 `npm run build` 生成）
- `.git/`（这是工程快照，不是仓库副本）
- `.env.local`（换成了 `.env.local.example`）
- 后端代码与任何密钥

改完之后把 `src/` 的 diff 或整个目录发回来即可。
