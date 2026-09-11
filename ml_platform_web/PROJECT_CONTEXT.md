# ML 建模平台与天津 VPP 工程上下文

梳理日期：2026-09-09。依据本机工作区源码、配置、项目说明和 skill；运行状态为本次检查时的快照。

当前建模平台与天津 VPP 是两个独立前端项目。`itmg-dtk-base` 与 `power-trade` 存在明确的微前端挂载和代码依赖关系；在已检查的源码、依赖与注册配置中，未发现 ML 平台已经接入天津基座、交易应用或共用其鉴权的实现。

本文主体记录改造前的工程梳理。2026-09-09 后续已完成 React dark 样式、菜单和多页签改造；当前实现与测试结果见 [改造验证报告](docs/DARK_UI_VALIDATION.md)。天津参考工程及项目 skill 未修改，ML 开发代理仍指向用户指定的远程后端。

**工程对照**

| 维度 | 当前 ML 平台 | itmg-dtk-base | power-trade |
| --- | --- | --- | --- |
| 职责 | 数据、建模、训练诊断、报告、部署与推理 | 登录、布局、权限、菜单、页签、主题、请求、微应用挂载 | 电力辅助交易业务 |
| 技术栈 | React 18、React Router 6、Vite 5 | Vue 2.6、Vue Router 3、Vuex 3、Vue CLI 4、qiankun 2 | Vue 2、Vue Router 3、Vuex 3、Vue CLI 4 |
| UI | Ant Design 5、ECharts 5、CSS 变量 | Element UI、项目公共组件、Less/SCSS | Element UI、ECharts 5、业务组件、Less |
| 开发端口 | 3000 | 6600 | 6616 |
| 入口示例 | `/v3/tasks`、`/v3/runs` | 登录后通过菜单进入子应用 | 独立 `/home`；基座 `/power-trade/home` |
| 运行形态 | 独立 React SPA | qiankun 主应用 | 同一 UMD 产物支持挂载及完整独立运行 |
| 主题 | 浅色内容区、深色侧栏；无三主题切换实现 | dark / light / auroraBlue | 继承或打包复用 Base 三主题 |
| 当前进程 | 3000 监听，Node 18.18.0 | 本次未发现 6600 监听 | 6616 监听，Node 14.17.3 |
| Git 状态 | 已初始化 Git，master 尚无提交；末次核对已有初始文件暂存 | 天津仓库 feature-v1.0 | 同一天津仓库，存在未提交联调改动 |

当前终端默认 Node 为 14.17.3；不同工程不能直接共用这一个版本执行全部命令。

**一、当前 ML 平台：以建模任务为主线**

工程位置：[ml_platform_web](/Users/dubbing/工作/恒实/code/展会/ml_platform_web)。入口为 [main.jsx](/Users/dubbing/工作/恒实/code/展会/ml_platform_web/src/main.jsx)，路由、Ant Design 主题和应用外壳集中在 [App.jsx](/Users/dubbing/工作/恒实/code/展会/ml_platform_web/src/App.jsx)。当前有 19 个页面文件、26 个单元测试文件及 4 个 E2E 文件。

主业务链路是：数据集 → ModelingTask → Experiment → Run → 对比/诊断/报告 → 最终评估与部署 → 在线或批量推理。工作流界面实际分为四步：导入数据、模型配置、训练过程、部署上线；对比与可视化并入训练过程。

| 路由 | 主要实现 | 用途 |
| --- | --- | --- |
| `/v3/tasks` | ModelingTasks | 任务列表，当前落地页 |
| `/v3/tasks/new/workflow` | ModelingWorkflow | 新建任务与四步流程 |
| `/v3/tasks/:taskId/workflow` | ModelingWorkflow | 已有任务工作流 |
| `/v3/tasks/:taskId` | ModelingTaskDetail | 概览、实验编排、模型对比、任务报告 |
| `/v3/runs` | V3Runs + RunInspector | 跨任务运行列表、详情抽屉、诊断与 SHAP |
| `/v3/training-plans` | TrainingPlans | 可复用训练方案 |
| `/data` | DataManagement | 数据上传、预览和管理 |
| `/models`、`/deploy` | ModelManagement、ModelDeploy | 模型及部署管理，含加权融合 |
| `/ts/tasks` 及详情/新建路由 | TSMonitor、TSConfig、TSResults | 独立时序任务链路 |
| `/training/*`、`/dl/*` | 旧训练页面、统一结果页 | 历史兼容能力，部分页面仍实际可访问 |

README 将旧训练链路标为废弃，表示后续投入优先级；源码中仍保留配置、监控与结果路由，不能直接当成已全部重定向或可删除代码。`/dashboard`、`/tasks`、`/experiments` 等旧入口已有重定向。

页面状态主要由 React hooks 管理。虽然 package.json 声明了 Redux Toolkit / react-redux，但当前 src 中未发现 store、Provider 或 slice 接入，不应把 Redux 视为现有核心状态架构。

关键目录与修改入口：

- [workbench](/Users/dubbing/工作/恒实/code/展会/ml_platform_web/src/components/workbench)：核心工作台，包含模型配置、批量实验、进度树、对比、诊断、报告、部署和数据 Pipeline。
- [results](/Users/dubbing/工作/恒实/code/展会/ml_platform_web/src/components/results)：统一结果页及按结果类型选择面板的注册机制。
- [viz](/Users/dubbing/工作/恒实/code/展会/ml_platform_web/src/components/viz)：训练历史、交叉验证、校准、预测分布、诊断等可视化组件与可用性判断。
- [global.css](/Users/dubbing/工作/恒实/code/展会/ml_platform_web/src/styles/global.css)：全局语义变量、报告样式、Ant Design 覆盖和布局样式。Tailwind 存在，但源码主要使用 Ant Design 与手写 CSS。
- [useLogStream.js](/Users/dubbing/工作/恒实/code/展会/ml_platform_web/src/hooks/useLogStream.js)：历史日志与 `/ws/logs/{domainTaskId}` 实时流。

**ML 接口、代理与鉴权**

统一入口为 [services/api.js](/Users/dubbing/工作/恒实/code/展会/ml_platform_web/src/services/api.js)。它集中导出数据、训练、模型、部署、ML/DL/TS、平台任务、实验、Run、训练方案等 API 对象。

| 链路 | 实现 |
| --- | --- |
| 普通 API | axios `baseURL: '/api'`，如 `/api/v3/runs/` |
| 推理 API | 第二个 axios 实例 `baseURL: '/'`，使用 `/inference/...` |
| 登录态 | localStorage 的 `ml_platform_token`，请求附带 Bearer token |
| 过期处理 | API 返回 401 后清理 token 并跳转 `/login`；登录接口自身 401 由页面处理 |
| WebSocket | 同源 `/ws/...`，通过 query 传递登录 token |
| 开发代理 | Vite 转发 `/api`、`/inference`、`/ws` |

[vite.config.js](/Users/dubbing/工作/恒实/code/展会/ml_platform_web/vite.config.js) 读取 `VITE_API_TARGET`；未设置时回退到 `http://127.0.0.1:8000`。当前 [.env.local](/Users/dubbing/工作/恒实/code/展会/ml_platform_web/.env.local) 设置为 `http://203.176.93.249:18081`。

上一轮已验证：补配置前本地 `/api/v3/runs/` 返回 500；补配置后本地任务与 Run 接口均返回远端的未登录 401，远端 `/health` 返回 200。此结果证明代理可达，不代表已完成登录后的业务验收。

**ML 报告链路**

- `ReportDocument.jsx` 统一渲染总报告和分报告；Markdown 中的 `{{chart:id}}` 决定图表插入位置。
- [reportCharts.js](/Users/dubbing/工作/恒实/code/展会/ml_platform_web/src/components/workbench/reportCharts.js) 集中维护 `REPORT_CHART_THEME` 和七种语义图规格：hbar、dots、hist、stacked、lines、scatter_pair、matrix。
- 新报告由后端给语义数据、前端生成 ECharts option；历史归档仍兼容 `spec.option`，不能随意删除。
- `ReportPrintDocument.jsx`、`AiReportModal.jsx` 和视图模型负责打印、阅读器、封面及归档选择。改结构时需要兼顾屏幕和打印输出。

**二、itmg-dtk-base：公共能力和微应用容器**

工程位置：[itmg-dtk-base](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/itmg-dtk-base)。整个天津仓库遵循根 [AGENTS.md](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/AGENTS.md)，经验记录在 [CODEX_NOTES.md](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/CODEX_NOTES.md)，后者优先级低于正式规则。

基座并非只有布局。它控制用户态、菜单/按钮权限、路由可访问性、多页签缓存、主题、请求安全处理及子应用生命周期。

| 关键文件 | 职责 |
| --- | --- |
| [main.js](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/itmg-dtk-base/src/main.js) | 注入全局对象、初始化用户数据、组件和样式，创建 Base Vue 实例 |
| [router/index.js](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/itmg-dtk-base/src/router/index.js) | 从微应用配置生成路由，验证登录及菜单权限，通过后挂载 |
| [BasicLayout.vue](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/itmg-dtk-base/src/page/BasicLayout.vue) | 主布局、菜单及业务容器 |
| [store/index.js](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/itmg-dtk-base/src/store/index.js) | 用户、菜单、按钮权限、页签、token 刷新等状态和动作 |
| [utils/axios.js](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/itmg-dtk-base/src/utils/axios.js) | 统一鉴权、刷新、SM3/SM4 处理、完整性签名、前缀、Blob 与错误分支 |
| [core/loadMicroApp.js](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/itmg-dtk-base/src/core/loadMicroApp.js) | qiankun 手动挂载、复用已挂载实例、按页签关闭情况卸载 |
| [shared/actions.js](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/itmg-dtk-base/src/shared/actions.js) | qiankun 全局状态，包含页签变化、Base window、图表配置等 |

[microApps.json](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/itmg-dtk-base/public/config/microApps.json) 已注册 `power-trade`，端口后缀 `16`、activeRule `/power-trade/`。开发入口由当前主机名和端口前缀计算；生产入口为同源 `/<app.name>/index.html`。当前注册表含 vpp、eem、asm、power-trade、carbon、AIF 和其他应用，不等于根目录每个工程都已经注册。

基座使用 `window._tsx.apirequest` 向子应用提供请求能力。注册微应用只解决加载入口，账号仍须获得对应菜单权限；详情路由、路由守卫、菜单选中和页签缓存需要一起考虑。

三主题代码集中在 [style/theme](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/itmg-dtk-base/src/style/theme)，Element 覆盖在 [style/element](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/itmg-dtk-base/src/style/element)，图表语义在 [themeConfig/chartConfig.js](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/itmg-dtk-base/src/themeConfig/chartConfig.js)。

| 基础主题变量 | dark | light | auroraBlue |
| --- | --- | --- | --- |
| 主色 `--color-primary` | #1a8dff | #06a299 | #42caff |
| 页面背景 `--background-body` | #061a3e | #f3f4f7 | #052449 |
| 主要文字 `--font-color` | 白色 90% | 黑色 90% | 白色 90% |
| 表头 `--table-header-background` | #113883 | #f0f5f5 | #113883 |

以上为基础主题值；局部选择器可能继续覆盖。auroraBlue 不能简单按 dark 的具体色值复制。

**三、power-trade：双模式交易应用**

工程位置：[power-trade](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/power-trade)。实际路由定义有 30 个业务路由，不计根重定向、兜底路由及独立登录路由；其中包括隐藏详情、历史兼容及当前隐藏的菜单项。

| 业务域 | 主要内容 |
| --- | --- |
| 总览 | 数据看板、经营概览 |
| 客户 | 客户、详情、设备绑定、分成合同、居间商 |
| 代理用电 | 负荷采集、历史用电、偏差分析 |
| 交易决策 | 省份及详情、电价预测、算法回测、负荷预测、报价 |
| 交易执行 | 策略、用电申报、交易日历、交易结果与持仓 |
| 复盘风控 | 复盘、预警、风险评估 |
| 结算与辅助信息 | 收益、偏差考核、日清分、月结算、营收统计、天气 |

[router/index.js](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/power-trade/src/router/index.js) 组合两种页面实现：普通列表/表单由 `configuredRoute`、`pageDefinitions.js`、`pages/business/index.vue` 驱动；复杂场景使用独立页面及相应 service。电价预测中心与算法回测中心当前共用 `AlgorithmCenter`，以 mode 区分，不能仅凭旧页面文件存在判断现行入口。

常用业务组件在 [components/business](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/power-trade/src/components/business)：PageShell、MetricGrid、TradeChart、PanelHeader、RecordFormDialog、RecordDetailDialog 等。后续新页面先检索这些实现及相近页面。

运行与请求路径：

```text
qiankun 模式
Base 登录/菜单/页签/主题 → power-trade 内容页
业务 service → powerTradeRequest → utils/request → Base 注入的 apirequest

独立模式
standalone-config.json → 场景/路由/鉴权/主题 → StandaloneApp
业务 service → powerTradeRequest → utils/request → 独立安全请求实例

数据源
页面 → 通用或专用 service → Mock provider / real provider
```

[main.js](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/power-trade/src/main.js) 保留 public-path、bootstrap/mount/update/unmount；运行时按 qiankun 标志选择路由、布局与守卫。[runtime.js](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/power-trade/src/runtime.js) 先加载独立场景，再创建路由和主题。

[standalone-config.json](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/power-trade/public/standalone-config.json) 当前 activeScenario 为 `local-static-auth`，另有 `local-permission-auth`、`local-debug-no-auth`、`subpath-static-auth`。开发环境 Mock 开启时会自动选择免登录调试场景；这不代表真实鉴权已通过。独立权限请求在 session 中使用 `systemType: power-trade`。

独立模式打包复用 Base 的主题、部分登录资源、图表配置和加密工具，所以开发/构建仍依赖相邻 Base 源码与依赖；构建完成后的运行环境不要求启动 Base。独立 access/refresh token 使用应用前缀键，但 `themeType`、`hgdata`、`refresh_expiresIn` 等兼容键仍共用，不能把所有浏览器存储都描述成完全隔离。

**power-trade 数据源与接口契约**

- [dataSource.js](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/power-trade/src/config/dataSource.js) 读取 `VUE_APP_POWER_TRADE_MOCK`；当前 development 为 true，production/test 为 false。
- [pageService.js](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/power-trade/src/services/pageService.js) 分流通用查询、详情、操作与日历；专用页面还有各自 service。
- [realProvider.js](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/power-trade/src/services/realProvider.js) 集中映射 endpoint、query/body、返回字段、单位和省份，当前体量较大，修改前应定位到具体业务分支。
- [api/powerTrade.js](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/power-trade/src/api/powerTrade.js) 统一添加服务前缀 `/vpp-power-trade-aux`；多数业务路径进一步包含 `/api/...`。
- 独立开发代理将业务服务转发到 `http://192.168.3.43:43305`；鉴权、用户、文件服务使用 `http://192.168.3.40:43080/api/`。本轮只核对配置，未验证这些远程地址的可达性。
- 省份为 GLOBAL、安徽、江苏、浙江、广东、山西。顶部省份是查询上下文，新建客户或预测的归属省份由表单明确提供；请求使用 `X-Province`，不能将请求头本身当成后端授权。
- 工作区最新改动通过 [operatorHeaders.js](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/power-trade/src/utils/operatorHeaders.js) 集中生成合法 `X-User`，移除了中文默认头；没有合法账号时省略，真实鉴权仍走统一 token 链。

Mock 数据主要使用版本化 localStorage；客户营业执照 Mock 附件存入 IndexedDB。新版算法流程目前可在 Mock 中完成回测、修正、版本保存、预测选用与结果快照。`VUE_APP_POWER_TRADE_ALGORITHM_VERSION_API` 默认关闭，开启只意味着尝试读取约定版本列表，不等于训练、重训或完整版本化预测能力已经实现。

**power-trade 当前进度与边界**

当前 feature-v1.0 工作区有 10 个已跟踪前端文件修改，另有 operatorHeaders、请求头验证脚本、后端交接文档、隔离验证工程及补丁等未跟踪项。本文梳理的是这些改动存在时的工作区，不是单纯的 HEAD。

资料索引：

- [BACKEND_INTEGRATION.md](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/power-trade/BACKEND_INTEGRATION.md)：按 C/L/D/A/P/Q/R/G 编号整理现状、前端兼容方案、后端缺口和验收条件。
- [BACKEND_OFFLINE_TEST_REPORT.md](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/power-trade/BACKEND_OFFLINE_TEST_REPORT.md)：2026-09-09 的离线编译、测试与修复结果，优先于较早报告理解最新进展。
- [BACKEND_IMPLEMENTATION.md](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/power-trade/BACKEND_IMPLEMENTATION.md)：首批后端快照修改与补丁交接，部分正文是早期阶段记录。
- [docs](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/power-trade/docs/vpp-power-trade-aux/docs)：OpenAPI、业务设计、联调导读及 HTML 原型；该目录被 Git 忽略。
- [backend-verification](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/power-trade/backend-verification/README.md)：隔离验证边界；不能替代正式后端父工程、数据库和真实权限测试。

历史报告记载：227 个业务 Java 源文件与 14 个实际 DTK 类型完成隔离编译，78 项 JUnit 通过，16 项前端请求隔离回归通过。这些为已有文档中的验证记录，本轮未重跑，也不表示线上部署或完整业务验收完成。

仍需保留的业务限制包括：客户申报量目前由平台总量均摊的 D01 数据源问题；新版算法训练/版本/快照契约缺口；真实文件归属、租户及省份权限；采集批次审计、幂等及数据库事务实测；结构化报价与完整收益归因。快照中已补部分 DTO、导入校验、采集和评估逻辑，但最新报告明确未部署、未执行迁移。

**四、天津项目 skill 清单与选用方式**

`.codex/skills` 下实际有四个项目 skill；根目录另有一个提交规范 skill。以下是本次阅读与盘点结果，未执行生成、安装或迁移脚本。

| Skill | 适用范围 | 核心内容 |
| --- | --- | --- |
| [vpp-ui-design-system](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/.codex/skills/vpp-ui-design-system/SKILL.md) | 天津项目 Vue2 页面、组件与 UI 变更 | 项目组件复用、Element 映射、三主题、列表/弹窗/详情模式及验证 |
| [vpp-figma-to-vue2](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/.codex/skills/vpp-figma-to-vue2/SKILL.md) | 从 Figma 实现天津 Vue2 界面 | 与 UI skill 联用，获取节点上下文，映射组件/主题，通过基座路径验证 |
| [vpp-dark-frontend](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/.codex/skills/vpp-dark-frontend/SKILL.md) | 将 VPP 深蓝视觉系统用于现有或新前端 | 跨框架 Token、管理后台/驾驶舱/混合布局、组件状态、ECharts、通用素材及视觉验收 |
| [create-vpp-microapp](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/.codex/skills/create-vpp-microapp/SKILL.md) | 创建与注册新的 Vue2 + qiankun 子应用 | 默认 EEM 模板，明确 standalone 模式，先 dry-run，保留生命周期/主题/请求/页签契约 |
| [aegispipe-commit](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/aegispipe-commit/SKILL.md) | 提交信息规范 | 中文类型标签、标题和正文长度、禁用字符及句式校验；位于根目录而非 `.codex/skills` |

UI skill 的四份主要 references 分别维护设计来源、组件映射、三主题 Token、页面组合模式；团队入口是 [docs/ui-design-system/README.md](/Users/dubbing/工作/恒实/code/天津虚拟电厂/vpp-tianjin-frontend/docs/ui-design-system/README.md)。dark/light 的 Figma 节点索引已在本地维护，auroraBlue 以代码为依据。本轮未访问 Figma，节点有效性未在线复核。

组件复用顺序是：现有成熟项目组件 → Element UI / 现有库及 Base 样式 → 局部业务组合。skill 明确区分成熟组件与空目录/占位文件，不能因名字像公共组件就直接推荐。主题值优先使用既有语义变量，业务页面不应全局重置 `.el-*`。

`vpp-dark-frontend` 与当前 ML 项目的关联最直接：它已经提供 React + Ant Design 5 适配说明、`react-antd.css`、Tailwind 映射、ECharts 预设、CSS/JSON/TS Token、管理页/驾驶舱预览、通用 SVG 与素材清单。还包括扫描、安装、适配器生成、验证四类脚本。

如果后续要求 ML 平台采用 VPP 风格，应保留 React/Ant Design 和业务链，先将已有 `--surface-*`、`--text-*`、`--brand-*`、ConfigProvider 主题映射到 VPP 语义，再处理布局、控件和图表。报告样式应继续由 `REPORT_CHART_THEME` 集中管理，并核对打印效果。此项是后续实施判断，本轮没有开始改版。

该视觉 skill 不用于复制交易业务、请求、路由、权限或生产素材。天津项目内部的三主题要求仍应以 AGENTS 和 UI skill 为准，不能用单一暗色模板替代。建立微应用与改视觉也是两类不同任务，当前并未提出创建或接入新微应用。

项目规则存在需结合上下文解释的地方：UI 参考中的“子应用不复制登录/布局”适用于基座内容模式；power-trade 与创建 skill 已明确支持完整独立模式。新增工程脚本的 name 校验目前仅允许小写字母和数字，不能直接用其默认校验重建含连字符的 `power-trade` 名称。根 AGENTS/部分参考的模块示例较早，未完整列出后加工程，应结合注册表和目标工程源码判断。

**五、验证现状与后续开发注意点**

本次实际检查了源码结构、关键路由/请求/主题链、环境开关、端口进程、Git 工作区及 skill 文件；没有运行天津业务服务、真实写接口、E2E、数据库迁移或部署。

ML 单测启动验证结果：

1. 默认 Node 14.17.3 下执行 `npm run test:unit`，Vitest 解析 `??=` 时失败，未进入用例。
2. 改用机器现有 Node 20.20.2 显式执行 Vitest，缺少 `@rolldown/binding-darwin-x64`，仍未进入用例。
3. 已安装 Vite 为 5.4.21，声明需要 Node 18+；Vitest 为 4.1.10，声明支持 Node 20 / 22 / 24+ 的相应范围。当前运行开发服务的 Node 18 能启动 Vite，并不意味着能跑 Vitest 4。

本轮未重装依赖、删除锁文件或切换用户默认 Node。README 的“26 文件 / 228 用例”属于工程说明，不能作为本次测试通过结论。天津旧工具链保留 node-sass 4，现有报告记录在 Node 14 下构建；请求头验证脚本另需 Node 20+。

已发现的配置或文档差异：

| 项目 | 观察 | 后续影响 |
| --- | --- | --- |
| ML Git | README 描述无 `.git` 的快照，但工作区现已初始化且尚无提交 | 首次提交前重新核对待跟踪内容 |
| ML ignore | `.gitignore` 当前仅有 `/node_modules`；末次核对 `.env.local` 已进入暂存区 | README 声称已忽略本地配置与事实不符；后续提交前应核对本地配置的跟踪边界 |
| ML health | Header 调用 `/health`，Vite 仅代理 `/api`、`/inference`、`/ws` | 开发环境健康信息可能落入 SPA 回退，环境标签需专项核对 |
| 天津 Base 代理 | power-trade 已注册，但 Base 开发代理列表未包含 `/vpp-power-trade-aux/`；子应用自身有此代理 | 挂载模式关闭 Mock 后，应核对实际请求 origin 与 Base 网关转发；本轮未验证该场景 |
| 天津运行 | 6616 正在监听，6600 未监听 | 不能用当前独立运行状态证明基座挂载可用 |
| 交易后端资料 | docs 快照及 OpenAPI 被 Git 忽略，补丁/交接文件另存 | 分享与提交时需明确包含源码快照还是补丁 |
| 历史验证记录 | 旧报告与新报告描述不同阶段 | 使用日期和明确验证范围判断，不将源码修改当成部署成功 |

后续改 ML 页面时优先定位 App、global.css、对应页面及 workbench 公共层；改交易业务时先定位路由/页面配置、页面 service、realProvider 和 OpenAPI；改主子应用行为时同步核对 Base 路由、权限、页签、挂载及请求链。所有修改前重新检查 Git 状态，保留用户已有改动。
