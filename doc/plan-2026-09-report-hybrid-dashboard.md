# 规划：报告可读性 / 混合策略 / 仪表盘

日期 2026-09-08 · 分支 `feat/v3-unified-workflow` · 基线 `8966625`（本地 = 远程 = 云端）

三个工作流互不依赖，可并行派发。每个工作流的**验收标准**是我验收时逐条核对的清单，写得可机器检查就按机器检查。

---

## 同步协议（所有工作流共用）

云端是测试环境，但它跑的是**构建镜像**，不是挂载源码。"热更新"只能这样做：

1. 本地改完 → `git commit`（每个可验收的最小改动一个提交）
2. 单元测试在云端一次性容器跑：`docker run --rm -v <改动文件>:/app/... -w /app docker-backend python -m pytest tests/ -q`
3. 部署：`git bundle create` → `scp` → 云端 `git merge --ff-only` → `docker compose build <backend|frontend>` → `up -d --no-deps <svc>`
4. 云端 HEAD 必须始终等于本地 HEAD。**禁止**直接 `docker compose cp` 覆盖运行容器里的文件当作交付——那会让云端和 git 分叉，下次构建就丢。

数据红线：数据集 `电力负荷预测数据.csv`（87312×35）不得删除或重建。`.mcp.json`、`docker/.env`、`.deploy_secrets*` 不得打印、不得提交。

---

## 工作流 A · 报告（图文并茂版）

### 现状诊断（已读：`report_facts.py`、`report_template.py`、三个模板、`AiReportModal.jsx`、`ai_report_service.py` 图表构建段、`trainer.py`）

报告"不如人意"不是数字错——数字现在是对的——是三件事：

1. **它不肯下结论。** 结论第一句"当前不存在可直接认定的全局最优模型"（`report_facts.py:309`），而 mlp_dl 132 对 xgboost 72 差 82%，任何口径差异都盖不住。0054f7f 把"口径不同不能直接比"做过了头。
2. **图和文各说各的。** 总报告唯一一张图是 7 个模型的损失曲线叠在一起（`ai_report_service.py:1040`），正文从不提它；分报告的图由模板放在固定位置，但图注是泛泛的说明而非对这张图的解读。库里有目标列直方图分箱、每个 run 的 SHAP 前 8、DL 的 500 点留出预测——总报告一张都没画。
3. **免责声明占了结论的位置。** 结论段 = 领先句 + 未做最终评估 + 就绪度 + 下一步 + 时序泄漏 + 重复 run，5 条警告之后读者才知道有没有结论；封面摘要又把这段原文复制一遍（`aiReportViewModel.js:40`）；"评估就绪度 60/100"是封面最大的元素，它只是三项 40/30/30 的检查表。最重要的发现——历史 run 用随机 KFold 训练含滞后特征的时序数据，72.47 是泄漏后的乐观分——被放在最后一段。

### 设计原则（这是"图文并茂"和"图文堆砌"的分界）

1. **每张图紧挨着读它的那段话之前**；那段话第一句就是对图的解读。
2. **图注是解读，不是说明**，由后端算出：写"xgboost 与 lightgbm 的误差条重叠"，不写"每折单独的得分与均值线"。
3. **正文不放表。** 表里的字段全部进图的悬停窗：排行榜 → 图 1 悬停；折表 → 图 2 悬停；字段表 → 图 4 悬停（列名列表）；SHAP 表 → 图 5 本身。只有逐列数据概况、参数设置两张宽表留在附录折叠区。
4. **可读性硬指标**：每句 ≤ 2 个数字；每段 ≤ 3 句；数字取到读者需要的精度（72.5 而不是 72.4673，0.8% 而不是 0.81%，精确值在悬停里）；不用"误差量级为目标列均值的"这类套话，说"相当于把 8897 的平均负荷预测偏差 0.8%"。
5. **图由前端统一渲染。** 后端不再拼 ECharts option，只出**语义图规格**：`{kind: "hbar"|"dots"|"hist"|"stacked"|"lines", title, caption, series, tooltip_fields, reference_lines}`。前端一个 `renderReportChart(spec)` 把它映射成 ECharts option，主题（配色 / 字号 / 边距 / 悬停格式 / 高度）只在这一处定义。现有 `withReportChartDefaults`（`AiReportModal.jsx:112`）扩成这个渲染器。这样图的样式问题只需改一处，不再像现在那样一张图一张图地补 `grid.left`。
6. 每节最多两张图；只有"实际 vs 预测 + 残差分布"允许并排；一图一行，占满正文宽度，高度 280–320px。
7. 豆包只写四处：总报告"下一步"一句、构造特征在解决什么两句、分报告"依赖在什么场景失效"一句、DL 收敛解读两句。其余全部计算。
8. 总报告与分报告**同一套渲染**：正文用 `{{chart:id}}` 标记，前端一个 `ReportDocument` 组件按标记就地渲染；`report_blocks` 与正文表格退役。

### 总报告 · 阅读顺序（样稿见 `doc/report-mock-overview.md`，数字全部取自任务 0bc692d9）

| 节 | 图（悬停内容） | 文 |
|---|---|---|
| 封面 | 无 | 一行元信息 + **一句判定**大字。没有分数卡 |
| 结论 | **图1 七个模型的误差**：横向条形，按口径着色，误差条 = 折间波动，虚线 = 均值 1%（悬停：模型·RMSE·占均值·R²·折间标准差·验证方式） | 两段：谁赢、差多少、谁落后 / 最大风险一条 + 下一步`<<豆包>>` |
| 模型差距 | **图2 五折散点**：每个 CV 模型一行 5 点 + 均值刻度（悬停：折·RMSE·MAE·R²） | 一段：哪些重叠、哪个分开 + `<<豆包>>` |
| 数据集 | **图3 负荷分布**：直方图 + 均值/四分位线（悬停：区间·样本数·占比）<br>**图4 35 列是怎么来的**：堆叠横条（悬停：该类别全部列名） | 规模一句 + `<<豆包>>`构造特征在解决什么 |
| 特征依赖 | **图5 最依赖的 8 个特征**：SHAP 横向条形（悬停：特征·SHAP·占首位 %） | 一段：集中度 + `<<豆包>>`部署风险 |
| 附录（折叠） | 无 | 逐列数据概况、参数设置两张宽表 |

### 分报告 · 阅读顺序

| 节 | ML（树模型） | DL |
|---|---|---|
| 结论 | 三句：口径内名次与差距 / 稳定性判定 / 部署风险 | 三句：名次与差距 / 收敛判定 / 口径说明 |
| 训练过程 | **图 各折 RMSE**（悬停：折·RMSE·MAE·R²，折表退役）→ 读图段 | **图 损失曲线**（对数轴）+ 最优轮竖线 + 早停区阴影（悬停：轮·训练损失·验证损失·验证 RMSE）→ 读图段。学习率图仅在变动时出现 |
| 训练结果 | **图 实际 vs 预测 ‖ 残差分布**（并排；需 P1）→ 指标段（RMSE/MAE 比值现在有图可看）→ **图 SHAP 前 8** → 依赖段 + `<<豆包>>` | **图 实际 vs 预测 ‖ 残差分布**（已有数据）→ 指标段。无 SHAP |

### 前置条件

- **P1** `trainer.py:88` 已算出 `y_val_pred`，照 `dl_trainer.py:376` 的方式存一份连续有序的尾段（≤500 点）到 `metrics.val_scatter`。老 run 没有，需重跑本任务 7 个 run（ML 约 2 分钟）。
- **P2** `columns_info.*.histogram` 的 `counts` / `bin_edges` 是 JSON 字符串（`_compact_value` 深度截断所致），facts 层 `json.loads` 后使用；分箱只保留目标列。
- **P3** 新增图构建器（全部基于现有数据）：`leaderboard_bars`、`fold_dots`、`target_hist`、`field_composition`、`shap_bars`、`residual_hist`。删除 `training_curves`（多模型叠加）。

### 分阶段与验收门

**阶段 1 · 后端**（图构建器 + facts + 三个模板 + P1/P2/P3；`report_blocks` 停止生成，改为正文内标记）
- [x] `POST /v3/tasks/{id}/ai-report` 返回的 `markdown` 含 5 个 `{{chart:…}}` 标记、**0 个** markdown 表格（`^\|` 行）；每个标记在 `charts` 里有同 id 的语义规格（含 `kind`、`tooltip_fields`），不再是 ECharts option
- [x] 每张图规格的 `tooltip_fields` 覆盖原来对应表格的全部列
- [x] 可读性：总报告正文每句 ≤ 2 个数字、每段 ≤ 3 句（脚本检查）
- [x] 结论第一句含具体模型名与百分比；不含"不存在""无法认定"；结论段 ≤ 3 句
- [x] 每张图的 `description` 含至少一个来自数据的模型名或数字（解读，不是说明）
- [x] 7 个分报告 markdown 不匹配 `不值得|建议|应当|优先`
- [x] 后端 `tests/` 全过；云端一次性容器跑

**阶段 2 · 前端**（`ReportDocument` 统一渲染；封面改版；附录折叠；导出/打印覆盖新图）
- [x] 总报告和分报告用同一组件渲染；`report_blocks`、正文 `TableBlock` 相关代码删除；附录用折叠面板放两张宽表
- [x] 所有图经 `renderReportChart(spec)` 渲染，后端不再出现 `"grid"`/`"nameGap"` 等 ECharts 字段（`grep -rn '"grid"' report_*.py ai_report_narrative.py` 零命中）
- [x] 悬停窗字段与规格一致（我在浏览器逐图悬停核对）
- [x] 封面无 60/100 大字；封面判定句 ≠ 结论第一段
- [x] 每张图上方 ≤ 1 行空白、下方紧跟读图段（我在浏览器量）
- [x] 打印/导出 PDF 含全部新图（走 ffe9aa2 的离屏文档）
- [x] vitest 全过、build 成功

**阶段 3 · 数据与整体验收**
- [x] 重跑本任务 7 个 run，ML run 的 `metrics.val_scatter` 存在
- [x] 重新生成报告；我在浏览器逐页核对总报告 5 图 + 7 个分报告
- [x] 云端 HEAD == 本地 HEAD

## 工作流 B · 混合策略

### 现状诊断

- 前端 `ModelConfigTabs.jsx:157 submitMixed`：提交 `strategy_type: 'baseline', model_family: 'mixed', selected_models: [...]`
- 后端 `grep hybrid|mixed` 在 services 里**零命中**；`model_family` 只是 `training_plans` 表上的一个标签字段
- 所以"混合策略"= **把 ML 和 DL 模型放进同一个批次各训一遍**。既不并联也不串联，没有任何融合逻辑

用户直觉里的"并联"已经存在：`ensemble_service.py` + `ensemble_fusion.py` 是部署期的加权融合（模型部署 → 多模型部署）。"串联"（stacking：底层模型出 OOF 预测 → 元学习器）代码里完全没有。

### 决策

| 选项 | 工作量 | 建议 |
|---|---|---|
| 实现 stacking | 新训练路径 + OOF 预测存储 + 元学习器 + 部署路径，≥ 1 周 | **不做**——不是"简单落地" |
| 重命名为「多模型对照」+ 指向多模型部署 | 半天 | 若「调参策略」tab 不支持跨族多选，选这个 |
| 删除该 tab | 半天 | **推荐**——若「调参策略」tab 的 `ExperimentBatchForm` 已支持跨族多选，则「混合」完全冗余 |

执行者第一步先确认 `ExperimentBatchForm` 是否支持一次选多个跨族模型，据此二选一。后端 `model_family='mixed'` 保留合法（历史数据用到）。

### 验收标准

- [x] 模型配置 step 不再出现"混合策略" tab（或已重命名且 Alert 文案指向"模型部署 → 多模型部署"做融合）
- [x] 仍能在一个批次里同时启动 ≥1 个 ML + ≥1 个 DL 模型（通过保留的入口）
- [x] 历史批次 `报告验证-混合` 在编排进度中仍正常显示
- [x] `TrainingPlans.jsx` 里 `mixed` 的筛选/标签与新命名一致
- [x] 前端测试全过；云端 HEAD == 本地 HEAD

---

## 工作流 C · 仪表盘

### 现状诊断

`Dashboard.jsx:292-298` 读的是 `/training/list`（旧 `TrainingTask` 表）和 `/models/list`（旧模型表）。云端库里：

| 表 | 行数 | 属于 |
|---|---:|---|
| TrainingTask | 4 | 旧流水线（已废弃页面 `/training/*`） |
| ModelingTask | 1 | V3 |
| ExperimentRun | 7 | V3 |
| AIReportArchive | 24 | V3 |

仪表盘的"任务 4 / 成功 N / 模型性能对比 / 近期活跃度"全部来自那 4 条旧记录。用户现在做的每一件事都在 V3，仪表盘一个都看不见——"参数不太对"是因为它在统计另一个宇宙。

### 决策

**推荐：先删，不重建。** 一个显示错误数字的仪表盘比没有仪表盘更糟。登录后落地页改为「建模 → 任务列表」。

若之后要重建，数据源已经齐备（`GET /v3/tasks`、`GET /v3/runs`、`/v3/tasks/{id}/leaderboard`），四张卡应是：建模任务数 / Run 数（成功·失败·运行中）/ 已部署模型数 / 数据集数；一张"最近 Run"表；一张"各任务最优模型"表。不要"模型性能对比"柱状图（跨任务的指标不可比）和"近期活跃度"折线（伪数据）。这是**另一个**工作流，不在本次范围。

### 验收标准

- [x] 侧边栏无"仪表盘"入口；`/dashboard` 路由 302 → `/v3/tasks`（或任务列表实际路径）
- [x] 登录成功后落地到任务列表
- [x] `Dashboard.jsx` 及仅被它引用的组件/样式已删除（`grep -rn Dashboard src/` 零命中）
- [x] 旧 API `trainingApi.listTasks / modelApi.listModels` 若已无其他调用方，一并删除；若仍有调用方，保留并在本文附录列出调用方
- [x] 前端 build 成功、测试全过；云端 HEAD == 本地 HEAD

---

## 派发方式

三个工作流各一个子代理，各自独立分支不需要——都在 `feat/v3-unified-workflow` 上顺序 ff 即可，但**每个代理只碰自己工作流的文件**：

| 工作流 | 允许改动的路径 |
|---|---|
| A 报告 | `ml_platform/app/services/report_*.py`、`report_templates/**`、`ai_report_service.py`（仅 headline_metrics 相关）、`ml_platform_web/src/components/workbench/AiReport*.jsx`、`aiReportViewModel*.js`、`RunReportPanel.jsx`、`styles/global.css`（仅 `.ai-report-*`）、对应测试 |
| B 混合 | `ModelConfigTabs.jsx`、`ExperimentBatchForm.jsx`、`TrainingPlans.jsx`、对应测试 |
| C 仪表盘 | `pages/Dashboard.jsx`、路由表、`Sidebar.jsx`、`services/api.js`（仅删除无调用方的函数）、对应测试 |

A 最重，先派；B、C 各半天，可与 A 并行。三个都完成后由我按上面的清单在浏览器 + 测试逐条验收。


---

## 验收记录 · 2026-09-09

云端 HEAD = 本地 HEAD = `227f3c6`。B（多模型对照，重命名路线）、C（仪表盘删除）、A（报告：后端 `report_charts.py` 语义规格 + 前端 `reportCharts.js`/`ReportDocument` 统一渲染）均已合并部署，最终归档 `91ce2217`（18 Run / 5 种模型）在浏览器逐项核对：封面一行元信息 + 判定句、无分数卡；总报告 5 图 0 表；hbar 整行悬停出 tooltip（字段 = `tooltip_fields`）；树模型分报告有"实际值 vs 预测值（交叉验证末折）"并排残差图；附录折叠。

验收中额外修掉的四处（均有测试）：worker 镜像从未重建导致训练走旧代码；选型模式下 `X_val=None`，树模型改用末折外样本预测并标注来源；`val_scatter_source` 与 `model_count` 分别被 16 键上限和 top_k 榜单截断；hbar tooltip 由 item 改 axis 触发。

**遗留观察（未改，待定）**
1. 封面判定句"这批模型是随机切分训练的"：现在 18 个 run 里 9 个已是时间序列切分，但领先的 72.47 仍来自旧的随机切分 run。措辞应改为"领先的模型是随机切分训练的"更准确。
2. 分报告上限 8 篇按名次取，18 个 run 后 DL 模型全部排在 8 名之外，报告里不再出现任何 DL 分报告。要么放宽上限，要么"每种模型至少一篇"。
