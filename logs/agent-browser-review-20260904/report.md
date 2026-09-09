# Dogfood Report: ML Learning Platform

| Field | Value |
|-------|-------|
| **Date** | 2026-09-04 |
| **App URL** | http://203.176.93.249:18081 |
| **Session** | ml-platform-review |
| **Scope** | 模型管理、统一结果页、主要导航与响应式表现 |

## Summary

| Severity | Count |
|----------|-------|
| Critical | 0 |
| High | 0 |
| Medium | 5 |
| Low | 2 |
| **Total** | **7** |

## Issues

### ISSUE-001: 1280px 桌面宽度下模型表格信息被挤成竖排

| Field | Value |
|-------|-------|
| **Severity** | medium |
| **Category** | visual / responsive |
| **URL** | http://203.176.93.249:18081/models |
| **Repro Video** | N/A（静态问题） |

**Description**

在 agent-browser 默认的 1280px 桌面视口中，侧栏占用约 220px 后，模型表格仍强制展示全部 8 列。数据集名称被压缩到每行约 2～3 个汉字，模型标识也只能依赖标签内部压缩，操作按钮拥挤在最右侧。页面虽然可操作，但扫描效率明显下降；这是常见笔记本宽度，不应退化成近似竖排文本。建议为低优先级列设置断点隐藏/合并，或给表格明确的横向滚动与固定关键列。

**Repro Steps**

<!-- Each step has a screenshot. A reader should be able to follow along visually. -->

1. 在 1280px 宽度打开模型管理页。
2. **观察：**“数据集”列被挤成多行竖排，表格的信息密度失衡。
   ![Result](screenshots/initial-model-management.png)

---

### ISSUE-002: “通用模型”仍暴露旧时序结果页，主操作会进入无效死路

| Field | Value |
|-------|-------|
| **Severity** | medium |
| **Category** | functional / navigation / information architecture |
| **URL** | http://203.176.93.249:18081/models → http://203.176.93.249:18081/ts/results |
| **Repro Video** | N/A（静态导航问题） |

**Description**

“通用模型”Tab 展示的实际内容是“时序预测记录”，与 Tab 名称不符，并且和左侧独立的“时序任务”模块形成重复入口。更严重的是，在没有任何记录时点击该 Tab 顶部的“结果可视化”，页面仍会跳转到不带任务 ID 的旧 `/ts/results` 路由，最终只显示“任务不存在或已被删除”，浏览器控制台同时记录一次 HTTP 400。这个面向所有用户可见的入口是确定的死路，也说明此前准备淘汰的旧结果页仍然挂在导航中。建议删除这个无参数入口；若通用模型就是时序模型，应统一命名并从具体任务行进入结果页。

**Repro Steps**

1. 打开模型管理，切换到“通用模型”。
   ![通用模型实际展示时序记录](screenshots/general-models.png)
2. 点击顶部“结果可视化”。
3. **观察：**跳转到 `/ts/results`，只显示“任务不存在或已被删除”。
   ![旧结果页死路](screenshots/general-results-route.png)

---

### ISSUE-003: 深度学习训练图表的 Y 轴刻度被容器裁切

| Field | Value |
|-------|-------|
| **Severity** | medium |
| **Category** | visual / data visualization |
| **URL** | http://203.176.93.249:18081/training/results?taskId=95776508-7d36-4e96-9b6b-cac37980e4ff&family=dl |
| **Repro Video** | N/A（静态问题） |

**Description**

在 1280px 桌面视口中向下查看深度学习“训练可视化”，损失曲线和过拟合观察图左侧的 Y 轴大数值刻度超出 ECharts 网格，首位数字被整齐裁掉。用户无法判断数量级，尤其“验证损失 - 训练损失”本来就有较大的负值。建议启用 `containLabel`，或根据格式化后的最大刻度动态增加 `grid.left`，同时对大数使用紧凑单位（万/百万或科学计数法）。

**Repro Steps**

1. 打开任一深度学习模型结果页。
2. 进入“训练可视化”并向下滚动到损失曲线。
3. **观察：**两个上方图表的 Y 轴标签左半部分被裁掉。
   ![Y 轴刻度被裁切](screenshots/dl-result-viz-lower-clean.png)

---

### ISSUE-004: 两类模型的关键训练配置均显示为缺失值

| Field | Value |
|-------|-------|
| **Severity** | medium |
| **Category** | content / data mapping |
| **URL** | 统一模型训练结果页（ML 与 DL） |
| **Repro Video** | N/A（静态问题） |

**Description**

统一页面的 config 适配仍有字段未接通：机器学习训练可视化把“验证集比例”显示为 `—`，深度学习训练可视化把“批大小”显示为 `—`。这两个值都是解释训练过程的基础参数，并不是可有可无的装饰信息。两个 family 各缺一个字段，说明页面框架虽然统一了，但后端 payload 到统一 view-model 的映射仍不完整。建议由适配层统一兜底读取任务配置，而不是只从可视化接口的某个局部字段读取。

**Repro Steps**

1. 进入机器学习模型的“训练可视化”。
2. **观察：**“验证集比例”为 `—`。
   ![机器学习配置缺失](screenshots/ml-result-viz.png)
3. 进入深度学习模型的“训练可视化”。
4. **观察：**“批大小”为 `—`。
   ![深度学习配置缺失](screenshots/dl-result-viz.png)

---

### ISSUE-005: 390px 窄屏下模型管理和结果页无法正常阅读

| Field | Value |
|-------|-------|
| **Severity** | medium |
| **Category** | visual / responsive |
| **URL** | http://203.176.93.249:18081/models 及统一结果页 |
| **Repro Video** | N/A（静态问题） |

**Description**

在 390×844 视口中，侧栏虽已折叠，但仍固定占用约 72px；顶部用户名被压成逐字竖排，结果页标题断成两行，模型标识溢出，Tab 和日志筛选区互相挤压。模型表格仅能看到任务名称的一部分，且没有明显的横向滚动提示或移动端卡片替代。页面元素仍可被 DOM 访问，但真实用户难以辨认和操作。建议移动端完全抽屉化侧栏、隐藏用户名文字，并为表格切换为关键字段卡片或提供明确横向滚动容器。

**Repro Steps**

1. 将视口设为 390×844，打开模型管理。
2. **观察：**用户名竖排，表格只露出首列且文本被截断。
   ![移动端模型管理](screenshots/mobile-model-management.png)
3. 打开任一机器学习结果页。
4. **观察：**标题、模型名、Tab 和日志工具栏在狭窄区域内重叠或截断。
   ![移动端统一结果页](screenshots/mobile-ml-result.png)

---

### ISSUE-006: MAPE 在同一深度学习结果中采用两种不同量纲

| Field | Value |
|-------|-------|
| **Severity** | low |
| **Category** | content / consistency |
| **URL** | 深度学习模型详情及统一结果页 |
| **Repro Video** | N/A（静态问题） |

**Description**

深度学习模型详情和结果页顶部把“验证 MAPE”显示为原始小数 `0.0118`，而同一任务的结果回测把 MAPE 显示为百分数 `1.15%`。两者在数值上基本对应，但量纲展示不一致，用户很容易把 `0.0118` 理解为 `0.0118%`。建议所有 MAPE 统一按百分比格式化并带 `%`，必要时在 tooltip 中保留原始比率。

**Repro Steps**

1. 打开深度学习模型详情，观察“验证 MAPE 0.0118”。
   ![详情中的原始比率](screenshots/dl-detail.png)
2. 进入同一任务“结果回测”，观察“MAPE 1.15%”。
   ![回测中的百分比](screenshots/dl-result-backtest-loaded.png)

---

### ISSUE-007: 日志时间语言不统一，图标按钮缺少可理解名称

| Field | Value |
|-------|-------|
| **Severity** | low |
| **Category** | content / accessibility |
| **URL** | 统一模型训练结果页 → 训练日志 |
| **Repro Video** | N/A（静态问题） |

**Description**

中文界面的日志时间显示为英文相对时间（如 `47h ago`）。同时，右侧五个日志工具按钮在可访问树中仅暴露为 `clock-circle`、`pause-circle`、`vertical-align-bottom`、`clear`、`download`，视觉上也只有图标；新用户和屏幕阅读器用户都无法直接知道它们分别代表时间格式、暂停跟随、滚到底部、清空筛选还是下载。建议本地化为“47 小时前”，并给每个按钮设置中文 `aria-label` 与 tooltip。

**Repro Steps**

1. 打开任一结果页的“训练日志”。
2. **观察：**日志左侧显示 `47h ago`，工具栏只有无文字图标。
   ![日志本地化与可访问名称](screenshots/dl-result-logs.png)

---
