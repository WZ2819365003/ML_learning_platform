# 建模任务报告（Markdown）— 设计

- 日期：2026-07-22
- 分支：`feat/v3-unified-workflow`
- 里程碑：M3 第一块

## 要解决的问题

一次建模任务跑完之后，结果散落在平台各处：leaderboard 在一个页面，最终评估在另一个，
超参在 run 详情里，SHAP 在第三个 tab。想把「这个模型是怎么来的、为什么可信」讲给
别人听，只能靠人工截图拼凑。

报告把这些收敛成一份可传阅、可存档的产物。

## 受众

两类，一份报告同时服务：

| 受众 | 关心什么 | 对应章节 |
|---|---|---|
| 外部（导师 / 甲方 / 领导） | 做了什么、结果多好、为什么可信 | 上半部「结论」 |
| 自己（几个月后回溯） | 数据集版本、超参、评估口径、能否复现 | 下半部「技术附录」 |

## 形态与产出

**后端出 Markdown，前端渲染补图。**

- 后端：`GET /api/v3/tasks/{task_id}/report.md` → `text/markdown`，
  带 `Content-Disposition: attachment` 可直接下载
- 前端：任务详情页「导出报告」入口，渲染 md 正文，并用**现有 ECharts 组件**
  在对应锚点补图；需要交付时浏览器打印成 PDF

**md 本身是纯文本，不含内嵌图片。** 报告要可 diff、可 grep、可直接粘进任何文档，
base64 图片会毁掉这三点。

### 被否掉的替代方案

- **后端直接生成自包含 HTML**：逼着服务端引入 matplotlib 只为画静态图，
  而前端已有更好的 ECharts 实现。重复建设。
- **前端全量生成**：必须打开页面手动导出，不能被脚本调用，且要把 ECharts
  （~700KB）内联进产物。

## 生成时机

**仅在 finalize 之后。**

未 finalize 时返回 **409**，body 给出明确指引（"该任务尚未执行最终评估，
请先在任务详情页执行「最终评估」"），而不是返回一份半成品报告。

理由：报告的核心价值是给出确定结论。未 finalize 时封存测试集尚未开启，
没有 `final_test_*`，报告只能展示 selection 指标——而那恰恰是最容易被误读成
"模型性能"的数字。

判定依据：`modeling_task_service.task_final_evaluation_state(task)` 返回的
`state == "FINALIZED"`（存储于 `ModelingTask.config[FINAL_EVALUATION_CONFIG_KEY]`）。

## 报告结构

### 上半部 · 结论

1. **一句话结论** — 选定模型 + 目标指标在封存测试集上的值
2. **任务概览** — 数据集名称、样本量、任务类型、目标列、目标指标与方向
3. **最终评估** — `final_evaluation.final_metrics`，标注"封存测试集，全程仅开启一次"
4. **候选对比** — leaderboard top-10

### 下半部 · 技术附录

5. **评估方法** — CV 折数、train/test 划分比例、随机种子、防泄漏说明
6. **冠军超参** — 冠军 run 的完整 `params`
7. **特征重要性** — `run.metrics["shap_importances"]` top-10
8. **复现信息** — `dataset_version_id`、`winner_run_id`、`evaluation_id`、
   平台版本、报告生成时间（UTC）

## 关键约束：selection 与 final 指标不得并排

**这是本设计最重要的一条。**

- leaderboard 上是 `selection_cv_mean_*`（交叉验证均值，用于**选择**模型）
- 最终评估是 `final_test_*`（封存测试集，用于**报告**性能）

两者口径不同，数值通常也不同。如果并排放进同一张表，读者必然拿它们比大小，
得出"模型在测试集上掉点了"这类错误结论——而这正是平台 B0/B1 评估完整性设计
要防的事。

因此：

- 第 3 节与第 4 节**物理分开**，不共表
- 第 4 节表头明确标注：「以下为模型选择阶段指标（交叉验证均值），
  **不可与最终评估结果直接比较**」
- 第 3 节标注：「封存测试集结果，全程仅开启一次」

这条要有专门的回归测试。

## 实现

新增 `app/services/report_service.py`：

```python
async def build_task_report(db, task_id: str) -> str:
    """返回完整 Markdown 文本。未 finalize 时抛 HTTPException(409)。"""
```

**纯函数特性是有意的**：不碰 HTTP、不写文件、不渲染，输入 task_id 输出字符串。
测试可以直接对返回的 md 文本断言，不需要起服务、不需要 mock 响应对象。

路由挂在既有的 `app/api/routes/modeling_tasks.py`（`prefix="/v3/tasks"`）。

### 数据来源（均为既有接口，不新增查询逻辑）

| 章节 | 来源 |
|---|---|
| 任务概览（名称/任务类型/目标列/目标指标） | `ModelingTask` 列 |
| 任务概览（数据集名称/样本量） | `ModelingTask.dataset_name`；样本量取自关联 `Dataset.row_count` |
| 最终评估 | `task_final_evaluation_state(task)["final_metrics"]` |
| 候选对比 | `modeling_task_service.task_leaderboard(db, task_id, top_k=10)` |
| 冠军 run | `task_final_evaluation_state(task)["winner_run_id"]` 定位，**不是**取 leaderboard 第一名 |
| 冠军超参 | 冠军 `ExperimentRun.params` |
| 特征重要性 | 冠军 `ExperimentRun.metrics["shap_importances"]` |
| 评估方法（CV 折数/划分比例/随机种子） | 冠军 run 的 `params` 与 `search_meta` |
| 平台版本 | `app.main` 中的 `version`（当前 2.0.0），不硬编码 |

> 冠军 run 必须由 `winner_run_id` 定位而非 leaderboard 第一名：leaderboard 按
> selection 指标排序，而 finalize 时的冠军是当时冻结的那一个。若之后又有新
> 实验跑出更高的 selection 分数，两者会不一致——报告必须忠实反映**已 finalize
> 的那次决策**，否则报告内容会随后续实验漂移。

## 约定

- **语言**：简体中文（与平台 UI 一致，见 CLAUDE.md）
- **候选对比**：只列 top-10。完整榜单指向平台 leaderboard 页面，
  避免报告里塞几十行噪音
- **数值格式**：指标保留 4 位小数；缺失值显示 `—` 而非留空或 `null`

## 降级行为

报告不能因为某一块数据缺失就整体失败。缺失时该节显示明确说明，其余照常输出：

| 缺失项 | 行为 |
|---|---|
| 无 SHAP（未计算或计算失败） | 第 7 节写「该模型未生成特征重要性」 |
| 无 `dataset_version_id` | 第 8 节该行显示 `—` |
| leaderboard 为空 | 第 4 节写「无其他候选」 |
| 冠军 run 已被删除 | 整体 409，说明数据不完整 |

## 测试

| 测试 | 断言 |
|---|---|
| 未 finalize | 抛 409，且响应体含指引文案 |
| 已 finalize | md 含最终指标数值与冠军模型名 |
| **口径不混淆** | selection 指标与 final 指标不出现在同一个 Markdown 表格内；第 4 节表头含警示文案 |
| 无 SHAP | 不抛异常，第 7 节含降级文案 |
| 无 dataset_version_id | 不抛异常，显示 `—` |
| leaderboard 为空 | 不抛异常 |

## 不在本次范围

- 报告存档到 MinIO（按需再加；finalize 后内容是确定的，即时生成即可）
- PDF 服务端渲染（交给浏览器打印）
- 多任务对比报告
- 报告模板自定义
