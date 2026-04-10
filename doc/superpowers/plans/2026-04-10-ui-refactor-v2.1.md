# UI 重构设计文档 v2.1

日期：2026-04-10  
分支：`feat/ui-refactor-v2.1`

---

## 一、目标

1. 侧边栏导航重组——"训练管理"改为"机器学习"，将"结果可视化"纳入该组；深度学习"结果详情"改名"结果可视化"
2. 机器学习 & 深度学习训练监控页拆分为两层：任务列表（默认）→ 任务详情（?taskId=...）
3. 任务列表支持名称内联编辑、删除（确认弹窗）、翻页
4. 模型管理页：Tab 分页（机器学习模型 | 深度学习模型 | 通用模型），同一页面展示
5. 模型部署页：同样 Tab 结构

---

## 二、新侧边栏菜单树

```
仪表盘             /dashboard
数据管理           /data
机器学习           (group)
  ├─ 训练配置      /training/config
  ├─ 训练监控      /training/monitor        ← 默认：任务列表；?taskId → 详情
  └─ 结果可视化    /training/results        ← 原 /results，路由迁移
深度学习           (group)
  ├─ 模型配置      /dl/config
  ├─ 训练监控      /dl/monitor              ← 默认：任务列表；?taskId → 详情
  └─ 结果可视化    /dl/results              ← 原"结果详情"，仅改名
模型管理           /models                  ← Tab 内部分页
模型部署           /deploy                  ← Tab 内部分页
系统设置           /settings
```

**路由变更汇总：**

| 旧路由 | 新路由 | 变化说明 |
|--------|--------|---------|
| `/results` | `/training/results` | 移入"机器学习"组 |
| `/dl/results` | `/dl/results` | 路径不变，页面改名 |
| `/training/monitor` | `/training/monitor` | 行为变化：新增列表层 |
| `/dl/monitor` | `/dl/monitor` | 行为变化：新增列表层 |

---

## 三、任务列表页设计（ML 监控 & DL 监控复用同一模式）

### 3.1 页面结构

```
[ 页面标题 ]                              [ 新建训练 ] 按钮
─────────────────────────────────────────────────────────
Table: 任务列表
  列：任务名称(可内联编辑) | 模型 | 状态 Tag | 进度 | 创建时间 | 操作
  操作列：[ 查看 ]  [ 删除 ]
─────────────────────────────────────────────────────────
Pagination: 每页 10 条，总数 N
```

### 3.2 Table 列定义

| 列 | 字段 | 宽度 | 说明 |
|----|------|------|------|
| 任务名称 | `name` | ~180px | 双击进入内联编辑（`EditableCell`），失焦/回车保存，Esc 取消 |
| 模型 | `model_type` | ~120px | 纯文本显示 |
| 状态 | `status` | ~90px | `<Tag color=...>` |
| 进度 | `progress` | ~120px | `<Progress percent={...} size="small" />` |
| 创建时间 | `created_at` | ~160px | 格式化显示 |
| 操作 | — | ~120px | `查看` (Primary Link) + `删除` (Danger) |

### 3.3 交互细节

- **"新建训练"** → 导航到 `/training/config`（ML）或 `/dl/config`（DL）
- **"查看"** → 导航到 `/training/monitor?taskId={id}` 同一页面切换到详情视图
- **内联编辑任务名称：**
  - 双击名称单元格显示 `<Input>` + 确认/取消图标
  - 保存时调用 `PATCH /api/training/{id}/name`
- **删除：**
  - 点击"删除"弹出 `Modal.confirm`
  - 确认后调用 `DELETE /api/training/{id}`
  - 成功后刷新列表
- **列表自动刷新：** RUNNING 任务每 5 秒轮询一次列表（仅刷新状态+进度列）

### 3.4 URL 路由模式

```
/training/monitor          → 显示列表视图
/training/monitor?taskId=X → 显示任务 X 的详情视图（原 DLMonitor/TrainingMonitor 内容）
```

详情视图顶部增加 `← 返回列表` 按钮，点击 `navigate('/training/monitor')`

---

## 四、后端变更

### 4.1 数据库新增字段

**TrainingTask**（`app/models/database.py`）：
```python
name: Mapped[str | None] = mapped_column(String(200), nullable=True)
```

**DLTrainingTask**：
```python
name: Mapped[str | None] = mapped_column(String(200), nullable=True)
```

新建任务时 `name` 默认为 `f"{model_type}_{task_id[:8]}"` 格式。

### 4.2 新增 API 端点

**机器学习训练（`/api/training/`）：**

```
GET    /api/training/list              分页查询任务列表
                                       query: page(1) + page_size(10)
                                       response: { tasks: [...], total: N }

PATCH  /api/training/{task_id}/name   修改任务名称
                                       body: { name: str }

DELETE /api/training/{task_id}        删除任务（仅允许非 RUNNING 状态）
                                       response: { ok: true }
```

**深度学习（`/api/dl/`）：**

```
GET    /api/dl/list                   已有，补充 page/page_size 分页参数
                                       response 补充 total 字段

PATCH  /api/dl/{task_id}/name         修改任务名称
                                       body: { name: str }

DELETE /api/dl/{task_id}              删除任务（仅允许非 RUNNING 状态）
```

### 4.3 Schema 变更

**ML 任务 Schema** 新增：
```python
class TaskListItem(BaseModel):
    task_id: str
    name: str | None
    model_type: str
    status: str
    progress: float
    created_at: datetime

class TaskListResponse(BaseModel):
    tasks: list[TaskListItem]
    total: int
```

**DL 任务 Schema** 的 `DLTaskListResponse` 同样补充 `total` + `name`。

---

## 五、模型管理页 Tab 设计（`/models`）

### 5.1 Tab 结构

```
[ 机器学习模型 | 深度学习模型 | 通用模型 ]
──────────────────────────────────────────
Tab: 机器学习模型
  数据来源：GET /api/models  (现有，返回 ML 模型列表)
  Table 列：模型名 | 算法 | 任务类型 | 创建时间 | 状态 | 操作

Tab: 深度学习模型
  数据来源：GET /api/dl/models-trained  (新端点，返回已训练 DL 模型)
  Table 列：模型名 | 架构 | 任务类型 | 创建时间 | 状态 | 操作

Tab: 通用模型
  占位内容：「Google TimesFM 等通用时序模型，敬请期待」
  预留集成接口
```

### 5.2 后端新端点

```
GET /api/dl/trained-models            返回已完成(SUCCESS)的 DL 训练任务列表
                                       作为"深度学习模型"的数据来源
```

---

## 六、模型部署页 Tab 设计（`/deploy`）

```
[ 机器学习部署 | 深度学习部署 | 通用模型部署 ]
──────────────────────────────────────────────
Tab: 机器学习部署
  保留现有 ModelDeploy 页面内容

Tab: 深度学习部署
  Select 选择已完成 DL 训练任务 → 部署（占位/后续实现）

Tab: 通用模型部署
  占位：「通用模型部署功能即将上线」
```

---

## 七、实施任务清单

| # | 任务 | 文件 | 难度 |
|---|------|------|------|
| T1 | 后端：TrainingTask + DLTrainingTask 新增 `name` 列 | `database.py` | 低 |
| T2 | 后端：ML `GET /list`（分页）、`PATCH /name`、`DELETE` | `training.py` + `training_service.py` | 中 |
| T3 | 后端：DL `GET /list` 加分页+total、`PATCH /name`、`DELETE` | `dl.py` + `dl_service.py` | 中 |
| T4 | 前端：`TrainingMonitor.jsx` 重构为列表/详情双视图 | `TrainingMonitor.jsx` | 高 |
| T5 | 前端：`DLMonitor.jsx` 重构为列表/详情双视图 | `DLMonitor.jsx` | 高 |
| T6 | 前端：Sidebar 改名 + 路由调整（结果可视化移位） | `Sidebar.jsx`, `App.jsx` | 低 |
| T7 | 前端：`Results.jsx` 路由从 `/results` 改为 `/training/results` | `App.jsx` + 内部导航 | 低 |
| T8 | 前端：`ModelManagement.jsx` 增加三 Tab 分页 | `ModelManagement.jsx` | 中 |
| T9 | 前端：`ModelDeploy.jsx` 增加三 Tab 分页 | `ModelDeploy.jsx` | 中 |
| T10 | 后端：`GET /api/dl/trained-models` 端点 | `dl.py` | 低 |

**实施顺序：** T1 → T2 → T3 → T10 → T6/T7（同步）→ T4 → T5 → T8 → T9

---

## 八、前端可复用组件规划

```
src/components/
  TaskListTable.jsx     通用任务列表表格（ML/DL 共用，通过 props 区分数据源和操作 API）
  EditableCell.jsx      内联编辑单元格（双击→Input，确认/取消）
```

`TaskListTable` Props：
```js
{
  fetchList: (page, pageSize) => Promise<{tasks, total}>,  // 数据获取函数
  onRename: (taskId, name) => Promise<void>,               // 改名
  onDelete: (taskId) => Promise<void>,                     // 删除
  onView: (taskId) => void,                                // 查看（navigate）
  newTaskPath: string,                                     // 新建任务路由
}
```

---

## 九、注意事项

1. 删除任务时，`RUNNING` 状态禁止删除，后端需返回 422 并在前端显示提示
2. 任务名称编辑长度上限 100 字符，前端加 `maxLength` 约束
3. `/training/results` 路由变更后，老链接（如 `/results`）应加 301 重定向（`<Navigate to="/training/results" />`）
4. DL 监控详情视图内"返回监控"按钮原来跳到 `/dl/monitor?taskId=X`，改为 `navigate('/dl/monitor')`（返回列表）
5. `DLResults.jsx` "返回监控"按钮同上，改为返回 `/dl/monitor`（列表）
