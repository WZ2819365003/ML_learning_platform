# 技术债清单

记录已识别但**故意暂不偿还**的架构问题，避免下个人重新踩坑。每条债务包含：
1. 现状 — 现在是什么样
2. 期望 — 为什么这是债
3. 真实工作量 — 不是"看着像"，是"真做完"
4. 触发条件 — 什么情况下必须还

---

## TD-1: V3 提交链路绕过 Scheduler 抽象 —— **已基本偿还（v3.3.x / M2a–M2c）**

> 本条原文写于 v3.3.1，描述"99% 流量绕过 Scheduler、只有 retry 端点走 B 路径"。
> 该描述**已不成立**，保留标题作为历史索引，内容改写为当前状态。

### 已偿还

全部提交路径现在都经 `get_scheduler(kind).submit(platform_task_id)`：

| 路径 | 位置 |
|---|---|
| ML 训练 | `training_service.py:554` |
| DL 训练 | `dl_service.py:614` |
| 调参并发批次 | `tuning_service.py:1032` |
| SHAP 自动触发 | `tuning_service.py:1344` |
| 贝叶斯逐 trial | `tuning_service.py:1520` |

配套完成：
- **worker 容器已进生产编排**（`docker-compose.yml` 的 `worker` 服务），
  与 backend 共享 `ml_backend_storage` 卷
- **DAG gate 已生效**（`scheduler._gate_upstream`），不再是"预留字段"
- **写回责任已从调用方挪到统一入口** `run_writeback.complete_platform_task`
  （M2c）：终态由条件 UPDATE 的 rowcount 仲裁，Run 与 Task 同事务写入
- **日志镜像**已挪进该入口的 post-commit 尾巴
- **错误传播**：新增 `experiment_runs.error_message`，worker 失败原因可持久化
- **恢复机制**：`reconcile_queued_tasks`（未进 broker）+ `recover_stalled_tasks`
  （已执行但终态未落库），由 lifespan 周期 sweep 驱动

### 未偿还的残余

1. **executor 没有 attempt fencing** —— 见 TD-2，这是当前最硬的阻断项
2. **贝叶斯 study 跨进程** —— Celery 下贝叶斯仍被 422 拒绝。原方案
   （RDBStorage + slot 表 + 协调器 lease）经评审后**推翻**：贝叶斯本身串行，
   分布式只买到"训练下沉到 worker"，不值得那套复杂度。替代方案是编排器留在
   进程内、只把训练发给 worker（等待改为轮询 Run 终态）。未实施。
3. **测试仍假定进程内完成时序**，没有真实 worker 往返的集成测试

### 触发条件

- [x] backend 重启杀掉训练 —— 已由 worker + recovery 缓解
- [ ] 生产开启 `CELERY_KINDS=train` —— **必须先解决 TD-2**

---

## TD-2: executor 没有 attempt fencing（🔴 生产开启 Celery train 的硬阻断）

### 现状

M2c 保证了**数据库终态写回**幂等：重复投递时只有一个 attempt 能写终态，
其余 no-op。但**执行副作用不在这个保证范围内**：

- ML 模型直接写 `storage/models/{task_id}.joblib`，路径只含 task_id，不含 attempt
- 日志、domain task 同理

因此两个 attempt 并存时（Celery 重试、停滞任务恢复、soft cancel 后 worker 继续跑），
**落后完成的那个会覆盖先完成者的产物**，导致 `experiment_runs.metrics` 描述的
模型与磁盘上的文件不是同一个。这种不一致没有任何报错，只能靠人工比对发现。

### 期望

产物路径带 attempt 维度，或采用"先写临时路径、终态提交时原子提升"的方案，
让只有胜出的 attempt 的产物可见。

### 触发条件

- [ ] 生产设置 `CELERY_KINDS=train`（或 `SCHEDULER_MODE=celery`）**之前必须完成**
- 当前默认 `CELERY_KINDS=""`，train 仍在进程内，不会产生并发 attempt

---

## TD-3: 遗留 per-kind Celery 任务绕过 M2c 写回

### 现状

`celery_tasks.run_train_task` / `run_explain_task` 及 `task_runner._KIND_TO_CELERY`
/ `_dispatch_to_celery` 是更早一层的兼容残留：

- **正常业务流不可达**：`CeleryScheduler.submit` 只发布 `run_platform_task_generic`
- **但操作上可达**：这两个 task name 仍注册在 worker 上，任何知道名字的外部
  producer、旧版本 backend 或 broker 中的残留消息都能触发
- 它们用的是 M2c 之前的写回方式（`_mark_task_running/_success`，只更新
  PlatformTask），**没有 claim、没有 CAS、不保证 Run/Task 一致**

### 删除前置条件

1. 确认无外部 producer 调用这两个 task name
2. 排空 broker 中 active/reserved/scheduled 的残留消息（或先改成调用
   `_execute_generic` 的安全 shim 做滚动兼容）
3. 缺失 executor 时改为 fail-closed，不再静默回落到旧任务
4. 补齐契约测试（见 TD-4）

---

## TD-4: 调度链路存在零测试覆盖的分支

### 现状

一次实验性删除暴露了这个问题：把上述遗留路径整条删掉后，**449 个测试全部通过**，
没有任何一条抓到破坏。零覆盖的包括：

- `submit_task` / `_dispatch_to_celery`
- `run_train_task` / `run_explain_task`
- `celery_app.task_routes` 指向的任务是否真实注册在 worker registry
- 缺失 executor 时的回落行为
- 真实 broker/worker 往返

项目**没有覆盖率门禁**，因此"测试全过"对未覆盖路径不构成任何保证。

### 应补的契约

1. 所有业务创建入口在提交事务后调用 `get_scheduler(kind).submit()`
2. `CeleryScheduler` 对所有 kind 只发布 `run_platform_task_generic`
3. `task_routes` 中每个名称都必须存在于 worker task registry
4. 每个可达 worker task 必须先 claim；重复投递不得再次执行 executor
5. 每个终态路径必须同事务更新 Run + Task
6. retry 一次只增加一次 `retry_count`
7. 至少一条 Celery eager 或真实 worker 的端到端集成测试

---

## TD-5: 贝叶斯搜索默认预算下退化为随机搜索

### 现状

`n_trials_per_model` 默认 **10**，而 Optuna `TPESampler` 的 `n_startup_trials`
默认也是 **10** —— 启动期全部使用随机采样。实测：默认配置跑完 10 个 trial，
进入 TPE 建模阶段的次数为 **0**。

用户在 UI 上选择"贝叶斯搜索"，实际得到的是随机搜索，且更慢（串行）。

### 需要的决策（产品层面）

- 若要真正的贝叶斯行为：默认预算提到 30–50，或调低 `n_startup_trials`
- 若 10 次足够：该策略应如实命名为"随机搜索"

这不是纯技术问题，会影响耗时与用户预期，需产品决策后再改。

---
