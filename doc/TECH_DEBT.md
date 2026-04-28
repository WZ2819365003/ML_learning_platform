# 技术债清单

记录已识别但**故意暂不偿还**的架构问题，避免下个人重新踩坑。每条债务包含：
1. 现状 — 现在是什么样
2. 期望 — 为什么这是债
3. 真实工作量 — 不是"看着像"，是"真做完"
4. 触发条件 — 什么情况下必须还

---

## TD-1: V3 提交链路绕过 Scheduler 抽象（v3.3.x 调查后挂起）

### 现状

代码里有两套并行的"任务派发"基础设施：

| 系统 | 入口 | 在哪运行 |
|---|---|---|
| **A. asyncio in-process** | `asyncio.create_task(executor(...))` | backend 容器的 FastAPI 事件循环 |
| **B. Scheduler abstraction** | `dispatch_platform_task → get_scheduler() → CeleryScheduler.submit` | 独立 Celery worker 容器 |

99% 的真实流量走 A，B 写好了但只接住一个边缘端点：

```
路径                                          走 A？  走 B？
POST /api/training/start                      ✓       ✗
POST /api/v3/tasks/{id}/experiments           ✓       ✗
POST /api/dl/train                            ✓       ✗
POST /api/platform/tasks                      ✓       ✗
_schedule_shap_for_top_runs (auto-SHAP)       ✓       ✗
POST /api/platform/tasks/{id}/retry           ✗       ✓   ← 唯一例外
```

### 期望

`SCHEDULER_MODE=celery` 应该把所有任务路由到 worker 容器，让：
- backend 重启不杀掉运行中的训练
- worker 横向扩展承载并发训练
- CPU/GPU 重活不阻塞 FastAPI 事件循环

### 为什么是债

`scheduler.py` 顶部注释明确写了"切个 SCHEDULER_MODE 就生效"。但实际上：
- `_build_scheduler()` 只决定 `get_scheduler()` 返回啥
- `get_scheduler()` 只在 `dispatch_platform_task` 里用
- `dispatch_platform_task` 只被 retry 端点调用

**误导性强**：v3.3.1 的方案 A 实验中我搭好了 worker 容器 + 切 mode，跑得 healthy，结果 worker 0 任务接收量。下个人按注释来，会踩同一个坑。

### 真实工作量（v3.3.x 重新估）：5–8 个工作日

需要改的：

1. **5 个提交路径全部改写**
   - `training_service.start_training` — 砍掉 `asyncio.create_task(_execute_training)`，改 `await get_scheduler().submit(platform_task_id)`
   - `tuning_service._run_concurrent_batch` — 同上，每个 trial 一个 submit
   - `tuning_service._run_bayesian_search` — 顺序 submit 后 wait
   - `dl_service.start_dl_training` — 砍掉 `asyncio.create_task(_execute_dl_training)`
   - `_schedule_shap_for_top_runs` — 改 submit
   - `platform_tasks` 手动 POST — 改 submit

2. **executor 完成后回写 DB 的语义重设计**

   现在 `_execute_single_trial` 是同步等：
   ```python
   result = await executor(domain_task_id, platform_task_id)
   await update_run_metrics(db, run_id, result.metrics, status="SUCCESS")
   ```

   切 Celery 后 backend 立刻返回，executor 在 worker 进程里跑，回写 DB 的责任要挪到 `celery_tasks.py` 里的 success/failure 回调。

3. **Optuna study state 跨进程共享**

   `_run_bayesian_search` 现在用 `optuna.create_study()`（默认 in-memory）。TPE 采样器要看上一 trial 的结果决定下一组超参。切 Celery 后多 worker 各自跑 trial，必须把 study 切到 RDBStorage（Optuna 内置 MySQL 后端）。

4. **V3 native logs mirror 重新接线**（v3.3.0 刚做的功能）

   `_mirror_logs_to_v3` 现在是 `_execute_single_trial` 里 `executor` 返回后同步调的。切 Celery 后挪到 `celery_tasks.py` 的 success 回调里。

5. **测试套件全部加 polling**

   现在 64 条 milestone gate 全部假定"4 秒等到 baseline 完成"。切 Celery 后任务在 worker 上跑，必须改成 polling + 长 timeout，spec 变脆弱。

6. **错误传播**

   现在 `asyncio.gather(..., return_exceptions=True)` 自然收单 trial 失败。Celery 模式下要从 result_backend 把 exception 信息拉回来塞进 ExperimentRun.error_message。

### 触发条件（什么情况下必须还）

- [ ] 用户报"backend 重启把训练任务杀了"——目前 demo 阶段用户重启 docker 时手动停所有任务，没踩到
- [ ] 单机并发训练超过 4 个，FastAPI 事件循环被阻塞，UI 卡顿
- [ ] 上线 GPU 训练，V100 满载时 backend healthcheck 跟不上
- [ ] 多机部署需求

### 不偿还的临时缓解

- `scheduler.py` 顶部 docstring 标记 `⚠ TECH DEBT`，列出 5 条 bypass 路径，提醒后人**不要被注释误导**（v3.3.0 commit 已加）
- `docker-compose.yml` 不带 worker 容器、不切 SCHEDULER_MODE，避免假象
- 本文件作为持久化记录

### 历史

- v3.3.x 调查：方案 A（worker 容器）实验后确认 0 派发量
- 2026-04 决定走方案 C：撤回 worker 容器、记录债务、把工时挪到 ROI 更高的项目
