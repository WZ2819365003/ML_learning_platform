# 时序模型接入 V3 平台 — 设计文档

- **状态:** 已批准（待实现）
- **日期:** 2026-04-29
- **范围:** 把时序预测能力作为第三个一等 family `ts` 接入 V3 ModelingTask → Experiment → Run 主链路
- **作者:** AI 协作产出
- **关联任务:** doc/todo.md P1 "模型扩展契约"
- **更新策略:** 需要一次镜像 rebuild（新增 `statsmodels` 依赖），后续业务代码改动都落在 `ml_platform/app/` 内，可通过 `docker compose restart backend` 快速生效（约 15-30s downtime）。**不引入开发态 bind mount**（保留现有 production 风格 compose）

---

## 1. 总体架构

把 `ts` 作为第三个一等 family（与 `ml` / `dl` 同构）接入 V3 调度链：

```
ModelingTask (task_type=forecasting)
    └── PlatformExperiment (strategy_type=baseline | grid_search | bayesian_search)
            └── ExperimentRun (model_token=arima | ets | lstm_forecaster | tcn_forecaster | timesfm_1)
                    └── PlatformTask (kind=ts_train) ──→ ts_executor → 训练/预测产物
```

### 模块清单

| 文件 | 类别 | 改动 |
|---|---|---|
| `app/core/ts_registry.py` | 新增 | 镜像 `dl_registry.py` 结构，定义 5 个模型的 `ParamSpec` / 类别 / 默认参数 |
| `app/core/ts_trainer.py` | 新增 | `BaseTSTrainer` 抽象基类 + 5 个具体 trainer 实现 |
| `app/services/ts_service.py` | 新增 | `_run_ts_training_sync` 协程 + `register_executor("ts_train", ...)` |
| `app/api/routes/ts_v3.py` | 新增 | V3 风格的 `/api/ts/models`、`/api/ts/forecast/{run_id}` |
| `app/core/model_registry.py` | 改 | `resolve_model_family()` 加 `ts` 分支；`list_models(family="ts")` |
| `app/services/tuning_service.py` | 改 | trial 派发处加 `elif family == 'ts'` |
| `app/services/resolver.py` | 改 | `TaskFacade` 支持 `task_kind = "forecasting"`；新增 `is_forecaster()`；**模型文件路径解析改为多扩展名兼容**（`.joblib` / `.pt` / `.json`），不再硬编码 `.joblib` |
| `ml_platform/requirements.txt` | 改 | 新增 `statsmodels>=0.14` |
| `app/api/routes/visualization.py` | 改 | 新增 forecasting 专属图表端点（折线 + 预测区间） |
| `app/services/platform_task_detail_service.py` | 改 | forecasting 分支取指标和图表元数据 |
| `app/main.py` | 改 | 注册 `ts_v3_router` |
| `ml_platform_web/src/pages/ModelingTaskDetail.jsx` | 改 | 任务类型选项加 forecasting；条件渲染 TS 表单/图表 |
| `ml_platform_web/src/components/forecasting/` | 新增 | `TimeSeriesPlanForm.jsx`、`ForecastChart.jsx` |
| `playwright_test/test/13-v3-forecasting.spec.js` | 新增 | E2E：建任务 → baseline → metrics → Inspector |

### 关键不变量

- **不新增数据库表** — forecasting 走 `Dataset` + `TrainingPlan.payload` 元数据
- **依赖新增 1 个** — `statsmodels>=0.14`（ARIMA/ETS）。`torch` 已有；TimesFM 走现有 `timesfm_env` venv，不影响主进程
- **不修改 docker-compose.yml** — 仅 `requirements.txt` 加一行 + 一次镜像 rebuild；后续代码改动 `restart backend` 即生效
- **老路径保留** — `/ts/tasks`、`TimeSeriesForecastTask` 表、`/api/timesfm/*` 标记 deprecated 不删除

---

## 2. Forecasting task_type & TrainingPlan payload schema

### task_type 扩展

```python
# schemas.py
TaskType = Literal["classification", "regression", "forecasting"]
```

`ModelingTask.task_type = "forecasting"` 时，绑定的 `TrainingPlan` 必须满足下列 payload 结构。

### TrainingPlan.payload schema（forecasting 专属字段）

```jsonc
{
  // —— 通用字段（与 ml/dl 一致）——
  "name": "demo-forecasting-plan",
  "task_type": "forecasting",
  "model_family": "ts",
  "strategy_type": "baseline",            // baseline | grid_search | bayesian_search
  "model_tokens": ["arima", "ets"],       // ts_registry 已知 token 之一或多个

  // —— forecasting 专属 ——
  "time_series": {
    "timestamp_col":  "ds",               // 必填，必须存在于 dataset.columns_info
    "target_col":     "y",                // 必填
    "series_id_col":  null,               // 选填：多序列时填 group 列；单序列填 null
    "exogenous_cols": [],                 // 选填：外生特征列名
    "freq":           "D",                // pandas freq alias: D/H/M/W/15min...
    "horizon":        14,                 // 必填：预测步长
    "lookback":       28,                 // 神经/foundation；ARIMA/ETS 忽略
    "validation": {
      "method":    "holdout",             // holdout | rolling_origin | expanding_window
      "test_size": 14,                    // holdout 时是步数；rolling_origin 时是切片数
      "step":      1                      // 仅 rolling_origin / expanding_window
    },
    "interval_levels": [80, 95]           // 预测区间分位数；无区间能力的模型忽略
  }
}
```

### Pydantic schema 落点

新增 `app/models/schemas.py::TimeSeriesPlanConfig` (BaseModel)，挂在 `TrainingPlanPayload` 上，**只在 `task_type == "forecasting"` 时校验非空**。

```python
class TimeSeriesPlanConfig(BaseModel):
    timestamp_col: str
    target_col: str
    series_id_col: str | None = None
    exogenous_cols: list[str] = []
    freq: str
    horizon: int = Field(gt=0, le=10000)
    lookback: int = Field(gt=0, default=28)
    validation: TimeSeriesValidationConfig
    interval_levels: list[int] = [80, 95]
```

### 数据预处理契约（trainer 入口前完成）

`ts_service._run_ts_training_sync` 统一做四件事，trainer 拿到的就是干净的 `pd.DataFrame`：

1. `pd.read_csv(dataset.path)` → 排序 `[series_id_col, timestamp_col]`
2. 校验 freq 一致性 — 用 `pd.infer_freq()` 校验；若失败或缺失率 > 5% 直接 fail，错误结构化进 `PlatformTask.error`
3. 按 `validation.method` 切训练/验证集
4. 标准化为 `(train_df, val_df, test_df, meta)` 四元组，trainer 接口统一

### 指标 — `FORECAST_EVAL_METRICS`

完全独立于 regression：

| metric | 含义 |
|---|---|
| MAE | 平均绝对误差 |
| RMSE | 均方根误差 |
| MAPE | 平均绝对百分比误差 |
| sMAPE | 对称平均绝对百分比误差 |
| MASE | 相对季节性 naive baseline 的相对误差 |
| coverage_80 | 80% 预测区间覆盖率（区间模型） |
| coverage_95 | 95% 预测区间覆盖率（区间模型） |

### 不做的事 (YAGNI)

- ❌ 多变量预测（multi-target）— 单 target，多 target 留下一期
- ❌ 数据集级别 schema 自动扫描（自动识别时间列）— 人工选
- ❌ Hierarchical / aggregation — 超 A 方案范围
- ❌ 在线流式预测 — 只做批量

---

## 3. 5 个模型接入细节

### 3.1 BaseTSTrainer 抽象

```python
# app/core/ts_trainer.py
class BaseTSTrainer(ABC):
    """统一接口：所有 ts family trainer 必须实现。"""

    name: str                                   # token，与 ts_registry 一致
    supports_intervals: bool = False            # 是否原生输出预测区间
    supports_exogenous: bool = False            # 是否支持外生特征

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, meta: TSMeta) -> None: ...

    @abstractmethod
    def predict(self, horizon: int, exog: pd.DataFrame | None = None
                ) -> ForecastResult: ...
        # ForecastResult: { mean: np.ndarray, intervals: dict[int, (low, high)] | None }

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseTSTrainer": ...
```

### 3.2 各 trainer 实现要点

| Token | 库 | 模型文件 | 区间 | 外生 | 备注 |
|---|---|---|---|---|---|
| `arima` | `statsmodels.tsa.arima.model.ARIMA` | `.joblib`（pickle SARIMAXResults） | ✅ | ✅ | 参数: order=(p,d,q), seasonal_order |
| `ets` | `statsmodels.tsa.holtwinters.ExponentialSmoothing` | `.joblib` | ✅ | ❌ | 参数: trend, seasonal, seasonal_periods |
| `lstm_forecaster` | `torch`（**新建** `dl_models/lstm_forecaster.py`，参考 `lstm.py` 但输出层改为 horizon 维） | `.pt` | ❌ (基线) | ✅ | direct multi-step：linear(hidden→horizon)；MSE loss；不复用 `lstm_regressor` 类（避免破坏现有回归） |
| `tcn_forecaster` | `torch`（**新建** `dl_models/tcn_forecaster.py`） | `.pt` | ❌ | ✅ | causal conv stack；同样 horizon 维输出 |
| `timesfm_1` | 现有 `timesfm_runner.py` 子进程 | 子进程 cold-start 不缓存 | ❌ | ❌ | foundation 模型，无 fit；只调用 forecast |

### 3.3 TimesFM 接入特殊处理

TimesFM 1.0 必须在 Python 3.10 venv（FastAPI 主进程是 3.13）。**不改既有架构**：

- `TimesFMAdapter` 实现 `BaseTSTrainer.fit()` 是 no-op（foundation 模型不需要训练）
- `predict()` 内部调用 `timesfm_service._run_forecast_subprocess(...)`，复用现有的 `ThreadPoolExecutor` + venv 路径
- `save()` 写一个 `marker.json`（记录模型版本），`load()` 校验 venv 仍可用
- 当 `timesfm_env/` 缺失时，`ts_registry.list_available_models()` 把 `timesfm_1` 标为 `available=false`，前端禁用对应选项

---

## 4. 调度链改动

### 4.1 model_registry.resolve_model_family

```python
def resolve_model_family(model_id: str) -> str | None:
    from app.core.dl_registry import get_dl_model_spec
    from app.core.ts_registry import get_ts_model_spec   # NEW

    if get_ml_model_spec(model_id) is not None:
        return "ml"
    if get_dl_model_spec(model_id) is not None:
        return "dl"
    if get_ts_model_spec(model_id) is not None:           # NEW
        return "ts"
    return None
```

### 4.2 tuning_service 派发分支

`_persist_trials` / `_baseline_trial` / `_dispatch_trial` 等已有 `elif family == 'dl'` 的位置，全部加上 `elif family == 'ts'` 分支，调用 `ts_service.run_ts_trial(...)`。

### 4.3 ts_service 执行链

```python
# app/services/ts_service.py
from app.scheduler.executors import register_executor

async def run_ts_executor(domain_id: str, platform_task_id: str) -> dict[str, Any]:
    """domain_id 是 ExperimentRun.id（与 ml/dl 一致）"""
    run = await load_experiment_run(domain_id)
    plan = await load_training_plan_snapshot(run.experiment_id)

    # 1. 加载 dataset → DataFrame
    # 2. 按 plan.payload.time_series.validation 切片
    # 3. 实例化 trainer（ts_registry 查 token）
    # 4. fit → predict on val → 计算 FORECAST_EVAL_METRICS
    # 5. 持久化模型到 storage/models/{run_id}.{ext}
    # 6. 写 ExperimentRunLog（V3 native）
    # 7. 写 metrics_snapshot 回 PlatformTask
    return {"metrics": {...}, "artifacts": {"model_path": ..., "forecast_path": ...}}

register_executor("ts_train", run_ts_executor)
```

### 4.4 platform_task_detail_service forecasting 分支

`build_run_detail()` 在装配 metrics / chart spec 时，根据 `run.task_type == "forecasting"` 走专属图表 list（`forecast_line`、`residual`、`coverage_calibration`），不复用 regression 的 `predicted_vs_actual`。

---

## 5. 前端

### 5.1 已有页面改动

| 页面 | 改动 |
|---|---|
| `ModelingTaskDetail.jsx` | 创建任务时 task_type 下拉新增 "时序预测 (forecasting)" 选项 |
| `TrainingPlans.jsx` | 列表/编辑器支持 forecasting plan；表单条件渲染 |
| `V3Runs.jsx` | 列表筛选器加 family=ts；列展示 horizon / freq |

### 5.2 新增组件

```
src/components/forecasting/
  ├── TimeSeriesPlanForm.jsx    — 采集 time_series payload (列选择器 + horizon/freq/validation)
  ├── ForecastChart.jsx         — ECharts 折线 + 预测区间 (band) + 真值对照
  ├── ForecastMetrics.jsx       — MAE/RMSE/MAPE/sMAPE/MASE/coverage 卡片
  └── ts/
      ├── ColumnSelector.jsx    — 时间列/目标列/series_id 选择，从 dataset.columns_info 拉
      └── ValidationConfig.jsx  — holdout / rolling_origin / expanding_window 参数表单
```

### 5.3 老路径处理

- `/ts/tasks`、`/ts/monitor`、`/ts/results`、`/timesfm` 路由保留
- 顶部加 deprecated banner："此路径将于下一期清理，请使用新建模任务 → 时序预测"
- 不再从主导航暴露老入口

---

## 6. 测试 & 验证 & 弃用计划

### 6.1 后端单元测试

新增 `tests/test_ts_*.py`：

- `test_ts_registry.py` — 5 个 token 都注册；schema 完整
- `test_ts_trainer_arima.py` — ARIMA 拟合 + 预测正确；区间覆盖率合理
- `test_ts_trainer_ets.py` — 同上
- `test_ts_trainer_lstm.py` — 神经 trainer 在 sin 波数据上 RMSE < threshold
- `test_ts_metrics.py` — MAPE/sMAPE/MASE 在已知 ground truth 上数值正确
- `test_ts_service_executor.py` — executor 端到端：构造 run/plan → run executor → 检查 metrics 落表

### 6.2 E2E 测试

新增 `playwright_test/test/13-v3-forecasting.spec.js`：

1. 上传时序数据集（一份 sin 波 CSV，提前放 `examples/data/`）
2. 创建 ModelingTask (task_type=forecasting)
3. 创建 TrainingPlan，绑定 arima + ets 双模型，baseline strategy
4. 触发 experiment → 等 run 跑完
5. 校验 V3 leaderboard 有数据，Inspector 可点开看 ForecastChart

发布门禁加这条：必须 13/13 全绿 + 现有 67/67 不退化。

### 6.3 弃用计划（下下期）

本任务发布版本预定 **v3.4.0**。在 v3.5 或 v3.6 执行下列清理（保留至少 1 个版本的 deprecated 期）：

- 删除 `app/api/routes/timesfm.py` 中的 `/timesfm/*` 老接口
- 删除 `app/services/timesfm_service.py` 顶层公共接口（保留内部 `_run_forecast_subprocess` 给新 adapter 用）
- 删除 `TSConfig.jsx` / `TSMonitor.jsx` / `TSResults.jsx` / `TimesFM.jsx` 四个老页面
- 删除 `TimeSeriesForecastTask` / `TimeSeriesDeployment` 表（用 alembic 或 migration script）
- 老接口在删除前 banner 提示三个版本

### 6.4 漏洞修复（独立任务）

dependabot 报的 34 个 npm 漏洞与本任务**正交**，单开 PR 处理：

- 大概率全部是 transitive deps，`npm audit fix` + 必要时手动 bump 顶层包
- 不在本设计范围内

---

## 7. 实施顺序（高层 milestone）

1. **M1 — 依赖 & 镜像** (0.5d)：`requirements.txt` 加 `statsmodels`；rebuild backend 镜像；启动后健康检查；这是**整个任务里唯一一次镜像 rebuild**
2. **M2 — 后端骨架** (1d)：`ts_registry.py` + 空的 `BaseTSTrainer` + `ts_service` 注册 executor + `resolve_model_family` 加分支 + `resolver.py` 多扩展名兼容
3. **M3 — ARIMA + ETS** (1d)：两个统计 trainer 完整实现 + 单测
4. **M4 — LSTM + TCN forecaster** (1.5d)：新建 `dl_models/{lstm,tcn}_forecaster.py` + 单测
5. **M5 — TimesFM adapter** (0.5d)：包装现有 venv 调用，挂在 `BaseTSTrainer` 接口上
6. **M6 — 前端** (2d)：TimeSeriesPlanForm + ForecastChart + ModelingTaskDetail 集成
7. **M7 — 端到端验证** (1d)：13-v3-forecasting.spec.js + 67+13 全绿
8. **M8 — 文档 & 发布** (0.5d)：功能说明.md / 系统架构.md 更新；版本号 → v3.4.0

**总工作量：~8 天**（实际 dev 人日，不含等待）。M1 之后无新依赖，业务代码改动 `restart backend` 即生效。

---

## 8. 风险与回滚

| 风险 | 影响 | 缓解 |
|---|---|---|
| forecasting Pydantic 校验过严挡住已有 ml/dl 创建流 | 阻塞 | 校验只在 `task_type=="forecasting"` 时启用 |
| TimesFM 子进程在容器里失败 | timesfm_1 不可用 | `available=false` 优雅降级，不阻塞其它 4 个模型 |
| 前端老 `/ts/tasks` 路径有用户书签 | 体验 | 加 deprecated banner + 自动跳转引导，三版本后再删 |
| 神经 forecaster 在小数据集上不收敛 | 单 run 失败 | trainer 内 try/except 写结构化错误进 ExperimentRun.error，不挂掉整个 experiment |
| 热更新时 ts_registry 与已运行的 task_runner 内存视图不一致 | 调度异常 | scheduler 每次 trial 派发现查 registry，不 cache |

回滚策略：所有改动通过 feature flag `ENABLE_FORECASTING`（默认 true）开关；出问题改 `false` 重启即可，老 `/ts/tasks` 路径保留作为 fallback。
