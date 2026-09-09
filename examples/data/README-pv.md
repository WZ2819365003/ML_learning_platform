# 光伏电站出力数据（合成）

`make_pv_dataset.py` 生成，用于端到端验证建模流程与 AI 报告。**不是实测数据**——
真实光伏数据需要从外部数据源下载，未经许可不做；这份用物理关系合成，可复现
（种子固定 20260909）。

## 物理关系

不是随机数堆出来的，模型要学的关系是真的：

- **太阳几何**：赤纬角 δ = 23.45°·sin(2π(284+n)/365)，时角 ω = 15°(h−12)，
  由此得天顶角余弦与太阳高度角（北纬 32°）
- **晴空辐照**：Haurwitz 模型 `1098·cosZ·exp(−0.059/cosZ)`
- **云量**：AR(1) 自相关过程（φ=0.92）经 logistic 压到 0–1，冬季偏阴；
  晴空指数 kt = 1 − 0.78·cloud + 噪声
- **组件温度**：环温 + 0.028·GHI − 0.9·min(风速, 8)（NOCT 式抬升，风冷）
- **出力**：装机 5 MW × (GHI/1000) × (1 + γ(T_mod − 25))，γ = −0.004/°C；
  含积灰衰减（季度清洗）、逆变器 94% 限功率、0.4% 概率的限电事件

## 规模

17376 行 × 39 列，逐小时，2 年整。目标列 `ac_power`：均值 748 kW，最大 4423 kW，
**夜间零值占 25.7%**（白天 8388 行均值 1541 kW）——目标分布零膨胀，是这份数据
和电力负荷那份最不一样的地方。

## 字段构成

| 类别 | 列 |
|---|---|
| 原始观测与日历 | ghi、dni、dhi、ambient_temp、module_temp、humidity、wind_speed、pressure、cloud_cover、solar_elevation、solar_azimuth、hour_of_day、day_of_year、month、day_of_week |
| 周期三角变换 | hour_sin/cos、doy_sin/cos、month_sin/cos、azimuth_sin/cos |
| 滞后项 | ac_power_lag_{1,2,24,168}、ghi_lag_1 |
| 滚动统计 | ac_power_roll_mean_{3,24}、ac_power_roll_std_24、ghi_roll_mean_3 |
| 派生与交互 | clearness_index、temp_x_ghi、ghi_sq、cooling_degree、temp_spread_module_ambient、days_since_start |

注意 `ghi` 与 `ac_power` 相关性 0.998——辐照本来就是出力的主导项。这让模型很容易
做好，但也正是真实光伏"估算"场景的样子；若要做真正的**预测**，应移除同时刻的
辐照观测，只留滞后项与气象预报。
