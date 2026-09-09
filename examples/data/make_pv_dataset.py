"""光伏电站出力数据：按太阳几何 + 晴空辐照 + 云量随机过程生成，再做特征工程。

不是随机数堆出来的：辐照来自赤纬角/时角算出的太阳高度角，云量是有自相关的
AR(1) 过程，组件温度按 NOCT 模型抬升，出力受温度系数负反馈并有逆变器限功率。
这些关系是模型真正要学的东西。
"""
import numpy as np, pandas as pd

rng = np.random.default_rng(20260909)
LAT = np.radians(32.0)          # 约北纬 32°（华东）
CAP_KW = 5000.0                 # 装机 5 MW
GAMMA = -0.0040                 # 功率温度系数 /°C
NOCT_K = 0.028                  # 辐照抬升组件温度的系数

idx = pd.date_range("2024-01-01", periods=24 * 731, freq="h")   # 2 年整
doy = idx.dayofyear.to_numpy(); hour = idx.hour.to_numpy()

decl = np.radians(23.45) * np.sin(2 * np.pi * (284 + doy) / 365)
omega = np.radians(15.0 * (hour - 12))
cos_z = np.sin(LAT) * np.sin(decl) + np.cos(LAT) * np.cos(decl) * np.cos(omega)
cos_z = np.clip(cos_z, 0, None)
elev = np.degrees(np.arcsin(np.clip(cos_z, -1, 1)))
azim = np.degrees(np.arctan2(np.sin(omega), np.cos(omega) * np.sin(LAT) - np.tan(decl) * np.cos(LAT)))

# 晴空 GHI（Haurwitz）
ghi_clear = np.where(cos_z > 0, 1098.0 * cos_z * np.exp(-0.059 / np.maximum(cos_z, 1e-3)), 0.0)

# 云量：AR(1)，冬季更阴
noise = rng.normal(0, 1, len(idx)); cloud = np.zeros(len(idx))
for i in range(1, len(idx)):
    cloud[i] = 0.92 * cloud[i - 1] + 0.38 * noise[i]
cloud = 1 / (1 + np.exp(-(cloud - 0.35 * np.cos(2 * np.pi * (doy - 15) / 365))))
kt = np.clip(1.0 - 0.78 * cloud + rng.normal(0, 0.035, len(idx)), 0.05, 1.0)

ghi = ghi_clear * kt
dhi = ghi * np.clip(0.28 + 0.62 * cloud, 0, 1)
dni = np.where(cos_z > 0.02, np.clip((ghi - dhi) / np.maximum(cos_z, 0.02), 0, 1000), 0.0)

temp = (14.5 - 11.5 * np.cos(2 * np.pi * (doy - 20) / 365)
        - 5.0 * np.cos(2 * np.pi * (hour - 15) / 24)
        - 2.5 * cloud + rng.normal(0, 1.6, len(idx)))
humid = np.clip(58 + 26 * cloud - 0.85 * (temp - 15) + rng.normal(0, 6, len(idx)), 12, 100)
wind = np.clip(rng.gamma(2.0, 1.35, len(idx)) + 0.9 * cloud, 0, None)
press = 1013 + 7 * np.cos(2 * np.pi * (doy - 15) / 365) + rng.normal(0, 3.5, len(idx))

mod_temp = temp + NOCT_K * ghi - 0.9 * np.minimum(wind, 8)
soil = 1.0 - 0.035 * (1 - np.exp(-((doy % 90) / 60.0)))          # 积灰，季度清洗
ac = CAP_KW * (ghi / 1000.0) * (1 + GAMMA * (mod_temp - 25.0)) * soil * 0.965
ac = np.clip(ac, 0, CAP_KW * 0.94)                                # 逆变器限功率
ac = np.where(elev > 1.5, ac, 0.0) * (1 - 0.35 * rng.binomial(1, 0.004, len(idx)))  # 偶发限电
ac = np.clip(ac + rng.normal(0, 22, len(idx)), 0, None).round(2)

df = pd.DataFrame({
    "timestamp": idx, "ac_power": ac,
    "ghi": ghi.round(2), "dni": dni.round(2), "dhi": dhi.round(2),
    "ambient_temp": temp.round(2), "module_temp": mod_temp.round(2),
    "humidity": humid.round(1), "wind_speed": wind.round(2), "pressure": press.round(1),
    "cloud_cover": cloud.round(4), "solar_elevation": elev.round(3), "solar_azimuth": azim.round(2),
    "hour_of_day": hour, "day_of_year": doy, "month": idx.month.to_numpy(),
    "day_of_week": idx.dayofweek.to_numpy(),
})

# ---- 特征工程 ----
two_pi = 2 * np.pi
for name, val, period in (("hour", hour, 24), ("doy", doy, 365), ("month", df.month, 12)):
    df[f"{name}_sin"] = np.sin(two_pi * val / period).round(5)
    df[f"{name}_cos"] = np.cos(two_pi * val / period).round(5)
df["azimuth_sin"] = np.sin(np.radians(azim)).round(5)
df["azimuth_cos"] = np.cos(np.radians(azim)).round(5)

for lag in (1, 2, 24, 168):
    df[f"ac_power_lag_{lag}"] = df.ac_power.shift(lag)
df["ghi_lag_1"] = df.ghi.shift(1)
for w in (3, 24):
    df[f"ac_power_roll_mean_{w}"] = df.ac_power.shift(1).rolling(w).mean().round(2)
df["ac_power_roll_std_24"] = df.ac_power.shift(1).rolling(24).std().round(2)
df["ghi_roll_mean_3"] = df.ghi.shift(1).rolling(3).mean().round(2)

df["clearness_index"] = np.where(ghi_clear > 5, ghi / np.maximum(ghi_clear, 1e-6), 0).round(4)
df["temp_x_ghi"] = (df.ambient_temp * df.ghi / 1000).round(3)
df["ghi_sq"] = (df.ghi ** 2 / 1000).round(2)
df["cooling_degree"] = np.clip(df.ambient_temp - 26, 0, None).round(2)
df["temp_spread_module_ambient"] = (df.module_temp - df.ambient_temp).round(2)
df["days_since_start"] = (idx - idx[0]).days

df = df.dropna().reset_index(drop=True)
df = df.drop(columns=["timestamp"])
out = "/private/tmp/claude-501/-Users-petersmith-Public-github-repo-ML-learning-platform/f5652b76-ef60-421f-82fc-825da61772f8/scratchpad/pv/光伏电站出力数据.csv"
df.to_csv(out, index=False, encoding="utf-8")

print(f"{len(df)} 行 × {df.shape[1]} 列 -> {out}")
print(f"ac_power: 均值 {df.ac_power.mean():.1f}  最大 {df.ac_power.max():.1f}  零值占比 {(df.ac_power == 0).mean():.1%}")
day = df[df.solar_elevation > 1.5]
print(f"白天 {len(day)} 行，均值 {day.ac_power.mean():.1f}")
print("与出力相关性 top:", df.corr(numeric_only=True).ac_power.drop("ac_power").abs().nlargest(5).round(3).to_dict())
