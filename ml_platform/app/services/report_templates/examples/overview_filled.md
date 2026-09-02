# 测试1-电力负荷预测 · 建模报告

## 结论

最优模型为 xgboost_regressor，交叉验证 RMSE 72.4673，误差量级为目标列均值的 0.81%。该成绩取自模型选择阶段，封存测试集上的最终评估尚未执行，泛化能力未经确认；就绪评分 60/100，扣分项即为此项。建议先完成最终评估，再决定是否上线。

需注意排名第一与第二为同一模型的两次独立训练，结果完全一致；真正的次优模型是 lightgbm_regressor，且与冠军的差距不具备统计意义（见下节）。

## 模型表现

七个 Run 分三批执行，全部成功，分属两族，成绩呈明显断层：四个树模型的 RMSE 集中在 72–80，三个深度学习模型在 132–188，后者误差约为前者的两倍。

树模型内部差异极小。lightgbm 与 xgboost 相差 0.3183，而 xgboost 自身的折间标准差为 0.8539——差距落在交叉验证噪声之内，当前数据不足以判定二者优劣；若在推理速度或模型体积上有偏好，可据此选型。random_forest 落后 6.8167，是树模型中唯一可辨识的差距。

两族分数口径不同：树模型为五折交叉验证均值，深度学习模型为留出验证集单次结果，二者不宜直接横向比较，表中排名仅供参考。

| 排名 | 模型 | RMSE | 占均值 | R² | 折间标准差 | 口径 |
|---:|---|---:|---:|---:|---:|---|
| 1 | xgboost_regressor | 72.4673 | 0.81% | 0.9974 | 0.8539 | 5 折 CV |
| 2 | xgboost_regressor | 72.4673 | 0.81% | 0.9974 | 0.8539 | 5 折 CV |
| 3 | lightgbm_regressor | 72.7856 | 0.82% | 0.9973 | 0.9344 | 5 折 CV |
| 4 | random_forest_regressor | 79.2840 | 0.89% | 0.9968 | 1.0957 | 5 折 CV |
| 5 | mlp_dl | 132.0422 | 1.48% | — | — | 留出验证 |
| 6 | lstm | 136.3387 | 1.53% | — | — | 留出验证 |
| 7 | lstm | 187.3476 | 2.11% | — | — | 留出验证 |

就绪评分满分 100，由最终评估（40）、跨折稳定性（30）、Run 全部成功（30）三项构成。本次后两项达成，最终评估未执行。

## 数据集

87312 行 × 35 列，目标列 load，无缺失；均值 8896.59，取值范围 5498.36–14274.15。

35 列中 11 列为原始采集与日历字段，24 列为训练流程构造的特征，构造特征占 69%。模型对其依赖显著：最优模型的 SHAP 首位为 load_lag_1，平均绝对贡献 976.4，是次位 load_lag_2（95.6）的 10.2 倍。即预测主要来自负荷自身的短期自相关，气象与日历字段仅提供次级修正。

| 类别 | 列数 | 字段 |
|---|---:|---|
| 原始采集与日历 | 11 | load、month、day_of_week、day_of_year、hour_of_day、is_weekend、humidity、wind_speed、dry_bulb_temp、wet_bulb_temp、dew_point_temp |
| 周期三角变换 | 8 | hour_sin/cos、dow_sin/cos、doy_sin/cos、month_sin/cos |
| 滞后项 | 4 | load_lag_1、load_lag_2、load_lag_48、load_lag_336 |
| 滚动统计 | 3 | load_roll_mean_6、load_roll_mean_48、load_roll_std_48 |
| 气象与交互衍生 | 9 | cooling_degree、heating_degree、discomfort_index、dew_point_depression、temp_spread_dry_wet、dry_bulb_temp_sq、temp_x_hour、weekend_x_hour、days_since_start |
