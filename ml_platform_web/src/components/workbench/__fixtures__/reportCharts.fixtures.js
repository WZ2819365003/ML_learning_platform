/**
 * Chart-spec fixtures, one per kind, following the ai-report payload contract.
 * Numbers come from the report mock (doc/report-mock-overview.md, 样稿 v2):
 * 测试1-电力负荷预测, 87312 rows × 35 columns, target `load`, mean 8897.
 *
 * The backend agent builds these from real runs; the frontend renders them.
 * Field names here are the contract — do not rename.
 */

const MEAN_LOAD = 8897
const ONE_PERCENT = Math.round(MEAN_LOAD * 0.01 * 100) / 100 // 88.97

const LEADERBOARD_ROWS = [
  { model: 'xgboost_regressor', rmse: 72.4673, pct_of_mean: 0.81, r2: 0.9812, fold_std: 3.1, validation: '交叉验证' },
  { model: 'lightgbm_regressor', rmse: 72.7931, pct_of_mean: 0.82, r2: 0.9809, fold_std: 3.4, validation: '交叉验证' },
  { model: 'random_forest_regressor', rmse: 79.2518, pct_of_mean: 0.89, r2: 0.9774, fold_std: 4.2, validation: '交叉验证' },
  { model: 'gradient_boosting_regressor', rmse: 84.1064, pct_of_mean: 0.95, r2: 0.9746, fold_std: 5.0, validation: '交叉验证' },
  { model: 'mlp_dl', rmse: 132.4102, pct_of_mean: 1.49, r2: 0.9371, fold_std: null, validation: '留出验证' },
  { model: 'lstm_dl', rmse: 158.9377, pct_of_mean: 1.79, r2: 0.9094, fold_std: null, validation: '留出验证' },
  { model: 'gru_dl', rmse: 186.7245, pct_of_mean: 2.10, r2: 0.8750, fold_std: null, validation: '留出验证' },
]

/** 图 1 · 七个模型的误差 — hbar, two series by validation scheme, error bars, 1% line. */
export const leaderboardSpec = {
  id: 'leaderboard',
  kind: 'hbar',
  title: '七个模型的误差',
  caption: '树模型全在 1% 线以内，xgboost 与 lightgbm 的误差条重叠；深度学习模型在 1.5%–2.1%。',
  unit: 'RMSE',
  categories: LEADERBOARD_ROWS.map((r) => r.model),
  series: [
    {
      name: '交叉验证',
      values: LEADERBOARD_ROWS.map((r) => (r.validation === '交叉验证' ? r.rmse : null)),
      error: LEADERBOARD_ROWS.map((r) => (r.validation === '交叉验证' ? r.fold_std : null)),
      color_role: 'primary',
    },
    {
      name: '留出验证',
      values: LEADERBOARD_ROWS.map((r) => (r.validation === '留出验证' ? r.rmse : null)),
      error: null,
      color_role: 'muted',
    },
  ],
  reference_lines: [{ value: ONE_PERCENT, label: '平均负荷的 1%' }],
  tooltip_fields: [
    { key: 'model', label: '模型' },
    { key: 'rmse', label: 'RMSE', format: '.1f' },
    { key: 'pct_of_mean', label: '占均值', format: 'percent' },
    { key: 'r2', label: 'R²', format: '.3f' },
    { key: 'fold_std', label: '折间标准差', format: '.1f' },
    { key: 'validation', label: '验证方式' },
  ],
  rows: LEADERBOARD_ROWS,
}

const FOLDS = {
  xgboost_regressor: [
    { fold: 1, rmse: 69.8, mae: 52.1, r2: 0.983 },
    { fold: 2, rmse: 71.2, mae: 53.4, r2: 0.982 },
    { fold: 3, rmse: 72.9, mae: 54.8, r2: 0.981 },
    { fold: 4, rmse: 74.1, mae: 55.9, r2: 0.980 },
    { fold: 5, rmse: 74.3, mae: 56.0, r2: 0.980 },
  ],
  lightgbm_regressor: [
    { fold: 1, rmse: 69.5, mae: 51.9, r2: 0.983 },
    { fold: 2, rmse: 71.8, mae: 53.9, r2: 0.982 },
    { fold: 3, rmse: 72.6, mae: 54.5, r2: 0.981 },
    { fold: 4, rmse: 74.9, mae: 56.4, r2: 0.979 },
    { fold: 5, rmse: 75.2, mae: 56.6, r2: 0.979 },
  ],
  random_forest_regressor: [
    { fold: 1, rmse: 75.4, mae: 57.0, r2: 0.979 },
    { fold: 2, rmse: 77.9, mae: 58.8, r2: 0.978 },
    { fold: 3, rmse: 79.6, mae: 60.2, r2: 0.977 },
    { fold: 4, rmse: 81.3, mae: 61.5, r2: 0.976 },
    { fold: 5, rmse: 82.1, mae: 62.0, r2: 0.975 },
  ],
}

/** 图 2 · 交叉验证五折散点 — dots, 3 models × 5 folds with a mean tick. */
export const foldDotsSpec = {
  id: 'fold_dots',
  kind: 'dots',
  title: '交叉验证五折散点',
  caption: 'xgboost 与 lightgbm 的五折范围大面积重叠；random_forest 整体右移，与前两者不相交。',
  unit: 'RMSE',
  categories: Object.keys(FOLDS),
  points: Object.entries(FOLDS).map(([category, rows]) => ({
    category,
    values: rows.map((r) => r.rmse),
    rows,
  })),
  mean_marker: true,
  tooltip_fields: [
    { key: 'fold', label: '第几折', format: 'int' },
    { key: 'rmse', label: 'RMSE', format: '.1f' },
    { key: 'mae', label: 'MAE', format: '.1f' },
    { key: 'r2', label: 'R²', format: '.3f' },
  ],
  rows: Object.entries(FOLDS).flatMap(([category, rows]) => rows.map((r) => ({ category, ...r }))),
}

// 20 bins from 5000 to 14274; single peak, slight left skew, 9 samples in the last bin.
const HIST_COUNTS = [
  62, 210, 640, 1580, 3420, 6210, 9430, 12080, 13560, 12490,
  10120, 7480, 4830, 2690, 1330, 690, 310, 120, 41, 9,
]
const HIST_TOTAL = HIST_COUNTS.reduce((a, b) => a + b, 0) // 87312
const HIST_MIN = 5000
const HIST_MAX = 14274
const HIST_WIDTH = (HIST_MAX - HIST_MIN) / HIST_COUNTS.length

/** 图 3 · 负荷分布 — hist, 20 bins with mean and quartile markers. */
export const targetHistSpec = {
  id: 'target_hist',
  kind: 'hist',
  title: '负荷分布',
  caption: '单峰、略左偏，主体在 7900–9900；最大值 14274 在长尾末端，只有 9 个样本。',
  unit: 'load',
  bins: HIST_COUNTS.map((count, i) => ({
    from: Math.round((HIST_MIN + i * HIST_WIDTH) * 10) / 10,
    to: Math.round((HIST_MIN + (i + 1) * HIST_WIDTH) * 10) / 10,
    count,
    pct: Math.round((count / HIST_TOTAL) * 10000) / 100,
  })),
  markers: [
    { value: 7900, label: 'Q1' },
    { value: MEAN_LOAD, label: '均值' },
    { value: 9900, label: 'Q3' },
  ],
  tooltip_fields: [
    { key: 'range', label: '区间' },
    { key: 'count', label: '样本数', format: 'int' },
    { key: 'pct', label: '占比', format: 'percent' },
  ],
  rows: [],
}

/** 图 4 · 35 列是怎么来的 — stacked, five feature groups with their column names. */
export const fieldCompositionSpec = {
  id: 'field_composition',
  kind: 'stacked',
  title: '35 列是怎么来的',
  caption: '24 列（69%）是训练流程构造出来的。',
  unit: '列',
  segments: [
    {
      name: '原始', count: 11,
      items: ['timestamp', 'load', 'temperature', 'humidity', 'wind_speed', 'pressure',
        'cloud_cover', 'precipitation', 'is_holiday', 'region', 'price'],
    },
    {
      name: '周期', count: 8,
      items: ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'is_weekend', 'is_peak_hour'],
    },
    { name: '滞后', count: 4, items: ['load_lag_1', 'load_lag_2', 'load_lag_24', 'load_lag_168'] },
    { name: '滚动', count: 3, items: ['load_roll_mean_24', 'load_roll_std_24', 'load_roll_mean_168'] },
    {
      name: '气象衍生', count: 9,
      items: ['temp_sq', 'heat_index', 'wind_chill', 'temp_lag_1', 'temp_lag_24', 'temp_roll_mean_24',
        'humidity_lag_1', 'dew_point', 'temp_x_hour'],
    },
  ],
  tooltip_fields: [
    { key: 'name', label: '类别' },
    { key: 'count', label: '列数', format: 'int' },
    { key: 'items', label: '列名' },
  ],
  rows: [],
}

const SHAP_ROWS = [
  { feature: 'load_lag_1', mean_abs_shap: 412.6, share_of_top: 100 },
  { feature: 'load_roll_mean_24', mean_abs_shap: 41.3, share_of_top: 10.0 },
  { feature: 'load_lag_24', mean_abs_shap: 36.8, share_of_top: 8.9 },
  { feature: 'hour_sin', mean_abs_shap: 22.4, share_of_top: 5.4 },
  { feature: 'temperature', mean_abs_shap: 18.9, share_of_top: 4.6 },
  { feature: 'load_lag_168', mean_abs_shap: 15.2, share_of_top: 3.7 },
  { feature: 'load_roll_std_24', mean_abs_shap: 9.8, share_of_top: 2.4 },
  { feature: 'is_weekend', mean_abs_shap: 6.1, share_of_top: 1.5 },
]

/** 图 5 · xgboost 最依赖的 8 个特征 — hbar, one series, no error bars. */
export const shapBarsSpec = {
  id: 'shap_bars',
  kind: 'hbar',
  title: 'xgboost 最依赖的 8 个特征',
  caption: 'load_lag_1 一个特征的贡献是第二名的 10 倍；前 8 里有 5 个是负荷自身的滞后或滚动项。',
  unit: '平均绝对 SHAP',
  categories: SHAP_ROWS.map((r) => r.feature),
  series: [{ name: '平均绝对 SHAP', values: SHAP_ROWS.map((r) => r.mean_abs_shap), error: null, color_role: 'primary' }],
  reference_lines: [],
  tooltip_fields: [
    { key: 'feature', label: '特征' },
    { key: 'mean_abs_shap', label: '平均绝对 SHAP', format: '.1f' },
    { key: 'share_of_top', label: '占首位', format: 'percent' },
  ],
  rows: SHAP_ROWS,
}

// 38 epochs of a DL run: loss falls fast, flattens, best at epoch 31, early stop 31–38.
const EPOCHS = Array.from({ length: 38 }, (_, i) => i + 1)
const LOSS_ROWS = EPOCHS.map((epoch) => {
  const train = Math.round((0.0420 * Math.exp(-epoch / 6) + 0.0031 + 0.0002 * Math.sin(epoch)) * 1e5) / 1e5
  const valBase = 0.0480 * Math.exp(-epoch / 6.5) + 0.0038
  const val = Math.round((valBase + (epoch > 31 ? (epoch - 31) * 0.00004 : 0) + 0.0003 * Math.cos(epoch * 1.3)) * 1e5) / 1e5
  return { epoch, train_loss: train, val_loss: val, val_rmse: Math.round(Math.sqrt(val) * MEAN_LOAD * 0.28 * 10) / 10 }
})

/** DL 分报告 · 损失曲线 — lines on a log axis, best-epoch marker, early-stop shade. */
export const lossHistorySpec = {
  id: 'loss_history',
  kind: 'lines',
  title: '损失曲线',
  caption: '第 12 轮后验证损失基本走平，第 31 轮最优；之后 7 轮没有改善，早停生效。',
  unit: 'loss',
  x_label: 'epoch',
  x: EPOCHS,
  series: [
    { name: '训练损失', values: LOSS_ROWS.map((r) => r.train_loss) },
    { name: '验证损失', values: LOSS_ROWS.map((r) => r.val_loss) },
  ],
  y_log: true,
  markers: [{ x: 31, label: '最优轮' }],
  shade: { from: 31, to: 38, label: '早停区' },
  tooltip_fields: [
    { key: 'epoch', label: '轮', format: 'int' },
    { key: 'train_loss', label: '训练损失', format: '.5f' },
    { key: 'val_loss', label: '验证损失', format: '.5f' },
    { key: 'val_rmse', label: '验证 RMSE', format: '.1f' },
  ],
  rows: LOSS_ROWS,
}

// 120 consecutive held-out points: a daily cycle around the mean plus a small
// prediction lag, which is what the residual histogram then shows.
const PAIR_X = Array.from({ length: 120 }, (_, i) => i)
const PAIR_ACTUAL = PAIR_X.map((i) => Math.round(MEAN_LOAD + 1450 * Math.sin((i / 24) * 2 * Math.PI - 1.2) + 260 * Math.sin(i / 3.7) + 90 * Math.cos(i * 1.9)))
const PAIR_PREDICTED = PAIR_X.map((i, k) => Math.round(PAIR_ACTUAL[k] - 55 * Math.cos((i / 24) * 2 * Math.PI - 1.2) + 38 * Math.sin(i * 0.77)))
const RESIDUALS = PAIR_ACTUAL.map((a, k) => a - PAIR_PREDICTED[k])
const RES_MIN = -120
const RES_WIDTH = 20
const RESIDUAL_BINS = Array.from({ length: 12 }, (_, i) => {
  const from = RES_MIN + i * RES_WIDTH
  const to = from + RES_WIDTH
  return { from, to, count: RESIDUALS.filter((r) => r >= from && (i === 11 ? r <= to : r < to)).length }
})

/** 分报告 · 实际 vs 预测 ‖ 残差分布 — scatter_pair, the only side-by-side figure. */
export const scatterPairSpec = {
  id: 'scatter_pair',
  kind: 'scatter_pair',
  title: '实际 vs 预测 与 残差分布',
  caption: '预测曲线贴着实际走，峰值处略滞后；残差集中在 ±60 内，两侧对称。',
  unit: 'load',
  x_label: '留出集顺序',
  pair: { x: PAIR_X, actual: PAIR_ACTUAL, predicted: PAIR_PREDICTED },
  residual_bins: RESIDUAL_BINS,
  stats: { rmse: 72.5, mae: 54.9, ratio: 1.32 },
  tooltip_fields: [
    { key: 'x', label: '位置', format: 'int' },
    { key: 'actual', label: '实际', format: 'int' },
    { key: 'predicted', label: '预测', format: 'int' },
    { key: 'residual', label: '残差', format: 'int' },
  ],
  rows: [],
}

/** Every fixture, keyed by kind, for table-driven tests. */
export const specsByKind = {
  hbar: leaderboardSpec,
  dots: foldDotsSpec,
  hist: targetHistSpec,
  stacked: fieldCompositionSpec,
  lines: lossHistorySpec,
  scatter_pair: scatterPairSpec,
}

export const allSpecs = [
  leaderboardSpec, foldDotsSpec, targetHistSpec, fieldCompositionSpec, shapBarsSpec, lossHistorySpec, scatterPairSpec,
]

/** A legacy archive chart: a ready ECharts option, no `kind`. */
export const legacyOptionChart = {
  id: 'training_curves',
  title: '训练曲线（旧归档）',
  description: '旧归档由后端拼好的 ECharts option。',
  option: {
    xAxis: { type: 'category', data: ['baseline#2', 'grid_search#3'] },
    yAxis: { type: 'value' },
    series: [{ type: 'line', data: [0.9731, 0.9764] }],
  },
}

/** The overview markdown of the mock, with chart markers where the figures go. */
export const overviewMarkdown = `# 测试1-电力负荷预测 · 建模报告

## 结论

xgboost_regressor 表现最好，交叉验证 RMSE 72.5，相当于把 8897 的平均负荷预测偏差 0.8%。lightgbm 只差 0.3——这个差距比模型自身的折间波动还小，两者其实分不出高下。三个深度学习模型的误差是树模型的两倍，明显落后。

{{chart:leaderboard}}

这批 Run 用的是随机切分。数据里有滞后特征，随机切分会让"未来"混进训练，所以上面的分数比真实上线时乐观。

## 模型差距

{{chart:fold_dots}}

random_forest 落后 6.8，是树模型里唯一能确认的差距。其余两个树模型用哪个都行。

## 数据集

87312 行、35 列，目标列无缺失。

{{chart:target_hist}}

{{chart:field_composition}}

## 特征依赖

{{chart:shap_bars}}

模型主要在用"上一时刻的负荷"做预测，天气和日历只是修正。
`

/** A full overview payload in the contract shape. */
export const overviewReport = {
  task_id: '0bc692d9-0000-0000-0000-000000000000',
  archive_id: 'fb837adc-7150-4a9a-8b4a-000000000000',
  generated_at: '2026-09-04T10:12:00Z',
  model: 'doubao',
  headline: 'xgboost 胜出，误差不到平均负荷的 1%；但这批模型是随机切分训练的，分数偏乐观。',
  meta: {
    dataset_name: '电力负荷预测数据.csv',
    target_column: 'load',
    run_count: 7,
    model_count: 5,
    task_name: '测试1-电力负荷预测',
  },
  markdown: overviewMarkdown,
  charts: [leaderboardSpec, foldDotsSpec, targetHistSpec, fieldCompositionSpec, shapBarsSpec],
  appendix_tables: [
    {
      id: 'data_profile',
      title: '逐列数据概况',
      columns: [
        { key: 'column', title: '列名' }, { key: 'dtype', title: '类型' }, { key: 'missing', title: '缺失' },
        { key: 'mean', title: '均值' }, { key: 'std', title: '标准差' }, { key: 'min', title: '最小' }, { key: 'max', title: '最大' },
      ],
      rows: [
        { column: 'load', dtype: 'float64', missing: 0, mean: 8897.2, std: 1433.8, min: 5000.1, max: 14274 },
        { column: 'temperature', dtype: 'float64', missing: 12, mean: 17.4, std: 8.9, min: -9.2, max: 39.6 },
      ],
    },
    {
      id: 'parameter_settings',
      title: '参数设置',
      columns: [
        { key: 'model_type', title: '模型' }, { key: 'strategy_type', title: '策略' },
        { key: 'key_params', title: '关键参数' }, { key: 'validation_setting', title: '验证设置' },
      ],
      rows: [
        { model_type: 'xgboost_regressor', strategy_type: 'grid_search', key_params: 'n_estimators=600, max_depth=6, learning_rate=0.05', validation_setting: 'KFold(5), shuffle=True' },
        { model_type: 'mlp_dl', strategy_type: 'baseline', key_params: 'hidden=[128,64], lr=1e-3, epochs=38', validation_setting: 'holdout 20%' },
      ],
    },
  ],
  run_reports: [
    {
      run_id: 'run-xgb-0001',
      model_type: 'xgboost_regressor',
      trial_no: 3,
      validation_scheme: '交叉验证',
      markdown: '# xgboost_regressor · 分报告\n\n## 训练过程\n\n{{chart:fold_dots}}\n\n五折 RMSE 在 69.8–74.3 之间。\n\n## 训练结果\n\n{{chart:scatter_pair}}\n\nRMSE 与 MAE 的比值 1.32，误差分布没有重尾。',
      charts: [foldDotsSpec, scatterPairSpec],
    },
    {
      run_id: 'run-mlp-0001',
      model_type: 'mlp_dl',
      validation_scheme: '留出验证',
      markdown: '# mlp_dl · 分报告\n\n## 训练过程\n\n{{chart:loss_history}}\n\n第 31 轮最优，之后早停。',
      charts: [lossHistorySpec],
    },
  ],
  runs_total: 7,
  runs_reported: 2,
  best_run_id: 'run-xgb-0001',
  evidence: ['leaderboard 取自 7 个 run 的最终评估。'],
}

/** A legacy archive payload: ECharts options and `tables`, no headline/meta. */
export const legacyReport = {
  task_id: 'legacy-task',
  archive_id: 'legacy-archive-0001',
  generated_at: '2026-07-26T10:00:00Z',
  model: 'doubao',
  markdown: '# AI 建模报告\n\n**总分：60/100。** 当前不存在可直接认定的全局最优模型。\n\n## 第二章 过程与评价\n\n正文。',
  charts: [legacyOptionChart],
  tables: [{ id: 'data_profile', title: '数据概况', columns: [{ key: 'column', title: '列名' }], rows: [{ column: 'load' }] }],
  run_reports: [],
}
