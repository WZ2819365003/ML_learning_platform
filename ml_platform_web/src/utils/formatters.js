export const metricLabels = {
  accuracy: '准确率',
  f1: 'F1',
  precision: '精确率',
  recall: '召回率',
  roc_auc: 'ROC AUC',
  cv_avg_accuracy: '交叉验证准确率',
  cv_avg_f1: '交叉验证 F1',
  cv_std_accuracy: '准确率波动',
  cv_std_f1: 'F1 波动',
  val_acc: '验证准确率',
  val_f1_macro: '验证 F1',
  val_auc_roc: '验证 AUC',
  val_precision: '验证精确率',
  val_recall: '验证召回率',
  val_rmse: '验证 RMSE',
  val_mae: '验证 MAE',
  val_r2: '验证 R2',
  val_mape: '验证 MAPE',
  val_loss: '验证损失',
  best_val_loss: '最佳验证损失',
};

/**
 * Parse a timestamp coming from the API into a Date.
 *
 * The backend stores UTC but serializes DB columns *without* an offset —
 * `created_at` arrives as "2026-08-25 12:52:01". `new Date()` reads a
 * timezone-less string as **local** time, so in UTC+8 every fresh log entry
 * was dated 8 hours into the past: a log written 30 seconds ago rendered as
 * "8小时前", which made a perfectly live log panel look completely stale.
 *
 * Values that already carry a zone (the WebSocket payloads send
 * "…T13:24:14+00:00") are left alone, as are date-only strings, which the ECMA
 * spec already defines as UTC.
 */
export function parseServerDate(value) {
  if (value === null || value === undefined || value === '') return null;
  if (value instanceof Date) return Number.isNaN(value.getTime()) ? null : value;

  let text = String(value).trim();
  const hasTimePart = /\d{2}:\d{2}/.test(text);
  const hasZone = /(?:[zZ]|[+-]\d{2}:?\d{2})$/.test(text);
  if (hasTimePart && !hasZone) {
    text = `${text.replace(' ', 'T')}Z`;
  }

  const date = new Date(text);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function formatDateTime(value) {
  if (!value) {
    return '-';
  }

  const date = parseServerDate(value);
  if (date === null) {
    return String(value);
  }

  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

export function formatMetric(value, { percent = false, digits = 4 } = {}) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '-';
  }

  const numericValue = Number(value);
  if (percent) {
    return `${(numericValue * 100).toFixed(2)}%`;
  }

  return numericValue.toFixed(digits);
}

/**
 * Return a percentage value for metrics whose unit is percent.
 *
 * Accuracy-like metrics and MAPE can arrive from different trainers either as
 * a ratio (0.0118) or an already-scaled percentage (1.18). Normalising at the
 * display boundary keeps the shared result page and model-management detail in
 * the same unit without changing persisted training output.
 */
export function percentageMetricValue(key, value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return null;
  if (!/(^|_)(acc|accuracy|precision|recall|f1|auc|mape)(_|$)/i.test(String(key))) return null;

  const numericValue = Number(value);
  return Math.abs(numericValue) <= 1 ? numericValue * 100 : numericValue;
}

export function formatMetricByKey(key, value, { digits = 4, percentageDigits = 2 } = {}) {
  const percentageValue = percentageMetricValue(key, value);
  if (percentageValue !== null) return `${percentageValue.toFixed(percentageDigits)}%`;
  return formatMetric(value, { digits });
}

export function formatBytes(bytes) {
  if (!bytes && bytes !== 0) {
    return '-';
  }

  const units = ['B', 'KB', 'MB', 'GB'];
  let value = Number(bytes);
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

export function pickPrimaryMetric(metrics = {}) {
  if (typeof metrics.accuracy === 'number') {
    return { key: 'accuracy', value: metrics.accuracy };
  }

  const entry = Object.entries(metrics).find(([, value]) => typeof value === 'number');
  if (!entry) {
    return null;
  }

  return { key: entry[0], value: entry[1] };
}
