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
};

export function formatDateTime(value) {
  if (!value) {
    return '-';
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
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
