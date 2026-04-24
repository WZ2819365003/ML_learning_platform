/**
 * TrainingHistoryChart — draw a multi-metric line chart from an epoch/fold
 * history dict, with series picked dynamically by task_type.
 *
 * Accepts `history` in any of these shapes:
 *   { train_loss: [...], val_loss: [...], val_acc: [...] }          // DL epoch-level
 *   { fold_idx: [...], accuracy: [...], roc_auc: [...] }            // ML CV-level
 *   [{ step: 1, loss: 0.5, acc: 0.8 }, ...]                         // array of snapshots
 *
 * Regression tasks get loss + RMSE/MAE series; classification gets loss + acc/F1.
 * This component replaces the per-page ad-hoc builders in DLResults.jsx and
 * the TrainingViz.jsx scratch file in RunInspector.
 */
import { Empty } from 'antd'
import EChart from '../EChart'

// Which metrics to plot for each task type. First key found wins.
const SERIES_PRESETS = {
  classification: [
    { key: 'train_loss', name: 'Train Loss', color: '#f97316' },
    { key: 'val_loss', name: 'Val Loss', color: '#ef4444' },
    { key: 'val_acc', name: 'Val Acc', color: '#10b981' },
    { key: 'accuracy', name: 'Accuracy', color: '#10b981' },
    { key: 'val_f1_macro', name: 'Val F1', color: '#2563eb' },
    { key: 'f1', name: 'F1', color: '#2563eb' },
    { key: 'val_auc_roc', name: 'Val AUC', color: '#8b5cf6' },
    { key: 'roc_auc', name: 'AUC', color: '#8b5cf6' },
  ],
  regression: [
    { key: 'train_loss', name: 'Train Loss', color: '#f97316' },
    { key: 'val_loss', name: 'Val Loss', color: '#ef4444' },
    { key: 'val_rmse', name: 'Val RMSE', color: '#2563eb' },
    { key: 'rmse', name: 'RMSE', color: '#2563eb' },
    { key: 'val_mae', name: 'Val MAE', color: '#8b5cf6' },
    { key: 'mae', name: 'MAE', color: '#8b5cf6' },
    { key: 'val_r2', name: 'Val R²', color: '#10b981' },
    { key: 'r2', name: 'R²', color: '#10b981' },
  ],
}

function normaliseHistory(history) {
  if (!history) return null
  // Array-of-snapshots form: [{step, ...metrics}, ...]
  if (Array.isArray(history)) {
    const keys = new Set()
    history.forEach(row => {
      Object.keys(row || {}).forEach(k => {
        if (k !== 'step' && k !== 'epoch' && k !== 'fold_idx') keys.add(k)
      })
    })
    const out = {}
    keys.forEach(k => {
      out[k] = history.map(row => row?.[k])
    })
    return out
  }
  // Dict-of-arrays form
  if (typeof history === 'object') return history
  return null
}

export default function TrainingHistoryChart({
  history,
  taskType = 'classification',
  height = 300,
}) {
  const normalised = normaliseHistory(history)
  if (!normalised) {
    return <Empty description="暂无训练过程数据" style={{ padding: 40 }} />
  }

  const presets = SERIES_PRESETS[taskType] || SERIES_PRESETS.classification

  // Pick the first length among candidate series to set the x-axis
  const activeSeries = []
  const seen = new Set()
  for (const preset of presets) {
    const arr = normalised[preset.key]
    if (Array.isArray(arr) && arr.length > 0 && !seen.has(preset.name)) {
      activeSeries.push({ ...preset, data: arr })
      seen.add(preset.name)
    }
  }

  if (activeSeries.length === 0) {
    return <Empty description="训练历史字段无已知指标（期望 train_loss / val_acc / rmse 等）" />
  }

  const xLen = Math.max(...activeSeries.map(s => s.data.length))
  const xData = Array.from({ length: xLen }, (_, i) => i + 1)

  const option = {
    tooltip: { trigger: 'axis' },
    legend: { data: activeSeries.map(s => s.name), top: 0, fontSize: 11 },
    grid: { left: 50, right: 20, top: 40, bottom: 30 },
    xAxis: { type: 'category', data: xData, name: 'epoch / fold' },
    yAxis: [
      { type: 'value', scale: true },
    ],
    series: activeSeries.map(s => ({
      name: s.name,
      type: 'line',
      smooth: true,
      showSymbol: false,
      data: s.data.map(v => (typeof v === 'number' ? Number(v.toFixed(4)) : v)),
      lineStyle: { color: s.color, width: 2 },
      itemStyle: { color: s.color },
    })),
  }

  return <EChart option={option} style={{ height }} />
}
