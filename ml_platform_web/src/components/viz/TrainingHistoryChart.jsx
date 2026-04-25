/**
 * TrainingHistoryChart — multi-metric line chart for DL epoch history.
 *
 * Accepts `history` in any of these shapes:
 *   { train_loss: [...], val_loss: [...], val_acc: [...] }          // DL epoch-level
 *   { fold_idx: [...], accuracy: [...], roc_auc: [...] }            // ML CV-level (legacy)
 *   [{ step: 1, loss: 0.5, acc: 0.8 }, ...]                         // array of snapshots
 *   [{ epoch: 1, train_loss, val_loss, val_acc, ... }, ...]         // /api/dl/{id}/epochs row form
 *
 * Loss series go on the LEFT y-axis, accuracy/AUC/F1/RMSE/MAE go on the
 * RIGHT y-axis — they have very different scales and forcing them onto
 * one axis flattens the loss curve into the floor.
 *
 * Regression tasks get loss + RMSE/MAE/R² series; classification gets
 * loss + acc/F1/AUC.  Each preset entry declares which axis it belongs
 * to so we can split cleanly.
 */
import { useMemo } from 'react'
import { Empty } from 'antd'
import EChart from '../EChart'

// Each preset entry: which key in the dict, the display name, the axis it
// belongs to ('loss' = left, 'metric' = right), and a colour.
const SERIES_PRESETS = {
  classification: [
    { key: 'train_loss',   name: 'Train Loss', color: '#f97316', axis: 'loss' },
    { key: 'val_loss',     name: 'Val Loss',   color: '#ef4444', axis: 'loss' },
    { key: 'val_acc',      name: 'Val Acc',    color: '#10b981', axis: 'metric' },
    { key: 'accuracy',     name: 'Accuracy',   color: '#10b981', axis: 'metric' },
    { key: 'val_f1_macro', name: 'Val F1',     color: '#2563eb', axis: 'metric' },
    { key: 'f1',           name: 'F1',         color: '#2563eb', axis: 'metric' },
    { key: 'val_auc_roc',  name: 'Val AUC',    color: '#8b5cf6', axis: 'metric' },
    { key: 'roc_auc',      name: 'AUC',        color: '#8b5cf6', axis: 'metric' },
  ],
  regression: [
    { key: 'train_loss', name: 'Train Loss', color: '#f97316', axis: 'loss' },
    { key: 'val_loss',   name: 'Val Loss',   color: '#ef4444', axis: 'loss' },
    { key: 'val_rmse',   name: 'Val RMSE',   color: '#2563eb', axis: 'metric' },
    { key: 'rmse',       name: 'RMSE',       color: '#2563eb', axis: 'metric' },
    { key: 'val_mae',    name: 'Val MAE',    color: '#8b5cf6', axis: 'metric' },
    { key: 'mae',        name: 'MAE',        color: '#8b5cf6', axis: 'metric' },
    { key: 'val_r2',     name: 'Val R²',     color: '#10b981', axis: 'metric' },
    { key: 'r2',         name: 'R²',         color: '#10b981', axis: 'metric' },
  ],
}

function normaliseHistory(history) {
  if (!history) return null
  // Array-of-snapshots form: [{epoch|step|fold_idx, ...metrics}, ...]
  if (Array.isArray(history)) {
    const keys = new Set()
    history.forEach(row => {
      Object.keys(row || {}).forEach(k => {
        if (!['step', 'epoch', 'fold_idx', 'fold', 'id', 'task_id', 'created_at', 'total_epochs'].includes(k)) {
          keys.add(k)
        }
      })
    })
    const out = {}
    keys.forEach(k => {
      out[k] = history.map(row => {
        const v = row?.[k]
        return typeof v === 'number' ? v : null
      })
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
  height = 320,
  xAxisName = 'Epoch',
}) {
  const normalised = normaliseHistory(history)

  const { activeSeries, xLen } = useMemo(() => {
    if (!normalised) return { activeSeries: [], xLen: 0 }
    const presets = SERIES_PRESETS[taskType] || SERIES_PRESETS.classification
    const seen = new Set()
    const series = []
    for (const preset of presets) {
      const arr = normalised[preset.key]
      if (Array.isArray(arr) && arr.length > 0 && !seen.has(preset.name)) {
        // Filter out series that are entirely null (e.g. val_rmse on a
        // classification task) — keeps the legend clean.
        const numeric = arr.filter(v => typeof v === 'number' && Number.isFinite(v))
        if (numeric.length === 0) continue
        series.push({ ...preset, data: arr })
        seen.add(preset.name)
      }
    }
    const len = series.length ? Math.max(...series.map(s => s.data.length)) : 0
    return { activeSeries: series, xLen: len }
  }, [normalised, taskType])

  if (!normalised) {
    return <Empty description="暂无训练过程数据" style={{ padding: 40 }} />
  }
  if (activeSeries.length === 0) {
    return <Empty description="训练历史字段无已知指标（期望 train_loss / val_acc / rmse 等）" />
  }

  const xData = Array.from({ length: xLen }, (_, i) => i + 1)

  const hasLoss = activeSeries.some(s => s.axis === 'loss')
  const hasMetric = activeSeries.some(s => s.axis === 'metric')

  // Build y-axes: left = loss (always log-friendly scale), right = metric.
  // When only one side exists, we still emit a single axis so series indices stay valid.
  const yAxes = []
  if (hasLoss) {
    yAxes.push({
      type: 'value',
      name: 'Loss',
      nameTextStyle: { fontSize: 11, color: '#ef4444' },
      scale: true,
      axisLine: { show: true, lineStyle: { color: '#ef4444' } },
      splitLine: { lineStyle: { type: 'dashed', opacity: 0.3 } },
    })
  }
  if (hasMetric) {
    yAxes.push({
      type: 'value',
      name: taskType === 'regression' ? '指标 / 误差' : '指标',
      nameTextStyle: { fontSize: 11, color: '#10b981' },
      scale: true,
      axisLine: { show: true, lineStyle: { color: '#10b981' } },
      splitLine: { show: false },
      ...(taskType === 'classification' ? { min: 0, max: 1 } : {}),
    })
  }

  // axis index per series. When one side is absent, all series fall to index 0.
  const lossIdx = hasLoss ? 0 : null
  const metricIdx = hasMetric ? (hasLoss ? 1 : 0) : null

  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' },
      formatter: (params) => {
        const epoch = params[0]?.axisValueLabel
        const lines = params
          .filter(p => p.value != null)
          .map(p => `<span style="color:${p.color}">●</span> ${p.seriesName}: <b>${typeof p.value === 'number' ? p.value.toFixed(4) : p.value}</b>`)
        return [`<b>${xAxisName} ${epoch}</b>`, ...lines].join('<br/>')
      },
    },
    legend: {
      data: activeSeries.map(s => s.name),
      top: 0,
      fontSize: 11,
    },
    grid: {
      left: 56,
      right: hasLoss && hasMetric ? 56 : 24,
      top: 36,
      bottom: 30,
    },
    xAxis: {
      type: 'category',
      data: xData,
      name: xAxisName,
      nameLocation: 'middle',
      nameGap: 22,
      nameTextStyle: { fontSize: 11 },
    },
    yAxis: yAxes,
    series: activeSeries.map(s => ({
      name: s.name,
      type: 'line',
      smooth: true,
      showSymbol: false,
      yAxisIndex: s.axis === 'loss' ? lossIdx : metricIdx,
      data: s.data.map(v => (typeof v === 'number' ? Number(v.toFixed(4)) : v)),
      lineStyle: { color: s.color, width: 2 },
      itemStyle: { color: s.color },
      // Mark min for loss (where the model converged best) and max for metric.
      markPoint: s.axis === 'loss'
        ? {
            symbolSize: 36,
            itemStyle: { color: s.color, opacity: 0.85 },
            data: [{ type: 'min', name: '最低', label: { fontSize: 10 } }],
          }
        : {
            symbolSize: 36,
            itemStyle: { color: s.color, opacity: 0.85 },
            data: [{ type: 'max', name: '最高', label: { fontSize: 10 } }],
          },
    })),
  }

  return <EChart option={option} style={{ height }} />
}
