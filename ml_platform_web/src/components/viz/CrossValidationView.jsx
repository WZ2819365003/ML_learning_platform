/**
 * CrossValidationView — K-Fold CV stability view for ML training.
 *
 * Replaces the old «multi-metric line chart on fold steps» — folds are
 * NOT a temporal sequence, so a line chart misrepresented them.  Instead
 * we render three pieces stacked vertically:
 *
 *   1. Per-metric summary cards: mean ± std (one Statistic each), color
 *      by metric.  Tells you «how well the model performed on average,
 *      and how much it wobbled across folds».
 *
 *   2. Bar chart: x = fold index, y = metric value, one series per metric.
 *      A horizontal mean line per metric is drawn via markLine so the
 *      eye can read «which folds were above/below average».
 *
 *   3. Per-fold table: rows = folds, cols = metrics. Verifies the chart
 *      with raw numbers.
 *
 * Accepts `payload` in the shape returned by GET /api/viz/:id/learning_curve:
 *   {
 *     steps: [
 *       { step: 1, total_steps: 5, metrics: { accuracy, f1, roc_auc, fold } },
 *       ...
 *     ],
 *     model_type, ...
 *   }
 *
 * Props:
 *   - payload: the learning_curve API response
 *   - taskKind: 'classification' | 'regression' — picks which metrics to
 *     surface and how to color «good» values
 *   - height: chart height (default 320)
 */
import { useMemo } from 'react'
import { Card, Col, Empty, Row, Space, Statistic, Table, Typography } from 'antd'
import EChart from '../EChart'

const { Text, Paragraph } = Typography

// Metric ordering + display names. Higher-is-better is encoded in `direction`
// so the summary card can color-code the value.
const METRIC_PRESETS = {
  classification: [
    { key: 'accuracy', name: 'Accuracy', color: '#10b981', direction: 'max' },
    { key: 'f1',       name: 'F1',       color: '#2563eb', direction: 'max' },
    { key: 'roc_auc',  name: 'ROC AUC',  color: '#8b5cf6', direction: 'max' },
    { key: 'precision',name: 'Precision',color: '#f59e0b', direction: 'max' },
    { key: 'recall',   name: 'Recall',   color: '#06b6d4', direction: 'max' },
  ],
  regression: [
    { key: 'r2',   name: 'R²',   color: '#10b981', direction: 'max' },
    { key: 'rmse', name: 'RMSE', color: '#ef4444', direction: 'min' },
    { key: 'mae',  name: 'MAE',  color: '#f59e0b', direction: 'min' },
    { key: 'mse',  name: 'MSE',  color: '#7c3aed', direction: 'min' },
  ],
}

function mean(arr) {
  const n = arr.length
  if (!n) return null
  return arr.reduce((a, b) => a + b, 0) / n
}

function std(arr) {
  const n = arr.length
  if (n < 2) return 0
  const m = mean(arr)
  return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / (n - 1))
}

function fmt(v, digits = 4) {
  if (v == null || !Number.isFinite(v)) return '-'
  return Number(v).toFixed(digits)
}

export default function CrossValidationView({ payload, taskKind = 'classification', height = 320 }) {
  const steps = Array.isArray(payload?.steps) ? payload.steps : []

  const presets = METRIC_PRESETS[taskKind] || METRIC_PRESETS.classification

  // Find which metrics actually have data, in preset order.
  const activeMetrics = useMemo(() => {
    return presets
      .map(p => {
        const values = steps
          .map(s => s.metrics?.[p.key])
          .filter(v => typeof v === 'number' && Number.isFinite(v))
        if (values.length === 0) return null
        return { ...p, values, mean: mean(values), std: std(values) }
      })
      .filter(Boolean)
  }, [steps, presets])

  if (steps.length === 0 || activeMetrics.length === 0) {
    return <Empty description="暂无 CV 数据（仅 K-Fold 训练任务可用）" />
  }

  const folds = steps.map((s, i) => s.metrics?.fold ?? s.step ?? (i + 1))

  // Bar chart: grouped bars per fold, one cluster per metric. We normalize
  // 'min' metrics for visualisation by NOT scaling — RMSE/MAE/MSE coexist with
  // R² on the same axis poorly, so we put them on a second y-axis.
  const isMinMetric = (key) => ['rmse', 'mae', 'mse'].includes(key)

  const seriesData = activeMetrics.map((m, idx) => ({
    name: m.name,
    type: 'bar',
    yAxisIndex: isMinMetric(m.key) ? 1 : 0,
    data: m.values.map(v => Number(v.toFixed(6))),
    itemStyle: { color: m.color },
    barGap: '10%',
    markLine: {
      symbol: 'none',
      lineStyle: { type: 'dashed', color: m.color, opacity: 0.65, width: 1 },
      label: {
        formatter: `${m.name} 均值 ${fmt(m.mean)}`,
        fontSize: 10, color: m.color,
      },
      data: [{ yAxis: m.mean }],
    },
  }))

  // Use dual y-axis only when metrics from both sides exist.
  const hasMaxMetric = activeMetrics.some(m => !isMinMetric(m.key))
  const hasMinMetric = activeMetrics.some(m => isMinMetric(m.key))
  const yAxes = []
  if (hasMaxMetric) {
    yAxes.push({
      type: 'value',
      name: taskKind === 'regression' ? 'R²' : '指标',
      ...(taskKind === 'classification' ? { min: 0, max: 1 } : {}),
      nameTextStyle: { fontSize: 11 },
    })
  }
  if (hasMinMetric) {
    yAxes.push({
      type: 'value',
      name: '误差',
      nameTextStyle: { fontSize: 11 },
    })
  }
  // When only one side exists, every series must point to index 0.
  if (!hasMinMetric) seriesData.forEach(s => { s.yAxisIndex = 0 })
  if (!hasMaxMetric) seriesData.forEach(s => { s.yAxisIndex = 0 })

  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (params) => {
        const fold = params[0]?.axisValueLabel
        const lines = params.map(p => `<span style="color:${p.color}">●</span> ${p.seriesName}: <b>${fmt(p.value)}</b>`)
        return [`<b>Fold ${fold}</b>`, ...lines].join('<br/>')
      },
    },
    legend: { data: activeMetrics.map(m => m.name), top: 0, fontSize: 11 },
    grid: { left: 60, right: hasMinMetric ? 70 : 30, top: 36, bottom: 30 },
    xAxis: {
      type: 'category',
      data: folds.map(f => `${f}`),
      name: 'Fold',
      nameLocation: 'middle',
      nameGap: 22,
    },
    yAxis: yAxes,
    series: seriesData,
  }

  // Per-fold table: rows = folds, cols = metrics.
  const tableColumns = [
    {
      title: 'Fold',
      dataIndex: 'fold',
      key: 'fold',
      align: 'center',
      width: 70,
      render: v => <Text code>{v}</Text>,
    },
    ...activeMetrics.map(m => ({
      title: m.name,
      dataIndex: m.key,
      key: m.key,
      align: 'right',
      render: v => fmt(v),
    })),
    {
      title: '与均值偏差',
      key: '__delta',
      align: 'right',
      render: (_, row) => {
        // Use the first active metric as the reference for «delta».
        const ref = activeMetrics[0]
        if (!ref) return '-'
        const v = row[ref.key]
        if (typeof v !== 'number') return '-'
        const d = v - ref.mean
        const sign = d >= 0 ? '+' : ''
        const color = Math.abs(d) > ref.std ? '#ef4444' : '#64748b'
        return <Text style={{ color, fontSize: 12 }}>{sign}{fmt(d)}</Text>
      },
    },
  ]

  const tableRows = folds.map((f, i) => {
    const row = { key: `fold-${f}`, fold: f }
    activeMetrics.forEach(m => { row[m.key] = m.values[i] })
    return row
  })

  // Summary footer row showing mean ± std per metric.
  const summaryRow = {
    key: '__summary',
    fold: '均值 ± std',
    isSummary: true,
  }
  activeMetrics.forEach(m => {
    summaryRow[m.key] = `${fmt(m.mean)} ± ${fmt(m.std)}`
  })

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      {/* 1. Per-metric mean ± std cards */}
      <Row gutter={[12, 12]}>
        {activeMetrics.map(m => {
          const cv = m.mean ? Math.abs(m.std / m.mean) : 0
          // CV (coefficient of variation) > 5% on a max-metric is the typical
          // «is this CV stable enough?» rule of thumb.
          const isStable = cv < 0.05
          return (
            <Col xs={12} md={Math.max(4, Math.floor(24 / Math.max(activeMetrics.length, 1)))} key={m.key}>
              <Card
                size="small"
                styles={{ body: { padding: '12px 16px' } }}
                style={{ borderTop: `3px solid ${m.color}`, borderRadius: 8 }}
              >
                <Statistic
                  title={<span style={{ fontSize: 12 }}>{m.name}（K-Fold 均值）</span>}
                  value={m.mean}
                  precision={4}
                  valueStyle={{ color: m.color, fontSize: 18 }}
                />
                <Text type="secondary" style={{ fontSize: 11 }}>
                  ± {fmt(m.std)} （CV {(cv * 100).toFixed(1)}%
                  {isStable ? '，稳定' : '，波动较大'}）
                </Text>
              </Card>
            </Col>
          )
        })}
      </Row>

      {/* 2. Bar chart with mean reference line */}
      <Card
        size="small"
        title={<Text strong style={{ fontSize: 13 }}>各折表现（横向虚线为该指标均值）</Text>}
        styles={{ body: { padding: 12 } }}
      >
        <EChart option={option} style={{ height }} />
      </Card>

      {/* 3. Per-fold table */}
      <Card
        size="small"
        title={<Text strong style={{ fontSize: 13 }}>逐折数值</Text>}
        styles={{ body: { padding: 0 } }}
      >
        <Table
          columns={tableColumns}
          dataSource={tableRows}
          pagination={false}
          size="small"
          summary={() => (
            <Table.Summary fixed>
              <Table.Summary.Row style={{ background: '#f8fafc', fontWeight: 600 }}>
                <Table.Summary.Cell index={0} align="center">
                  <Text strong>均值 ± std</Text>
                </Table.Summary.Cell>
                {activeMetrics.map((m, idx) => (
                  <Table.Summary.Cell index={idx + 1} key={m.key} align="right">
                    <Text strong style={{ color: m.color }}>
                      {fmt(m.mean)} ± {fmt(m.std)}
                    </Text>
                  </Table.Summary.Cell>
                ))}
                <Table.Summary.Cell index={activeMetrics.length + 1} align="right">
                  <Text type="secondary" style={{ fontSize: 11 }}>K = {steps.length}</Text>
                </Table.Summary.Cell>
              </Table.Summary.Row>
            </Table.Summary>
          )}
        />
      </Card>

      <Paragraph type="secondary" style={{ marginBottom: 0, fontSize: 12 }}>
        K-Fold 交叉验证用 K 个相互独立的训练 / 验证划分评估稳定性。
        均值反映整体水平，标准差越小说明模型对数据划分越不敏感。
        <Text strong> CV 系数 ≥ 5%</Text> 通常提示模型对训练子集敏感，需要更多数据或正则化。
      </Paragraph>
    </Space>
  )
}
