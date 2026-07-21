/* eslint-disable react-refresh/only-export-components */
import { Card, Col, Empty, Row, Space } from 'antd'
import EChart from '../EChart'

function chartSeries(name, data, color) {
  return {
    name,
    type: 'line',
    smooth: true,
    showSymbol: false,
    data,
    lineStyle: { width: 2 },
    itemStyle: { color },
  }
}

export function buildLossHistoryOption(history = []) {
  const epochs = history.map((row, index) => row?.epoch ?? row?.step ?? index + 1)
  return {
    tooltip: { trigger: 'axis' },
    legend: { data: ['训练损失', '验证损失'], top: 0 },
    grid: { top: 40, left: 56, right: 24, bottom: 46 },
    xAxis: { type: 'category', data: epochs, name: 'Epoch', nameLocation: 'middle', nameGap: 28 },
    yAxis: { type: 'value', name: 'Loss', scale: true },
    series: [
      chartSeries('训练损失', history.map(row => row?.train_loss ?? null), '#f97316'),
      chartSeries('验证损失', history.map(row => row?.val_loss ?? null), '#ef4444'),
    ],
  }
}

export function buildTaskMetricHistoryOption(history = [], taskType = 'classification') {
  const epochs = history.map((row, index) => row?.epoch ?? row?.step ?? index + 1)
  const candidates = taskType === 'regression'
    ? [
        ['val_rmse', '验证 RMSE', '#2563eb', 0],
        ['val_mae', '验证 MAE', '#8b5cf6', 0],
        ['val_r2', '验证 R²', '#10b981', 1],
      ]
    : [
        ['val_acc', '验证准确率', '#10b981', 0],
        ['val_f1_macro', '验证 F1', '#2563eb', 0],
      ]
  const active = candidates.filter(([key]) =>
    history.some(row => Number.isFinite(row?.[key])))
  const hasR2 = taskType === 'regression' && active.some(([, , , axis]) => axis === 1)

  return {
    tooltip: { trigger: 'axis' },
    legend: { data: active.map(([, name]) => name), top: 0 },
    grid: { top: 40, left: 60, right: hasR2 ? 58 : 24, bottom: 46 },
    xAxis: { type: 'category', data: epochs, name: 'Epoch', nameLocation: 'middle', nameGap: 28 },
    yAxis: taskType === 'regression'
      ? [
          { type: 'value', name: '误差', scale: true },
          ...(hasR2 ? [{ type: 'value', name: 'R²', min: -1, max: 1, splitLine: { show: false } }] : []),
        ]
      : { type: 'value', name: '指标', min: 0, max: 1 },
    series: active.map(([key, name, color, axis]) => ({
      ...chartSeries(name, history.map(row => row?.[key] ?? null), color),
      ...(taskType === 'regression' ? { yAxisIndex: hasR2 ? axis : 0 } : {}),
    })),
  }
}

function buildConfusionMatrixOption(matrix) {
  const labels = matrix.map((_, index) => String(index))
  return {
    tooltip: { formatter: ({ value }) => `真实=${labels[value[1]]}<br/>预测=${labels[value[0]]}<br/>数量=${value[2]}` },
    grid: { top: 28, left: 64, right: 24, bottom: 64 },
    xAxis: { type: 'category', data: labels, name: '预测类别' },
    yAxis: { type: 'category', data: labels, name: '真实类别' },
    visualMap: {
      min: 0,
      max: Math.max(...matrix.flat(), 1),
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 0,
      inRange: { color: ['#e0f2fe', '#2563eb', '#1e3a8a'] },
    },
    series: [{
      type: 'heatmap',
      data: matrix.flatMap((row, rowIndex) => row.map((value, columnIndex) => [columnIndex, rowIndex, value])),
      label: { show: true },
    }],
  }
}

function buildRocOption(fpr, tpr, auc) {
  return {
    tooltip: { trigger: 'axis' },
    legend: { data: [`ROC (AUC ${Number(auc).toFixed(3)})`, '随机基线'], bottom: 0 },
    grid: { top: 24, left: 56, right: 24, bottom: 52 },
    xAxis: { type: 'value', name: 'FPR', min: 0, max: 1 },
    yAxis: { type: 'value', name: 'TPR', min: 0, max: 1 },
    series: [
      {
        name: `ROC (AUC ${Number(auc).toFixed(3)})`,
        type: 'line',
        showSymbol: false,
        data: fpr.map((value, index) => [value, tpr[index]]),
        lineStyle: { width: 2, color: '#2563eb' },
        areaStyle: { color: 'rgba(37,99,235,0.12)' },
      },
      {
        name: '随机基线', type: 'line', symbol: 'none',
        data: [[0, 0], [1, 1]], lineStyle: { type: 'dashed', color: '#94a3b8' },
      },
    ],
  }
}

function buildConfidenceOption(values) {
  const bins = 20
  const counts = Array(bins).fill(0)
  values.forEach(value => { counts[Math.min(Math.floor(value * bins), bins - 1)] += 1 })
  return {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { top: 20, left: 56, right: 24, bottom: 58 },
    xAxis: {
      type: 'category',
      data: counts.map((_, index) => `${(index / bins).toFixed(2)}-${((index + 1) / bins).toFixed(2)}`),
      name: '预测置信度',
      axisLabel: { rotate: 30, fontSize: 10 },
    },
    yAxis: { type: 'value', name: '样本数' },
    series: [{ type: 'bar', data: counts, itemStyle: { color: '#6366f1' } }],
  }
}

function buildPredictionScatterOption(scatter) {
  const all = [...scatter.actual, ...scatter.predicted]
  const min = Math.min(...all)
  const max = Math.max(...all)
  return {
    tooltip: { formatter: ({ value }) => `实际=${value[0]}<br/>预测=${value[1]}` },
    grid: { top: 24, left: 60, right: 24, bottom: 48 },
    xAxis: { type: 'value', name: '实际值', scale: true },
    yAxis: { type: 'value', name: '预测值', scale: true },
    series: [
      {
        name: '样本', type: 'scatter', symbolSize: 6,
        data: scatter.actual.map((value, index) => [value, scatter.predicted[index]]),
        itemStyle: { color: 'rgba(14,165,233,0.65)' },
      },
      {
        name: '理想预测', type: 'line', symbol: 'none', data: [[min, min], [max, max]],
        lineStyle: { type: 'dashed', color: '#ef4444' },
      },
    ],
  }
}

export function buildResidualOption(scatter) {
  return {
    tooltip: { formatter: ({ value }) => `预测=${value[0]}<br/>残差=${value[1]}` },
    grid: { top: 24, left: 60, right: 24, bottom: 48 },
    xAxis: { type: 'value', name: '预测值', scale: true },
    yAxis: { type: 'value', name: '残差（实际-预测）', scale: true },
    series: [{
      name: '残差',
      type: 'scatter',
      symbolSize: 6,
      data: scatter.actual.map((actual, index) => [
        scatter.predicted[index],
        actual - scatter.predicted[index],
      ]),
      itemStyle: { color: 'rgba(139,92,246,0.65)' },
      markLine: {
        silent: true,
        symbol: 'none',
        data: [{ yAxis: 0 }],
        lineStyle: { type: 'dashed', color: '#ef4444' },
      },
    }],
  }
}

function DiagnosticCard({ title, option, height = 300 }) {
  return (
    <Card size="small" title={title} styles={{ body: { padding: 12 } }}>
      {option?.series?.length
        ? <EChart option={option} style={{ height }} />
        : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无数据" style={{ padding: 48 }} />}
    </Card>
  )
}

export default function DLDiagnostics({ metrics = {}, taskType = 'classification' }) {
  const history = Array.isArray(metrics.history) ? metrics.history : []
  const matrix = Array.isArray(metrics.confusion_matrix) ? metrics.confusion_matrix : null
  const fpr = Array.isArray(metrics.val_roc_fpr) ? metrics.val_roc_fpr : null
  const tpr = Array.isArray(metrics.val_roc_tpr) ? metrics.val_roc_tpr : null
  const confidence = Array.isArray(metrics.val_confidence_dist) ? metrics.val_confidence_dist : null
  const scatter = metrics.val_scatter?.actual?.length && metrics.val_scatter?.predicted?.length
    ? metrics.val_scatter
    : null

  if (!history.length && !matrix && !fpr && !confidence && !scatter) {
    return <Empty description="暂无训练过程与评估数据" />
  }

  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      {history.length > 0 && (
        <Row gutter={[12, 12]}>
          <Col xs={24} xl={12}>
            <DiagnosticCard title="训练与验证损失" option={buildLossHistoryOption(history)} />
          </Col>
          <Col xs={24} xl={12}>
            <DiagnosticCard
              title={taskType === 'regression' ? '回归验证指标' : '分类验证指标'}
              option={buildTaskMetricHistoryOption(history, taskType)}
            />
          </Col>
        </Row>
      )}

      {taskType === 'classification' && (matrix || (fpr && tpr)) && (
        <Row gutter={[12, 12]}>
          {matrix && <Col xs={24} xl={12}><DiagnosticCard title="混淆矩阵" option={buildConfusionMatrixOption(matrix)} /></Col>}
          {fpr && tpr && (
            <Col xs={24} xl={12}>
              <DiagnosticCard title="ROC 曲线" option={buildRocOption(fpr, tpr, metrics.val_auc_roc)} />
            </Col>
          )}
        </Row>
      )}

      {taskType === 'classification' && confidence && (
        <DiagnosticCard title="预测置信度分布" option={buildConfidenceOption(confidence)} height={260} />
      )}

      {taskType === 'regression' && scatter && (
        <Row gutter={[12, 12]}>
          <Col xs={24} xl={12}>
            <DiagnosticCard title="预测值 vs 实际值" option={buildPredictionScatterOption(scatter)} />
          </Col>
          <Col xs={24} xl={12}>
            <DiagnosticCard title="残差图" option={buildResidualOption(scatter)} />
          </Col>
        </Row>
      )}
    </Space>
  )
}
