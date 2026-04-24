/**
 * CalibrationCurveChart — reliability (calibration) curve + diagonal +
 * ECE / Brier score summary.
 *
 * Accepts payload from GET /api/viz/:id/calibration:
 *   { prob_pred: [...], prob_true: [...], n_bins, ece, brier, positive_label }
 *
 * A well-calibrated model's curve hugs the diagonal — deviations above
 * mean under-confidence, below mean over-confidence.
 */
import { Card, Empty, Space, Statistic, Typography } from 'antd'
import EChart from '../EChart'

const { Text } = Typography

export default function CalibrationCurveChart({ payload, height = 360 }) {
  if (!payload) return <Empty description="暂无校准曲线数据" />
  const probPred = Array.isArray(payload.prob_pred) ? payload.prob_pred : []
  const probTrue = Array.isArray(payload.prob_true) ? payload.prob_true : []
  if (probPred.length === 0) return <Empty description="校准曲线无数据点" />

  const curveData = probPred.map((p, i) => [p, probTrue[i]])
  const diagonal = [[0, 0], [1, 1]]

  const option = {
    title: {
      text: '校准曲线 (Reliability Diagram)',
      left: 'center',
      textStyle: { fontSize: 13 },
    },
    tooltip: {
      trigger: 'axis',
      formatter: params => {
        const pt = params.find(p => p.seriesName === '模型') || params[0]
        return `预测概率: ${pt.value[0]}<br/>实际正类频率: ${pt.value[1]}`
      },
    },
    legend: { top: 24, data: ['模型', '完美校准'], textStyle: { fontSize: 11 } },
    grid: { left: 50, right: 20, top: 60, bottom: 40 },
    xAxis: { type: 'value', name: 'Mean predicted probability', min: 0, max: 1 },
    yAxis: { type: 'value', name: 'Fraction of positives', min: 0, max: 1 },
    series: [
      {
        name: '完美校准',
        type: 'line',
        data: diagonal,
        smooth: false,
        showSymbol: false,
        lineStyle: { type: 'dashed', color: '#9ca3af', width: 1 },
      },
      {
        name: '模型',
        type: 'line',
        data: curveData,
        smooth: true,
        symbol: 'circle',
        symbolSize: 6,
        lineStyle: { color: '#2563eb', width: 2 },
        itemStyle: { color: '#2563eb' },
      },
    ],
  }

  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      <Card variant="outlined" size="small">
        <Space wrap size={[24, 12]}>
          <Statistic
            title="ECE (期望校准误差)"
            value={payload.ece ?? 0}
            precision={4}
            valueStyle={{ fontSize: 18, color: (payload.ece ?? 0) < 0.05 ? '#10b981' : '#f59e0b' }}
          />
          <Statistic
            title="Brier Score"
            value={payload.brier ?? 0}
            precision={4}
            valueStyle={{ fontSize: 18 }}
          />
          {payload.n_bins != null && (
            <Statistic title="Bins" value={payload.n_bins} valueStyle={{ fontSize: 18 }} />
          )}
        </Space>
        <Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 8 }}>
          曲线越贴近对角线校准越好；ECE &lt; 0.05 一般视为校准良好。
        </Text>
      </Card>
      <EChart option={option} style={{ height }} />
    </Space>
  )
}
