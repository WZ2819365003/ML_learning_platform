/**
 * PredictionDistributionChart — renders one of three distributions based
 * on payload.kind:
 *
 *   - "classification_binary_proba": stacked histogram of P(positive)
 *     split by true class; a good model shows two well-separated humps.
 *   - "classification_confidence_multiclass": histogram of max-prob per
 *     sample (model confidence).
 *   - "regression_residuals": residual histogram + Statistic strip
 *     (mean / std / min / max).
 *
 * Accepts payload from GET /api/viz/:id/distribution.
 */
import { Card, Empty, Space, Statistic, Tag, Typography } from 'antd'
import EChart from '../EChart'

const { Text } = Typography

function binCenters(edges) {
  if (!Array.isArray(edges) || edges.length < 2) return []
  const out = []
  for (let i = 0; i < edges.length - 1; i += 1) {
    out.push(Number(((edges[i] + edges[i + 1]) / 2).toFixed(4)))
  }
  return out
}

export default function PredictionDistributionChart({ payload, height = 340 }) {
  if (!payload) return <Empty description="暂无预测分布数据" />

  const centers = binCenters(payload.bin_edges)
  if (centers.length === 0) return <Empty description="分布数据为空" />

  if (payload.kind === 'regression_residuals') {
    const counts = payload.counts || []
    const option = {
      title: { text: '残差分布 (y_true − y_pred)', left: 'center', textStyle: { fontSize: 13 } },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: 50, right: 20, top: 50, bottom: 40 },
      xAxis: { type: 'category', data: centers.map(String), name: 'residual' },
      yAxis: { type: 'value', name: 'count' },
      series: [
        {
          name: 'residual',
          type: 'bar',
          data: counts,
          itemStyle: { color: '#2563eb' },
        },
      ],
    }
    return (
      <Space direction="vertical" size={12} style={{ width: '100%' }}>
        <Card variant="outlined" size="small">
          <Space wrap size={[24, 12]}>
            <Statistic title="mean" value={payload.mean ?? 0} precision={4} valueStyle={{ fontSize: 18 }} />
            <Statistic title="std" value={payload.std ?? 0} precision={4} valueStyle={{ fontSize: 18 }} />
            <Statistic title="min" value={payload.min ?? 0} precision={4} valueStyle={{ fontSize: 18 }} />
            <Statistic title="max" value={payload.max ?? 0} precision={4} valueStyle={{ fontSize: 18 }} />
          </Space>
          <Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 8 }}>
            残差应以 0 为中心、近正态；若偏斜明显，说明模型有系统性偏差。
          </Text>
        </Card>
        <EChart option={option} style={{ height }} />
      </Space>
    )
  }

  if (payload.kind === 'classification_binary_proba') {
    const posCounts = payload.positive_counts || []
    const negCounts = payload.negative_counts || []
    const option = {
      title: {
        text: '正类概率分布 (按真实类别分列)',
        left: 'center',
        textStyle: { fontSize: 13 },
      },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { top: 24, data: ['正类样本', '负类样本'], textStyle: { fontSize: 11 } },
      grid: { left: 50, right: 20, top: 60, bottom: 40 },
      xAxis: { type: 'category', data: centers.map(v => Number(v).toFixed(2)), name: 'P(正类)' },
      yAxis: { type: 'value', name: 'count' },
      series: [
        {
          name: '正类样本',
          type: 'bar',
          stack: 'dist',
          data: posCounts,
          itemStyle: { color: '#10b981' },
        },
        {
          name: '负类样本',
          type: 'bar',
          stack: 'dist',
          data: negCounts,
          itemStyle: { color: '#ef4444' },
        },
      ],
    }
    return (
      <Space direction="vertical" size={12} style={{ width: '100%' }}>
        <Card variant="outlined" size="small">
          <Space wrap size={[12, 8]}>
            <div>
              <div style={{ fontSize: 12, color: '#8c8c8c', marginBottom: 4 }}>正类</div>
              <Tag color="green">{String(payload.positive_label ?? '-')}</Tag>
            </div>
            <Text type="secondary" style={{ fontSize: 12 }}>
              理想情况下正类样本应集中在右侧、负类集中在左侧——两峰重叠越少模型判别越强。
            </Text>
          </Space>
        </Card>
        <EChart option={option} style={{ height }} />
      </Space>
    )
  }

  // Multiclass confidence
  const counts = payload.counts || []
  const option = {
    title: { text: '最大预测概率分布 (模型自信度)', left: 'center', textStyle: { fontSize: 13 } },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 50, right: 20, top: 50, bottom: 40 },
    xAxis: { type: 'category', data: centers.map(v => Number(v).toFixed(2)), name: 'max probability' },
    yAxis: { type: 'value', name: 'count' },
    series: [
      {
        name: 'confidence',
        type: 'bar',
        data: counts,
        itemStyle: { color: '#8b5cf6' },
      },
    ],
  }
  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      {payload.n_classes != null && (
        <Card variant="outlined" size="small">
          <Statistic title="类别数" value={payload.n_classes} valueStyle={{ fontSize: 18 }} />
          <Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 8 }}>
            多分类任务下展示每个样本的最大预测概率——分布偏右代表模型整体自信度高。
          </Text>
        </Card>
      )}
      <EChart option={option} style={{ height }} />
    </Space>
  )
}
