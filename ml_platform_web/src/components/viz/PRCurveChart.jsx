/**
 * PRCurveChart — Precision-Recall curve for binary (single curve + best
 * threshold markLine) and multiclass (one-vs-rest, colored per class).
 *
 * Accepts the payload shape from GET /api/viz/:id/pr_curve:
 *   binary:     { multiclass: false, precision: [...], recall: [...],
 *                 thresholds: [...], average_precision, best_threshold, best_f1,
 *                 positive_label }
 *   multiclass: { multiclass: true, curves: [{class, precision, recall,
 *                 average_precision}, ...] }
 */
import { Card, Empty, Space, Statistic, Tag } from 'antd'
import EChart from '../EChart'

const PALETTE = ['#2563eb', '#10b981', '#f97316', '#8b5cf6', '#ef4444', '#eab308', '#06b6d4']

export default function PRCurveChart({ payload, height = 380 }) {
  if (!payload) return <Empty description="暂无 PR 曲线数据" />

  if (payload.multiclass) {
    const curves = Array.isArray(payload.curves) ? payload.curves : []
    if (curves.length === 0) return <Empty description="多分类 PR 曲线为空" />

    const option = {
      title: { text: 'Precision-Recall 曲线 (One-vs-Rest)', left: 'center', textStyle: { fontSize: 13 } },
      tooltip: { trigger: 'axis' },
      legend: { top: 24, textStyle: { fontSize: 11 } },
      grid: { left: 50, right: 20, top: 60, bottom: 40 },
      xAxis: { type: 'value', name: 'Recall', min: 0, max: 1 },
      yAxis: { type: 'value', name: 'Precision', min: 0, max: 1 },
      series: curves.map((c, i) => ({
        name: `${c.class} (AP=${Number(c.average_precision ?? 0).toFixed(3)})`,
        type: 'line',
        smooth: true,
        showSymbol: false,
        data: (c.recall || []).map((r, idx) => [r, (c.precision || [])[idx]]),
        lineStyle: { color: PALETTE[i % PALETTE.length], width: 2 },
      })),
    }
    return <EChart option={option} style={{ height }} />
  }

  const precision = Array.isArray(payload.precision) ? payload.precision : []
  const recall = Array.isArray(payload.recall) ? payload.recall : []
  if (precision.length === 0 || recall.length === 0) {
    return <Empty description="二分类 PR 曲线为空" />
  }

  const option = {
    title: { text: 'Precision-Recall 曲线', left: 'center', textStyle: { fontSize: 13 } },
    tooltip: { trigger: 'axis', formatter: p => {
      const pt = p[0]
      return `Recall: ${pt.value[0]}<br/>Precision: ${pt.value[1]}`
    } },
    grid: { left: 50, right: 20, top: 50, bottom: 40 },
    xAxis: { type: 'value', name: 'Recall', min: 0, max: 1 },
    yAxis: { type: 'value', name: 'Precision', min: 0, max: 1 },
    series: [
      {
        name: 'PR',
        type: 'line',
        smooth: true,
        showSymbol: false,
        data: recall.map((r, i) => [r, precision[i]]),
        lineStyle: { color: '#2563eb', width: 2 },
        areaStyle: { color: 'rgba(37, 99, 235, 0.1)' },
      },
    ],
  }

  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      <Card variant="outlined" size="small">
        <Space wrap size={[24, 12]}>
          <Statistic title="Average Precision" value={payload.average_precision ?? 0} precision={4} valueStyle={{ fontSize: 18 }} />
          <Statistic title="Best Threshold" value={payload.best_threshold ?? 0} precision={4} valueStyle={{ fontSize: 18 }} />
          <Statistic title="Best F1" value={payload.best_f1 ?? 0} precision={4} valueStyle={{ fontSize: 18 }} />
          {payload.positive_label != null && (
            <div>
              <div style={{ fontSize: 12, color: '#8c8c8c', marginBottom: 4 }}>正类</div>
              <Tag color="blue">{String(payload.positive_label)}</Tag>
            </div>
          )}
        </Space>
      </Card>
      <EChart option={option} style={{ height }} />
    </Space>
  )
}
