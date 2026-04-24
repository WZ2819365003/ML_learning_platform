/**
 * PerClassMetricsTable — per-class precision/recall/F1/support table.
 *
 * Accepts `payload` as returned by GET /api/viz/:id/per_class, e.g.
 *   { classes: [{ label: "0", precision, recall, f1, support }, ...],
 *     macro_avg: {...}, weighted_avg: {...}, accuracy: 0.92 }
 *
 * Two summary rows (macro avg / weighted avg) are pinned at the bottom
 * with a divider; the header shows an accuracy Statistic.
 */
import { Card, Empty, Space, Statistic, Table, Typography } from 'antd'

const { Text } = Typography

function fmt(v, digits = 4) {
  if (v == null || Number.isNaN(v)) return '-'
  return Number(v).toFixed(digits)
}

const COLUMNS = [
  {
    title: '类别',
    dataIndex: 'label',
    key: 'label',
    render: (v, row) => (row.isSummary
      ? <Text strong>{v}</Text>
      : <Text code>{v}</Text>),
  },
  {
    title: 'Precision',
    dataIndex: 'precision',
    key: 'precision',
    align: 'right',
    render: v => fmt(v),
  },
  {
    title: 'Recall',
    dataIndex: 'recall',
    key: 'recall',
    align: 'right',
    render: v => fmt(v),
  },
  {
    title: 'F1',
    dataIndex: 'f1',
    key: 'f1',
    align: 'right',
    render: v => fmt(v),
  },
  {
    title: 'Support',
    dataIndex: 'support',
    key: 'support',
    align: 'right',
    render: v => (v == null ? '-' : Number(v).toLocaleString()),
  },
]

export default function PerClassMetricsTable({ payload }) {
  // Backend returns `rows` (one per class) plus macro_avg / weighted_avg.
  const classArr = Array.isArray(payload?.rows)
    ? payload.rows
    : Array.isArray(payload?.classes)
      ? payload.classes
      : []
  if (classArr.length === 0) {
    return <Empty description="暂无逐类指标" />
  }

  const classRows = classArr.map((c, i) => ({
    key: `class-${i}`,
    ...c,
    isSummary: false,
  }))

  const summaryRows = []
  if (payload.macro_avg) {
    summaryRows.push({ key: 'macro_avg', label: 'Macro Avg', ...payload.macro_avg, isSummary: true })
  }
  if (payload.weighted_avg) {
    summaryRows.push({ key: 'weighted_avg', label: 'Weighted Avg', ...payload.weighted_avg, isSummary: true })
  }

  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      {payload.accuracy != null && (
        <Card variant="outlined" size="small">
          <Statistic title="Accuracy" value={payload.accuracy} precision={4} valueStyle={{ fontSize: 20 }} />
        </Card>
      )}
      <Table
        columns={COLUMNS}
        dataSource={[...classRows, ...summaryRows]}
        pagination={false}
        size="small"
        rowClassName={row => (row.isSummary ? 'per-class-summary-row' : '')}
      />
    </Space>
  )
}
