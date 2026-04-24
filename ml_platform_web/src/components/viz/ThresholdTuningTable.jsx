/**
 * ThresholdTuningTable — sweep classification thresholds and show per-
 * threshold precision / recall / F1 / accuracy.  The row matching
 * payload.best_threshold (best F1) is highlighted.
 *
 * Accepts payload from GET /api/viz/:id/threshold:
 *   { rows: [{threshold, precision, recall, f1, accuracy}, ...],
 *     best_threshold, positive_label }
 *
 * Only applicable to binary classifiers with predict_proba.
 */
import { Card, Empty, Space, Statistic, Table, Tag, Typography } from 'antd'

const { Text } = Typography

function fmt(v) {
  if (v == null || Number.isNaN(v)) return '-'
  return Number(v).toFixed(4)
}

export default function ThresholdTuningTable({ payload }) {
  if (!payload || !Array.isArray(payload.rows) || payload.rows.length === 0) {
    return <Empty description="暂无阈值分析数据" />
  }

  const bestThreshold = payload.best_threshold
  const dataSource = payload.rows.map((r, i) => ({
    key: `thr-${i}`,
    ...r,
    isBest: Math.abs((r.threshold ?? -1) - (bestThreshold ?? -2)) < 1e-6,
  }))

  const columns = [
    {
      title: '阈值',
      dataIndex: 'threshold',
      key: 'threshold',
      render: (v, row) => row.isBest
        ? <Tag color="gold" style={{ fontWeight: 500 }}>{fmt(v)}</Tag>
        : fmt(v),
      width: 100,
    },
    { title: 'Precision', dataIndex: 'precision', key: 'precision', align: 'right', render: fmt },
    { title: 'Recall', dataIndex: 'recall', key: 'recall', align: 'right', render: fmt },
    {
      title: 'F1',
      dataIndex: 'f1',
      key: 'f1',
      align: 'right',
      render: (v, row) => row.isBest
        ? <Text strong style={{ color: '#d97706' }}>{fmt(v)}</Text>
        : fmt(v),
    },
    { title: 'Accuracy', dataIndex: 'accuracy', key: 'accuracy', align: 'right', render: fmt },
  ]

  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      <Card variant="outlined" size="small">
        <Space wrap size={[24, 12]}>
          <Statistic title="Best Threshold (F1)" value={bestThreshold ?? 0} precision={4} valueStyle={{ fontSize: 18, color: '#d97706' }} />
          {payload.positive_label != null && (
            <div>
              <div style={{ fontSize: 12, color: '#8c8c8c', marginBottom: 4 }}>正类</div>
              <Tag color="blue">{String(payload.positive_label)}</Tag>
            </div>
          )}
        </Space>
        <Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 8 }}>
          不同业务场景优先不同指标：重复营销看 Recall，风险审批看 Precision。以 F1 为综合参考。
        </Text>
      </Card>
      <Table
        columns={columns}
        dataSource={dataSource}
        pagination={false}
        size="small"
        rowClassName={row => (row.isBest ? 'threshold-best-row' : '')}
      />
    </Space>
  )
}
