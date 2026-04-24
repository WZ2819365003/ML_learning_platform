/**
 * ShapView — honest, method-aware SHAP renderer.
 *
 * Renders one of three bodies based on `payload.method`:
 *   - "tree" / "kernel": full summary (mean |SHAP| bar chart + base_value +
 *     sample_count + top-3 feature narrative with direction tags)
 *   - "permutation": narrative explains SHAP was unavailable; shows only
 *     the permutation-importance bar chart (no per-sample data exists).
 *
 * This component replaces the old ad-hoc SHAP blocks in Results.jsx and
 * RunInspector.jsx.  Both pages now have consistent labeling of the SHAP
 * rung — the user can trust whether they're looking at real SHAP or a
 * fallback.
 */
import { Alert, Card, Empty, Space, Statistic, Tag, Typography, Spin } from 'antd'
import EChart from '../EChart'

const { Text, Paragraph } = Typography

const METHOD_META = {
  tree: {
    label: 'TreeExplainer',
    color: '#10b981',
    description: '基于树模型结构的精确 SHAP 值，每个样本的特征贡献都已计算。',
  },
  kernel: {
    label: 'KernelExplainer',
    color: '#2563eb',
    description: '基于采样近似的 SHAP 值，可能略慢于 Tree 但同样提供逐样本解释。',
  },
  permutation: {
    label: 'Permutation Importance',
    color: '#f59e0b',
    description:
      'SHAP 不可用（numpy 兼容或模型不支持），降级为置换重要度：只衡量整体影响，不提供单样本方向。',
  },
}

export default function ShapView({ payload, loading, onRetry }) {
  if (loading) {
    return (
      <Card variant="outlined">
        <div style={{ textAlign: 'center', padding: 60 }}>
          <Spin tip="正在计算 SHAP 解释..." />
        </div>
      </Card>
    )
  }

  if (!payload || payload.status === 'error') {
    return (
      <Card variant="outlined">
        <Empty description={payload?.detail || '暂无 SHAP 数据'}>
          {onRetry && <a onClick={onRetry}>重新计算</a>}
        </Empty>
      </Card>
    )
  }

  const meta = METHOD_META[payload.method] || METHOD_META.permutation
  const top = Array.isArray(payload.top_features) ? payload.top_features.slice(0, 10) : []
  const hasPerSample = payload.shap_values != null && payload.feature_values != null
  const topNarrative = top.slice(0, 3)

  // Bar chart: mean |SHAP| per feature
  const barOption = top.length
    ? {
        title: {
          text: hasPerSample ? 'Top 特征的平均 |SHAP| 值' : 'Top 特征的置换重要度',
          left: 'center',
          textStyle: { fontSize: 13, fontWeight: 500 },
        },
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
        grid: { left: 90, right: 20, top: 40, bottom: 30 },
        xAxis: { type: 'value', name: hasPerSample ? 'mean |SHAP|' : 'importance' },
        yAxis: {
          type: 'category',
          data: [...top].reverse().map(t => t.feature),
          axisLabel: { fontSize: 11 },
        },
        series: [
          {
            type: 'bar',
            data: [...top].reverse().map(t => Number((t.importance ?? 0).toFixed(4))),
            itemStyle: { color: meta.color },
            label: { show: true, position: 'right', fontSize: 10, formatter: ({ value }) => value.toFixed(3) },
          },
        ],
      }
    : null

  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      {/* Header: method + base_value + sample_count */}
      <Card variant="outlined" size="small">
        <Space wrap size={[24, 12]}>
          <div>
            <Tag color={meta.color} style={{ fontSize: 13, padding: '4px 10px' }}>
              {meta.label}
            </Tag>
            <Text type="secondary" style={{ fontSize: 12, marginLeft: 8 }}>
              {meta.description}
            </Text>
          </div>
          {payload.base_value != null && (
            <Statistic
              title="基线值 (base_value)"
              value={payload.base_value}
              precision={4}
              valueStyle={{ fontSize: 18 }}
            />
          )}
          {payload.sample_count != null && (
            <Statistic
              title="解释样本数"
              value={payload.sample_count}
              valueStyle={{ fontSize: 18 }}
              suffix={payload.feature_count ? `· ${payload.feature_count} 个特征` : null}
            />
          )}
        </Space>
      </Card>

      {payload.method === 'permutation' && (
        <Alert
          type="warning"
          showIcon
          message="真 SHAP 不可用，当前展示的是置换重要度 (Permutation Importance)"
          description="这意味着可以看到哪些特征对模型总体影响大，但无法看每一条样本的方向（推高 / 拉低）。这通常由 numpy 版本与 SHAP 库不兼容触发，平台已自动降级而非伪造 SHAP 值。"
        />
      )}

      {/* Top features bar chart */}
      {barOption ? (
        <Card variant="outlined" size="small">
          <EChart option={barOption} style={{ height: 320 }} />
        </Card>
      ) : (
        <Empty description="无可展示的特征重要度" />
      )}

      {/* Narrative for top-3 features */}
      {topNarrative.length > 0 && (
        <Card size="small" title="Top 3 特征解读" variant="outlined">
          <Space direction="vertical" size={8} style={{ width: '100%' }}>
            {topNarrative.map((f, i) => (
              <Paragraph key={f.feature} style={{ margin: 0, fontSize: 13 }}>
                <Tag color={meta.color}>#{i + 1}</Tag>
                <Text strong>{f.feature}</Text>
                <Text type="secondary" style={{ marginLeft: 6 }}>
                  重要度 {(f.importance ?? 0).toFixed(4)}
                </Text>
                {f.direction === 'up' && (
                  <Tag color="red" style={{ marginLeft: 6 }}>推高预测</Tag>
                )}
                {f.direction === 'down' && (
                  <Tag color="blue" style={{ marginLeft: 6 }}>拉低预测</Tag>
                )}
                {f.direction == null && hasPerSample === false && (
                  <Tag style={{ marginLeft: 6 }}>无方向信息（permutation）</Tag>
                )}
              </Paragraph>
            ))}
          </Space>
        </Card>
      )}
    </Space>
  )
}
