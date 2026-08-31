/**
 * BacktestPanel — how the model's predictions line up against the truth.
 *
 * Two readings of the same pair of arrays:
 *   curve   — actual and predicted over the sample index: where it goes wrong,
 *             which peaks get flattened, whether the prediction lags
 *   scatter — predicted against actual with a y=x diagonal: how tight the fit
 *             is overall, and whether the model systematically over/under-shoots
 *
 * Where the numbers come from differs by family, and so does one caveat:
 *   ML — /viz/{id}/predicted_vs_actual, the full ordered hold-out
 *   DL — metrics.val_scatter, a trailing window saved during training
 *
 * DL runs trained before that field became ordered hold a *random* subsample.
 * Those points are fine as a scatter but meaningless as a curve — joining
 * randomly ordered points draws a line that looks like data and is not — so
 * the curve is withheld for them rather than drawn misleadingly.
 */
import React, { useEffect, useMemo, useState } from 'react'
import { Alert, Card, Col, Empty, Row, Space, Spin, Statistic, Tooltip, Typography } from 'antd'
import { InfoCircleOutlined, LineChartOutlined, DotChartOutlined } from '@ant-design/icons'

import EChart from '../EChart'
import PredictedActualCurve from '../viz/PredictedActualCurve'
import { vizApi } from '../../services/api'

const { Text } = Typography

/** Error summary over the sampled pairs — the numbers behind the picture. */
export function backtestStats(actual = [], predicted = []) {
  const n = Math.min(actual.length, predicted.length)
  if (n === 0) return null
  let se = 0, ae = 0, ape = 0, apeCount = 0
  for (let i = 0; i < n; i += 1) {
    const err = actual[i] - predicted[i]
    se += err * err
    ae += Math.abs(err)
    // MAPE is undefined at zero, so those rows are skipped rather than
    // inflating the average to Infinity.
    if (Math.abs(actual[i]) > 1e-8) { ape += Math.abs(err / actual[i]); apeCount += 1 }
  }
  return {
    count: n,
    rmse: Math.sqrt(se / n),
    mae: ae / n,
    mape: apeCount > 0 ? (ape / apeCount) * 100 : null,
  }
}

export function buildScatterOption(actual = [], predicted = []) {
  const n = Math.min(actual.length, predicted.length)
  const points = Array.from({ length: n }, (_, i) => [actual[i], predicted[i]])
  const all = [...actual.slice(0, n), ...predicted.slice(0, n)]
  const min = Math.min(...all)
  const max = Math.max(...all)
  return {
    grid: { left: 60, right: 20, top: 20, bottom: 44 },
    tooltip: { formatter: ({ value }) => `实际=${value[0]}<br/>预测=${value[1]}` },
    xAxis: { type: 'value', name: '实际值', scale: true, nameLocation: 'middle', nameGap: 26 },
    yAxis: { type: 'value', name: '预测值', scale: true },
    series: [
      { type: 'scatter', symbolSize: 5, data: points, itemStyle: { color: '#2563eb', opacity: 0.55 } },
      {
        name: '理想预测', type: 'line', symbol: 'none', data: [[min, min], [max, max]],
        lineStyle: { type: 'dashed', color: '#94a3b8', width: 1.4 },
      },
    ],
  }
}

export default function BacktestPanel({ family, taskId, taskType, metrics }) {
  const [payload, setPayload] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const isDl = family === 'dl'

  useEffect(() => {
    if (isDl || !taskId) return undefined
    let cancelled = false
    setLoading(true)
    setError(null)
    vizApi.getPredictedVsActual(taskId)
      .then(resp => { if (!cancelled) setPayload(resp) })
      .catch(err => {
        if (!cancelled) setError(err?.response?.data?.detail || '回测数据加载失败')
      })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [isDl, taskId])

  const source = isDl ? (metrics?.val_scatter || null) : payload
  const actual = Array.isArray(source?.actual) ? source.actual : []
  const predicted = Array.isArray(source?.predicted) ? source.predicted : []

  // Older DL runs sampled randomly and carry no ordering flag.
  const curveTrustworthy = !isDl || source?.ordered === true

  const stats = useMemo(() => backtestStats(actual, predicted), [actual, predicted])
  const scatterOption = useMemo(() => buildScatterOption(actual, predicted), [actual, predicted])

  if (taskType === 'classification') {
    return (
      <Empty image={Empty.PRESENTED_IMAGE_SIMPLE}
        description="结果回测面向回归任务；分类任务请看训练可视化里的混淆矩阵与 ROC" />
    )
  }

  if (loading) return <Spin><div style={{ height: 220 }} /></Spin>
  if (error) return <Alert type="error" showIcon message="回测数据加载失败" description={error} />
  if (actual.length === 0) {
    return (
      <Empty image={Empty.PRESENTED_IMAGE_SIMPLE}
        description={isDl
          ? '该模型训练时没有保存验证集预测样本，重新训练后即可回测'
          : '暂无回测数据'} />
    )
  }

  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      {stats && (
        <Row gutter={[12, 12]}>
          <Col xs={12} md={6}><Card size="small"><Statistic title="样本数" value={stats.count} /></Card></Col>
          <Col xs={12} md={6}><Card size="small"><Statistic title="RMSE" value={stats.rmse} precision={4} /></Card></Col>
          <Col xs={12} md={6}><Card size="small"><Statistic title="MAE" value={stats.mae} precision={4} /></Card></Col>
          <Col xs={12} md={6}>
            <Card size="small">
              <Statistic title="MAPE" value={stats.mape ?? 0} precision={2} suffix="%" />
            </Card>
          </Col>
        </Row>
      )}

      {!curveTrustworthy && (
        <Alert
          type="warning" showIcon
          message="该模型的样本是随机抽取的，无法按顺序绘制曲线"
          description="它训练于样本改为按顺序保存之前。散点图仍然可信；重新训练后曲线即可用。"
        />
      )}

      {curveTrustworthy && (
        <Card size="small" variant="outlined"
          title={
            <Space size={6}>
              <LineChartOutlined style={{ color: '#2563eb' }} />
              <span>预测值 vs 实际值（曲线）</span>
              <Tooltip title="两条线贴合越紧越好。看峰值是否被削平、预测是否整体滞后。">
                <InfoCircleOutlined style={{ color: '#94a3b8', fontSize: 12 }} />
              </Tooltip>
            </Space>
          }>
          <PredictedActualCurve payload={{ actual, predicted }} height={300} />
        </Card>
      )}

      <Card size="small" variant="outlined"
        title={
          <Space size={6}>
            <DotChartOutlined style={{ color: '#f59e0b' }} />
            <span>预测值 vs 实际值（散点）</span>
            <Tooltip title="点越贴近虚线越好。整体偏在一侧说明模型系统性高估或低估。">
              <InfoCircleOutlined style={{ color: '#94a3b8', fontSize: 12 }} />
            </Tooltip>
          </Space>
        }>
        <EChart option={scatterOption} style={{ height: 300 }} />
      </Card>

      <Text type="secondary" style={{ fontSize: 12 }}>
        {isDl
          ? '数据来自训练时保存的验证集尾部样本。'
          : '数据来自该模型的测试集预测结果。'}
      </Text>
    </Space>
  )
}
