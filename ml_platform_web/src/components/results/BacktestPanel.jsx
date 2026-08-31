/**
 * BacktestPanel — how the model's predictions line up against the truth.
 *
 * Two readings of the same pair of arrays:
 *   curve   — actual and predicted over the sample index: where it goes wrong,
 *             which peaks get flattened, whether the prediction lags
 *   scatter — predicted against actual with a y=x diagonal: how tight the fit
 *             is overall, and whether the model systematically over/under-shoots
 *
 * Both families read /viz/{id}/predicted_vs_actual, which replays the task's
 * hold-out and predicts it in row order. DL used to fall back to the
 * `val_scatter` saved during training, but runs from before that field became
 * contiguous hold a *random* subsample — fine as a scatter, meaningless as a
 * curve, since joining randomly ordered points draws a line that looks like
 * data and is not. Replaying server-side gives ordered rows for every model,
 * including ones trained before that changed, with nobody having to retrain.
 */
import React, { useEffect, useMemo, useState } from 'react'
import { Alert, Card, Col, Empty, Row, Space, Spin, Statistic, Tooltip, Typography } from 'antd'
import { InfoCircleOutlined, LineChartOutlined, DotChartOutlined } from '@ant-design/icons'

import EChart from '../EChart'
import PredictedActualCurve from '../viz/PredictedActualCurve'
import { buildConfusionMatrixOption, buildRocCurveOption } from '../workbench/TrainingViz'
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

/** Residual against predicted — derived, so it costs no extra request. */
export function buildResidualOption(actual = [], predicted = []) {
  const n = Math.min(actual.length, predicted.length)
  const points = Array.from({ length: n }, (_, i) => [predicted[i], actual[i] - predicted[i]])
  return {
    grid: { left: 62, right: 20, top: 20, bottom: 44 },
    tooltip: { formatter: ({ value }) => `预测=${value[0]}<br/>残差=${value[1]}` },
    xAxis: { type: 'value', name: '预测值', scale: true, nameLocation: 'middle', nameGap: 26 },
    yAxis: { type: 'value', name: '残差（实际−预测）', scale: true },
    series: [
      { type: 'scatter', symbolSize: 5, data: points, itemStyle: { color: '#8b5cf6', opacity: 0.5 } },
      // A flat line at zero: residuals fanning out or curving away from it is
      // the pattern worth catching.
      { type: 'line', symbol: 'none', markLine: {
          silent: true, symbol: 'none',
          lineStyle: { type: 'dashed', color: '#94a3b8' },
          data: [{ yAxis: 0 }],
        }, data: [] },
    ],
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

/**
 * The classification half of 结果回测.
 *
 * A confusion matrix *is* predictions lined up against truth — the same
 * question the regression curve answers, asked of labels instead of numbers.
 * These lived under 训练可视化 until the tabs were split by what they answer,
 * and would otherwise have had nowhere to go.
 */
function ClassificationBacktest({ taskId }) {
  const [cm, setCm] = useState(null)
  const [roc, setRoc] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!taskId) return undefined
    let cancelled = false
    setLoading(true)
    // allSettled: a model without predict_proba has no ROC, which must not
    // take the confusion matrix down with it.
    Promise.allSettled([
      vizApi.getConfusionMatrix(taskId),
      vizApi.getRocCurve(taskId),
    ]).then(([c, r]) => {
      if (cancelled) return
      setCm(c.status === 'fulfilled' ? c.value : null)
      setRoc(r.status === 'fulfilled' ? r.value : null)
    }).finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [taskId])

  const cmOption = useMemo(() => buildConfusionMatrixOption(cm), [cm])
  const rocOption = useMemo(() => buildRocCurveOption(roc), [roc])

  if (loading) return <Spin><div style={{ height: 220 }} /></Spin>
  if (!cmOption && !rocOption) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无回测数据" />
  }

  return (
    <Row gutter={[12, 12]}>
      <Col xs={24} xl={12}>
        <Card size="small" variant="outlined"
          title={
            <Space size={6}>
              <span>混淆矩阵</span>
              <Tooltip title="对角线是预测正确的样本。非对角线上的大数字指出模型把哪一类错认成了哪一类。">
                <InfoCircleOutlined style={{ color: '#94a3b8', fontSize: 12 }} />
              </Tooltip>
            </Space>
          }>
          {cmOption
            ? <EChart option={cmOption} style={{ height: 320 }} />
            : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无混淆矩阵" />}
        </Card>
      </Col>
      <Col xs={24} xl={12}>
        <Card size="small" variant="outlined"
          title={
            <Space size={6}>
              <span>ROC 曲线</span>
              <Tooltip title="越贴近左上角越好。模型不支持 predict_proba 时无法绘制。">
                <InfoCircleOutlined style={{ color: '#94a3b8', fontSize: 12 }} />
              </Tooltip>
            </Space>
          }>
          {rocOption
            ? <EChart option={rocOption} style={{ height: 320 }} />
            : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE}
                description="该模型不支持 predict_proba，无法绘制 ROC" />}
        </Card>
      </Col>
    </Row>
  )
}

export default function BacktestPanel({ family, taskId, taskType }) {
  const [payload, setPayload] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const isDl = family === 'dl'

  useEffect(() => {
    if (!taskId) return undefined
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
  }, [taskId])

  const actual = Array.isArray(payload?.actual) ? payload.actual : []
  const predicted = Array.isArray(payload?.predicted) ? payload.predicted : []

  const stats = useMemo(() => backtestStats(actual, predicted), [actual, predicted])
  const scatterOption = useMemo(() => buildScatterOption(actual, predicted), [actual, predicted])
  const residualOption = useMemo(() => buildResidualOption(actual, predicted), [actual, predicted])

  if (taskType === 'classification') {
    return <ClassificationBacktest taskId={taskId} />
  }

  if (loading) return <Spin><div style={{ height: 220 }} /></Spin>
  if (error) return <Alert type="error" showIcon message="回测数据加载失败" description={error} />
  if (actual.length === 0) {
    return (
      <Empty image={Empty.PRESENTED_IMAGE_SIMPLE}
        description="暂无回测数据" />
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

      <Card size="small" variant="outlined"
        title={
          <Space size={6}>
            <DotChartOutlined style={{ color: '#8b5cf6' }} />
            <span>残差分布</span>
            <Tooltip title="残差应随机散布在 0 线两侧。呈喇叭状或弯曲，说明模型漏掉了某种结构。">
              <InfoCircleOutlined style={{ color: '#94a3b8', fontSize: 12 }} />
            </Tooltip>
          </Space>
        }>
        <EChart option={residualOption} style={{ height: 280 }} />
      </Card>

      <Text type="secondary" style={{ fontSize: 12 }}>
        {payload?.total_count
          ? `数据来自该模型留出集的最近 ${actual.length} 条（共 ${payload.total_count} 条）。`
          : '数据来自该模型留出集的预测结果。'}
      </Text>
    </Space>
  )
}
