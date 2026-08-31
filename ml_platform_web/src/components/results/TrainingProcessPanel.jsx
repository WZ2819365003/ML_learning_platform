/**
 * TrainingProcessPanel — what happened *while* the model was fitting.
 *
 * The old 训练可视化 tab showed a confusion matrix, a predicted-vs-actual
 * scatter and a feature-importance bar. Those all describe the *finished*
 * model, not the process: they belong to 结果回测 and 模型解释, which is where
 * they now live. What is left here is the training run itself, which the
 * registry has always grouped under `resultsTabs: ['training']`:
 *
 *   ML — the learning curve, plus how stable the score was across CV folds
 *   DL — loss and metric per epoch, plus where early stopping landed
 *
 * Per-fold scores are not persisted (training_service strips `cv_folds` before
 * saving), so fold stability is shown from the mean/std aggregates that are.
 */
import React, { useEffect, useMemo, useState } from 'react'
import { Alert, Card, Col, Empty, Row, Space, Spin, Statistic, Tooltip, Typography } from 'antd'
import { InfoCircleOutlined } from '@ant-design/icons'

import EChart from '../EChart'
import {
  buildLossHistoryOption, buildTaskMetricHistoryOption,
} from '../viz/DLDiagnostics'
import { vizApi } from '../../services/api'

const { Text } = Typography

/** Pull cv_avg_x / cv_std_x pairs out of the metric bag. */
export function crossValidationStability(metrics = {}) {
  const rows = []
  for (const [key, value] of Object.entries(metrics)) {
    if (!key.startsWith('cv_avg_') || typeof value !== 'number') continue
    const name = key.slice('cv_avg_'.length)
    const std = metrics[`cv_std_${name}`]
    rows.push({
      metric: name.toUpperCase(),
      mean: value,
      std: typeof std === 'number' ? std : null,
      // Spread relative to the mean — the comparable number across metrics on
      // wildly different scales (rmse in thousands, r2 in [0,1]).
      cv: typeof std === 'number' && Math.abs(value) > 1e-9
        ? Math.abs(std / value) * 100
        : null,
    })
  }
  return rows
}

/** Where training actually stopped, versus how long it was allowed to run. */
export function earlyStoppingSummary(metrics = {}, trainConfig = {}) {
  const history = Array.isArray(metrics.history) ? metrics.history : []
  const ran = metrics.final_epoch ?? history.length
  const budget = trainConfig.epochs ?? history[0]?.total ?? null
  if (!ran) return null
  return {
    ran,
    budget,
    stoppedEarly: Boolean(budget) && ran < budget,
    bestValLoss: typeof metrics.best_val_loss === 'number' ? metrics.best_val_loss : null,
  }
}

function SectionCard({ title, hint, children }) {
  return (
    <Card size="small" variant="outlined"
      title={
        <Space size={6}>
          <span>{title}</span>
          {hint && (
            <Tooltip title={hint}>
              <InfoCircleOutlined style={{ color: '#94a3b8', fontSize: 12 }} />
            </Tooltip>
          )}
        </Space>
      }>
      {children}
    </Card>
  )
}

export default function TrainingProcessPanel({ family, taskId, taskType, metrics, trainConfig }) {
  const isDl = family === 'dl'
  const [learningCurve, setLearningCurve] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (isDl || !taskId) return undefined
    let cancelled = false
    setLoading(true)
    vizApi.getLearningCurve(taskId)
      .then(resp => { if (!cancelled) setLearningCurve(resp) })
      .catch(() => { if (!cancelled) setLearningCurve(null) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [isDl, taskId])

  const stability = useMemo(() => crossValidationStability(metrics), [metrics])
  const stopping = useMemo(
    () => earlyStoppingSummary(metrics, trainConfig), [metrics, trainConfig],
  )
  const history = Array.isArray(metrics?.history) ? metrics.history : []

  const stepOption = useMemo(() => {
    const steps = learningCurve?.steps || []
    if (steps.length === 0) return null
    const keys = Object.keys(steps[0] || {}).filter(k => k !== 'step' && typeof steps[0][k] === 'number')
    return {
      grid: { left: 56, right: 20, top: 30, bottom: 40 },
      tooltip: { trigger: 'axis' },
      legend: { top: 0 },
      xAxis: { type: 'category', data: steps.map(s => s.step), name: '步骤' },
      yAxis: { type: 'value', scale: true },
      series: keys.map(k => ({
        name: k, type: 'line', smooth: true, showSymbol: steps.length < 30,
        data: steps.map(s => s[k]),
      })),
    }
  }, [learningCurve])

  if (loading) return <Spin><div style={{ height: 220 }} /></Spin>

  const hasAnything = isDl ? history.length > 0 : (stepOption || stability.length > 0)
  if (!hasAnything) {
    return (
      <Empty image={Empty.PRESENTED_IMAGE_SIMPLE}
        description="该模型没有留下训练过程数据（训练步骤日志与交叉验证聚合值均为空）" />
    )
  }

  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      {isDl && stopping && (
        <Row gutter={[12, 12]}>
          <Col xs={12} md={8}>
            <Card size="small">
              <Statistic title="实际训练轮数"
                value={stopping.ran}
                suffix={stopping.budget ? `/ ${stopping.budget}` : undefined} />
            </Card>
          </Col>
          <Col xs={12} md={8}>
            <Card size="small">
              <Statistic title="最佳验证损失"
                value={stopping.bestValLoss ?? 0} precision={6} />
            </Card>
          </Col>
          <Col xs={24} md={8}>
            <Card size="small">
              <Statistic title="是否早停"
                value={stopping.stoppedEarly ? '是' : '否'}
                valueStyle={{ color: stopping.stoppedEarly ? '#d97706' : '#0f172a' }} />
            </Card>
          </Col>
        </Row>
      )}

      {isDl && history.length > 0 && (
        <>
          <SectionCard title="损失曲线（按 Epoch）"
            hint="训练损失持续下降而验证损失回升，是过拟合开始的信号。">
            <EChart option={buildLossHistoryOption(history)} style={{ height: 280 }} />
          </SectionCard>
          <SectionCard title="指标曲线（按 Epoch）"
            hint="验证指标趋于平坦说明继续训练收益有限。">
            <EChart option={buildTaskMetricHistoryOption(history, taskType)} style={{ height: 280 }} />
          </SectionCard>
        </>
      )}

      {!isDl && stepOption && (
        <SectionCard title="训练过程指标"
          hint="训练期间记录的分步指标，反映拟合是如何推进的。">
          <EChart option={stepOption} style={{ height: 300 }} />
        </SectionCard>
      )}

      {!isDl && stability.length > 0 && (
        <SectionCard title="交叉验证稳定性"
          hint="各折得分的均值与标准差。变异系数越大，说明模型对数据划分越敏感、结果越不稳。">
          <Row gutter={[12, 12]}>
            {stability.map(row => (
              <Col xs={24} sm={12} md={8} key={row.metric}>
                <Card size="small" variant="outlined">
                  <Statistic title={row.metric} value={row.mean} precision={4} />
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    ± {row.std != null ? row.std.toFixed(4) : '—'}
                    {row.cv != null && `　变异系数 ${row.cv.toFixed(1)}%`}
                  </Text>
                </Card>
              </Col>
            ))}
          </Row>
          {stability.some(r => r.cv != null && r.cv > 15) && (
            <Alert style={{ marginTop: 10 }} type="warning" showIcon
              message="部分指标跨折波动较大"
              description="变异系数超过 15% 通常意味着样本量偏小或数据分布不均，单次得分不宜过度解读。" />
          )}
        </SectionCard>
      )}
    </Space>
  )
}
