/**
 * TrainingProcessPanel — what happened *while* the model was fitting.
 *
 * Deliberately holds nothing that describes the finished model. A confusion
 * matrix, a predicted-vs-actual scatter and a feature-importance bar all
 * answer "how good is it" rather than "how did it get there", so they live in
 * 结果回测 and 模型解释. What is here is the run itself.
 *
 * Layout is a fixed 2×2 of equal-height cards. A panel that dropped empty
 * charts left a ragged shape and made the tab look broken rather than sparse,
 * so a chart with no data renders a placeholder of the same size and says what
 * is missing.
 */
import React, { useEffect, useMemo, useState } from 'react'
import { Alert, Card, Col, Descriptions, Empty, Row, Space, Spin, Statistic, Tooltip, Typography } from 'antd'
import { InfoCircleOutlined } from '@ant-design/icons'

import EChart from '../EChart'
import { vizApi } from '../../services/api'

const { Text } = Typography

const CHART_HEIGHT = 280

/** Keep large loss values readable without forcing a very wide Y-axis gutter. */
export function compactChartNumber(value) {
  const number = Number(value)
  if (!Number.isFinite(number)) return String(value ?? '')
  const absolute = Math.abs(number)
  const formatUnit = (divisor, suffix) => {
    const scaled = number / divisor
    const precision = Math.abs(scaled) >= 100 ? 0 : Math.abs(scaled) >= 10 ? 1 : 2
    return `${scaled.toFixed(precision).replace(/\.0+$|(?<=\.[0-9])0+$/, '')}${suffix}`
  }
  if (absolute >= 100_000_000) return formatUnit(100_000_000, '亿')
  if (absolute >= 10_000) return formatUnit(10_000, '万')
  return new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 2 }).format(number)
}

const containedGrid = (top = 32, bottom = 40) => ({
  left: 16,
  right: 20,
  top,
  bottom,
  containLabel: true,
})

// ---------------------------------------------------------------------------
// Pure derivations — exported for tests
// ---------------------------------------------------------------------------

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
      // Spread relative to the mean — comparable across metrics on wildly
      // different scales (rmse in thousands, r2 in [0,1]).
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

/** Train and validation loss on one axis — where they part is where it overfits. */
export function buildLossOption(history = []) {
  if (history.length === 0) return null
  const epochs = history.map(r => r.epoch ?? r.step)
  const train = history.map(r => r.train_loss ?? null)
  const val = history.map(r => r.val_loss ?? null)
  if (train.every(v => v == null) && val.every(v => v == null)) return null
  return {
    grid: containedGrid(),
    tooltip: { trigger: 'axis' },
    legend: { top: 0, data: ['训练损失', '验证损失'] },
    xAxis: { type: 'category', data: epochs, name: 'Epoch', nameLocation: 'middle', nameGap: 24 },
    yAxis: { type: 'value', scale: true, axisLabel: { formatter: compactChartNumber } },
    series: [
      { name: '训练损失', type: 'line', data: train, showSymbol: false, lineStyle: { color: '#2563eb', width: 1.8 }, itemStyle: { color: '#2563eb' } },
      { name: '验证损失', type: 'line', data: val, showSymbol: false, lineStyle: { color: '#dc2626', width: 1.8 }, itemStyle: { color: '#dc2626' } },
    ],
  }
}

/**
 * The gap between validation and training loss, epoch by epoch.
 *
 * The pair of curves above shows this implicitly; plotting the difference makes
 * the turn obvious — it is flat while the model is still learning and climbs
 * once it starts memorising.
 */
export function buildOverfitGapOption(history = []) {
  const rows = history.filter(r => typeof r.train_loss === 'number' && typeof r.val_loss === 'number')
  if (rows.length === 0) return null
  const gaps = rows.map(r => r.val_loss - r.train_loss)
  const worst = gaps.indexOf(Math.max(...gaps))
  return {
    grid: containedGrid(26),
    tooltip: { trigger: 'axis', valueFormatter: v => (typeof v === 'number' ? v.toFixed(6) : v) },
    xAxis: { type: 'category', data: rows.map(r => r.epoch), name: 'Epoch', nameLocation: 'middle', nameGap: 24 },
    yAxis: {
      type: 'value', name: '验证 − 训练', scale: true,
      axisLabel: { formatter: compactChartNumber },
    },
    series: [{
      type: 'line', data: gaps, showSymbol: false, smooth: true,
      areaStyle: { opacity: 0.12 }, lineStyle: { color: '#d97706', width: 1.8 },
      itemStyle: { color: '#d97706' },
      markPoint: {
        symbolSize: 40,
        data: [{ name: '最大差距', coord: [worst, gaps[worst]], value: compactChartNumber(gaps[worst]) }],
      },
    }],
  }
}

/** Learning rate per epoch — already recorded by the trainer, never plotted. */
export function buildLearningRateOption(history = []) {
  const lrs = history.map(r => r.lr).filter(v => typeof v === 'number')
  if (lrs.length === 0) return null
  return {
    grid: containedGrid(26),
    tooltip: { trigger: 'axis', valueFormatter: v => (typeof v === 'number' ? v.toExponential(2) : v) },
    xAxis: { type: 'category', data: history.map(r => r.epoch), name: 'Epoch', nameLocation: 'middle', nameGap: 24 },
    // Log scale: a scheduler halves the rate, so successive steps are invisible
    // on a linear axis once the value gets small.
    yAxis: { type: 'log', name: '学习率', axisLabel: { formatter: v => Number(v).toExponential(0) } },
    series: [{
      type: 'line', step: 'end', showSymbol: false,
      data: history.map(r => r.lr ?? null),
      lineStyle: { color: '#8b5cf6', width: 1.8 }, itemStyle: { color: '#8b5cf6' },
    }],
  }
}

/** Validation metrics per epoch, whatever the trainer recorded. */
export function buildMetricHistoryOption(history = []) {
  if (history.length === 0) return null
  const skip = new Set(['epoch', 'total', 'step', 'lr', 'train_loss', 'val_loss'])
  const keys = Object.keys(history[0] || {}).filter(k => !skip.has(k) && typeof history[0][k] === 'number')
  if (keys.length === 0) return null
  return {
    grid: containedGrid(),
    tooltip: { trigger: 'axis' },
    legend: { top: 0 },
    xAxis: { type: 'category', data: history.map(r => r.epoch), name: 'Epoch', nameLocation: 'middle', nameGap: 24 },
    yAxis: { type: 'value', scale: true, axisLabel: { formatter: compactChartNumber } },
    series: keys.map(k => ({ name: k, type: 'line', data: history.map(r => r[k] ?? null), showSymbol: false })),
  }
}

/** Per-fold scores, when the trainer persisted them. */
export function buildFoldScoresOption(folds = [], metricKey = null) {
  if (!Array.isArray(folds) || folds.length === 0) return null
  const key = metricKey || Object.keys(folds[0] || {}).find(k => k !== 'fold' && typeof folds[0][k] === 'number')
  if (!key) return null
  const values = folds.map(f => f[key]).filter(v => typeof v === 'number')
  if (values.length === 0) return null
  const mean = values.reduce((a, b) => a + b, 0) / values.length
  return {
    grid: containedGrid(26),
    tooltip: { trigger: 'axis', valueFormatter: v => (typeof v === 'number' ? v.toFixed(4) : v) },
    xAxis: { type: 'category', data: folds.map(f => `第 ${f.fold} 折`) },
    yAxis: { type: 'value', scale: true, name: key, axisLabel: { formatter: compactChartNumber } },
    series: [{
      type: 'bar', data: values, itemStyle: { color: '#2563eb', borderRadius: [4, 4, 0, 0] },
      // The mean line is what turns five bars into a statement about spread.
      markLine: {
        silent: true, symbol: 'none',
        label: { formatter: `均值 ${mean.toFixed(4)}` },
        lineStyle: { type: 'dashed', color: '#dc2626' },
        data: [{ yAxis: mean }],
      },
    }],
  }
}

/** Chart slot that keeps its footprint when there is nothing to draw. */
function ChartCard({ title, hint, option, emptyText }) {
  return (
    <Col xs={24} xl={12}>
      <Card size="small" variant="outlined"
        styles={{ body: { height: CHART_HEIGHT + 24, padding: 12 } }}
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
        {option
          ? <EChart option={option} style={{ height: CHART_HEIGHT }} />
          : (
            <div style={{ height: CHART_HEIGHT, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Empty image={Empty.PRESENTED_IMAGE_SIMPLE}
                description={<Text type="secondary" style={{ fontSize: 12 }}>{emptyText}</Text>} />
            </div>
          )}
      </Card>
    </Col>
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

  const history = useMemo(
    () => (Array.isArray(metrics?.history) ? metrics.history : []), [metrics],
  )
  const folds = useMemo(
    () => (Array.isArray(metrics?.cv_folds) ? metrics.cv_folds : []), [metrics],
  )
  const stability = useMemo(() => crossValidationStability(metrics), [metrics])
  const stopping = useMemo(() => earlyStoppingSummary(metrics, trainConfig), [metrics, trainConfig])

  const stepOption = useMemo(() => {
    const steps = learningCurve?.steps || []
    if (steps.length === 0) return null
    const keys = Object.keys(steps[0] || {}).filter(k => k !== 'step' && typeof steps[0][k] === 'number')
    if (keys.length === 0) return null
    return {
      grid: containedGrid(),
      tooltip: { trigger: 'axis' },
      legend: { top: 0 },
      xAxis: { type: 'category', data: steps.map(s => s.step), name: '步骤', nameLocation: 'middle', nameGap: 24 },
      yAxis: { type: 'value', scale: true, axisLabel: { formatter: compactChartNumber } },
      series: keys.map(k => ({
        name: k, type: 'line', smooth: true, showSymbol: steps.length < 30,
        data: steps.map(s => s[k]),
      })),
    }
  }, [learningCurve])

  const stabilityOption = useMemo(() => {
    if (stability.length === 0) return null
    return {
      grid: containedGrid(26, 46),
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: stability.map(r => r.metric) },
      yAxis: { type: 'value', name: '变异系数 %', max: v => Math.max(20, v.max) },
      series: [{
        type: 'bar', data: stability.map(r => (r.cv == null ? 0 : Number(r.cv.toFixed(2)))),
        itemStyle: { color: ({ value }) => (value > 15 ? '#d97706' : '#10b981'), borderRadius: [4, 4, 0, 0] },
        label: { show: true, position: 'top', formatter: ({ value }) => `${value}%` },
        markLine: {
          silent: true, symbol: 'none',
          label: { formatter: '15%' },
          lineStyle: { type: 'dashed', color: '#94a3b8' },
          data: [{ yAxis: 15 }],
        },
      }],
    }
  }, [stability])

  if (loading) return <Spin><div style={{ height: 260 }} /></Spin>

  const nothingAtAll = isDl
    ? history.length === 0
    : (!stepOption && stability.length === 0 && folds.length === 0)
  if (nothingAtAll) {
    return (
      <Empty image={Empty.PRESENTED_IMAGE_SIMPLE}
        description="该模型没有留下训练过程数据（训练步骤日志与交叉验证聚合值均为空）" />
    )
  }

  const overview = isDl
    ? [
      { title: '实际训练轮数', value: stopping?.ran ?? '—', suffix: stopping?.budget ? `/ ${stopping.budget}` : undefined },
      { title: '是否早停', value: stopping?.stoppedEarly ? '是' : '否' },
      { title: '最佳验证损失', value: stopping?.bestValLoss ?? '—', precision: 6 },
      { title: '批大小', value: trainConfig?.batch_size ?? '—' },
    ]
    : [
      { title: '交叉验证折数', value: folds.length || trainConfig?.cv_folds || metrics?.cv_folds_count || '—' },
      { title: '验证集比例', value: trainConfig?.test_size ?? '—' },
      { title: '记录指标数', value: stability.length || '—' },
      {
        title: '最大变异系数',
        value: stability.length ? Math.max(...stability.map(r => r.cv ?? 0)) : '—',
        precision: 1, suffix: stability.length ? '%' : undefined,
      },
    ]

  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      <Row gutter={[12, 12]}>
        {overview.map(item => (
          <Col xs={12} md={6} key={item.title}>
            <Card size="small" styles={{ body: { padding: '12px 14px' } }}>
              <Statistic title={item.title} value={item.value}
                precision={item.precision} suffix={item.suffix} />
            </Card>
          </Col>
        ))}
      </Row>

      {/* Fixed 2×2 — empty slots keep their box so the panel stays rectangular. */}
      <Row gutter={[12, 12]}>
        {isDl ? (
          <>
            <ChartCard title="损失曲线" option={buildLossOption(history)}
              hint="训练损失持续下降而验证损失回升，就是过拟合的起点。"
              emptyText="该模型没有逐轮损失记录" />
            <ChartCard title="过拟合观察（验证 − 训练）" option={buildOverfitGapOption(history)}
              hint="差距由平到升的拐点，就是模型从学习转向记忆的位置。"
              emptyText="缺少训练/验证损失对，无法计算差距" />
            <ChartCard title="学习率变化" option={buildLearningRateOption(history)}
              hint="对数轴。学习率下降的位置应当对应验证损失的平台期。"
              emptyText="该模型没有记录学习率" />
            <ChartCard title="验证指标曲线" option={buildMetricHistoryOption(history)}
              hint="指标趋于平坦，说明继续训练收益有限。"
              emptyText="该模型没有逐轮验证指标" />
          </>
        ) : (
          <>
            <ChartCard title="分步训练指标" option={stepOption}
              hint="训练期间记录的分步指标，反映拟合是如何推进的。"
              emptyText="该模型没有分步训练记录" />
            <ChartCard title="各折得分" option={buildFoldScoresOption(folds)}
              hint="每一折单独的得分与均值线。个别折明显偏低，说明数据分布不均而非模型整体不行。"
              emptyText="该模型训练时未保存每折明细（重新训练后可见）" />
            <ChartCard title="跨折稳定性（变异系数）" option={stabilityOption}
              hint="标准差占均值的比例。超过 15%（虚线）说明结果对数据划分敏感。"
              emptyText="没有交叉验证聚合值" />
            <Col xs={24} xl={12}>
              <Card size="small" variant="outlined" title="训练配置"
                styles={{ body: { height: CHART_HEIGHT + 24, padding: 12, overflow: 'auto' } }}>
                <Descriptions column={1} size="small" bordered>
                  <Descriptions.Item label="任务类型">{taskType || '—'}</Descriptions.Item>
                  <Descriptions.Item label="交叉验证折数">
                    {folds.length || trainConfig?.cv_folds || '—'}
                  </Descriptions.Item>
                  <Descriptions.Item label="验证集比例">{trainConfig?.test_size ?? '—'}</Descriptions.Item>
                  {stability.map(row => (
                    <Descriptions.Item key={row.metric} label={row.metric}>
                      {row.mean.toFixed(4)}
                      {row.std != null && ` ± ${row.std.toFixed(4)}`}
                    </Descriptions.Item>
                  ))}
                </Descriptions>
              </Card>
            </Col>
          </>
        )}
      </Row>

      {stability.some(r => r.cv != null && r.cv > 15) && (
        <Alert type="warning" showIcon
          message="部分指标跨折波动较大"
          description="变异系数超过 15% 通常意味着样本量偏小或数据分布不均，单次得分不宜过度解读。" />
      )}
    </Space>
  )
}
