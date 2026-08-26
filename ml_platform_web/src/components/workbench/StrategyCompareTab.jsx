/**
 * StrategyCompareTab — baseline vs grid_search vs bayesian_search.
 *
 * Three summary cards (one per canonical strategy), a box-plot over the
 * objective metric keyed by strategy_type, and a ranking table of every
 * SUCCESS run — sortable by value.  Data from GET
 * /api/v3/tasks/:id/strategy-comparison (commit 8).
 *
 * Design choices:
 *   - Cards always show the 3 canonical strategies — strategies not yet run
 *     display a neutral "该策略未运行" placeholder rather than being hidden,
 *     so the user sees what's missing and can start that batch.
 *   - Box plot requires ≥2 SUCCESS runs per strategy to draw a box; single-
 *     run strategies degrade to a scatter dot.
 *   - The ranking table surfaces run_id, which is wired up via `onInspect`
 *     so clicking a row opens RunInspector on the SHAP tab (matches the
 *     navigation pattern introduced in v3.1.2 for drill-down).
 */
import React, { useEffect, useMemo, useState } from 'react'
import {
  Alert, Card, Col, Empty, Row, Space, Spin, Statistic, Tag, Typography,
} from 'antd'
import { TrophyOutlined, ReloadOutlined, ExperimentOutlined } from '@ant-design/icons'
import { modelingTaskApi } from '../../services/api'
import { buildStrategyCardVM } from '../../utils/comparison'
import EChart from '../EChart'

const { Text } = Typography

const CANONICAL = ['baseline', 'grid_search', 'bayesian_search']

const STRATEGY_META = {
  baseline:        { label: 'Baseline',        color: '#10b981', desc: '默认超参，一轮快速建模' },
  grid_search:     { label: 'Grid Search',     color: '#2563eb', desc: '笛卡尔积穷举超参' },
  bayesian_search: { label: 'Bayesian (TPE)',  color: '#8b5cf6', desc: 'Optuna TPE 自适应采样' },
}

function fmt(v, digits = 4) {
  if (v == null || Number.isNaN(v)) return '-'
  return Number(v).toFixed(digits)
}

export default function StrategyCompareTab({ taskId, onInspect }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const load = React.useCallback(async () => {
    if (!taskId) return
    setLoading(true)
    setError(null)
    try {
      const resp = await modelingTaskApi.strategyComparison(taskId)
      setData(resp)
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || '加载策略对比失败')
    } finally {
      setLoading(false)
    }
  }, [taskId])

  useEffect(() => { void load() }, [load])

  const strategyMap = useMemo(() => {
    const out = {}
    for (const s of (data?.strategies || [])) out[s.strategy_type] = s
    return out
  }, [data])

  const boxSeriesData = useMemo(() => {
    // ECharts boxplot format: [[min, Q1, median, Q3, max], ...].
    // One box per strategy that has ≥2 success runs.
    const strategies = data?.strategies || []
    return strategies
      .filter(s => s.stats && s.run_count >= 2)
      .map(s => [s.stats.min, s.stats.q1, s.stats.median, s.stats.q3, s.stats.max])
  }, [data])

  // Strip plot points (one dot per SUCCESS run) so single-run strategies
  // still have a visible marker even when the box plot skips them.
  const stripPoints = useMemo(() => {
    const points = data?.raw_points || []
    const idxByStrategy = new Map(
      (data?.strategies || []).map((s, i) => [s.strategy_type, i]),
    )
    return points.map(p => ({
      value: [idxByStrategy.get(p.strategy_type) ?? 0, p.value],
      ...p,
    }))
  }, [data])

  const boxOption = useMemo(() => {
    if (!data?.strategies?.length) return null
    const categories = (data.strategies || []).map(
      s => STRATEGY_META[s.strategy_type]?.label || s.strategy_type,
    )
    return {
      title: {
        text: `${data.metric_name} (${data.objective_direction === 'max' ? '越高越好' : '越低越好'})`,
        left: 'center', top: 4, textStyle: { fontSize: 13 },
      },
      tooltip: { trigger: 'item' },
      legend: { bottom: 4, textStyle: { fontSize: 11 } },
      grid: { top: 44, left: 60, right: 20, bottom: 56 },
      xAxis: { type: 'category', data: categories, boundaryGap: true, axisLabel: { fontSize: 12 } },
      yAxis: { type: 'value', scale: true, name: data.metric_name },
      series: [
        {
          name: '分布（箱线）',
          type: 'boxplot',
          data: boxSeriesData,
          itemStyle: { borderColor: '#2563eb', color: 'rgba(37, 99, 235, 0.2)' },
        },
        {
          name: '单个 Run',
          type: 'scatter',
          data: stripPoints,
          symbolSize: 7,
          itemStyle: {
            color: p => {
              const strat = p.data.strategy_type
              return STRATEGY_META[strat]?.color || '#94a3b8'
            },
            opacity: 0.75,
          },
          tooltip: {
            formatter: p => {
              const d = p.data
              return `<b>${STRATEGY_META[d.strategy_type]?.label || d.strategy_type}</b><br/>`
                + `${data.metric_name}: ${fmt(d.value[1])}<br/>`
                + `model: ${d.model_type || '-'}<br/>`
                + `trial #${d.trial_no ?? '-'}`
            },
          },
        },
      ],
    }
  }, [data, boxSeriesData, stripPoints])


  if (loading && !data) {
    return <div style={{ textAlign: 'center', padding: 60 }}><Spin tip="加载策略对比…" /></div>
  }
  if (error) {
    return <Alert type="error" showIcon message="加载失败" description={error} />
  }
  if (!data || (data.strategies?.length ?? 0) === 0) {
    return (
      <Empty description={
        <div>
          <div>该任务尚无任何实验批次</div>
          <Text type="secondary" style={{ fontSize: 12 }}>
            启动 baseline / grid / bayesian 批次后，可在此处对比各策略的表现分布。
          </Text>
        </div>
      } />
    )
  }

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      {/* Three strategy cards — always show all canonical strategies */}
      <Row gutter={[12, 12]}>
        {CANONICAL.map(key => {
          const s = strategyMap[key]
          const meta = STRATEGY_META[key]
          const card = s ? buildStrategyCardVM(s) : null
          const best = card?.bestRun
          return (
            <Col xs={24} md={8} key={key}>
              <Card
                variant="outlined"
                size="small"
                title={
                  <Space>
                    <Tag color={meta.color} style={{ fontWeight: 500 }}>{meta.label}</Tag>
                    <Text type="secondary" style={{ fontSize: 11 }}>{meta.desc}</Text>
                  </Space>
                }
                styles={{ body: { padding: '12px 16px' } }}
              >
                {s && card.hasBestRun ? (
                  <Space direction="vertical" size={6} style={{ width: '100%' }}>
                    <Statistic
                      title={`最佳 ${data.metric_name}`}
                      value={best?.objective_value ?? NaN}
                      precision={4}
                      valueStyle={{ fontSize: 22, color: meta.color }}
                      prefix={<TrophyOutlined style={{ fontSize: 16 }} />}
                    />
                    <div style={{ fontSize: 12, color: '#475569' }}>
                      <Text type="secondary">模型: </Text>
                      <Text code>{best?.model_type || '-'}</Text>
                      <br />
                      <Text type="secondary">Run 数: </Text>
                      <Text>{card.runCount}/{card.fullRunCount}（成功/总）</Text>
                      {best?.run_id && (
                        <>
                          <br />
                          <a onClick={() => onInspect?.(best.run_id)}>查看最佳 Run →</a>
                        </>
                      )}
                    </div>
                  </Space>
                ) : (
                  <Empty
                    imageStyle={{ height: 50 }}
                    description={<Text type="secondary" style={{ fontSize: 12 }}>
                      {s ? `暂无成功 Run（0/${card.fullRunCount}）` : '该策略未运行'}
                    </Text>}
                  />
                )}
              </Card>
            </Col>
          )
        })}
      </Row>

      {/* Box plot */}
      <Card
        variant="outlined"
        size="small"
        title={<Space><ExperimentOutlined />策略表现分布</Space>}
        extra={<a onClick={load}><ReloadOutlined /> 刷新</a>}
      >
        {boxOption ? (
          <EChart option={boxOption} style={{ height: 340 }} />
        ) : (
          <Empty description="暂无足够的 Run 数据渲染分布图（每个策略至少需要 2 个成功 Run）" />
        )}
      </Card>

      {/* The Run 总排行榜 table that used to sit here was the same rows, the
          same ordering and the same actions as 模型对比 directly above it in
          the 模型排名 tab — two leaderboards, one screen. This panel now keeps
          only what is genuinely per-*strategy*: the spread comparison. */}
    </Space>
  )
}
