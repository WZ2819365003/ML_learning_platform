/**
 * V3 Experiment Detail — leaderboard, run list, and metric comparison charts.
 * Loaded at /experiments/:experimentId
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  Alert, Button, Card, Col, Descriptions, Empty, message,
  Row, Skeleton, Space, Spin, Table, Tabs, Tag, Tooltip, Typography,
} from 'antd'
import {
  ArrowLeftOutlined, BulbOutlined, ReloadOutlined,
  ThunderboltOutlined, TrophyOutlined,
} from '@ant-design/icons'
import * as echarts from 'echarts'
import { platformExperimentsApi } from '../services/api'

const { Title, Text } = Typography

function useChart(ref, option) {
  useEffect(() => {
    if (!ref.current || !option) return
    const inst = echarts.init(ref.current)
    inst.setOption(option, true)
    const onResize = () => inst.resize()
    window.addEventListener('resize', onResize)
    return () => { window.removeEventListener('resize', onResize); inst.dispose() }
  }, [option])
}

const STATUS_TAG = { SUCCESS: 'success', RUNNING: 'processing', FAILED: 'error', PENDING: 'warning' }

const ExperimentDetail = () => {
  const { experimentId } = useParams()
  const navigate = useNavigate()
  const barRef  = useRef(null)
  const radarRef = useRef(null)

  const [experiment, setExperiment] = useState(null)
  const [leaderboard, setLeaderboard] = useState([])
  const [loading, setLoading] = useState(true)

  // SHAP explain state
  const [shapRunId, setShapRunId]         = useState(null)   // which run is being explained
  const [shapData, setShapData]           = useState(null)   // { feature_importances: {...} }
  const [shapTriggering, setShapTriggering] = useState(false)
  const [shapLoading, setShapLoading]     = useState(false)
  const shapRef = useRef(null)

  const fetchAll = async () => {
    setLoading(true)
    try {
      const [expData, lbData] = await Promise.all([
        platformExperimentsApi.get(experimentId),
        platformExperimentsApi.getLeaderboard(experimentId),
      ])
      setExperiment(expData)
      setLeaderboard(lbData)
    } catch { message.error('加载实验详情失败') }
    finally { setLoading(false) }
  }

  useEffect(() => { fetchAll() }, [experimentId])

  // ── SHAP explain handlers ────────────────────────────────────────────────

  /** Load an already-computed SHAP result for a run. */
  const loadExplain = useCallback(async (runId) => {
    setShapRunId(runId)
    setShapLoading(true)
    setShapData(null)
    try {
      const res = await platformExperimentsApi.getExplain(experimentId, runId)
      setShapData(res)
    } catch (err) {
      if (err?.response?.status === 404) {
        setShapData(null)  // not computed yet — user needs to trigger
      } else {
        message.error('加载 SHAP 结果失败')
      }
    } finally {
      setShapLoading(false)
    }
  }, [experimentId])

  /** Trigger SHAP computation for a run, then load the result. */
  const handleTriggerExplain = useCallback(async (runId) => {
    setShapRunId(runId)
    setShapTriggering(true)
    setShapData(null)
    try {
      const res = await platformExperimentsApi.triggerExplain(experimentId, runId)
      if (res.already_computed) {
        // Already done — just fetch
        await loadExplain(runId)
        return
      }
      message.info('SHAP 计算已启动，请稍候…')
      // Poll every 2 s, up to 60 s
      const deadline = Date.now() + 60_000
      const poll = async () => {
        try {
          const result = await platformExperimentsApi.getExplain(experimentId, runId)
          setShapData(result)
        } catch (err) {
          if (err?.response?.status === 404 && Date.now() < deadline) {
            setTimeout(poll, 2000)
          } else if (Date.now() >= deadline) {
            message.warning('SHAP 计算超时，请稍后刷新重试')
          }
        }
      }
      setTimeout(poll, 2000)
    } catch (err) {
      const detail = err?.response?.data?.detail
      message.error(detail ? `触发失败: ${detail}` : '触发 SHAP 解释失败')
    } finally {
      setShapTriggering(false)
    }
  }, [experimentId, loadExplain])

  // Auto-load SHAP for best run when leaderboard is ready
  useEffect(() => {
    if (experiment?.best_run_id && leaderboard.length > 0) {
      loadExplain(experiment.best_run_id)
    }
  }, [experiment?.best_run_id, leaderboard.length, loadExplain])

  // ── Bar chart: metric comparison across top runs ─────────────────────────
  const barOption = useMemo(() => {
    if (leaderboard.length === 0) return null
    const top = leaderboard.slice(0, 8)
    const metricKey = experiment?.objective_metric || 'accuracy'
    return {
      grid: { top: 12, right: 16, bottom: 32, left: 16, containLabel: true },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        backgroundColor: 'rgba(15,23,42,0.85)',
        borderColor: 'transparent',
        textStyle: { color: '#f8fafc', fontSize: 12 },
      },
      xAxis: {
        type: 'category',
        data: top.map((r, i) => r.params?.model_type ?? `Run ${i + 1}`),
        axisLabel: { color: '#94a3b8', fontSize: 11, rotate: top.length > 5 ? 15 : 0 },
        axisTick: { show: false },
        axisLine: { lineStyle: { color: '#e2e8f0' } },
      },
      yAxis: {
        type: 'value',
        max: 1,
        axisLabel: { color: '#94a3b8', fontSize: 11, formatter: v => `${(v * 100).toFixed(0)}%` },
        splitLine: { lineStyle: { color: '#f1f5f9' } },
      },
      series: [{
        type: 'bar',
        data: top.map(r => +(r.metrics?.[metricKey] ?? 0)),
        barMaxWidth: 40,
        itemStyle: {
          borderRadius: [6, 6, 0, 0],
          color: (params) => {
            const colors = ['#2563eb', '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe', '#dbeafe', '#eff6ff', '#f0f9ff']
            return colors[params.dataIndex % colors.length]
          },
        },
        label: {
          show: true, position: 'top', fontSize: 11,
          formatter: p => `${(p.value * 100).toFixed(1)}%`,
          color: '#64748b',
        },
      }],
    }
  }, [leaderboard, experiment])

  // ── Radar chart: multi-metric comparison of top 3 runs ──────────────────
  const radarOption = useMemo(() => {
    const top3 = leaderboard.slice(0, 3)
    if (top3.length < 2) return null

    const allMetricKeys = [...new Set(top3.flatMap(r => Object.keys(r.metrics || {})))]
      .filter(k => typeof top3[0]?.metrics?.[k] === 'number')
      .slice(0, 6)

    if (allMetricKeys.length < 2) return null

    const indicators = allMetricKeys.map(k => ({
      name: k,
      max: Math.max(...top3.map(r => +(r.metrics?.[k] ?? 0))) * 1.1 || 1,
    }))

    const COLORS = ['#2563eb', '#10b981', '#f59e0b']

    return {
      tooltip: {
        backgroundColor: 'rgba(15,23,42,0.85)',
        borderColor: 'transparent',
        textStyle: { color: '#f8fafc', fontSize: 12 },
      },
      legend: {
        bottom: 0, icon: 'circle', itemWidth: 8,
        textStyle: { color: '#64748b', fontSize: 11 },
        data: top3.map((r, i) => r.params?.model_type ?? `Run ${i + 1}`),
      },
      radar: {
        indicator: indicators,
        shape: 'polygon',
        splitNumber: 4,
        axisName: { color: '#64748b', fontSize: 11 },
        splitLine: { lineStyle: { color: '#e2e8f0' } },
        splitArea: { show: false },
        axisLine: { lineStyle: { color: '#e2e8f0' } },
      },
      series: [{
        type: 'radar',
        data: top3.map((r, i) => ({
          name: r.params?.model_type ?? `Run ${i + 1}`,
          value: allMetricKeys.map(k => +(r.metrics?.[k] ?? 0)),
          lineStyle: { color: COLORS[i] },
          areaStyle: { color: COLORS[i], opacity: 0.1 },
          symbol: 'circle',
          symbolSize: 5,
        })),
      }],
    }
  }, [leaderboard])

  // ── SHAP feature importance chart ───────────────────────────────────────
  const shapOption = useMemo(() => {
    if (!shapData?.feature_importances) return null
    const entries = Object.entries(shapData.feature_importances)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 15)
    const features = entries.map(([k]) => k).reverse()
    const values   = entries.map(([, v]) => v).reverse()
    return {
      grid: { top: 12, right: 24, bottom: 24, left: 16, containLabel: true },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params) => {
          const p = params[0]
          return `${p.name}<br/><b>${p.value.toFixed(4)}</b>`
        },
      },
      xAxis: {
        type: 'value',
        name: '平均 |SHAP| 值',
        nameLocation: 'end',
        nameTextStyle: { color: '#94a3b8', fontSize: 11 },
        axisLabel: { color: '#94a3b8', fontSize: 11 },
        splitLine: { lineStyle: { color: '#f1f5f9' } },
      },
      yAxis: {
        type: 'category',
        data: features,
        axisLabel: { color: '#374151', fontSize: 12 },
        axisTick: { show: false },
        axisLine: { lineStyle: { color: '#e2e8f0' } },
      },
      series: [{
        type: 'bar',
        data: values,
        barMaxWidth: 32,
        itemStyle: {
          borderRadius: [0, 4, 4, 0],
          color: {
            type: 'linear', x: 0, y: 0, x2: 1, y2: 0,
            colorStops: [
              { offset: 0, color: '#6366f1' },
              { offset: 1, color: '#a78bfa' },
            ],
          },
        },
        label: {
          show: true, position: 'right', fontSize: 11, color: '#64748b',
          formatter: p => p.value.toFixed(3),
        },
      }],
    }
  }, [shapData])

  useChart(barRef, barOption)
  useChart(radarRef, radarOption)
  useChart(shapRef, shapOption)

  const leaderboardColumns = [
    { title: '排名', dataIndex: 'rank', width: 60, render: v => v ? <Text strong>#{v}</Text> : '—' },
    {
      title: 'Run ID',
      dataIndex: 'id',
      width: 100,
      render: (v, r) => (
        <Space>
          {r.rank === 1 && <TrophyOutlined style={{ color: '#f59e0b' }} />}
          <Text code style={{ fontSize: 11 }}>{v.slice(0, 8)}…</Text>
        </Space>
      ),
    },
    {
      title: '模型',
      render: (_, r) => r.params?.model_type
        ? <Tag color="blue">{r.params.model_type}</Tag>
        : <Text type="secondary">—</Text>,
    },
    {
      title: '主要指标',
      render: (_, r) => {
        const k = experiment?.objective_metric || 'accuracy'
        const v = r.metrics?.[k]
        return v != null
          ? <Text style={{ color: '#10b981', fontWeight: 700 }}>{(v * 100).toFixed(2)}%</Text>
          : '—'
      },
    },
    {
      title: '全部指标',
      render: (_, r) => (
        <Space size={4} wrap>
          {Object.entries(r.metrics || {}).slice(0, 4).map(([k, v]) => (
            <Tag key={k} style={{ fontSize: 10 }}>
              {k}: {typeof v === 'number' ? (v * 100).toFixed(1) + '%' : String(v)}
            </Tag>
          ))}
        </Space>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      width: 90,
      render: v => <Tag color={STATUS_TAG[v] ?? 'default'} style={{ fontSize: 11 }}>{v}</Tag>,
    },
    {
      title: '完成时间',
      dataIndex: 'finished_at',
      width: 150,
      render: v => v
        ? <Text style={{ fontSize: 11, color: '#94a3b8' }}>{new Date(v).toLocaleString('zh-CN', { hour12: false })}</Text>
        : '—',
    },
    {
      title: 'SHAP',
      width: 80,
      render: (_, r) => r.status !== 'SUCCESS' ? null : (
        <Tooltip title="触发/查看 SHAP 解释">
          <Button
            size="small"
            type={shapRunId === r.id && shapData ? 'primary' : 'default'}
            ghost={shapRunId === r.id && !!shapData}
            icon={<BulbOutlined />}
            loading={shapTriggering && shapRunId === r.id}
            onClick={() => handleTriggerExplain(r.id)}
            style={{ fontSize: 11 }}
          />
        </Tooltip>
      ),
    },
  ]

  if (loading) return <Card style={{ margin: '20px 0' }}><Skeleton active /></Card>
  if (!experiment) return null

  return (
    <div style={{ padding: '20px 0' }}>
      <Space direction="vertical" size={20} style={{ width: '100%', display: 'flex' }}>

        {/* Back + title */}
        <div className="hero-banner">
          <Button
            icon={<ArrowLeftOutlined />} ghost size="small"
            style={{ marginBottom: 12 }}
            onClick={() => navigate('/experiments')}
          >
            返回实验列表
          </Button>
          <Title level={2} style={{ color: '#fff', margin: 0, fontWeight: 800 }}>{experiment.name}</Title>
          <Text style={{ color: 'rgba(255,255,255,0.65)', fontSize: 13 }}>
            {experiment.description || `${experiment.kind} · ${experiment.objective_metric} (${experiment.objective_direction})`}
          </Text>
        </div>

        {/* Experiment meta */}
        <Card title="实验信息" extra={<Button icon={<ReloadOutlined />} size="small" onClick={fetchAll}>刷新</Button>}>
          <Descriptions size="small" column={3} bordered>
            <Descriptions.Item label="状态">
              <Tag color={STATUS_TAG[experiment.status] ?? 'default'}>{experiment.status}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="类型">{experiment.kind}</Descriptions.Item>
            <Descriptions.Item label="数据集">{experiment.dataset_name ?? '—'}</Descriptions.Item>
            <Descriptions.Item label="优化目标">{experiment.objective_metric}</Descriptions.Item>
            <Descriptions.Item label="优化方向">{experiment.objective_direction === 'max' ? '最大化' : '最小化'}</Descriptions.Item>
            <Descriptions.Item label="最优 Run ID">
              {experiment.best_run_id
                ? <Text code style={{ fontSize: 11 }}>{experiment.best_run_id.slice(0, 12)}…</Text>
                : '—'}
            </Descriptions.Item>
          </Descriptions>
        </Card>

        {/* Tabbed content: leaderboard / charts / SHAP */}
        <Card bodyStyle={{ padding: '0 24px 24px' }}>
          <Tabs
            defaultActiveKey="leaderboard"
            items={[
              {
                key: 'leaderboard',
                label: `排行榜 · ${leaderboard.length} 个完成 Run`,
                children: leaderboard.length === 0
                  ? <Empty description="还没有完成的 Run — 请先提交训练任务" style={{ padding: '32px 0' }} />
                  : <Table
                      rowKey="id"
                      dataSource={leaderboard.map(r => ({ ...r, key: r.id }))}
                      columns={leaderboardColumns}
                      pagination={false}
                      size="small"
                      rowClassName={r => r.rank === 1 ? 'ant-table-row-selected' : ''}
                    />,
              },
              {
                key: 'charts',
                label: '指标对比图',
                children: leaderboard.length === 0
                  ? <Empty description="暂无完成的 Run" style={{ padding: '32px 0' }} />
                  : (
                    <Row gutter={[16, 16]}>
                      <Col xs={24} lg={barOption && radarOption ? 14 : 24}>
                        <Card
                          title={`${experiment.objective_metric} 对比`}
                          bordered={false}
                          bodyStyle={{ padding: '8px 0 0' }}
                        >
                          <div ref={barRef} style={{ height: 260 }} />
                        </Card>
                      </Col>
                      {radarOption && (
                        <Col xs={24} lg={10}>
                          <Card
                            title="多指标雷达（前3名）"
                            bordered={false}
                            bodyStyle={{ padding: '8px 0 0' }}
                          >
                            <div ref={radarRef} style={{ height: 260 }} />
                          </Card>
                        </Col>
                      )}
                    </Row>
                  ),
              },
              {
                key: 'shap',
                label: (
                  <Space size={4}>
                    <BulbOutlined />
                    <span>SHAP 模型解释</span>
                    {shapData && <Tag color="green" style={{ fontSize: 10, marginLeft: 2 }}>已完成</Tag>}
                  </Space>
                ),
                children: (
                  <div>
                    {/* Run selector info */}
                    {shapRunId && (
                      <Alert
                        type="info"
                        showIcon
                        style={{ marginBottom: 16 }}
                        message={
                          <Space>
                            <span>
                              当前解释 Run：<Text code style={{ fontSize: 11 }}>{shapRunId.slice(0, 12)}…</Text>
                            </span>
                            <Button
                              size="small"
                              icon={<ThunderboltOutlined />}
                              loading={shapTriggering}
                              onClick={() => handleTriggerExplain(shapRunId)}
                            >
                              重新计算
                            </Button>
                          </Space>
                        }
                      />
                    )}

                    {!shapRunId && (
                      <Alert
                        type="info"
                        showIcon
                        style={{ marginBottom: 16 }}
                        message='在"排行榜"tab 的 SHAP 列点击灯泡图标，为指定 Run 触发 SHAP 解释。'
                      />
                    )}

                    <Spin spinning={shapLoading || shapTriggering}>
                      {shapData?.feature_importances ? (
                        <>
                          <div style={{ color: '#64748b', fontSize: 12, marginBottom: 12 }}>
                            特征重要性（平均 |SHAP| 值，前 15 个特征）
                            {shapData.sample_size && (
                              <Tag style={{ marginLeft: 8, fontSize: 10 }}>
                                采样 {shapData.sample_size} 行
                              </Tag>
                            )}
                          </div>
                          <div ref={shapRef} style={{ height: 420, width: '100%' }} />
                        </>
                      ) : (
                        !shapLoading && !shapTriggering && shapRunId && (
                          <Empty
                            description={
                              <span>
                                该 Run 尚未计算 SHAP —{' '}
                                <Button
                                  type="link" style={{ padding: 0 }}
                                  onClick={() => handleTriggerExplain(shapRunId)}
                                >
                                  立即触发
                                </Button>
                              </span>
                            }
                            style={{ padding: '48px 0' }}
                          />
                        )
                      )}
                    </Spin>
                  </div>
                ),
              },
            ]}
          />
        </Card>

      </Space>
    </div>
  )
}

export default ExperimentDetail
