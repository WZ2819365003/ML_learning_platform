/**
 * V3 Experiment Detail — leaderboard, run list, and metric comparison charts.
 * Loaded at /experiments/:experimentId
 */

import React, { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  Button, Card, Col, Descriptions, Empty, message,
  Row, Skeleton, Space, Table, Tag, Typography,
} from 'antd'
import {
  ArrowLeftOutlined, TrophyOutlined, ReloadOutlined, ThunderboltOutlined,
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

  useChart(barRef, barOption)
  useChart(radarRef, radarOption)

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

        {/* Charts */}
        <Row gutter={[16, 16]}>
          <Col xs={24} lg={barOption && radarOption ? 14 : 24}>
            <Card title={`${experiment.objective_metric} 排行`} bodyStyle={{ padding: '8px 20px 16px' }}>
              {leaderboard.length === 0
                ? <Empty description="暂无完成的 Run" />
                : <div ref={barRef} style={{ height: 260 }} />
              }
            </Card>
          </Col>
          {radarOption && (
            <Col xs={24} lg={10}>
              <Card title="多指标雷达图（前3名）" bodyStyle={{ padding: '8px 20px 16px' }}>
                <div ref={radarRef} style={{ height: 260 }} />
              </Card>
            </Col>
          )}
        </Row>

        {/* Leaderboard */}
        <Card title={`排行榜 · ${leaderboard.length} 个完成 Run`}>
          {leaderboard.length === 0
            ? <Empty description="还没有完成的 Run — 请先提交训练任务" />
            : <Table
                rowKey="id"
                dataSource={leaderboard.map(r => ({ ...r, key: r.id }))}
                columns={leaderboardColumns}
                pagination={false}
                size="small"
                rowClassName={(r) => r.rank === 1 ? 'ant-table-row-selected' : ''}
              />
          }
        </Card>

      </Space>
    </div>
  )
}

export default ExperimentDetail
