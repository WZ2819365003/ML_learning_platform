/**
 * V3 Task Center — unified view of all PlatformTasks across every kind.
 *
 * Features:
 *  - Global stat cards (total counts from /stats endpoint, not page-level)
 *  - Filter by kind and status
 *  - Auto-refresh every 5 s when active tasks exist
 *  - Retry / Cancel / Delete actions
 *  - Expandable rows showing error detail and metrics snapshot
 *  - Duration column computed from started_at / finished_at
 */

import React, { useCallback, useEffect, useState } from 'react'
import {
  Button, Card, Col, Empty, message, Popconfirm,
  Row, Select, Space, Spin, Statistic, Table, Tag, Tooltip, Typography,
} from 'antd'
import {
  CheckCircleOutlined, ClockCircleOutlined, CloseCircleOutlined,
  ExclamationCircleOutlined, ReloadOutlined, RedoOutlined, SyncOutlined,
} from '@ant-design/icons'
import { platformTasksApi } from '../services/api'

const { Text, Title } = Typography

// ── Status display config ──────────────────────────────────────────────────
const STATUS_CONFIG = {
  SUCCESS:   { color: '#10b981', bg: 'rgba(16,185,129,0.10)',  icon: <CheckCircleOutlined />,       label: '成功' },
  RUNNING:   { color: '#3b82f6', bg: 'rgba(59,130,246,0.10)',  icon: <SyncOutlined spin />,         label: '运行中' },
  QUEUED:    { color: '#6366f1', bg: 'rgba(99,102,241,0.10)',  icon: <ClockCircleOutlined />,       label: '已排队' },
  PENDING:   { color: '#f59e0b', bg: 'rgba(245,158,11,0.10)',  icon: <ClockCircleOutlined />,       label: '等待中' },
  FAILED:    { color: '#ef4444', bg: 'rgba(239,68,68,0.10)',   icon: <ExclamationCircleOutlined />, label: '失败' },
  RETRY:     { color: '#f97316', bg: 'rgba(249,115,22,0.10)',  icon: <RedoOutlined />,              label: '重试' },
  CANCELLED: { color: '#94a3b8', bg: 'rgba(148,163,184,0.10)', icon: <CloseCircleOutlined />,      label: '已取消' },
}

const KIND_LABELS = {
  train:      '训练',
  dl_train:   'DL训练',
  explain:    'SHAP解释',
  eval:       '评估',
  predict:    '预测',
  preprocess: '预处理',
  automl:     'AutoML',
}

// ── Helpers ────────────────────────────────────────────────────────────────

function StatusBadge({ status }) {
  const key = status?.toUpperCase()
  const cfg = STATUS_CONFIG[key] ?? {
    color: '#94a3b8', bg: 'rgba(148,163,184,0.1)', icon: null, label: status,
  }
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 5,
      padding: '2px 10px', borderRadius: 99,
      fontSize: 11, fontWeight: 600,
      background: cfg.bg, color: cfg.color,
    }}>
      {cfg.icon} {cfg.label}
    </span>
  )
}

/** Format duration between two ISO strings. */
function formatDuration(startIso, endIso) {
  if (!startIso) return '—'
  const start = new Date(startIso).getTime()
  const end   = endIso ? new Date(endIso).getTime() : Date.now()
  const secs  = Math.round((end - start) / 1000)
  if (secs < 60)   return `${secs}s`
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`
  const h = Math.floor(secs / 3600)
  const m = Math.floor((secs % 3600) / 60)
  return `${h}h ${m}m`
}

const ACTIVE_STATUSES = new Set(['RUNNING', 'QUEUED', 'PENDING', 'RETRY'])

// ── Main component ─────────────────────────────────────────────────────────
const TaskCenter = () => {
  const [tasks, setTasks]               = useState([])
  const [total, setTotal]               = useState(0)
  const [page, setPage]                 = useState(1)
  const [loading, setLoading]           = useState(false)
  const [globalStats, setGlobalStats]   = useState(null)  // from /stats endpoint
  const [filterKind, setFilterKind]     = useState(undefined)
  const [filterStatus, setFilterStatus] = useState(undefined)

  const PAGE_SIZE = 20

  // ── Data fetching ─────────────────────────────────────────────────────────

  const fetchStats = useCallback(async () => {
    try {
      const res = await platformTasksApi.stats()
      setGlobalStats(res)
    } catch {
      // Non-critical — fall back to page-level counts
    }
  }, [])

  const fetchTasks = useCallback(async (p = 1) => {
    setLoading(true)
    try {
      const res = await platformTasksApi.list({
        page: p,
        page_size: PAGE_SIZE,
        ...(filterKind   ? { kind: filterKind }     : {}),
        ...(filterStatus ? { status: filterStatus } : {}),
      })
      setTasks(res.items || [])
      setTotal(res.total || 0)
      setPage(p)
    } catch {
      message.error('加载任务列表失败')
    } finally {
      setLoading(false)
    }
  }, [filterKind, filterStatus])

  // Refresh both list and stats on filter change
  useEffect(() => {
    fetchTasks(1)
    fetchStats()
  }, [filterKind, filterStatus])  // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-refresh every 5 s while active tasks exist on the current page
  useEffect(() => {
    const hasActive = tasks.some(t => ACTIVE_STATUSES.has(t.status?.toUpperCase()))
    if (!hasActive) return undefined
    const timer = setInterval(() => {
      fetchTasks(page)
      fetchStats()
    }, 5000)
    return () => clearInterval(timer)
  }, [tasks, page, fetchTasks, fetchStats])

  // ── Actions ───────────────────────────────────────────────────────────────

  const handleRetry = async (id) => {
    try {
      await platformTasksApi.retry(id)
      message.success('任务已重新提交')
      fetchTasks(page)
      fetchStats()
    } catch (err) {
      const detail = err?.response?.data?.detail
      message.error(detail ? `重试失败: ${detail}` : '重试失败')
    }
  }

  const handleCancel = async (id) => {
    try {
      await platformTasksApi.cancel(id)
      message.success('任务已取消')
      fetchTasks(page)
      fetchStats()
    } catch (err) {
      const detail = err?.response?.data?.detail
      message.error(detail ? `取消失败: ${detail}` : '取消失败')
    }
  }

  const handleDelete = async (id) => {
    try {
      await platformTasksApi.delete(id)
      message.success('已删除')
      fetchTasks(page)
      fetchStats()
    } catch (err) {
      const detail = err?.response?.data?.detail
      message.error(detail ? `删除失败: ${detail}` : '删除失败')
    }
  }

  // ── Summary stats (prefer global, fall back to page counts) ───────────────
  const pageCounts = tasks.reduce((acc, t) => {
    const s = t.status?.toUpperCase()
    acc[s] = (acc[s] || 0) + 1
    return acc
  }, {})

  const statsCards = globalStats
    ? [
        { label: '运行中',  value: globalStats.running,   color: '#3b82f6' },
        { label: '已排队',  value: globalStats.queued,    color: '#6366f1' },
        { label: '成功',    value: globalStats.succeeded, color: '#10b981' },
        { label: '失败',    value: globalStats.failed,    color: '#ef4444' },
      ]
    : [
        { label: '运行中',  value: pageCounts.RUNNING || 0,                           color: '#3b82f6' },
        { label: '已排队',  value: (pageCounts.QUEUED || 0) + (pageCounts.PENDING || 0), color: '#6366f1' },
        { label: '成功',    value: pageCounts.SUCCESS || 0,                           color: '#10b981' },
        { label: '失败',    value: (pageCounts.FAILED || 0) + (pageCounts.RETRY || 0), color: '#ef4444' },
      ]

  // ── Columns ───────────────────────────────────────────────────────────────
  const columns = [
    {
      title: '任务 ID',
      dataIndex: 'id',
      width: 120,
      render: v => (
        <Tooltip title={v}>
          <Text code style={{ fontSize: 11 }}>{v.slice(0, 8)}…</Text>
        </Tooltip>
      ),
    },
    {
      title: '类型',
      dataIndex: 'kind',
      width: 100,
      render: v => <Tag color="blue" style={{ fontSize: 11 }}>{KIND_LABELS[v] ?? v}</Tag>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      width: 110,
      render: v => <StatusBadge status={v} />,
    },
    {
      title: '耗时',
      width: 90,
      render: (_, r) => {
        const isActive = ACTIVE_STATUSES.has(r.status?.toUpperCase())
        return (
          <Text style={{ fontSize: 12, color: isActive ? '#3b82f6' : '#64748b' }}>
            {formatDuration(r.started_at, r.finished_at)}
            {isActive && <SyncOutlined spin style={{ marginLeft: 4, fontSize: 10 }} />}
          </Text>
        )
      },
    },
    {
      title: '进度',
      dataIndex: 'progress',
      width: 70,
      render: v => (
        <Text style={{ fontSize: 12, color: '#64748b' }}>
          {((v ?? 0) * 100).toFixed(0)}%
        </Text>
      ),
    },
    {
      title: '重试',
      width: 60,
      render: (_, r) => (
        <Text style={{ fontSize: 12, color: '#64748b' }}>
          {r.retry_count ?? 0}/{r.max_retries ?? 3}
        </Text>
      ),
    },
    {
      title: '关联任务',
      dataIndex: 'payload_ref',
      ellipsis: true,
      render: v => v
        ? <Tooltip title={v}><Text style={{ fontSize: 11, color: '#94a3b8' }}>{v}</Text></Tooltip>
        : '—',
    },
    {
      title: '入队时间',
      dataIndex: 'queued_at',
      width: 150,
      render: v => v
        ? <Text style={{ fontSize: 11, color: '#94a3b8' }}>{new Date(v).toLocaleString('zh-CN', { hour12: false })}</Text>
        : '—',
    },
    {
      title: '操作',
      width: 150,
      render: (_, r) => {
        const status = r.status?.toUpperCase()
        return (
          <Space size={4}>
            {(status === 'FAILED' || status === 'RETRY') && (
              <Button
                size="small"
                icon={<RedoOutlined />}
                onClick={() => handleRetry(r.id)}
              >
                重试
              </Button>
            )}
            {(status === 'PENDING' || status === 'QUEUED') && (
              <Popconfirm
                title="确认取消该任务？"
                onConfirm={() => handleCancel(r.id)}
                okText="取消任务"
                cancelText="返回"
              >
                <Button size="small" danger icon={<CloseCircleOutlined />}>取消</Button>
              </Popconfirm>
            )}
            {(status === 'SUCCESS' || status === 'FAILED' || status === 'CANCELLED') && (
              <Popconfirm
                title="确认删除该记录？"
                onConfirm={() => handleDelete(r.id)}
                okText="删除"
                cancelText="返回"
              >
                <Button size="small" type="text" danger>删除</Button>
              </Popconfirm>
            )}
          </Space>
        )
      },
    },
  ]

  // ── Expandable row — error message + metrics snapshot ─────────────────────
  const expandedRowRender = (record) => {
    const hasError   = record.status?.toUpperCase() === 'FAILED' && record.error_message
    const hasMetrics = record.metrics_snapshot && Object.keys(record.metrics_snapshot).length > 0

    if (!hasError && !hasMetrics) return null

    return (
      <div style={{ padding: '8px 48px 12px', background: '#fafafa', borderRadius: 6 }}>
        {hasError && (
          <div style={{ marginBottom: hasMetrics ? 12 : 0 }}>
            <Text type="secondary" style={{ fontSize: 11, display: 'block', marginBottom: 4 }}>
              错误信息
            </Text>
            <Text
              code
              style={{
                fontSize: 11, color: '#ef4444',
                background: '#fff1f2', padding: '6px 10px',
                borderRadius: 4, display: 'block', whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}
            >
              {record.error_message}
            </Text>
          </div>
        )}
        {hasMetrics && (
          <div>
            <Text type="secondary" style={{ fontSize: 11, display: 'block', marginBottom: 4 }}>
              指标快照
            </Text>
            <Space wrap size={[8, 4]}>
              {Object.entries(record.metrics_snapshot).map(([k, v]) => (
                <Tag key={k} style={{ fontSize: 11 }}>
                  {k}: {typeof v === 'number' ? v.toFixed(4) : String(v)}
                </Tag>
              ))}
            </Space>
          </div>
        )}
      </div>
    )
  }

  const rowExpandable = (record) =>
    (record.status?.toUpperCase() === 'FAILED' && !!record.error_message) ||
    (record.metrics_snapshot && Object.keys(record.metrics_snapshot).length > 0)

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={{ padding: '20px 0' }}>
      <Space direction="vertical" size={20} style={{ width: '100%', display: 'flex' }}>

        {/* Hero */}
        <div className="hero-banner">
          <Row align="middle" justify="space-between">
            <Col>
              <Title level={2} style={{ color: '#fff', margin: 0, fontWeight: 800 }}>任务中心</Title>
              <Text style={{ color: 'rgba(255,255,255,0.65)', fontSize: 14 }}>
                统一管理所有异步任务 — 训练、解释、评估、预测
              </Text>
            </Col>
            <Col>
              <Button
                icon={<ReloadOutlined />}
                ghost
                onClick={() => { fetchTasks(1); fetchStats() }}
              >
                刷新
              </Button>
            </Col>
          </Row>
        </div>

        {/* Global stat cards */}
        <Row gutter={[16, 16]}>
          {statsCards.map((item, i) => (
            <Col xs={12} sm={6} key={i}>
              <Card bodyStyle={{ padding: '16px 20px' }}>
                <Statistic
                  title={
                    <Space size={4}>
                      <span style={{ color: item.color, fontWeight: 700, fontSize: 12 }}>{item.label}</span>
                      {!globalStats && (
                        <Tooltip title="仅统计当前页">
                          <Text style={{ fontSize: 10, color: '#94a3b8' }}>(当前页)</Text>
                        </Tooltip>
                      )}
                    </Space>
                  }
                  value={item.value}
                  valueStyle={{ color: item.color, fontSize: 24, fontWeight: 800 }}
                />
              </Card>
            </Col>
          ))}
        </Row>

        {/* Task table */}
        <Card
          title="任务列表"
          extra={
            <Space>
              <Select
                allowClear
                placeholder="任务类型"
                style={{ width: 120 }}
                options={Object.entries(KIND_LABELS).map(([k, v]) => ({ value: k, label: v }))}
                onChange={v => { setFilterKind(v) }}
                value={filterKind}
              />
              <Select
                allowClear
                placeholder="状态筛选"
                style={{ width: 120 }}
                options={Object.entries(STATUS_CONFIG).map(([k, v]) => ({ value: k, label: v.label }))}
                onChange={v => { setFilterStatus(v) }}
                value={filterStatus}
              />
            </Space>
          }
        >
          {tasks.length === 0 && !loading
            ? <Empty description="暂无任务记录" />
            : (
              <Table
                rowKey="id"
                dataSource={tasks}
                columns={columns}
                loading={loading}
                size="small"
                expandable={{
                  expandedRowRender,
                  rowExpandable,
                  expandRowByClick: false,
                }}
                pagination={{
                  total,
                  current: page,
                  pageSize: PAGE_SIZE,
                  onChange: fetchTasks,
                  showTotal: t => `共 ${t} 条`,
                  showSizeChanger: false,
                }}
              />
            )
          }
        </Card>

      </Space>
    </div>
  )
}

export default TaskCenter
