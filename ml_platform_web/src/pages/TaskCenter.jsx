/**
 * V3 Task Center — unified view of all PlatformTasks across every kind.
 * Replaces the per-module monitor pages with a single, filterable view.
 */

import React, { useEffect, useState, useCallback } from 'react'
import {
  Button, Card, Col, Empty, message, Popconfirm,
  Row, Select, Space, Statistic, Table, Tag, Typography,
} from 'antd'
import {
  ReloadOutlined, CloseCircleOutlined, RedoOutlined,
  CheckCircleOutlined, SyncOutlined, ClockCircleOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons'
import { platformTasksApi } from '../services/api'

const { Title, Text } = Typography

// ── Status config ─────────────────────────────────────────────────────────
const STATUS_CONFIG = {
  SUCCESS:   { color: '#10b981', bg: 'rgba(16,185,129,0.10)', icon: <CheckCircleOutlined />,    label: '成功' },
  RUNNING:   { color: '#3b82f6', bg: 'rgba(59,130,246,0.10)', icon: <SyncOutlined spin />,      label: '运行中' },
  QUEUED:    { color: '#6366f1', bg: 'rgba(99,102,241,0.10)', icon: <ClockCircleOutlined />,    label: '已排队' },
  PENDING:   { color: '#f59e0b', bg: 'rgba(245,158,11,0.10)', icon: <ClockCircleOutlined />,    label: '等待中' },
  FAILED:    { color: '#ef4444', bg: 'rgba(239,68,68,0.10)',  icon: <ExclamationCircleOutlined />, label: '失败' },
  RETRY:     { color: '#f97316', bg: 'rgba(249,115,22,0.10)', icon: <RedoOutlined />,           label: '重试' },
  CANCELLED: { color: '#94a3b8', bg: 'rgba(148,163,184,0.10)', icon: <CloseCircleOutlined />,  label: '已取消' },
}

function StatusBadge({ status }) {
  const cfg = STATUS_CONFIG[status?.toUpperCase()] ?? { color: '#94a3b8', bg: 'rgba(148,163,184,0.1)', icon: null, label: status }
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

const KIND_LABELS = {
  train: '训练', dl_train: 'DL训练', explain: 'SHAP解释',
  eval: '评估', predict: '预测', preprocess: '预处理', automl: 'AutoML',
}

// ── Main component ─────────────────────────────────────────────────────────
const TaskCenter = () => {
  const [tasks, setTasks] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(false)
  const [filterKind, setFilterKind] = useState(undefined)
  const [filterStatus, setFilterStatus] = useState(undefined)

  const PAGE_SIZE = 20

  const fetchTasks = useCallback(async (p = page) => {
    setLoading(true)
    try {
      const res = await platformTasksApi.list({
        page: p, page_size: PAGE_SIZE,
        ...(filterKind ? { kind: filterKind } : {}),
        ...(filterStatus ? { status: filterStatus } : {}),
      })
      setTasks((res.items || []).map((t, i) => ({ ...t, key: t.id })))
      setTotal(res.total || 0)
      setPage(p)
    } catch (err) {
      message.error('加载任务列表失败')
    } finally {
      setLoading(false)
    }
  }, [page, filterKind, filterStatus])

  useEffect(() => { fetchTasks(1) }, [filterKind, filterStatus])

  // Auto-refresh every 5s if any task is running/queued
  useEffect(() => {
    const hasActive = tasks.some(t => ['RUNNING', 'QUEUED', 'PENDING', 'RETRY'].includes(t.status?.toUpperCase()))
    if (!hasActive) return
    const timer = setInterval(() => fetchTasks(page), 5000)
    return () => clearInterval(timer)
  }, [tasks, page, fetchTasks])

  const handleRetry = async (id) => {
    try {
      await platformTasksApi.retry(id)
      message.success('任务已重新提交')
      fetchTasks(page)
    } catch { message.error('重试失败') }
  }

  const handleCancel = async (id) => {
    try {
      await platformTasksApi.cancel(id)
      message.success('任务已取消')
      fetchTasks(page)
    } catch { message.error('取消失败') }
  }

  const handleDelete = async (id) => {
    try {
      await platformTasksApi.delete(id)
      message.success('已删除')
      fetchTasks(page)
    } catch { message.error('删除失败') }
  }

  // Stat counts
  const counts = tasks.reduce((acc, t) => {
    const s = t.status?.toUpperCase()
    acc[s] = (acc[s] || 0) + 1
    return acc
  }, {})

  const columns = [
    {
      title: '任务 ID',
      dataIndex: 'id',
      width: 120,
      render: v => <Text code style={{ fontSize: 11 }}>{v.slice(0, 8)}…</Text>,
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
      title: '进度',
      dataIndex: 'progress',
      width: 80,
      render: v => <Text style={{ fontSize: 12, color: '#64748b' }}>{((v || 0) * 100).toFixed(0)}%</Text>,
    },
    {
      title: '重试',
      render: (_, r) => <Text style={{ fontSize: 12, color: '#64748b' }}>{r.retry_count ?? 0}/{r.max_retries ?? 3}</Text>,
      width: 60,
    },
    {
      title: '关联任务',
      dataIndex: 'payload_ref',
      ellipsis: true,
      render: v => v ? <Text style={{ fontSize: 11, color: '#94a3b8' }}>{v}</Text> : '—',
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      width: 150,
      render: v => v ? <Text style={{ fontSize: 11, color: '#94a3b8' }}>{new Date(v).toLocaleString('zh-CN', { hour12: false })}</Text> : '—',
    },
    {
      title: '操作',
      width: 140,
      render: (_, r) => (
        <Space size={4}>
          {['FAILED', 'RETRY'].includes(r.status?.toUpperCase()) && (
            <Button size="small" icon={<RedoOutlined />} onClick={() => handleRetry(r.id)}>重试</Button>
          )}
          {['PENDING', 'QUEUED'].includes(r.status?.toUpperCase()) && (
            <Popconfirm title="确认取消？" onConfirm={() => handleCancel(r.id)} okText="取消任务" cancelText="返回">
              <Button size="small" danger icon={<CloseCircleOutlined />}>取消</Button>
            </Popconfirm>
          )}
          {['SUCCESS', 'FAILED', 'CANCELLED'].includes(r.status?.toUpperCase()) && (
            <Popconfirm title="确认删除记录？" onConfirm={() => handleDelete(r.id)} okText="删除" cancelText="返回">
              <Button size="small" type="text" danger>删除</Button>
            </Popconfirm>
          )}
        </Space>
      ),
    },
  ]

  return (
    <div style={{ padding: '20px 0' }}>
      <Space direction="vertical" size={20} style={{ width: '100%', display: 'flex' }}>

        {/* Hero */}
        <div className="hero-banner">
          <Title level={2} style={{ color: '#fff', margin: 0, fontWeight: 800 }}>任务中心</Title>
          <Text style={{ color: 'rgba(255,255,255,0.65)', fontSize: 14 }}>
            统一管理所有异步任务 — 训练、解释、评估、预测
          </Text>
        </div>

        {/* Stat cards */}
        <Row gutter={[16, 16]}>
          {[
            { label: '运行中', count: counts.RUNNING || 0, color: '#3b82f6' },
            { label: '已排队', count: (counts.QUEUED || 0) + (counts.PENDING || 0), color: '#6366f1' },
            { label: '成功',   count: counts.SUCCESS || 0, color: '#10b981' },
            { label: '失败',   count: (counts.FAILED || 0) + (counts.RETRY || 0), color: '#ef4444' },
          ].map((item, i) => (
            <Col xs={12} sm={6} key={i}>
              <Card bodyStyle={{ padding: '16px 20px' }}>
                <Statistic
                  title={<span style={{ color: item.color, fontWeight: 700, fontSize: 12 }}>{item.label}</span>}
                  value={item.count}
                  valueStyle={{ color: item.color, fontSize: 24, fontWeight: 800 }}
                />
              </Card>
            </Col>
          ))}
        </Row>

        {/* Table */}
        <Card
          title="任务列表"
          extra={
            <Space>
              <Select
                allowClear
                placeholder="任务类型"
                style={{ width: 120 }}
                options={Object.entries(KIND_LABELS).map(([k, v]) => ({ value: k, label: v }))}
                onChange={setFilterKind}
              />
              <Select
                allowClear
                placeholder="状态"
                style={{ width: 110 }}
                options={Object.entries(STATUS_CONFIG).map(([k, v]) => ({ value: k, label: v.label }))}
                onChange={setFilterStatus}
              />
              <Button icon={<ReloadOutlined />} onClick={() => fetchTasks(1)}>刷新</Button>
            </Space>
          }
        >
          {tasks.length === 0 && !loading
            ? <Empty description="暂无任务" />
            : <Table
                rowKey="id"
                dataSource={tasks}
                columns={columns}
                loading={loading}
                size="small"
                pagination={{
                  total,
                  current: page,
                  pageSize: PAGE_SIZE,
                  onChange: (p) => fetchTasks(p),
                  showTotal: (t) => `共 ${t} 条`,
                  showSizeChanger: false,
                }}
              />
          }
        </Card>

      </Space>
    </div>
  )
}

export default TaskCenter
