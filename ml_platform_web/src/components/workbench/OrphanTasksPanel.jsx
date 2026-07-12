import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert, Button, Space, Tag, Tooltip, Select, Popconfirm, Table, Typography, message,
} from 'antd'
import {
  CheckCircleOutlined, ClockCircleOutlined, CloseCircleOutlined,
  ExclamationCircleOutlined, EyeOutlined, RedoOutlined, SyncOutlined,
} from '@ant-design/icons'
import { platformTasksApi } from '../../services/api'
import OrphanTaskDetailDrawer from './OrphanTaskDetailDrawer'

const { Text } = Typography

const STATUS_CONFIG = {
  SUCCESS:   { color: '#10b981', bg: 'rgba(16,185,129,0.10)',  icon: <CheckCircleOutlined />,       label: '成功' },
  COMPLETED: { color: '#10b981', bg: 'rgba(16,185,129,0.10)',  icon: <CheckCircleOutlined />,       label: '已完成' },
  RUNNING:   { color: '#3b82f6', bg: 'rgba(59,130,246,0.10)',  icon: <SyncOutlined spin />,         label: '运行中' },
  QUEUED:    { color: '#6366f1', bg: 'rgba(99,102,241,0.10)',  icon: <ClockCircleOutlined />,       label: '已排队' },
  PENDING:   { color: '#f59e0b', bg: 'rgba(245,158,11,0.10)',  icon: <ClockCircleOutlined />,       label: '等待中' },
  CREATED:   { color: '#94a3b8', bg: 'rgba(148,163,184,0.10)', icon: <ClockCircleOutlined />,       label: '已创建' },
  FAILED:    { color: '#ef4444', bg: 'rgba(239,68,68,0.10)',   icon: <ExclamationCircleOutlined />, label: '失败' },
  RETRY:     { color: '#f97316', bg: 'rgba(249,115,22,0.10)',  icon: <RedoOutlined />,              label: '重试' },
  CANCELLED: { color: '#94a3b8', bg: 'rgba(148,163,184,0.10)', icon: <CloseCircleOutlined />,       label: '已取消' },
  ARCHIVED:  { color: '#94a3b8', bg: 'rgba(148,163,184,0.10)', icon: <CloseCircleOutlined />,       label: '已归档' },
}

const KIND_LABELS = {
  train: '训练', dl_train: 'DL训练', explain: 'SHAP解释',
  eval: '评估', predict: '预测', preprocess: '预处理', automl: 'AutoML',
}

function StatusBadge({ status }) {
  const key = status?.toUpperCase()
  const cfg = STATUS_CONFIG[key] ?? { color: '#94a3b8', bg: 'rgba(148,163,184,0.1)', icon: null, label: status }
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 5, padding: '2px 10px',
      borderRadius: 99, fontSize: 11, fontWeight: 600, background: cfg.bg, color: cfg.color,
    }}>
      {cfg.icon} {cfg.label}
    </span>
  )
}

function formatDuration(startIso, endIso) {
  if (!startIso) return '—'
  const start = new Date(startIso).getTime()
  const end = endIso ? new Date(endIso).getTime() : Date.now()
  const secs = Math.round((end - start) / 1000)
  if (secs < 60) return `${secs}s`
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`
  const h = Math.floor(secs / 3600)
  const m = Math.floor((secs % 3600) / 60)
  return `${h}h ${m}m`
}

/**
 * OrphanTasksPanel — bare PlatformTasks not linked to any ModelingTask Run
 * (standalone predict / SHAP / preprocess jobs). Moved verbatim from the old
 * TaskCenter「孤立任务」tab so 运行诊断 can host it as a second tab. Owns the
 * platformTasksApi list + retry/cancel/delete/batch-retry/filter/selection and
 * the OrphanTaskDetailDrawer.
 */
export default function OrphanTasksPanel() {
  const [tasks, setTasks] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(false)
  const [filterKind, setFilterKind] = useState(undefined)
  const [filterStatus, setFilterStatus] = useState(undefined)
  const [selectedRowKeys, setSelectedRowKeys] = useState([])
  const [batchRetrying, setBatchRetrying] = useState(false)
  const [detailTaskId, setDetailTaskId] = useState(null)
  const PAGE_SIZE = 20

  const fetchTasks = useCallback(async (p = 1) => {
    setLoading(true)
    try {
      const res = await platformTasksApi.list({
        page: p, page_size: PAGE_SIZE,
        ...(filterKind ? { kind: filterKind } : {}),
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

  useEffect(() => { fetchTasks(1) }, [filterKind, filterStatus]) // eslint-disable-line

  const handleRetry = async (id) => {
    try { await platformTasksApi.retry(id); message.success('任务已重新提交'); fetchTasks(page) }
    catch (err) { message.error(err?.response?.data?.detail || '重试失败') }
  }
  const handleCancel = async (id) => {
    try { await platformTasksApi.cancel(id); message.success('任务已取消'); fetchTasks(page) }
    catch (err) { message.error(err?.response?.data?.detail || '取消失败') }
  }
  const handleDelete = async (id) => {
    try { await platformTasksApi.delete(id); message.success('已删除'); fetchTasks(page) }
    catch (err) { message.error(err?.response?.data?.detail || '删除失败') }
  }

  const handleBatchRetry = async () => {
    if (!selectedRowKeys.length) return
    setBatchRetrying(true)
    const results = await Promise.allSettled(selectedRowKeys.map(id => platformTasksApi.retry(id)))
    const ok = results.filter(r => r.status === 'fulfilled').length
    const fail = results.length - ok
    if (ok) message.success(`已重新提交 ${ok} 个任务`)
    if (fail) message.error(`${fail} 个任务重试失败`)
    setSelectedRowKeys([])
    setBatchRetrying(false)
    fetchTasks(page)
  }

  const retriableSelectedCount = useMemo(
    () => tasks.filter(t => selectedRowKeys.includes(t.id) && ['FAILED', 'RETRY'].includes(t.status?.toUpperCase())).length,
    [tasks, selectedRowKeys],
  )

  const columns = [
    { title: '任务 ID', dataIndex: 'id', width: 120,
      render: v => <Tooltip title={v}><Text code style={{ fontSize: 11 }}>{v.slice(0, 8)}…</Text></Tooltip> },
    { title: '类型', dataIndex: 'kind', width: 100,
      render: v => <Tag color="blue" style={{ fontSize: 11 }}>{KIND_LABELS[v] ?? v}</Tag> },
    { title: '状态', dataIndex: 'status', width: 110, render: v => <StatusBadge status={v} /> },
    { title: '耗时', width: 90, render: (_, r) => <Text style={{ fontSize: 12, color: '#64748b' }}>{formatDuration(r.started_at, r.finished_at)}</Text> },
    { title: '进度', dataIndex: 'progress', width: 70, render: v => <Text style={{ fontSize: 12 }}>{((v ?? 0) * 100).toFixed(0)}%</Text> },
    { title: '关联', dataIndex: 'payload_ref', ellipsis: true,
      render: v => v ? <Tooltip title={v}><Text style={{ fontSize: 11, color: '#94a3b8' }}>{v}</Text></Tooltip> : '—' },
    { title: '入队', dataIndex: 'queued_at', width: 140,
      render: v => v ? <Text style={{ fontSize: 11, color: '#94a3b8' }}>{new Date(v).toLocaleString('zh-CN', { hour12: false })}</Text> : '—' },
    { title: '操作', width: 180, render: (_, r) => {
      const s = r.status?.toUpperCase()
      return (
        <Space size={4}>
          <Button size="small" icon={<EyeOutlined />} onClick={() => setDetailTaskId(r.id)}>详情</Button>
          {(s === 'FAILED' || s === 'RETRY') && (
            <Button size="small" icon={<RedoOutlined />} onClick={() => handleRetry(r.id)}>重试</Button>
          )}
          {(s === 'PENDING' || s === 'QUEUED') && (
            <Popconfirm title="确认取消？" onConfirm={() => handleCancel(r.id)}>
              <Button size="small" danger>取消</Button>
            </Popconfirm>
          )}
          {(s === 'SUCCESS' || s === 'FAILED' || s === 'CANCELLED') && (
            <Popconfirm title="确认删除？" onConfirm={() => handleDelete(r.id)}>
              <Button size="small" type="text" danger>删除</Button>
            </Popconfirm>
          )}
        </Space>
      )
    } },
  ]

  return (
    <div>
      <Alert type="info" showIcon style={{ marginBottom: 12 }}
        message="孤立任务视图"
        description={
          <span>
            这里只显示<strong>未关联到建模任务</strong>的平台调度任务（例如独立的预测、SHAP 解释、数据预处理）。
            训练类任务请到<strong>「任务列表」</strong>查看完整的 ModelingTask → 实验批次 → Run 层级。
          </span>
        } />
      <Space style={{ marginBottom: 12 }} wrap>
        <Select allowClear placeholder="任务类型" style={{ width: 120 }} value={filterKind} onChange={setFilterKind}
          options={Object.entries(KIND_LABELS).map(([k, v]) => ({ value: k, label: v }))} />
        <Select allowClear placeholder="状态筛选" style={{ width: 120 }} value={filterStatus} onChange={setFilterStatus}
          options={Object.entries(STATUS_CONFIG).map(([k, v]) => ({ value: k, label: v.label }))} />
        <Popconfirm title={`确认重试选中的 ${retriableSelectedCount} 个任务？`} onConfirm={handleBatchRetry} disabled={retriableSelectedCount === 0}>
          <Button icon={<RedoOutlined />} type="primary" disabled={retriableSelectedCount === 0} loading={batchRetrying}>
            批量重试 {retriableSelectedCount > 0 ? `(${retriableSelectedCount})` : ''}
          </Button>
        </Popconfirm>
        {selectedRowKeys.length > 0 && (
          <Button type="text" onClick={() => setSelectedRowKeys([])}>清空选择</Button>
        )}
      </Space>
      <Table rowKey="id" dataSource={tasks} columns={columns} loading={loading} size="small"
        rowSelection={{
          selectedRowKeys, onChange: setSelectedRowKeys,
          getCheckboxProps: (r) => ({ disabled: !['FAILED', 'RETRY'].includes(r.status?.toUpperCase()) }),
        }}
        pagination={{ total, current: page, pageSize: PAGE_SIZE, onChange: fetchTasks, showTotal: t => `共 ${t} 条`, showSizeChanger: false }} />
      <OrphanTaskDetailDrawer
        taskId={detailTaskId}
        open={!!detailTaskId}
        onClose={() => setDetailTaskId(null)}
        onRetry={async (id) => { await handleRetry(id) }}
        onCancel={async (id) => { await handleCancel(id) }}
        onDelete={async (id) => { await handleDelete(id) }}
      />
    </div>
  )
}
