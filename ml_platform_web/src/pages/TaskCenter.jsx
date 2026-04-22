/**
 * V3 Task Center — hierarchical view.
 *
 *   ModelingTask (user's "job")   ── row, always visible
 *     └─ Experiment batch         ── nested, revealed on expand
 *          └─ Run + PlatformTask  ── deepest, per-trial details
 *
 * Why hierarchical?  One modeling task often fans out into multiple AutoML
 * experiments, each fanning out into dozens of parallel runs.  Flattening
 * all of them into a single table makes it impossible to understand what
 * the user actually submitted.  Here the top level is the user's intent,
 * and expansion reveals the scheduler's fan-out.
 *
 * A second "孤立任务" tab surfaces bare PlatformTasks not linked to any Run
 * (e.g. standalone forecast/predict jobs) so nothing is lost from the old
 * flat view.
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert, Button, Card, Col, Empty, message, Popconfirm, Progress,
  Row, Select, Space, Statistic, Table, Tabs, Tag, Tooltip, Typography,
} from 'antd'
import {
  CheckCircleOutlined, ClockCircleOutlined, CloseCircleOutlined,
  DatabaseOutlined, ExclamationCircleOutlined, ExperimentOutlined,
  EyeOutlined, LineChartOutlined, ReloadOutlined, RedoOutlined, SyncOutlined,
} from '@ant-design/icons'
import { Link } from 'react-router-dom'
import { platformTasksApi } from '../services/api'
import RunInspector from '../components/workbench/RunInspector'
import OrphanTaskDetailDrawer from '../components/workbench/OrphanTaskDetailDrawer'

const { Text, Title } = Typography

// ── Config ────────────────────────────────────────────────────────────────
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

const STRATEGY_LABELS = {
  baseline: '基线',
  grid_search: '网格搜索',
  bayesian_search: '贝叶斯搜索',
  automl: 'AutoML',
}

const ACTIVE_STATUSES = new Set(['RUNNING', 'QUEUED', 'PENDING', 'RETRY'])

// ── Small components ──────────────────────────────────────────────────────

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

function StatsChips({ stats }) {
  if (!stats) return null
  const chips = [
    { label: '总数', value: stats.total_runs ?? stats.total ?? 0, color: '#64748b' },
    { label: '成功', value: stats.success ?? 0, color: '#10b981' },
    { label: '失败', value: stats.failed ?? 0, color: '#ef4444' },
    { label: '运行中', value: stats.running ?? 0, color: '#3b82f6' },
    { label: '排队', value: stats.queued ?? 0, color: '#6366f1' },
  ]
  return (
    <Space size={6} wrap>
      {chips.map(c => c.value > 0 && (
        <Tag key={c.label} style={{ fontSize: 11, color: c.color, border: `1px solid ${c.color}33` }}>
          {c.label}: <b>{c.value}</b>
        </Tag>
      ))}
    </Space>
  )
}

// ── Runs table (deepest level) ────────────────────────────────────────────

function RunsTable({ runs, objectiveMetric, onInspect }) {
  const columns = [
    { title: '#', dataIndex: 'trial_no', width: 50,
      render: (v, r) => v ?? r.rank ?? '—' },
    { title: 'Run ID', dataIndex: 'id', width: 110,
      render: (v) => <Text code style={{ fontSize: 11 }}>{v.slice(0, 8)}…</Text> },
    { title: '模型', width: 120, render: (_, r) =>
      <Text style={{ fontSize: 12 }}>{r.params?.model_type || '—'}</Text> },
    { title: '状态', dataIndex: 'status', width: 100,
      render: (v) => <StatusBadge status={v} /> },
    {
      title: objectiveMetric ? `目标(${objectiveMetric})` : '目标指标',
      width: 130,
      render: (_, r) => {
        const v = objectiveMetric && r.metrics?.[objectiveMetric]
        return (
          <Text code style={{
            color: typeof v === 'number' ? '#2563eb' : '#94a3b8',
            fontSize: 12,
          }}>
            {typeof v === 'number' ? v.toFixed(4) : '—'}
          </Text>
        )
      },
    },
    { title: '进度', width: 90, render: (_, r) => {
      const p = r.task?.progress ?? 0
      const active = ACTIVE_STATUSES.has((r.task?.status || r.status || '').toUpperCase())
      if (!active && (r.status === 'SUCCESS' || r.status === 'FAILED')) {
        return <Text style={{ fontSize: 11, color: '#94a3b8' }}>{(p * 100).toFixed(0)}%</Text>
      }
      return <Progress percent={(p * 100).toFixed(0)} size="small" showInfo={false}
        strokeColor={r.status === 'FAILED' ? '#ef4444' : '#3b82f6'} />
    } },
    { title: '耗时', width: 80, render: (_, r) =>
      <Text style={{ fontSize: 11, color: '#64748b' }}>
        {formatDuration(r.started_at, r.finished_at)}
      </Text> },
    { title: '重试', width: 60, render: (_, r) =>
      <Text style={{ fontSize: 11, color: r.task?.retry_count ? '#f97316' : '#94a3b8' }}>
        {r.task?.retry_count ?? 0}/{r.task?.max_retries ?? 3}
      </Text> },
    { title: '', width: 70, render: (_, r) => (
      <Button size="small" type="link" onClick={() => onInspect(r.id)}>
        详情
      </Button>
    ) },
  ]

  return (
    <Table
      size="small"
      rowKey="id"
      columns={columns}
      dataSource={runs}
      pagination={false}
      style={{ marginLeft: 24 }}
    />
  )
}

// ── Experiment row (middle level) ─────────────────────────────────────────

function ExperimentBlock({ experiment, objectiveMetric, onInspect }) {
  return (
    <Card
      size="small"
      style={{ marginBottom: 8, background: '#fafbfc', border: '1px solid #e2e8f0' }}
      title={
        <Space size={8}>
          <ExperimentOutlined style={{ color: '#8b5cf6' }} />
          <Text strong>{experiment.name}</Text>
          <Tag color="purple" style={{ fontSize: 11 }}>
            {STRATEGY_LABELS[experiment.strategy_type] || experiment.strategy_type}
          </Tag>
          <StatusBadge status={experiment.status} />
          {experiment.selected_models?.length > 0 && (
            <Text type="secondary" style={{ fontSize: 11 }}>
              模型: {experiment.selected_models.join(', ')}
            </Text>
          )}
        </Space>
      }
      extra={<StatsChips stats={experiment.stats} />}
      bodyStyle={{ padding: '8px 0' }}
    >
      {experiment.runs?.length > 0 ? (
        <RunsTable
          runs={experiment.runs}
          objectiveMetric={objectiveMetric}
          onInspect={onInspect}
        />
      ) : (
        <Empty
          description="暂无 Run"
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          style={{ padding: 16 }}
        />
      )}
    </Card>
  )
}

// ── Hierarchical tab ──────────────────────────────────────────────────────

function HierarchicalView({ onInspect, filterStatus, setFilterStatus }) {
  const [tree, setTree] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(false)
  const PAGE_SIZE = 10

  const fetchTree = useCallback(async (p = 1) => {
    setLoading(true)
    try {
      const res = await platformTasksApi.tree({
        page: p, page_size: PAGE_SIZE,
        ...(filterStatus ? { status: filterStatus } : {}),
      })
      setTree(res.items || [])
      setTotal(res.total || 0)
      setPage(p)
    } catch {
      message.error('加载任务树失败')
    } finally {
      setLoading(false)
    }
  }, [filterStatus])

  useEffect(() => { fetchTree(1) }, [filterStatus]) // eslint-disable-line

  // Auto-refresh while any task is active
  useEffect(() => {
    const hasActive = tree.some(mt =>
      ACTIVE_STATUSES.has(mt.status?.toUpperCase()) ||
      (mt.stats?.running ?? 0) > 0 ||
      (mt.stats?.queued ?? 0) > 0,
    )
    if (!hasActive) return undefined
    const timer = setInterval(() => fetchTree(page), 5000)
    return () => clearInterval(timer)
  }, [tree, page, fetchTree])

  const columns = [
    {
      title: '建模任务',
      dataIndex: 'name',
      render: (v, r) => (
        <Space direction="vertical" size={0}>
          <Link
            to={`/v3/tasks/${r.id}`}
            style={{ fontWeight: 600, color: '#0f172a', fontSize: 14 }}
          >
            {v}
          </Link>
          {r.description && (
            <Text type="secondary" style={{ fontSize: 11 }}>{r.description}</Text>
          )}
          <Space size={4}>
            {r.dataset_name && (
              <Tag icon={<DatabaseOutlined />} style={{ fontSize: 11 }}>
                {r.dataset_name}
              </Tag>
            )}
            {r.task_type && (
              <Tag color="blue" style={{ fontSize: 11 }}>{r.task_type}</Tag>
            )}
          </Space>
        </Space>
      ),
    },
    { title: '状态', dataIndex: 'status', width: 100,
      render: v => <StatusBadge status={v} /> },
    { title: '目标指标', width: 140, render: (_, r) =>
      <Text code style={{ fontSize: 11 }}>
        {r.objective_metric} ({r.objective_direction})
      </Text> },
    { title: '统计', width: 260, render: (_, r) => <StatsChips stats={r.stats} /> },
    { title: '实验数', width: 70, align: 'center',
      render: (_, r) => <Tag>{r.experiments?.length || 0}</Tag> },
    { title: '创建时间', dataIndex: 'created_at', width: 150,
      render: v => v
        ? <Text style={{ fontSize: 11, color: '#94a3b8' }}>
            {new Date(v).toLocaleString('zh-CN', { hour12: false })}
          </Text>
        : '—' },
    {
      title: '操作', key: 'actions', width: 110, fixed: 'right',
      render: (_, r) => (
        <Link to={`/v3/tasks/${r.id}`}>
          <Button size="small" type="link" icon={<EyeOutlined />}>查看详情</Button>
        </Link>
      ),
    },
  ]

  const expandedRowRender = (record) => {
    if (!record.experiments?.length) {
      return <Empty description="该建模任务还没有提交实验" image={Empty.PRESENTED_IMAGE_SIMPLE} />
    }
    return (
      <div style={{ padding: '8px 16px', background: '#f8fafc' }}>
        {record.experiments.map(exp => (
          <ExperimentBlock
            key={exp.id}
            experiment={exp}
            objectiveMetric={record.objective_metric}
            onInspect={onInspect}
          />
        ))}
      </div>
    )
  }

  return (
    <Table
      rowKey="id"
      dataSource={tree}
      columns={columns}
      loading={loading}
      size="middle"
      expandable={{
        expandedRowRender,
        rowExpandable: (r) => (r.experiments?.length || 0) > 0,
      }}
      pagination={{
        total,
        current: page,
        pageSize: PAGE_SIZE,
        onChange: fetchTree,
        showTotal: t => `共 ${t} 个建模任务`,
        showSizeChanger: false,
      }}
      locale={{ emptyText: <Empty description="暂无建模任务" /> }}
    />
  )
}

// ── Flat view — bare PlatformTasks (legacy, for orphan kinds) ─────────────

function FlatView() {
  const [tasks, setTasks] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage]   = useState(1)
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
    try {
      await platformTasksApi.retry(id)
      message.success('任务已重新提交')
      fetchTasks(page)
    } catch (err) {
      message.error(err?.response?.data?.detail || '重试失败')
    }
  }

  const handleCancel = async (id) => {
    try {
      await platformTasksApi.cancel(id)
      message.success('任务已取消')
      fetchTasks(page)
    } catch (err) {
      message.error(err?.response?.data?.detail || '取消失败')
    }
  }

  const handleDelete = async (id) => {
    try {
      await platformTasksApi.delete(id)
      message.success('已删除')
      fetchTasks(page)
    } catch (err) {
      message.error(err?.response?.data?.detail || '删除失败')
    }
  }

  // Batch retry — re-submit every selected FAILED / RETRY task in parallel.
  // Rows in other statuses are disabled via getCheckboxProps so selection is safe.
  const handleBatchRetry = async () => {
    if (!selectedRowKeys.length) return
    setBatchRetrying(true)
    const results = await Promise.allSettled(
      selectedRowKeys.map(id => platformTasksApi.retry(id)),
    )
    const ok   = results.filter(r => r.status === 'fulfilled').length
    const fail = results.length - ok
    if (ok)   message.success(`已重新提交 ${ok} 个任务`)
    if (fail) message.error(`${fail} 个任务重试失败`)
    setSelectedRowKeys([])
    setBatchRetrying(false)
    fetchTasks(page)
  }

  const retriableSelectedCount = useMemo(
    () => tasks.filter(t =>
      selectedRowKeys.includes(t.id) &&
      ['FAILED', 'RETRY'].includes(t.status?.toUpperCase()),
    ).length,
    [tasks, selectedRowKeys],
  )

  const columns = [
    { title: '任务 ID', dataIndex: 'id', width: 120,
      render: v => (
        <Tooltip title={v}>
          <Text code style={{ fontSize: 11 }}>{v.slice(0, 8)}…</Text>
        </Tooltip>
      ) },
    { title: '类型', dataIndex: 'kind', width: 100,
      render: v => <Tag color="blue" style={{ fontSize: 11 }}>{KIND_LABELS[v] ?? v}</Tag> },
    { title: '状态', dataIndex: 'status', width: 110,
      render: v => <StatusBadge status={v} /> },
    { title: '耗时', width: 90, render: (_, r) =>
      <Text style={{ fontSize: 12, color: '#64748b' }}>
        {formatDuration(r.started_at, r.finished_at)}
      </Text> },
    { title: '进度', dataIndex: 'progress', width: 70,
      render: v => <Text style={{ fontSize: 12 }}>{((v ?? 0) * 100).toFixed(0)}%</Text> },
    { title: '关联', dataIndex: 'payload_ref', ellipsis: true,
      render: v => v
        ? <Tooltip title={v}>
            <Text style={{ fontSize: 11, color: '#94a3b8' }}>{v}</Text>
          </Tooltip>
        : '—' },
    { title: '入队', dataIndex: 'queued_at', width: 140,
      render: v => v
        ? <Text style={{ fontSize: 11, color: '#94a3b8' }}>
            {new Date(v).toLocaleString('zh-CN', { hour12: false })}
          </Text>
        : '—' },
    { title: '操作', width: 180, render: (_, r) => {
      const s = r.status?.toUpperCase()
      return (
        <Space size={4}>
          <Button
            size="small"
            icon={<EyeOutlined />}
            onClick={() => setDetailTaskId(r.id)}
          >详情</Button>
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
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 12 }}
        message="孤立任务视图"
        description={
          <span>
            这里只显示<strong>未关联到建模任务</strong>的平台调度任务（例如独立的预测、SHAP 解释、数据预处理）。
            训练类任务请到<strong>「建模任务视图」</strong>查看完整的 ModelingTask → 实验批次 → Run 层级。
          </span>
        }
      />
      <Space style={{ marginBottom: 12 }} wrap>
        <Select allowClear placeholder="任务类型" style={{ width: 120 }}
          value={filterKind} onChange={setFilterKind}
          options={Object.entries(KIND_LABELS).map(([k, v]) => ({ value: k, label: v }))} />
        <Select allowClear placeholder="状态筛选" style={{ width: 120 }}
          value={filterStatus} onChange={setFilterStatus}
          options={Object.entries(STATUS_CONFIG).map(([k, v]) => ({ value: k, label: v.label }))} />
        <Popconfirm
          title={`确认重试选中的 ${retriableSelectedCount} 个任务？`}
          onConfirm={handleBatchRetry}
          disabled={retriableSelectedCount === 0}
        >
          <Button
            icon={<RedoOutlined />}
            type="primary"
            disabled={retriableSelectedCount === 0}
            loading={batchRetrying}
          >
            批量重试 {retriableSelectedCount > 0 ? `(${retriableSelectedCount})` : ''}
          </Button>
        </Popconfirm>
        {selectedRowKeys.length > 0 && (
          <Button type="text" onClick={() => setSelectedRowKeys([])}>
            清空选择
          </Button>
        )}
      </Space>
      <Table
        rowKey="id"
        dataSource={tasks}
        columns={columns}
        loading={loading}
        size="small"
        rowSelection={{
          selectedRowKeys,
          onChange: setSelectedRowKeys,
          getCheckboxProps: (r) => ({
            disabled: !['FAILED', 'RETRY'].includes(r.status?.toUpperCase()),
          }),
        }}
        pagination={{
          total, current: page, pageSize: PAGE_SIZE,
          onChange: fetchTasks,
          showTotal: t => `共 ${t} 条`, showSizeChanger: false,
        }}
      />
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

// ── Page ───────────────────────────────────────────────────────────────────

const TaskCenter = () => {
  const [globalStats, setGlobalStats] = useState(null)
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const [inspectorRunId, setInspectorRunId] = useState(null)
  const [treeFilterStatus, setTreeFilterStatus] = useState(undefined)
  const [activeTab, setActiveTab] = useState('tree')

  const fetchStats = useCallback(async () => {
    try { setGlobalStats(await platformTasksApi.stats()) } catch { /* non-critical */ }
  }, [])

  useEffect(() => { fetchStats() }, [fetchStats])
  useEffect(() => {
    const timer = setInterval(fetchStats, 10000)
    return () => clearInterval(timer)
  }, [fetchStats])

  const statsCards = useMemo(() => [
    { label: '运行中', value: globalStats?.running ?? 0,   color: '#3b82f6' },
    { label: '已排队', value: globalStats?.queued ?? 0,    color: '#6366f1' },
    { label: '成功',   value: globalStats?.succeeded ?? 0, color: '#10b981' },
    { label: '失败',   value: globalStats?.failed ?? 0,    color: '#ef4444' },
  ], [globalStats])

  const openInspector = (runId) => {
    setInspectorRunId(runId)
    setInspectorOpen(true)
  }

  return (
    <div style={{ padding: '20px 0' }}>
      <Space direction="vertical" size={20} style={{ width: '100%', display: 'flex' }}>
        <div className="hero-banner">
          <Row align="middle" justify="space-between">
            <Col>
              <Title level={2} style={{ color: '#fff', margin: 0, fontWeight: 800 }}>
                任务中心
              </Title>
              <Text style={{ color: 'rgba(255,255,255,0.65)', fontSize: 14 }}>
                按建模任务组织 — 展开查看实验批次与每个 Run 的调度状态
              </Text>
            </Col>
            <Col>
              <Button icon={<ReloadOutlined />} ghost onClick={fetchStats}>
                刷新统计
              </Button>
            </Col>
          </Row>
        </div>

        <Row gutter={[16, 16]}>
          {statsCards.map((item, i) => (
            <Col xs={12} sm={6} key={i}>
              <Card bodyStyle={{ padding: '16px 20px' }}>
                <Statistic
                  title={<span style={{ color: item.color, fontWeight: 700, fontSize: 12 }}>
                    {item.label}
                  </span>}
                  value={item.value}
                  valueStyle={{ color: item.color, fontSize: 24, fontWeight: 800 }}
                />
              </Card>
            </Col>
          ))}
        </Row>

        <Card>
          <Tabs
            activeKey={activeTab}
            onChange={setActiveTab}
            items={[
              {
                key: 'tree',
                label: <span><LineChartOutlined /> 建模任务视图</span>,
                children: (
                  <>
                    <Space style={{ marginBottom: 12 }}>
                      <Select
                        allowClear
                        placeholder="按状态筛选"
                        style={{ width: 140 }}
                        value={treeFilterStatus}
                        onChange={setTreeFilterStatus}
                        options={['CREATED', 'RUNNING', 'COMPLETED', 'FAILED', 'ARCHIVED']
                          .map(k => ({ value: k, label: STATUS_CONFIG[k]?.label || k }))}
                      />
                    </Space>
                    <HierarchicalView
                      onInspect={openInspector}
                      filterStatus={treeFilterStatus}
                      setFilterStatus={setTreeFilterStatus}
                    />
                  </>
                ),
              },
              {
                key: 'flat',
                label: <span><ExperimentOutlined /> 孤立任务</span>,
                children: <FlatView />,
              },
            ]}
          />
        </Card>
      </Space>

      <RunInspector
        open={inspectorOpen}
        runId={inspectorRunId}
        onClose={() => setInspectorOpen(false)}
      />
    </div>
  )
}

export default TaskCenter
