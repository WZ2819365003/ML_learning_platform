import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Card, Table, Tag, Space, Button, Input, Select, Tooltip, Empty, message,
  Typography,
} from 'antd'
import {
  ReloadOutlined, EyeOutlined, ExperimentOutlined, BranchesOutlined,
  CheckCircleFilled, CloseCircleFilled, ClockCircleFilled, TrophyOutlined,
  BulbOutlined, SafetyOutlined,
} from '@ant-design/icons'
import { v3RunsApi } from '../services/api'
import RunInspector from '../components/workbench/RunInspector'

const { Title, Text } = Typography

// ── Status metadata (mirrors ModelingTasks.jsx for visual consistency) ───────
const STATUS_META = {
  SUCCESS:  { color: 'success',    icon: <CheckCircleFilled />,       label: '成功' },
  FAILED:   { color: 'error',      icon: <CloseCircleFilled />,       label: '失败' },
  RUNNING:  { color: 'processing', icon: <ClockCircleFilled spin />,  label: '运行中' },
  PENDING:  { color: 'default',    icon: <ClockCircleFilled />,       label: '等待中' },
  QUEUED:   { color: 'default',    icon: <ClockCircleFilled />,       label: '排队中' },
  CANCELED: { color: 'warning',    icon: null,                        label: '已取消' },
}

const STRATEGY_COLOR = {
  baseline:        'blue',
  grid_search:     'purple',
  bayesian_search: 'magenta',
}

const STATUS_OPTIONS = [
  { value: 'SUCCESS', label: '成功' },
  { value: 'FAILED',  label: '失败' },
  { value: 'RUNNING', label: '运行中' },
  { value: 'PENDING', label: '等待中' },
]

const STRATEGY_OPTIONS = [
  { value: 'baseline',        label: 'Baseline' },
  { value: 'grid_search',     label: 'Grid Search' },
  { value: 'bayesian_search', label: 'Bayesian Search' },
]

const TASK_TYPE_OPTIONS = [
  { value: 'classification', label: '分类' },
  { value: 'regression',     label: '回归' },
]

/**
 * V3 Run 诊断中心 — flat cross-task Run listing.
 *
 * Why this page exists: the pre-existing workflow forced users to drill
 * through ModelingTask → detail → 实验编排 → RunInspector just to reach
 * SHAP / 自动诊断 for a specific Run.  This page skips the two middle
 * hops: it lists every ExperimentRun across every ModelingTask in one
 * filterable table and opens the RunInspector Drawer directly.
 *
 * The Drawer is intentional — closing it returns the user to the same
 * filtered table state, which is the right 「返回」 semantic.
 */
export default function V3Runs() {
  const [loading, setLoading] = useState(false)
  const [items, setItems] = useState([])
  const [total, setTotal] = useState(0)
  const [error, setError] = useState(null)

  // Server-side filters → v3RunsApi query params
  const [filterStatus, setFilterStatus] = useState(undefined)
  const [filterStrategy, setFilterStrategy] = useState(undefined)
  const [filterTaskType, setFilterTaskType] = useState(undefined)

  // Client-side filters (cheap to do in-memory)
  const [searchText, setSearchText] = useState('')
  const [filterModel, setFilterModel] = useState(undefined)

  // RunInspector drawer state
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [activeRunId, setActiveRunId] = useState(null)
  const [drawerDefaultTab, setDrawerDefaultTab] = useState('overview')

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const resp = await v3RunsApi.list({
        status:        filterStatus,
        strategy_type: filterStrategy,
        task_type:     filterTaskType,
        limit:         500,
      })
      setItems(resp?.items || [])
      setTotal(resp?.total || 0)
    } catch (err) {
      const detail = err?.response?.data?.detail || err.message || '加载 Run 列表失败'
      setError(detail)
      message.error(detail)
    } finally {
      setLoading(false)
    }
  }, [filterStatus, filterStrategy, filterTaskType])

  useEffect(() => { load() }, [load])

  // Distinct model types from current result set, for the client-side filter.
  const modelOptions = useMemo(() => {
    const set = new Set()
    items.forEach((r) => r.model_type && set.add(r.model_type))
    return Array.from(set).sort().map((m) => ({ value: m, label: m }))
  }, [items])

  // Client-side filtering: search covers task_name / experiment_name / run_id tail
  const filteredItems = useMemo(() => {
    const q = searchText.trim().toLowerCase()
    return items.filter((r) => {
      if (filterModel && r.model_type !== filterModel) return false
      if (!q) return true
      const idTail = String(r.run_id || '').slice(0, 8).toLowerCase()
      return (
        String(r.task_name       || '').toLowerCase().includes(q) ||
        String(r.experiment_name || '').toLowerCase().includes(q) ||
        idTail.includes(q)
      )
    })
  }, [items, filterModel, searchText])

  const openInspector = (runId, defaultTab) => {
    setActiveRunId(runId)
    setDrawerDefaultTab(defaultTab)
    setDrawerOpen(true)
  }

  // ── Table columns ─────────────────────────────────────────────────────────
  const columns = [
    {
      title: 'Run',
      dataIndex: 'run_id',
      width: 130,
      fixed: 'left',
      render: (id, row) => (
        <Space size={4} direction="vertical" style={{ lineHeight: 1.25 }}>
          <Text code style={{ fontSize: 11 }}>{String(id).slice(0, 8)}</Text>
          {row.rank === 1 && <Tag color="gold" style={{ margin: 0 }}><TrophyOutlined /> Top-1</Tag>}
        </Space>
      ),
    },
    {
      title: '建模任务',
      dataIndex: 'task_name',
      width: 220,
      ellipsis: true,
      render: (v, row) => (
        <Tooltip title={v}>
          <Space direction="vertical" size={2} style={{ lineHeight: 1.25 }}>
            <Text strong style={{ fontSize: 13 }}>{v || '-'}</Text>
            <Text type="secondary" style={{ fontSize: 11 }}>
              {row.task_type === 'regression' ? '回归' : '分类'}
            </Text>
          </Space>
        </Tooltip>
      ),
      sorter: (a, b) => String(a.task_name || '').localeCompare(String(b.task_name || '')),
    },
    {
      title: '实验',
      dataIndex: 'experiment_name',
      width: 180,
      ellipsis: true,
      render: (v, row) => (
        <Space direction="vertical" size={2} style={{ lineHeight: 1.25 }}>
          <Text style={{ fontSize: 12 }}>{v || '-'}</Text>
          <Tag color={STRATEGY_COLOR[row.strategy_type] || 'default'} style={{ margin: 0, fontSize: 10 }}>
            {row.strategy_type || '-'}
          </Tag>
        </Space>
      ),
      filters: STRATEGY_OPTIONS.map((s) => ({ text: s.label, value: s.value })),
      onFilter: (val, r) => r.strategy_type === val,
    },
    {
      title: '模型',
      dataIndex: 'model_type',
      width: 120,
      render: (v) => v
        ? <Tag style={{ margin: 0, fontFamily: 'monospace', fontSize: 11 }}>{v}</Tag>
        : <Text type="secondary" style={{ fontSize: 11 }}>-</Text>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      width: 100,
      render: (s) => {
        const meta = STATUS_META[s] || { color: 'default', label: s }
        return <Tag icon={meta.icon} color={meta.color}>{meta.label}</Tag>
      },
    },
    {
      title: (
        <Tooltip title="每行展示该 Run 所属建模任务的优化指标值；不同任务的指标可能不同">
          <span>目标值 <SafetyOutlined style={{ color: '#94a3b8', fontSize: 11 }} /></span>
        </Tooltip>
      ),
      dataIndex: 'objective_value',
      width: 140,
      align: 'right',
      render: (val, row) => {
        if (val == null) return <Text type="secondary" style={{ fontSize: 11 }}>-</Text>
        const num = typeof val === 'number' ? val : Number(val)
        return (
          <Space direction="vertical" size={0} style={{ lineHeight: 1.2, alignItems: 'flex-end' }}>
            <code style={{ fontSize: 13, fontWeight: 600, color: '#2563eb' }}>
              {Number.isFinite(num) ? num.toFixed(4) : String(val)}
            </code>
            <Text type="secondary" style={{ fontSize: 10 }}>
              {row.objective_metric}（{row.objective_direction === 'min' ? '↓ 越低越好' : '↑ 越高越好'}）
            </Text>
          </Space>
        )
      },
      sorter: (a, b) => {
        // Direction-aware sort: if 'min', smaller is better (ascending good); else descending good.
        const dir = a.objective_direction === 'min' ? 1 : -1
        const av = a.objective_value ?? (dir > 0 ? Infinity : -Infinity)
        const bv = b.objective_value ?? (dir > 0 ? Infinity : -Infinity)
        return dir * (av - bv)
      },
    },
    {
      title: 'Trial',
      dataIndex: 'trial_no',
      width: 75,
      align: 'center',
      render: (v, row) => (
        <Space direction="vertical" size={0} style={{ lineHeight: 1.2 }}>
          <Text style={{ fontSize: 12 }}>#{v ?? '?'}</Text>
          {row.rank != null && (
            <Text type="secondary" style={{ fontSize: 10 }}>rank {row.rank}</Text>
          )}
        </Space>
      ),
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      width: 150,
      render: (v) => v
        ? <Text style={{ fontSize: 11, fontFamily: 'monospace' }}>
            {new Date(v).toLocaleString('zh-CN', { hour12: false })}
          </Text>
        : <Text type="secondary">-</Text>,
      sorter: (a, b) => new Date(a.created_at || 0) - new Date(b.created_at || 0),
      defaultSortOrder: 'descend',
    },
    {
      title: '操作',
      key: 'actions',
      width: 190,
      fixed: 'right',
      render: (_, row) => (
        <Space size={4}>
          <Tooltip title="查看解释 (SHAP 特征重要性)">
            <Button
              size="small"
              type="primary"
              ghost
              icon={<BulbOutlined />}
              onClick={() => openInspector(row.run_id, 'shap')}
            >
              解释
            </Button>
          </Tooltip>
          <Tooltip title="打开 Run 诊断 (自动过拟合/失败归因)">
            <Button
              size="small"
              icon={<SafetyOutlined />}
              onClick={() => openInspector(row.run_id, 'context')}
            >
              诊断
            </Button>
          </Tooltip>
          <Tooltip title="打开 Run 详情概览">
            <Button
              size="small"
              type="text"
              icon={<EyeOutlined />}
              onClick={() => openInspector(row.run_id, 'overview')}
            />
          </Tooltip>
        </Space>
      ),
    },
  ]

  // ── Header strip stat chips ────────────────────────────────────────────────
  const statsRow = useMemo(() => {
    const byStatus = { SUCCESS: 0, FAILED: 0, RUNNING: 0 }
    filteredItems.forEach((r) => {
      if (byStatus[r.status] != null) byStatus[r.status]++
    })
    return byStatus
  }, [filteredItems])

  return (
    <div style={{ padding: '20px 0' }}>
      <Card
        bordered={false}
        bodyStyle={{ padding: 20 }}
        style={{ borderRadius: 12, boxShadow: '0 1px 3px rgba(15, 23, 42, 0.06)' }}
      >
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          marginBottom: 16, flexWrap: 'wrap', gap: 12,
        }}>
          <Space align="center">
            <div style={{
              width: 40, height: 40, borderRadius: 10,
              background: 'rgba(37, 99, 235, 0.1)', color: '#2563eb',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 20,
            }}>
              <BranchesOutlined />
            </div>
            <div>
              <Title level={4} style={{ margin: 0, color: '#0f172a' }}>Run 诊断中心</Title>
              <Text type="secondary" style={{ fontSize: 12 }}>
                跨所有建模任务的 Run 平铺列表 · 一键打开 SHAP 解释 / 自动诊断
              </Text>
            </div>
          </Space>
          <Space>
            <Tag color="success" style={{ fontSize: 12, padding: '2px 10px' }}>
              成功 {statsRow.SUCCESS}
            </Tag>
            <Tag color="error" style={{ fontSize: 12, padding: '2px 10px' }}>
              失败 {statsRow.FAILED}
            </Tag>
            <Tag color="processing" style={{ fontSize: 12, padding: '2px 10px' }}>
              运行中 {statsRow.RUNNING}
            </Tag>
            <Text type="secondary" style={{ fontSize: 11 }}>
              · 共 {filteredItems.length}/{total}
            </Text>
          </Space>
        </div>

        {/* Filters */}
        <Space wrap style={{ marginBottom: 14 }}>
          <Input.Search
            placeholder="搜索任务名 / 实验名 / Run ID"
            allowClear
            style={{ width: 260 }}
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            onSearch={setSearchText}
          />
          <Select
            placeholder="任务类型"
            allowClear
            style={{ width: 130 }}
            value={filterTaskType}
            options={TASK_TYPE_OPTIONS}
            onChange={setFilterTaskType}
          />
          <Select
            placeholder="策略"
            allowClear
            style={{ width: 170 }}
            value={filterStrategy}
            options={STRATEGY_OPTIONS}
            onChange={setFilterStrategy}
          />
          <Select
            placeholder="状态"
            allowClear
            style={{ width: 130 }}
            value={filterStatus}
            options={STATUS_OPTIONS}
            onChange={setFilterStatus}
          />
          <Select
            placeholder="模型"
            allowClear
            style={{ width: 150 }}
            value={filterModel}
            options={modelOptions}
            onChange={setFilterModel}
            disabled={modelOptions.length === 0}
          />
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>刷新</Button>
        </Space>

        {/* Table */}
        <Table
          size="small"
          rowKey="run_id"
          loading={loading}
          columns={columns}
          dataSource={filteredItems}
          scroll={{ x: 1200 }}
          pagination={{
            pageSize: 20,
            showSizeChanger: true,
            pageSizeOptions: ['10', '20', '50', '100'],
            showTotal: (t) => `共 ${t} 条`,
          }}
          locale={{
            emptyText: error
              ? <Empty description={`加载失败：${error}`} />
              : <Empty description={<span>暂无 Run — 先去 <ExperimentOutlined /> 建模工作台 创建实验</span>} />,
          }}
        />
      </Card>

      {/* Inspector Drawer — close returns to table; no route change */}
      <RunInspector
        open={drawerOpen}
        runId={activeRunId}
        defaultTab={drawerDefaultTab}
        onClose={() => setDrawerOpen(false)}
      />
    </div>
  )
}
