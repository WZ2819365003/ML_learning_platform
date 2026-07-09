import React, { useState, useEffect, useCallback } from 'react'
import {
  Card, Button, Table, Tag, Space, Tooltip, message, Popconfirm, Empty,
} from 'antd'
import {
  PlusOutlined, AppstoreOutlined, ExperimentOutlined, TrophyOutlined,
  ReloadOutlined, DeleteOutlined, EyeOutlined, CheckCircleFilled,
  ClockCircleFilled, CloseCircleFilled, FireOutlined, EditOutlined,
  CloudUploadOutlined,
} from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { modelingTaskApi } from '../services/api'

const STATUS_META = {
  CREATED:   { color: 'default', icon: <ClockCircleFilled />, label: '待启动' },
  RUNNING:   { color: 'processing', icon: <ClockCircleFilled spin />, label: '运行中' },
  COMPLETED: { color: 'success', icon: <CheckCircleFilled />, label: '已完成' },
  FAILED:    { color: 'error', icon: <CloseCircleFilled />, label: '失败' },
  ARCHIVED:  { color: 'default', icon: null, label: '已归档' },
}

// ── Inline stat chip (compact, lives in the card header strip) ───────────────
function StatChip({ icon, label, value, color }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8,
      padding: '6px 14px', borderRadius: 8,
      background: 'rgba(148, 163, 184, 0.08)',
      border: '1px solid rgba(148, 163, 184, 0.14)',
      minWidth: 120,
    }}>
      <div style={{
        width: 28, height: 28, borderRadius: 6,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: `${color}1a`, color,
      }}>
        {icon}
      </div>
      <div style={{ lineHeight: 1.15 }}>
        <div style={{ fontSize: 11, color: '#64748b', fontWeight: 500 }}>{label}</div>
        <div style={{ fontSize: 16, fontWeight: 700, color: '#0f172a' }}>{value}</div>
      </div>
    </div>
  )
}

export default function ModelingTasks() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [data, setData] = useState({ items: [], total: 0 })
  const [pagination, setPagination] = useState({ page: 1, pageSize: 10 })

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const resp = await modelingTaskApi.list({
        page: pagination.page,
        page_size: pagination.pageSize,
      })
      setData(resp)
    } catch (err) {
      message.error(err?.response?.data?.detail || '加载建模任务失败')
    } finally {
      setLoading(false)
    }
  }, [pagination.page, pagination.pageSize])

  useEffect(() => { load() }, [load])

  const handleDelete = async (taskId) => {
    try {
      await modelingTaskApi.delete(taskId)
      message.success('已删除')
      await load()
    } catch (err) {
      message.error(err?.response?.data?.detail || '删除失败')
    }
  }

  // ── Header stats (computed from current page; for MVP)
  const stats = (() => {
    const items = Array.isArray(data?.items) ? data.items : []
    const byStatus = items.reduce((acc, t) => {
      acc[t.status] = (acc[t.status] || 0) + 1
      return acc
    }, {})
    return {
      total: data?.total ?? items.length,
      running: byStatus.RUNNING || 0,
      completed: byStatus.COMPLETED || 0,
      failed: byStatus.FAILED || 0,
    }
  })()

  const columns = [
    {
      title: '任务',
      dataIndex: 'name',
      key: 'name',
      render: (name, row) => (
        <div>
          <a onClick={() => navigate(`/v3/tasks/${row.id}`)} style={{ fontWeight: 600, fontSize: 13 }}>
            {name}
          </a>
          {row.description && (
            <div style={{ color: '#64748b', fontSize: 12, marginTop: 2, lineHeight: 1.35 }}>
              {row.description}
            </div>
          )}
          <div style={{ color: '#94a3b8', fontSize: 11, marginTop: 2 }}>
            {row.dataset_name && <>数据集: {row.dataset_name}</>}
            {row.target_column && <> · 目标: {row.target_column}</>}
          </div>
        </div>
      ),
    },
    {
      title: '类型',
      dataIndex: 'task_type',
      key: 'task_type',
      width: 80,
      render: (t) => <Tag color={t === 'regression' ? 'geekblue' : 'cyan'}>
        {t === 'regression' ? '回归' : '分类'}
      </Tag>,
    },
    {
      title: '优化目标',
      dataIndex: 'objective_metric',
      key: 'objective_metric',
      width: 140,
      render: (m, row) => (
        <span style={{ fontFamily: 'monospace', fontSize: 12 }}>
          {m} <span style={{ color: '#94a3b8' }}>({row.objective_direction})</span>
        </span>
      ),
    },
    {
      title: '实验 / 成功 Run',
      key: 'counts',
      width: 150,
      render: (_, row) => (
        <Space size={4}>
          <Tag color="blue">{row.experiment_count ?? 0} 批次</Tag>
          <Tag color="green">{row.successful_run_count ?? 0} 成功</Tag>
        </Space>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (s) => {
        const meta = STATUS_META[s] || STATUS_META.CREATED
        return <Tag icon={meta.icon} color={meta.color}>{meta.label}</Tag>
      },
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 150,
      render: (v) => v ? new Date(v).toLocaleString('zh-CN', { hour12: false }) : '-',
    },
    {
      title: '操作',
      key: 'actions',
      width: 220,
      fixed: 'right',
      render: (_, row) => {
        const canDeploy = (row.successful_run_count ?? 0) > 0
        return (
          <Space size={4}>
            <Tooltip title="进入工作流（数据→配置→训练→可视化→部署）">
              <Button size="small" type="primary" ghost icon={<EditOutlined />}
                onClick={() => navigate(`/v3/tasks/${row.id}/workflow`)}>工作流</Button>
            </Tooltip>
            <Tooltip title={canDeploy ? '部署最佳模型' : '暂无成功的 Run，无法部署'}>
              <Button size="small" icon={<CloudUploadOutlined />} disabled={!canDeploy}
                onClick={() => navigate(`/v3/tasks/${row.id}/workflow?step=3`)} />
            </Tooltip>
            <Tooltip title="查看详情（Tab 视图）">
              <Button size="small" icon={<EyeOutlined />}
                onClick={() => navigate(`/v3/tasks/${row.id}`)} />
            </Tooltip>
            <Popconfirm
              title="确认删除此建模任务？"
              description="其下所有实验与 Run 都会被级联清理。"
              onConfirm={() => handleDelete(row.id)}
            >
              <Button size="small" danger icon={<DeleteOutlined />} disabled={row.status === 'RUNNING'} />
            </Popconfirm>
          </Space>
        )
      },
    },
  ]

  return (
    <div style={{ padding: 16 }}>
      {/* ── Title bar + actions (compact, single row) ───────────────────── */}
      <Card
        bordered={false}
        style={{ marginBottom: 12, boxShadow: '0 1px 2px rgba(15, 23, 42, 0.04)' }}
        bodyStyle={{ padding: '12px 16px' }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
          <div style={{ minWidth: 260 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <FireOutlined style={{ color: '#2563eb', fontSize: 20 }} />
              <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700, lineHeight: 1.2 }}>建模任务工作台</h2>
              <Tag color="blue" style={{ marginLeft: 4 }}>V3</Tag>
            </div>
            <div style={{ color: '#64748b', fontSize: 12, marginTop: 4 }}>
              以「任务」为单位组织建模流程：一个任务可挂多组实验（基线/网格/贝叶斯），自动汇总最佳 Run。
            </div>
          </div>

          {/* Stats chips — inline, no wasted vertical space */}
          <Space size={8} wrap>
            <StatChip icon={<AppstoreOutlined />} label="总任务" value={stats.total} color="#2563eb" />
            <StatChip icon={<ExperimentOutlined />} label="运行中" value={stats.running} color="#0ea5e9" />
            <StatChip icon={<TrophyOutlined />} label="已完成" value={stats.completed} color="#10b981" />
            <StatChip icon={<CloseCircleFilled />} label="失败" value={stats.failed} color="#ef4444" />
          </Space>

          <Space>
            <Button icon={<ReloadOutlined />} onClick={load}>刷新</Button>
            <Button type="primary" icon={<PlusOutlined />} onClick={() => navigate('/v3/tasks/new/workflow')}>
              新建建模任务
            </Button>
          </Space>
        </div>
      </Card>

      {/* ── Main table card ─────────────────────────────────────────────── */}
      <Card bordered={false} bodyStyle={{ padding: 0 }} style={{ boxShadow: '0 1px 2px rgba(15, 23, 42, 0.04)' }}>
        <Table
          rowKey="id"
          size="middle"
          columns={columns}
          dataSource={Array.isArray(data?.items) ? data.items : []}
          loading={loading}
          scroll={{ x: 1080 }}
          locale={{
            emptyText: <div style={{ padding: '40px 0' }}>
              <Empty description={
                <span style={{ color: '#64748b' }}>还没有建模任务，先点右上角「新建建模任务」开始</span>
              } />
            </div>
          }}
          pagination={{
            current: pagination.page,
            pageSize: pagination.pageSize,
            total: data?.total ?? 0,
            showSizeChanger: true,
            showQuickJumper: true,
            pageSizeOptions: ['10', '20', '50'],
            showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条 / 共 ${total} 条`,
            onChange: (page, pageSize) => setPagination({ page, pageSize }),
            style: { padding: '12px 16px', margin: 0 },
          }}
        />
      </Card>
    </div>
  )
}
