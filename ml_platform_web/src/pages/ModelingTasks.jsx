import { useActiveEffect } from '../hooks/useActiveEffect'
import React, { useState, useCallback } from 'react'
import {
  Card, Button, Table, Tag, Space, Tooltip, message, Popconfirm, Empty,
} from '../ui'
import { PlusOutlined, AppstoreOutlined, ExperimentOutlined, TrophyOutlined, ReloadOutlined, CheckCircleFilled, ClockCircleFilled, CloseCircleFilled } from '@ant-design/icons'
import { Link, useNavigate } from 'react-router-dom'
import { modelingTaskApi } from '../services/api'
import { formatDateTime } from '../utils/formatters'
import MetricCard from '../components/layout/MetricCard'

const STATUS_META = {
  CREATED:   { color: 'default', icon: <ClockCircleFilled />, label: '待启动' },
  RUNNING:   { color: 'processing', icon: <ClockCircleFilled spin />, label: '运行中' },
  COMPLETED: { color: 'success', icon: <CheckCircleFilled />, label: '已完成' },
  FAILED:    { color: 'error', icon: <CloseCircleFilled />, label: '失败' },
  ARCHIVED:  { color: 'default', icon: null, label: '已归档' },
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

  useActiveEffect(() => { load() }, [load])

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
          <Link to={`/v3/tasks/${row.id}`} style={{ fontWeight: 600, fontSize: 13 }}>
            {name}
          </Link>
          {row.description && (
            <div style={{ color: 'var(--text-secondary)', fontSize: 12, marginTop: 2, lineHeight: 1.35 }}>
              {row.description}
            </div>
          )}
          <div style={{ color: 'var(--text-muted)', fontSize: 11, marginTop: 2 }}>
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
          {m} <span style={{ color: 'var(--text-muted)' }}>({row.objective_direction})</span>
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
      width: 110,
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
      render: (v) => v ? formatDateTime(v) : '-',
    },
    {
      title: '操作',
      key: 'actions',
      width: 220,
      fixed: 'right',
      render: (_, row) => {
        const canDeploy = (row.successful_run_count ?? 0) > 0
        return (
          <Space size={16} className="table-actions">
            <Tooltip title="进入工作流（数据→配置→训练→可视化→部署）">
              <Button size="small" onClick={() => navigate(`/v3/tasks/${row.id}/workflow`)} type="link" className="table-action">工作流</Button>
            </Tooltip>
            <Tooltip title={canDeploy ? '部署最佳模型' : '暂无成功的 Run，无法部署'}>
              <Button size="small" disabled={!canDeploy} onClick={() => navigate(`/v3/tasks/${row.id}/workflow?step=3`)} type="link" className="table-action">部署</Button>
            </Tooltip>
            <Tooltip title="查看详情（Tab 视图）">
              <Button size="small" onClick={() => navigate(`/v3/tasks/${row.id}`)} type="link" className="table-action">详情</Button>
            </Tooltip>
            <Popconfirm okButtonProps={{ danger: true }}
              title="确认删除此建模任务？"
              description="其下所有实验与 Run 都会被级联清理。"
              onConfirm={() => handleDelete(row.id)}
            >
              <Button size="small" danger disabled={row.status === 'RUNNING'} type="link" className="table-action">删除</Button>
            </Popconfirm>
          </Space>
        )
      },
    },
  ]

  return (
    <div style={{ padding: 16 }}>
      <div className="page-heading">
        <div><h1>建模任务工作台</h1><p>按任务组织数据、训练、评估与部署，查看各实验的运行结果。</p></div>
        <Space><Button icon={<ReloadOutlined />} onClick={load}>刷新</Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => navigate('/v3/tasks/new/workflow')}>新建建模任务</Button></Space>
      </div>
      <div className="metric-grid">
        <MetricCard icon={<AppstoreOutlined />} label="总任务" value={stats.total} />
        <MetricCard icon={<ExperimentOutlined />} label="运行中（本页）" value={stats.running} color="var(--info)" />
        <MetricCard icon={<TrophyOutlined />} label="已完成（本页）" value={stats.completed} color="var(--success)" />
        <MetricCard icon={<CloseCircleFilled />} label="失败（本页）" value={stats.failed} color="var(--error)" />
      </div>

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
                <span style={{ color: 'var(--text-secondary)' }}>还没有建模任务，先点右上角「新建建模任务」开始</span>
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
            onChange: (page, pageSize) => setPagination({showTotal: total => `共 ${total} 条`, showSizeChanger: false,  page, pageSize }),
          }}
        />
      </Card>
    </div>
  )
}
