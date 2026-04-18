import React, { useCallback, useEffect, useState } from 'react'
import {
  Card, Button, Tabs, Tag, Space, Table, Descriptions, Typography, Empty,
  Row, Col, Spin, Alert, Statistic, Tooltip, Breadcrumb, message, Popconfirm,
} from 'antd'
import {
  ArrowLeftOutlined, ReloadOutlined, PlusOutlined, TrophyOutlined,
  NodeIndexOutlined, LineChartOutlined, BulbOutlined, ExperimentOutlined,
  CheckCircleFilled, CloseCircleFilled, ClockCircleFilled, FireOutlined,
} from '@ant-design/icons'
import { useNavigate, useParams } from 'react-router-dom'
import { modelingTaskApi, platformRunsApi } from '../services/api'
import ExperimentBatchModal from '../components/workbench/ExperimentBatchModal'
import RunInspector from '../components/workbench/RunInspector'

const { Text, Paragraph } = Typography

const STATUS_META = {
  CREATED:   { color: 'default', label: '待启动', icon: <ClockCircleFilled /> },
  RUNNING:   { color: 'processing', label: '运行中', icon: <ClockCircleFilled spin /> },
  COMPLETED: { color: 'success', label: '已完成', icon: <CheckCircleFilled /> },
  FAILED:    { color: 'error', label: '失败', icon: <CloseCircleFilled /> },
  ARCHIVED:  { color: 'default', label: '已归档' },
}

const STRATEGY_COLOR = {
  baseline: 'green',
  grid_search: 'blue',
  bayesian_search: 'purple',
  automl_registry: 'gold',
}

function renderStatus(status) {
  const meta = STATUS_META[status] || { color: 'default', label: status }
  return <Tag icon={meta.icon} color={meta.color}>{meta.label}</Tag>
}

function fmt(n, digits = 4) {
  if (typeof n !== 'number' || !Number.isFinite(n)) return '-'
  return n.toFixed(digits)
}

export default function ModelingTaskDetail() {
  const { taskId } = useParams()
  const navigate = useNavigate()

  const [loading, setLoading] = useState(false)
  const [task, setTask] = useState(null)
  const [experiments, setExperiments] = useState([])
  const [runStats, setRunStats] = useState({ total: 0, success: 0, running: 0, failed: 0 })
  const [leaderboard, setLeaderboard] = useState([])
  const [lbLoading, setLbLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')

  // Modals
  const [batchOpen, setBatchOpen] = useState(false)
  const [inspectorRunId, setInspectorRunId] = useState(null)

  const loadTask = useCallback(async () => {
    setLoading(true)
    try {
      const resp = await modelingTaskApi.get(taskId)
      setTask(resp)
      setExperiments(resp.experiments || [])
      setRunStats(resp.run_stats || {})
    } catch (err) {
      message.error(err?.response?.data?.detail || '加载建模任务失败')
    } finally {
      setLoading(false)
    }
  }, [taskId])

  const loadLeaderboard = useCallback(async () => {
    setLbLoading(true)
    try {
      const resp = await modelingTaskApi.leaderboard(taskId, 50)
      setLeaderboard(Array.isArray(resp) ? resp : [])
    } catch (err) {
      message.error(err?.response?.data?.detail || '加载排行榜失败')
    } finally {
      setLbLoading(false)
    }
  }, [taskId])

  useEffect(() => { loadTask() }, [loadTask])
  useEffect(() => {
    if (activeTab === 'runs' || activeTab === 'explain') loadLeaderboard()
  }, [activeTab, loadLeaderboard])

  // Auto-refresh when task is running
  useEffect(() => {
    if (task?.status !== 'RUNNING') return
    const id = setInterval(() => { loadTask(); if (activeTab === 'runs') loadLeaderboard() }, 5000)
    return () => clearInterval(id)
  }, [task?.status, activeTab, loadTask, loadLeaderboard])

  const refreshAll = async () => {
    await loadTask()
    if (activeTab === 'runs' || activeTab === 'explain') await loadLeaderboard()
  }

  if (loading && !task) {
    return (
      <div style={{ padding: 40, display: 'flex', justifyContent: 'center' }}>
        <Spin size="large" tip="加载建模任务中..." />
      </div>
    )
  }

  if (!task) {
    return <Alert type="error" showIcon message="建模任务不存在" style={{ margin: 16 }} />
  }

  const bestRun = leaderboard.find(r => r.rank === 1)

  // ── Tab: Overview ─────────────────────────────────────────────────────────
  const overviewTab = (
    <Row gutter={[12, 12]}>
      <Col span={16}>
        <Card size="small" title="任务信息" bodyStyle={{ padding: 14 }}>
          <Descriptions column={2} size="small" bordered
            labelStyle={{ background: '#f8fafc', width: 110 }}>
            <Descriptions.Item label="任务 ID" span={2}>
              <code style={{ fontSize: 12 }}>{task.id}</code>
            </Descriptions.Item>
            <Descriptions.Item label="数据集">{task.dataset_name || '-'}</Descriptions.Item>
            <Descriptions.Item label="目标列">{task.target_column || '-'}</Descriptions.Item>
            <Descriptions.Item label="类型">
              <Tag color={task.task_type === 'regression' ? 'geekblue' : 'cyan'}>
                {task.task_type === 'regression' ? '回归' : '分类'}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label="优化目标">
              <code>{task.objective_metric} ({task.objective_direction})</code>
            </Descriptions.Item>
            <Descriptions.Item label="状态">{renderStatus(task.status)}</Descriptions.Item>
            <Descriptions.Item label="创建时间">
              {task.created_at ? new Date(task.created_at).toLocaleString('zh-CN', { hour12: false }) : '-'}
            </Descriptions.Item>
            {task.description && (
              <Descriptions.Item label="描述" span={2}>
                <Paragraph style={{ margin: 0 }}>{task.description}</Paragraph>
              </Descriptions.Item>
            )}
          </Descriptions>
        </Card>
      </Col>

      <Col span={8}>
        <Card size="small" title={<span><TrophyOutlined style={{ color: '#f59e0b' }} /> 最佳 Run</span>}
          bodyStyle={{ padding: 14 }}>
          {task.best_run_id ? (
            <>
              <Statistic
                title={task.objective_metric}
                value={bestRun?.objective_value ?? '-'}
                precision={4}
                valueStyle={{ color: '#10b981' }}
              />
              <Space direction="vertical" size={4} style={{ marginTop: 8, width: '100%' }}>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  实验：{bestRun?.experiment_name || task.best_experiment_id?.slice(0, 8)}
                </Text>
                <Button size="small" type="primary" ghost block
                  onClick={() => setInspectorRunId(task.best_run_id)}>
                  查看最佳 Run 详情
                </Button>
              </Space>
            </>
          ) : (
            <Empty image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={<Text type="secondary" style={{ fontSize: 12 }}>还没有成功的 Run</Text>} />
          )}
        </Card>
      </Col>

      <Col span={6}>
        <Card size="small" bodyStyle={{ padding: 14 }}>
          <Statistic title="实验批次数" value={experiments.length} prefix={<NodeIndexOutlined />} />
        </Card>
      </Col>
      <Col span={6}>
        <Card size="small" bodyStyle={{ padding: 14 }}>
          <Statistic title="Run 总数" value={runStats.total || 0} prefix={<LineChartOutlined />} />
        </Card>
      </Col>
      <Col span={6}>
        <Card size="small" bodyStyle={{ padding: 14 }}>
          <Statistic title="成功 Run" value={runStats.success || 0}
            valueStyle={{ color: '#10b981' }} prefix={<CheckCircleFilled />} />
        </Card>
      </Col>
      <Col span={6}>
        <Card size="small" bodyStyle={{ padding: 14 }}>
          <Statistic title="失败 Run" value={runStats.failed || 0}
            valueStyle={{ color: '#ef4444' }} prefix={<CloseCircleFilled />} />
        </Card>
      </Col>

      {task.summary_snapshot && Object.keys(task.summary_snapshot).length > 0 && (
        <Col span={24}>
          <Card size="small" title="任务快照" bodyStyle={{ padding: 14 }}>
            <pre style={{ margin: 0, fontSize: 11, background: '#f8fafc',
              padding: 8, borderRadius: 4, maxHeight: 220, overflow: 'auto' }}>
              {JSON.stringify(task.summary_snapshot, null, 2)}
            </pre>
          </Card>
        </Col>
      )}
    </Row>
  )

  // ── Tab: Experiments ──────────────────────────────────────────────────────
  const experimentsColumns = [
    {
      title: '批次名称', dataIndex: 'name', key: 'name',
      render: (n, row) => (
        <div>
          <div style={{ fontWeight: 600, fontSize: 13 }}>{n}</div>
          <Text type="secondary" style={{ fontSize: 11 }}>
            {row.id?.slice(0, 8)}
          </Text>
        </div>
      ),
    },
    {
      title: '策略', dataIndex: 'strategy_type', key: 'strategy_type', width: 130,
      render: (s) => s ? <Tag color={STRATEGY_COLOR[s] || 'default'}>{s}</Tag> : '-',
    },
    {
      title: '参与模型', dataIndex: 'selected_models', key: 'selected_models',
      render: (models) => (
        <Space size={4} wrap>
          {(models || []).slice(0, 4).map(m => <Tag key={m} style={{ margin: 0 }}>{m}</Tag>)}
          {models?.length > 4 && <Tag>+{models.length - 4}</Tag>}
        </Space>
      ),
    },
    {
      title: '预算', dataIndex: 'budget_config', key: 'budget', width: 180,
      render: (b) => b && Object.keys(b).length ? (
        <Text code style={{ fontSize: 11 }}>
          {b.max_trials && `max=${b.max_trials}`}
          {b.n_trials_per_model && ` / model=${b.n_trials_per_model}`}
        </Text>
      ) : '-',
    },
    { title: '状态', dataIndex: 'status', key: 'status', width: 110, render: renderStatus },
    {
      title: '创建时间', dataIndex: 'created_at', key: 'created_at', width: 150,
      render: (v) => v ? new Date(v).toLocaleString('zh-CN', { hour12: false }) : '-',
    },
    {
      title: '操作', key: 'actions', width: 120,
      render: (_, row) => (
        <Button size="small" type="link"
          onClick={() => { setActiveTab('runs') }}>查看 Run →</Button>
      ),
    },
  ]

  const experimentsTab = (
    <Card size="small" bodyStyle={{ padding: 0 }}>
      <div style={{ padding: '10px 14px', borderBottom: '1px solid #f1f5f9',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Space>
          <Text strong>实验批次</Text>
          <Tag>{experiments.length}</Tag>
        </Space>
        <Button type="primary" size="small" icon={<PlusOutlined />}
          onClick={() => setBatchOpen(true)}>
          启动新批次
        </Button>
      </div>
      <Table
        size="small"
        rowKey="id"
        columns={experimentsColumns}
        dataSource={experiments}
        pagination={experiments.length > 10 ? {
          pageSize: 10, showSizeChanger: true, showTotal: (t) => `共 ${t} 条`,
          style: { padding: '10px 14px', margin: 0 },
        } : false}
        locale={{ emptyText: <div style={{ padding: 24 }}>
          <Empty description="还没有实验批次" />
        </div> }}
      />
    </Card>
  )

  // ── Tab: Runs (leaderboard) ───────────────────────────────────────────────
  const leaderboardColumns = [
    {
      title: 'Rank', dataIndex: 'rank', key: 'rank', width: 70,
      render: (r) => r === 1 ? <Tag color="gold" icon={<TrophyOutlined />}>{r}</Tag>
        : r <= 3 ? <Tag color="blue">{r}</Tag> : <span style={{ color: '#64748b' }}>{r}</span>,
    },
    {
      title: '实验', dataIndex: 'experiment_name', key: 'experiment_name', ellipsis: true,
      render: (name, row) => (
        <div>
          <div style={{ fontSize: 12, fontWeight: 500 }}>{name}</div>
          <Tag color={STRATEGY_COLOR[row.strategy_type] || 'default'} style={{ fontSize: 10, marginTop: 2 }}>
            {row.strategy_type}
          </Tag>
        </div>
      ),
    },
    { title: 'Trial', dataIndex: 'trial_no', key: 'trial_no', width: 70 },
    {
      title: task?.objective_metric || '目标值',
      dataIndex: 'objective_value', key: 'objective_value', width: 120,
      render: (v) => <code style={{ color: '#2563eb', fontSize: 12, fontWeight: 600 }}>{fmt(v)}</code>,
      sorter: (a, b) => (a.objective_value ?? 0) - (b.objective_value ?? 0),
    },
    {
      title: '超参', key: 'params', ellipsis: true,
      render: (_, row) => {
        const entries = Object.entries(row.params || {}).slice(0, 3)
        return (
          <Tooltip title={<pre style={{ margin: 0, fontSize: 10, whiteSpace: 'pre-wrap' }}>
            {JSON.stringify(row.params, null, 2)}
          </pre>}>
            <Space size={4} wrap>
              {entries.map(([k, v]) => (
                <Tag key={k} style={{ margin: 0, fontSize: 10 }}>
                  {k}={typeof v === 'object' ? '...' : String(v)}
                </Tag>
              ))}
              {Object.keys(row.params || {}).length > 3 && <Tag>...</Tag>}
            </Space>
          </Tooltip>
        )
      },
    },
    {
      title: '完成时间', dataIndex: 'finished_at', key: 'finished_at', width: 140,
      render: (v) => v ? new Date(v).toLocaleString('zh-CN', { hour12: false }) : '-',
    },
    {
      title: '操作', key: 'actions', width: 80,
      render: (_, row) => (
        <Button size="small" type="link" onClick={() => setInspectorRunId(row.run_id)}>详情</Button>
      ),
    },
  ]

  const runsTab = (
    <Card size="small" bodyStyle={{ padding: 0 }}>
      <div style={{ padding: '10px 14px', borderBottom: '1px solid #f1f5f9',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Space>
          <Text strong>Run 排行榜</Text>
          <Tag>{leaderboard.length}</Tag>
          <Text type="secondary" style={{ fontSize: 12 }}>
            按 <code>{task?.objective_metric}</code> ({task?.objective_direction}) 排序
          </Text>
        </Space>
        <Button size="small" icon={<ReloadOutlined />} onClick={loadLeaderboard} loading={lbLoading}>
          刷新
        </Button>
      </div>
      <Table
        size="small"
        rowKey="run_id"
        loading={lbLoading}
        columns={leaderboardColumns}
        dataSource={leaderboard}
        pagination={leaderboard.length > 10 ? {
          pageSize: 10, showSizeChanger: true, showTotal: (t) => `共 ${t} 条`,
          style: { padding: '10px 14px', margin: 0 },
        } : false}
        onRow={(row) => ({
          onClick: () => setInspectorRunId(row.run_id),
          style: { cursor: 'pointer' },
        })}
        locale={{ emptyText: <div style={{ padding: 24 }}>
          <Empty description="还没有成功的 Run" />
        </div> }}
      />
    </Card>
  )

  // ── Tab: Explain (top-3 runs SHAP) ───────────────────────────────────────
  const topRuns = leaderboard.slice(0, 3)
  const explainTab = (
    <Card size="small" bodyStyle={{ padding: 14 }}>
      {topRuns.length === 0 ? (
        <Empty description="暂无可解释的 Run" />
      ) : (
        <Row gutter={[12, 12]}>
          {topRuns.map((r) => (
            <Col key={r.run_id} span={8}>
              <Card size="small" hoverable
                onClick={() => setInspectorRunId(r.run_id)}
                title={<Space>
                  {r.rank === 1 ? <TrophyOutlined style={{ color: '#f59e0b' }} /> : null}
                  <span>Rank #{r.rank}</span>
                </Space>}
                extra={<Text code style={{ color: '#2563eb' }}>{fmt(r.objective_value)}</Text>}
              >
                <Text type="secondary" style={{ fontSize: 12 }}>{r.experiment_name}</Text>
                <div style={{ marginTop: 6 }}>
                  <Tag color={STRATEGY_COLOR[r.strategy_type]}>{r.strategy_type}</Tag>
                  <Tag>trial #{r.trial_no}</Tag>
                </div>
                <Button size="small" type="primary" ghost block style={{ marginTop: 12 }}
                  icon={<BulbOutlined />}>
                  查看 SHAP 解释 →
                </Button>
              </Card>
            </Col>
          ))}
        </Row>
      )}
    </Card>
  )

  return (
    <div style={{ padding: 16 }}>
      {/* ── Header strip ────────────────────────────────────────────────── */}
      <Card bordered={false} bodyStyle={{ padding: '12px 16px' }}
        style={{ marginBottom: 12, boxShadow: '0 1px 2px rgba(15, 23, 42, 0.04)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: 280 }}>
            <Breadcrumb style={{ fontSize: 12, marginBottom: 6 }}
              items={[
                { title: <a onClick={() => navigate('/v3/tasks')}>建模工作台</a> },
                { title: task.name },
              ]}
            />
            <Space align="center" size={10}>
              <FireOutlined style={{ color: '#2563eb', fontSize: 22 }} />
              <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700 }}>{task.name}</h2>
              {renderStatus(task.status)}
              <Tag color="blue">{task.task_type === 'regression' ? '回归' : '分类'}</Tag>
              <Tag>
                <code>{task.objective_metric}</code> · {task.objective_direction}
              </Tag>
            </Space>
            {task.description && (
              <div style={{ color: '#64748b', fontSize: 12, marginTop: 4 }}>{task.description}</div>
            )}
          </div>
          <Space>
            <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/v3/tasks')}>返回</Button>
            <Button icon={<ReloadOutlined />} onClick={refreshAll}>刷新</Button>
            <Button type="primary" icon={<PlusOutlined />} onClick={() => setBatchOpen(true)}>
              启动新批次
            </Button>
          </Space>
        </div>
      </Card>

      {/* ── Tabs ────────────────────────────────────────────────────────── */}
      <Card bordered={false} bodyStyle={{ padding: '0 16px 16px' }}
        style={{ boxShadow: '0 1px 2px rgba(15, 23, 42, 0.04)' }}>
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          items={[
            { key: 'overview',    label: <span><ExperimentOutlined /> 任务概览</span>, children: overviewTab },
            { key: 'experiments', label: <span><NodeIndexOutlined /> 实验编排 ({experiments.length})</span>, children: experimentsTab },
            { key: 'runs',        label: <span><LineChartOutlined /> Run 对比 ({runStats.success || 0})</span>, children: runsTab },
            { key: 'explain',     label: <span><BulbOutlined /> 模型解释</span>, children: explainTab },
          ]}
        />
      </Card>

      {/* ── Modals ──────────────────────────────────────────────────────── */}
      <ExperimentBatchModal
        open={batchOpen}
        task={task}
        onClose={() => setBatchOpen(false)}
        onSubmitted={async () => {
          await loadTask()
          setActiveTab('experiments')
        }}
      />
      <RunInspector
        open={!!inspectorRunId}
        runId={inspectorRunId}
        onClose={() => setInspectorRunId(null)}
      />
    </div>
  )
}
