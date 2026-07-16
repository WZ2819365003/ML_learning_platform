import React, { useCallback, useEffect, useState } from 'react'
import {
  Card, Button, Tabs, Tag, Space, Table, Descriptions, Typography, Empty,
  Row, Col, Spin, Alert, Statistic, Breadcrumb, message,
} from 'antd'
import {
  ArrowLeftOutlined, ReloadOutlined, PlusOutlined, TrophyOutlined,
  NodeIndexOutlined, LineChartOutlined, ExperimentOutlined,
  CheckCircleFilled, CloseCircleFilled, ClockCircleFilled, FireOutlined,
} from '@ant-design/icons'
import { useNavigate, useParams } from 'react-router-dom'
import { modelingTaskApi } from '../services/api'
import ExperimentBatchModal from '../components/workbench/ExperimentBatchModal'
import RunInspector from '../components/workbench/RunInspector'
import TrainingPlanSnapshotView from '../components/workbench/TrainingPlanSnapshotView'
import ProgressTree from '../components/workbench/ProgressTree'
import ModelComparison from '../components/workbench/ModelComparison'

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

export default function ModelingTaskDetail() {
  const { taskId } = useParams()
  const navigate = useNavigate()

  const [loading, setLoading] = useState(false)
  const [task, setTask] = useState(null)
  const [experiments, setExperiments] = useState([])
  const [runStats, setRunStats] = useState({ total: 0, success: 0, running: 0, failed: 0 })
  const [leaderboard, setLeaderboard] = useState([])
  const [lbLoading, setLbLoading] = useState(false)
  const [runs, setRuns] = useState([])
  const [runsLoading, setRunsLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')

  // Modals
  const [batchOpen, setBatchOpen] = useState(false)
  const [inspectorRunId, setInspectorRunId] = useState(null)
  const [inspectorTab, setInspectorTab] = useState('overview')

  // Open the run inspector on a specific tab (overview for baseline click,
  // 'shap' when the user asks for explanation from the Runs 表格).
  const openInspector = useCallback((runId, tab = 'overview') => {
    setInspectorRunId(runId)
    setInspectorTab(tab)
  }, [])

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

  const loadRuns = useCallback(async () => {
    setRunsLoading(true)
    try {
      const resp = await modelingTaskApi.runs(taskId)
      setRuns(resp?.items || [])
    } catch (err) {
      message.error(err?.response?.data?.detail || '加载 Run 矩阵失败')
    } finally {
      setRunsLoading(false)
    }
  }, [taskId])

  useEffect(() => { loadTask() }, [loadTask])
  // Always load the leaderboard once on mount so Overview's "最佳 Run" card
  // has data; Runs/Explain tabs trigger a refresh on their own.
  useEffect(() => { loadLeaderboard() }, [loadLeaderboard])
  useEffect(() => { loadRuns() }, [loadRuns])
  useEffect(() => {
    if (activeTab === 'compare') {
      loadLeaderboard()
      loadRuns()
    }
  }, [activeTab, loadLeaderboard, loadRuns])

  // Auto-refresh when task is running
  useEffect(() => {
    if (task?.status !== 'RUNNING') return
    const id = setInterval(() => { loadTask(); loadLeaderboard(); loadRuns() }, 5000)
    return () => clearInterval(id)
  }, [task?.status, loadTask, loadLeaderboard, loadRuns])

  const refreshAll = async () => {
    await loadTask()
    await loadRuns()
    if (activeTab === 'compare') await loadLeaderboard()
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
  const finalizationLocked = task.final_evaluation?.state && task.final_evaluation.state !== 'OPEN'

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
                  onClick={() => openInspector(task.best_run_id, 'overview')}>
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

      <Col span={24}>
        <TrainingPlanSnapshotView
          snapshot={task.training_plan_snapshot}
          status={task.training_plan_status}
        />
      </Col>

      <Col span={24}>
        <ProgressTree modelingTaskId={task.id} />
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
      render: () => (
        <Button size="small" type="link" onClick={() => setActiveTab('compare')}>查看对比 →</Button>
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
        <Button type="primary" size="small" icon={<PlusOutlined />} disabled={finalizationLocked}
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

  // ── Tab: 模型对比 (shared ModelComparison over all runs) ─────────────────
  const compareTab = (
    <ModelComparison
      task={task}
      rows={runs.length ? runs : leaderboard}
      loading={runsLoading || lbLoading}
      error={null}
      onRefresh={refreshAll}
    />
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
            <Button type="primary" icon={<PlusOutlined />} disabled={finalizationLocked}
              onClick={() => setBatchOpen(true)}>
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
            { key: 'compare',     label: <span><LineChartOutlined /> 模型对比 ({runStats.total || 0})</span>, children: compareTab },
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
          await loadRuns()
          setActiveTab('experiments')
        }}
      />
      <RunInspector
        open={!!inspectorRunId}
        runId={inspectorRunId}
        defaultTab={inspectorTab}
        onClose={() => setInspectorRunId(null)}
      />
    </div>
  )
}
