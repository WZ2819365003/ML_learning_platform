import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Card, Steps, Button, Space, Select, Input, Upload, Form, Row, Col, Tag,
  Typography, message, Table, Empty, Progress, Tooltip, Divider, Alert,
} from 'antd'
import {
  DatabaseOutlined, ExperimentOutlined, ThunderboltOutlined, LineChartOutlined,
  CloudUploadOutlined, InboxOutlined, PlusOutlined, ReloadOutlined,
  ArrowLeftOutlined, ArrowRightOutlined, TrophyOutlined, BulbOutlined, DownloadOutlined,
} from '@ant-design/icons'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import {
  modelingTaskApi, dataApi, runModelDownloadUrl,
} from '../services/api'
import ExperimentBatchModal from '../components/workbench/ExperimentBatchModal'
import ProgressTree from '../components/workbench/ProgressTree'
import StrategyCompareTab from '../components/workbench/StrategyCompareTab'
import RunInspector from '../components/workbench/RunInspector'
import DeployStep from '../components/workbench/DeployStep'

const { Text, Title } = Typography

const OBJECTIVE_PRESETS = {
  classification: [
    { value: 'accuracy', label: 'Accuracy (越高越好)' },
    { value: 'f1', label: 'F1 (越高越好)' },
    { value: 'roc_auc', label: 'ROC-AUC (越高越好)' },
  ],
  regression: [
    { value: 'rmse', label: 'RMSE (越低越好)' },
    { value: 'mae', label: 'MAE (越低越好)' },
    { value: 'r2', label: 'R² (越高越好)' },
  ],
}
const _dir = (m) => (['rmse', 'mae', 'mse', 'mape'].includes(m) ? 'min' : 'max')

const STEP_ITEMS = [
  { title: '数据', icon: <DatabaseOutlined /> },
  { title: '配置模型', icon: <ExperimentOutlined /> },
  { title: '训练', icon: <ThunderboltOutlined /> },
  { title: '可视化结果', icon: <LineChartOutlined /> },
  { title: '部署上线', icon: <CloudUploadOutlined /> },
]

export default function ModelingWorkflow() {
  const { taskId } = useParams()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const isNew = taskId === 'new'

  const initialStep = isNew ? 0 : Math.min(4, Math.max(0, Number(searchParams.get('step')) || 0))
  const [current, setCurrent] = useState(initialStep)
  const [task, setTask] = useState(null)
  const [runs, setRuns] = useState([])
  const [leaderboard, setLeaderboard] = useState([])
  const [datasets, setDatasets] = useState([])
  const [columnInfo, setColumnInfo] = useState(null)
  const [batchOpen, setBatchOpen] = useState(false)
  const [inspectorRunId, setInspectorRunId] = useState(null)
  const [inspectorTab, setInspectorTab] = useState('overview')
  const [saving, setSaving] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [form] = Form.useForm()

  const loadDatasets = useCallback(async () => {
    try {
      const resp = await dataApi.listDatasets({ page: 1, page_size: 100 })
      setDatasets(resp?.items || resp?.datasets || [])
    } catch {/* non-fatal */}
  }, [])

  const loadTask = useCallback(async () => {
    if (isNew) return
    try {
      const t = await modelingTaskApi.get(taskId)
      setTask(t)
      form.setFieldsValue({
        dataset_id: t.dataset_id, target_column: t.target_column,
        task_type: t.task_type, objective_metric: t.objective_metric,
        name: t.name,
      })
    } catch (err) {
      message.error(err?.response?.data?.detail || '加载任务失败')
    }
  }, [taskId, isNew, form])

  const loadRuns = useCallback(async () => {
    if (isNew) return
    try {
      const [r, lb] = await Promise.all([
        modelingTaskApi.runs(taskId),
        modelingTaskApi.leaderboard(taskId, 50),
      ])
      setRuns(r?.items || [])
      setLeaderboard(Array.isArray(lb) ? lb : [])
    } catch {/* non-fatal */}
  }, [taskId, isNew])

  useEffect(() => { loadDatasets() }, [loadDatasets])
  useEffect(() => { loadTask() }, [loadTask])
  useEffect(() => { loadRuns() }, [loadRuns])

  // Poll while task running so 训练/可视化 stay fresh
  useEffect(() => {
    if (isNew || task?.status !== 'RUNNING') return
    const id = setInterval(() => { loadTask(); loadRuns() }, 5000)
    return () => clearInterval(id)
  }, [isNew, task?.status, loadTask, loadRuns])

  const taskTypeWatch = Form.useWatch('task_type', form) || task?.task_type || 'classification'
  const datasetIdWatch = Form.useWatch('dataset_id', form)

  // Fetch column headers for target-column dropdown when a dataset is chosen
  useEffect(() => {
    if (!datasetIdWatch) { setColumnInfo(null); return }
    let cancelled = false
    dataApi.previewDataset(datasetIdWatch)
      .then((resp) => { if (!cancelled) setColumnInfo(resp?.columns_info || null) })
      .catch(() => { if (!cancelled) setColumnInfo(null) })
    return () => { cancelled = true }
  }, [datasetIdWatch])

  const targetOptions = useMemo(() => {
    if (!columnInfo) return []
    return Object.entries(columnInfo).map(([col, meta]) => ({
      value: col,
      label: <Space size={6}><span>{col}</span>
        <Tag style={{ fontSize: 10, margin: 0 }}>{meta.dtype}</Tag></Space>,
    }))
  }, [columnInfo])

  const handleUpload = async (file) => {
    setUploading(true)
    try {
      const resp = await dataApi.uploadDataset(file)
      message.success(`已上传 ${resp.name}`)
      await loadDatasets()
      form.setFieldsValue({ dataset_id: resp.id, target_column: undefined })
    } catch (err) {
      message.error(err?.response?.data?.detail || '上传失败')
    } finally {
      setUploading(false)
    }
    return false // prevent antd auto-upload
  }

  // Persist step-1 (create new task or update existing), then go to 配置模型
  const saveDataStep = async () => {
    let values
    try { values = await form.validateFields() } catch { return }
    setSaving(true)
    try {
      const payload = {
        name: values.name,
        dataset_id: values.dataset_id || null,
        target_column: values.target_column || null,
        task_type: values.task_type,
        objective_metric: values.objective_metric,
        objective_direction: _dir(values.objective_metric),
      }
      if (isNew) {
        const created = await modelingTaskApi.create(payload)
        message.success('任务已创建')
        navigate(`/v3/tasks/${created.id}/workflow`, { replace: true })
        setTask(created)
        setCurrent(1)
      } else {
        await modelingTaskApi.update(taskId, payload)
        message.success('已保存')
        await loadTask()
        setCurrent(1)
      }
    } catch (err) {
      message.error(err?.response?.data?.detail || '保存失败')
    } finally {
      setSaving(false)
    }
  }

  const bestRunId = leaderboard.find(r => r.rank === 1)?.run_id
  const runStatusCounts = runs.reduce((acc, r) => {
    const s = String(r.status).toUpperCase()
    acc[s] = (acc[s] || 0) + 1
    return acc
  }, {})

  // ── Step renderers ─────────────────────────────────────────────────────────
  const dataStep = (
    <Card size="small" bodyStyle={{ padding: 20 }}>
      <Form form={form} layout="vertical" initialValues={{ task_type: 'classification', objective_metric: 'accuracy' }}>
        <Form.Item name="name" label="任务名称" rules={[{ required: true, min: 2, message: '至少 2 个字符' }]}>
          <Input placeholder="例：鸢尾花分类 v1" style={{ maxWidth: 420 }} />
        </Form.Item>

        <Upload.Dragger accept=".csv,.xlsx,.parquet" showUploadList={false}
          beforeUpload={handleUpload} disabled={uploading} style={{ marginBottom: 16 }}>
          <p className="ant-upload-drag-icon"><InboxOutlined /></p>
          <p className="ant-upload-text">{uploading ? '上传中…' : '点击或拖拽上传新数据集（CSV / Excel / Parquet）'}</p>
          <p className="ant-upload-hint" style={{ fontSize: 12 }}>也可以直接在下方选择已上传的数据集</p>
        </Upload.Dragger>

        <Row gutter={12}>
          <Col span={14}>
            <Form.Item name="dataset_id" label="数据集" rules={[{ required: true, message: '请选择数据集' }]}>
              <Select showSearch allowClear placeholder="选择已上传的数据集"
                options={datasets.map(d => ({ value: d.id, label: `${d.name} (${d.row_count || '?'} 行)` }))}
                filterOption={(i, o) => String(o.label).toLowerCase().includes(i.toLowerCase())}
                onChange={() => form.setFieldValue('target_column', undefined)} />
            </Form.Item>
          </Col>
          <Col span={10}>
            <Form.Item name="target_column" label="目标列" rules={[{ required: true, message: '请选择目标列' }]}>
              <Select showSearch placeholder={datasetIdWatch ? '选择目标列' : '请先选择数据集'}
                disabled={!datasetIdWatch} options={targetOptions} optionFilterProp="value" />
            </Form.Item>
          </Col>
        </Row>
        <Row gutter={12}>
          <Col span={10}>
            <Form.Item name="task_type" label="任务类型" rules={[{ required: true }]}>
              <Select options={[{ value: 'classification', label: '分类' }, { value: 'regression', label: '回归' }]}
                onChange={(t) => form.setFieldValue('objective_metric', t === 'regression' ? 'rmse' : 'accuracy')} />
            </Form.Item>
          </Col>
          <Col span={14}>
            <Form.Item name="objective_metric" label="优化目标" rules={[{ required: true }]}>
              <Select options={OBJECTIVE_PRESETS[taskTypeWatch]} />
            </Form.Item>
          </Col>
        </Row>
      </Form>
    </Card>
  )

  const configStep = (
    <Card size="small" bodyStyle={{ padding: 20 }}>
      <Space direction="vertical" size={14} style={{ width: '100%' }}>
        <Alert type="info" showIcon
          message="配置一组或多组模型进行训练"
          description="点击下方按钮选择模型（可多选）、策略（基线 / 网格 / 贝叶斯）与超参数空间。可多次启动，形成多组对照实验。" />
        <div>
          <Button type="primary" icon={<PlusOutlined />} size="large"
            onClick={() => setBatchOpen(true)} disabled={!task}>
            配置模型并启动训练
          </Button>
        </div>
        {task?.experiments?.length > 0 && (
          <>
            <Divider style={{ margin: '4px 0' }}><Text type="secondary" style={{ fontSize: 12 }}>已配置的实验批次</Text></Divider>
            <Space wrap>
              {task.experiments.map(e => (
                <Tag key={e.id} color="blue">{e.name} · {e.strategy_type} · {(e.selected_models || []).length} 模型</Tag>
              ))}
            </Space>
          </>
        )}
      </Space>
    </Card>
  )

  const trainStep = (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      <Card size="small" bodyStyle={{ padding: '12px 16px' }}>
        <Space size={16} wrap>
          <Text strong>训练进度</Text>
          <Tag color="green">成功 {runStatusCounts.SUCCESS || 0}</Tag>
          <Tag color="processing">运行中 {(runStatusCounts.RUNNING || 0) + (runStatusCounts.PENDING || 0)}</Tag>
          <Tag color="error">失败 {runStatusCounts.FAILED || 0}</Tag>
          <Button size="small" icon={<ReloadOutlined />} onClick={() => { loadTask(); loadRuns() }}>刷新</Button>
          <Button size="small" type="primary" ghost icon={<PlusOutlined />} onClick={() => setBatchOpen(true)}>再加一组</Button>
        </Space>
      </Card>
      {task && <ProgressTree modelingTaskId={task.id} />}
    </Space>
  )

  const vizColumns = [
    { title: '排名', key: 'rank', width: 70,
      render: (_, __, i) => i === 0 ? <Tag color="gold" icon={<TrophyOutlined />}>1</Tag> : <span>{i + 1}</span> },
    { title: '模型', key: 'model', render: (_, r) => <Tag>{r.params?.model_type || '-'}</Tag> },
    { title: '策略', dataIndex: 'strategy_type', key: 'strategy_type',
      render: (s) => <Tag color="blue">{s}</Tag> },
    { title: task?.objective_metric || '目标值', dataIndex: 'objective_value', key: 'objective_value',
      render: (v) => <code style={{ color: '#2563eb', fontWeight: 600 }}>{typeof v === 'number' ? v.toFixed(4) : '-'}</code> },
    { title: '操作', key: 'actions', width: 200, render: (_, r) => (
      <Space size={2}>
        <Button size="small" type="link" onClick={() => { setInspectorRunId(r.run_id); setInspectorTab('overview') }}>详情</Button>
        <Button size="small" type="link" icon={<BulbOutlined />}
          onClick={() => { setInspectorRunId(r.run_id); setInspectorTab('shap') }}>解释</Button>
        {r.domain_task_id && (
          <Tooltip title="下载模型文件">
            <Button size="small" type="link" icon={<DownloadOutlined />}
              href={runModelDownloadUrl(r.domain_task_id)} target="_blank" />
          </Tooltip>
        )}
      </Space>
    ) },
  ]

  const vizStep = (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      <Card size="small" title={<span><TrophyOutlined style={{ color: '#f59e0b' }} /> 排行榜</span>}
        bodyStyle={{ padding: 0 }}
        extra={<Button size="small" icon={<ReloadOutlined />} onClick={loadRuns}>刷新</Button>}>
        <Table size="small" rowKey="run_id" columns={vizColumns} dataSource={leaderboard}
          pagination={leaderboard.length > 8 ? { pageSize: 8 } : false}
          locale={{ emptyText: <div style={{ padding: 24 }}><Empty description="还没有成功的 Run" /></div> }} />
      </Card>
      {task && <StrategyCompareTab taskId={task.id}
        onInspect={(rid) => { setInspectorRunId(rid); setInspectorTab('shap') }} />}
    </Space>
  )

  const deployStep = <DeployStep task={task} runs={runs} bestRunId={bestRunId} />

  const stepContent = [dataStep, configStep, trainStep, vizStep, deployStep][current]

  return (
    <div style={{ padding: 16 }}>
      <Card bordered={false} bodyStyle={{ padding: '14px 20px' }}
        style={{ marginBottom: 12, boxShadow: '0 1px 2px rgba(15,23,42,0.04)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 16, marginBottom: 14 }}>
          <Space align="center">
            <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/v3/tasks')}>返回工作台</Button>
            <Title level={4} style={{ margin: 0 }}>
              {isNew ? '新建建模任务' : (task?.name || '建模工作流')}
            </Title>
            {task && <Tag color={task.task_type === 'regression' ? 'geekblue' : 'cyan'}>
              {task.task_type === 'regression' ? '回归' : '分类'}</Tag>}
          </Space>
        </div>
        <Steps current={current} items={STEP_ITEMS}
          onChange={(c) => { if (!isNew || c === 0) setCurrent(c) }} />
      </Card>

      {stepContent}

      {/* Footer nav */}
      <Card bordered={false} bodyStyle={{ padding: '12px 20px' }}
        style={{ marginTop: 12, boxShadow: '0 1px 2px rgba(15,23,42,0.04)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <Button disabled={current === 0} icon={<ArrowLeftOutlined />}
            onClick={() => setCurrent(c => Math.max(0, c - 1))}>上一步</Button>
          {current === 0 ? (
            <Button type="primary" loading={saving} onClick={saveDataStep}>
              {isNew ? '创建并继续' : '保存并继续'} <ArrowRightOutlined />
            </Button>
          ) : current < STEP_ITEMS.length - 1 ? (
            <Button type="primary" disabled={isNew && !task}
              onClick={() => setCurrent(c => c + 1)}>下一步 <ArrowRightOutlined /></Button>
          ) : (
            <Button type="primary" onClick={() => navigate('/v3/tasks')}>完成</Button>
          )}
        </div>
      </Card>

      <ExperimentBatchModal open={batchOpen} task={task}
        onClose={() => setBatchOpen(false)}
        onSubmitted={async () => { await loadTask(); await loadRuns(); setCurrent(2) }} />
      <RunInspector open={!!inspectorRunId} runId={inspectorRunId} defaultTab={inspectorTab}
        onClose={() => setInspectorRunId(null)} />
    </div>
  )
}
