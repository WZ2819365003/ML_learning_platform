/**
 * TSMonitor — 时序任务列表页
 * 路由: /ts/tasks
 * 功能: 任务统计 + 筛选表格 + 新建任务弹窗（多步骤向导）
 */
import React, { useEffect, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  Badge, Button, Card, Col, Descriptions, Empty, Form,
  InputNumber, Modal, Popconfirm, Radio, Row,
  Select, Space, Steps, Statistic, Table, Tag, Typography, message,
} from 'antd'
import {
  DeleteOutlined, EyeOutlined,
  LineChartOutlined, PlusOutlined, ReloadOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import { dataApi, tsApi } from '../services/api'
import { formatDateTime } from '../utils/formatters'

const { Title, Text } = Typography
const PAGE_SIZE = 12

const STATUS_META = {
  PENDING: { badge: 'default',    label: '排队中' },
  RUNNING: { badge: 'processing', label: '运行中' },
  SUCCESS: { badge: 'success',    label: '已完成' },
  FAILED:  { badge: 'error',      label: '失败'   },
}

const FREQ_LABELS  = { high: '高频', medium: '中频', low: '低频' }
const FREQ_OPTIONS = [
  { label: '高频（日/小时级）', value: 'high' },
  { label: '中频（周级）', value: 'medium' },
  { label: '低频（月级）', value: 'low' },
]

// ── 新建任务弹窗：3 步骤向导 ──────────────────────────────────────────────────
function CreateTaskModal({ open, onClose, onCreated }) {
  const [step, setStep]         = useState(0)
  const [form]                  = Form.useForm()
  const [datasets, setDatasets] = useState([])
  const [deployments, setDeps]  = useState([])
  const [columns, setColumns]   = useState([])
  const [dsLoading, setDsLoad]  = useState(false)
  const [depLoading, setDepLoad]= useState(false)
  const [submitting, setSub]    = useState(false)
  const [preview, setPreview]   = useState({})

  useEffect(() => {
    if (!open) { setStep(0); form.resetFields(); setPreview({}); setColumns([]); return }
    void loadDatasets()
    void loadDeployments()
  }, [form, open])

  async function loadDatasets() {
    setDsLoad(true)
    try {
      const res = await dataApi.listDatasets({ page: 1, page_size: 100 })
      setDatasets(res.items ?? res.datasets ?? [])
    } finally { setDsLoad(false) }
  }

  async function loadDeployments() {
    setDepLoad(true)
    try {
      const res = await tsApi.listDeployments({ page: 1, page_size: 100 })
      setDeps(res.items ?? [])
    } finally { setDepLoad(false) }
  }

  async function onDatasetChange(id) {
    form.setFieldsValue({ value_column: undefined, time_column: undefined })
    setColumns([])
    if (!id) return
    try {
      const res = await dataApi.previewDataset(id)
      if (res?.rows?.[0]) setColumns(Object.keys(res.rows[0]))
    } catch { /* silent */ }
  }

  async function validateStep() {
    try {
      if (step === 0) await form.validateFields(['dataset_id', 'value_column'])
      if (step === 1) await form.validateFields(['horizon', 'frequency'])  // deployment_id is optional
      return true
    } catch { return false }
  }

  async function goNext() {
    if (!await validateStep()) return
    const vals = form.getFieldsValue()
    const ds   = datasets.find(d => d.id === vals.dataset_id)
    const dep  = deployments.find(d => d.deployment_id === vals.deployment_id)
    setPreview({ ds, dep, vals })
    setStep(s => s + 1)
  }

  async function handleSubmit() {
    setSub(true)
    try {
      const vals = form.getFieldsValue(true)   // true = include all fields even if untouched
      const payload = {
        dataset_id:   vals.dataset_id,
        value_column: vals.value_column,
        time_column:  vals.time_column  || null,
        horizon:      Number(vals.horizon) || 24,
        frequency:    vals.frequency    || 'high',
      }
      // deployment_id is optional — omit if not selected (backend auto-picks default)
      if (vals.deployment_id) payload.deployment_id = vals.deployment_id

      const task = await tsApi.createTask(payload)
      message.success('预测任务已提交！')
      onCreated(task)
      onClose()
    } catch (err) {
      const detail = err?.response?.data?.detail
      // Pydantic 422 returns detail as an array — format it nicely
      const msg = Array.isArray(detail)
        ? detail.map(e => `${String(e.loc?.slice(-1)[0] ?? '')} ${e.msg}`).join('；')
        : (typeof detail === 'string' ? detail : '提交失败，请检查输入')
      message.error(msg)
    } finally { setSub(false) }
  }

  const footer = (
    <Space style={{ justifyContent: 'flex-end', width: '100%' }}>
      <Button onClick={onClose}>取消</Button>
      {step > 0 && <Button onClick={() => setStep(s => s - 1)}>上一步</Button>}
      {step < 2  && <Button type="primary" onClick={goNext}>下一步</Button>}
      {step === 2 && (
        <Button type="primary" loading={submitting} icon={<ThunderboltOutlined />} onClick={handleSubmit}>
          提交任务
        </Button>
      )}
    </Space>
  )

  return (
    <Modal
      title={<Space><LineChartOutlined />新建时序预测任务</Space>}
      open={open} onCancel={onClose} footer={footer} width={620} destroyOnClose
    >
      <Steps
        current={step} size="small" style={{ marginBottom: 28 }}
        items={[{ title: '选择数据' }, { title: '配置参数' }, { title: '确认提交' }]}
      />

      <Form form={form} layout="vertical" initialValues={{ horizon: 24, frequency: 'high' }}>
        {/* Step 0 */}
        {step === 0 && <>
          <Form.Item label="数据集" name="dataset_id" rules={[{ required: true, message: '请选择数据集' }]}>
            <Select showSearch optionFilterProp="label" loading={dsLoading}
              placeholder="选择已上传的数据集" onChange={onDatasetChange}
              options={datasets.map(d => ({ value: d.id, label: d.name }))} />
          </Form.Item>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="时间列（可选）" name="time_column">
                <Select allowClear placeholder="用于图表时间轴"
                  options={columns.map(c => ({ value: c, label: c }))} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="预测目标列" name="value_column" rules={[{ required: true, message: '请选择目标列' }]}>
                <Select placeholder="选择数值列"
                  options={columns.map(c => ({ value: c, label: c }))} />
              </Form.Item>
            </Col>
          </Row>
        </>}

        {/* Step 1 */}
        {step === 1 && <>
          <Form.Item
            label="TimesFM 部署实例（可选）"
            name="deployment_id"
            extra={
              deployments.length === 0
                ? <Text type="secondary" style={{ fontSize: 11 }}>暂无部署实例，将自动使用 Chronos-Small 默认模型</Text>
                : <Text type="secondary" style={{ fontSize: 11 }}>不选择则使用默认 Chronos-Small 模型</Text>
            }
          >
            <Select
              showSearch optionFilterProp="label" loading={depLoading}
              allowClear placeholder="（可选）选择运行中的部署实例"
              options={deployments.map(d => ({
                value: d.deployment_id,
                label: `${d.name} · ${d.backend_label?.split('/').pop()} · ${d.status === 'active' ? '运行中' : '已暂停'}`,
                disabled: d.status !== 'active',
              }))}
            />
          </Form.Item>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="预测步长" name="horizon" rules={[{ required: true }]}>
                <InputNumber min={1} max={512} style={{ width: '100%' }} addonAfter="步" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="数据频率" name="frequency">
                <Radio.Group optionType="button" buttonStyle="solid" options={FREQ_OPTIONS} />
              </Form.Item>
            </Col>
          </Row>
        </>}

        {/* Step 2 */}
        {step === 2 && (
          <Descriptions bordered column={1} size="small">
            <Descriptions.Item label="数据集">{preview.ds?.name ?? preview.vals?.dataset_id}</Descriptions.Item>
            <Descriptions.Item label="目标列">{preview.vals?.value_column}</Descriptions.Item>
            <Descriptions.Item label="时间列">{preview.vals?.time_column ?? '—'}</Descriptions.Item>
            <Descriptions.Item label="部署实例">
              {preview.dep
                ? `${preview.dep.name} · ${preview.dep.backend_label?.split('/').pop()}`
                : <Text type="secondary">默认 Chronos-Small（自动创建）</Text>}
            </Descriptions.Item>
            <Descriptions.Item label="预测步长">{preview.vals?.horizon} 步</Descriptions.Item>
            <Descriptions.Item label="频率">{FREQ_LABELS[preview.vals?.frequency] ?? preview.vals?.frequency}</Descriptions.Item>
          </Descriptions>
        )}
      </Form>
    </Modal>
  )
}

// ── 主页面 ──────────────────────────────────────────────────────────────────────
export default function TSMonitor() {
  const navigate = useNavigate()
  const location = useLocation()

  const [tasks, setTasks]       = useState([])
  const [total, setTotal]       = useState(0)
  const [page, setPage]         = useState(1)
  const [loading, setLoading]   = useState(false)
  const [statusFilter, setFilter] = useState(null)
  const [createOpen, setCreate] = useState(
    () => new URLSearchParams(location.search).get('drawer') === 'create',
  )

  useEffect(() => {
    if (new URLSearchParams(location.search).get('drawer') === 'create') {
      setCreate(true)
    }
  }, [location.search])

  const closeCreate = () => {
    setCreate(false)
    if (new URLSearchParams(location.search).get('drawer') === 'create') {
      navigate('/ts/tasks', { replace: true })
    }
  }
  const refreshTimer            = useRef(null)

  useEffect(() => {
    void fetchTasks(1, statusFilter)
    return () => clearInterval(refreshTimer.current)
  }, [statusFilter])

  useEffect(() => {
    clearInterval(refreshTimer.current)
    const hasActive = tasks.some(t => t.status === 'RUNNING' || t.status === 'PENDING')
    if (hasActive) {
      refreshTimer.current = setInterval(() => void fetchTasks(page, statusFilter), 3000)
    }
    return () => clearInterval(refreshTimer.current)
  }, [tasks, page, statusFilter])

  async function fetchTasks(p, sf) {
    setLoading(true)
    try {
      const params = { page: p, page_size: PAGE_SIZE }
      if (sf) params.status = sf
      const res = await tsApi.listTasks(params)
      setTasks(res.items ?? [])
      setTotal(res.total ?? 0)
      setPage(p)
    } catch { message.error('加载任务列表失败') }
    finally { setLoading(false) }
  }

  async function handleDelete(id) {
    try {
      await tsApi.deleteTask(id)
      message.success('已删除')
      void fetchTasks(page, statusFilter)
    } catch (err) {
      message.error(err?.response?.data?.detail ?? '删除失败')
    }
  }

  const counts = {
    total,
    running: tasks.filter(t => t.status === 'RUNNING' || t.status === 'PENDING').length,
    success: tasks.filter(t => t.status === 'SUCCESS').length,
    failed:  tasks.filter(t => t.status === 'FAILED').length,
  }

  const columns = [
    {
      title: '任务 / 数据集',
      key: 'name',
      render: (_, r) => (
        <Space direction="vertical" size={2}>
          <Text strong style={{ fontSize: 13 }}>{r.dataset_name ?? r.dataset_id?.slice(0, 8)}</Text>
          <Space size={4}>
            <Tag style={{ fontSize: 11, margin: 0 }}>{r.value_column}</Tag>
            <Text type="secondary" style={{ fontSize: 11 }}>
              {r.horizon}步 · {FREQ_LABELS[r.frequency] ?? r.frequency}
            </Text>
          </Space>
        </Space>
      ),
    },
    {
      title: '部署 / 引擎',
      key: 'deploy',
      width: 170,
      render: (_, r) => (
        <Space direction="vertical" size={2}>
          <Text style={{ fontSize: 12 }}>{r.deployment_name ?? '默认'}</Text>
          <Tag color="blue" style={{ fontSize: 10 }}>
            {r.backend_label?.split('/').pop() ?? r.model_name?.split('/').pop() ?? '—'}
          </Tag>
        </Space>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      width: 95,
      render: s => {
        const m = STATUS_META[s] ?? { badge: 'default', label: s }
        return <Badge status={m.badge} text={m.label} />
      },
    },
    {
      title: '提交时间',
      dataIndex: 'created_at',
      width: 145,
      render: v => <Text style={{ fontSize: 12 }}>{formatDateTime(v)}</Text>,
    },
    {
      title: '操作',
      key: 'action',
      width: 110,
      render: (_, r) => (
        <Space size={4} onClick={e => e.stopPropagation()}>
          <Button size="small" icon={<EyeOutlined />}
            onClick={() => navigate(`/ts/tasks/${r.id}`)}>
            详情
          </Button>
          <Popconfirm
            title="确认删除此任务？" disabled={r.status === 'RUNNING'}
            onConfirm={() => void handleDelete(r.id)}>
            <Button size="small" danger icon={<DeleteOutlined />} disabled={r.status === 'RUNNING'} />
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <Space direction="vertical" size={20} style={{ width: '100%' }}>
      {/* Header */}
      <Row align="middle" justify="space-between" wrap={false}>
        <Col>
          <Title level={2} style={{ margin: 0 }}>时序预测任务</Title>
          <Text type="secondary">Amazon Chronos 零样本时序预测 · 任务管理中心</Text>
        </Col>
        <Col>
          <Space>
            <Button icon={<ReloadOutlined />} onClick={() => void fetchTasks(page, statusFilter)}>刷新</Button>
            <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreate(true)}>新建任务</Button>
          </Space>
        </Col>
      </Row>

      {/* Stats */}
      <Row gutter={[12, 12]}>
        {[
          { label: '任务总数', value: counts.total,   color: '#1890ff' },
          { label: '进行中',   value: counts.running,  color: '#faad14' },
          { label: '已完成',   value: counts.success,  color: '#52c41a' },
          { label: '失败',     value: counts.failed,   color: '#ff4d4f' },
        ].map(({ label, value, color }) => (
          <Col xs={12} sm={6} key={label}>
            <Card size="small" style={{ textAlign: 'center', borderTop: `3px solid ${color}` }}>
              <Statistic
                title={<Text type="secondary" style={{ fontSize: 12 }}>{label}</Text>}
                value={value}
                valueStyle={{ color, fontSize: 22, fontWeight: 700 }}
              />
            </Card>
          </Col>
        ))}
      </Row>

      {/* Table */}
      <Card
        title={<Space><LineChartOutlined />任务列表</Space>}
        extra={
          <Select size="small" style={{ width: 110 }} placeholder="全部状态" allowClear
            value={statusFilter}
            onChange={v => { setFilter(v ?? null); void fetchTasks(1, v ?? null) }}
            options={Object.entries(STATUS_META).map(([v, m]) => ({ value: v, label: m.label }))} />
        }
      >
        <Table
          rowKey="id"
          dataSource={tasks}
          columns={columns}
          loading={loading}
          size="middle"
          pagination={{
            current: page, pageSize: PAGE_SIZE, total,
            onChange: p => { setPage(p); void fetchTasks(p, statusFilter) },
            showTotal: t => `共 ${t} 条`, showSizeChanger: false,
          }}
          locale={{ emptyText: <Empty description="暂无任务，点击「新建任务」开始预测" /> }}
          onRow={r => ({
            style: { cursor: 'pointer' },
            onClick: () => navigate(`/ts/tasks/${r.id}`),
          })}
        />
      </Card>

      {/* Create Modal */}
      <CreateTaskModal
        open={createOpen}
        onClose={closeCreate}
        onCreated={task => { void fetchTasks(1, statusFilter); navigate(`/ts/tasks/${task.id}`) }}
      />
    </Space>
  )
}
