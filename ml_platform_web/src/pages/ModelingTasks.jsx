import React, { useState, useEffect, useCallback } from 'react'
import {
  Card, Button, Table, Tag, Space, Modal, Form, Input, Select, Tooltip,
  message, Popconfirm, Row, Col, Empty, Divider,
} from 'antd'
import {
  PlusOutlined, AppstoreOutlined, ExperimentOutlined, TrophyOutlined,
  ReloadOutlined, DeleteOutlined, EyeOutlined, CheckCircleFilled,
  ClockCircleFilled, CloseCircleFilled, FireOutlined,
} from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { modelingTaskApi, dataApi } from '../services/api'

const STATUS_META = {
  CREATED:   { color: 'default', icon: <ClockCircleFilled />, label: '待启动' },
  RUNNING:   { color: 'processing', icon: <ClockCircleFilled spin />, label: '运行中' },
  COMPLETED: { color: 'success', icon: <CheckCircleFilled />, label: '已完成' },
  FAILED:    { color: 'error', icon: <CloseCircleFilled />, label: '失败' },
  ARCHIVED:  { color: 'default', icon: null, label: '已归档' },
}

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

const _objectiveDirection = (metric) =>
  ['rmse', 'mae', 'mse', 'mape'].includes(metric) ? 'min' : 'max'

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
  const [datasets, setDatasets] = useState([])
  const [pagination, setPagination] = useState({ page: 1, pageSize: 10 })
  const [createOpen, setCreateOpen] = useState(false)
  const [createLoading, setCreateLoading] = useState(false)
  // columnInfo: { columnName: { dtype, missing_rate, missing_count } }
  const [columnInfo, setColumnInfo] = useState(null)
  const [columnsLoading, setColumnsLoading] = useState(false)
  const [form] = Form.useForm()

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

  const loadDatasets = useCallback(async () => {
    try {
      const resp = await dataApi.listDatasets({ page: 1, page_size: 100 })
      setDatasets(resp?.items || resp?.datasets || [])
    } catch {/* non-fatal */}
  }, [])

  useEffect(() => { load() }, [load])
  useEffect(() => { loadDatasets() }, [loadDatasets])

  const taskTypeWatch = Form.useWatch('task_type', form) || 'classification'
  const datasetIdWatch = Form.useWatch('dataset_id', form)

  // When the user picks a dataset in the create modal, fetch its column headers
  // so the 目标列 field becomes a dropdown.  We use the preview endpoint (which
  // already returns columns_info: { col: {dtype, missing_rate, ...} }) rather
  // than adding a new route — the payload is tiny when rows=1.
  useEffect(() => {
    if (!createOpen || !datasetIdWatch) { setColumnInfo(null); return }
    let cancelled = false
    setColumnsLoading(true)
    dataApi.previewDataset(datasetIdWatch)
      .then((resp) => { if (!cancelled) setColumnInfo(resp?.columns_info || null) })
      .catch(() => { if (!cancelled) setColumnInfo(null) })
      .finally(() => { if (!cancelled) setColumnsLoading(false) })
    // Clear any previously selected target_column when the dataset changes
    form.setFieldValue('target_column', undefined)
    return () => { cancelled = true }
  }, [datasetIdWatch, createOpen, form])

  // Build target-column select options, filtered by task type.  For regression
  // we only surface numeric columns (float/int); for classification we allow
  // everything but sort numeric first (common convention — label is an int
  // class id or a string).  Also surface missing_rate as a subtle warning.
  const targetColumnOptions = React.useMemo(() => {
    if (!columnInfo) return []
    const entries = Object.entries(columnInfo)
    const isNumeric = (dt) => /int|float|double|number/i.test(String(dt))
    const filtered = taskTypeWatch === 'regression'
      ? entries.filter(([, m]) => isNumeric(m.dtype))
      : entries
    filtered.sort((a, b) => {
      const an = isNumeric(a[1].dtype) ? 0 : 1
      const bn = isNumeric(b[1].dtype) ? 0 : 1
      return an - bn || a[0].localeCompare(b[0])
    })
    return filtered.map(([col, meta]) => ({
      value: col,
      label: (
        <Space size={6}>
          <span style={{ fontWeight: 500 }}>{col}</span>
          <Tag style={{ fontSize: 10, margin: 0 }} color={isNumeric(meta.dtype) ? 'blue' : 'default'}>
            {meta.dtype}
          </Tag>
          {meta.missing_rate > 0 && (
            <Tag style={{ fontSize: 10, margin: 0 }} color="warning">
              缺失 {(meta.missing_rate * 100).toFixed(0)}%
            </Tag>
          )}
        </Space>
      ),
    }))
  }, [columnInfo, taskTypeWatch])

  const handleCreate = async () => {
    const values = await form.validateFields()
    setCreateLoading(true)
    try {
      const payload = {
        name: values.name,
        description: values.description,
        dataset_id: values.dataset_id || null,
        target_column: values.target_column || null,
        task_type: values.task_type,
        objective_metric: values.objective_metric,
        objective_direction: _objectiveDirection(values.objective_metric),
      }
      const result = await modelingTaskApi.create(payload)
      message.success(`建模任务 "${result.name}" 已创建`)
      setCreateOpen(false)
      form.resetFields()
      await load()
      navigate(`/v3/tasks/${result.id}`)
    } catch (err) {
      if (err?.errorFields) return  // form validation
      message.error(err?.response?.data?.detail || '创建失败')
    } finally {
      setCreateLoading(false)
    }
  }

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
    const byStatus = data.items.reduce((acc, t) => {
      acc[t.status] = (acc[t.status] || 0) + 1
      return acc
    }, {})
    return {
      total: data.total,
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
      width: 120,
      fixed: 'right',
      render: (_, row) => (
        <Space size={4}>
          <Tooltip title="查看详情">
            <Button size="small" type="primary" ghost icon={<EyeOutlined />}
              onClick={() => navigate(`/v3/tasks/${row.id}`)}>详情</Button>
          </Tooltip>
          <Popconfirm
            title="确认删除此建模任务？"
            description="其下所有实验与 Run 都会被级联清理。"
            onConfirm={() => handleDelete(row.id)}
          >
            <Button size="small" danger icon={<DeleteOutlined />} disabled={row.status === 'RUNNING'} />
          </Popconfirm>
        </Space>
      ),
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
              以"任务"为单位组织建模流程：一个任务可挂多组实验（基线/网格/贝叶斯），自动汇总最佳 Run。
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
            <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateOpen(true)}>
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
          dataSource={data.items}
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
            total: data.total,
            showSizeChanger: true,
            showQuickJumper: true,
            pageSizeOptions: ['10', '20', '50'],
            showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条 / 共 ${total} 条`,
            onChange: (page, pageSize) => setPagination({ page, pageSize }),
            style: { padding: '12px 16px', margin: 0 },
          }}
        />
      </Card>

      {/* ── Create modal ────────────────────────────────────────────────── */}
      <Modal
        title="新建建模任务"
        open={createOpen}
        onCancel={() => { setCreateOpen(false); form.resetFields() }}
        onOk={handleCreate}
        confirmLoading={createLoading}
        width={640}
        okText="创建"
        cancelText="取消"
      >
        <Divider style={{ margin: '0 0 16px' }} />
        <Form form={form} layout="vertical" requiredMark="optional" initialValues={{
          task_type: 'classification',
          objective_metric: 'accuracy',
        }}>
          <Form.Item name="name" label="任务名称" rules={[{ required: true, min: 2, message: '至少 2 个字符' }]}>
            <Input placeholder="例: 客户流失预测 v1" />
          </Form.Item>
          <Form.Item name="description" label="描述">
            <Input.TextArea rows={2} placeholder="选填，简述本次建模目标（如业务场景、期望指标）" />
          </Form.Item>
          <Row gutter={12}>
            <Col span={14}>
              <Form.Item name="dataset_id" label="数据集">
                <Select placeholder="选择数据集（可稍后在实验批次中指定）"
                  options={datasets.map(d => ({ value: d.id, label: `${d.name} (${d.row_count || '?'} 行)` }))}
                  allowClear showSearch
                  filterOption={(input, opt) => String(opt.label).toLowerCase().includes(input.toLowerCase())}
                />
              </Form.Item>
            </Col>
            <Col span={10}>
              <Form.Item
                name="target_column"
                label={
                  <Space size={4}>
                    <span>目标列</span>
                    {columnInfo && (
                      <span style={{ fontSize: 11, color: '#64748b' }}>
                        (共 {Object.keys(columnInfo).length} 列)
                      </span>
                    )}
                  </Space>
                }
                tooltip={!datasetIdWatch ? '请先选择数据集' : undefined}
              >
                <Select
                  placeholder={
                    !datasetIdWatch ? '请先选择数据集'
                    : columnsLoading ? '正在读取列头…'
                    : columnInfo ? '从列表中选择目标列'
                    : '无法加载列信息'
                  }
                  disabled={!datasetIdWatch}
                  loading={columnsLoading}
                  options={targetColumnOptions}
                  showSearch optionFilterProp="value"
                  filterOption={(input, opt) =>
                    String(opt.value).toLowerCase().includes(input.toLowerCase())
                  }
                  notFoundContent={columnsLoading ? '加载中...' : '无可用列（请检查任务类型是否匹配数据集）'}
                  allowClear
                />
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={12}>
            <Col span={10}>
              <Form.Item name="task_type" label="任务类型" rules={[{ required: true }]}>
                <Select
                  options={[
                    { value: 'classification', label: '分类' },
                    { value: 'regression', label: '回归' },
                  ]}
                  onChange={() => {
                    const t = form.getFieldValue('task_type')
                    form.setFieldValue('objective_metric', t === 'regression' ? 'rmse' : 'accuracy')
                  }}
                />
              </Form.Item>
            </Col>
            <Col span={14}>
              <Form.Item name="objective_metric" label="优化目标指标" rules={[{ required: true }]}>
                <Select options={OBJECTIVE_PRESETS[taskTypeWatch]} />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>
    </div>
  )
}
