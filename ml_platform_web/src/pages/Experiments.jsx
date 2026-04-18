/**
 * V3 Experiments — list, create, and navigate to experiment detail pages.
 * Each experiment groups one or more ExperimentRuns for comparison.
 */

import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Alert, Button, Card, Col, Divider, Empty, Form, Input, InputNumber,
  message, Modal, Popconfirm, Row, Select, Space, Statistic, Table, Tag, Typography,
} from 'antd'
import {
  PlusOutlined, TrophyOutlined, ExperimentOutlined,
  ArrowRightOutlined, DeleteOutlined, ReloadOutlined,
  ThunderboltOutlined, RocketOutlined,
} from '@ant-design/icons'
import { platformExperimentsApi, dataApi } from '../services/api'

const { Title, Text } = Typography

const STATUS_COLOR = {
  CREATED: 'default', RUNNING: 'processing', COMPLETED: 'success', FAILED: 'error',
}
const KIND_LABEL = { single: '单次训练', automl: 'AutoML', grid_search: '网格搜索' }
const METRIC_OPTIONS = [
  { value: 'accuracy', label: '准确率 (Accuracy)' },
  { value: 'f1',       label: 'F1 Score' },
  { value: 'roc_auc',  label: 'ROC AUC' },
  { value: 'rmse',     label: 'RMSE (回归)' },
  { value: 'mae',      label: 'MAE (回归)' },
  { value: 'r2',       label: 'R² Score' },
]

const Experiments = () => {
  const navigate = useNavigate()
  const [experiments, setExperiments] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(false)
  const [createOpen, setCreateOpen] = useState(false)
  const [datasets, setDatasets] = useState([])
  const [form] = Form.useForm()

  /* ── AutoML launch state ── */
  const [automlOpen, setAutomlOpen]       = useState(false)
  const [automlTarget, setAutomlTarget]   = useState(null)  // experiment to launch against
  const [automlLoading, setAutomlLoading] = useState(false)
  const [automlForm] = Form.useForm()

  const PAGE_SIZE = 20

  const fetchExperiments = async (p = 1) => {
    setLoading(true)
    try {
      const res = await platformExperimentsApi.list({ page: p, page_size: PAGE_SIZE })
      setExperiments((res.items || []).map(e => ({ ...e, key: e.id })))
      setTotal(res.total || 0)
      setPage(p)
    } catch { message.error('加载实验列表失败') }
    finally { setLoading(false) }
  }

  const fetchDatasets = async () => {
    try {
      const res = await dataApi.listDatasets({ page_size: 100 })
      setDatasets((res.items || []).map(d => ({ value: d.id, label: d.name })))
    } catch { /* silent */ }
  }

  useEffect(() => {
    fetchExperiments()
    fetchDatasets()
  }, [])

  const handleCreate = async () => {
    try {
      const values = await form.validateFields()
      await platformExperimentsApi.create(values)
      message.success('实验创建成功')
      setCreateOpen(false)
      form.resetFields()
      fetchExperiments(1)
    } catch (err) {
      if (err?.errorFields) return  // validation error
      message.error('创建失败')
    }
  }

  const openAutoml = (experiment) => {
    setAutomlTarget(experiment)
    const isRegression = ['rmse', 'mae', 'r2'].includes(experiment.objective_metric)
    automlForm.resetFields()
    automlForm.setFieldsValue({
      dataset_id: experiment.dataset_id || undefined,
      task_type: isRegression ? 'regression' : 'classification',
      max_candidates: 8,
    })
    setAutomlOpen(true)
  }

  const handleAutomlLaunch = async () => {
    try {
      const values = await automlForm.validateFields()
      setAutomlLoading(true)
      const payload = {
        dataset_id: values.dataset_id,
        target_column: values.target_column.trim(),
        task_type: values.task_type,
        max_candidates: values.max_candidates ?? 8,
        eval_metrics: values.task_type === 'regression'
          ? ['rmse', 'mae', 'r2']
          : ['accuracy', 'f1', 'roc_auc'],
      }
      await platformExperimentsApi.submitAutomlRegistry(automlTarget.id, payload)
      message.success(`AutoML 已启动，最多 ${payload.max_candidates} 个候选模型并发训练`)
      setAutomlOpen(false)
      fetchExperiments(page)
    } catch (err) {
      if (err?.errorFields) return  // validation
      const detail = err?.response?.data?.detail
      message.error(detail ? `启动失败: ${detail}` : '启动 AutoML 失败')
    } finally {
      setAutomlLoading(false)
    }
  }

  const handleDelete = async (id) => {
    try {
      await platformExperimentsApi.delete(id)
      message.success('已删除')
      fetchExperiments(page)
    } catch { message.error('删除失败') }
  }

  // Summary stats
  const runningCount   = experiments.filter(e => e.status === 'RUNNING').length
  const completedCount = experiments.filter(e => e.status === 'COMPLETED').length
  const failedCount    = experiments.filter(e => e.status === 'FAILED').length

  const columns = [
    {
      title: '实验名称',
      dataIndex: 'name',
      render: (name, r) => (
        <Button
          type="link"
          style={{ padding: 0, fontWeight: 600, fontSize: 13 }}
          onClick={() => navigate(`/experiments/${r.id}`)}
        >
          <ExperimentOutlined style={{ marginRight: 6 }} />{name}
        </Button>
      ),
    },
    {
      title: '类型',
      dataIndex: 'kind',
      width: 110,
      render: v => <Tag color="blue" style={{ fontSize: 11 }}>{KIND_LABEL[v] ?? v}</Tag>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      width: 100,
      render: v => <Tag color={STATUS_COLOR[v] ?? 'default'}>{v}</Tag>,
    },
    {
      title: '数据集',
      dataIndex: 'dataset_name',
      width: 160,
      ellipsis: true,
      render: v => v ?? <Text type="secondary">—</Text>,
    },
    {
      title: '优化目标',
      width: 130,
      render: (_, r) => (
        <Text style={{ fontSize: 12, color: '#64748b' }}>
          {r.objective_metric} ({r.objective_direction})
        </Text>
      ),
    },
    {
      title: '最优 Run',
      dataIndex: 'best_run_id',
      width: 100,
      render: (v, r) => v
        ? <Button type="link" size="small" icon={<TrophyOutlined />}
            onClick={() => navigate(`/experiments/${r.id}`)}>查看</Button>
        : <Text type="secondary" style={{ fontSize: 12 }}>—</Text>,
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      width: 150,
      render: v => v
        ? <Text style={{ fontSize: 11, color: '#94a3b8' }}>{new Date(v).toLocaleString('zh-CN', { hour12: false })}</Text>
        : '—',
    },
    {
      title: '操作',
      width: 200,
      render: (_, r) => (
        <Space wrap>
          <Button
            size="small" type="text" icon={<ArrowRightOutlined />}
            onClick={() => navigate(`/experiments/${r.id}`)}
          >详情</Button>
          {(r.kind === 'automl' || r.kind === 'grid_search') && r.status !== 'RUNNING' && (
            <Button
              size="small"
              type="primary"
              ghost
              icon={<ThunderboltOutlined />}
              onClick={() => openAutoml(r)}
            >
              AutoML
            </Button>
          )}
          <Popconfirm
            title="确认删除该实验及所有 Run？"
            onConfirm={() => handleDelete(r.id)}
            okText="删除" cancelText="取消"
            disabled={r.status === 'RUNNING'}
          >
            <Button size="small" type="text" danger icon={<DeleteOutlined />}
              disabled={r.status === 'RUNNING'}>删除</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <div style={{ padding: '20px 0' }}>
      <Space direction="vertical" size={20} style={{ width: '100%', display: 'flex' }}>

        {/* Hero */}
        <div className="hero-banner">
          <Row align="middle" justify="space-between">
            <Col>
              <Title level={2} style={{ color: '#fff', margin: 0, fontWeight: 800 }}>实验管理</Title>
              <Text style={{ color: 'rgba(255,255,255,0.65)', fontSize: 14 }}>
                组织多次训练 Run，对比模型性能，自动筛选最优
              </Text>
            </Col>
            <Col>
              <Button
                type="primary" icon={<PlusOutlined />} size="large"
                style={{ borderRadius: 10, fontWeight: 600 }}
                onClick={() => setCreateOpen(true)}
              >
                新建实验
              </Button>
            </Col>
          </Row>
        </div>

        {/* Stat cards */}
        <Row gutter={[16, 16]}>
          {[
            { label: '总实验数', value: total,          color: '#2563eb' },
            { label: '运行中',   value: runningCount,   color: '#3b82f6' },
            { label: '已完成',   value: completedCount, color: '#10b981' },
            { label: '失败',     value: failedCount,    color: '#ef4444' },
          ].map((s, i) => (
            <Col xs={12} sm={6} key={i}>
              <Card bodyStyle={{ padding: '16px 20px' }}>
                <Statistic
                  title={<span style={{ color: s.color, fontWeight: 700, fontSize: 12 }}>{s.label}</span>}
                  value={s.value}
                  valueStyle={{ color: s.color, fontSize: 24, fontWeight: 800 }}
                />
              </Card>
            </Col>
          ))}
        </Row>

        {/* Table */}
        <Card
          title="实验列表"
          extra={<Button icon={<ReloadOutlined />} onClick={() => fetchExperiments(1)}>刷新</Button>}
        >
          {experiments.length === 0 && !loading
            ? <Empty
                description={<span>还没有实验 — <Button type="link" onClick={() => setCreateOpen(true)}>新建一个</Button></span>}
              />
            : <Table
                rowKey="id"
                dataSource={experiments}
                columns={columns}
                loading={loading}
                size="small"
                pagination={{
                  total,
                  current: page,
                  pageSize: PAGE_SIZE,
                  onChange: fetchExperiments,
                  showTotal: t => `共 ${t} 条`,
                  showSizeChanger: false,
                }}
              />
          }
        </Card>

      </Space>

      {/* AutoML launch modal */}
      <Modal
        title={
          <Space>
            <RocketOutlined style={{ color: '#6366f1' }} />
            <span>一键 AutoML — {automlTarget?.name}</span>
          </Space>
        }
        open={automlOpen}
        onOk={handleAutomlLaunch}
        onCancel={() => setAutomlOpen(false)}
        okText={<span><ThunderboltOutlined /> 启动训练</span>}
        okButtonProps={{ loading: automlLoading, style: { background: '#6366f1', borderColor: '#6366f1' } }}
        cancelText="取消"
        width={560}
        destroyOnClose
      >
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message="平台将从内置候选库中选取算法，并发训练多个模型，自动排名选出最优。"
        />
        <Form form={automlForm} layout="vertical">
          <Form.Item
            label="数据集"
            name="dataset_id"
            rules={[{ required: true, message: '请选择数据集' }]}
          >
            <Select
              showSearch
              placeholder="选择训练数据集"
              options={datasets}
              filterOption={(input, opt) => opt.label.toLowerCase().includes(input.toLowerCase())}
            />
          </Form.Item>
          <Form.Item
            label="目标列（标签列）"
            name="target_column"
            rules={[{ required: true, message: '请输入目标列名' }]}
          >
            <Input placeholder="如：label、target、survived" />
          </Form.Item>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="任务类型" name="task_type" initialValue="classification">
                <Select
                  options={[
                    { value: 'classification', label: '分类 (Classification)' },
                    { value: 'regression',     label: '回归 (Regression)' },
                  ]}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="最大候选数"
                name="max_candidates"
                initialValue={8}
                extra="越多越准，但耗时更长"
              >
                <InputNumber min={1} max={20} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>
          <Divider style={{ margin: '8px 0 16px' }} />
          <div style={{ color: '#64748b', fontSize: 12 }}>
            <b>候选模型池包括：</b>逻辑回归、随机森林、ExtraTrees、梯度提升树 (GBDT)、XGBoost、LightGBM
            — 每种算法搭配 2~3 组超参，共约 12 个候选（可通过最大候选数截断）。
          </div>
        </Form>
      </Modal>

      {/* Create experiment modal */}
      <Modal
        title="新建实验"
        open={createOpen}
        onOk={handleCreate}
        onCancel={() => { setCreateOpen(false); form.resetFields() }}
        okText="创建"
        cancelText="取消"
        width={540}
      >
        <Form form={form} layout="vertical" style={{ marginTop: 16 }}>
          <Form.Item label="实验名称" name="name" rules={[{ required: true, message: '请输入实验名称' }]}>
            <Input placeholder="如：泰坦尼克号分类实验" />
          </Form.Item>
          <Form.Item label="描述" name="description">
            <Input.TextArea rows={2} placeholder="可选说明" />
          </Form.Item>
          <Form.Item label="关联数据集" name="dataset_id">
            <Select
              allowClear showSearch
              placeholder="选择数据集（可选）"
              options={datasets}
              filterOption={(input, opt) => opt.label.toLowerCase().includes(input.toLowerCase())}
            />
          </Form.Item>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="优化指标" name="objective_metric" initialValue="accuracy">
                <Select options={METRIC_OPTIONS} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="优化方向" name="objective_direction" initialValue="max">
                <Select options={[{ value: 'max', label: '最大化' }, { value: 'min', label: '最小化' }]} />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item label="实验类型" name="kind" initialValue="single">
            <Select options={Object.entries(KIND_LABEL).map(([k, v]) => ({ value: k, label: v }))} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default Experiments
