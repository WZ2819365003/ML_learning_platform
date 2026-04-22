/**
 * TrainingPlans — list + editor for reusable training-plan templates.
 *
 * A plan captures a reusable recipe of (task_type × strategy × models ×
 * search_space × budget × eval_metrics) — dataset-agnostic.  When a user
 * launches a new experiment batch on a ModelingTask, they can pick a plan
 * and the batch-config form gets prefilled.
 *
 * The page is one screen:
 *   - Top action bar (新建 · 刷新 · task_type filter)
 *   - Table of plans with inline 编辑 / 删除
 *   - Right-hand Drawer for create/edit (loads tuning-spaces registry so the
 *     "可选模型" list is data-driven rather than hard-coded)
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Card, Table, Button, Space, Tag, Popconfirm, Drawer, Form, Input, Select,
  Switch, InputNumber, Divider, message, Typography, Tooltip, Empty, Segmented,
  Row, Col,
} from 'antd'
import {
  PlusOutlined, ReloadOutlined, EditOutlined, DeleteOutlined,
  ThunderboltOutlined, InfoCircleOutlined, CopyOutlined,
} from '@ant-design/icons'
import { trainingPlansApi, modelingTaskApi, dlApi } from '../services/api'
import DLConfigPanel from '../components/workbench/DLConfigPanel'

const { Text, Paragraph } = Typography

const STRATEGY_LABELS = {
  baseline:        { label: '基线（默认超参）',      color: 'default' },
  grid_search:     { label: '网格搜索',               color: 'blue' },
  bayesian_search: { label: '贝叶斯搜索（Optuna）',   color: 'purple' },
}

const TASK_TYPE_OPTIONS = [
  { value: 'classification', label: '分类' },
  { value: 'regression',     label: '回归' },
]

const METRIC_PRESETS = {
  classification: [
    { value: 'accuracy', label: 'Accuracy' },
    { value: 'f1',       label: 'F1' },
    { value: 'precision', label: 'Precision' },
    { value: 'recall',   label: 'Recall' },
    { value: 'roc_auc',  label: 'ROC-AUC' },
  ],
  regression: [
    { value: 'rmse', label: 'RMSE' },
    { value: 'mae',  label: 'MAE' },
    { value: 'mse',  label: 'MSE' },
    { value: 'r2',   label: 'R²' },
  ],
}

const _objectiveDirection = (metric) =>
  ['rmse', 'mae', 'mse', 'mape'].includes(metric) ? 'min' : 'max'

const FAMILY_OPTIONS = [
  { label: 'ML 经典', value: 'ml' },
  { label: 'DL 深度', value: 'dl' },
  { label: '混合',    value: 'mixed' },
]

// Build a dl_config entry from a registry spec using its defaults.
const _buildDefaultDLConfig = (modelSpec, optimizerParams, trainParams) => {
  const fromSpecs = (specs) =>
    Object.fromEntries((specs || []).map(s => [s.name, s.default]))
  return {
    arch:  fromSpecs(modelSpec?.arch_params),
    opt:   fromSpecs(optimizerParams),
    train: fromSpecs(trainParams),
  }
}

export default function TrainingPlans() {
  const [loading, setLoading] = useState(false)
  const [data, setData] = useState({ items: [], total: 0 })
  const [taskType, setTaskType] = useState('all')
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [editingId, setEditingId] = useState(null)
  const [saving, setSaving] = useState(false)
  const [tuningSpaces, setTuningSpaces] = useState({ classification: {}, regression: {} })
  const [dlRegistry, setDlRegistry] = useState({
    models: [], optimizer_params: [], train_params: [],
  })
  const [form] = Form.useForm()

  const formTaskType   = Form.useWatch('task_type',    form) || 'classification'
  const formFamily     = Form.useWatch('model_family', form) || 'ml'
  const formSelected   = Form.useWatch('selected_models', form) || []
  const formDlConfig   = Form.useWatch('dl_config',    form) || {}
  const availableModels = tuningSpaces[formTaskType] || {}

  // DL models filtered by current task_type — index by id for quick lookup.
  const dlModelsForTask = useMemo(
    () => (dlRegistry.models || []).filter(m =>
      !m.task_types || m.task_types.includes(formTaskType)
    ),
    [dlRegistry.models, formTaskType],
  )
  const dlModelById = useMemo(
    () => Object.fromEntries(dlModelsForTask.map(m => [m.id, m])),
    [dlModelsForTask],
  )

  // Compose the Select options based on selected model_family.
  const modelOptions = useMemo(() => {
    const mlOpts = Object.entries(availableModels).map(([key, meta]) => ({
      value: key,
      label: (
        <Space>
          <Tag color="blue" style={{ margin: 0 }}>ML</Tag>
          <span>{meta?.display_name || key}</span>
          <code style={{ fontSize: 10, color: '#64748b' }}>{key}</code>
        </Space>
      ),
    }))
    const dlOpts = dlModelsForTask.map(m => ({
      value: m.id,
      label: (
        <Space>
          <Tag color="purple" style={{ margin: 0 }}>DL</Tag>
          <span>{m.display_name || m.id}</span>
          <code style={{ fontSize: 10, color: '#64748b' }}>{m.id}</code>
        </Space>
      ),
    }))
    if (formFamily === 'ml')  return mlOpts
    if (formFamily === 'dl')  return dlOpts
    return [
      { label: 'ML 经典模型', options: mlOpts },
      { label: 'DL 深度模型', options: dlOpts },
    ]
  }, [availableModels, dlModelsForTask, formFamily])

  // Of the currently-selected tokens, which are DL? (need a per-model panel)
  const selectedDlTokens = useMemo(
    () => (formSelected || []).filter(t => !!dlModelById[t]),
    [formSelected, dlModelById],
  )

  const loadPlans = useCallback(async () => {
    setLoading(true)
    try {
      const params = taskType === 'all' ? {} : { task_type: taskType }
      const resp = await trainingPlansApi.list(params)
      setData(resp)
    } catch (e) {
      message.error(e?.response?.data?.detail || '加载失败')
    } finally {
      setLoading(false)
    }
  }, [taskType])

  const loadSpaces = useCallback(async () => {
    try {
      const [cls, reg] = await Promise.all([
        modelingTaskApi.tuningSpaces('classification'),
        modelingTaskApi.tuningSpaces('regression'),
      ])
      setTuningSpaces({
        classification: cls?.models || {},
        regression:     reg?.models || {},
      })
    } catch {/* non-fatal */}
  }, [])

  const loadDlRegistry = useCallback(async () => {
    try {
      const resp = await dlApi.listModels()
      setDlRegistry({
        models:           resp?.models           || [],
        optimizer_params: resp?.optimizer_params || [],
        train_params:     resp?.train_params     || [],
      })
    } catch {/* non-fatal — DL just won't appear in the picker */}
  }, [])

  useEffect(() => { loadPlans() }, [loadPlans])
  useEffect(() => { loadSpaces() }, [loadSpaces])
  useEffect(() => { loadDlRegistry() }, [loadDlRegistry])

  const handleCreate = () => {
    setEditingId(null)
    form.resetFields()
    form.setFieldsValue({
      task_type: 'classification',
      strategy_type: 'baseline',
      model_family: 'ml',
      selected_models: [],
      dl_config: {},
      eval_metrics: ['accuracy', 'f1'],
      default_objective_metric: 'accuracy',
      budget_config: { max_trials: 20, test_size: 0.2 },
    })
    setDrawerOpen(true)
  }

  const handleEdit = async (plan) => {
    setEditingId(plan.id)
    form.resetFields()
    form.setFieldsValue({
      name: plan.name,
      description: plan.description,
      task_type: plan.task_type,
      strategy_type: plan.strategy_type,
      model_family: plan.model_family || 'ml',
      selected_models: plan.selected_models || [],
      dl_config: plan.dl_config || {},
      eval_metrics: plan.eval_metrics || [],
      default_objective_metric: plan.default_objective_metric,
      budget_config: plan.budget_config || {},
    })
    setDrawerOpen(true)
  }

  const handleDuplicate = async (plan) => {
    try {
      const { id: _id, created_at, updated_at, use_count, last_used_at, ...rest } = plan
      await trainingPlansApi.create({ ...rest, name: `${plan.name} (副本)` })
      message.success('已复制')
      await loadPlans()
    } catch (e) {
      message.error(e?.response?.data?.detail || '复制失败')
    }
  }

  const handleDelete = async (plan) => {
    try {
      await trainingPlansApi.remove(plan.id)
      message.success('已删除')
      await loadPlans()
    } catch (e) {
      message.error(e?.response?.data?.detail || '删除失败')
    }
  }

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields()
      // Only keep dl_config entries for currently-selected DL tokens so stale
      // entries (from deselected models) don't leak into the saved plan.
      const cleanedDlConfig = Object.fromEntries(
        Object.entries(values.dl_config || {}).filter(
          ([k]) => (values.selected_models || []).includes(k),
        ),
      )
      const payload = {
        name: values.name,
        description: values.description,
        task_type: values.task_type,
        strategy_type: values.strategy_type,
        model_family: values.model_family || 'ml',
        selected_models: values.selected_models,
        dl_config: cleanedDlConfig,
        eval_metrics: values.eval_metrics,
        default_objective_metric: values.default_objective_metric,
        default_objective_direction: _objectiveDirection(values.default_objective_metric),
        budget_config: values.budget_config,
        // search_space overrides are not exposed in this MVP UI — the plan
        // uses the registry defaults until the user edits it via a JSON API.
      }
      setSaving(true)
      if (editingId) {
        await trainingPlansApi.update(editingId, payload)
        message.success('已更新')
      } else {
        await trainingPlansApi.create(payload)
        message.success('已创建')
      }
      setDrawerOpen(false)
      await loadPlans()
    } catch (err) {
      if (err?.errorFields) return  // antd validation — already rendered
      message.error(err?.response?.data?.detail || '保存失败')
    } finally {
      setSaving(false)
    }
  }

  const columns = useMemo(() => [
    {
      title: '方案名称', dataIndex: 'name', key: 'name',
      render: (v, row) => (
        <div>
          <div style={{ fontWeight: 600, color: '#0f172a' }}>{v}</div>
          {row.description && (
            <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{row.description}</div>
          )}
        </div>
      ),
    },
    {
      title: '任务类型', dataIndex: 'task_type', width: 100,
      render: (v) => <Tag color={v === 'regression' ? 'geekblue' : 'cyan'}>
        {v === 'regression' ? '回归' : '分类'}
      </Tag>,
    },
    {
      title: '模型族', dataIndex: 'model_family', width: 90,
      render: (v) => {
        const family = v || 'ml'
        const meta = {
          ml:     { label: 'ML',    color: 'blue' },
          dl:     { label: 'DL',    color: 'purple' },
          mixed:  { label: '混合',  color: 'volcano' },
        }[family] || { label: family, color: 'default' }
        return <Tag color={meta.color}>{meta.label}</Tag>
      },
    },
    {
      title: '策略', dataIndex: 'strategy_type', width: 180,
      render: (v) => {
        const m = STRATEGY_LABELS[v] || { label: v, color: 'default' }
        return <Tag color={m.color}>{m.label}</Tag>
      },
    },
    {
      title: '模型', dataIndex: 'selected_models', key: 'models',
      render: (v) => (
        <Space wrap size={[4, 4]}>
          {(v || []).map(m => <Tag key={m} style={{ fontSize: 11 }}>{m}</Tag>)}
        </Space>
      ),
    },
    {
      title: '优化目标', dataIndex: 'default_objective_metric', width: 130,
      render: (v, row) => v ? (
        <Space size={4}>
          <code style={{ fontSize: 11 }}>{v}</code>
          <Tag color={row.default_objective_direction === 'min' ? 'orange' : 'green'} style={{ fontSize: 10 }}>
            {row.default_objective_direction === 'min' ? '越低越好' : '越高越好'}
          </Tag>
        </Space>
      ) : '-',
    },
    {
      title: '使用', dataIndex: 'use_count', width: 70, align: 'center',
      render: (v) => v > 0 ? <Tag color="blue">{v}</Tag> : <Text type="secondary">0</Text>,
    },
    {
      title: '操作', key: 'actions', width: 200, fixed: 'right',
      render: (_, row) => (
        <Space size={4}>
          <Tooltip title="编辑"><Button size="small" type="text" icon={<EditOutlined />}
            onClick={() => handleEdit(row)} /></Tooltip>
          <Tooltip title="复制"><Button size="small" type="text" icon={<CopyOutlined />}
            onClick={() => handleDuplicate(row)} /></Tooltip>
          <Popconfirm title="删除此方案？" onConfirm={() => handleDelete(row)} okText="删除" cancelText="取消">
            <Button size="small" type="text" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ], [])

  return (
    <div style={{ padding: '20px 4px' }}>
      <Card
        bordered={false}
        style={{ borderRadius: 16, boxShadow: '0 1px 4px rgba(15,23,42,0.06)' }}
        bodyStyle={{ padding: 0 }}
      >
        <div style={{ padding: '16px 20px', borderBottom: '1px solid #f1f5f9' }}>
          <Row justify="space-between" align="middle">
            <Col>
              <Space size={12}>
                <ThunderboltOutlined style={{ fontSize: 22, color: '#2563eb' }} />
                <div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: '#0f172a' }}>训练方案</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>
                    预设训练配置（模型 + 超参搜索空间 + 评估指标），建模任务可直接套用
                  </div>
                </div>
              </Space>
            </Col>
            <Col>
              <Space>
                <Segmented
                  size="middle"
                  value={taskType}
                  onChange={setTaskType}
                  options={[
                    { label: '全部', value: 'all' },
                    { label: '分类', value: 'classification' },
                    { label: '回归', value: 'regression' },
                  ]}
                />
                <Button icon={<ReloadOutlined />} onClick={loadPlans} loading={loading}>刷新</Button>
                <Button type="primary" icon={<PlusOutlined />} onClick={handleCreate}>
                  新建方案
                </Button>
              </Space>
            </Col>
          </Row>
        </div>

        <Table
          rowKey="id"
          loading={loading}
          dataSource={data.items}
          columns={columns}
          pagination={{ pageSize: 10, showSizeChanger: true }}
          locale={{ emptyText: <Empty image={Empty.PRESENTED_IMAGE_SIMPLE}
            description="暂无训练方案 — 点击右上角新建一个" /> }}
          style={{ padding: '0 4px' }}
        />
      </Card>

      {/* ─────── Create / Edit drawer ─────── */}
      <Drawer
        title={editingId ? '编辑训练方案' : '新建训练方案'}
        width={620}
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        destroyOnClose
        footer={
          <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
            <Button onClick={() => setDrawerOpen(false)}>取消</Button>
            <Button type="primary" loading={saving} onClick={handleSubmit}>
              {editingId ? '保存' : '创建'}
            </Button>
          </Space>
        }
      >
        <Form form={form} layout="vertical">
          <Form.Item name="name" label="方案名称"
            rules={[{ required: true, min: 2, message: '至少 2 个字符' }]}>
            <Input placeholder="例: 快速分类筛选 (LR + XGB 默认超参)" />
          </Form.Item>

          <Form.Item name="description" label="描述">
            <Input.TextArea rows={2} placeholder="说明该方案的适用场景" />
          </Form.Item>

          <Row gutter={12}>
            <Col span={12}>
              <Form.Item name="task_type" label="任务类型" rules={[{ required: true }]}>
                <Select
                  options={TASK_TYPE_OPTIONS}
                  onChange={() => {
                    // clear model selection + switch default metric when task_type changes
                    form.setFieldsValue({
                      selected_models: [],
                      dl_config: {},
                      eval_metrics: [],
                      default_objective_metric: undefined,
                    })
                  }}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="strategy_type" label={
                <Space size={4}>
                  <span>调优策略</span>
                  <Tooltip title="baseline=只跑默认超参；grid_search=网格遍历；bayesian_search=Optuna 贝叶斯搜索">
                    <InfoCircleOutlined style={{ color: '#94a3b8', fontSize: 12 }} />
                  </Tooltip>
                </Space>
              } rules={[{ required: true }]}>
                <Select options={Object.entries(STRATEGY_LABELS).map(([v, m]) => ({
                  value: v, label: m.label,
                }))} />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item name="model_family" label={
            <Space size={4}>
              <span>模型族</span>
              <Tooltip title="ML=sklearn/XGB/LGB 等经典模型；DL=基于 PyTorch 的深度模型；混合=同时包含两类。DL 当前仅支持 baseline 策略（grid/bayesian 会自动降级）。">
                <InfoCircleOutlined style={{ color: '#94a3b8', fontSize: 12 }} />
              </Tooltip>
            </Space>
          } rules={[{ required: true }]}>
            <Segmented
              options={FAMILY_OPTIONS}
              onChange={(nextFamily) => {
                // When family narrows, drop tokens that no longer belong.
                const currentSelected = form.getFieldValue('selected_models') || []
                const currentDlCfg    = form.getFieldValue('dl_config') || {}
                let filtered = currentSelected
                if (nextFamily === 'ml') {
                  filtered = currentSelected.filter(t => !dlModelById[t])
                } else if (nextFamily === 'dl') {
                  filtered = currentSelected.filter(t => !!dlModelById[t])
                }
                const cleanedDl = Object.fromEntries(
                  Object.entries(currentDlCfg).filter(([k]) => filtered.includes(k)),
                )
                form.setFieldsValue({ selected_models: filtered, dl_config: cleanedDl })
              }}
            />
          </Form.Item>

          <Form.Item name="selected_models" label={
            <Space size={4}>
              <span>候选模型</span>
              <Text type="secondary" style={{ fontSize: 11 }}>
                （可多选，来自注册表）
              </Text>
            </Space>
          } rules={[{ required: true, message: '至少选择一个模型' }]}>
            <Select
              mode="multiple"
              placeholder="选择参与训练的模型"
              options={modelOptions}
              onChange={(nextTokens) => {
                // Backfill dl_config defaults for newly selected DL tokens,
                // drop entries for removed tokens.
                const prevDl = form.getFieldValue('dl_config') || {}
                const nextDl = { ...prevDl }
                for (const t of nextTokens) {
                  if (dlModelById[t] && !nextDl[t]) {
                    nextDl[t] = _buildDefaultDLConfig(
                      dlModelById[t],
                      dlRegistry.optimizer_params,
                      dlRegistry.train_params,
                    )
                  }
                }
                for (const k of Object.keys(nextDl)) {
                  if (!nextTokens.includes(k)) delete nextDl[k]
                }
                form.setFieldsValue({ dl_config: nextDl })
              }}
            />
          </Form.Item>

          {/* Per-DL-model config panels (only visible when DL models are selected).
              Note: dl_config is already a tracked form field (via setFieldsValue
              from each DLConfigPanel's onChange) — the outer Form.Item here is a
              label-only wrapper, so we intentionally omit `name` to avoid antd
              trying to inject value/onChange into the <div> child. */}
          {selectedDlTokens.length > 0 && (
            <Form.Item label={
              <Space size={4}>
                <span>DL 超参配置</span>
                <Text type="secondary" style={{ fontSize: 11 }}>
                  （每个 DL 模型独立配置；未修改项使用注册表默认值）
                </Text>
              </Space>
            }>
              <div>
                {selectedDlTokens.map(token => (
                  <DLConfigPanel
                    key={token}
                    modelId={token}
                    modelSpec={dlModelById[token]}
                    optimizerParams={dlRegistry.optimizer_params}
                    trainParams={dlRegistry.train_params}
                    value={formDlConfig?.[token]}
                    onChange={(next) => {
                      form.setFieldsValue({
                        dl_config: { ...(formDlConfig || {}), [token]: next },
                      })
                    }}
                  />
                ))}
              </div>
            </Form.Item>
          )}

          <Divider style={{ margin: '16px 0' }}>评估与预算</Divider>

          <Form.Item name="eval_metrics" label="评估指标"
            rules={[{ required: true, message: '至少选择一个评估指标' }]}>
            <Select
              mode="multiple"
              placeholder="选择要记录的指标（优化目标必须在其中）"
              options={METRIC_PRESETS[formTaskType] || []}
            />
          </Form.Item>

          <Form.Item name="default_objective_metric" label="默认优化目标"
            rules={[{ required: true, message: '请选择优化目标' }]}>
            <Select
              placeholder="主要优化的单一指标"
              options={METRIC_PRESETS[formTaskType] || []}
            />
          </Form.Item>

          <Row gutter={12}>
            <Col span={8}>
              <Form.Item name={['budget_config', 'max_trials']} label="最大 Trial 数"
                tooltip="baseline 策略下等于模型数；搜索策略下限制迭代次数">
                <InputNumber min={1} max={200} style={{ width: '100%' }} placeholder="20" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name={['budget_config', 'test_size']} label="测试集比例"
                tooltip="留作 holdout 的数据占比">
                <InputNumber min={0.05} max={0.5} step={0.05} style={{ width: '100%' }}
                  placeholder="0.2" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name={['budget_config', 'cv_folds']} label="CV 折数">
                <InputNumber min={2} max={10} style={{ width: '100%' }} placeholder="5" />
              </Form.Item>
            </Col>
          </Row>

          <Paragraph type="secondary" style={{ fontSize: 11, marginTop: 4 }}>
            高级搜索空间（per-model 覆盖）将使用 tuning-spaces 注册表默认值；
            如需自定义，请通过 API 直接 PATCH search_space。
          </Paragraph>
        </Form>
      </Drawer>
    </div>
  )
}
