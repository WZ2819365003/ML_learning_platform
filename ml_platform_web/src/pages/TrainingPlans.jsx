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
 *   - Modal create/edit: select models -> generated config table -> per-model
 *     parameter modal (loads tuning-spaces registry; no hard-coded model list)
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Card, Table, Button, Space, Tag, Popconfirm, Modal, Form, Input, Select,
  Switch, InputNumber, Divider, message, Typography, Tooltip, Empty, Segmented,
  Row, Col,
} from 'antd'
import {
  PlusOutlined, ReloadOutlined, EditOutlined, DeleteOutlined,
  ThunderboltOutlined, InfoCircleOutlined, CopyOutlined,
  CheckCircleOutlined, UndoOutlined, SaveOutlined,
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

/**
 * MLParamsPanel — per-ML-model override editor.
 *
 * Baseline edits scalar defaults; grid_search edits value lists; bayesian
 * edits distribution JSON. All changes write back to search_space[modelId],
 * which the tuning service already consumes per strategy.
 */
function MLParamsPanel({ modelId, meta, strategyType = 'baseline', value, onChange }) {
  const fixed = meta?.fixed || {}
  const gridValues = meta?.grid_values || {}
  const distribution = meta?.distribution || {}
  const paramKeys = Array.from(new Set([
    ...Object.keys(fixed),
    ...Object.keys(gridValues),
    ...Object.keys(distribution),
  ]))

  const defaultFor = (key) => {
    if (strategyType === 'grid_search') {
      if (Array.isArray(gridValues[key])) return gridValues[key]
      if (fixed[key] !== undefined) return [fixed[key]]
      return []
    }
    if (strategyType === 'bayesian_search') {
      if (distribution[key] !== undefined) return distribution[key]
      if (fixed[key] !== undefined) {
        return { type: typeof fixed[key] === 'number' ? 'float' : 'categorical', default: fixed[key] }
      }
      return {}
    }
    if (fixed[key] !== undefined) return fixed[key]
    if (Array.isArray(gridValues[key]) && gridValues[key].length > 0) return gridValues[key][0]
    const dist = distribution[key]
    if (dist?.default !== undefined) return dist.default
    if (dist?.low !== undefined) return dist.low
    if (Array.isArray(dist?.choices) && dist.choices.length > 0) return dist.choices[0]
    return ''
  }

  const defaults = Object.fromEntries(paramKeys.map(k => [k, defaultFor(k)]))
  const merged = { ...defaults, ...(value || {}) }

  const parseList = (raw, fallback) => {
    if (Array.isArray(raw)) return raw
    const tokens = String(raw ?? '')
      .split(',')
      .map(v => v.trim())
      .filter(Boolean)
    if (tokens.length === 0) return fallback
    return tokens.map(v => {
      if (v === 'null') return null
      if (v === 'true') return true
      if (v === 'false') return false
      const n = Number(v)
      return Number.isFinite(n) ? n : v
    })
  }

  const handleFieldChange = (k, v) => {
    if (strategyType === 'grid_search') {
      onChange?.({ ...(value || {}), [k]: parseList(v, defaults[k]) })
      return
    }
    if (strategyType === 'bayesian_search') {
      try {
        onChange?.({ ...(value || {}), [k]: JSON.parse(v || '{}') })
      } catch {
        onChange?.({ ...(value || {}), [k]: v })
      }
      return
    }
    // Coerce numeric strings back to numbers so payload matches registry dtype.
    let coerced = v
    if (typeof defaults[k] === 'number') {
      const n = Number(v)
      coerced = Number.isFinite(n) ? n : v
    } else if (typeof defaults[k] === 'boolean') {
      coerced = !!v
    }
    onChange?.({ ...(value || {}), [k]: coerced })
  }

  const handleReset = () => {
    onChange?.({})
    message.success(`已恢复 ${modelId} 的默认参数`)
  }

  const handleApply = () => {
    message.success(`${modelId} 参数已应用`)
  }

  if (paramKeys.length === 0) {
    return (
      <Text type="secondary" style={{ fontSize: 12 }}>
        此模型暂无注册表参数，使用训练器默认值。
      </Text>
    )
  }

  return (
    <div>
      <Row gutter={[12, 8]}>
        {paramKeys.map(k => {
          const defaultVal = defaults[k]
          const currentVal = merged[k]
          const isNum = typeof defaultVal === 'number'
          const isBool = typeof defaultVal === 'boolean'
          const overridden = value && Object.prototype.hasOwnProperty.call(value, k)
          const rangeHint = distribution[k] || gridValues[k]
            ? `搜索模板: ${JSON.stringify(distribution[k] || gridValues[k])}`
            : null
          return (
            <Col key={k} xs={24} sm={12}>
              <div style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}>
                <code style={{ fontSize: 11, color: '#0f172a' }}>{k}</code>
                {overridden && (
                  <Tag color="orange" style={{ marginLeft: 6, fontSize: 10 }}>已修改</Tag>
                )}
                {rangeHint && (
                  <Tooltip title={rangeHint}>
                    <InfoCircleOutlined style={{ marginLeft: 6, color: '#94a3b8', fontSize: 11 }} />
                  </Tooltip>
                )}
              </div>
              {strategyType === 'grid_search' ? (
                <Input
                  value={Array.isArray(currentVal) ? currentVal.map(v => v === null ? 'null' : String(v)).join(', ') : String(currentVal ?? '')}
                  onChange={e => handleFieldChange(k, e.target.value)}
                  placeholder="逗号分隔，例如: 100, 200, 400"
                />
              ) : strategyType === 'bayesian_search' ? (
                <Input.TextArea
                  autoSize={{ minRows: 2, maxRows: 5 }}
                  value={typeof currentVal === 'string' ? currentVal : JSON.stringify(currentVal ?? {}, null, 2)}
                  onChange={e => handleFieldChange(k, e.target.value)}
                  style={{ fontFamily: 'Consolas, Monaco, monospace', fontSize: 12 }}
                />
              ) : isBool ? (
                <Switch
                  checked={!!currentVal}
                  onChange={v => handleFieldChange(k, v)}
                />
              ) : isNum ? (
                <InputNumber
                  value={currentVal}
                  onChange={v => handleFieldChange(k, v)}
                  style={{ width: '100%' }}
                  placeholder={`默认 ${defaultVal}`}
                />
              ) : (
                <Input
                  value={currentVal ?? ''}
                  onChange={e => handleFieldChange(k, e.target.value)}
                  placeholder={`默认 ${defaultVal}`}
                />
              )}
            </Col>
          )
        })}
      </Row>
      <Space style={{ marginTop: 12 }}>
        <Button
          size="small"
          icon={<CheckCircleOutlined />}
          type="primary"
          ghost
          onClick={handleApply}
        >
          应用参数
        </Button>
        <Button
          size="small"
          icon={<UndoOutlined />}
          onClick={handleReset}
          disabled={!value || Object.keys(value).length === 0}
        >
          还原默认值
        </Button>
      </Space>
    </div>
  )
}

export default function TrainingPlans() {
  const [loading, setLoading] = useState(false)
  const [data, setData] = useState({ items: [], total: 0 })
  const [taskType, setTaskType] = useState('all')
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [paramModalOpen, setParamModalOpen] = useState(false)
  const [modelSelectOpen, setModelSelectOpen] = useState(false)
  const [editingModelToken, setEditingModelToken] = useState(null)
  const [savedParamTokens, setSavedParamTokens] = useState({})
  const [editingId, setEditingId] = useState(null)
  const [saving, setSaving] = useState(false)
  const [tuningSpaces, setTuningSpaces] = useState({ classification: {}, regression: {} })
  const [dlRegistry, setDlRegistry] = useState({
    models: [], optimizer_params: [], train_params: [],
  })
  const [form] = Form.useForm()

  const formTaskType   = Form.useWatch('task_type',    form) || 'classification'
  const formStrategyType = Form.useWatch('strategy_type', form) || 'baseline'
  const formFamily     = Form.useWatch('model_family', form) || 'ml'
  const formSelected   = Form.useWatch('selected_models', form) || []
  const formDlConfig   = Form.useWatch('dl_config',    form) || {}
  const formSearchSpace = Form.useWatch('search_space', form) || {}
  const formBudget      = Form.useWatch('budget_config', form) || {}
  const availableModels = tuningSpaces[formTaskType] || {}

  // Run count estimator — shown as a chip next to the submit button so the
  // user sees how expensive their plan will be before clicking 创建.
  // Formula per strategy:
  //   baseline        : N_models × 1
  //   grid_search     : Σ_models Π_params len(values[param])
  //   bayesian_search : N_models × max_trials
  const planEstimate = useMemo(() => {
    const nModels = formSelected.length
    if (nModels === 0) return { runs: 0, label: '请先选择模型', tone: 'default' }

    if (formStrategyType === 'baseline') {
      return { runs: nModels, label: `${nModels} 个 run`, tone: 'default' }
    }
    if (formStrategyType === 'grid_search') {
      let total = 0
      for (const m of formSelected) {
        const space = formSearchSpace?.[m] || {}
        // each value must be an array; product of lengths
        const lens = Object.values(space)
          .map(v => (Array.isArray(v) ? v.length : 1))
          .filter(n => n > 0)
        const combos = lens.length ? lens.reduce((a, b) => a * b, 1) : 1
        total += combos
      }
      return {
        runs: total,
        label: `约 ${total} 个 run (${nModels} 个模型的网格组合)`,
        tone: total > 50 ? 'danger' : total > 20 ? 'warning' : 'default',
      }
    }
    if (formStrategyType === 'bayesian_search') {
      const trials = Number(formBudget?.max_trials) || 20
      const total = nModels * trials
      return {
        runs: total,
        label: `约 ${total} 个 run (${nModels} 模型 × ${trials} trials)`,
        tone: total > 100 ? 'danger' : total > 50 ? 'warning' : 'default',
      }
    }
    return { runs: 0, label: '', tone: 'default' }
  }, [formStrategyType, formSelected, formSearchSpace, formBudget])

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

  const selectedModelRows = useMemo(
    () => (formSelected || []).map(token => {
      const isDl = !!dlModelById[token]
      const meta = isDl ? dlModelById[token] : availableModels[token]
      const customConfig = isDl ? formDlConfig?.[token] : formSearchSpace?.[token]
      const hasCustom = !!customConfig && Object.keys(customConfig).length > 0
      return {
        token,
        family: isDl ? 'dl' : 'ml',
        name: meta?.display_name || token,
        description: meta?.description || '',
        param_count: isDl
          ? ((meta?.arch_params || []).length + (dlRegistry.optimizer_params || []).length + (dlRegistry.train_params || []).length)
          : Array.from(new Set([
              ...Object.keys(meta?.fixed || {}),
              ...Object.keys(meta?.grid_values || {}),
              ...Object.keys(meta?.distribution || {}),
            ])).length,
        hasCustom,
        saved: !!savedParamTokens[token],
      }
    }),
    [formSelected, dlModelById, availableModels, formDlConfig, formSearchSpace, dlRegistry, savedParamTokens],
  )

  const editingIsDl = !!dlModelById[editingModelToken]
  const editingModelMeta = editingIsDl
    ? dlModelById[editingModelToken]
    : availableModels[editingModelToken]

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
    setSavedParamTokens({})
    setEditingModelToken(null)
    setModelSelectOpen(false)
    form.resetFields()
    form.setFieldsValue({
      task_type: 'classification',
      strategy_type: 'baseline',
      model_family: 'ml',
      selected_models: [],
      dl_config: {},
      search_space: {},
      eval_metrics: ['accuracy', 'f1'],
      default_objective_metric: 'accuracy',
      budget_config: { max_trials: 20, test_size: 0.2 },
    })
    setDrawerOpen(true)
  }

  const handleEdit = async (plan) => {
    setEditingId(plan.id)
    setEditingModelToken(null)
    form.resetFields()
    form.setFieldsValue({
      name: plan.name,
      description: plan.description,
      task_type: plan.task_type,
      strategy_type: plan.strategy_type,
      model_family: plan.model_family || 'ml',
      selected_models: plan.selected_models || [],
      dl_config: plan.dl_config || {},
      search_space: plan.search_space || {},
      eval_metrics: plan.eval_metrics || [],
      default_objective_metric: plan.default_objective_metric,
      budget_config: plan.budget_config || {},
    })
    setSavedParamTokens(Object.fromEntries((plan.selected_models || []).map(t => [t, true])))
    setModelSelectOpen(false)
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
      const selected = values.selected_models || []
      // Only keep dl_config entries for currently-selected DL tokens so stale
      // entries (from deselected models) don't leak into the saved plan.
      const cleanedDlConfig = Object.fromEntries(
        Object.entries(values.dl_config || {}).filter(
          ([k]) => selected.includes(k),
        ),
      )
      // Same hygiene for per-ML-model overrides: only keep entries that are
      // (a) still selected, (b) non-empty (skip {} noise).
      const cleanedSearchSpace = Object.fromEntries(
        Object.entries(values.search_space || {})
          .filter(([k, v]) => selected.includes(k) && v && Object.keys(v).length > 0),
      )
      const payload = {
        name: values.name,
        description: values.description,
        task_type: values.task_type,
        strategy_type: values.strategy_type,
        model_family: values.model_family || 'ml',
        selected_models: selected,
        dl_config: cleanedDlConfig,
        search_space: Object.keys(cleanedSearchSpace).length ? cleanedSearchSpace : null,
        eval_metrics: values.eval_metrics,
        default_objective_metric: values.default_objective_metric,
        default_objective_direction: _objectiveDirection(values.default_objective_metric),
        budget_config: values.budget_config,
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
        variant="borderless"
        style={{ borderRadius: 16, boxShadow: '0 1px 4px rgba(15,23,42,0.06)' }}
        styles={{ body: { padding: 0 } }}
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

      {/* ─────── Create / Edit modal ─────── */}
      {/*
        Migrated from Drawer to Modal per v3.1.2 UX feedback: the side Drawer
        felt detached from the table below, and the original single "创建"
        button left users no way to tune per-model params. The Modal version
        exposes a per-model param editor (MLParamsPanel / DLConfigPanel) and
        a two-step footer: 取消 · [保存参数] (local validate) · 创建/保存.
      */}
      <Modal
        title={editingId ? '编辑训练方案' : '新建训练方案'}
        open={drawerOpen}
        onCancel={() => {
          setDrawerOpen(false)
          setParamModalOpen(false)
        }}
        destroyOnHidden
        width={920}
        centered
        maskClosable={false}
        footer={
          <Space style={{ width: '100%', justifyContent: 'space-between' }}>
            <Tooltip
              title={
                formStrategyType === 'grid_search'
                  ? '网格搜索按每个模型的参数笛卡尔积累加。>50 会被标红，建议缩小搜索空间。'
                  : formStrategyType === 'bayesian_search'
                  ? 'Optuna 按 max_trials × 模型数采样。'
                  : 'baseline 对每个选中的模型跑一次默认超参。'
              }
            >
              <Tag
                color={
                  planEstimate.tone === 'danger' ? 'red'
                    : planEstimate.tone === 'warning' ? 'orange'
                    : 'blue'
                }
                style={{ fontSize: 12, padding: '2px 10px' }}
              >
                预估：{planEstimate.label}
              </Tag>
            </Tooltip>
            <Space>
              <Button onClick={() => setDrawerOpen(false)}>取消</Button>
              <Button
                onClick={async () => {
                  try {
                    await form.validateFields()
                    message.success('当前参数已应用（尚未提交到服务器）')
                  } catch {
                    /* antd renders errors inline */
                  }
                }}
              >
                保存模型表单
              </Button>
              <Button type="primary" loading={saving} onClick={handleSubmit}>
                {editingId ? '保存' : '创建'}
              </Button>
            </Space>
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
                      search_space: {},
                      eval_metrics: [],
                      default_objective_metric: undefined,
                    })
                    setSavedParamTokens({})
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
                setSavedParamTokens(prev => Object.fromEntries(
                  filtered.map(t => [t, prev[t] ?? false]),
                ))
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
              open={modelSelectOpen}
              onOpenChange={setModelSelectOpen}
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
                setSavedParamTokens(prev => Object.fromEntries(
                  nextTokens.map(t => [t, prev[t] ?? false]),
                ))
                setModelSelectOpen(false)
              }}
            />
          </Form.Item>

          <Form.Item label={
            <Space size={4}>
              <span>候选模型配置表</span>
              <Text type="secondary" style={{ fontSize: 11 }}>
                （选中模型后生成配置行；点击"编辑参数"进入独立参数配置页）
              </Text>
            </Space>
          }>
            <Table
              size="small"
              rowKey="token"
              dataSource={selectedModelRows}
              pagination={false}
              style={{ border: '1px solid #e2e8f0', borderRadius: 12, overflow: 'hidden' }}
              locale={{
                emptyText: <Empty image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description="先选择候选模型，系统会在这里生成配置表" />,
              }}
              columns={[
                {
                  title: '模型',
                  key: 'model',
                  render: (_, row) => (
                    <Space direction="vertical" size={2}>
                      <Space>
                        <Tag color={row.family === 'dl' ? 'purple' : 'blue'} style={{ margin: 0 }}>
                          {row.family.toUpperCase()}
                        </Tag>
                        <Text strong>{row.name}</Text>
                      </Space>
                      <Text type="secondary" style={{ fontSize: 11 }}>{row.token}</Text>
                    </Space>
                  ),
                },
                {
                  title: '参数',
                  key: 'params',
                  width: 180,
                  render: (_, row) => (
                    <Space size={4} wrap>
                      <Tag>{row.param_count} 项</Tag>
                      {row.hasCustom ? <Tag color="orange">已自定义</Tag> : <Tag>默认</Tag>}
                    </Space>
                  ),
                },
                {
                  title: '保存状态',
                  key: 'saved',
                  width: 110,
                  render: (_, row) => (
                    <Tag color={row.saved ? 'green' : 'orange'}>
                      {row.saved ? '已保存' : '未保存'}
                    </Tag>
                  ),
                },
                {
                  title: '操作',
                  key: 'actions',
                  width: 210,
                  render: (_, row) => (
                    <Space size={4}>
                      <Button
                        size="small"
                        icon={<EditOutlined />}
                        onClick={() => {
                          setEditingModelToken(row.token)
                          setParamModalOpen(true)
                        }}
                      >
                        编辑参数
                      </Button>
                      <Button
                        size="small"
                        icon={<SaveOutlined />}
                        onClick={() => {
                          setSavedParamTokens(prev => ({ ...prev, [row.token]: true }))
                          message.success(`${row.name} 参数已保存到当前方案草稿`)
                        }}
                      >
                        保存
                      </Button>
                    </Space>
                  ),
                },
              ]}
            />
          </Form.Item>

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
            未修改的模型将使用 tuning-spaces 注册表默认值；
            在"候选模型配置表"中点击"编辑参数"即可进入该模型的独立参数配置页。
          </Paragraph>
        </Form>
      </Modal>

      <Modal
        title="模型参数配置"
        open={paramModalOpen}
        onCancel={() => setParamModalOpen(false)}
        destroyOnHidden
        width={780}
        centered
        footer={
          <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
            <Button onClick={() => setParamModalOpen(false)}>取消</Button>
            <Button
              type="primary"
              icon={<SaveOutlined />}
              disabled={!editingModelToken}
              onClick={() => {
                setSavedParamTokens(prev => ({ ...prev, [editingModelToken]: true }))
                setParamModalOpen(false)
                message.success('模型参数已保存到配置表')
              }}
            >
              保存到配置表
            </Button>
          </Space>
        }
      >
        {!editingModelToken || !editingModelMeta ? (
          <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="未选择模型" />
        ) : (
          <Space direction="vertical" size={14} style={{ width: '100%' }}>
            <div style={{
              padding: 16,
              borderRadius: 16,
              border: '1px solid #dbeafe',
              background: 'linear-gradient(135deg, #eff6ff 0%, #ffffff 100%)',
            }}>
              <Space direction="vertical" size={6}>
                <Space wrap>
                  <Tag color={editingIsDl ? 'purple' : 'blue'} style={{ margin: 0 }}>
                    {editingIsDl ? 'DL' : 'ML'}
                  </Tag>
                  <Text strong style={{ fontSize: 16 }}>
                    {editingModelMeta?.display_name || editingModelToken}
                  </Text>
                  <code style={{ fontSize: 11, color: '#64748b' }}>{editingModelToken}</code>
                </Space>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {editingModelMeta?.description || '编辑该模型在当前训练方案中的参数配置。'}
                </Text>
              </Space>
            </div>

            {editingIsDl ? (
              <DLConfigPanel
                modelId={editingModelToken}
                modelSpec={editingModelMeta}
                optimizerParams={dlRegistry.optimizer_params}
                trainParams={dlRegistry.train_params}
                value={formDlConfig?.[editingModelToken]}
                onChange={(next) => {
                  form.setFieldsValue({
                    dl_config: { ...(formDlConfig || {}), [editingModelToken]: next },
                  })
                  setSavedParamTokens(prev => ({ ...prev, [editingModelToken]: false }))
                }}
              />
            ) : (
              <MLParamsPanel
                modelId={editingModelToken}
                meta={editingModelMeta}
                strategyType={formStrategyType}
                value={formSearchSpace?.[editingModelToken]}
                onChange={(next) => {
                  form.setFieldsValue({
                    search_space: { ...(formSearchSpace || {}), [editingModelToken]: next },
                  })
                  setSavedParamTokens(prev => ({ ...prev, [editingModelToken]: false }))
                }}
              />
            )}
          </Space>
        )}
      </Modal>
    </div>
  )
}
