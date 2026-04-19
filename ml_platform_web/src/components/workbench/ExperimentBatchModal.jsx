import React, { useEffect, useState, useMemo } from 'react'
import {
  Modal, Form, Input, Select, Radio, InputNumber, Row, Col, Tabs, Tag,
  Space, Divider, Typography, Alert, Checkbox, message, Tooltip,
} from 'antd'
import { ThunderboltOutlined, NodeIndexOutlined, FunctionOutlined, InfoCircleOutlined, BookOutlined } from '@ant-design/icons'
import { modelingTaskApi, trainingPlansApi } from '../../services/api'

const { Text } = Typography

const STRATEGY_META = {
  baseline: {
    label: '基线 (Baseline)',
    icon: <ThunderboltOutlined />,
    description: '每个模型跑一次默认超参，用最少成本拿到基准指标。适合快速验证数据通路。',
    color: '#10b981',
  },
  grid_search: {
    label: '网格搜索 (Grid)',
    icon: <NodeIndexOutlined />,
    description: '枚举离散候选的笛卡尔积，解释性最强但组合数会指数膨胀。建议每个模型 ≤ 30 组合。',
    color: '#2563eb',
  },
  bayesian_search: {
    label: '贝叶斯寻优 (Optuna TPE)',
    icon: <FunctionOutlined />,
    description: '基于 TPE 顺序采样，每次利用历史反馈选下一组参数。少量 Trial 就能逼近好解。',
    color: '#8b5cf6',
  },
}

// Render per-model tuning controls given the YAML spec for that model
function ModelTuningPanel({ strategy, modelKey, spec, value, onChange }) {
  if (!spec) return <Alert type="warning" showIcon message={`未找到模型 ${modelKey} 的调参空间`} />

  // value shape:
  //   baseline          → { overrides: { [modelKey]: { param: value } } }
  //   grid_search       → { grid: { [modelKey]: { param: [v1, v2] } } }
  //   bayesian_search   → { distribution: { [modelKey]: { param: {...dist spec} } } }

  if (strategy === 'baseline') {
    const overrides = value?.overrides?.[modelKey] || {}
    const gridKeys = Object.keys(spec.grid_values || {})
    if (gridKeys.length === 0) {
      return <Text type="secondary" style={{ fontSize: 12 }}>该模型无需额外超参，使用 fixed 默认值。</Text>
    }
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 10 }}>
        {gridKeys.map(param => {
          const choices = spec.grid_values[param]
          const first = choices?.[0]
          const isNum = typeof first === 'number'
          return (
            <div key={param}>
              <div style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}>
                {param} <Text type="secondary" style={{ fontSize: 11 }}>(默认 {JSON.stringify(first)})</Text>
              </div>
              {isNum ? (
                <InputNumber
                  style={{ width: '100%' }}
                  placeholder={`默认 ${first}`}
                  value={overrides[param]}
                  onChange={(v) => onChange({ [param]: v })}
                />
              ) : (
                <Select
                  style={{ width: '100%' }}
                  placeholder={`默认 ${JSON.stringify(first)}`}
                  value={overrides[param]}
                  allowClear
                  options={choices.map(c => ({ value: String(c), label: String(c) }))}
                  onChange={(v) => onChange({ [param]: v })}
                />
              )}
            </div>
          )
        })}
      </div>
    )
  }

  if (strategy === 'grid_search') {
    const grid = value?.grid?.[modelKey] || {}
    const gridKeys = Object.keys(spec.grid_values || {})
    if (gridKeys.length === 0) {
      return <Text type="secondary" style={{ fontSize: 12 }}>该模型无可网格调参的超参。</Text>
    }
    return (
      <div style={{ display: 'grid', gap: 10 }}>
        {gridKeys.map(param => {
          const defaults = spec.grid_values[param]
          const current = grid[param] ?? defaults
          return (
            <div key={param}>
              <div style={{ fontSize: 12, color: '#475569', marginBottom: 4, display: 'flex', justifyContent: 'space-between' }}>
                <span>{param}</span>
                <Text type="secondary" style={{ fontSize: 11 }}>
                  默认 {defaults.length} 个候选 · 当前 {current.length}
                </Text>
              </div>
              <Select
                mode="tags"
                style={{ width: '100%' }}
                value={current.map(v => JSON.stringify(v))}
                tokenSeparators={[',']}
                onChange={(vals) => {
                  const parsed = vals.map(v => {
                    if (v === 'null') return null
                    if (v === 'true') return true
                    if (v === 'false') return false
                    const num = Number(v)
                    return Number.isFinite(num) && v.trim() !== '' ? num : v
                  })
                  onChange({ [param]: parsed })
                }}
                options={defaults.map(v => ({ value: JSON.stringify(v), label: String(v) }))}
              />
            </div>
          )
        })}
      </div>
    )
  }

  if (strategy === 'bayesian_search') {
    const dist = value?.distribution?.[modelKey] || {}
    const distKeys = Object.keys(spec.distribution || {})
    if (distKeys.length === 0) {
      return <Text type="secondary" style={{ fontSize: 12 }}>该模型未声明 Bayesian 分布，退回默认空间。</Text>
    }
    return (
      <div style={{ display: 'grid', gap: 8 }}>
        {distKeys.map(param => {
          const d = spec.distribution[param]
          const active = dist[param] || d
          const summary = d.type === 'float' || d.type === 'int'
            ? `${d.type} [${d.low}, ${d.high}]${d.log ? ' (log)' : ''}${d.step ? ` step=${d.step}` : ''}`
            : `categorical (${(d.choices || []).length} 选项)`
          return (
            <div key={param} style={{
              padding: '8px 12px', borderRadius: 6,
              background: 'rgba(139, 92, 246, 0.06)',
              border: '1px solid rgba(139, 92, 246, 0.15)',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 12, fontWeight: 500 }}>{param}</span>
                <Tag color="purple" style={{ fontSize: 11 }}>{summary}</Tag>
              </div>
              {(d.type === 'float' || d.type === 'int') && (
                <Row gutter={8} style={{ marginTop: 6 }}>
                  <Col span={12}>
                    <InputNumber size="small" style={{ width: '100%' }} placeholder="low"
                      value={active.low}
                      onChange={(v) => onChange({ [param]: { ...active, low: v } })} />
                  </Col>
                  <Col span={12}>
                    <InputNumber size="small" style={{ width: '100%' }} placeholder="high"
                      value={active.high}
                      onChange={(v) => onChange({ [param]: { ...active, high: v } })} />
                  </Col>
                </Row>
              )}
            </div>
          )
        })}
      </div>
    )
  }

  return null
}


export default function ExperimentBatchModal({ open, task, onClose, onSubmitted }) {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [tuningSpaces, setTuningSpaces] = useState({})
  const [submitting, setSubmitting] = useState(false)

  const strategy = Form.useWatch('strategy_type', form) || 'baseline'
  const selectedModels = Form.useWatch('selected_models', form) || []
  const [modelParams, setModelParams] = useState({})  // nested per-strategy params
  const [plans, setPlans] = useState([])
  const [appliedPlanId, setAppliedPlanId] = useState(null)

  // Reset state when modal opens / task changes
  useEffect(() => {
    if (open && task) {
      form.resetFields()
      form.setFieldsValue({
        name: `${task.name}-${new Date().toLocaleString('zh-CN', { hour12: false }).replace(/[/\s:]/g, '')}`,
        strategy_type: 'baseline',
        selected_models: [],
        max_trials: 20,
        n_trials_per_model: 10,
        cv_folds: 5,
        test_size: 0.2,
      })
      setModelParams({})
    }
  }, [open, task, form])

  // Load tuning spaces when task type is known
  useEffect(() => {
    if (!open || !task?.task_type) return
    setLoading(true)
    modelingTaskApi.tuningSpaces(task.task_type)
      .then(resp => setTuningSpaces(resp || {}))
      .catch(err => message.error('加载调参空间失败：' + (err?.response?.data?.detail || err.message)))
      .finally(() => setLoading(false))
  }, [open, task?.task_type])

  // Load applicable training plans (filtered by task_type so regression plans
  // don't pollute a classification task's picker).
  useEffect(() => {
    if (!open || !task?.task_type) { setPlans([]); return }
    trainingPlansApi.list({ task_type: task.task_type, page_size: 50 })
      .then(resp => setPlans(resp?.items || []))
      .catch(() => setPlans([]))
  }, [open, task?.task_type])

  // Apply a plan by prefilling the form from its config.  This is best-effort:
  // search_space overrides aren't re-hydrated into the per-model tuning UI
  // (that would require replaying every Tab's state), but strategy + models +
  // budget copy cleanly, which is the 80% benefit.
  const applyPlan = (planId) => {
    const plan = plans.find(p => p.id === planId)
    setAppliedPlanId(planId || null)
    if (!plan) return
    form.setFieldsValue({
      strategy_type: plan.strategy_type,
      selected_models: plan.selected_models || [],
      max_trials: plan.budget_config?.max_trials ?? 20,
      cv_folds: plan.budget_config?.cv_folds ?? 5,
      test_size: plan.budget_config?.test_size ?? 0.2,
      n_trials_per_model: plan.budget_config?.n_trials_per_model ?? 10,
    })
    setModelParams({})
    message.success(`已套用方案：${plan.name}`)
    // Fire-and-forget usage bump
    trainingPlansApi.markUsed(planId).catch(() => {})
  }

  const modelOptions = useMemo(
    () => Object.keys(tuningSpaces).map(k => ({ value: k, label: k })),
    [tuningSpaces]
  )

  const updateModelParam = (modelKey, patch) => {
    setModelParams(prev => {
      const next = { ...prev }
      if (strategy === 'baseline') {
        next.overrides = { ...(next.overrides || {}) }
        next.overrides[modelKey] = { ...(next.overrides[modelKey] || {}), ...patch }
        // strip nulls
        for (const k of Object.keys(next.overrides[modelKey])) {
          if (next.overrides[modelKey][k] === null || next.overrides[modelKey][k] === undefined) {
            delete next.overrides[modelKey][k]
          }
        }
      } else if (strategy === 'grid_search') {
        next.grid = { ...(next.grid || {}) }
        next.grid[modelKey] = { ...(next.grid[modelKey] || {}), ...patch }
      } else if (strategy === 'bayesian_search') {
        next.distribution = { ...(next.distribution || {}) }
        next.distribution[modelKey] = { ...(next.distribution[modelKey] || {}), ...patch }
      }
      return next
    })
  }

  const handleSubmit = async () => {
    let values
    try { values = await form.validateFields() }
    catch { return }

    if (!values.selected_models?.length) {
      message.warning('请至少选择一个模型')
      return
    }

    // Build search_space payload matching backend expectations
    let search_space = {}
    if (strategy === 'baseline') {
      search_space = modelParams.overrides || {}
    } else if (strategy === 'grid_search') {
      search_space = modelParams.grid || {}
    } else if (strategy === 'bayesian_search') {
      search_space = modelParams.distribution || {}
    }

    const budget_config = {
      max_trials: values.max_trials,
      cv_folds: values.cv_folds,
      test_size: values.test_size,
    }
    if (strategy === 'bayesian_search') {
      budget_config.n_trials_per_model = values.n_trials_per_model
    }

    setSubmitting(true)
    try {
      await modelingTaskApi.createExperimentBatch(task.id, {
        name: values.name,
        strategy_type: strategy,
        selected_models: values.selected_models,
        search_space,
        budget_config,
      })
      message.success('实验批次已提交，开始训练')
      onSubmitted?.()
      onClose?.()
    } catch (err) {
      message.error(err?.response?.data?.detail || '提交失败')
    } finally {
      setSubmitting(false)
    }
  }

  const strategyMeta = STRATEGY_META[strategy]

  return (
    <Modal
      title={<span><ThunderboltOutlined style={{ marginRight: 8, color: '#2563eb' }} />启动新的实验批次</span>}
      open={open}
      onCancel={onClose}
      onOk={handleSubmit}
      confirmLoading={submitting}
      width={780}
      okText="提交并开始训练"
      cancelText="取消"
      destroyOnClose
    >
      <Alert
        type="info" showIcon
        style={{ marginBottom: 12 }}
        message={<Space><strong>{strategyMeta?.label}</strong><Text type="secondary" style={{ fontSize: 12 }}>{strategyMeta?.description}</Text></Space>}
      />

      {plans.length > 0 && (
        <div style={{
          marginBottom: 12, padding: '10px 12px', borderRadius: 6,
          background: 'rgba(37,99,235,0.04)', border: '1px dashed rgba(37,99,235,0.25)',
        }}>
          <Space align="center" size={10} style={{ width: '100%' }}>
            <BookOutlined style={{ color: '#2563eb' }} />
            <Text style={{ fontSize: 12, color: '#475569' }}>从已保存方案快速套用：</Text>
            <Select
              placeholder="选择方案（覆盖策略 · 模型 · 预算）"
              style={{ flex: 1, minWidth: 260 }}
              allowClear
              value={appliedPlanId}
              onChange={applyPlan}
              options={plans.map(p => ({
                value: p.id,
                label: (
                  <Space>
                    <span style={{ fontWeight: 500 }}>{p.name}</span>
                    <Tag color="blue" style={{ fontSize: 10 }}>{p.strategy_type}</Tag>
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      {(p.selected_models || []).length} 个模型
                    </Text>
                  </Space>
                ),
              }))}
            />
          </Space>
        </div>
      )}

      <Form form={form} layout="vertical" size="middle">
        <Row gutter={12}>
          <Col span={14}>
            <Form.Item name="name" label="批次名称" rules={[{ required: true, min: 2 }]}>
              <Input placeholder="例：rf+xgb 基线对比" />
            </Form.Item>
          </Col>
          <Col span={10}>
            <Form.Item name="strategy_type" label="策略" rules={[{ required: true }]}>
              <Radio.Group optionType="button" buttonStyle="solid" style={{ width: '100%' }}>
                <Radio.Button value="baseline" style={{ width: '33.33%', textAlign: 'center' }}>基线</Radio.Button>
                <Radio.Button value="grid_search" style={{ width: '33.33%', textAlign: 'center' }}>网格</Radio.Button>
                <Radio.Button value="bayesian_search" style={{ width: '33.33%', textAlign: 'center' }}>贝叶斯</Radio.Button>
              </Radio.Group>
            </Form.Item>
          </Col>
        </Row>

        <Form.Item name="selected_models" label="参与模型" rules={[{ required: true }]}>
          <Select
            mode="multiple"
            loading={loading}
            placeholder={`选择要参与 ${task?.task_type === 'regression' ? '回归' : '分类'} 的模型`}
            options={modelOptions}
            maxTagCount="responsive"
          />
        </Form.Item>

        <Row gutter={12}>
          <Col span={strategy === 'bayesian_search' ? 6 : 8}>
            <Form.Item name="max_trials" label={<Tooltip title="整个批次的最大训练次数上限（安全阀）"><span>最大 Trial 数 <InfoCircleOutlined style={{ color: '#94a3b8' }} /></span></Tooltip>}>
              <InputNumber min={1} max={500} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          {strategy === 'bayesian_search' && (
            <Col span={6}>
              <Form.Item name="n_trials_per_model" label="每模型 Trial">
                <InputNumber min={1} max={200} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          )}
          <Col span={strategy === 'bayesian_search' ? 6 : 8}>
            <Form.Item name="cv_folds" label="CV 折数">
              <InputNumber min={2} max={10} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={strategy === 'bayesian_search' ? 6 : 8}>
            <Form.Item name="test_size" label="测试集比例">
              <InputNumber min={0.05} max={0.5} step={0.05} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
        </Row>

        {selectedModels.length > 0 && (
          <>
            <Divider style={{ margin: '4px 0 12px' }}>
              <Text type="secondary" style={{ fontSize: 12 }}>每模型调参空间</Text>
            </Divider>
            <Tabs
              size="small"
              type="card"
              items={selectedModels.map(m => ({
                key: m,
                label: m,
                children: (
                  <div style={{ padding: '4px 2px' }}>
                    <ModelTuningPanel
                      strategy={strategy}
                      modelKey={m}
                      spec={tuningSpaces[m]}
                      value={modelParams}
                      onChange={(patch) => updateModelParam(m, patch)}
                    />
                  </div>
                ),
              }))}
            />
          </>
        )}
      </Form>
    </Modal>
  )
}
