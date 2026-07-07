import React, { useEffect, useMemo, useState } from 'react'
import {
  Tabs, Form, Select, InputNumber, Input, Switch, Button, Space, Divider,
  Typography, Tooltip, message, Alert, Card, Modal,
} from 'antd'
import {
  RocketOutlined, CodeOutlined, QuestionCircleOutlined, ThunderboltOutlined,
  SettingOutlined,
} from '@ant-design/icons'
import { trainingApi, dlApi, modelingTaskApi } from '../../services/api'
import ModelSelector from '../ModelSelector'
import DynamicParamForm from '../DynamicParamForm'
import CodeConfigModal from './CodeConfigModal'

const { Text, Paragraph } = Typography

// Comma-separated numeric strings → number[] (matches TrainingConfig behaviour)
function parseHyperparameters(raw = {}) {
  const out = {}
  for (const [k, v] of Object.entries(raw)) {
    if (v === undefined || v === null || v === '') continue
    if (typeof v === 'string' && v.includes(',')) {
      const nums = v.split(',').map(s => Number(s.trim())).filter(n => !Number.isNaN(n))
      out[k] = nums.length ? nums : v
    } else {
      out[k] = v
    }
  }
  return out
}

// Compact param renderer for DL arch/opt/train groups (mirrors DLConfig).
function renderDlField(param) {
  const { type, default: def, min, max, step, options } = param
  if (type === 'int') return <InputNumber style={{ width: '100%' }} min={min} max={max} step={step ?? 1} precision={0} placeholder={def != null ? String(def) : undefined} />
  if (type === 'float') return <InputNumber style={{ width: '100%' }} min={min} max={max} step={step ?? 0.01} placeholder={def != null ? String(def) : undefined} />
  if (type === 'bool') return <Switch />
  if (type === 'str' && options?.length) return <Select allowClear placeholder={def != null ? String(def) : '请选择'} options={options.map(o => ({ value: o === 'None' ? null : o, label: o }))} />
  if (type === 'list') return <Input placeholder={Array.isArray(def) ? def.join(',') : (def ?? '例: 128,64,32')} />
  return <Input placeholder={def != null ? String(def) : undefined} />
}

function DlParamGroup({ params = [], prefix, advancedMode }) {
  const visible = advancedMode ? params : params.filter(p => !p.advanced)
  if (!visible.length) return <Text type="secondary" style={{ fontSize: 12 }}>暂无可配置参数。</Text>
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0 16px' }}>
      {visible.map(p => (
        <Form.Item key={p.name} name={[prefix, p.name]} initialValue={p.default ?? undefined}
          valuePropName={p.type === 'bool' ? 'checked' : 'value'}
          label={<span>{p.display_name}{p.description && <Tooltip title={p.description}><QuestionCircleOutlined style={{ marginLeft: 4, color: '#999', fontSize: 12 }} /></Tooltip>}</span>}>
          {renderDlField(p)}
        </Form.Item>
      ))}
    </div>
  )
}

const _tsName = (task, tag) =>
  `${task?.name || 'task'}-${tag}-${new Date().toLocaleString('zh-CN', { hour12: false }).replace(/[/\s:]/g, '')}`

/**
 * 模型配置 step — 3 tabs (机器学习 / 深度学习 / 混合策略). Each configures real
 * ML/DL parameters and dispatches a baseline batch through the existing V3
 * pipeline (modelingTaskApi.createExperimentBatch), so 训练/结果/部署 are unchanged.
 * A 「代码配置」 button (Python executor) is available on every tab.
 */
export default function ModelConfigTabs({ task, onSubmitted }) {
  const taskType = task?.task_type || 'classification'
  const [mlForm] = Form.useForm()
  const [dlForm] = Form.useForm()
  const [mixedForm] = Form.useForm()
  const [mlReg, setMlReg] = useState({ categories: [], models: [], classification_metrics: [], regression_metrics: [] })
  const [dlReg, setDlReg] = useState({ categories: [], models: [], optimizer_params: [], train_params: [] })
  const [submitting, setSubmitting] = useState(false)
  const [mlAdvOpen, setMlAdvOpen] = useState(false)
  const [dlAdvOpen, setDlAdvOpen] = useState(false)
  const [codeOpen, setCodeOpen] = useState(false)
  const [codeTemplate, setCodeTemplate] = useState('')

  useEffect(() => {
    trainingApi.listModels()
      .then(r => setMlReg({
        categories: r.categories ?? [], models: r.models ?? [],
        classification_metrics: r.classification_metrics ?? [], regression_metrics: r.regression_metrics ?? [],
      })).catch(() => {})
    dlApi.listModels()
      .then(r => setDlReg({
        categories: r.categories ?? [], models: r.models ?? [],
        optimizer_params: r.optimizer_params ?? [], train_params: r.train_params ?? [],
      })).catch(() => {})
  }, [])

  const mlModel = Form.useWatch('model_type', mlForm)
  const mlSpec = useMemo(() => mlReg.models.find(m => m.id === mlModel) ?? null, [mlReg.models, mlModel])
  const metricOptions = (taskType === 'regression' ? mlReg.regression_metrics : mlReg.classification_metrics)
    .map(m => ({ label: m.label, value: m.value }))

  const dlModel = Form.useWatch('model_type', dlForm)
  const dlSpec = useMemo(() => dlReg.models.find(m => m.id === dlModel) ?? null, [dlReg.models, dlModel])
  const dlModelOptions = dlReg.models
    .filter(m => !m.task_types || m.task_types.includes(taskType))
    .map(m => ({ value: m.id, label: m.display_name || m.id }))
  const hasDlAdvanced = (dlSpec?.arch_params || []).some(p => p.advanced)
    || dlReg.optimizer_params.some(p => p.advanced)
    || dlReg.train_params.some(p => p.advanced)

  const mixedOptions = useMemo(() => ([
    {
      label: '机器学习',
      options: mlReg.models.filter(m => m.task_types?.includes(taskType))
        .map(m => ({ value: m.id, label: `${m.display_name || m.id}` })),
    },
    {
      label: '深度学习',
      options: dlModelOptions,
    },
  ]), [mlReg.models, dlModelOptions, taskType])

  const dispatch = async (payload) => {
    setSubmitting(true)
    try {
      await modelingTaskApi.createExperimentBatch(task.id, payload)
      message.success('已提交训练')
      onSubmitted?.()
    } catch (err) {
      message.error(err?.response?.data?.detail || '提交失败')
    } finally {
      setSubmitting(false)
    }
  }

  const submitMl = async () => {
    let v; try { v = await mlForm.validateFields() } catch { return }
    const params = parseHyperparameters(v.hyperparameters || {})
    await dispatch({
      name: _tsName(task, v.model_type),
      strategy_type: 'baseline',
      selected_models: [v.model_type],
      search_space: Object.keys(params).length ? { [v.model_type]: params } : {},
      budget_config: { cv_folds: v.cv_folds ?? 5, test_size: v.test_size ?? 0.2 },
      eval_metrics: v.eval_metrics,
      model_family: 'ml',
    })
  }

  const submitDl = async () => {
    let v; try { v = await dlForm.validateFields() } catch { return }
    await dispatch({
      name: _tsName(task, v.model_type),
      strategy_type: 'baseline',
      selected_models: [v.model_type],
      model_family: 'dl',
      dl_config: { [v.model_type]: { arch: v.arch_config || {}, opt: v.opt_config || {}, train: v.train_config || {} } },
    })
  }

  const submitMixed = async () => {
    let v; try { v = await mixedForm.validateFields() } catch { return }
    if (!v.models?.length) { message.warning('请至少选择一个模型'); return }
    await dispatch({
      name: _tsName(task, 'mixed'),
      strategy_type: 'baseline',
      selected_models: v.models,
      model_family: 'mixed',
    })
  }

  const openCode = (kind) => {
    setCodeTemplate(CODE_TEMPLATES[kind])
    setCodeOpen(true)
  }

  const CodeButton = ({ kind }) => (
    <Button icon={<CodeOutlined />} onClick={() => openCode(kind)}>代码配置</Button>
  )

  const mlTab = (
    <Form form={mlForm} layout="vertical" initialValues={{ test_size: 0.2, cv_folds: 5, eval_metrics: taskType === 'regression' ? ['rmse', 'r2'] : ['accuracy', 'f1'] }}>
      <Form.Item name="model_type" label="模型" rules={[{ required: true, message: '请选择机器学习模型' }]}>
        <ModelSelector models={mlReg.models} categories={mlReg.categories} taskFilter={taskType}
          value={mlModel} onChange={v => mlForm.setFieldValue('model_type', v)} />
      </Form.Item>
      <Space size={16} wrap>
        <Form.Item name="test_size" label="测试集比例" style={{ marginBottom: 8 }}>
          <InputNumber min={0.05} max={0.5} step={0.05} style={{ width: 120 }} />
        </Form.Item>
        <Form.Item name="cv_folds" label="交叉验证折数" style={{ marginBottom: 8 }}>
          <InputNumber min={2} max={20} step={1} style={{ width: 120 }} />
        </Form.Item>
      </Space>
      {mlSpec?.params?.length > 0 && (
        <>
          <Divider orientation="left" style={{ margin: '8px 0 12px' }}>
            <Space>
              模型参数
              {mlSpec.params.some(p => p.advanced) && (
                <Button size="small" type="link" icon={<SettingOutlined />} onClick={() => setMlAdvOpen(true)}>
                  高级设置
                </Button>
              )}
            </Space>
          </Divider>
          <DynamicParamForm params={mlSpec.params} advancedMode={false} />
        </>
      )}
      {/* Advanced params live in a modal (rendered inside this Form so fields
          bind to mlForm via context) to keep the main panel uncluttered. */}
      <Modal title={<span><SettingOutlined /> 机器学习 · 高级参数</span>} open={mlAdvOpen} width={640}
        onCancel={() => setMlAdvOpen(false)}
        footer={<Button type="primary" onClick={() => setMlAdvOpen(false)}>完成</Button>}>
        <Paragraph type="secondary" style={{ fontSize: 12 }}>不常改的进阶参数，保存后随本次训练一起提交。</Paragraph>
        <DynamicParamForm params={(mlSpec?.params || []).filter(p => p.advanced)} advancedMode />
      </Modal>
      <Form.Item name="eval_metrics" label="评估指标">
        <Select mode="multiple" options={metricOptions} placeholder="请选择评估指标" />
      </Form.Item>
      <Space>
        <Button type="primary" icon={<RocketOutlined />} loading={submitting} onClick={submitMl}>启动机器学习训练</Button>
        <CodeButton kind="ml" />
      </Space>
    </Form>
  )

  const dlTab = (
    <Form form={dlForm} layout="vertical" initialValues={{ task_type: taskType }}>
      <Form.Item name="model_type" label="模型" rules={[{ required: true, message: '请选择深度学习模型' }]}>
        <Select placeholder="请选择深度学习模型" options={dlModelOptions} />
      </Form.Item>
      {dlSpec?.description && (
        <Card size="small" style={{ marginBottom: 12, background: '#f8fafc' }}>
          <Text type="secondary" style={{ fontSize: 12 }}>{dlSpec.description}</Text>
        </Card>
      )}
      {dlSpec && hasDlAdvanced && (
        <div style={{ marginBottom: 8 }}>
          <Button size="small" type="link" icon={<SettingOutlined />} onClick={() => setDlAdvOpen(true)}>
            高级设置（网络/优化器/训练细节）
          </Button>
        </div>
      )}
      {dlSpec?.arch_params?.length > 0 && (
        <>
          <Divider orientation="left" style={{ margin: '4px 0 8px' }}>网络结构</Divider>
          <DlParamGroup params={dlSpec.arch_params} prefix="arch_config" advancedMode={false} />
        </>
      )}
      {dlReg.optimizer_params.length > 0 && (
        <>
          <Divider orientation="left" style={{ margin: '4px 0 8px' }}>优化器</Divider>
          <DlParamGroup params={dlReg.optimizer_params} prefix="opt_config" advancedMode={false} />
        </>
      )}
      {dlReg.train_params.length > 0 && (
        <>
          <Divider orientation="left" style={{ margin: '4px 0 8px' }}>训练控制</Divider>
          <DlParamGroup params={dlReg.train_params} prefix="train_config" advancedMode={false} />
        </>
      )}
      {/* DL advanced params modal — inside dlForm's Form so fields bind via context. */}
      <Modal title={<span><SettingOutlined /> 深度学习 · 高级参数</span>} open={dlAdvOpen} width={700}
        onCancel={() => setDlAdvOpen(false)}
        footer={<Button type="primary" onClick={() => setDlAdvOpen(false)}>完成</Button>}>
        <Paragraph type="secondary" style={{ fontSize: 12 }}>进阶网络 / 优化器 / 训练参数，保存后随本次训练提交。</Paragraph>
        {(dlSpec?.arch_params || []).some(p => p.advanced) && (
          <><Divider orientation="left" style={{ margin: '4px 0 8px' }}>网络结构</Divider>
            <DlParamGroup params={(dlSpec.arch_params || []).filter(p => p.advanced)} prefix="arch_config" advancedMode /></>
        )}
        {dlReg.optimizer_params.some(p => p.advanced) && (
          <><Divider orientation="left" style={{ margin: '4px 0 8px' }}>优化器</Divider>
            <DlParamGroup params={dlReg.optimizer_params.filter(p => p.advanced)} prefix="opt_config" advancedMode /></>
        )}
        {dlReg.train_params.some(p => p.advanced) && (
          <><Divider orientation="left" style={{ margin: '4px 0 8px' }}>训练控制</Divider>
            <DlParamGroup params={dlReg.train_params.filter(p => p.advanced)} prefix="train_config" advancedMode /></>
        )}
      </Modal>
      <Space style={{ marginTop: 8 }}>
        <Button type="primary" icon={<RocketOutlined />} loading={submitting} onClick={submitDl}>启动深度学习训练</Button>
        <CodeButton kind="dl" />
      </Space>
    </Form>
  )

  const mixedTab = (
    <Form form={mixedForm} layout="vertical">
      <Alert type="info" showIcon style={{ marginBottom: 12 }}
        message="混合策略：同时训练机器学习 + 深度学习模型"
        description="选择跨族模型一次性对照，或直接用「代码配置」以 Python 精确描述实验（推荐）。" />
      <Form.Item name="models" label="参与模型（机器学习 + 深度学习）" rules={[{ required: true, message: '请至少选择一个模型' }]}>
        <Select mode="multiple" options={mixedOptions} placeholder="从机器学习 / 深度学习中多选" maxTagCount="responsive" />
      </Form.Item>
      <Space>
        <Button type="primary" icon={<ThunderboltOutlined />} loading={submitting} onClick={submitMixed}>启动混合训练</Button>
        <CodeButton kind="mixed" />
      </Space>
    </Form>
  )

  return (
    <>
      <Paragraph type="secondary" style={{ marginBottom: 8 }}>
        为本任务（{task?.dataset_name || '数据集'} · 目标 {task?.target_column || '-'}）配置模型。可多次启动，累积成多组对照实验。
      </Paragraph>
      <Tabs
        items={[
          { key: 'ml', label: '机器学习', children: mlTab },
          { key: 'dl', label: '深度学习', children: dlTab },
          { key: 'mixed', label: '混合策略', children: mixedTab },
        ]}
      />
      <CodeConfigModal open={codeOpen} task={task} defaultCode={codeTemplate}
        onClose={() => setCodeOpen(false)}
        onSubmitted={() => { setCodeOpen(false); onSubmitted?.() }} />
    </>
  )
}

// Python templates seeded into the 代码配置 editor per tab.
const CODE_TEMPLATES = {
  ml: `# 定义一个名为 config 的 dict。机器学习示例：
config = {
    "name": "ML 代码配置",
    "strategy_type": "baseline",      # baseline | grid_search | bayesian_search
    "model_family": "ml",
    "selected_models": ["random_forest", "xgboost"],
    "search_space": {
        "random_forest": {"n_estimators": 300, "max_depth": 8},
    },
    "budget_config": {"cv_folds": 5, "test_size": 0.2},
    "eval_metrics": ["accuracy", "f1"],
}
`,
  dl: `# 深度学习示例：
config = {
    "name": "DL 代码配置",
    "strategy_type": "baseline",
    "model_family": "dl",
    "selected_models": ["mlp_dl"],
    "dl_config": {
        "mlp_dl": {"arch": {"hidden_layers": [256, 128]}, "train": {"epochs": 30}},
    },
}
`,
  mixed: `# 混合策略：机器学习 + 深度学习一次对照。
ml = ["random_forest", "xgboost", "lightgbm"]
dl = ["mlp_dl"]
config = {
    "name": "混合代码配置",
    "strategy_type": "baseline",
    "model_family": "mixed",
    "selected_models": ml + dl,
    "search_space": {m: {"n_estimators": 200} for m in ["random_forest"]},
    "eval_metrics": ["accuracy", "f1"],
}
`,
}
