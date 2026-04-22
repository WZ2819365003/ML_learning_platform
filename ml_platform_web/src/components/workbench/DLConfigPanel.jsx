/**
 * DLConfigPanel — compact per-DL-model hyperparameter editor.
 *
 * Renders three small sub-forms (架构 / 优化器 / 训练) driven by the DL registry
 * metadata returned by `/api/dl/models`.  Used inside TrainingPlans create/edit
 * drawer and (later) in RunInspector for "override baseline" flows.
 *
 * The component is fully controlled — parent owns the `value` object shaped as
 *   { arch: { hidden_dim: 256, ... }, opt: { learning_rate: 1e-3, ... },
 *     train: { epochs: 50, ... } }
 * and receives `onChange(nextValue)` on every field edit.
 */
import React, { useState } from 'react'
import {
  Card, Form, Select, InputNumber, Switch, Row, Col, Tooltip, Tag, Typography,
  Space, Button,
} from 'antd'
import { InfoCircleOutlined, DownOutlined, UpOutlined } from '@ant-design/icons'

const { Text } = Typography

const _isNumeric = (t) => t === 'int' || t === 'float'

/**
 * Render a single param control based on its type spec.
 */
function ParamField({ spec, value, onChange }) {
  const {
    name, display_name, type, min, max, step, options, description, advanced,
  } = spec
  const label = (
    <Space size={4}>
      <span>{display_name || name}</span>
      {description && (
        <Tooltip title={description}>
          <InfoCircleOutlined style={{ color: '#94a3b8', fontSize: 11 }} />
        </Tooltip>
      )}
      {advanced && <Tag color="default" style={{ fontSize: 10 }}>高级</Tag>}
    </Space>
  )

  let control
  if (options && Array.isArray(options) && options.length > 0) {
    control = (
      <Select
        size="small"
        value={value ?? spec.default}
        onChange={onChange}
        options={options.map(o => ({ value: o, label: String(o) }))}
      />
    )
  } else if (type === 'bool') {
    control = (
      <Switch
        size="small"
        checked={!!(value ?? spec.default)}
        onChange={onChange}
      />
    )
  } else if (_isNumeric(type)) {
    control = (
      <InputNumber
        size="small"
        value={value ?? spec.default}
        onChange={onChange}
        min={min}
        max={max}
        step={step ?? (type === 'int' ? 1 : 0.001)}
        precision={type === 'int' ? 0 : undefined}
        style={{ width: '100%' }}
      />
    )
  } else {
    // default: treat as free-text via InputNumber-less fallback
    control = (
      <Select
        size="small"
        value={value ?? spec.default}
        onChange={onChange}
        mode="tags"
        maxTagCount={1}
      />
    )
  }

  return (
    <Form.Item
      label={label}
      style={{ marginBottom: 8 }}
      labelCol={{ style: { paddingBottom: 2, lineHeight: '16px' } }}
    >
      {control}
    </Form.Item>
  )
}

function Section({ title, specs, values, onField, cols = 2, hideAdvanced }) {
  const visible = (specs || []).filter(s => !hideAdvanced || !s.advanced)
  if (visible.length === 0) return null
  const span = 24 / cols
  return (
    <div style={{ marginBottom: 8 }}>
      <Text strong style={{ fontSize: 12, color: '#475569' }}>{title}</Text>
      <Row gutter={[12, 0]} style={{ marginTop: 4 }}>
        {visible.map(s => (
          <Col key={s.name} span={span}>
            <ParamField
              spec={s}
              value={values?.[s.name]}
              onChange={(v) => onField(s.name, v)}
            />
          </Col>
        ))}
      </Row>
    </div>
  )
}

/**
 * @param {object}   props
 * @param {string}   props.modelId
 * @param {object}   props.modelSpec   — entry from DL_MODEL_REGISTRY
 * @param {Array}    props.optimizerParams
 * @param {Array}    props.trainParams
 * @param {object}   props.value       — { arch, opt, train }
 * @param {Function} props.onChange    — (nextValue) => void
 * @param {boolean}  [props.hideAdvanced=true]
 */
export default function DLConfigPanel({
  modelId, modelSpec, optimizerParams, trainParams,
  value, onChange,
}) {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const safeValue = value || { arch: {}, opt: {}, train: {} }

  const updateSection = (section) => (field, fieldValue) => {
    onChange({
      ...safeValue,
      [section]: { ...(safeValue[section] || {}), [field]: fieldValue },
    })
  }

  const hasAdvanced = [
    ...(modelSpec?.arch_params || []),
    ...(optimizerParams || []),
    ...(trainParams || []),
  ].some(s => s.advanced)

  const headerTitle = (
    <Space size={6}>
      <Tag color="purple" style={{ margin: 0 }}>DL</Tag>
      <span style={{ fontWeight: 600 }}>{modelSpec?.display_name || modelId}</span>
      <code style={{ fontSize: 11, color: '#64748b' }}>{modelId}</code>
    </Space>
  )

  const extra = hasAdvanced ? (
    <Button
      size="small"
      type="link"
      onClick={() => setShowAdvanced(v => !v)}
      icon={showAdvanced ? <UpOutlined /> : <DownOutlined />}
      style={{ padding: 0, fontSize: 11 }}
    >
      {showAdvanced ? '收起高级' : '展开高级'}
    </Button>
  ) : null

  return (
    <Card
      size="small"
      title={headerTitle}
      extra={extra}
      style={{ marginBottom: 10, borderRadius: 8 }}
      bodyStyle={{ padding: '12px 14px' }}
    >
      <Form layout="vertical" size="small">
        <Section
          title="架构参数"
          specs={modelSpec?.arch_params || []}
          values={safeValue.arch}
          onField={updateSection('arch')}
          hideAdvanced={!showAdvanced}
        />
        <Section
          title="优化器"
          specs={optimizerParams || []}
          values={safeValue.opt}
          onField={updateSection('opt')}
          hideAdvanced={!showAdvanced}
        />
        <Section
          title="训练控制"
          specs={trainParams || []}
          values={safeValue.train}
          onField={updateSection('train')}
          hideAdvanced={!showAdvanced}
        />
      </Form>
    </Card>
  )
}
