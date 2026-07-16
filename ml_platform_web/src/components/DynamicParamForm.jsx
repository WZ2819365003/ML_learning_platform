/**
 * DynamicParamForm — renders a list of ParamSpec objects as Ant Design Form fields.
 *
 * Props:
 *   params        - array of ParamSpec objects (from model_registry / API response)
 *   advancedMode  - bool; when false, hides params where advanced === true
 *   formItemProps - optional extra props to spread onto each Form.Item (e.g. labelCol)
 *
 * Usage: place inside an Ant Design <Form> — reads/writes via Form context.
 */
import React from 'react';
import { Form, InputNumber, Select, Switch, Input, Tooltip } from 'antd';
import { QuestionCircleOutlined } from '@ant-design/icons';

const { Option } = Select;

function ParamLabel({ displayName, description }) {
  return (
    <span>
      {displayName}
      {description && (
        <Tooltip title={description}>
          <QuestionCircleOutlined style={{ marginLeft: 4, color: '#999', fontSize: 12 }} />
        </Tooltip>
      )}
    </span>
  );
}

function renderField(param) {
  const { type, default: defaultVal, min, max, step, options } = param;

  if (type === 'int') {
    return (
      <InputNumber
        style={{ width: '100%' }}
        min={min}
        max={max}
        step={step ?? 1}
        precision={0}
        placeholder={defaultVal != null ? String(defaultVal) : undefined}
      />
    );
  }

  if (type === 'float') {
    return (
      <InputNumber
        style={{ width: '100%' }}
        min={min}
        max={max}
        step={step ?? 0.01}
        placeholder={defaultVal != null ? String(defaultVal) : undefined}
      />
    );
  }

  if (type === 'bool') {
    return <Switch />;
  }

  if (type === 'str' && options?.length) {
    return (
      <Select placeholder={defaultVal != null ? String(defaultVal) : '请选择'} allowClear>
        {options.map(opt => (
          <Option key={opt} value={opt === 'None' ? null : opt}>
            {opt}
          </Option>
        ))}
      </Select>
    );
  }

  if (type === 'list') {
    // Accepts comma-separated integers, e.g. "100,50" → stored as string, parsed at submit
    return (
      <Input
        placeholder={
          Array.isArray(defaultVal) ? defaultVal.join(',') : (defaultVal ?? '例: 100,50')
        }
      />
    );
  }

  // Fallback: plain text input
  return <Input placeholder={defaultVal != null ? String(defaultVal) : undefined} />;
}

export default function DynamicParamForm({ params = [], advancedMode = false, formItemProps = {} }) {
  const visible = advancedMode ? params : params.filter(p => !p.advanced);

  if (visible.length === 0) return null;

  return (
    <>
      {visible.map(param => {
        const isSwitch = param.type === 'bool';
        return (
          <Form.Item
            key={param.name}
            name={['hyperparameters', param.name]}
            label={<ParamLabel displayName={param.display_name} description={param.description} />}
            valuePropName={isSwitch ? 'checked' : 'value'}
            initialValue={param.default ?? undefined}
            rules={param.required ? [{ required: true, message: `请填写 ${param.display_name}` }] : []}
            {...formItemProps}
          >
            {renderField(param)}
          </Form.Item>
        );
      })}
    </>
  );
}
