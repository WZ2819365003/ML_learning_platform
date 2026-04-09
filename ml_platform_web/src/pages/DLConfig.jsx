import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Button, Card, Divider, Form, Input, InputNumber, Select,
  Space, Switch, Tabs, Tooltip, Typography, message,
} from 'antd';
import { QuestionCircleOutlined, RocketOutlined, SettingOutlined } from '@ant-design/icons';
import { dataApi, dlApi } from '../services/api';

const { Paragraph, Text, Title } = Typography;

// ── Shared param label with tooltip ──────────────────────────────────────────
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

// ── Render a single param field based on its type ────────────────────────────
function renderParamField(param) {
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
          <Select.Option key={opt} value={opt === 'None' ? null : opt}>
            {opt}
          </Select.Option>
        ))}
      </Select>
    );
  }

  if (type === 'list') {
    return (
      <Input
        placeholder={
          Array.isArray(defaultVal) ? defaultVal.join(',') : (defaultVal ?? '例: 128,64,32')
        }
      />
    );
  }

  return <Input placeholder={defaultVal != null ? String(defaultVal) : undefined} />;
}

// ── Render a param group under a given form namespace prefix ─────────────────
function ParamGroup({ params = [], prefix, advancedMode }) {
  const visible = advancedMode ? params : params.filter(p => !p.advanced);
  if (visible.length === 0) return <Text type="secondary">暂无可配置参数。</Text>;

  return (
    <>
      {visible.map(param => {
        const isSwitch = param.type === 'bool';
        return (
          <Form.Item
            key={param.name}
            name={[prefix, param.name]}
            label={<ParamLabel displayName={param.display_name} description={param.description} />}
            valuePropName={isSwitch ? 'checked' : 'value'}
            initialValue={param.default ?? undefined}
          >
            {renderParamField(param)}
          </Form.Item>
        );
      })}
    </>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
const DLConfig = () => {
  const [form] = Form.useForm();
  const navigate = useNavigate();

  const [datasets, setDatasets] = useState([]);
  const [registry, setRegistry] = useState({ categories: [], models: [], optimizer_params: [], train_params: [] });
  const [activeCategoryTab, setActiveCategoryTab] = useState('all');
  const [advancedMode, setAdvancedMode] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const selectedDatasetId = Form.useWatch('dataset_id', form);
  const selectedModelType = Form.useWatch('model_type', form);

  const selectedDataset = useMemo(
    () => datasets.find(d => d.id === selectedDatasetId) ?? null,
    [datasets, selectedDatasetId]
  );

  const targetOptions = useMemo(() => {
    return Object.keys(selectedDataset?.columns_info ?? {}).map(col => ({ label: col, value: col }));
  }, [selectedDataset]);

  const selectedModelSpec = useMemo(
    () => registry.models.find(m => m.id === selectedModelType) ?? null,
    [registry.models, selectedModelType]
  );

  const filteredModels = useMemo(() => {
    if (activeCategoryTab === 'all') return registry.models;
    return registry.models.filter(m => m.category === activeCategoryTab);
  }, [registry.models, activeCategoryTab]);

  // ── Load data on mount ────────────────────────────────────────────────────
  useEffect(() => {
    void loadPageData();
  }, []);

  async function loadPageData() {
    try {
      const [datasetResp, registryResp] = await Promise.all([
        dataApi.listDatasets(),
        dlApi.listModels(),
      ]);
      setDatasets(datasetResp.items ?? []);
      setRegistry({
        categories: registryResp.categories ?? [],
        models: registryResp.models ?? [],
        optimizer_params: registryResp.optimizer_params ?? [],
        train_params: registryResp.train_params ?? [],
      });
    } catch (err) {
      console.error('加载深度学习配置失败:', err);
      message.error('加载深度学习配置失败');
    }
  }

  // ── Auto-select last column as target when dataset changes ────────────────
  useEffect(() => {
    if (!selectedDataset) return;
    const cols = Object.keys(selectedDataset.columns_info ?? {});
    const current = form.getFieldValue('target_column');
    if (current && cols.includes(current)) return;
    form.setFieldValue('target_column', cols.at(-1));
  }, [form, selectedDataset]);

  // ── Clear model when category tab changes ─────────────────────────────────
  useEffect(() => {
    const currentModel = form.getFieldValue('model_type');
    if (currentModel && activeCategoryTab !== 'all') {
      const stillVisible = registry.models.find(
        m => m.id === currentModel && m.category === activeCategoryTab
      );
      if (!stillVisible) form.setFieldValue('model_type', undefined);
    }
  }, [activeCategoryTab, form, registry.models]);

  // ── Submit ────────────────────────────────────────────────────────────────
  async function handleSubmit(values) {
    setSubmitting(true);
    try {
      const task = await dlApi.startTraining({
        dataset_id: values.dataset_id,
        target_column: values.target_column,
        model_type: values.model_type,
        task_type: values.task_type || 'auto',
        arch_config: values.arch_config || {},
        opt_config: values.opt_config || {},
        train_config: values.train_config || {},
      });
      message.success('深度学习训练任务已启动');
      navigate(`/dl/monitor?taskId=${task.id}`);
    } catch (err) {
      console.error('启动深度学习训练失败:', err);
      message.error('启动训练失败');
    } finally {
      setSubmitting(false);
    }
  }

  // ── Category tabs: 全部 + one tab per category ────────────────────────────
  const categoryTabItems = [
    { key: 'all', label: '全部' },
    ...registry.categories.map(cat => ({
      key: cat.id,
      label: `${cat.icon ?? ''} ${cat.display_name}`.trim(),
    })),
  ];

  const taskTypeOptions = [
    { label: '自动检测', value: 'auto' },
    { label: '分类', value: 'classification' },
    { label: '回归', value: 'regression' },
  ];

  return (
    <div>
      <Title level={2}>深度学习训练配置</Title>
      <Card>
        <Paragraph type="secondary" style={{ marginBottom: 0 }}>
          配置深度学习模型的网络结构、优化器和训练参数，然后启动训练任务。
        </Paragraph>

        <Form
          form={form}
          layout="vertical"
          style={{ marginTop: 24 }}
          initialValues={{ task_type: 'auto' }}
          onFinish={values => void handleSubmit(values)}
        >
          {/* ── 数据 & 任务 ────────────────────────────────────────────────── */}
          <Divider orientation="left">
            <Space>
              <DatabaseIcon />
              数据 &amp; 任务
            </Space>
          </Divider>

          <Form.Item
            label="数据集"
            name="dataset_id"
            rules={[{ required: true, message: '请选择数据集' }]}
          >
            <Select
              placeholder="请选择数据集"
              options={datasets.map(d => ({
                label: `${d.name} (${d.row_count ?? 0} 行 / ${d.column_count ?? 0} 列)`,
                value: d.id,
              }))}
            />
          </Form.Item>

          <Form.Item
            label="目标列"
            name="target_column"
            rules={[{ required: true, message: '请选择目标列' }]}
          >
            <Select
              placeholder="请选择目标列"
              options={targetOptions}
              disabled={!selectedDataset}
            />
          </Form.Item>

          <Form.Item label="任务类型" name="task_type">
            <Select options={taskTypeOptions} />
          </Form.Item>

          {/* ── 模型选择 ─────────────────────────────────────────────────── */}
          <Divider orientation="left">
            <Space>
              <SettingOutlined />
              模型选择
            </Space>
          </Divider>

          <Tabs
            activeKey={activeCategoryTab}
            onChange={setActiveCategoryTab}
            items={categoryTabItems}
            size="small"
            style={{ marginBottom: 16 }}
          />

          <Form.Item
            label="模型"
            name="model_type"
            rules={[{ required: true, message: '请选择模型' }]}
          >
            <Select
              placeholder="请选择深度学习模型"
              options={filteredModels.map(m => ({
                label: m.display_name,
                value: m.id,
              }))}
            />
          </Form.Item>

          {selectedModelSpec && (
            <Card size="small" style={{ marginBottom: 16, background: '#f9f9f9' }}>
              <Text type="secondary">{selectedModelSpec.description}</Text>
            </Card>
          )}

          {/* ── 网络结构参数 ──────────────────────────────────────────────── */}
          {selectedModelSpec?.arch_params?.length > 0 && (
            <>
              <Divider orientation="left">
                <Space>
                  <SettingOutlined />
                  网络结构参数
                  <Switch
                    size="small"
                    checked={advancedMode}
                    onChange={setAdvancedMode}
                    checkedChildren="高级"
                    unCheckedChildren="基础"
                  />
                </Space>
              </Divider>
              <ParamGroup
                params={selectedModelSpec.arch_params}
                prefix="arch_config"
                advancedMode={advancedMode}
              />
            </>
          )}

          {/* ── 优化器配置 ───────────────────────────────────────────────── */}
          {registry.optimizer_params.length > 0 && (
            <>
              <Divider orientation="left">
                <Space>
                  <SettingOutlined />
                  优化器配置
                  <Switch
                    size="small"
                    checked={advancedMode}
                    onChange={setAdvancedMode}
                    checkedChildren="高级"
                    unCheckedChildren="基础"
                  />
                </Space>
              </Divider>
              <ParamGroup
                params={registry.optimizer_params}
                prefix="opt_config"
                advancedMode={advancedMode}
              />
            </>
          )}

          {/* ── 训练控制 ─────────────────────────────────────────────────── */}
          {registry.train_params.length > 0 && (
            <>
              <Divider orientation="left">
                <Space>
                  <SettingOutlined />
                  训练控制
                  <Switch
                    size="small"
                    checked={advancedMode}
                    onChange={setAdvancedMode}
                    checkedChildren="高级"
                    unCheckedChildren="基础"
                  />
                </Space>
              </Divider>
              <ParamGroup
                params={registry.train_params}
                prefix="train_config"
                advancedMode={advancedMode}
              />
            </>
          )}

          {/* ── Submit ──────────────────────────────────────────────────── */}
          <Form.Item style={{ marginTop: 24 }}>
            <Button
              type="primary"
              htmlType="submit"
              icon={<RocketOutlined />}
              loading={submitting}
              size="large"
            >
              启动训练
            </Button>
          </Form.Item>
        </Form>
      </Card>
    </div>
  );
};

// Inline icon to avoid adding more antd icon imports
function DatabaseIcon() {
  return <span style={{ fontSize: 14 }}>📂</span>;
}

export default DLConfig;
