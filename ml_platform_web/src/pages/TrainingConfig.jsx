import React, { useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Alert, Button, Card, Divider, Form, InputNumber, Select, Space, Switch,
  Tag, Typography, message,
} from 'antd';
import { RocketOutlined, SettingOutlined } from '@ant-design/icons';
import { dataApi, trainingApi } from '../services/api';
import ModelSelector from '../components/ModelSelector';
import DynamicParamForm from '../components/DynamicParamForm';

const { Paragraph, Text, Title } = Typography;

// Parse list-type hyperparams from comma-separated string → number[]
function parseHyperparameters(raw = {}) {
  const result = {};
  for (const [k, v] of Object.entries(raw)) {
    if (typeof v === 'string' && v.includes(',')) {
      const nums = v.split(',').map(s => Number(s.trim())).filter(n => !Number.isNaN(n));
      result[k] = nums.length > 0 ? nums : v;
    } else {
      result[k] = v;
    }
  }
  return result;
}

const TrainingConfig = () => {
  const [form] = Form.useForm();
  const navigate = useNavigate();
  const location = useLocation();

  const [datasets, setDatasets] = useState([]);
  const [registryData, setRegistryData] = useState({
    categories: [], models: [], classificationMetrics: [], regressionMetrics: [],
  });
  const [advancedMode, setAdvancedMode] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const selectedDatasetId = Form.useWatch('dataset_id', form);
  const selectedModel = Form.useWatch('model_type', form);

  const selectedModelSpec = useMemo(
    () => registryData.models.find(m => m.id === selectedModel) ?? null,
    [registryData.models, selectedModel]
  );
  const isRegression = selectedModelSpec?.task_types?.includes('regression') && !selectedModelSpec?.task_types?.includes('classification');
  const metricOptions = isRegression ? registryData.regressionMetrics : registryData.classificationMetrics;

  const selectedDataset = useMemo(
    () => datasets.find(d => d.id === selectedDatasetId) ?? null,
    [datasets, selectedDatasetId]
  );
  const targetOptions = useMemo(() => {
    return Object.keys(selectedDataset?.columns_info ?? {}).map(col => ({ label: col, value: col }));
  }, [selectedDataset]);

  // ── Load page data ──────────────────────────────────────────────────────────
  useEffect(() => { void loadPageData(); }, []);

  async function loadPageData() {
    try {
      const [datasetResponse, modelResponse] = await Promise.all([
        dataApi.listDatasets(),
        trainingApi.listModels(),
      ]);
      setDatasets(datasetResponse.items ?? []);
      setRegistryData({
        categories: modelResponse.categories ?? [],
        models: modelResponse.models ?? [],
        classificationMetrics: (modelResponse.classification_metrics ?? []).map(m => ({
          label: m.label, value: m.value,
        })),
        regressionMetrics: (modelResponse.regression_metrics ?? []).map(m => ({
          label: m.label, value: m.value,
        })),
      });
    } catch (error) {
      console.error('加载训练配置失败:', error);
      message.error('加载训练配置失败');
    }
  }

  // ── Auto-select dataset from navigation state ───────────────────────────────
  useEffect(() => {
    const preferredId = location.state?.preferredDatasetId;
    if (!preferredId || datasets.length === 0) return;
    if (datasets.some(d => d.id === preferredId)) {
      form.setFieldValue('dataset_id', preferredId);
    }
  }, [datasets, form, location.state]);

  // ── Auto-select default target column ──────────────────────────────────────
  useEffect(() => {
    if (!selectedDataset) return;
    const cols = Object.keys(selectedDataset.columns_info ?? {});
    const current = form.getFieldValue('target_column');
    if (current && cols.includes(current)) return;
    const defaultTarget = cols.includes('Target') ? 'Target' : cols.at(-1);
    form.setFieldValue('target_column', defaultTarget);
  }, [form, selectedDataset]);

  // ── Reset eval_metrics when task type changes ───────────────────────────────
  useEffect(() => {
    if (!selectedModelSpec) return;
    const defaults = selectedModelSpec.task_types?.includes('regression') &&
      !selectedModelSpec.task_types?.includes('classification')
      ? ['rmse', 'r2']
      : ['accuracy', 'f1'];
    form.setFieldValue('eval_metrics', defaults);
  }, [selectedModelSpec, form]);

  // ── Submit ──────────────────────────────────────────────────────────────────
  async function handleSubmit(values) {
    setSubmitting(true);
    try {
      // Lift class_weight from hyperparameters to top-level field
      const { class_weight: classWeight, ...rawHp } = values.hyperparameters ?? {};
      const hyperparameters = parseHyperparameters(rawHp);

      const task = await trainingApi.startTraining({
        dataset_id: values.dataset_id,
        target_column: values.target_column,
        model_type: values.model_type,
        hyperparameters,
        test_size: values.test_size ?? 0.2,
        eval_metrics: values.eval_metrics ?? [],
        class_weight: classWeight ?? null,
        cross_validation: { enabled: true, folds: values.cv_folds ?? 5 },
      });

      message.success('训练任务已启动');
      navigate(`/training/monitor?taskId=${task.id}`);
    } catch (error) {
      console.error('启动训练失败:', error);
      message.error('启动训练失败');
    } finally {
      setSubmitting(false);
    }
  }

  const selectedTarget = Form.useWatch('target_column', form);

  return (
    <div>
      <Title level={2}>训练配置</Title>
      <Card>
        <Space direction="vertical" size={20} style={{ width: '100%' }}>
          <Paragraph type="secondary" style={{ marginBottom: 0 }}>
            从已上传的数据集中选择目标列和模型，前端会直接向后端提交训练任务。
          </Paragraph>

          {selectedTarget === 'Target' && selectedDataset?.columns_info?.['Failure Type'] ? (
            <Alert
              type="warning"
              showIcon
              message="当前数据集同时包含 Failure Type 和 Target。后端已自动排除泄漏列。"
            />
          ) : null}

          <Form
            form={form}
            layout="vertical"
            initialValues={{ test_size: 0.2, eval_metrics: ['accuracy', 'f1'], cv_folds: 5 }}
            onFinish={values => void handleSubmit(values)}
          >
            {/* ── Dataset & Target ── */}
            <Form.Item label="数据集" name="dataset_id" rules={[{ required: true, message: '请选择数据集' }]}>
              <Select
                data-testid="dataset-select"
                placeholder="请选择数据集"
                options={datasets.map(d => ({
                  label: `${d.name} (${d.row_count ?? 0} 行 / ${d.column_count ?? 0} 列)`,
                  value: d.id,
                }))}
              />
            </Form.Item>

            <Form.Item label="目标列" name="target_column" rules={[{ required: true, message: '请选择目标列' }]}>
              <Select
                data-testid="target-select"
                placeholder="请选择目标列"
                options={targetOptions}
                disabled={!selectedDataset}
              />
            </Form.Item>

            {/* ── Model Selector ── */}
            <Form.Item
              label={
                <Space>
                  模型类型
                  {selectedModelSpec && (
                    <Tag color={isRegression ? 'green' : 'blue'}>
                      {isRegression ? '回归' : '分类'}
                    </Tag>
                  )}
                </Space>
              }
              name="model_type"
              rules={[{ required: true, message: '请选择模型类型' }]}
            >
              <ModelSelector
                models={registryData.models}
                categories={registryData.categories}
                value={selectedModel}
                onChange={v => form.setFieldValue('model_type', v)}
              />
            </Form.Item>

            {/* ── Common Params ── */}
            <Space size={16} wrap style={{ width: '100%', marginBottom: 8 }}>
              <Form.Item label="测试集比例" name="test_size" style={{ marginBottom: 0 }}>
                <InputNumber min={0.05} max={0.5} step={0.05} style={{ width: 120 }} />
              </Form.Item>
              <Form.Item label="交叉验证折数" name="cv_folds" style={{ marginBottom: 0 }}>
                <InputNumber min={2} max={20} step={1} style={{ width: 120 }} />
              </Form.Item>
            </Space>

            {/* ── Model-specific Params ── */}
            {selectedModelSpec && (
              <>
                <Divider orientation="left" style={{ marginTop: 8, marginBottom: 16 }}>
                  <Space>
                    <SettingOutlined />
                    模型参数
                    <Switch
                      size="small"
                      checked={advancedMode}
                      onChange={setAdvancedMode}
                      checkedChildren="高级"
                      unCheckedChildren="基础"
                    />
                  </Space>
                </Divider>
                <DynamicParamForm
                  params={selectedModelSpec.params ?? []}
                  advancedMode={advancedMode}
                />
              </>
            )}

            {/* ── Eval Metrics ── */}
            <Form.Item label="评估指标" name="eval_metrics">
              <Select
                mode="multiple"
                options={metricOptions}
                placeholder="请选择评估指标"
              />
            </Form.Item>

            {/* ── Dataset Info ── */}
            {selectedDataset ? (
              <Card size="small" style={{ marginBottom: 24 }}>
                <Space direction="vertical" size={4}>
                  <Text>当前数据集: {selectedDataset.name}</Text>
                  <Text type="secondary">
                    列: {Object.keys(selectedDataset.columns_info ?? {}).join(', ')}
                  </Text>
                </Space>
              </Card>
            ) : null}

            <Button
              type="primary"
              htmlType="submit"
              icon={<RocketOutlined />}
              loading={submitting}
              data-testid="start-training-button"
            >
              启动训练
            </Button>
          </Form>
        </Space>
      </Card>
    </div>
  );
};

export default TrainingConfig;
