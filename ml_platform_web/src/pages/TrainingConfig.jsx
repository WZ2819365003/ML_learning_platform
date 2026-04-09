import React, { useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Alert, Button, Card, Form, InputNumber, Select, Space, Typography, message } from 'antd';
import { RocketOutlined } from '@ant-design/icons';
import { dataApi, trainingApi } from '../services/api';

const { Paragraph, Text, Title } = Typography;

const metricOptions = [
  { label: 'accuracy', value: 'accuracy' },
  { label: 'f1', value: 'f1' },
];

const TrainingConfig = () => {
  const [form] = Form.useForm();
  const navigate = useNavigate();
  const location = useLocation();
  const [datasets, setDatasets] = useState([]);
  const [models, setModels] = useState([]);
  const [submitting, setSubmitting] = useState(false);

  const selectedDatasetId = Form.useWatch('dataset_id', form);
  const selectedTarget = Form.useWatch('target_column', form);
  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === selectedDatasetId) ?? null,
    [datasets, selectedDatasetId]
  );
  const targetOptions = useMemo(() => {
    const columnsInfo = selectedDataset?.columns_info ?? {};
    return Object.keys(columnsInfo).map((column) => ({
      label: column,
      value: column,
    }));
  }, [selectedDataset]);

  useEffect(() => {
    void loadPageData();
  }, []);

  useEffect(() => {
    const preferredDatasetId = location.state?.preferredDatasetId;
    if (!preferredDatasetId || datasets.length === 0) {
      return;
    }

    const datasetExists = datasets.some((dataset) => dataset.id === preferredDatasetId);
    if (datasetExists) {
      form.setFieldValue('dataset_id', preferredDatasetId);
    }
  }, [datasets, form, location.state]);

  useEffect(() => {
    if (!selectedDataset) {
      return;
    }

    const columns = Object.keys(selectedDataset.columns_info ?? {});
    const currentTarget = form.getFieldValue('target_column');
    if (currentTarget && columns.includes(currentTarget)) {
      return;
    }

    const defaultTarget = columns.includes('Target') ? 'Target' : columns.at(-1);
    form.setFieldValue('target_column', defaultTarget);
  }, [form, selectedDataset]);

  useEffect(() => {
    if (models.length === 0) {
      return;
    }

    const currentModel = form.getFieldValue('model_type');
    if (currentModel && models.includes(currentModel)) {
      return;
    }

    const defaultModel = models.includes('random_forest') ? 'random_forest' : models[0];
    form.setFieldValue('model_type', defaultModel);
  }, [form, models]);

  async function loadPageData() {
    try {
      const [datasetResponse, modelResponse] = await Promise.all([
        dataApi.listDatasets(),
        trainingApi.listModels(),
      ]);

      setDatasets(datasetResponse.items ?? []);
      setModels(Array.isArray(modelResponse) ? modelResponse : modelResponse.value ?? []);
    } catch (error) {
      console.error('加载训练配置失败:', error);
      message.error('加载训练配置失败');
    }
  }

  async function handleSubmit(values) {
    setSubmitting(true);
    try {
      const task = await trainingApi.startTraining({
        dataset_id: values.dataset_id,
        target_column: values.target_column,
        model_type: values.model_type,
        hyperparameters: {
          n_estimators: values.n_estimators,
          max_depth: values.max_depth,
        },
        test_size: values.test_size,
        eval_metrics: values.eval_metrics,
        cross_validation: {
          enabled: true,
          folds: values.cv_folds,
        },
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
            initialValues={{
              test_size: 0.2,
              eval_metrics: ['accuracy', 'f1'],
              n_estimators: 100,
              max_depth: 6,
              cv_folds: 3,
            }}
            onFinish={(values) => void handleSubmit(values)}
          >
            <Form.Item
              label="数据集"
              name="dataset_id"
              rules={[{ required: true, message: '请选择数据集' }]}
            >
              <Select
                data-testid="dataset-select"
                placeholder="请选择数据集"
                options={datasets.map((dataset) => ({
                  label: `${dataset.name} (${dataset.row_count ?? 0} 行 / ${dataset.column_count ?? 0} 列)`,
                  value: dataset.id,
                }))}
              />
            </Form.Item>

            <Form.Item
              label="目标列"
              name="target_column"
              rules={[{ required: true, message: '请选择目标列' }]}
            >
              <Select
                data-testid="target-select"
                placeholder="请选择目标列"
                options={targetOptions}
                disabled={!selectedDataset}
              />
            </Form.Item>

            <Form.Item
              label="模型类型"
              name="model_type"
              rules={[{ required: true, message: '请选择模型类型' }]}
            >
              <Select
                data-testid="model-select"
                placeholder="请选择模型类型"
                options={models.map((model) => ({ label: model, value: model }))}
              />
            </Form.Item>

            <Space size={16} wrap style={{ width: '100%', marginBottom: 16 }}>
              <Form.Item label="测试集比例" name="test_size" style={{ marginBottom: 0 }}>
                <InputNumber min={0.1} max={0.5} step={0.05} />
              </Form.Item>
              <Form.Item label="n_estimators" name="n_estimators" style={{ marginBottom: 0 }}>
                <InputNumber min={10} max={500} step={10} />
              </Form.Item>
              <Form.Item label="max_depth" name="max_depth" style={{ marginBottom: 0 }}>
                <InputNumber min={2} max={20} />
              </Form.Item>
              <Form.Item label="交叉验证折数" name="cv_folds" style={{ marginBottom: 0 }}>
                <InputNumber min={2} max={10} />
              </Form.Item>
            </Space>

            <Form.Item label="评估指标" name="eval_metrics">
              <Select mode="multiple" options={metricOptions} />
            </Form.Item>

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
