import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Form,
  Input,
  Modal,
  Row,
  Space,
  Table,
  Tag,
  Typography,
  message,
} from 'antd';
import {
  ApiOutlined,
  BarChartOutlined,
  CloudUploadOutlined,
  CopyOutlined,
  DeleteOutlined,
  EyeOutlined,
  PlayCircleOutlined,
} from '@ant-design/icons';
import * as echarts from 'echarts';
import api, { dataApi, deployApi, modelApi } from '../services/api';
import { formatBytes, formatDateTime, formatMetric, metricLabels } from '../utils/formatters';

const { Paragraph, Text, Title } = Typography;
const { TextArea } = Input;

const metricOrder = [
  'accuracy',
  'f1',
  'precision',
  'recall',
  'roc_auc',
  'cv_avg_accuracy',
  'cv_avg_f1',
];

function sanitizePreviewRow(row, targetColumn) {
  if (!row) {
    return {};
  }

  const nextRow = { ...row };
  delete nextRow[targetColumn];
  if (targetColumn === 'Target') {
    delete nextRow['Failure Type'];
  }
  if (targetColumn === 'Failure Type') {
    delete nextRow.Target;
  }
  return nextRow;
}

const ModelManagement = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedRowKeys, setSelectedRowKeys] = useState([]);
  const [compareModalOpen, setCompareModalOpen] = useState(false);
  const [compareData, setCompareData] = useState([]);
  const [detailModalOpen, setDetailModalOpen] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  const [predictionPayload, setPredictionPayload] = useState('[]');
  const [predictionResult, setPredictionResult] = useState('');
  const [predictionRunning, setPredictionRunning] = useState(false);
  const [deployTarget, setDeployTarget] = useState(null);
  const [deployLoading, setDeployLoading] = useState(false);
  const [deployForm] = Form.useForm();
  const compareChartRef = useRef(null);

  useEffect(() => {
    void loadModels();
  }, []);

  useEffect(() => {
    if (!compareModalOpen || !compareChartRef.current || compareData.length === 0) {
      return undefined;
    }

    const chart = echarts.init(compareChartRef.current);
    const metricKeys = metricOrder.filter((metricName) =>
      compareData.some((item) => typeof item.result_metrics?.[metricName] === 'number')
    );

    chart.setOption({
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { bottom: 0 },
      grid: { top: 24, left: 56, right: 20, bottom: 56 },
      xAxis: {
        type: 'category',
        data: metricKeys.map((metricName) => metricLabels[metricName] ?? metricName),
      },
      yAxis: { type: 'value', min: 0, max: 1 },
      series: compareData.map((item) => ({
        name: `${item.model_type} · ${item.dataset_name ?? item.task_id.slice(0, 8)}`,
        type: 'bar',
        data: metricKeys.map((metricName) => item.result_metrics?.[metricName] ?? 0),
      })),
    });

    const handleResize = () => chart.resize();
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.dispose();
    };
  }, [compareData, compareModalOpen]);

  const selectedModels = useMemo(
    () => models.filter((model) => selectedRowKeys.includes(model.task_id)),
    [models, selectedRowKeys]
  );

  async function loadModels() {
    setLoading(true);
    try {
      const response = await modelApi.listModels({ page_size: 50 });
      setModels(response.items ?? []);
    } catch (error) {
      console.error('加载模型列表失败:', error);
      message.error('加载模型列表失败');
    } finally {
      setLoading(false);
    }
  }

  async function openModelDetail(record) {
    setDetailModalOpen(true);
    setDetailLoading(true);
    setPredictionResult('');
    try {
      const [detail, preview] = await Promise.all([
        modelApi.getModelDetail(record.task_id),
        dataApi.previewDataset(record.dataset_id),
      ]);

      setSelectedModel(detail);
      setPredictionPayload(JSON.stringify([sanitizePreviewRow(preview.rows?.[0], detail.target_column)], null, 2));
    } catch (error) {
      console.error('加载模型详情失败:', error);
      message.error('加载模型详情失败');
    } finally {
      setDetailLoading(false);
    }
  }

  async function handleDeleteModel(taskId) {
    try {
      await modelApi.deleteModel(taskId);
      message.success('模型文件已删除');
      setSelectedRowKeys((current) => current.filter((key) => key !== taskId));
      await loadModels();
    } catch (error) {
      console.error('删除模型失败:', error);
      message.error('删除模型失败');
    }
  }

  async function handleCompareModels() {
    if (selectedRowKeys.length < 2) {
      message.warning('至少选择两个模型再进行对比');
      return;
    }

    try {
      const comparison = await modelApi.compareModels(selectedRowKeys);
      setCompareData(comparison);
      setCompareModalOpen(true);
    } catch (error) {
      console.error('加载模型对比失败:', error);
      message.error('加载模型对比失败');
    }
  }

  async function handlePredict() {
    if (!selectedModel) {
      return;
    }

    let rows;
    try {
      rows = JSON.parse(predictionPayload);
    } catch (error) {
      message.error('预测输入必须是合法的 JSON 数组');
      return;
    }

    if (!Array.isArray(rows) || rows.length === 0) {
      message.error('请至少提供一行待预测数据');
      return;
    }

    setPredictionRunning(true);
    try {
      const result = await modelApi.predict(selectedModel.task_id, {
        rows,
        include_probabilities: true,
      });
      setPredictionResult(JSON.stringify(result, null, 2));
    } catch (error) {
      console.error('模型预测失败:', error);
      const detail = error?.response?.data?.detail;
      message.error(typeof detail === 'string' ? detail : '模型预测失败');
    } finally {
      setPredictionRunning(false);
    }
  }

  async function handleDeploy(values) {
    if (!deployTarget) return;
    setDeployLoading(true);
    try {
      await deployApi.createDeployment(deployTarget.task_id, {
        name: values.name,
        description: values.description || null,
        max_batch_size: 100,
      });
      message.success('部署成功，请前往「模型部署」页查看接口 URL');
      setDeployTarget(null);
      deployForm.resetFields();
    } catch (e) {
      message.error('部署失败: ' + (e?.response?.data?.detail || e?.message || '未知错误'));
    } finally {
      setDeployLoading(false);
    }
  }

  const predictionUrl = selectedModel
    ? `${api.defaults.baseURL}/models/${selectedModel.task_id}/predict`
    : '';

  return (
    <Space direction="vertical" size={20} style={{ width: '100%' }}>
      <Card>
        <Space direction="vertical" size={12} style={{ width: '100%' }}>
          <Title level={2} style={{ margin: 0 }}>
            模型管理
          </Title>
          <Paragraph type="secondary" style={{ marginBottom: 0 }}>
            这里只显示真实保存下来的模型文件。你可以查看训练细节、删除模型文件、对比多模型指标，也可以直接调用预测接口。
          </Paragraph>
          <Space wrap>
            <Button
              type="primary"
              icon={<BarChartOutlined />}
              disabled={selectedRowKeys.length < 2}
              onClick={handleCompareModels}
            >
              对比已选模型 ({selectedRowKeys.length}/3)
            </Button>
            <Text type="secondary">最多同时对比 3 个模型</Text>
          </Space>
        </Space>
      </Card>

      <Card>
        <Table
          loading={loading}
          rowKey="task_id"
          dataSource={models}
          locale={{ emptyText: <Empty description="还没有已保存模型" /> }}
          rowSelection={{
            selectedRowKeys,
            onChange: (nextKeys) => {
              if (nextKeys.length > 3) {
                message.warning('最多只能同时选择 3 个模型');
                return;
              }
              setSelectedRowKeys(nextKeys);
            },
          }}
          columns={[
            {
              title: '数据集',
              dataIndex: 'dataset_name',
              render: (value, record) => value ?? record.dataset_id,
            },
            {
              title: '模型',
              dataIndex: 'model_type',
              render: (value) => <Tag color="blue">{value}</Tag>,
            },
            {
              title: '目标列',
              dataIndex: 'target_column',
            },
            {
              title: '准确率',
              render: (_, record) => formatMetric(record.result_metrics?.accuracy, { percent: true }),
            },
            {
              title: 'F1',
              render: (_, record) => formatMetric(record.result_metrics?.f1),
            },
            {
              title: '大小',
              dataIndex: 'model_size',
              render: (value) => formatBytes(value),
            },
            {
              title: '完成时间',
              dataIndex: 'finished_at',
              render: (value) => formatDateTime(value),
            },
            {
              title: '操作',
              render: (_, record) => (
                <Space size="small">
                  <Button type="text" icon={<EyeOutlined />} onClick={() => void openModelDetail(record)}>
                    详情
                  </Button>
                  <Button
                    type="text"
                    icon={<CloudUploadOutlined />}
                    onClick={() => { setDeployTarget(record); deployForm.setFieldValue('name', `${record.model_type}-deploy`); }}
                  >
                    部署
                  </Button>
                  <Button type="text" danger icon={<DeleteOutlined />} onClick={() => void handleDeleteModel(record.task_id)}>
                    删除
                  </Button>
                </Space>
              ),
            },
          ]}
        />
      </Card>

      <Modal
        title="模型详情与预测接口"
        open={detailModalOpen}
        onCancel={() => setDetailModalOpen(false)}
        footer={null}
        width={980}
        destroyOnHidden
      >
        {detailLoading || !selectedModel ? (
          <Card loading />
        ) : (
          <Space direction="vertical" size={16} style={{ width: '100%' }}>
            <Descriptions bordered column={{ xs: 1, md: 2, xl: 3 }}>
              <Descriptions.Item label="数据集">{selectedModel.dataset?.name ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="模型">{selectedModel.model_type}</Descriptions.Item>
              <Descriptions.Item label="目标列">{selectedModel.target_column}</Descriptions.Item>
              <Descriptions.Item label="状态">{selectedModel.status}</Descriptions.Item>
              <Descriptions.Item label="完成时间">{formatDateTime(selectedModel.finished_at)}</Descriptions.Item>
              <Descriptions.Item label="模型文件">{selectedModel.model_path ?? '-'}</Descriptions.Item>
            </Descriptions>

            <Row gutter={[16, 16]}>
              {metricOrder
                .filter((metricName) => typeof selectedModel.result_metrics?.[metricName] === 'number')
                .slice(0, 4)
                .map((metricName) => (
                  <Col xs={24} sm={12} xl={6} key={metricName}>
                    <Card size="small">
                      <Text type="secondary">{metricLabels[metricName] ?? metricName}</Text>
                      <Title level={4} style={{ marginTop: 8, marginBottom: 0 }}>
                        {metricName.includes('accuracy')
                          ? formatMetric(selectedModel.result_metrics?.[metricName], { percent: true })
                          : formatMetric(selectedModel.result_metrics?.[metricName])}
                      </Title>
                    </Card>
                  </Col>
                ))}
            </Row>

            <Card
              size="small"
              title={<Space><ApiOutlined /> 预测接口</Space>}
              extra={
                <Button
                  icon={<CopyOutlined />}
                  onClick={async () => {
                    await navigator.clipboard.writeText(predictionUrl);
                    message.success('接口地址已复制');
                  }}
                >
                  复制 URL
                </Button>
              }
            >
              <Space direction="vertical" size={12} style={{ width: '100%' }}>
                <Alert
                  type="info"
                  showIcon
                  message="向下面的 URL 发送 POST 请求，Body 传入 rows 数组即可拿到预测结果。"
                />
                <Text code data-testid="prediction-url" style={{ display: 'block', whiteSpace: 'pre-wrap' }}>
                  {predictionUrl}
                </Text>
                <Text strong>请求体示例</Text>
                <TextArea
                  value={predictionPayload}
                  onChange={(event) => setPredictionPayload(event.target.value)}
                  rows={12}
                  data-testid="prediction-payload"
                />
                <Space>
                  <Button
                    type="primary"
                    icon={<PlayCircleOutlined />}
                    loading={predictionRunning}
                    onClick={() => void handlePredict()}
                    data-testid="run-prediction-button"
                  >
                    运行预测
                  </Button>
                </Space>
                <Text strong>预测结果</Text>
                <pre
                  data-testid="prediction-result"
                  style={{
                    margin: 0,
                    padding: 16,
                    borderRadius: 12,
                    background: '#0f172a',
                    color: '#e2e8f0',
                    minHeight: 120,
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                  }}
                >
                  {predictionResult || '运行一次预测后，这里会显示后端返回的 JSON 结果。'}
                </pre>
              </Space>
            </Card>
          </Space>
        )}
      </Modal>

      <Modal
        title="模型指标对比"
        open={compareModalOpen}
        onCancel={() => setCompareModalOpen(false)}
        footer={null}
        width={980}
        destroyOnHidden
      >
        {compareData.length === 0 ? (
          <Alert type="info" showIcon message="还没有可展示的对比结果。" />
        ) : (
          <div ref={compareChartRef} style={{ width: '100%', height: 420 }} />
        )}
      </Modal>

      <Modal
        title={<Space><CloudUploadOutlined /> 部署模型</Space>}
        open={!!deployTarget}
        onCancel={() => { setDeployTarget(null); deployForm.resetFields(); }}
        onOk={() => deployForm.submit()}
        okText="确认部署"
        confirmLoading={deployLoading}
        destroyOnHidden
      >
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message={`将为模型 ${deployTarget?.model_type ?? ''} 创建部署，部署成功后可在「模型部署」页查看接口 URL。`}
        />
        <Form form={deployForm} layout="vertical" onFinish={(v) => void handleDeploy(v)}>
          <Form.Item
            label="部署名称"
            name="name"
            rules={[{ required: true, message: '请输入部署名称' }]}
          >
            <Input placeholder="例：生产预测服务-v1" />
          </Form.Item>
          <Form.Item label="描述（可选）" name="description">
            <Input.TextArea rows={2} placeholder="简短描述此部署的用途" />
          </Form.Item>
        </Form>
      </Modal>
    </Space>
  );
};

export default ModelManagement;
