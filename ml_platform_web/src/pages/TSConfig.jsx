import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Alert,
  Badge,
  Button,
  Card,
  Col,
  Empty,
  Form,
  InputNumber,
  Popconfirm,
  Radio,
  Row,
  Select,
  Space,
  Spin,
  Table,
  Tag,
  Typography,
  message,
} from 'antd';
import {
  BarChartOutlined,
  CloudDownloadOutlined,
  DeleteOutlined,
  EyeOutlined,
  LineChartOutlined,
  MonitorOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import { dataApi, timesfmApi } from '../services/api';
import { formatDateTime } from '../utils/formatters';

const { Title, Text } = Typography;

const FREQ_LABELS = { high: '高频（日数据）', medium: '中频（周数据）', low: '低频（月数据）' };
const STATUS_COLOR = { PENDING: 'default', RUNNING: 'processing', SUCCESS: 'success', FAILED: 'error' };
const STATUS_LABEL = { PENDING: '等待中', RUNNING: '运行中', SUCCESS: '完成', FAILED: '失败' };
const MODEL_OPTIONS = [
  { value: 'amazon/chronos-t5-tiny',  label: 'Chronos-T5-Tiny (~8M) — 最快' },
  { value: 'amazon/chronos-t5-small', label: 'Chronos-T5-Small (~46M) — 默认' },
  { value: 'amazon/chronos-t5-base',  label: 'Chronos-T5-Base (~200M)' },
];
const PAGE_SIZE = 10;

export default function TSConfig() {
  const navigate = useNavigate();
  const [form] = Form.useForm();

  const [modelStatus, setModelStatus] = useState(null);
  const [statusLoading, setStatusLoading] = useState(false);
  const [preloading, setPreloading] = useState(false);

  const [datasets, setDatasets] = useState([]);
  const [columns, setColumns] = useState([]);
  const [dsLoading, setDsLoading] = useState(false);

  const [submitting, setSubmitting] = useState(false);
  const [tasks, setTasks] = useState([]);
  const [tasksLoading, setTasksLoading] = useState(false);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);

  const refreshTimer = useRef(null);

  useEffect(() => {
    void fetchModelStatus();
    void fetchDatasets();
    void fetchTasks(1);
    return () => clearInterval(refreshTimer.current);
  }, []);

  useEffect(() => {
    clearInterval(refreshTimer.current);
    const hasActive = tasks.some((t) => t.status === 'RUNNING' || t.status === 'PENDING');
    if (hasActive) {
      refreshTimer.current = setInterval(() => void fetchTasks(page), 3000);
    }
    return () => clearInterval(refreshTimer.current);
  }, [tasks, page]);

  async function fetchModelStatus() {
    setStatusLoading(true);
    try {
      const res = await timesfmApi.modelStatus();
      setModelStatus(res);
    } catch {
      setModelStatus(null);
    } finally {
      setStatusLoading(false);
    }
  }

  async function fetchDatasets() {
    setDsLoading(true);
    try {
      const res = await dataApi.listDatasets({ page: 1, page_size: 100 });
      setDatasets(res.datasets ?? res.items ?? []);
    } catch {
      message.error('加载数据集失败');
    } finally {
      setDsLoading(false);
    }
  }

  async function onDatasetChange(datasetId) {
    form.setFieldsValue({ time_column: undefined, value_column: undefined });
    setColumns([]);
    if (!datasetId) return;
    try {
      const res = await dataApi.previewDataset(datasetId);
      if (res?.rows?.[0]) setColumns(Object.keys(res.rows[0]));
    } catch {
      message.error('加载列信息失败');
    }
  }

  async function fetchTasks(p) {
    setTasksLoading(true);
    try {
      const res = await timesfmApi.listForecasts({ page: p, page_size: PAGE_SIZE });
      setTasks(res.items ?? []);
      setTotal(res.total ?? 0);
    } catch {
      message.error('加载预测记录失败');
    } finally {
      setTasksLoading(false);
    }
  }

  async function handleSubmit(values) {
    setSubmitting(true);
    try {
      const task = await timesfmApi.startForecast({
        dataset_id: values.dataset_id,
        value_column: values.value_column,
        time_column: values.time_column ?? null,
        horizon: values.horizon,
        frequency: values.frequency,
        model_name: values.model_name,
      });
      message.success('预测任务已提交，正在跳转监控页面…');
      setTimeout(() => navigate(`/ts/monitor?id=${task.id}`), 800);
    } catch (err) {
      message.error(err?.response?.data?.detail ?? '提交失败');
    } finally {
      setSubmitting(false);
    }
  }

  async function handlePreload() {
    const modelName = form.getFieldValue('model_name') ?? 'amazon/chronos-t5-small';
    setPreloading(true);
    try {
      await timesfmApi.preloadModel(modelName);
      message.info('模型后台预加载已启动（首次约需 1-3 分钟下载权重）');
    } catch (err) {
      message.error(err?.response?.data?.detail ?? '预加载失败');
    } finally {
      setPreloading(false);
    }
  }

  async function handleDelete(id) {
    try {
      await timesfmApi.deleteForecast(id);
      message.success('已删除');
      void fetchTasks(page);
    } catch (err) {
      message.error(err?.response?.data?.detail ?? '删除失败');
    }
  }

  const available = modelStatus?.available;

  const tableColumns = [
    {
      title: '数据集 / 预测列',
      key: 'name',
      render: (_, r) => (
        <Space direction="vertical" size={0}>
          <Text strong>{r.dataset_name ?? r.dataset_id?.slice(0, 8)}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>{r.value_column} · {r.horizon}步</Text>
        </Space>
      ),
    },
    {
      title: '模型',
      dataIndex: 'model_name',
      render: (v) => <Tag color="purple">{v?.split('/').pop() ?? v}</Tag>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      width: 95,
      render: (s) => <Badge status={STATUS_COLOR[s] ?? 'default'} text={STATUS_LABEL[s] ?? s} />,
    },
    {
      title: '提交时间',
      dataIndex: 'created_at',
      render: (v) => formatDateTime(v),
      width: 140,
    },
    {
      title: '操作',
      key: 'actions',
      width: 140,
      render: (_, r) => (
        <Space size={4} onClick={(e) => e.stopPropagation()}>
          <Button
            size="small"
            icon={<MonitorOutlined />}
            onClick={() => navigate(`/ts/monitor?id=${r.id}`)}
          >
            监控
          </Button>
          <Button
            size="small"
            icon={<EyeOutlined />}
            disabled={r.status !== 'SUCCESS'}
            onClick={() => navigate(`/ts/results?id=${r.id}`)}
          />
          <Popconfirm
            title="确认删除此预测记录？"
            onConfirm={() => void handleDelete(r.id)}
            disabled={r.status === 'RUNNING'}
          >
            <Button size="small" danger icon={<DeleteOutlined />} disabled={r.status === 'RUNNING'} />
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <Space direction="vertical" size={20} style={{ width: '100%' }}>
      <Row align="middle" justify="space-between" wrap={false}>
        <Col>
          <Title level={2} style={{ margin: 0 }}>时序预测配置</Title>
          <Text type="secondary">Amazon Chronos 零样本时序预测 — 选择数据集、配置参数后一键运行</Text>
        </Col>
        <Col>
          {statusLoading ? (
            <Spin size="small" />
          ) : modelStatus ? (
            <Badge
              status={available ? (modelStatus.loaded ? 'success' : 'warning') : 'error'}
              text={
                available
                  ? modelStatus.loaded
                    ? `模型就绪 · ${modelStatus.model?.split('/').pop()}`
                    : `已安装 / 未加载`
                  : 'Chronos 未安装'
              }
            />
          ) : (
            <Button size="small" onClick={() => void fetchModelStatus()}>检查状态</Button>
          )}
        </Col>
      </Row>

      {modelStatus && !available && (
        <Alert
          type="warning"
          showIcon
          message="Chronos 预测引擎未安装"
          description={
            <Space direction="vertical" size={4}>
              <Text>请在后端虚拟环境中执行：</Text>
              <Text code>pip install chronos-forecasting torch</Text>
            </Space>
          }
        />
      )}

      <Row gutter={[20, 20]}>
        {/* ── Config form ── */}
        <Col xs={24} lg={9}>
          <Card
            title={<Space><BarChartOutlined />配置预测任务</Space>}
            style={{ height: '100%' }}
          >
            <Form
              form={form}
              layout="vertical"
              initialValues={{ horizon: 24, frequency: 'high', model_name: 'amazon/chronos-t5-small' }}
              onFinish={(v) => void handleSubmit(v)}
            >
              <Form.Item
                label="① 选择数据集"
                name="dataset_id"
                rules={[{ required: true, message: '请选择数据集' }]}
              >
                <Select
                  loading={dsLoading}
                  placeholder="选择已上传的数据集"
                  showSearch
                  optionFilterProp="label"
                  options={datasets.map((d) => ({ value: d.id, label: d.name }))}
                  onChange={onDatasetChange}
                />
              </Form.Item>

              <Form.Item label="② 时间列（可选）" name="time_column">
                <Select
                  placeholder="用于 X 轴时间坐标"
                  allowClear
                  options={columns.map((c) => ({ value: c, label: c }))}
                />
              </Form.Item>

              <Form.Item
                label={
                  <>
                    ③ 预测目标列&nbsp;
                    <Tag color="red" style={{ marginLeft: 4 }}>必填</Tag>
                  </>
                }
                name="value_column"
                rules={[{ required: true, message: '请选择预测目标列' }]}
              >
                <Select
                  placeholder="选择数值列作为预测目标"
                  options={columns.map((c) => ({ value: c, label: c }))}
                />
              </Form.Item>

              <Form.Item label="预测步数（Horizon）" name="horizon">
                <InputNumber min={1} max={512} style={{ width: '100%' }} addonAfter="步" />
              </Form.Item>

              <Form.Item label="数据频率" name="frequency">
                <Radio.Group buttonStyle="solid">
                  <Radio.Button value="high">高频（日）</Radio.Button>
                  <Radio.Button value="medium">中频（周）</Radio.Button>
                  <Radio.Button value="low">低频（月）</Radio.Button>
                </Radio.Group>
              </Form.Item>

              <Form.Item label="模型规格" name="model_name">
                <Select options={MODEL_OPTIONS} />
              </Form.Item>

              <Space direction="vertical" style={{ width: '100%' }} size={8}>
                <Button
                  type="primary"
                  htmlType="submit"
                  icon={<PlayCircleOutlined />}
                  loading={submitting}
                  disabled={!available}
                  block
                  size="large"
                >
                  运行预测
                </Button>
                <Button
                  icon={<CloudDownloadOutlined />}
                  loading={preloading}
                  disabled={!available}
                  block
                  onClick={() => void handlePreload()}
                >
                  预加载模型权重
                </Button>
              </Space>
            </Form>
          </Card>
        </Col>

        {/* ── Task history ── */}
        <Col xs={24} lg={15}>
          <Card
            title={<Space><LineChartOutlined />预测任务历史</Space>}
            extra={
              <Space size={8}>
                <Button size="small" onClick={() => navigate('/ts/results')}>查看结果</Button>
                <Button size="small" icon={<ReloadOutlined />} onClick={() => void fetchTasks(page)}>刷新</Button>
              </Space>
            }
          >
            <Table
              rowKey="id"
              dataSource={tasks}
              columns={tableColumns}
              loading={tasksLoading}
              size="middle"
              pagination={{
                current: page,
                pageSize: PAGE_SIZE,
                total,
                onChange: (p) => { setPage(p); void fetchTasks(p); },
                showTotal: (t) => `共 ${t} 条`,
                showSizeChanger: false,
              }}
              locale={{ emptyText: <Empty description="暂无记录，配置参数后点击「运行预测」" /> }}
              expandable={{
                expandedRowRender: (r) =>
                  r.status === 'FAILED' ? (
                    <Alert type="error" showIcon message="预测失败" description={r.error_message ?? '未知错误'} />
                  ) : null,
                rowExpandable: (r) => r.status === 'FAILED',
              }}
            />
          </Card>
        </Col>
      </Row>
    </Space>
  );
}
