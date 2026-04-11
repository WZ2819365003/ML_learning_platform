import React, { useEffect, useRef, useState } from 'react';
import {
  Alert,
  Badge,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Form,
  InputNumber,
  Modal,
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
  PlayCircleOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import * as echarts from 'echarts';
import { dataApi, timesfmApi } from '../services/api';
import { formatDateTime } from '../utils/formatters';

const { Title, Text } = Typography;

const FREQ_LABELS = { high: '高频（日数据）', medium: '中频（周数据）', low: '低频（月数据）' };
const STATUS_COLOR = { PENDING: 'default', RUNNING: 'processing', SUCCESS: 'success', FAILED: 'error' };
const STATUS_LABEL = { PENDING: '等待中', RUNNING: '运行中', SUCCESS: '完成', FAILED: '失败' };
const MODEL_OPTIONS = [
  { value: 'amazon/chronos-t5-tiny',  label: 'Chronos-T5-Tiny  (~8M)  — 最快' },
  { value: 'amazon/chronos-t5-small', label: 'Chronos-T5-Small (~46M) — 默认' },
  { value: 'amazon/chronos-t5-base',  label: 'Chronos-T5-Base  (~200M)' },
];
const PAGE_SIZE = 10;

// ── Forecast result chart ────────────────────────────────────────────────────
function ForecastChart({ result }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current || !result) return;
    const chart = echarts.init(ref.current);

    const hist = result.historical ?? [];
    const pf   = result.point_forecast ?? [];
    const q10  = result.q10 ?? [];
    const q90  = result.q90 ?? [];
    const histAxis = result.time_axis?.historical;
    const nHist = Math.min(hist.length, 200);
    const nFc   = pf.length;

    const xHist = histAxis
      ? histAxis.slice(-nHist)
      : Array.from({ length: nHist }, (_, i) => `t-${nHist - i}`);
    const xFc = Array.from({ length: nFc }, (_, i) => `t+${i + 1}`);
    const xAll = [...xHist, ...xFc];
    const histDisplay = hist.slice(-nHist);

    chart.setOption({
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
      legend: { bottom: 0, data: ['历史数据', '点预测', '90% 置信带'] },
      grid: { top: 16, left: 60, right: 20, bottom: 56 },
      xAxis: {
        type: 'category',
        data: xAll,
        axisLabel: { rotate: 30, interval: Math.floor(xAll.length / 8) },
      },
      yAxis: { type: 'value' },
      series: [
        {
          name: '历史数据',
          type: 'line',
          data: [...histDisplay, ...Array(nFc).fill(null)],
          lineStyle: { color: '#1890ff', width: 1.5 },
          symbol: 'none',
        },
        {
          name: '点预测',
          type: 'line',
          data: [...Array(nHist).fill(null), ...pf],
          lineStyle: { color: '#f5222d', width: 2, type: 'dashed' },
          symbol: 'none',
        },
        // Band top
        {
          name: '90% 置信带',
          type: 'line',
          data: [...Array(nHist).fill(null), ...q90],
          lineStyle: { opacity: 0 },
          areaStyle: { color: 'rgba(245,34,45,0.12)' },
          symbol: 'none',
          stack: 'band',
          legendHoverLink: false,
        },
        // Band bottom (fills downward, overwriting with white)
        {
          type: 'line',
          data: [...Array(nHist).fill(null), ...q10],
          lineStyle: { opacity: 0 },
          areaStyle: { color: '#fff' },
          symbol: 'none',
          stack: 'band',
          legendHoverLink: false,
          showInLegend: false,
        },
      ],
    });

    const onResize = () => chart.resize();
    window.addEventListener('resize', onResize);
    return () => { window.removeEventListener('resize', onResize); chart.dispose(); };
  }, [result]);

  return <div ref={ref} style={{ width: '100%', height: 420 }} />;
}

// ── Result modal ─────────────────────────────────────────────────────────────
function ResultModal({ task, open, onClose }) {
  const result = task?.result;
  if (!task) return null;
  return (
    <Modal
      title={`预测结果 — ${task.value_column} / ${task.dataset_name ?? task.dataset_id}`}
      open={open}
      onCancel={onClose}
      footer={<Button onClick={onClose}>关闭</Button>}
      width={1100}
      destroyOnHidden
    >
      {!result ? (
        <Empty description="暂无结果" />
      ) : (
        <Space direction="vertical" size={16} style={{ width: '100%' }}>
          <Descriptions bordered size="small" column={{ xs: 1, sm: 2, xl: 4 }}>
            <Descriptions.Item label="数据集">{task.dataset_name}</Descriptions.Item>
            <Descriptions.Item label="预测列">{task.value_column}</Descriptions.Item>
            <Descriptions.Item label="预测步数">{task.horizon}</Descriptions.Item>
            <Descriptions.Item label="频率">{FREQ_LABELS[task.frequency] ?? task.frequency}</Descriptions.Item>
            <Descriptions.Item label="模型">{result.model_name ?? task.model_name}</Descriptions.Item>
            <Descriptions.Item label="历史数据点">{result.historical?.length ?? '—'}</Descriptions.Item>
            <Descriptions.Item label="完成时间" span={2}>{formatDateTime(task.finished_at)}</Descriptions.Item>
          </Descriptions>

          <ForecastChart result={result} />

          <Card size="small" title="预测数值（前 24 步）">
            <Row gutter={[8, 8]}>
              {(result.point_forecast ?? []).slice(0, 24).map((v, i) => (
                <Col xs={12} sm={8} md={6} xl={4} key={i}>
                  <Card size="small" style={{ textAlign: 'center' }}>
                    <Text type="secondary" style={{ fontSize: 11 }}>t+{i + 1}</Text>
                    <div style={{ fontWeight: 600, fontSize: 13 }}>{v.toFixed(3)}</div>
                    <Text type="secondary" style={{ fontSize: 10 }}>
                      [{(result.q10?.[i] ?? v).toFixed(2)}, {(result.q90?.[i] ?? v).toFixed(2)}]
                    </Text>
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>
        </Space>
      )}
    </Modal>
  );
}

// ── Main page ────────────────────────────────────────────────────────────────
export default function TimesFM() {
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

  const [resultTask, setResultTask] = useState(null);
  const [resultOpen, setResultOpen] = useState(false);
  const [resultLoading, setResultLoading] = useState(false);

  const refreshTimer = useRef(null);

  useEffect(() => {
    void fetchModelStatus();
    void fetchDatasets();
    void fetchTasks(1);
    return () => clearInterval(refreshTimer.current);
  }, []);

  // Auto-refresh every 3s while any task is running
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
      await timesfmApi.startForecast({
        dataset_id: values.dataset_id,
        value_column: values.value_column,
        time_column: values.time_column ?? null,
        horizon: values.horizon,
        frequency: values.frequency,
        model_name: values.model_name,
      });
      message.success('预测任务已提交，请稍等…');
      void fetchTasks(1);
      setPage(1);
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

  async function openResult(record) {
    setResultTask(record);
    setResultOpen(true);
    setResultLoading(true);
    try {
      const full = await timesfmApi.getForecast(record.id);
      setResultTask(full);
    } catch {
      message.error('加载结果失败');
    } finally {
      setResultLoading(false);
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

  const tableColumns = [
    {
      title: '数据集',
      dataIndex: 'dataset_name',
      render: (v, r) => v ?? r.dataset_id?.slice(0, 8),
    },
    { title: '预测列', dataIndex: 'value_column' },
    {
      title: '步数 / 频率',
      key: 'hf',
      render: (_, r) => `${r.horizon} 步 / ${FREQ_LABELS[r.frequency] ?? r.frequency}`,
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
      width: 155,
    },
    {
      title: '操作',
      key: 'actions',
      width: 110,
      render: (_, r) => (
        <Space size={4} onClick={(e) => e.stopPropagation()}>
          <Button
            size="small"
            icon={<EyeOutlined />}
            disabled={r.status !== 'SUCCESS'}
            onClick={() => void openResult(r)}
          >
            查看
          </Button>
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

  const available = modelStatus?.available;

  return (
    <Space direction="vertical" size={20} style={{ width: '100%' }}>
      {/* Header */}
      <Row align="middle" justify="space-between" wrap={false}>
        <Col>
          <Title level={2} style={{ margin: 0 }}>时序预测</Title>
          <Text type="secondary">Amazon Chronos 零样本时序基础模型</Text>
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
                    ? `模型就绪 — ${modelStatus.model}`
                    : `已安装 / 未加载 — ${modelStatus.model}`
                  : 'Chronos 未安装'
              }
            />
          ) : <Button size="small" onClick={() => void fetchModelStatus()}>检查状态</Button>}
        </Col>
      </Row>

      {/* Install warning */}
      {modelStatus && !available && (
        <Alert
          type="warning"
          showIcon
          message="Chronos 预测引擎未安装"
          description={
            <Space direction="vertical" size={4}>
              <Text>请在后端虚拟环境中执行：</Text>
              <Text code>pip install chronos-forecasting torch</Text>
              <Text type="secondary">安装后重启后端即可，模型权重首次预测时自动从 HuggingFace 下载（约 200 MB）。</Text>
            </Space>
          }
        />
      )}

      <Row gutter={[20, 20]}>
        {/* ── Configure panel ── */}
        <Col xs={24} lg={9}>
          <Card title={<Space><BarChartOutlined />配置预测任务</Space>} style={{ height: '100%' }}>
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
                  placeholder="用于生成时间坐标轴"
                  allowClear
                  options={columns.map((c) => ({ value: c, label: c }))}
                />
              </Form.Item>

              <Form.Item
                label={<><span>③ 预测目标列 </span><Tag color="red" style={{ marginLeft: 4 }}>必填</Tag></>}
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

              <Space direction="vertical" style={{ width: '100%' }}>
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
            title={<Space><ReloadOutlined />预测历史</Space>}
            extra={<Button size="small" onClick={() => void fetchTasks(page)}>刷新</Button>}
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
              onRow={(r) => ({
                style: r.status === 'SUCCESS' ? { cursor: 'pointer' } : {},
                onClick: () => r.status === 'SUCCESS' && void openResult(r),
              })}
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

      {/* Result detail modal */}
      <Modal
        title={`预测结果 — ${resultTask?.value_column ?? ''}`}
        open={resultOpen}
        onCancel={() => setResultOpen(false)}
        footer={<Button onClick={() => setResultOpen(false)}>关闭</Button>}
        width={1100}
        destroyOnHidden
      >
        {resultLoading ? (
          <div style={{ textAlign: 'center', padding: 60 }}><Spin size="large" /></div>
        ) : (
          <ResultModal task={resultTask} open={resultOpen} onClose={() => setResultOpen(false)} />
        )}
      </Modal>
    </Space>
  );
}
