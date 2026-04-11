import React, { useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Alert,
  Badge,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Popconfirm,
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
  ArrowLeftOutlined,
  BarChartOutlined,
  DeleteOutlined,
  EyeOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import * as echarts from 'echarts';
import { timesfmApi } from '../services/api';
import { formatDateTime } from '../utils/formatters';

const { Title, Text } = Typography;

const FREQ_LABELS = { high: '高频（日数据）', medium: '中频（周数据）', low: '低频（月数据）' };
const STATUS_COLOR = { PENDING: 'default', RUNNING: 'processing', SUCCESS: 'success', FAILED: 'error' };
const STATUS_LABEL = { PENDING: '等待中', RUNNING: '运行中', SUCCESS: '已完成', FAILED: '失败' };
const PAGE_SIZE = 15;

function useQuery() {
  return new URLSearchParams(useLocation().search);
}

// ── Forecast chart ────────────────────────────────────────────────────────────
function ForecastChart({ result }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current || !result) return;
    const chart = echarts.init(ref.current);

    const hist    = result.historical ?? [];
    const pf      = result.point_forecast ?? [];
    const q10     = result.q10 ?? [];
    const q90     = result.q90 ?? [];
    const histAxis = result.time_axis?.historical;
    const nHist   = Math.min(hist.length, 200);
    const nFc     = pf.length;

    const xHist = histAxis
      ? histAxis.slice(-nHist)
      : Array.from({ length: nHist }, (_, i) => `t-${nHist - i}`);
    const xFc   = Array.from({ length: nFc }, (_, i) => `t+${i + 1}`);
    const xAll  = [...xHist, ...xFc];
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

// ── Forecast value grid ───────────────────────────────────────────────────────
function ForecastGrid({ result, maxRows = 24 }) {
  if (!result?.point_forecast?.length) return null;
  return (
    <Card size="small" title={`预测数值（前 ${maxRows} 步）`}>
      <Row gutter={[8, 8]}>
        {result.point_forecast.slice(0, maxRows).map((v, i) => (
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
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function TSResults() {
  const navigate = useNavigate();
  const query = useQuery();
  const targetId = query.get('id');

  const [tasks, setTasks] = useState([]);
  const [tasksLoading, setTasksLoading] = useState(false);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState(null);

  const [selectedTask, setSelectedTask] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    void fetchTasks(1, statusFilter);
  }, [statusFilter]);

  useEffect(() => {
    if (targetId) void openTask(targetId);
  }, [targetId]);

  async function fetchTasks(p, sf) {
    setTasksLoading(true);
    try {
      const params = { page: p, page_size: PAGE_SIZE };
      if (sf) params.status = sf;
      const res = await timesfmApi.listForecasts(params);
      setTasks(res.items ?? []);
      setTotal(res.total ?? 0);
      setPage(p);
    } catch {
      message.error('加载预测记录失败');
    } finally {
      setTasksLoading(false);
    }
  }

  async function openTask(id) {
    setDetailLoading(true);
    try {
      const full = await timesfmApi.getForecast(id);
      setSelectedTask(full);
    } catch {
      message.error('加载结果失败');
    } finally {
      setDetailLoading(false);
    }
  }

  async function handleDelete(id) {
    try {
      await timesfmApi.deleteForecast(id);
      message.success('已删除');
      if (selectedTask?.id === id) setSelectedTask(null);
      void fetchTasks(page, statusFilter);
    } catch (err) {
      message.error(err?.response?.data?.detail ?? '删除失败');
    }
  }

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
      title: '频率',
      dataIndex: 'frequency',
      render: (v) => FREQ_LABELS[v] ?? v,
    },
    {
      title: '状态',
      dataIndex: 'status',
      render: (s) => <Badge status={STATUS_COLOR[s] ?? 'default'} text={STATUS_LABEL[s] ?? s} />,
    },
    {
      title: '完成时间',
      dataIndex: 'finished_at',
      render: (v) => v ? formatDateTime(v) : '—',
      width: 140,
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
            type={selectedTask?.id === r.id ? 'primary' : 'default'}
            disabled={r.status !== 'SUCCESS'}
            onClick={() => void openTask(r.id)}
          >
            结果
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

  return (
    <Space direction="vertical" size={20} style={{ width: '100%' }}>
      {/* Header */}
      <Row align="middle" justify="space-between">
        <Col>
          <Space>
            <Button icon={<ArrowLeftOutlined />} type="text" onClick={() => navigate('/ts/config')} />
            <div>
              <Title level={2} style={{ margin: 0 }}>结果可视化</Title>
              <Text type="secondary">查看时序预测图表与预测数值</Text>
            </div>
          </Space>
        </Col>
        <Col>
          <Button icon={<BarChartOutlined />} type="primary" onClick={() => navigate('/ts/config')}>
            新建预测
          </Button>
        </Col>
      </Row>

      <Row gutter={[20, 20]}>
        {/* ── Task list ── */}
        <Col xs={24} lg={8}>
          <Card
            title="预测记录"
            extra={
              <Space size={8}>
                <Select
                  size="small"
                  style={{ width: 90 }}
                  placeholder="状态"
                  allowClear
                  value={statusFilter}
                  onChange={(v) => setStatusFilter(v ?? null)}
                  options={[
                    { value: 'SUCCESS', label: '完成' },
                    { value: 'RUNNING', label: '运行中' },
                    { value: 'PENDING', label: '等待中' },
                    { value: 'FAILED', label: '失败' },
                  ]}
                />
                <Button size="small" icon={<ReloadOutlined />} onClick={() => void fetchTasks(page, statusFilter)} />
              </Space>
            }
          >
            <Table
              rowKey="id"
              dataSource={tasks}
              columns={tableColumns}
              loading={tasksLoading}
              size="small"
              rowClassName={(r) => r.id === selectedTask?.id ? 'ant-table-row-selected' : ''}
              pagination={{
                current: page,
                pageSize: PAGE_SIZE,
                total,
                onChange: (p) => void fetchTasks(p, statusFilter),
                showTotal: (t) => `共 ${t} 条`,
                showSizeChanger: false,
                size: 'small',
              }}
              locale={{ emptyText: <Empty description="暂无记录" /> }}
              onRow={(r) => ({
                style: r.status === 'SUCCESS' ? { cursor: 'pointer' } : {},
                onClick: () => r.status === 'SUCCESS' && void openTask(r.id),
              })}
              expandable={{
                expandedRowRender: (r) =>
                  r.status === 'FAILED' ? (
                    <Alert type="error" showIcon message={r.error_message ?? '未知错误'} />
                  ) : null,
                rowExpandable: (r) => r.status === 'FAILED',
              }}
            />
          </Card>
        </Col>

        {/* ── Result detail ── */}
        <Col xs={24} lg={16}>
          {detailLoading ? (
            <Card><div style={{ textAlign: 'center', padding: 60 }}><Spin size="large" /></div></Card>
          ) : selectedTask?.result ? (
            <Space direction="vertical" size={16} style={{ width: '100%' }}>
              {/* Meta info */}
              <Card size="small">
                <Descriptions bordered size="small" column={{ xs: 1, sm: 2, xl: 3 }}>
                  <Descriptions.Item label="数据集">{selectedTask.dataset_name}</Descriptions.Item>
                  <Descriptions.Item label="预测列">{selectedTask.value_column}</Descriptions.Item>
                  <Descriptions.Item label="预测步数">{selectedTask.horizon}</Descriptions.Item>
                  <Descriptions.Item label="频率">{FREQ_LABELS[selectedTask.frequency] ?? selectedTask.frequency}</Descriptions.Item>
                  <Descriptions.Item label="模型">{selectedTask.result?.model_name ?? selectedTask.model_name}</Descriptions.Item>
                  <Descriptions.Item label="历史数据点">{selectedTask.result?.historical?.length ?? '—'}</Descriptions.Item>
                  <Descriptions.Item label="完成时间" span={2}>{formatDateTime(selectedTask.finished_at)}</Descriptions.Item>
                </Descriptions>
              </Card>

              {/* Chart */}
              <Card title="预测图表" size="small">
                <ForecastChart result={selectedTask.result} />
              </Card>

              {/* Forecast grid */}
              <ForecastGrid result={selectedTask.result} />
            </Space>
          ) : (
            <Card style={{ height: '100%', minHeight: 400 }}>
              <Empty
                style={{ paddingTop: 80 }}
                description={
                  <Space direction="vertical">
                    <Text type="secondary">点击左侧完成的预测记录查看图表</Text>
                    <Button type="primary" onClick={() => navigate('/ts/config')}>
                      新建预测任务
                    </Button>
                  </Space>
                }
              />
            </Card>
          )}
        </Col>
      </Row>
    </Space>
  );
}
