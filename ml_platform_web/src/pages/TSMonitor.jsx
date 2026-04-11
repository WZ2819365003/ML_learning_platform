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
  Row,
  Space,
  Spin,
  Statistic,
  Table,
  Tag,
  Typography,
  message,
} from 'antd';
import {
  ArrowLeftOutlined,
  BarChartOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  EyeOutlined,
  LoadingOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import { timesfmApi } from '../services/api';
import { formatDateTime } from '../utils/formatters';

const { Title, Text } = Typography;

const FREQ_LABELS = { high: '高频（日数据）', medium: '中频（周数据）', low: '低频（月数据）' };
const STATUS_COLOR = { PENDING: 'default', RUNNING: 'processing', SUCCESS: 'success', FAILED: 'error' };
const STATUS_LABEL = { PENDING: '等待中', RUNNING: '运行中', SUCCESS: '已完成', FAILED: '失败' };

function useQuery() {
  return new URLSearchParams(useLocation().search);
}

// ── Duration tracker ─────────────────────────────────────────────────────────
function useDuration(startedAt, finishedAt) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!startedAt || finishedAt) {
      if (startedAt && finishedAt) {
        const diff = (new Date(finishedAt) - new Date(startedAt)) / 1000;
        setElapsed(Math.round(diff));
      }
      return;
    }
    const start = new Date(startedAt).getTime();
    const timer = setInterval(() => {
      setElapsed(Math.round((Date.now() - start) / 1000));
    }, 1000);
    return () => clearInterval(timer);
  }, [startedAt, finishedAt]);
  return elapsed;
}

function fmtSec(s) {
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

// ── Single task detail card ───────────────────────────────────────────────────
function TaskDetailCard({ task, onViewResult }) {
  const elapsed = useDuration(task?.started_at, task?.finished_at);
  if (!task) return null;

  const isActive = task.status === 'PENDING' || task.status === 'RUNNING';

  return (
    <Card
      title={
        <Space>
          <Badge status={STATUS_COLOR[task.status] ?? 'default'} />
          <span>
            {task.dataset_name ?? task.dataset_id?.slice(0, 8)} / <Text code>{task.value_column}</Text> / {task.horizon}步预测
          </span>
        </Space>
      }
      extra={
        task.status === 'SUCCESS' && (
          <Button type="primary" icon={<EyeOutlined />} onClick={onViewResult}>
            查看结果
          </Button>
        )
      }
    >
      {isActive && (
        <div style={{ textAlign: 'center', padding: '24px 0 16px' }}>
          <Spin
            indicator={<LoadingOutlined style={{ fontSize: 48, color: '#1890ff' }} spin />}
          />
          <div style={{ marginTop: 16 }}>
            <Text type="secondary">
              {task.status === 'PENDING' ? '任务排队中，即将开始…' : '正在运行 Chronos 预测引擎…'}
            </Text>
          </div>
        </div>
      )}

      {task.status === 'FAILED' && (
        <Alert
          type="error"
          showIcon
          message="预测失败"
          description={task.error_message ?? '未知错误'}
          style={{ marginBottom: 16 }}
        />
      )}

      {task.status === 'SUCCESS' && (
        <Alert
          type="success"
          showIcon
          icon={<CheckCircleOutlined />}
          message="预测完成！"
          description="点击右上角「查看结果」可查看预测图表和数值。"
          style={{ marginBottom: 16 }}
        />
      )}

      <Row gutter={[16, 16]} style={{ marginTop: 8 }}>
        <Col span={6}>
          <Statistic
            title="状态"
            value={STATUS_LABEL[task.status] ?? task.status}
            prefix={task.status === 'RUNNING' ? <LoadingOutlined /> : null}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="耗时"
            value={fmtSec(elapsed)}
            prefix={<ClockCircleOutlined />}
          />
        </Col>
        <Col span={6}>
          <Statistic title="预测步数" value={task.horizon} suffix="步" />
        </Col>
        <Col span={6}>
          <Statistic
            title="模型"
            value={task.model_name?.split('/').pop() ?? '—'}
            valueStyle={{ fontSize: 14 }}
          />
        </Col>
      </Row>

      <Descriptions
        size="small"
        bordered
        column={{ xs: 1, sm: 2, lg: 3 }}
        style={{ marginTop: 20 }}
      >
        <Descriptions.Item label="数据集">{task.dataset_name ?? task.dataset_id}</Descriptions.Item>
        <Descriptions.Item label="预测目标列">{task.value_column}</Descriptions.Item>
        <Descriptions.Item label="时间列">{task.time_column ?? '未指定'}</Descriptions.Item>
        <Descriptions.Item label="频率">{FREQ_LABELS[task.frequency] ?? task.frequency}</Descriptions.Item>
        <Descriptions.Item label="提交时间">{formatDateTime(task.created_at)}</Descriptions.Item>
        <Descriptions.Item label="开始时间">{task.started_at ? formatDateTime(task.started_at) : '—'}</Descriptions.Item>
        <Descriptions.Item label="完成时间" span={2}>{task.finished_at ? formatDateTime(task.finished_at) : '—'}</Descriptions.Item>
      </Descriptions>
    </Card>
  );
}

// ── Active tasks table ────────────────────────────────────────────────────────
function ActiveTasksTable({ tasks, loading, onSelect, onRefresh }) {
  const navigate = useNavigate();
  const columns = [
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
      render: (s) => <Badge status={STATUS_COLOR[s] ?? 'default'} text={STATUS_LABEL[s] ?? s} />,
    },
    {
      title: '提交时间',
      dataIndex: 'created_at',
      render: (v) => formatDateTime(v),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, r) => (
        <Space size={4}>
          <Button size="small" onClick={() => onSelect(r.id)}>查看详情</Button>
          {r.status === 'SUCCESS' && (
            <Button
              size="small"
              type="primary"
              icon={<EyeOutlined />}
              onClick={() => navigate(`/ts/results?id=${r.id}`)}
            >
              结果
            </Button>
          )}
        </Space>
      ),
    },
  ];

  return (
    <Card
      title={<Space><ReloadOutlined />所有监控任务</Space>}
      extra={<Button size="small" icon={<ReloadOutlined />} onClick={onRefresh}>刷新</Button>}
    >
      <Table
        rowKey="id"
        dataSource={tasks}
        columns={columns}
        loading={loading}
        size="small"
        pagination={false}
        locale={{ emptyText: <Empty description="暂无 PENDING/RUNNING 任务" /> }}
      />
    </Card>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function TSMonitor() {
  const navigate = useNavigate();
  const query = useQuery();
  const targetId = query.get('id');

  const [focusTask, setFocusTask] = useState(null);
  const [focusLoading, setFocusLoading] = useState(false);

  const [activeTasks, setActiveTasks] = useState([]);
  const [tableLoading, setTableLoading] = useState(false);

  const refreshTimer = useRef(null);

  useEffect(() => {
    void fetchActiveTasks();
    if (targetId) void fetchFocusTask(targetId);
    return () => clearInterval(refreshTimer.current);
  }, [targetId]);

  // Auto-refresh while there are active tasks
  useEffect(() => {
    clearInterval(refreshTimer.current);
    const hasActive = activeTasks.some(
      (t) => t.status === 'RUNNING' || t.status === 'PENDING'
    ) || (focusTask && (focusTask.status === 'RUNNING' || focusTask.status === 'PENDING'));

    if (hasActive) {
      refreshTimer.current = setInterval(() => {
        void fetchActiveTasks();
        if (focusTask?.id) void fetchFocusTask(focusTask.id);
      }, 2000);
    }
    return () => clearInterval(refreshTimer.current);
  }, [activeTasks, focusTask]);

  async function fetchFocusTask(id) {
    setFocusLoading(true);
    try {
      const res = await timesfmApi.getForecast(id);
      setFocusTask(res);
    } catch {
      message.error('加载任务详情失败');
    } finally {
      setFocusLoading(false);
    }
  }

  async function fetchActiveTasks() {
    setTableLoading(true);
    try {
      // Load all recent tasks (not just active) for the table
      const res = await timesfmApi.listForecasts({ page: 1, page_size: 50 });
      setActiveTasks(res.items ?? []);
    } catch {
      // silently ignore
    } finally {
      setTableLoading(false);
    }
  }

  function handleSelectTask(id) {
    navigate(`/ts/monitor?id=${id}`);
    void fetchFocusTask(id);
  }

  return (
    <Space direction="vertical" size={20} style={{ width: '100%' }}>
      {/* Header */}
      <Row align="middle" justify="space-between">
        <Col>
          <Space>
            <Button
              icon={<ArrowLeftOutlined />}
              onClick={() => navigate('/ts/config')}
              type="text"
            />
            <div>
              <Title level={2} style={{ margin: 0 }}>任务监控</Title>
              <Text type="secondary">实时监控时序预测任务进度</Text>
            </div>
          </Space>
        </Col>
        <Col>
          <Space>
            <Button icon={<BarChartOutlined />} onClick={() => navigate('/ts/config')}>
              新建预测
            </Button>
            <Button
              icon={<EyeOutlined />}
              type="primary"
              onClick={() => navigate('/ts/results')}
            >
              查看结果
            </Button>
          </Space>
        </Col>
      </Row>

      {/* Focus task detail */}
      {focusLoading && !focusTask ? (
        <Card><div style={{ textAlign: 'center', padding: 40 }}><Spin /></div></Card>
      ) : focusTask ? (
        <TaskDetailCard
          task={focusTask}
          onViewResult={() => navigate(`/ts/results?id=${focusTask.id}`)}
        />
      ) : (
        <Card>
          <Empty
            description={
              <Space direction="vertical">
                <Text>点击下方任务行查看详情，或</Text>
                <Button type="primary" onClick={() => navigate('/ts/config')}>
                  新建预测任务
                </Button>
              </Space>
            }
          />
        </Card>
      )}

      {/* All tasks table */}
      <ActiveTasksTable
        tasks={activeTasks}
        loading={tableLoading}
        onSelect={handleSelectTask}
        onRefresh={() => { void fetchActiveTasks(); if (focusTask?.id) void fetchFocusTask(focusTask.id); }}
      />
    </Space>
  );
}
