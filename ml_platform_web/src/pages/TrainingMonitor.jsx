import React, { useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Button, Card, Descriptions, Input, Pagination,
  Popconfirm, Progress, Space, Table, Tag, Typography, message,
} from 'antd';
import {
  ArrowLeftOutlined, CheckOutlined, CloseOutlined,
  DeleteOutlined, EditOutlined, EyeOutlined,
  PlusOutlined, ReloadOutlined, StopOutlined,
} from '@ant-design/icons';
import { dataApi, logsApi, trainingApi } from '../services/api';
import { useLogStream } from '../hooks/useLogStream';

const { Text, Title } = Typography;

function useQuery() {
  return new URLSearchParams(useLocation().search);
}

const statusColors = {
  PENDING: 'default',
  RUNNING: 'processing',
  SUCCESS: 'success',
  FAILED: 'error',
};

const PAGE_SIZE = 10;

// ── Inline-editable name cell ─────────────────────────────────────────────────
function EditableNameCell({ record, onSave }) {
  const [editing, setEditing] = useState(false);
  const [val, setVal] = useState(record.name ?? record.id.slice(0, 8));
  const inputRef = useRef(null);

  function startEdit(e) {
    e.stopPropagation();
    setVal(record.name ?? record.id.slice(0, 8));
    setEditing(true);
    setTimeout(() => inputRef.current?.focus(), 50);
  }

  async function save(e) {
    e?.stopPropagation();
    const trimmed = val.trim();
    if (!trimmed) { setEditing(false); return; }
    try {
      await onSave(record.id, trimmed);
      setEditing(false);
    } catch {
      message.error('重命名失败');
    }
  }

  function cancel(e) {
    e?.stopPropagation();
    setEditing(false);
  }

  if (editing) {
    return (
      <Space size={4} onClick={e => e.stopPropagation()}>
        <Input
          ref={inputRef}
          size="small"
          value={val}
          maxLength={100}
          style={{ width: 160 }}
          onChange={e => setVal(e.target.value)}
          onPressEnter={save}
          onKeyDown={e => e.key === 'Escape' && cancel(e)}
        />
        <Button size="small" type="text" icon={<CheckOutlined />} onClick={save} />
        <Button size="small" type="text" icon={<CloseOutlined />} onClick={cancel} />
      </Space>
    );
  }

  return (
    <Space size={4}>
      <Text>{record.name ?? <Text type="secondary">{record.id.slice(0, 8)}</Text>}</Text>
      <Button
        size="small"
        type="text"
        icon={<EditOutlined />}
        onClick={startEdit}
        style={{ opacity: 0.5 }}
      />
    </Space>
  );
}

// ── List view ─────────────────────────────────────────────────────────────────
function TaskListView({ navigate }) {
  const [tasks, setTasks] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [datasetsById, setDatasetsById] = useState({});
  const [loading, setLoading] = useState(false);

  useEffect(() => { void loadDatasets(); }, []);
  useEffect(() => { void loadPage(page); }, [page]);

  // Auto-refresh running tasks every 5s
  useEffect(() => {
    const timer = setInterval(() => {
      const hasRunning = tasks.some(t => t.status === 'RUNNING' || t.status === 'PENDING');
      if (hasRunning) void loadPage(page);
    }, 5000);
    return () => clearInterval(timer);
  }, [tasks, page]);

  async function loadDatasets() {
    try {
      const res = await dataApi.listDatasets();
      setDatasetsById(Object.fromEntries((res.items ?? []).map(d => [d.id, d.name])));
    } catch { /* ignore */ }
  }

  async function loadPage(p) {
    setLoading(true);
    try {
      const res = await trainingApi.listTasks({ page: p, page_size: PAGE_SIZE });
      setTasks(res.items ?? []);
      setTotal(res.total ?? 0);
    } catch {
      message.error('加载任务列表失败');
    } finally {
      setLoading(false);
    }
  }

  async function handleRename(taskId, name) {
    await trainingApi.renameTask(taskId, name);
    setTasks(prev => prev.map(t => t.id === taskId ? { ...t, name } : t));
  }

  async function handleDelete(taskId) {
    try {
      await trainingApi.deleteTask(taskId);
      message.success('任务已删除');
      void loadPage(page);
    } catch (err) {
      message.error(err?.response?.data?.detail ?? '删除失败');
    }
  }

  async function handleStop(taskId) {
    try {
      await trainingApi.stopTraining(taskId);
      message.success('已发送停止指令');
      void loadPage(page);
    } catch {
      message.error('停止失败');
    }
  }

  const columns = [
    {
      title: '任务名称',
      key: 'name',
      render: (_, record) => (
        <EditableNameCell record={record} onSave={handleRename} />
      ),
    },
    {
      title: '模型',
      dataIndex: 'model_type',
      key: 'model_type',
      width: 130,
    },
    {
      title: '数据集',
      dataIndex: 'dataset_id',
      key: 'dataset_id',
      width: 160,
      render: v => datasetsById[v] ?? <Text code>{v.slice(0, 8)}</Text>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 90,
      render: v => {
        const s = (v ?? 'UNKNOWN').toUpperCase();
        return <Tag color={statusColors[s] ?? 'default'}>{s}</Tag>;
      },
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      width: 120,
      render: (v, record) => {
        const s = (record.status ?? '').toUpperCase();
        return (
          <Progress
            percent={Math.round(v ?? 0)}
            size="small"
            status={s === 'FAILED' ? 'exception' : s === 'SUCCESS' ? 'success' : 'active'}
            showInfo={false}
          />
        );
      },
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 160,
      render: v => new Date(v).toLocaleString('zh-CN'),
    },
    {
      title: '操作',
      key: 'actions',
      width: 140,
      render: (_, record) => {
        const s = (record.status ?? '').toUpperCase();
        return (
          <Space size={4} onClick={e => e.stopPropagation()}>
            <Button
              size="small"
              icon={<EyeOutlined />}
              onClick={() => navigate(`/training/monitor?taskId=${record.id}`)}
            >
              查看
            </Button>
            {s === 'RUNNING' && (
              <Button
                size="small"
                danger
                icon={<StopOutlined />}
                onClick={() => void handleStop(record.id)}
              >
                停止
              </Button>
            )}
            {s !== 'RUNNING' && (
              <Popconfirm
                title="确认删除该任务吗？"
                okText="删除"
                cancelText="取消"
                okButtonProps={{ danger: true }}
                onConfirm={() => void handleDelete(record.id)}
              >
                <Button size="small" danger icon={<DeleteOutlined />} />
              </Popconfirm>
            )}
          </Space>
        );
      },
    },
  ];

  return (
    <div>
      <Space style={{ width: '100%', justifyContent: 'space-between', marginBottom: 24 }}>
        <Title level={2} style={{ margin: 0 }}>机器学习训练监控</Title>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={() => void loadPage(page)}>刷新</Button>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => navigate('/training/config')}
          >
            新建训练
          </Button>
        </Space>
      </Space>

      <Card>
        <Table
          rowKey="id"
          dataSource={tasks}
          columns={columns}
          loading={loading}
          pagination={false}
          size="middle"
          onRow={record => ({
            style: { cursor: 'pointer' },
            onClick: () => navigate(`/training/monitor?taskId=${record.id}`),
          })}
        />
        <div style={{ marginTop: 16, textAlign: 'right' }}>
          <Pagination
            current={page}
            pageSize={PAGE_SIZE}
            total={total}
            showTotal={t => `共 ${t} 条`}
            onChange={p => setPage(p)}
            showSizeChanger={false}
          />
        </div>
      </Card>
    </div>
  );
}

// ── Detail view ───────────────────────────────────────────────────────────────
function TaskDetailView({ taskId, navigate }) {
  const [task, setTask] = useState(null);
  const [loading, setLoading] = useState(true);
  // WS log stream (auto-reconnect w/ exponential backoff, seeded from REST)
  const { logs, connected, seedHistorical } = useLogStream({
    domainTaskId: taskId,
    enabled: !!taskId,
    maxEntries: 500,
  });
  const loadTaskRef = useRef(null);
  const seedLogsRef = useRef(null);

  useEffect(() => {
    void loadTaskRef.current?.();
    void seedLogsRef.current?.();
    // Keep status/metrics polling at 3s for progress bar (logs now streamed).
    const timer = setInterval(() => { void loadTaskRef.current?.(); }, 3000);
    return () => clearInterval(timer);
  }, [taskId]);

  async function loadTask() {
    try {
      const data = await trainingApi.getTrainingStatus(taskId);
      setTask(data);
    } catch {
      message.error('加载任务状态失败');
    } finally {
      setLoading(false);
    }
  }

  async function seedLogsFromRest() {
    // Fetch the last page of historical logs once at mount so the user sees
    // context from before the WS connected; after that, the WS appends live.
    try {
      const res = await logsApi.getLogs(taskId, { page_size: 30 });
      seedHistorical(res.entries ?? []);
    } catch {
      /* no historical logs yet */
    }
  }

  loadTaskRef.current = loadTask;
  seedLogsRef.current = seedLogsFromRest;

  const logText = logs.length
    ? logs.map(e => `${e.timestamp ?? ''} | ${e.level ?? 'INFO'} | ${e.message}`).join('\n')
    : '暂无日志。';

  async function handleStop() {
    try {
      await trainingApi.stopTraining(taskId);
      message.success('已发送停止指令');
      void loadTask();
    } catch {
      message.error('停止失败');
    }
  }

  const s = (task?.status ?? '').toUpperCase();

  return (
    <div>
      <Space style={{ width: '100%', justifyContent: 'space-between', marginBottom: 24 }}>
        <Space>
          <Button
            icon={<ArrowLeftOutlined />}
            onClick={() => navigate('/training/monitor')}
          >
            返回列表
          </Button>
          <Title level={2} style={{ margin: 0 }}>训练任务详情</Title>
        </Space>
        <Space>
          {s === 'RUNNING' && (
            <Button danger icon={<StopOutlined />} onClick={handleStop}>停止训练</Button>
          )}
          <Button icon={<ReloadOutlined />} onClick={() => { void loadTask(); void seedLogsFromRest(); }}>
            刷新
          </Button>
        </Space>
      </Space>

      {loading && <Card><Text type="secondary">正在加载…</Text></Card>}

      {!loading && task && (
        <Space direction="vertical" size={16} style={{ width: '100%' }}>
          <Card>
            <Descriptions bordered column={2} size="small">
              <Descriptions.Item label="任务名称">{task.name ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="任务 ID"><Text code>{task.id}</Text></Descriptions.Item>
              <Descriptions.Item label="模型">{task.model_type}</Descriptions.Item>
              <Descriptions.Item label="状态">
                <Tag color={statusColors[s] ?? 'default'}>{s}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="进度" span={2}>
                <Progress
                  percent={Math.round(task.progress ?? 0)}
                  status={s === 'FAILED' ? 'exception' : s === 'SUCCESS' ? 'success' : 'active'}
                  strokeColor={s === 'RUNNING' ? { from: '#108ee9', to: '#87d068' } : undefined}
                />
              </Descriptions.Item>
              <Descriptions.Item label="开始时间">
                {task.started_at ? new Date(task.started_at).toLocaleString('zh-CN') : '-'}
              </Descriptions.Item>
              <Descriptions.Item label="结束时间">
                {task.finished_at ? new Date(task.finished_at).toLocaleString('zh-CN') : '-'}
              </Descriptions.Item>
            </Descriptions>
          </Card>

          {task.result_metrics && (
            <Card title="训练指标">
              <Descriptions bordered column={3} size="small">
                {Object.entries(task.result_metrics).map(([k, v]) => (
                  <Descriptions.Item key={k} label={k}>{String(v)}</Descriptions.Item>
                ))}
              </Descriptions>
            </Card>
          )}

          {task.error_message && (
            <Card title="错误信息">
              <Text type="danger">{task.error_message}</Text>
            </Card>
          )}

          <Card
            title={
              <Space>
                <span>实时日志</span>
                <Tag color={connected ? 'green' : 'default'}>
                  {connected ? '已连接' : '未连接'}
                </Tag>
              </Space>
            }
          >
            <pre style={{ margin: 0, whiteSpace: 'pre-wrap', maxHeight: 300, overflow: 'auto' }}>
              {logText}
            </pre>
          </Card>
        </Space>
      )}
    </div>
  );
}

// ── Root component ────────────────────────────────────────────────────────────
const TrainingMonitor = () => {
  const query = useQuery();
  const navigate = useNavigate();
  const taskId = query.get('taskId');

  if (taskId) {
    return <TaskDetailView taskId={taskId} navigate={navigate} />;
  }
  return <TaskListView navigate={navigate} />;
};

export default TrainingMonitor;
