import React, { useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Badge, Button, Card, Pagination, Popconfirm, Progress,
  Space, Table, Tag, Typography, message,
} from 'antd';
import {
  ArrowLeftOutlined, CheckOutlined, CloseOutlined,
  DeleteOutlined, EditOutlined, EyeOutlined,
  PlusOutlined, ReloadOutlined, StopOutlined, TrophyOutlined,
} from '@ant-design/icons';
import EChart from '../components/EChart';
import { dlApi } from '../services/api';

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

const statusLabels = {
  PENDING: '等待中',
  RUNNING: '训练中',
  SUCCESS: '成功',
  FAILED: '失败',
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
        <input
          ref={inputRef}
          size="small"
          value={val}
          maxLength={100}
          style={{ width: 160, border: '1px solid #d9d9d9', borderRadius: 4, padding: '2px 8px' }}
          onChange={e => setVal(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter') save(e);
            if (e.key === 'Escape') cancel(e);
          }}
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

// ── ECharts helpers ───────────────────────────────────────────────────────────
function buildLossOption(lossHistory) {
  const epochs = lossHistory.map(d => d.epoch);
  const trainLosses = lossHistory.map(d => d.train_loss ?? null);
  const valLosses = lossHistory.map(d => d.val_loss ?? null);
  return {
    tooltip: { trigger: 'axis' },
    legend: { data: ['train_loss', 'val_loss'] },
    xAxis: { type: 'category', data: epochs, name: 'Epoch' },
    yAxis: { type: 'value', name: 'Loss' },
    series: [
      { name: 'train_loss', type: 'line', smooth: true, data: trainLosses },
      { name: 'val_loss', type: 'line', smooth: true, data: valLosses },
    ],
  };
}

function buildMetricOption(lossHistory, taskType) {
  const epochs = lossHistory.map(d => d.epoch);
  const valLosses = lossHistory.map(d => d.val_loss ?? null);
  let secKey = null, secName = '';
  if (taskType === 'classification') { secKey = 'val_acc'; secName = 'val_acc'; }
  else if (taskType === 'regression') { secKey = 'val_rmse'; secName = 'val_rmse'; }
  else {
    const first = lossHistory.find(d => d.val_acc != null || d.val_rmse != null);
    if (first?.val_acc != null) { secKey = 'val_acc'; secName = 'val_acc'; }
    else if (first?.val_rmse != null) { secKey = 'val_rmse'; secName = 'val_rmse'; }
  }
  const secData = secKey ? lossHistory.map(d => d[secKey] ?? null) : [];
  const series = [
    { name: 'val_loss', type: 'line', smooth: true, yAxisIndex: 0, data: valLosses },
  ];
  if (secKey) series.push({ name: secName, type: 'line', smooth: true, yAxisIndex: 1, data: secData });
  return {
    tooltip: { trigger: 'axis' },
    legend: { data: secKey ? ['val_loss', secName] : ['val_loss'] },
    xAxis: { type: 'category', data: epochs, name: 'Epoch' },
    yAxis: [
      { type: 'value', name: 'Loss' },
      { type: 'value', name: secKey ? secName : '', position: 'right' },
    ],
    series,
  };
}

const epochColumns = [
  { title: 'Epoch', dataIndex: 'epoch', key: 'epoch', width: 80 },
  { title: 'Train Loss', dataIndex: 'train_loss', key: 'train_loss', render: v => v != null ? v.toFixed(6) : '-' },
  { title: 'Val Loss', dataIndex: 'val_loss', key: 'val_loss', render: v => v != null ? v.toFixed(6) : '-' },
  { title: 'Val Acc / RMSE', key: 'val_metric', render: (_, row) => row.val_acc != null ? row.val_acc.toFixed(4) : row.val_rmse != null ? row.val_rmse.toFixed(4) : '-' },
  { title: 'LR', dataIndex: 'lr', key: 'lr', render: v => v != null ? v.toExponential(3) : '-' },
];

// ── List view ─────────────────────────────────────────────────────────────────
function DLTaskListView({ navigate }) {
  const [tasks, setTasks] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);

  useEffect(() => { void loadPage(page); }, [page]);

  useEffect(() => {
    const timer = setInterval(() => {
      const hasRunning = tasks.some(t => t.status === 'RUNNING' || t.status === 'PENDING');
      if (hasRunning) void loadPage(page);
    }, 5000);
    return () => clearInterval(timer);
  }, [tasks, page]);

  async function loadPage(p) {
    setLoading(true);
    try {
      const res = await dlApi.listTasks({ page: p, page_size: PAGE_SIZE });
      setTasks(res.items ?? []);
      setTotal(res.total ?? 0);
    } catch {
      message.error('加载任务列表失败');
    } finally {
      setLoading(false);
    }
  }

  async function handleRename(taskId, name) {
    await dlApi.renameTask(taskId, name);
    setTasks(prev => prev.map(t => t.id === taskId ? { ...t, name } : t));
  }

  async function handleDelete(taskId) {
    try {
      await dlApi.deleteTask(taskId);
      message.success('任务已删除');
      void loadPage(page);
    } catch (err) {
      message.error(err?.response?.data?.detail ?? '删除失败');
    }
  }

  async function handleStop(taskId) {
    try {
      await dlApi.stopTask(taskId);
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
      render: (_, record) => <EditableNameCell record={record} onSave={handleRename} />,
    },
    { title: '架构', dataIndex: 'model_type', key: 'model_type', width: 120 },
    { title: '任务类型', dataIndex: 'task_type', key: 'task_type', width: 100 },
    {
      title: '状态', dataIndex: 'status', key: 'status', width: 90,
      render: v => {
        const s = (v ?? '').toUpperCase();
        return <Tag color={statusColors[s] ?? 'default'}>{statusLabels[s] ?? s}</Tag>;
      },
    },
    {
      title: '进度', dataIndex: 'progress', key: 'progress', width: 120,
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
      title: '创建时间', dataIndex: 'created_at', key: 'created_at', width: 160,
      render: v => new Date(v).toLocaleString('zh-CN'),
    },
    {
      title: '操作', key: 'actions', width: 160,
      render: (_, record) => {
        const s = (record.status ?? '').toUpperCase();
        return (
          <Space size={4} onClick={e => e.stopPropagation()}>
            <Button
              size="small"
              icon={<EyeOutlined />}
              onClick={() => navigate(`/dl/monitor?taskId=${record.id}`)}
            >
              查看
            </Button>
            {s === 'SUCCESS' && (
              <Button
                size="small"
                type="primary"
                icon={<TrophyOutlined />}
                onClick={() => navigate(`/dl/results?taskId=${record.id}`)}
              >
                结果
              </Button>
            )}
            {s === 'RUNNING' && (
              <Button size="small" danger icon={<StopOutlined />}
                onClick={() => void handleStop(record.id)}>停止</Button>
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
        <Title level={2} style={{ margin: 0 }}>深度学习训练监控</Title>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={() => void loadPage(page)}>刷新</Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => navigate('/dl/config')}>
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
            onClick: () => navigate(`/dl/monitor?taskId=${record.id}`),
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
function DLTaskDetailView({ taskId, navigate }) {
  const [taskInfo, setTaskInfo] = useState(null);
  const [lossHistory, setLossHistory] = useState([]);
  const [wsConnected, setWsConnected] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stopping, setStopping] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    void loadStatus();
    return () => {
      if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
    };
  }, [taskId]);

  async function loadStatus() {
    try {
      const data = await dlApi.getStatus(taskId);
      setTaskInfo(data);
      setProgress(data.progress ?? 0);
      const finalStatuses = ['SUCCESS', 'FAILED'];
      if (!finalStatuses.includes((data.status ?? '').toUpperCase())) {
        openWebSocket();
      }
    } catch {
      message.error('加载任务状态失败');
    }
  }

  function openWebSocket() {
    if (wsRef.current) return;
    const ws = new WebSocket(`ws://localhost:8000/api/dl/ws/${taskId}`);
    wsRef.current = ws;
    ws.onopen = () => setWsConnected(true);
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'epoch') {
          setProgress(msg.progress ?? 0);
          setLossHistory(prev => {
            const next = [...prev, msg];
            return next.length > 500 ? next.slice(-500) : next;
          });
          setTaskInfo(prev => prev ? { ...prev, progress: msg.progress ?? 0, status: 'RUNNING' } : prev);
        }
        if (msg.type === 'done') {
          setWsConnected(false);
          ws.close();
          wsRef.current = null;
          void loadStatus();
        }
      } catch { /* ignore */ }
    };
    ws.onerror = () => setWsConnected(false);
    ws.onclose = () => setWsConnected(false);
  }

  async function handleStop() {
    setStopping(true);
    try {
      await dlApi.stopTask(taskId);
      message.success('已发送停止指令');
      void loadStatus();
    } catch {
      message.error('停止任务失败');
    } finally {
      setStopping(false);
    }
  }

  const status = (taskInfo?.status ?? 'PENDING').toUpperCase();
  const isRunning = status === 'RUNNING';
  const taskType = taskInfo?.task_type ?? 'auto';
  const tableData = [...lossHistory].slice(-20).reverse().map((row, idx) => ({ ...row, key: idx }));

  return (
    <div>
      <Space style={{ width: '100%', justifyContent: 'space-between', marginBottom: 24 }}>
        <Space>
          <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/dl/monitor')}>
            返回列表
          </Button>
          <Title level={2} style={{ margin: 0 }}>深度学习训练详情</Title>
        </Space>
        <Space>
          {wsConnected && <Badge status="processing" text="WebSocket 已连接" />}
          {!wsConnected && <Badge status="default" text="未连接" />}
          {taskInfo?.status === 'SUCCESS' && (
            <Button
              icon={<TrophyOutlined />}
              type="primary"
              onClick={() => navigate(`/dl/results?taskId=${taskId}`)}
            >
              查看结果
            </Button>
          )}
          {isRunning && (
            <Button danger icon={<StopOutlined />} loading={stopping} onClick={handleStop}>
              停止训练
            </Button>
          )}
        </Space>
      </Space>

      <Card style={{ marginBottom: 24 }}>
        {taskInfo ? (
          <Space direction="vertical" size={12} style={{ width: '100%' }}>
            <Space wrap>
              <Text strong>任务名称：</Text>
              <Text>{taskInfo.name ?? '-'}</Text>
              <Text strong style={{ marginLeft: 16 }}>任务 ID：</Text>
              <Text code>{taskId}</Text>
              <Text strong style={{ marginLeft: 16 }}>模型：</Text>
              <Text>{taskInfo.model_type ?? '-'}</Text>
              <Text strong style={{ marginLeft: 16 }}>状态：</Text>
              <Tag color={statusColors[status] ?? 'default'}>
                {statusLabels[status] ?? status}
              </Tag>
            </Space>
            <Progress
              percent={Math.round(progress)}
              status={status === 'FAILED' ? 'exception' : status === 'SUCCESS' ? 'success' : 'active'}
              strokeColor={isRunning ? { from: '#108ee9', to: '#87d068' } : undefined}
            />
          </Space>
        ) : (
          <Text type="secondary">正在加载任务信息…</Text>
        )}
      </Card>

      {lossHistory.length > 0 && (
        <Space size={16} style={{ width: '100%', display: 'flex', marginBottom: 24 }}>
          <Card title="训练损失" style={{ flex: 1, minWidth: 0 }} bodyStyle={{ padding: 8 }}>
            <EChart option={buildLossOption(lossHistory)} style={{ height: 300 }} notMerge />
          </Card>
          <Card title="验证指标" style={{ flex: 1, minWidth: 0 }} bodyStyle={{ padding: 8 }}>
            <EChart option={buildMetricOption(lossHistory, taskType)} style={{ height: 300 }} notMerge />
          </Card>
        </Space>
      )}

      {lossHistory.length > 0 && (
        <Card title={`近期 Epoch 记录（最近 ${Math.min(20, lossHistory.length)} 条）`}>
          <Table
            rowKey="key"
            dataSource={tableData}
            columns={epochColumns}
            pagination={false}
            size="small"
          />
        </Card>
      )}

      {lossHistory.length === 0 && taskInfo && (
        <Card>
          <Text type="secondary">
            {status === 'PENDING' ? '训练尚未开始，等待 Epoch 数据…' : '暂无 Epoch 数据。'}
          </Text>
        </Card>
      )}
    </div>
  );
}

// ── Root component ────────────────────────────────────────────────────────────
const DLMonitor = () => {
  const query = useQuery();
  const navigate = useNavigate();
  const taskId = query.get('taskId');

  if (taskId) {
    return <DLTaskDetailView taskId={taskId} navigate={navigate} />;
  }
  return <DLTaskListView navigate={navigate} />;
};

export default DLMonitor;
