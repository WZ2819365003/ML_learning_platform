import React, { useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Badge, Button, Card, Col, Pagination, Popconfirm, Progress,
  Row, Space, Table, Tag, Typography, message,
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
  const epochs    = lossHistory.map(d => d.epoch);
  const valLosses = lossHistory.map(d => d.val_loss ?? null);

  // Detect task type from data if 'auto'
  let resolvedType = taskType;
  if (taskType === 'auto' || !taskType) {
    const first = lossHistory.find(d => d.val_acc != null || d.val_rmse != null);
    resolvedType = first?.val_acc != null ? 'classification' : 'regression';
  }

  const isClassification = resolvedType === 'classification';
  const legendItems = ['val_loss'];
  const series = [
    { name: 'val_loss', type: 'line', smooth: true, yAxisIndex: 0, data: valLosses },
  ];

  if (isClassification) {
    const hasF1 = lossHistory.some(d => d.val_f1_macro != null);
    legendItems.push('val_acc');
    series.push({
      name: 'val_acc', type: 'line', smooth: true, yAxisIndex: 1,
      data: lossHistory.map(d => d.val_acc ?? null),
    });
    if (hasF1) {
      legendItems.push('val_f1');
      series.push({
        name: 'val_f1', type: 'line', smooth: true, yAxisIndex: 1,
        lineStyle: { type: 'dashed' },
        data: lossHistory.map(d => d.val_f1_macro ?? null),
      });
    }
  } else {
    legendItems.push('val_rmse');
    series.push({
      name: 'val_rmse', type: 'line', smooth: true, yAxisIndex: 1,
      data: lossHistory.map(d => d.val_rmse ?? null),
    });
  }

  return {
    tooltip: { trigger: 'axis' },
    legend: { data: legendItems },
    xAxis: { type: 'category', data: epochs, name: 'Epoch' },
    yAxis: [
      { type: 'value', name: 'Loss' },
      { type: 'value', name: isClassification ? '准确率 / F1' : 'RMSE', position: 'right' },
    ],
    series,
  };
}

// Task-type-aware epoch table columns.
// Classification: focus on val_loss + val_acc (F1 is noisy on binary tasks;
// only shown when ≥3 classes and data actually contains it).
// Regression: replace accuracy/F1 with RMSE + MAE, which are the real
// optimization targets for regression runs.
function buildEpochColumns(taskType, rows = []) {
  const cols = [
    { title: 'Epoch',      dataIndex: 'epoch',      key: 'epoch',      width: 70 },
    { title: 'Train Loss', dataIndex: 'train_loss', key: 'train_loss', width: 120, render: v => v != null ? v.toFixed(6) : '-' },
    { title: 'Val Loss',   dataIndex: 'val_loss',   key: 'val_loss',   width: 120, render: v => v != null ? v.toFixed(6) : '-' },
  ];
  // Resolve effective task type when backend said 'auto'.
  let kind = taskType;
  if (!kind || kind === 'auto') {
    kind = rows.some(r => r.val_rmse != null || r.val_mae != null) ? 'regression' : 'classification';
  }
  if (kind === 'regression') {
    cols.push(
      { title: 'Val RMSE', dataIndex: 'val_rmse', key: 'val_rmse', width: 110, render: v => v != null ? v.toFixed(4) : '-' },
      { title: 'Val MAE',  dataIndex: 'val_mae',  key: 'val_mae',  width: 110, render: v => v != null ? v.toFixed(4) : '-' },
    );
  } else {
    cols.push({ title: 'Val Acc', dataIndex: 'val_acc', key: 'val_acc', width: 110, render: v => v != null ? v.toFixed(4) : '-' });
    // Only surface F1 when it's actually populated AND multiclass (>2 classes
    // makes F1 more informative than accuracy). Binary F1 ≈ accuracy, so we
    // hide it by default to reduce noise.
    const hasInformativeF1 = rows.some(r => r.val_f1_macro != null && r.val_f1_macro > 0);
    if (hasInformativeF1) {
      cols.push({ title: 'Val F1', dataIndex: 'val_f1_macro', key: 'val_f1', width: 110, render: v => v != null ? v.toFixed(4) : '-' });
    }
  }
  cols.push({ title: 'LR', dataIndex: 'lr', key: 'lr', width: 100, render: v => v != null ? v.toExponential(3) : '-' });
  return cols;
}

function formatLogExtra(extra) {
  if (!extra) {
    return '';
  }
  return Object.entries(extra)
    .map(([key, value]) => `${key}=${value}`)
    .join(' | ');
}

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

const EPOCH_PAGE_SIZE = 10;

// ── Detail view ───────────────────────────────────────────────────────────────
function DLTaskDetailView({ taskId, navigate }) {
  const [taskInfo, setTaskInfo] = useState(null);
  const [lossHistory, setLossHistory] = useState([]);
  const [wsConnected, setWsConnected] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stopping, setStopping] = useState(false);
  const [logEntries, setLogEntries] = useState([]);
  const [epochPage, setEpochPage] = useState(1);
  const logEndRef = useRef(null);
  const wsRef = useRef(null);
  const logWsRef = useRef(null);
  const wsBase = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.hostname}:8000`;

  useEffect(() => {
    void loadStatus();
    void loadLogs();
    return () => {
      if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
      if (logWsRef.current) { logWsRef.current.close(); logWsRef.current = null; }
    };
  }, [taskId]);

  // Heartbeat: poll logs every 4 s while training is running (catches missed WS messages)
  useEffect(() => {
    if (!taskInfo) return;
    const s = (taskInfo.status ?? '').toUpperCase();
    if (s !== 'RUNNING' && s !== 'PENDING') return;
    const timer = setInterval(async () => {
      try {
        const res = await dlApi.getLogs(taskId, { page: 1, page_size: 1000 });
        const newEntries = res.entries ?? [];
        setLogEntries(prev => newEntries.length >= prev.length ? newEntries : prev);
      } catch { /* ignore */ }
    }, 4000);
    return () => clearInterval(timer);
  }, [taskInfo?.status, taskId]);

  // Auto-scroll log panel when new entries arrive
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logEntries]);

  async function loadStatus() {
    try {
      const data = await dlApi.getStatus(taskId);
      setTaskInfo(data);
      setProgress(data.progress ?? 0);
      // Always load epoch history from DB (restores charts after page refresh)
      await loadEpochHistory();
      const finalStatuses = ['SUCCESS', 'FAILED'];
      if (!finalStatuses.includes((data.status ?? '').toUpperCase())) {
        openWebSocket();
        openLogWebSocket();
      }
    } catch {
      message.error('加载任务状态失败');
    }
  }

  async function loadEpochHistory() {
    try {
      const res = await dlApi.getEpochs(taskId, { page: 1, page_size: 2000 });
      const items = res.items ?? [];
      if (items.length > 0) {
        setLossHistory(items.map(e => ({
          epoch: e.epoch,
          total: e.total_epochs,
          train_loss: e.train_loss,
          val_loss: e.val_loss,
          val_acc: e.val_acc,
          val_f1_macro: e.val_f1_macro,
          val_rmse: e.val_rmse,
          val_mae: e.val_mae,
          val_r2: e.val_r2,
          lr: e.lr,
        })));
      }
    } catch {
      // No epoch data yet — not an error
    }
  }

  async function loadLogs() {
    try {
      const res = await dlApi.getLogs(taskId, { page: 1, page_size: 1000 });
      setLogEntries(res.entries ?? []);
    } catch {
      // Logs may not exist yet — not an error
    }
  }

  function openWebSocket() {
    if (wsRef.current) return;
    const ws = new WebSocket(`${wsBase}/api/dl/ws/${taskId}`);
    wsRef.current = ws;
    ws.onopen = () => setWsConnected(true);
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'epoch') {
          setProgress(msg.progress ?? 0);
          setLossHistory(prev => {
            // Deduplicate: skip if this epoch is already in the list (loaded from DB)
            const lastEpoch = prev.length > 0 ? prev[prev.length - 1].epoch : 0;
            if (msg.epoch <= lastEpoch) return prev;
            const next = [...prev, msg];
            return next.length > 2000 ? next.slice(-2000) : next;
          });
          setTaskInfo(prev => prev ? { ...prev, progress: msg.progress ?? 0, status: 'RUNNING' } : prev);
        }
        if (msg.type === 'done') {
          setWsConnected(false);
          ws.close();
          wsRef.current = null;
          void loadStatus();
          // Reload logs once training finishes
          setTimeout(() => {
            void loadLogs();
            if (logWsRef.current) {
              logWsRef.current.close();
              logWsRef.current = null;
            }
          }, 1000);
        }
      } catch { /* ignore */ }
    };
    ws.onerror = () => setWsConnected(false);
    ws.onclose = () => setWsConnected(false);
  }

  function openLogWebSocket() {
    if (logWsRef.current) return;
    const ws = new WebSocket(`${wsBase}/ws/logs/${taskId}`);
    logWsRef.current = ws;
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type !== 'log') return;
        const entry = {
          level: msg.level ?? 'INFO',
          message: msg.message ?? '',
          extra: msg.extra ?? null,
          created_at: msg.timestamp ?? msg.created_at ?? new Date().toISOString(),
        };
        setLogEntries(prev => {
          const next = [...prev, entry];
          return next.length > 1000 ? next.slice(-1000) : next;
        });
      } catch { /* ignore */ }
    };
    ws.onclose = () => {
      if (logWsRef.current === ws) logWsRef.current = null;
    };
    ws.onerror = () => {
      if (logWsRef.current === ws) logWsRef.current = null;
    };
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
  // Epoch table: all epochs descending, client-side pagination
  const tableData = [...lossHistory].reverse().map((row, idx) => ({ ...row, key: idx }));
  const pagedEpochData = tableData.slice((epochPage - 1) * EPOCH_PAGE_SIZE, epochPage * EPOCH_PAGE_SIZE);

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
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={24} xl={12}>
            <Card title="训练损失" bodyStyle={{ padding: 12 }}>
              <EChart option={buildLossOption(lossHistory)} style={{ height: 340, width: '100%' }} notMerge />
            </Card>
          </Col>
          <Col xs={24} xl={12}>
            <Card title="验证指标" bodyStyle={{ padding: 12 }}>
              <EChart option={buildMetricOption(lossHistory, taskType)} style={{ height: 340, width: '100%' }} notMerge />
            </Card>
          </Col>
        </Row>
      )}

      {lossHistory.length > 0 && (
        <Card
          title={`Epoch 记录（共 ${lossHistory.length} 条）`}
          style={{ marginBottom: 24 }}
          // Cap the Card body so it stops growing with rows — the Table below
          // handles its own vertical scroll. Previously the outer Card expanded
          // and pushed the log panel off-screen while training was running.
          styles={{ body: { maxHeight: 560, overflow: 'hidden', padding: 16 } }}
        >
          <Table
            rowKey="key"
            dataSource={pagedEpochData}
            columns={buildEpochColumns(taskType, lossHistory)}
            pagination={false}
            size="small"
            scroll={{ y: 440, x: 900 }}
          />
          <div style={{ marginTop: 12, textAlign: 'right' }}>
            <Pagination
              current={epochPage}
              pageSize={EPOCH_PAGE_SIZE}
              total={lossHistory.length}
              onChange={p => setEpochPage(p)}
              showTotal={t => `共 ${t} 条`}
              showSizeChanger={false}
              size="small"
            />
          </div>
        </Card>
      )}

      {lossHistory.length === 0 && taskInfo && (
        <Card style={{ marginBottom: 24 }}>
          <Text type="secondary">
            {status === 'PENDING' ? '训练尚未开始，等待 Epoch 数据…' : '暂无 Epoch 数据。'}
          </Text>
        </Card>
      )}

      {/* Training log panel */}
      <Card
        title={
          <Space>
            <span>训练日志</span>
            <Text type="secondary" style={{ fontSize: 12 }}>
              ({logEntries.length} 条)
            </Text>
          </Space>
        }
        extra={
          <Button size="small" onClick={() => void loadLogs()}>刷新日志</Button>
        }
      >
        <div
          data-testid="dl-log-panel"
          style={{
            height: 300,
            overflowY: 'auto',
            background: '#0f172a',
            borderRadius: 8,
            padding: 12,
            fontFamily: 'monospace',
            fontSize: 12,
          }}
        >
          {logEntries.length === 0 ? (
            <Text style={{ color: '#64748b' }}>
              暂无日志。训练开始后日志将在这里显示。
            </Text>
          ) : (
            logEntries.map((entry, i) => {
              const color = entry.level === 'ERROR' ? '#f87171'
                : entry.level === 'WARN' ? '#fbbf24'
                : '#86efac';
              const ts = entry.created_at
                ? new Date(entry.created_at).toLocaleTimeString('zh-CN')
                : '';
              return (
                <div key={i} data-testid="dl-log-entry" style={{ marginBottom: 2 }}>
                  <span style={{ color: '#64748b' }}>{ts} </span>
                  <span style={{ color }}>[{entry.level}] </span>
                  <span style={{ color: '#e2e8f0' }}>{entry.message}</span>
                  {entry.extra && (
                    <span style={{ color: '#93c5fd' }}> | {formatLogExtra(entry.extra)}</span>
                  )}
                </div>
              );
            })
          )}
          <div ref={logEndRef} />
        </div>
      </Card>
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
