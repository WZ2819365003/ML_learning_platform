import React, { useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Badge, Button, Card, Progress, Space, Table, Tag, Typography, message,
} from 'antd';
import { StopOutlined, TrophyOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
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

// ── Build ECharts option for loss curve ──────────────────────────────────────
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

// ── Build ECharts option for validation metrics (dual Y-axis) ────────────────
function buildMetricOption(lossHistory, taskType) {
  const epochs = lossHistory.map(d => d.epoch);
  const valLosses = lossHistory.map(d => d.val_loss ?? null);

  // Determine secondary metric based on task type
  let secKey = null;
  let secName = '';
  if (taskType === 'classification') {
    secKey = 'val_acc';
    secName = 'val_acc';
  } else if (taskType === 'regression') {
    secKey = 'val_rmse';
    secName = 'val_rmse';
  } else {
    // Auto-detect from first data point that has one of them
    const first = lossHistory.find(d => d.val_acc != null || d.val_rmse != null);
    if (first?.val_acc != null) { secKey = 'val_acc'; secName = 'val_acc'; }
    else if (first?.val_rmse != null) { secKey = 'val_rmse'; secName = 'val_rmse'; }
  }

  const secData = secKey ? lossHistory.map(d => d[secKey] ?? null) : [];
  const legendData = secKey ? ['val_loss', secName] : ['val_loss'];

  const series = [
    { name: 'val_loss', type: 'line', smooth: true, yAxisIndex: 0, data: valLosses },
  ];
  if (secKey) {
    series.push({ name: secName, type: 'line', smooth: true, yAxisIndex: 1, data: secData });
  }

  return {
    tooltip: { trigger: 'axis' },
    legend: { data: legendData },
    xAxis: { type: 'category', data: epochs, name: 'Epoch' },
    yAxis: [
      { type: 'value', name: 'Loss' },
      { type: 'value', name: secKey ? secName : '', position: 'right' },
    ],
    series,
  };
}

// ── Main Page ─────────────────────────────────────────────────────────────────
const DLMonitor = () => {
  const query = useQuery();
  const taskId = query.get('taskId');
  const navigate = useNavigate();

  const [taskInfo, setTaskInfo] = useState(null);
  const [lossHistory, setLossHistory] = useState([]);
  const [wsConnected, setWsConnected] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stopping, setStopping] = useState(false);

  const wsRef = useRef(null);

  // ── Load initial status and open WS ──────────────────────────────────────
  useEffect(() => {
    if (!taskId) return;
    void loadStatus();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
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
    } catch (err) {
      console.error('加载任务状态失败:', err);
      message.error('加载任务状态失败');
    }
  }

  function openWebSocket() {
    if (wsRef.current) return;

    const ws = new WebSocket(`ws://localhost:8000/api/dl/ws/${taskId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        if (msg.type === 'epoch') {
          setProgress(msg.progress ?? 0);
          setLossHistory(prev => {
            const next = [...prev, msg];
            // keep last 500 entries to avoid unbounded growth
            return next.length > 500 ? next.slice(-500) : next;
          });
          // update task info progress in place
          setTaskInfo(prev => prev ? { ...prev, progress: msg.progress ?? 0, status: 'RUNNING' } : prev);
        }

        if (msg.type === 'done') {
          setWsConnected(false);
          ws.close();
          wsRef.current = null;
          // reload full status from server
          void loadStatus();
        }
      } catch (e) {
        console.warn('WS消息解析失败:', e);
      }
    };

    ws.onerror = () => {
      setWsConnected(false);
    };

    ws.onclose = () => {
      setWsConnected(false);
    };
  }

  // ── Stop task ────────────────────────────────────────────────────────────
  async function handleStop() {
    setStopping(true);
    try {
      await dlApi.stopTask(taskId);
      message.success('已发送停止指令');
      void loadStatus();
    } catch (err) {
      console.error('停止任务失败:', err);
      message.error('停止任务失败');
    } finally {
      setStopping(false);
    }
  }

  // ── Derived ───────────────────────────────────────────────────────────────
  const status = (taskInfo?.status ?? 'PENDING').toUpperCase();
  const isRunning = status === 'RUNNING';
  const taskType = taskInfo?.task_type ?? 'auto';

  // Last 20 rows for the epoch table
  const tableData = [...lossHistory].slice(-20).reverse().map((row, idx) => ({
    ...row,
    key: idx,
  }));

  const epochColumns = [
    { title: 'Epoch', dataIndex: 'epoch', key: 'epoch', width: 80 },
    {
      title: 'Train Loss',
      dataIndex: 'train_loss',
      key: 'train_loss',
      render: v => v != null ? v.toFixed(6) : '-',
    },
    {
      title: 'Val Loss',
      dataIndex: 'val_loss',
      key: 'val_loss',
      render: v => v != null ? v.toFixed(6) : '-',
    },
    {
      title: 'Val Acc / RMSE',
      key: 'val_metric',
      render: (_, row) => {
        if (row.val_acc != null) return row.val_acc.toFixed(4);
        if (row.val_rmse != null) return row.val_rmse.toFixed(4);
        return '-';
      },
    },
    {
      title: 'LR',
      dataIndex: 'lr',
      key: 'lr',
      render: v => v != null ? v.toExponential(3) : '-',
    },
  ];

  return (
    <div>
      <Space style={{ width: '100%', justifyContent: 'space-between', marginBottom: 24 }}>
        <Title level={2} style={{ margin: 0 }}>
          深度学习训练监控
        </Title>
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
            <Button
              danger
              icon={<StopOutlined />}
              loading={stopping}
              onClick={handleStop}
            >
              停止训练
            </Button>
          )}
        </Space>
      </Space>

      {/* ── Task Info Card ──────────────────────────────────────────────── */}
      <Card style={{ marginBottom: 24 }}>
        {taskInfo ? (
          <Space direction="vertical" size={12} style={{ width: '100%' }}>
            <Space wrap>
              <Text strong>任务 ID：</Text>
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

      {/* ── Charts ─────────────────────────────────────────────────────── */}
      {lossHistory.length > 0 && (
        <Space size={16} style={{ width: '100%', display: 'flex', marginBottom: 24 }}>
          <Card
            title="训练损失"
            style={{ flex: 1, minWidth: 0 }}
            bodyStyle={{ padding: 8 }}
          >
            <ReactECharts
              option={buildLossOption(lossHistory)}
              style={{ height: 300 }}
              notMerge
            />
          </Card>
          <Card
            title="验证指标"
            style={{ flex: 1, minWidth: 0 }}
            bodyStyle={{ padding: 8 }}
          >
            <ReactECharts
              option={buildMetricOption(lossHistory, taskType)}
              style={{ height: 300 }}
              notMerge
            />
          </Card>
        </Space>
      )}

      {/* ── Epoch Table ────────────────────────────────────────────────── */}
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
};

export default DLMonitor;
