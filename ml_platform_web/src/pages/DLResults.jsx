import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Alert, Button, Card, Col, Empty, Pagination,
  Row, Space, Statistic, Table, Tag, Typography, message,
} from 'antd';
import { ArrowLeftOutlined, DownloadOutlined, EyeOutlined, ReloadOutlined } from '@ant-design/icons';
import EChart from '../components/EChart';
import { dlApi } from '../services/api';

const { Title, Text } = Typography;

function useQuery() {
  return new URLSearchParams(useLocation().search);
}

// ── Metric helpers ────────────────────────────────────────────────────────────
const metricDisplayNames = {
  best_val_loss: '最优验证损失',
  val_acc:       '验证准确率',
  val_rmse:      '验证 RMSE',
  val_mae:       '验证 MAE',
  val_r2:        '验证 R²',
  final_epoch:   '最终 Epoch',
  train_loss:    '训练损失',
};
function getMetricDisplayName(key) { return metricDisplayNames[key] ?? key; }
function formatMetricValue(key, value) {
  if (value == null) return '-';
  if (key === 'final_epoch') return String(Math.round(value));
  if (key === 'val_acc') return `${(value * 100).toFixed(2)}%`;
  if (typeof value === 'number') return value.toFixed(4);
  return String(value);
}

// ── Loss history chart ────────────────────────────────────────────────────────
function buildHistoryOption(history) {
  const epochs = history.map(d => d.epoch ?? d.step ?? '');
  return {
    tooltip: { trigger: 'axis' },
    legend: { data: ['train_loss', 'val_loss'] },
    xAxis: { type: 'category', data: epochs, name: 'Epoch' },
    yAxis: { type: 'value', name: 'Loss' },
    series: [
      { name: 'train_loss', type: 'line', smooth: true, data: history.map(d => d.train_loss ?? null) },
      { name: 'val_loss',   type: 'line', smooth: true, data: history.map(d => d.val_loss   ?? null) },
    ],
  };
}

const PAGE_SIZE = 10;

// ── List view ─────────────────────────────────────────────────────────────────
function DLResultListView({ navigate }) {
  const [tasks, setTasks] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);

  useEffect(() => { void load(page); }, [page]);

  async function load(p) {
    setLoading(true);
    try {
      const res = await dlApi.listTrainedModels({ page: p, page_size: PAGE_SIZE });
      setTasks(res.items ?? []);
      setTotal(res.total ?? 0);
    } catch {
      message.error('加载深度学习结果列表失败');
    } finally {
      setLoading(false);
    }
  }

  function bestMetric(r) {
    const m = r.result_metrics ?? {};
    if (m.val_acc  != null) return `准确率 ${(m.val_acc * 100).toFixed(1)}%`;
    if (m.val_rmse != null) return `RMSE ${m.val_rmse.toFixed(4)}`;
    if (m.val_mae  != null) return `MAE ${m.val_mae.toFixed(4)}`;
    return '-';
  }

  const columns = [
    {
      title: '任务名称',
      key: 'name',
      render: (_, r) => r.name ?? <Text type="secondary">{r.id.slice(0, 8)}</Text>,
    },
    {
      title: '架构',
      dataIndex: 'model_type',
      key: 'model_type',
      width: 120,
      render: v => <Tag color="purple">{v}</Tag>,
    },
    { title: '任务类型', dataIndex: 'task_type', key: 'task_type', width: 100 },
    {
      title: '最佳指标',
      key: 'metric',
      width: 160,
      render: (_, r) => bestMetric(r),
    },
    {
      title: '完成时间',
      dataIndex: 'finished_at',
      key: 'finished_at',
      width: 160,
      render: v => v ? new Date(v).toLocaleString('zh-CN') : '-',
    },
    {
      title: '操作',
      key: 'action',
      width: 80,
      render: (_, r) => (
        <Button
          size="small"
          icon={<EyeOutlined />}
          onClick={() => navigate(`/dl/results?taskId=${r.id}`)}
        >
          查看
        </Button>
      ),
    },
  ];

  return (
    <div>
      <Space style={{ width: '100%', justifyContent: 'space-between', marginBottom: 24 }}>
        <Title level={2} style={{ margin: 0 }}>深度学习结果可视化</Title>
        <Button icon={<ReloadOutlined />} onClick={() => void load(page)}>刷新</Button>
      </Space>
      <Card>
        <Text type="secondary" style={{ display: 'block', marginBottom: 16 }}>
          已完成训练的深度学习模型列表，点击「查看」查看详细训练指标和曲线。
        </Text>
        <Table
          rowKey="id"
          dataSource={tasks}
          columns={columns}
          loading={loading}
          pagination={false}
          size="middle"
          locale={{ emptyText: <Empty description="还没有完成训练的深度学习任务" /> }}
          onRow={r => ({
            style: { cursor: 'pointer' },
            onClick: () => navigate(`/dl/results?taskId=${r.id}`),
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
function DLResultDetailView({ taskId, navigate }) {
  const [taskInfo, setTaskInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => { void loadTask(); }, [taskId]);

  async function loadTask() {
    setLoading(true);
    try {
      const data = await dlApi.getStatus(taskId);
      setTaskInfo(data);
    } catch {
      message.error('加载任务结果失败');
    } finally {
      setLoading(false);
    }
  }

  const status  = (taskInfo?.status ?? '').toUpperCase();
  const metrics = taskInfo?.result_metrics ?? {};
  const metricEntries = Object.entries(metrics).filter(([, v]) => v != null);
  const history = taskInfo?.history ?? [];

  return (
    <div>
      <Space style={{ width: '100%', justifyContent: 'space-between', marginBottom: 24 }}>
        <Space>
          <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/dl/results')}>
            返回列表
          </Button>
          <Title level={2} style={{ margin: 0 }}>深度学习结果详情</Title>
        </Space>
        <Button icon={<ReloadOutlined />} loading={loading} onClick={() => void loadTask()}>
          刷新
        </Button>
      </Space>

      {/* Task meta */}
      {taskInfo && (
        <Card style={{ marginBottom: 16 }}>
          <Space wrap size={[24, 8]}>
            <span><Text type="secondary">任务名称：</Text><Text strong>{taskInfo.name ?? '-'}</Text></span>
            <span><Text type="secondary">架构：</Text><Tag color="purple">{taskInfo.model_type}</Tag></span>
            <span><Text type="secondary">任务类型：</Text><Text>{taskInfo.task_type}</Text></span>
            <span><Text type="secondary">完成时间：</Text>
              <Text>{taskInfo.finished_at ? new Date(taskInfo.finished_at).toLocaleString('zh-CN') : '-'}</Text>
            </span>
          </Space>
        </Card>
      )}

      {loading && <Card><Text type="secondary">正在加载结果…</Text></Card>}

      {!loading && status === 'FAILED' && (
        <Alert type="error" showIcon message="训练失败"
          description={taskInfo?.error_message ?? '未知错误，请查看日志。'} />
      )}

      {!loading && status === 'SUCCESS' && (
        <>
          {/* Metric cards */}
          {metricEntries.length > 0 && (
            <Card title="训练指标" style={{ marginBottom: 16 }}>
              <Row gutter={[16, 16]}>
                {metricEntries.map(([key, value]) => (
                  <Col key={key} xs={12} sm={8} md={6}>
                    <Card size="small" bordered>
                      <Statistic
                        title={getMetricDisplayName(key)}
                        value={formatMetricValue(key, value)}
                        valueStyle={{ fontSize: 18 }}
                      />
                    </Card>
                  </Col>
                ))}
              </Row>
            </Card>
          )}

          {/* Training history chart */}
          {history.length > 0 && (
            <Card title="训练曲线" style={{ marginBottom: 16 }}>
              <EChart option={buildHistoryOption(history)} style={{ height: 360 }} />
            </Card>
          )}

          {history.length === 0 && (
            <Card style={{ marginBottom: 16 }}>
              <Text type="secondary">训练历史数据不可用，仅展示最终指标。</Text>
            </Card>
          )}

          <Card title="模型文件">
            <Button icon={<DownloadOutlined />} disabled>
              下载模型文件（暂不可用）
            </Button>
          </Card>
        </>
      )}

      {!loading && status !== 'SUCCESS' && status !== 'FAILED' && (
        <Alert type="info" showIcon message="训练尚未完成"
          description={
            <span>
              当前状态：{status || '未知'}。
              <Button type="link" style={{ padding: 0, marginLeft: 4 }}
                onClick={() => navigate('/dl/monitor')}>
                前往监控页
              </Button>
            </span>
          }
        />
      )}
    </div>
  );
}

// ── Root ──────────────────────────────────────────────────────────────────────
const DLResults = () => {
  const query = useQuery();
  const navigate = useNavigate();
  const taskId = query.get('taskId');

  if (taskId) return <DLResultDetailView taskId={taskId} navigate={navigate} />;
  return <DLResultListView navigate={navigate} />;
};

export default DLResults;
