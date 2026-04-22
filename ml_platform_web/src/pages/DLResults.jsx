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
  best_val_loss:    '最优验证损失',
  val_acc:          '验证准确率',
  val_error:        '验证误差',
  val_rmse:         '验证 RMSE',
  val_mae:          '验证 MAE',
  val_r2:           '验证 R²',
  val_mape:         '验证 MAPE',
  final_epoch:      '最终 Epoch',
  train_loss:       '训练损失',
  val_precision:    '验证精确率',
  val_recall:       '验证召回率',
  val_auc_roc:      '验证 AUC-ROC',
};
function getMetricDisplayName(key) { return metricDisplayNames[key] ?? key; }
function formatMetricValue(key, value) {
  if (value == null) return '-';
  if (key === 'final_epoch') return String(Math.round(value));
  if (key === 'val_acc' || key === 'val_error') return `${(value * 100).toFixed(2)}%`;
  if (typeof value === 'number') return value.toFixed(4);
  return String(value);
}

// Scalar-only metric keys to display as Statistic cards.
// Order matters — the most task-relevant metrics go first (train/val loss
// are universal, then regression error metrics before classification
// accuracy/error, because the next filter step drops non-matching keys anyway).
const SCALAR_METRIC_DISPLAY_ORDER = [
  'best_val_loss',
  // Regression core (RMSE / MAE / R² / MAPE)
  'val_rmse', 'val_mae', 'val_r2', 'val_mape',
  // Classification headline metrics
  'val_acc', 'val_error',
  'val_precision', 'val_recall', 'val_auc_roc',
  'final_epoch',
];

// ── Chart option builders ─────────────────────────────────────────────────────
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

function buildMetricHistoryOption(history, taskType) {
  const epochs = history.map(d => d.epoch ?? d.step ?? '');
  // Resolve effective type: trust backend first, else sniff from data (acc
  // implies classification, rmse/mae implies regression). This keeps the
  // legend honest and stops us from drawing phantom metric lines on
  // regression runs (the old bug).
  let kind = taskType;
  if (!kind || kind === 'auto') {
    kind = history.some(d => d.val_rmse != null || d.val_mae != null) ? 'regression' : 'classification';
  }

  // Pick only the series whose key is actually populated — avoids phantom
  // zero/null lines cluttering the chart.
  const candidates = kind === 'regression'
    ? [
        { key: 'val_rmse', name: 'val_rmse' },
        { key: 'val_mae',  name: 'val_mae'  },
        { key: 'val_r2',   name: 'val_r2'   },
      ]
    : [
        { key: 'val_acc',      name: 'val_acc' },
        { key: 'val_error',    name: 'val_error' },
      ];
  const series = candidates
    .filter(c => history.some(d => c.key === 'val_error' ? d.val_acc != null : d[c.key] != null))
    .map(c => ({
      name: c.name, type: 'line', smooth: true,
      data: history.map(d => {
        if (c.key === 'val_error') return d.val_acc != null ? 1 - d.val_acc : null;
        return d[c.key] ?? null;
      }),
    }));

  return {
    tooltip: { trigger: 'axis' },
    legend: { data: series.map(s => s.name) },
    xAxis: { type: 'category', data: epochs, name: 'Epoch' },
    yAxis: {
      type: 'value',
      name: kind === 'regression' ? '误差' : '指标',
      ...(kind === 'regression' ? {} : { min: 0, max: 1 }),
    },
    series,
  };
}

function buildConfusionMatrixOption(cm) {
  const n = cm.length;
  const labels = Array.from({ length: n }, (_, i) => String(i));
  const data   = cm.flatMap((row, ri) => row.map((v, ci) => [ci, ri, v]));
  const maxVal = Math.max(...cm.flat(), 1);
  return {
    tooltip: {
      position: 'top',
      formatter: p => `真实类别: ${labels[p.data[1]]}<br/>预测类别: ${labels[p.data[0]]}<br/>数量: ${p.data[2]}`,
    },
    grid: { height: '65%', top: 36, left: 72, right: 28, bottom: 60 },
    xAxis: { type: 'category', data: labels, name: '预测类别', nameLocation: 'middle', nameGap: 30 },
    yAxis: { type: 'category', data: labels, name: '真实类别', nameLocation: 'middle', nameGap: 40 },
    visualMap: {
      min: 0, max: maxVal,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 0,
      inRange: { color: ['#edf8fb', '#006d77'] },
    },
    series: [{ type: 'heatmap', data, label: { show: true } }],
  };
}

function buildRocCurveOption(fpr, tpr, auc) {
  const data = fpr.map((v, i) => [v, tpr[i]]);
  return {
    tooltip: { trigger: 'axis' },
    legend: { data: [`ROC (AUC=${auc != null ? auc.toFixed(4) : '?'})`, '随机基线'], bottom: 0 },
    grid: { top: 24, left: 56, right: 20, bottom: 54 },
    xAxis: { type: 'value', name: 'FPR', min: 0, max: 1 },
    yAxis: { type: 'value', name: 'TPR', min: 0, max: 1 },
    series: [
      {
        name: `ROC (AUC=${auc != null ? auc.toFixed(4) : '?'})`,
        type: 'line', smooth: false,
        data,
        areaStyle: { opacity: 0.1 },
        lineStyle: { width: 2 },
        showSymbol: false,
      },
      {
        name: '随机基线', type: 'line',
        data: [[0, 0], [1, 1]],
        lineStyle: { type: 'dashed', color: '#94a3b8' },
        symbol: 'none',
      },
    ],
  };
}

function buildConfidenceDistOption(confidences) {
  // Build histogram with 20 bins [0,1]
  const bins = 20;
  const counts = Array(bins).fill(0);
  confidences.forEach(v => {
    const bin = Math.min(Math.floor(v * bins), bins - 1);
    counts[bin]++;
  });
  const labels = Array.from({ length: bins }, (_, i) => `${(i / bins).toFixed(2)}~${((i + 1) / bins).toFixed(2)}`);
  return {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { top: 24, left: 56, right: 20, bottom: 60 },
    xAxis: { type: 'category', data: labels, name: '预测置信度', axisLabel: { rotate: 30, fontSize: 10 } },
    yAxis: { type: 'value', name: '样本数' },
    series: [{
      type: 'bar',
      data: counts,
      itemStyle: { color: '#6366f1' },
    }],
  };
}

function buildScatterOption(actual, predicted) {
  const data = actual.map((a, i) => [a, predicted[i]]);
  const allVals = [...actual, ...predicted];
  const minV = Math.min(...allVals);
  const maxV = Math.max(...allVals);
  return {
    tooltip: { formatter: p => `实际: ${p.data[0].toFixed(4)}<br/>预测: ${p.data[1].toFixed(4)}` },
    grid: { top: 24, left: 64, right: 20, bottom: 54 },
    xAxis: { type: 'value', name: '实际值', min: minV, max: maxV },
    yAxis: { type: 'value', name: '预测值', min: minV, max: maxV },
    series: [
      {
        name: '预测 vs 实际',
        type: 'scatter',
        data,
        symbolSize: 5,
        itemStyle: { color: '#0ea5e9', opacity: 0.7 },
      },
      {
        name: '完美预测线',
        type: 'line',
        data: [[minV, minV], [maxV, maxV]],
        lineStyle: { type: 'dashed', color: '#f43f5e' },
        symbol: 'none',
      },
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

  const status          = (taskInfo?.status ?? '').toUpperCase();
  const rawMetrics      = taskInfo?.result_metrics ?? {};
  const history         = Array.isArray(rawMetrics.history) ? rawMetrics.history : [];
  const confusionMatrix = Array.isArray(rawMetrics.confusion_matrix) ? rawMetrics.confusion_matrix : null;
  const taskType        = taskInfo?.task_type ?? 'auto';

  // ROC curve (binary classification)
  const rocFpr = Array.isArray(rawMetrics.val_roc_fpr) ? rawMetrics.val_roc_fpr : null;
  const rocTpr = Array.isArray(rawMetrics.val_roc_tpr) ? rawMetrics.val_roc_tpr : null;
  const auc    = rawMetrics.val_auc_roc ?? null;

  // Prediction confidence distribution (classification)
  const confidenceDist = Array.isArray(rawMetrics.val_confidence_dist) ? rawMetrics.val_confidence_dist : null;

  // Predicted vs Actual scatter (regression)
  const scatter = rawMetrics.val_scatter ?? null;

  // Only show scalar numeric metrics in cards (exclude history, confusion_matrix, etc.)
  const derivedMetrics = {
    ...rawMetrics,
    ...(rawMetrics.val_error == null && rawMetrics.val_acc != null
      ? { val_error: 1 - rawMetrics.val_acc }
      : {}),
  };

  const metricEntries = SCALAR_METRIC_DISPLAY_ORDER
    .filter(k => derivedMetrics[k] != null && typeof derivedMetrics[k] === 'number')
    .map(k => [k, derivedMetrics[k]]);

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

          {/* Training history charts */}
          {history.length > 0 && (
            <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
              <Col xs={24} xl={12}>
                <Card title="训练损失曲线">
                  <EChart option={buildHistoryOption(history)} style={{ height: 320 }} />
                </Card>
              </Col>
              <Col xs={24} xl={12}>
                <Card title="验证指标曲线">
                  <EChart option={buildMetricHistoryOption(history, taskType)} style={{ height: 320 }} />
                </Card>
              </Col>
            </Row>
          )}

          {/* Confusion matrix + ROC curve */}
          {(confusionMatrix || (rocFpr && rocTpr)) && (
            <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
              {confusionMatrix && (
                <Col xs={24} xl={rocFpr ? 12 : 24}>
                  <Card title="混淆矩阵">
                    <EChart option={buildConfusionMatrixOption(confusionMatrix)} style={{ height: 340 }} />
                  </Card>
                </Col>
              )}
              {rocFpr && rocTpr && (
                <Col xs={24} xl={confusionMatrix ? 12 : 24}>
                  <Card title="ROC 曲线">
                    <EChart option={buildRocCurveOption(rocFpr, rocTpr, auc)} style={{ height: 340 }} />
                  </Card>
                </Col>
              )}
            </Row>
          )}

          {/* Prediction confidence distribution (classification) */}
          {confidenceDist && (
            <Card title="预测置信度分布" style={{ marginBottom: 16 }}>
              <EChart option={buildConfidenceDistOption(confidenceDist)} style={{ height: 280 }} />
            </Card>
          )}

          {/* Predicted vs Actual scatter (regression) */}
          {scatter?.actual && scatter?.predicted && (
            <Card title="预测值 vs 实际值" style={{ marginBottom: 16 }}>
              <EChart option={buildScatterOption(scatter.actual, scatter.predicted)} style={{ height: 320 }} />
            </Card>
          )}

          {history.length === 0 && !confusionMatrix && !rocFpr && !scatter && (
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
