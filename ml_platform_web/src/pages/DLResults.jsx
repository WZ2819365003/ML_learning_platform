import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Alert, Button, Card, Col, Row, Statistic, Typography, message,
} from 'antd';
import { ArrowLeftOutlined, DownloadOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import { dlApi } from '../services/api';

const { Title, Text } = Typography;

function useQuery() {
  return new URLSearchParams(useLocation().search);
}

// ── Metric display name mapping ───────────────────────────────────────────────
const metricDisplayNames = {
  best_val_loss: '最优验证损失',
  val_acc: '验证准确率',
  val_rmse: '验证 RMSE',
  val_mae: '验证 MAE',
  val_r2: '验证 R²',
  final_epoch: '最终 Epoch',
  train_loss: '训练损失',
};

function getMetricDisplayName(key) {
  return metricDisplayNames[key] ?? key;
}

// ── Format metric value ───────────────────────────────────────────────────────
function formatMetricValue(key, value) {
  if (value == null) return '-';
  if (key === 'final_epoch') return String(Math.round(value));
  if (typeof value === 'number') return value.toFixed(4);
  return String(value);
}

// ── Build history line chart ──────────────────────────────────────────────────
function buildHistoryOption(history) {
  const epochs = history.map(d => d.epoch ?? d.step ?? '');
  const trainLosses = history.map(d => d.train_loss ?? null);
  const valLosses = history.map(d => d.val_loss ?? null);

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

// ── Main Page ─────────────────────────────────────────────────────────────────
const DLResults = () => {
  const query = useQuery();
  const taskId = query.get('taskId');
  const navigate = useNavigate();

  const [taskInfo, setTaskInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!taskId) return;
    void loadTask();
  }, [taskId]);

  async function loadTask() {
    setLoading(true);
    try {
      const data = await dlApi.getStatus(taskId);
      setTaskInfo(data);
    } catch (err) {
      console.error('加载任务结果失败:', err);
      message.error('加载任务结果失败');
    } finally {
      setLoading(false);
    }
  }

  if (!taskId) {
    return (
      <div>
        <Title level={2}>深度学习结果</Title>
        <Alert type="warning" message="未提供任务 ID，请从训练监控页面跳转过来。" showIcon />
      </div>
    );
  }

  const status = (taskInfo?.status ?? '').toUpperCase();
  const metrics = taskInfo?.result_metrics ?? {};
  const metricEntries = Object.entries(metrics);
  const history = taskInfo?.history ?? [];

  return (
    <div>
      <Space style={{ width: '100%', justifyContent: 'space-between', marginBottom: 24 }}>
        <Title level={2} style={{ margin: 0 }}>
          深度学习结果
        </Title>
        <Button
          icon={<ArrowLeftOutlined />}
          onClick={() => navigate(`/dl/monitor?taskId=${taskId}`)}
        >
          返回监控
        </Button>
      </Space>

      {loading && (
        <Card>
          <Text type="secondary">正在加载结果…</Text>
        </Card>
      )}

      {/* ── FAILED ─────────────────────────────────────────────────────── */}
      {!loading && status === 'FAILED' && (
        <Alert
          type="error"
          showIcon
          message="训练失败"
          description={taskInfo?.error_message ?? '未知错误，请查看日志。'}
        />
      )}

      {/* ── SUCCESS ────────────────────────────────────────────────────── */}
      {!loading && status === 'SUCCESS' && (
        <>
          {/* Metrics cards */}
          {metricEntries.length > 0 && (
            <Card title="训练指标" style={{ marginBottom: 24 }}>
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
            <Card title="训练曲线" style={{ marginBottom: 24 }}>
              <ReactECharts
                option={buildHistoryOption(history)}
                style={{ height: 360 }}
                notMerge
              />
            </Card>
          )}

          {history.length === 0 && (
            <Card style={{ marginBottom: 24 }}>
              <Text type="secondary">训练历史数据不可用，仅展示最终指标。</Text>
            </Card>
          )}

          {/* Download (placeholder) */}
          <Card title="模型文件">
            <Button icon={<DownloadOutlined />} disabled>
              下载模型文件（暂不可用）
            </Button>
          </Card>
        </>
      )}

      {/* ── Not finished yet ───────────────────────────────────────────── */}
      {!loading && status !== 'SUCCESS' && status !== 'FAILED' && (
        <Alert
          type="info"
          showIcon
          message="训练尚未完成"
          description={
            <span>
              当前状态：{status || '未知'}。
              <Button
                type="link"
                style={{ padding: 0, marginLeft: 4 }}
                onClick={() => navigate(`/dl/monitor?taskId=${taskId}`)}
              >
                前往监控页
              </Button>
            </span>
          }
        />
      )}
    </div>
  );
};

// Small inline Space to avoid adding new import (antd Space)
function Space({ children, style }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, ...style }}>
      {children}
    </div>
  );
}

export default DLResults;
