import React, { useEffect, useState } from 'react'
import {
  Drawer, Descriptions, Tag, Space, Table, Tabs, Empty, Spin,
  Typography, Divider, Alert, Tooltip,
} from 'antd'
import {
  CheckCircleFilled, CloseCircleFilled, ClockCircleFilled,
  DatabaseOutlined, ExperimentOutlined, LineChartOutlined,
} from '@ant-design/icons'
import { platformRunsApi } from '../../services/api'
import LogViewer from './LogViewer'
import ShapView from './ShapView'
import TrainingViz from './TrainingViz'

const { Text, Paragraph } = Typography

const STATUS_TAG = {
  SUCCESS:  <Tag icon={<CheckCircleFilled />} color="success">成功</Tag>,
  FAILED:   <Tag icon={<CloseCircleFilled />} color="error">失败</Tag>,
  RUNNING:  <Tag icon={<ClockCircleFilled spin />} color="processing">运行中</Tag>,
  PENDING:  <Tag icon={<ClockCircleFilled />} color="default">等待中</Tag>,
  QUEUED:   <Tag icon={<ClockCircleFilled />} color="default">排队中</Tag>,
  CANCELED: <Tag color="warning">已取消</Tag>,
}

const LEVEL_COLOR = {
  INFO: '#3b82f6', WARN: '#f59e0b', WARNING: '#f59e0b', ERROR: '#ef4444', DEBUG: '#94a3b8',
}

function MetricsGrid({ metrics }) {
  if (!metrics || Object.keys(metrics).length === 0) return <Empty description="无指标" image={Empty.PRESENTED_IMAGE_SIMPLE} />
  const entries = Object.entries(metrics).filter(([, v]) => typeof v === 'number')
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 8 }}>
      {entries.map(([k, v]) => (
        <div key={k} style={{
          padding: '8px 12px', borderRadius: 6,
          background: 'rgba(37, 99, 235, 0.04)', border: '1px solid rgba(37, 99, 235, 0.1)',
        }}>
          <div style={{ fontSize: 11, color: '#64748b' }}>{k}</div>
          <div style={{ fontSize: 16, fontWeight: 700, color: '#0f172a', fontFamily: 'monospace' }}>
            {typeof v === 'number' ? v.toFixed(4) : String(v)}
          </div>
        </div>
      ))}
    </div>
  )
}

function ParamsTable({ params }) {
  if (!params || Object.keys(params).length === 0) return <Empty description="无超参" image={Empty.PRESENTED_IMAGE_SIMPLE} />
  return (
    <Table
      size="small"
      pagination={false}
      rowKey={(r) => r.key}
      dataSource={Object.entries(params).map(([k, v]) => ({ key: k, value: v }))}
      columns={[
        { title: '参数', dataIndex: 'key', width: '40%',
          render: (v) => <code style={{ fontSize: 12 }}>{v}</code> },
        { title: '取值', dataIndex: 'value',
          render: (v) => <code style={{ fontSize: 12, color: '#2563eb' }}>
            {typeof v === 'object' ? JSON.stringify(v) : String(v)}
          </code> },
      ]}
    />
  )
}

export default function RunInspector({ open, runId, onClose }) {
  const [loading, setLoading] = useState(false)
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!open || !runId) return
    setLoading(true)
    setError(null)
    platformRunsApi.inspect(runId, { log_limit: 200 })
      .then(resp => setData(resp))
      .catch(err => setError(err?.response?.data?.detail || err.message))
      .finally(() => setLoading(false))
  }, [open, runId])

  const run = data?.run
  const exp = data?.experiment
  const ptask = data?.platform_task
  const ttask = data?.training_task
  const ds = ttask?.dataset
  const shap = data?.shap
  const siblings = data?.siblings || []
  const logs = data?.logs || []

  return (
    <Drawer
      title={
        <Space>
          <LineChartOutlined style={{ color: '#2563eb' }} />
          <span>Run 诊断</span>
          {run && <Text type="secondary" style={{ fontSize: 12 }}>#{run.id?.slice(0, 8)}</Text>}
          {run && STATUS_TAG[run.status]}
        </Space>
      }
      placement="right"
      width={780}
      open={open}
      onClose={onClose}
      bodyStyle={{ padding: '16px 20px' }}
    >
      <Spin spinning={loading}>
        {error && <Alert type="error" showIcon message={error} style={{ marginBottom: 12 }} />}

        {!loading && !error && data && (
          <Tabs
            defaultActiveKey="overview"
            items={[
              // ── Overview ──
              {
                key: 'overview',
                label: '概览',
                children: (
                  <Space direction="vertical" size={14} style={{ width: '100%' }}>
                    <Descriptions size="small" column={2} bordered
                      labelStyle={{ background: '#f8fafc', width: 110 }}>
                      <Descriptions.Item label="Trial 号">{run?.trial_no ?? '-'}</Descriptions.Item>
                      <Descriptions.Item label="排名">{run?.rank ?? '-'}</Descriptions.Item>
                      <Descriptions.Item label="所属实验">{exp?.name || '-'}</Descriptions.Item>
                      <Descriptions.Item label="策略">
                        <Tag color="blue">{exp?.strategy_type || exp?.source_experiment_type || '-'}</Tag>
                      </Descriptions.Item>
                      <Descriptions.Item label="优化指标">{exp?.objective_metric}</Descriptions.Item>
                      <Descriptions.Item label="方向">{exp?.objective_direction}</Descriptions.Item>
                      <Descriptions.Item label="开始时间">
                        {ptask?.started_at ? new Date(ptask.started_at).toLocaleString('zh-CN', { hour12: false }) : '-'}
                      </Descriptions.Item>
                      <Descriptions.Item label="结束时间">
                        {ptask?.finished_at ? new Date(ptask.finished_at).toLocaleString('zh-CN', { hour12: false }) : '-'}
                      </Descriptions.Item>
                    </Descriptions>

                    <div>
                      <Text strong style={{ fontSize: 13 }}>指标</Text>
                      <div style={{ marginTop: 6 }}>
                        <MetricsGrid metrics={run?.metrics} />
                      </div>
                    </div>

                    {run?.search_meta && (
                      <div>
                        <Text strong style={{ fontSize: 13 }}>搜索元数据</Text>
                        <pre style={{
                          marginTop: 4, fontSize: 11, padding: 8, borderRadius: 6,
                          background: '#f8fafc', border: '1px solid #e2e8f0',
                          maxHeight: 140, overflow: 'auto',
                        }}>
                          {JSON.stringify(run.search_meta, null, 2)}
                        </pre>
                      </div>
                    )}
                  </Space>
                ),
              },

              // ── Training visualisation ──
              // Renders confusion matrix / residual plot / ROC / learning
              // curve / feature importance in a tight 2×2 grid.  Driven off
              // the domain TrainingTask id (not the Run id) — the backend
              // viz helpers re-load the model + dataset on demand.
              {
                key: 'training_viz',
                label: <span>训练可视化</span>,
                children: (
                  <TrainingViz
                    trainingTaskId={ttask?.id}
                    modelType={ttask?.model_type}
                    taskStatus={run?.status}
                  />
                ),
              },

              // ── Params ──
              {
                key: 'params',
                label: <span>超参 <Tag style={{ marginLeft: 4 }}>{Object.keys(run?.params || {}).length}</Tag></span>,
                children: <ParamsTable params={run?.params} />,
              },

              // ── Dataset + training task ──
              {
                key: 'context',
                label: '上下文',
                children: (
                  <Space direction="vertical" size={14} style={{ width: '100%' }}>
                    <div>
                      <Text strong><DatabaseOutlined /> 数据集</Text>
                      {ds ? (
                        <Descriptions size="small" column={2} style={{ marginTop: 6 }} bordered
                          labelStyle={{ background: '#f8fafc', width: 100 }}>
                          <Descriptions.Item label="名称">{ds.name}</Descriptions.Item>
                          <Descriptions.Item label="行数">{ds.row_count ?? '-'}</Descriptions.Item>
                          <Descriptions.Item label="列数">{ds.column_count ?? '-'}</Descriptions.Item>
                          <Descriptions.Item label="文件">
                            <Text ellipsis style={{ maxWidth: 220, fontSize: 12 }}>{ds.file_path || '-'}</Text>
                          </Descriptions.Item>
                        </Descriptions>
                      ) : <Empty description="无数据集记录" image={Empty.PRESENTED_IMAGE_SIMPLE} />}
                    </div>

                    <div>
                      <Text strong><ExperimentOutlined /> 训练任务</Text>
                      {ttask ? (
                        <Descriptions size="small" column={2} style={{ marginTop: 6 }} bordered
                          labelStyle={{ background: '#f8fafc', width: 100 }}>
                          <Descriptions.Item label="模型">{ttask.model_type}</Descriptions.Item>
                          <Descriptions.Item label="状态">{STATUS_TAG[ttask.status] || ttask.status}</Descriptions.Item>
                          <Descriptions.Item label="进度">{ttask.progress != null ? `${(ttask.progress * 100).toFixed(0)}%` : '-'}</Descriptions.Item>
                          <Descriptions.Item label="目标列">{ttask.target_column || '-'}</Descriptions.Item>
                          <Descriptions.Item label="模型文件" span={2}>
                            <Text ellipsis style={{ maxWidth: 400, fontSize: 12 }}>{ttask.model_path || '-'}</Text>
                          </Descriptions.Item>
                        </Descriptions>
                      ) : <Empty description="无训练任务绑定" image={Empty.PRESENTED_IMAGE_SIMPLE} />}
                    </div>
                  </Space>
                ),
              },

              // ── Logs ──
              // Professional log panel: REST-seeded historical entries + live
              // WS tail; level filter, search, pause, download. The WS channel
              // is keyed by the domain TrainingTask id (not the Run id).
              {
                key: 'logs',
                label: <span>日志 <Tag style={{ marginLeft: 4 }}>{logs.length}</Tag></span>,
                children: (
                  <LogViewer
                    historical={logs}
                    domainTaskId={ttask?.id}
                    isLive={run?.status === 'RUNNING' || run?.status === 'PENDING'}
                  />
                ),
              },

              // ── Siblings ──
              {
                key: 'siblings',
                label: <span>同批次 <Tag style={{ marginLeft: 4 }}>{siblings.length}</Tag></span>,
                children: (
                  <Table
                    size="small"
                    rowKey="id"
                    pagination={{ pageSize: 10, size: 'small' }}
                    dataSource={siblings}
                    columns={[
                      { title: '#', dataIndex: 'trial_no', width: 50 },
                      { title: 'Rank', dataIndex: 'rank', width: 60,
                        render: (v) => v === 1 ? <Tag color="gold">1</Tag> : v },
                      { title: '状态', dataIndex: 'status', width: 90,
                        render: (s) => STATUS_TAG[s] || s },
                      { title: '目标值', key: 'objective',
                        render: (_, r) => {
                          const m = exp?.objective_metric
                          const v = m && r.metrics?.[m]
                          return <Tooltip title={JSON.stringify(r.metrics)}>
                            <code style={{ color: '#2563eb' }}>{typeof v === 'number' ? v.toFixed(4) : '-'}</code>
                          </Tooltip>
                        } },
                    ]}
                  />
                ),
              },

              // ── SHAP ──
              {
                key: 'shap',
                label: (
                  <span>
                    SHAP
                    {shap?.has_explanation && (
                      <Tag color="success" style={{ marginLeft: 4 }}>就绪</Tag>
                    )}
                  </span>
                ),
                children: run ? (
                  <ShapView
                    runId={run.id}
                    initialSummary={shap}
                    experimentId={run.experiment_id || exp?.id}
                    runStatus={run.status}
                  />
                ) : (
                  <Empty description="无 Run 数据" />
                ),
              },
            ]}
          />
        )}
      </Spin>
    </Drawer>
  )
}
