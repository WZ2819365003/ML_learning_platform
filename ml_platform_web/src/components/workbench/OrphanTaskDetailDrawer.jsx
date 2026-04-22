import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert, Button, Descriptions, Drawer, Empty, Popconfirm, Space, Spin,
  Tabs, Tag, Tooltip, Typography, message,
} from 'antd'
import {
  ClockCircleOutlined, CloseCircleOutlined, CodeOutlined,
  DatabaseOutlined, ExperimentOutlined, FileTextOutlined, InfoCircleOutlined,
  LineChartOutlined, ReloadOutlined, RedoOutlined,
} from '@ant-design/icons'
import { platformTasksApi } from '../../services/api'

const { Paragraph, Text } = Typography

/**
 * OrphanTaskDetailDrawer — drill-down for the `孤立任务` tab in TaskCenter.
 *
 * Backed by `GET /api/platform/tasks/{id}/detail` which enriches the task
 * row with:
 *   - source_label        ─ human-friendly description of payload_ref kind
 *   - domain_kind/domain  ─ resolved domain entity (TrainingTask, DL…, ts…)
 *   - recent_logs         ─ tail of storage/logs/{domain_id}.log
 *
 * Tabs:
 *   1. 概览 — task core fields + timeline
 *   2. 源数据 — per-kind domain summary
 *   3. 日志 — tailing log lines (refresh while task is RUNNING)
 *   4. 操作 — retry / cancel / delete (delegated to parent callbacks)
 */
export default function OrphanTaskDetailDrawer({
  taskId,
  open,
  onClose,
  onRetry,
  onCancel,
  onDelete,
}) {
  const [detail, setDetail] = useState(null)
  const [loading, setLoading] = useState(false)

  const load = useCallback(async () => {
    if (!taskId) return
    setLoading(true)
    try {
      const data = await platformTasksApi.detail(taskId)
      setDetail(data)
    } catch (err) {
      message.error(err?.response?.data?.detail || '加载详情失败')
      setDetail(null)
    } finally {
      setLoading(false)
    }
  }, [taskId])

  useEffect(() => {
    if (open && taskId) load()
    // Clear out the stale payload so re-opening a different task doesn't
    // briefly flash the previous task's data.
    if (!open) setDetail(null)
  }, [open, taskId, load])

  // Light auto-refresh: while the task is active, poll every 3s so the
  // drawer reflects epoch-level DL progress + fresh log lines.
  useEffect(() => {
    const status = detail?.task?.status?.toUpperCase()
    if (!open || !['RUNNING', 'QUEUED', 'PENDING', 'RETRY'].includes(status)) return
    const t = setInterval(load, 3000)
    return () => clearInterval(t)
  }, [open, detail?.task?.status, load])

  const task   = detail?.task
  const domain = detail?.domain
  const kind   = detail?.domain_kind

  const tabs = useMemo(() => ([
    {
      key: 'overview',
      label: <span><InfoCircleOutlined /> 概览</span>,
      children: <OverviewTab task={task} sourceLabel={detail?.source_label} />,
    },
    {
      key: 'source',
      label: <span><DatabaseOutlined /> 源数据</span>,
      children: <SourceTab kind={kind} domain={domain} />,
    },
    {
      key: 'logs',
      label: <span><CodeOutlined /> 日志</span>,
      children: <LogsTab logs={detail?.recent_logs} onRefresh={load} />,
    },
    {
      key: 'actions',
      label: <span><ReloadOutlined /> 操作</span>,
      children: (
        <ActionsTab
          task={task}
          onRetry={async () => { await onRetry?.(task?.id); await load() }}
          onCancel={async () => { await onCancel?.(task?.id); await load() }}
          onDelete={async () => { await onDelete?.(task?.id); onClose?.() }}
        />
      ),
    },
  ]), [task, domain, kind, detail?.source_label, detail?.recent_logs, load, onRetry, onCancel, onDelete, onClose])

  return (
    <Drawer
      title={
        <Space>
          <ExperimentOutlined />
          <span>任务详情</span>
          {detail?.source_label && (
            <Tag color="blue" style={{ marginLeft: 4 }}>{detail.source_label}</Tag>
          )}
          {task?.status && (
            <Tag color={_statusColor(task.status)}>{task.status}</Tag>
          )}
        </Space>
      }
      placement="right"
      width={720}
      open={open}
      onClose={onClose}
      destroyOnClose
    >
      <Spin spinning={loading && !detail}>
        {!detail && !loading ? (
          <Empty description="暂无数据" />
        ) : detail ? (
          <Tabs items={tabs} defaultActiveKey="overview" />
        ) : null}
      </Spin>
    </Drawer>
  )
}

// ── Overview ───────────────────────────────────────────────────────────────
function OverviewTab({ task, sourceLabel }) {
  if (!task) return <Empty />
  return (
    <Descriptions size="small" column={2} bordered>
      <Descriptions.Item label="任务 ID" span={2}>
        <Text code copyable>{task.id}</Text>
      </Descriptions.Item>
      <Descriptions.Item label="来源">{sourceLabel || '—'}</Descriptions.Item>
      <Descriptions.Item label="类型"><Tag>{task.kind}</Tag></Descriptions.Item>
      <Descriptions.Item label="状态"><Tag color={_statusColor(task.status)}>{task.status}</Tag></Descriptions.Item>
      <Descriptions.Item label="优先级">{task.priority}</Descriptions.Item>
      <Descriptions.Item label="进度">{((task.progress ?? 0) * 100).toFixed(1)}%</Descriptions.Item>
      <Descriptions.Item label="重试次数">{task.retry_count} / {task.max_retries}</Descriptions.Item>
      <Descriptions.Item label="关联引用" span={2}>
        <Text code>{task.payload_ref || '—'}</Text>
      </Descriptions.Item>
      {task.celery_task_id && (
        <Descriptions.Item label="Celery ID" span={2}>
          <Text code>{task.celery_task_id}</Text>
        </Descriptions.Item>
      )}
      {task.worker_id && (
        <Descriptions.Item label="Worker">{task.worker_id}</Descriptions.Item>
      )}
      {Array.isArray(task.depends_on) && task.depends_on.length > 0 && (
        <Descriptions.Item label="依赖任务" span={2}>
          <Space size={4} wrap>
            {task.depends_on.map((d) => (
              <Tag key={d} icon={<ClockCircleOutlined />}>{d.slice(0, 8)}…</Tag>
            ))}
          </Space>
        </Descriptions.Item>
      )}
      <Descriptions.Item label="创建时间">{_fmt(task.created_at)}</Descriptions.Item>
      <Descriptions.Item label="入队时间">{_fmt(task.queued_at)}</Descriptions.Item>
      <Descriptions.Item label="开始时间">{_fmt(task.started_at)}</Descriptions.Item>
      <Descriptions.Item label="结束时间">{_fmt(task.finished_at)}</Descriptions.Item>
      {task.error_message && (
        <Descriptions.Item label="错误信息" span={2}>
          <Alert type="error" showIcon message={
            <Paragraph copyable style={{ marginBottom: 0, whiteSpace: 'pre-wrap' }}>
              {task.error_message}
            </Paragraph>
          } />
        </Descriptions.Item>
      )}
      {task.metrics_snapshot && Object.keys(task.metrics_snapshot).length > 0 && (
        <Descriptions.Item label="指标快照" span={2}>
          <Space wrap>
            {Object.entries(task.metrics_snapshot).map(([k, v]) => (
              <Tag color="geekblue" key={k}>{k}: {_fmtMetric(v)}</Tag>
            ))}
          </Space>
        </Descriptions.Item>
      )}
    </Descriptions>
  )
}

// ── Source data ────────────────────────────────────────────────────────────
function SourceTab({ kind, domain }) {
  if (!domain) {
    return (
      <Alert type="info" showIcon
        message="源数据不可用"
        description="关联的业务记录可能已被删除，或 payload_ref 前缀不受支持。任务本身仍可正常查看。"
      />
    )
  }
  if (kind === 'train')       return <TrainDomain d={domain} />
  if (kind === 'dl_train')    return <DLTrainDomain d={domain} />
  if (kind === 'explain')     return <ExplainDomain d={domain} />
  if (kind === 'ts_forecast') return <TSForecastDomain d={domain} />
  return (
    <pre style={{
      background: '#0f172a', color: '#e2e8f0',
      padding: 16, borderRadius: 8, fontSize: 12,
      maxHeight: 480, overflow: 'auto',
    }}>{JSON.stringify(domain, null, 2)}</pre>
  )
}

function TrainDomain({ d }) {
  return (
    <Descriptions size="small" column={2} bordered>
      <Descriptions.Item label="训练任务 ID" span={2}><Text code copyable>{d.id}</Text></Descriptions.Item>
      <Descriptions.Item label="名称" span={2}>{d.name || '—'}</Descriptions.Item>
      <Descriptions.Item label="模型"><Tag color="blue">{d.model_type}</Tag></Descriptions.Item>
      <Descriptions.Item label="目标列">{d.target_column}</Descriptions.Item>
      <Descriptions.Item label="数据集"><Text code>{d.dataset_id}</Text></Descriptions.Item>
      <Descriptions.Item label="状态"><Tag color={_statusColor(d.status)}>{d.status}</Tag></Descriptions.Item>
      <Descriptions.Item label="进度">{((d.progress ?? 0) * 100).toFixed(1)}%</Descriptions.Item>
      <Descriptions.Item label="模型文件" span={2}>{d.model_path || '—'}</Descriptions.Item>
      {d.result_metrics && Object.keys(d.result_metrics).length > 0 && (
        <Descriptions.Item label="训练指标" span={2}>
          <Space wrap>
            {Object.entries(d.result_metrics).map(([k, v]) => (
              <Tag color="purple" key={k}>{k}: {_fmtMetric(v)}</Tag>
            ))}
          </Space>
        </Descriptions.Item>
      )}
    </Descriptions>
  )
}

function DLTrainDomain({ d }) {
  return (
    <Descriptions size="small" column={2} bordered>
      <Descriptions.Item label="DL 任务 ID" span={2}><Text code copyable>{d.id}</Text></Descriptions.Item>
      <Descriptions.Item label="名称" span={2}>{d.name || '—'}</Descriptions.Item>
      <Descriptions.Item label="模型"><Tag color="purple">{d.model_type}</Tag></Descriptions.Item>
      <Descriptions.Item label="任务类型"><Tag>{d.task_type}</Tag></Descriptions.Item>
      <Descriptions.Item label="目标列">{d.target_column}</Descriptions.Item>
      <Descriptions.Item label="状态"><Tag color={_statusColor(d.status)}>{d.status}</Tag></Descriptions.Item>
      <Descriptions.Item label="Epoch">{d.current_epoch} / {d.total_epochs}</Descriptions.Item>
      <Descriptions.Item label="进度">{((d.progress ?? 0) * 100).toFixed(1)}%</Descriptions.Item>
      <Descriptions.Item label="结构配置" span={2}>
        <pre style={_jsonStyle}>{JSON.stringify(d.arch_config, null, 2)}</pre>
      </Descriptions.Item>
      <Descriptions.Item label="优化器配置" span={2}>
        <pre style={_jsonStyle}>{JSON.stringify(d.opt_config, null, 2)}</pre>
      </Descriptions.Item>
      <Descriptions.Item label="训练配置" span={2}>
        <pre style={_jsonStyle}>{JSON.stringify(d.train_config, null, 2)}</pre>
      </Descriptions.Item>
      {d.result_metrics && Object.keys(d.result_metrics).length > 0 && (
        <Descriptions.Item label="训练指标" span={2}>
          <Space wrap>
            {Object.entries(d.result_metrics).map(([k, v]) => (
              <Tag color="purple" key={k}>{k}: {_fmtMetric(v)}</Tag>
            ))}
          </Space>
        </Descriptions.Item>
      )}
    </Descriptions>
  )
}

function ExplainDomain({ d }) {
  return (
    <Descriptions size="small" column={2} bordered>
      <Descriptions.Item label="Run ID" span={2}><Text code copyable>{d.run_id}</Text></Descriptions.Item>
      <Descriptions.Item label="实验 ID" span={2}><Text code>{d.experiment_id}</Text></Descriptions.Item>
      <Descriptions.Item label="模型"><Tag color="blue">{d.model_type || '—'}</Tag></Descriptions.Item>
      <Descriptions.Item label="策略"><Tag>{d.source_experiment_type || '—'}</Tag></Descriptions.Item>
      <Descriptions.Item label="状态"><Tag color={_statusColor(d.status)}>{d.status}</Tag></Descriptions.Item>
      <Descriptions.Item label="排名">{d.rank ?? '—'}</Descriptions.Item>
      <Descriptions.Item label="目标列">{d.target_column || '—'}</Descriptions.Item>
      <Descriptions.Item label="数据集"><Text code>{d.dataset_id || '—'}</Text></Descriptions.Item>
      {d.linked_training_task_id && (
        <Descriptions.Item label="对应训练任务" span={2}>
          <Text code>{d.linked_training_task_id}</Text>
        </Descriptions.Item>
      )}
      {d.artifacts_uri && (
        <Descriptions.Item label="产物 URI" span={2}>
          <Text code>{d.artifacts_uri}</Text>
        </Descriptions.Item>
      )}
      {d.metrics && Object.keys(d.metrics).length > 0 && (
        <Descriptions.Item label="Run 指标" span={2}>
          <Space wrap>
            {Object.entries(d.metrics).map(([k, v]) => (
              <Tag color="geekblue" key={k}>{k}: {_fmtMetric(v)}</Tag>
            ))}
          </Space>
        </Descriptions.Item>
      )}
    </Descriptions>
  )
}

function TSForecastDomain({ d }) {
  const preview = d.result_preview || {}
  return (
    <Descriptions size="small" column={2} bordered>
      <Descriptions.Item label="预测任务 ID" span={2}><Text code copyable>{d.id}</Text></Descriptions.Item>
      <Descriptions.Item label="数据集">{d.dataset_name || d.dataset_id}</Descriptions.Item>
      <Descriptions.Item label="模型"><Tag color="cyan">{d.model_name}</Tag></Descriptions.Item>
      <Descriptions.Item label="值列">{d.value_column}</Descriptions.Item>
      <Descriptions.Item label="时间列">{d.time_column || '—'}</Descriptions.Item>
      <Descriptions.Item label="预测步长">{d.horizon}</Descriptions.Item>
      <Descriptions.Item label="频率">{d.frequency}</Descriptions.Item>
      <Descriptions.Item label="状态" span={2}>
        <Tag color={_statusColor(d.status)}>{d.status}</Tag>
      </Descriptions.Item>
      {Object.keys(preview).length > 0 && (
        <Descriptions.Item label="结果预览" span={2}>
          <pre style={_jsonStyle}>{JSON.stringify(preview, null, 2)}</pre>
        </Descriptions.Item>
      )}
    </Descriptions>
  )
}

// ── Logs ───────────────────────────────────────────────────────────────────
function LogsTab({ logs, onRefresh }) {
  if (!Array.isArray(logs) || logs.length === 0) {
    return (
      <div>
        <Button icon={<ReloadOutlined />} onClick={onRefresh} size="small" style={{ marginBottom: 12 }}>
          刷新
        </Button>
        <Empty description="暂无日志" />
      </div>
    )
  }
  return (
    <div>
      <Space style={{ marginBottom: 8 }}>
        <Button icon={<ReloadOutlined />} onClick={onRefresh} size="small">刷新</Button>
        <Text type="secondary" style={{ fontSize: 12 }}>
          显示最近 {logs.length} 行
        </Text>
      </Space>
      <pre style={{
        background: '#0f172a', color: '#e2e8f0',
        padding: 16, borderRadius: 8, fontSize: 12, lineHeight: 1.5,
        maxHeight: 520, overflow: 'auto',
        fontFamily: 'Menlo, Consolas, monospace',
      }}>
        {logs.join('\n')}
      </pre>
    </div>
  )
}

// ── Actions ────────────────────────────────────────────────────────────────
function ActionsTab({ task, onRetry, onCancel, onDelete }) {
  if (!task) return <Empty />
  const s = task.status?.toUpperCase()
  const canRetry  = s === 'FAILED' || s === 'RETRY'
  const canCancel = s === 'PENDING' || s === 'QUEUED'
  const canDelete = s === 'SUCCESS' || s === 'FAILED' || s === 'CANCELLED'
  return (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      {canRetry && (
        <Alert type="warning" showIcon
          message="重新提交"
          description="将以当前配置重新入队，原有的执行历史保留。"
          action={
            <Button icon={<RedoOutlined />} type="primary" onClick={onRetry}>重试</Button>
          }
        />
      )}
      {canCancel && (
        <Alert type="info" showIcon
          message="取消执行"
          description="任务将立即标记为 CANCELLED 并从调度队列移除。"
          action={
            <Popconfirm title="确认取消？" onConfirm={onCancel}>
              <Button danger icon={<CloseCircleOutlined />}>取消任务</Button>
            </Popconfirm>
          }
        />
      )}
      {canDelete && (
        <Alert type="error" showIcon
          message="删除记录"
          description="此操作不可撤销；仅清除调度层记录，不影响模型文件。"
          action={
            <Popconfirm title="确认删除？" onConfirm={onDelete}>
              <Button danger>删除</Button>
            </Popconfirm>
          }
        />
      )}
      {!canRetry && !canCancel && !canDelete && (
        <Alert type="info" showIcon message="当前状态无可用操作" />
      )}
    </Space>
  )
}

// ── Helpers ────────────────────────────────────────────────────────────────
function _fmt(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleString('zh-CN', { hour12: false })
}

function _fmtMetric(v) {
  if (typeof v === 'number') return Number.isInteger(v) ? v : v.toFixed(4)
  return String(v)
}

function _statusColor(status) {
  const k = status?.toUpperCase()
  if (k === 'SUCCESS' || k === 'COMPLETED') return 'green'
  if (k === 'RUNNING') return 'blue'
  if (k === 'QUEUED' || k === 'PENDING') return 'gold'
  if (k === 'FAILED') return 'red'
  if (k === 'RETRY') return 'orange'
  if (k === 'CANCELLED') return 'default'
  return 'default'
}

const _jsonStyle = {
  background: '#0f172a', color: '#e2e8f0',
  padding: 10, borderRadius: 6, fontSize: 11, lineHeight: 1.5,
  margin: 0, maxHeight: 180, overflow: 'auto',
}
