import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Button, Card, Empty, Modal, Popconfirm, Progress, Space, Spin, Tag, Tooltip,
  Tree, Typography, message,
} from 'antd'
import {
  AppstoreOutlined, CheckCircleOutlined, ClockCircleOutlined,
  CloseCircleOutlined, DeleteOutlined, ExperimentOutlined, FileTextOutlined,
  PoweroffOutlined, QuestionCircleOutlined,
  ReloadOutlined, SyncOutlined, ThunderboltOutlined,
} from '@ant-design/icons'
import {
  modelingTaskApi, platformExperimentsApi, platformRunsApi, platformTasksApi,
} from '../../services/api'
import LogViewer from './LogViewer'

const { Text } = Typography

/**
 * A node is "active" when the scheduler may still advance it. Drives three
 * things that must agree: whether 停止 is offered, whether 删除 is blocked
 * (the backend refuses to delete a RUNNING experiment), and whether the log
 * modal tails live. Exported for tests.
 */
export const ACTIVE_STATUSES = new Set(['RUNNING', 'PENDING', 'QUEUED', 'RETRY'])
export const isActive = (status) => ACTIVE_STATUSES.has((status || '').toUpperCase())

/**
 * ProgressTree — three-level orchestration view for a ModelingTask.
 *
 *   ModelingTask (root)
 *     └── PlatformExperiment (strategy_type)
 *           └── ExperimentRun (one leaf per selected model / per trial)
 *
 * Reads `/api/v3/tasks/{id}/progress-tree`.  While the backend reports
 * `has_active_runs=true`, we re-fetch every 3 seconds so per-epoch DL
 * updates and ML fold completions surface without a reload.
 *
 * Leaf rendering:
 *   ML: <ExperimentOutlined/> + blue tag
 *   DL: <ThunderboltOutlined/> + purple tag
 *   unknown: <QuestionCircleOutlined/>
 *
 * Per-node actions:
 *   Run   — 日志 (live tail modal), 停止 (cancel, only while active)
 *   批次  — 删除 (refused server-side while RUNNING, so disabled here too)
 */
export default function ProgressTree({ modelingTaskId, autoRefresh = true, pollMs = 3000 }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  // The run whose logs the modal is showing (null = closed). Holds the whole
  // run node so the modal keeps its title/status even as the tree re-polls.
  const [logRun, setLogRun] = useState(null)
  const [busyId, setBusyId] = useState(null)

  const load = useCallback(async () => {
    if (!modelingTaskId) return
    setLoading(true)
    try {
      const tree = await modelingTaskApi.progressTree(modelingTaskId)
      setData(tree)
    } catch (err) {
      // Swallow — the endpoint may not be ready on older backends, and a
      // stale tree is harmless.  The caller Card still renders.
      console.warn('加载进度树失败', err)
    } finally {
      setLoading(false)
    }
  }, [modelingTaskId])

  useEffect(() => { load() }, [load])

  useEffect(() => {
    if (!autoRefresh) return
    if (!data?.has_active_runs) return
    const t = setInterval(load, pollMs)
    return () => clearInterval(t)
  }, [autoRefresh, pollMs, data?.has_active_runs, load])

  // ── Run actions ──────────────────────────────────────────────────────────
  // Stop targets the PlatformTask, not the Run: `cancel_task` terminates the
  // Task and its ExperimentRun in one locked transaction and triggers batch
  // finalisation, so cancelling the last trial still closes the batch.
  const stopRun = useCallback(async (run) => {
    if (!run?.platform_task_id) {
      message.warning('该 Run 没有关联的调度任务，无法停止')
      return
    }
    setBusyId(run.id)
    try {
      await platformTasksApi.cancel(run.platform_task_id)
      message.success('已请求停止，运行中的任务会在当前步骤结束后终止')
      await load()
    } catch (err) {
      message.error(_errText(err) || '停止失败')
    } finally {
      setBusyId(null)
    }
  }, [load])

  // Delete is offered per *batch*: there is no per-run delete, and removing a
  // PlatformTask alone would orphan its ExperimentRun. The backend refuses to
  // delete a RUNNING experiment, so stop first, then delete.
  const deleteExperiment = useCallback(async (exp) => {
    setBusyId(exp.id)
    try {
      await platformExperimentsApi.delete(exp.id)
      message.success('批次已删除')
      await load()
    } catch (err) {
      message.error(_errText(err) || '删除失败')
    } finally {
      setBusyId(null)
    }
  }, [load])

  const treeData = useMemo(
    () => _buildTreeNodes(data, { onViewLogs: setLogRun, onStopRun: stopRun, onDeleteExperiment: deleteExperiment, busyId }),
    [data, stopRun, deleteExperiment, busyId],
  )
  // Expansion is user-controlled, but seeded so the interesting rows are open.
  //
  // It used to be pinned to *every* node with no way to collapse, so each
  // "再加一组" added ~76px of permanently-expanded tree and the step grew
  // without bound. Now finished batches start collapsed to a single line and
  // only batches with something still running are opened, which keeps the
  // panel a fixed height however many batches accumulate.
  const autoExpandedKeys = useMemo(() => _activeNodeKeys(data), [data])
  const [expandedKeys, setExpandedKeys] = useState(null)
  const [touched, setTouched] = useState(false)

  // Follow the data until the user expresses a preference, then stop fighting
  // them — otherwise every 3s poll would re-collapse what they just opened.
  useEffect(() => {
    if (!touched) setExpandedKeys(autoExpandedKeys)
  }, [autoExpandedKeys, touched])

  const allKeys = useMemo(() => _allNodeKeys(treeData), [treeData])
  const allExpanded = expandedKeys != null && allKeys.length > 0
    && allKeys.every((k) => expandedKeys.includes(k))

  return (
    <Card
      size="small"
      title={
        <Space>
          <AppstoreOutlined />
          <span>编排进度</span>
          {data?.modeling_task && (
            <Tag color="blue">
              {data.modeling_task.experiment_count} 个批次 · {data.modeling_task.run_count} 个 Run
            </Tag>
          )}
        </Space>
      }
      extra={
        <Space>
          {data?.has_active_runs && (
            <Tag icon={<SyncOutlined spin />} color="processing">自动刷新中</Tag>
          )}
          <Button
            size="small"
            onClick={() => {
              setTouched(true)
              setExpandedKeys(allExpanded ? [] : allKeys)
            }}
          >
            {allExpanded ? '全部折叠' : '全部展开'}
          </Button>
          <Button size="small" icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>
      }
    >
      {data?.modeling_task && (
        <div style={{ marginBottom: 12 }}>
          <Space align="center" style={{ width: '100%' }}>
            <Text strong>整体进度</Text>
            <div style={{ flex: 1, minWidth: 240 }}>
              <Progress
                percent={Math.round((data.modeling_task.progress_aggregated ?? 0) * 100)}
                size="small"
                status={data.has_active_runs ? 'active' : 'normal'}
              />
            </div>
          </Space>
        </div>
      )}

      <Spin spinning={loading && !data}>
        {treeData.length === 0 ? (
          <Empty description="暂无实验批次 — 创建实验后会在这里看到进度" />
        ) : (
          <div style={{ maxHeight: 340, overflowY: 'auto', overflowX: 'hidden' }}>
            <Tree
              treeData={treeData}
              expandedKeys={expandedKeys || []}
              onExpand={(keys) => { setTouched(true); setExpandedKeys(keys) }}
              selectable={false}
              showLine
              blockNode
            />
          </div>
        )}
      </Spin>

      <RunLogModal run={logRun} onClose={() => setLogRun(null)} />
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Live log modal
// ---------------------------------------------------------------------------

/**
 * RunLogModal — live log tail for a single Run.
 *
 * Seeds historical entries from the Run Inspector endpoint (which already owns
 * the messy id-resolution: V3 native logs first, legacy `training_logs` and the
 * on-disk file as fallbacks) and hands the live WebSocket tail to LogViewer.
 * The WS channel is keyed by the *domain* task id, not the Run id.
 */
function RunLogModal({ run, onClose }) {
  const [historical, setHistorical] = useState([])
  const [loading, setLoading] = useState(false)
  const runId = run?.id

  useEffect(() => {
    if (!runId) { setHistorical([]); return }
    let cancelled = false
    setLoading(true)
    platformRunsApi.inspect(runId, { log_limit: 500, include_siblings: false })
      .then((resp) => { if (!cancelled) setHistorical(resp?.logs || []) })
      .catch(() => { if (!cancelled) setHistorical([]) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [runId])

  return (
    <Modal
      open={!!run}
      onCancel={onClose}
      footer={null}
      width={960}
      destroyOnClose
      title={
        <Space>
          <FileTextOutlined />
          <span>训练日志</span>
          {run && <Text strong>{run.model_type || '未命名'}</Text>}
          {run?.trial_no != null && <Text type="secondary">#{run.trial_no}</Text>}
          {run && <Tag color={_statusColor(run.status)}>{run.status}</Tag>}
        </Space>
      }
    >
      <Spin spinning={loading}>
        {run && (
          run.domain_id ? (
            <LogViewer
              historical={historical}
              domainTaskId={run.domain_id}
              isLive={isActive(run.status)}
            />
          ) : (
            <Empty description="该 Run 还没有关联的训练任务，暂无日志" />
          )
        )}
      </Spin>
    </Modal>
  )
}

function _errText(err) {
  const body = err?.response?.data
  return (typeof body === 'string' ? body : body?.detail) || err?.message || ''
}

// ---------------------------------------------------------------------------
// Tree node construction
// ---------------------------------------------------------------------------

function _buildTreeNodes(data, actions = {}) {
  if (!data?.experiments) return []
  const { onViewLogs, onStopRun, onDeleteExperiment, busyId } = actions
  return data.experiments.map((exp) => ({
    key: `exp:${exp.id}`,
    title: (
      <ExperimentNode
        exp={exp}
        onDelete={onDeleteExperiment}
        busy={busyId === exp.id}
      />
    ),
    children: (exp.runs || []).map((run) => ({
      key: `run:${run.id}`,
      title: (
        <RunNode
          run={run}
          onViewLogs={onViewLogs}
          onStop={onStopRun}
          busy={busyId === run.id}
        />
      ),
      isLeaf: true,
    })),
  }))
}

/**
 * Keys to auto-expand: batches that still have a run the scheduler can advance.
 * A finished batch stays collapsed to one line — its runs are still reachable,
 * just not occupying space by default.
 */
export function _activeNodeKeys(data) {
  const keys = []
  for (const exp of data?.experiments || []) {
    const liveRun = (exp.runs || []).some((r) => isActive(r.status))
    if (isActive(exp.status) || liveRun) keys.push(`exp:${exp.id}`)
  }
  return keys
}

function _allNodeKeys(nodes) {
  const keys = []
  nodes.forEach((n) => {
    keys.push(n.key)
    if (n.children) n.children.forEach((c) => keys.push(c.key))
  })
  return keys
}

// ── Experiment node ────────────────────────────────────────────────────────

function ExperimentNode({ exp, onDelete, busy }) {
  const pct = Math.round((exp.progress_aggregated ?? 0) * 100)
  const running = isActive(exp.status)
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 0' }}>
      <AppstoreOutlined style={{ color: '#64748b' }} />
      <Text strong ellipsis style={{ minWidth: 140 }}>{exp.name}</Text>
      <Tag color={_strategyColor(exp.strategy_type)}>
        {_strategyLabel(exp.strategy_type)}
      </Tag>
      <Tag color={_statusColor(exp.status)}>{exp.status}</Tag>
      <Text type="secondary" style={{ fontSize: 11 }}>
        {exp.run_count} Run
      </Text>
      <div style={{ width: 160 }}>
        <Progress percent={pct} size="small" showInfo />
      </div>
      {/* Deleting a running batch is refused server-side — disable it here so
          the reason is visible before the click rather than as an error after.
          A disabled button fires no mouse events, so the tooltip explaining
          *why* would never show; the wrapper span is what receives the hover. */}
      <Tooltip title={running ? '批次运行中，请先停止其中的 Run' : '删除该批次(含其 Run 记录)'}>
        <span style={{ display: 'inline-flex' }}>
          <Popconfirm
            title="删除该实验批次？"
            description="批次及其 Run 记录会一并删除，不可恢复。"
            okText="删除"
            okButtonProps={{ danger: true }}
            cancelText="取消"
            disabled={running}
            onConfirm={() => onDelete?.(exp)}
          >
            <Button
              size="small"
              type="text"
              danger
              loading={busy}
              disabled={running}
              icon={<DeleteOutlined />}
              style={running ? { pointerEvents: 'none' } : undefined}
            />
          </Popconfirm>
        </span>
      </Tooltip>
    </div>
  )
}

// ── Run node ───────────────────────────────────────────────────────────────

function RunNode({ run, onViewLogs, onStop, busy }) {
  const pct = Math.round((run.progress ?? 0) * 100)
  const familyChip = _familyChip(run.family)
  const running = isActive(run.status)
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '3px 0' }}>
      {familyChip}
      <Text style={{ minWidth: 120 }} ellipsis>
        <Text strong>{run.model_type || '未命名'}</Text>
        {run.trial_no != null && (
          <Text type="secondary" style={{ fontSize: 11, marginLeft: 4 }}>
            #{run.trial_no}
          </Text>
        )}
      </Text>
      <Tag color={_statusColor(run.status)} icon={_statusIcon(run.status)} style={{ margin: 0 }}>
        {run.status}
      </Tag>
      {run.current_step && (
        <Tooltip title="当前步骤">
          <Text type="secondary" style={{ fontSize: 12, minWidth: 110 }}>
            {run.current_step}
          </Text>
        </Tooltip>
      )}
      <div style={{ flex: 1, minWidth: 120 }}>
        <Progress
          percent={pct}
          size="small"
          status={_progressStatus(run.status)}
          showInfo
        />
      </div>
      <Space size={2}>
        <Tooltip title="查看实时日志">
          <Button
            size="small"
            type="text"
            icon={<FileTextOutlined />}
            onClick={() => onViewLogs?.(run)}
          />
        </Tooltip>
        {running && (
          <Tooltip title="停止该 Run">
            <Popconfirm
              title="停止该 Run？"
              description="已完成的部分会保留，训练不会继续。"
              okText="停止"
              okButtonProps={{ danger: true }}
              cancelText="取消"
              onConfirm={() => onStop?.(run)}
            >
              <Button
                size="small"
                type="text"
                danger
                loading={busy}
                icon={<PoweroffOutlined />}
              />
            </Popconfirm>
          </Tooltip>
        )}
      </Space>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Styling helpers
// ---------------------------------------------------------------------------

function _familyChip(family) {
  if (family === 'dl') {
    return <Tag icon={<ThunderboltOutlined />} color="purple" style={{ margin: 0 }}>DL</Tag>
  }
  if (family === 'ml') {
    return <Tag icon={<ExperimentOutlined />} color="blue" style={{ margin: 0 }}>ML</Tag>
  }
  return <Tag icon={<QuestionCircleOutlined />} color="default" style={{ margin: 0 }}>?</Tag>
}

function _statusColor(status) {
  const k = status?.toUpperCase?.()
  if (k === 'SUCCESS' || k === 'COMPLETED') return 'green'
  if (k === 'RUNNING') return 'blue'
  if (k === 'QUEUED' || k === 'PENDING') return 'gold'
  if (k === 'FAILED') return 'red'
  if (k === 'RETRY') return 'orange'
  if (k === 'CANCELLED') return 'default'
  return 'default'
}

function _statusIcon(status) {
  const k = status?.toUpperCase?.()
  if (k === 'SUCCESS' || k === 'COMPLETED') return <CheckCircleOutlined />
  if (k === 'RUNNING') return <SyncOutlined spin />
  if (k === 'QUEUED' || k === 'PENDING') return <ClockCircleOutlined />
  if (k === 'FAILED') return <CloseCircleOutlined />
  return null
}

function _progressStatus(status) {
  const k = status?.toUpperCase?.()
  if (k === 'FAILED')    return 'exception'
  if (k === 'CANCELLED') return 'exception'
  if (k === 'SUCCESS' || k === 'COMPLETED') return 'success'
  if (k === 'RUNNING')   return 'active'
  return 'normal'
}

function _strategyLabel(s) {
  return ({
    baseline:        '基线',
    grid_search:     '网格',
    bayesian_search: '贝叶斯',
  })[s] || s || '—'
}

function _strategyColor(s) {
  return ({
    baseline:        'blue',
    grid_search:     'geekblue',
    bayesian_search: 'purple',
  })[s] || 'default'
}
