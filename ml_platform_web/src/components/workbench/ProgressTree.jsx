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
export default function ProgressTree({
  modelingTaskId, taskName, autoRefresh = true, pollMs = 3000,
  statusCounts, headerExtra, maxBodyHeight = 520,   // null → let the host scroll
  fillHeight = false,                               // stretch to the host's box
}) {
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
    () => _buildTreeNodes(data, { onViewLogs: setLogRun, onStopRun: stopRun, onDeleteExperiment: deleteExperiment, busyId, taskName }),
    [data, stopRun, deleteExperiment, busyId, taskName],
  )
  const expandedKeys = useMemo(() => _allNodeKeys(treeData), [treeData])

  return (
    <Card
      size="small"
      // fillHeight: the host gives us a fixed frame and expects us to fill it.
      // Without this the card sized itself to its rows and left a band of dead
      // space between its bottom edge and the frame's.
      style={fillHeight ? { height: '100%', display: 'flex', flexDirection: 'column' } : undefined}
      styles={fillHeight ? {
        body: { flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' },
      } : undefined}
      title={
        <Space size={8} wrap>
          <AppstoreOutlined />
          <span>编排进度</span>
          {data?.modeling_task && (
            <Tag color="blue" style={{ margin: 0 }}>
              {data.modeling_task.experiment_count} 批次 · {data.modeling_task.run_count} Run
            </Tag>
          )}
          {/* Absorbed from the separate 训练进度 card above: the same three
              numbers were being shown twice, one card apart. */}
          {statusCounts && (
            <>
              <Tag color="green" style={{ margin: 0 }}>成功 {statusCounts.SUCCESS || 0}</Tag>
              {(statusCounts.RUNNING || statusCounts.PENDING) ? (
                <Tag color="processing" style={{ margin: 0 }}>
                  运行中 {(statusCounts.RUNNING || 0) + (statusCounts.PENDING || 0)}
                </Tag>
              ) : null}
              {statusCounts.FAILED ? (
                <Tag color="error" style={{ margin: 0 }}>失败 {statusCounts.FAILED}</Tag>
              ) : null}
            </>
          )}

        </Space>
      }
      extra={
        <Space size={4}>
          {data?.has_active_runs && (
            <Tag icon={<SyncOutlined spin />} color="processing" style={{ margin: 0 }}>
              自动刷新中
            </Tag>
          )}
          <Button size="small" icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
          {headerExtra}
        </Space>
      }
    >
      {data?.modeling_task && (
        <div style={{ marginBottom: 10 }}>
          <Space align="center" style={{ width: '100%' }}>
            <Text strong style={{ fontSize: 12 }}>整体进度</Text>
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

      <Spin spinning={loading && !data}
        wrapperClassName={fillHeight ? 'pt-fill' : undefined}
        style={fillHeight ? { flex: 1, minHeight: 0 } : undefined}>
        {treeData.length === 0 ? (
          <Empty description="暂无实验批次 — 创建实验后会在这里看到进度" />
        ) : (
          // Scroll container: the panel keeps one height no matter how many
          // batches 再加一组 adds, instead of pushing everything below it
          // further down the page on every click.
          //
          // Skipped when maxBodyHeight is null. A caller that already renders
          // this inside its own fixed, scrolling box wants that box to be the
          // only scroller — nesting a second one just produces two scrollbars
          // for one list.
          fillHeight ? (
            <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', paddingRight: 4 }}>
              <Tree
                treeData={treeData}
                expandedKeys={expandedKeys}
                selectable={false}
                showLine
                blockNode
              />
            </div>
          ) : maxBodyHeight == null ? (
            <Tree
              treeData={treeData}
              expandedKeys={expandedKeys}
              selectable={false}
              showLine
              blockNode
            />
          ) : (
            <div style={{ maxHeight: maxBodyHeight, overflowY: 'auto', paddingRight: 4 }}>
              <Tree
                treeData={treeData}
                expandedKeys={expandedKeys}
                selectable={false}
                showLine
                blockNode
              />
            </div>
          )
        )}
      </Spin>

      {fillHeight && (
        <style>{`
          .pt-fill,
          .pt-fill > .ant-spin-container {
            flex: 1;
            min-height: 0;
            display: flex;
            flex-direction: column;
          }
        `}</style>
      )}

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
  const { onViewLogs, onStopRun, onDeleteExperiment, busyId, taskName } = actions
  return data.experiments.map((exp) => {
    const runs = exp.runs || []

    // A batch with exactly one run repeats itself: the batch name already
    // contains the model, and both rows carry the same status and the same
    // 100% bar. Render them as a single row — the parent/child split only
    // earns its height when a batch actually holds several trials.
    if (runs.length === 1) {
      return {
        key: `exp:${exp.id}`,
        isLeaf: true,
        title: (
          <MergedNode
            exp={exp}
            run={runs[0]}
            taskName={taskName}
            onViewLogs={onViewLogs}
            onStop={onStopRun}
            onDelete={onDeleteExperiment}
            busy={busyId === exp.id || busyId === runs[0].id}
          />
        ),
      }
    }

    return {
      key: `exp:${exp.id}`,
      title: (
        <ExperimentNode
          exp={exp}
          taskName={taskName}
          onDelete={onDeleteExperiment}
          busy={busyId === exp.id}
        />
      ),
      children: runs.map((run) => ({
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
    }
  })
}

/**
 * Batch names are built as `${taskName}-${model}-${timestamp}`. On this page
 * the task name is already the page title, so repeating it on every row costs
 * horizontal space and tells the reader nothing.
 */
function _shortBatchName(name, taskName) {
  if (!name) return '未命名批次'
  let short = taskName && name.startsWith(`${taskName}-`)
    ? name.slice(taskName.length + 1)
    : name
  // Trailing run-id/timestamp suffix — keep it, but let it recede visually.
  return short
}

function _allNodeKeys(nodes) {
  const keys = []
  nodes.forEach((n) => {
    keys.push(n.key)
    if (n.children) n.children.forEach((c) => keys.push(c.key))
  })
  return keys
}

// ── Shared row pieces ──────────────────────────────────────────────────────

/**
 * One status indicator per row, not four.
 *
 * A finished row used to carry the status tag, the literal text 已完成, a 100%
 * progress bar and a tick — four elements for one fact. The bar and the live
 * step text only say something while a run is still moving, so they render
 * only then.
 */
function StatusCell({ status, currentStep, progressPct }) {
  const running = isActive(status)
  return (
    <>
      <Tag color={_statusColor(status)} icon={_statusIcon(status)} style={{ margin: 0 }}>
        {status}
      </Tag>
      {/* The live step text is only meaningful while something is moving —
          finished rows used to render a literal 已完成 next to a 成功 tag. */}
      {running && currentStep && (
        <Tooltip title="当前步骤">
          <Text type="secondary" style={{ fontSize: 12, minWidth: 96 }}>{currentStep}</Text>
        </Tooltip>
      )}
      {/* Every row keeps its bar, finished or not. I had hidden the completed
          ones as "redundant with the status tag", but the row of full bars is
          what makes the batch readable at a glance — you see how far the whole
          set got without reading a single label. */}
      <div style={{ flex: 1, minWidth: 140 }}>
        <Progress percent={progressPct} size="small" status={_progressStatus(status)} />
      </div>
    </>
  )
}

function LogButton({ run, onViewLogs }) {
  return (
    <Tooltip title="查看实时日志">
      <Button size="small" type="text" icon={<FileTextOutlined />}
        onClick={() => onViewLogs?.(run)} />
    </Tooltip>
  )
}

function StopButton({ run, onStop, busy }) {
  return (
    <Tooltip title="停止该 Run">
      <Popconfirm
        title="停止该 Run？"
        description="已完成的部分会保留，训练不会继续。"
        okText="停止" okButtonProps={{ danger: true }} cancelText="取消"
        onConfirm={() => onStop?.(run)}
      >
        <Button size="small" type="text" danger loading={busy} icon={<PoweroffOutlined />} />
      </Popconfirm>
    </Tooltip>
  )
}

function DeleteButton({ exp, onDelete, busy, running }) {
  return (
    // A disabled button fires no mouse events, so the tooltip explaining why
    // would never show; the wrapper span is what receives the hover.
    <Tooltip title={running ? '批次运行中，请先停止其中的 Run' : '删除该批次(含其 Run 记录)'}>
      <span style={{ display: 'inline-flex' }}>
        <Popconfirm
          title="删除该实验批次？"
          description="批次及其 Run 记录会一并删除，不可恢复。"
          okText="删除" okButtonProps={{ danger: true }} cancelText="取消"
          disabled={running}
          onConfirm={() => onDelete?.(exp)}
        >
          <Button size="small" type="text" danger loading={busy} disabled={running}
            icon={<DeleteOutlined />}
            style={running ? { pointerEvents: 'none' } : undefined} />
        </Popconfirm>
      </span>
    </Tooltip>
  )
}

const ROW = { display: 'flex', alignItems: 'center', gap: 8, padding: '3px 0' }

// ── Merged node (batch with exactly one run) ───────────────────────────────

function MergedNode({ exp, run, taskName, onViewLogs, onStop, onDelete, busy }) {
  const running = isActive(run.status) || isActive(exp.status)
  return (
    <div style={ROW}>
      {_familyChip(run.family)}
      <Text strong style={{ minWidth: 150 }} ellipsis>
        {run.model_type || _shortBatchName(exp.name, taskName)}
      </Text>
      <Tag color={_strategyColor(exp.strategy_type)} style={{ margin: 0 }}>
        {_strategyLabel(exp.strategy_type)}
      </Tag>
      <StatusCell
        status={run.status}
        currentStep={run.current_step}
        progressPct={Math.round((run.progress ?? 0) * 100)}
      />
      <Tooltip title={exp.name}>
        <Text type="secondary" style={{ fontSize: 11 }}>
          {_shortBatchName(exp.name, taskName)}
        </Text>
      </Tooltip>
      <Space size={2}>
        <LogButton run={run} onViewLogs={onViewLogs} />
        {isActive(run.status) && <StopButton run={run} onStop={onStop} busy={busy} />}
        <DeleteButton exp={exp} onDelete={onDelete} busy={busy} running={running} />
      </Space>
    </div>
  )
}

// ── Experiment node (multi-run batch) ──────────────────────────────────────

function ExperimentNode({ exp, taskName, onDelete, busy }) {
  const running = isActive(exp.status)
  return (
    <div style={{ ...ROW, padding: '4px 0' }}>
      <AppstoreOutlined style={{ color: '#64748b' }} />
      <Text strong ellipsis style={{ minWidth: 150 }}>
        {_shortBatchName(exp.name, taskName)}
      </Text>
      <Tag color={_strategyColor(exp.strategy_type)} style={{ margin: 0 }}>
        {_strategyLabel(exp.strategy_type)}
      </Tag>
      <StatusCell
        status={exp.status}
        progressPct={Math.round((exp.progress_aggregated ?? 0) * 100)}
      />
      <Text type="secondary" style={{ fontSize: 11 }}>{exp.run_count} Run</Text>
      <DeleteButton exp={exp} onDelete={onDelete} busy={busy} running={running} />
    </div>
  )
}

// ── Run node (inside a multi-run batch) ────────────────────────────────────

function RunNode({ run, onViewLogs, onStop, busy }) {
  return (
    <div style={ROW}>
      {_familyChip(run.family)}
      <Text style={{ minWidth: 130 }} ellipsis>
        <Text strong>{run.model_type || '未命名'}</Text>
        {run.trial_no != null && (
          <Text type="secondary" style={{ fontSize: 11, marginLeft: 4 }}>#{run.trial_no}</Text>
        )}
      </Text>
      <StatusCell
        status={run.status}
        currentStep={run.current_step}
        progressPct={Math.round((run.progress ?? 0) * 100)}
      />
      <Space size={2}>
        <LogButton run={run} onViewLogs={onViewLogs} />
        {isActive(run.status) && <StopButton run={run} onStop={onStop} busy={busy} />}
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
